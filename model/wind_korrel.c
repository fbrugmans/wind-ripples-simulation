#include <stdio.h>
/* Wind-coupled grain model for wind ripples (Brugmans / Claude, 3 Oct. 2026).
 *
 * Each grain in the air is followed in time under gravity and drag in a logarithmic wind
 * profile above the local bed; the grains slow the wind down through their momentum flux
 * (Kok & Renno 2009, COMSALT, eqs. 26-30). Collisions with the bed follow the splash function
 * of COMSALT (Table 1) in the first version:
 *   rebound with probability B (1 - exp(-gam v)), B = 0.96, gam = 1 s/m;
 *     retained kinetic energy N(0.45; 0.22), angle ~ Exp(mean 40 degrees);
 *   otherwise the grain stays (on top of the column hit);
 *   ejections: number ~ Poisson(a v / sqrt(g D)), a = 0.02; speed ~ Exp(<v_ej>),
 *     <v_ej> = (alpha_ej / a) sqrt(g D) (1 - exp(-v / (40 sqrt(g D)))), alpha_ej = 0.15;
 *     angle ~ Exp(mean 50 degrees); total ejection momentum <= (1 - alpha_reb) m v;
 *     released at a distance eps ~ Exp(2 d) downwind (Lester et al. 2025).
 * In the article model the collision follows Beladjine et al. (2007) (splashmodel = 2, set in run_factor.py).
 * Aerodynamic entrainment (only to start/maintain transport): when the air shear stress at
 * the bed exceeds the fluid threshold.
 *
 * BED: columns (width dx) with h[i] layers of thickness dz; one grain = one cell. The
 * slab width W (across the wind) is such that a grain exactly fills a cell:
 * W = (pi d^3 / 6) / (dx dz phi_b). Avalanche: slope over Wl columns > tan 32, toppling >= 3.
 * The wind profile follows the terrain (height above the local bed); the flow modulation by the
 * topography was added later (windmod, set from wind_korrel.py).
 * Units: SI (m, s, kg).
 */
#include <stdint.h>
#include <math.h>
#include <stdlib.h>

static uint64_t S0 = 0x9E3779B97F4A7C15ULL, S1 = 0xD1B54A32D192ED03ULL;
static inline uint64_t xr(void) {
    uint64_t s1 = S0; const uint64_t s0 = S1; S0 = s0;
    s1 ^= s1 << 23; S1 = s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26); return S1 + s0;
}
static inline double U(void) { return (xr() >> 11) * (1.0 / 9007199254740992.0); }
static inline double nrm(void) { return sqrt(-2 * log(1 - U())) * cos(6.283185307179586 * U()); }
void zaai(uint64_t a) { S0 = a ^ 0x9E3779B97F4A7C15ULL; S1 = (a * 0xBF58476D1CE4E5B9ULL) ^ 0x94D049BB133111EBULL; for (int i = 0; i < 20; i++) xr(); }

/* ---- parameters (set from Python) ---- */
typedef struct {
    double d, rho_p, rho_a, nu, g;          /* grain and air */
    double ustar, kappa, z0;                 /* wind */
    double ust_fl;                           /* fluid threshold u* (m/s) for aerodynamic entrainment */
    double aero_rate;                        /* entrainment per metre per s per unit exceedance */
    double B, gam, ereb_m, ereb_s, threb;    /* rebound */
    double a, alpha_ej, thej, Dsp;           /* ejection; Dsp = D in sqrt(g D) */
    double eps_d;                            /* mean impact-ejection delay in d */
    double dx, dz, tanr; int Wl, toren;      /* bed */
    double dt, t_upd, relax;                 /* time step, interval of the wind update */
    int bevroren;                            /* 1: bed does not change (flux measurement) */
} Par;

static Par P;
double terrein = 0;   /* 23 Oct.: terrain angle (rad), > 0 = mean bed rises downwind; 0 = unchanged */
/* 24 Oct.: CREST DIAGNOSTICS (counters only, no random numbers). kam_vlag[i] = 1 for columns around the crest (from Python).
   kam_hist[q * KAM_NH + b]: q 0 creep displacement from the crest (columns), 1 displacement by wind-driven sliding from the crest, 2 jump
   after ejection from the crest, 3 jump after rebound from the crest (columns of dx, up to the first impact), 4 impact angle relative to the
   local bed at the crest (degrees), 5 descent angle relative to the mean bed at the crest (degrees); 4 and 5 only for fast (v > 1 m/s), descending grains. */
#define KAM_NH 400
static uint8_t *kam_vlag = 0; static double *kam_hist = 0;
void zet_kam(uint8_t *v_, double *h_) { kam_vlag = v_; kam_hist = h_; }
/* 24 Oct.: LAUNCHING BY A PUSH (lanceer_duw = 1; hypothesis: grains grazing the crest abrade the top). A
   pushed grain gets a speed along the x-axis u = Bx / (n_duw m_k): the momentum loss of the impact along the x-axis, shared equally
   over the pushes of that impact. If the bed (smoothed over 2 mm) 2 mm (5 columns) further downwind lies more than g_z s^2 / (2 u^2) lower than at the grain
   (s = 2 mm, g_z = g cos terrain angle), the bed drops away faster than the grain falls: the grain flies (is launched along the x-axis,
   angle 0.02 rad) instead of rolling. Otherwise it rolls as before. Counters: n_duw (pushes with a speed), n_lduw (launched). */
int lanceer_duw = 0; double kr_u = 0; long n_duw_u = 0, n_lduw = 0;
void lduw_uit(long *u) { u[0] = n_duw_u; u[1] = n_lduw; n_duw_u = 0; n_lduw = 0; }
static inline void kam_bij(int q, double x) { int b = (int)x; if (b < 0) b = 0; if (b >= KAM_NH) b = KAM_NH - 1; kam_hist[q * KAM_NH + b] += 1; }
/* 24 Oct.: COHESION in the ripple body (cohesion number Co = F_c / weight of a grain; Valance et al. 2025, EPJ Web Conf. 340, 06008;
   moisture in the ripple body). (1) angle of repose: force balance of a grain with an extra normal force F_c: tan phi_c = tan phi (1 + Co);
   (2) threshold for wind-driven sliding: u*_c^2 x (1 + 3 Co) (= (1 + 6 omega)^2); aerodynamic entrainment into saltation stays
   unchanged: in the model it only starts saltation (in the field saltation comes from upwind; also in the article the threshold
   for ongoing saltation stays the same);
   (3) ejections per impact unchanged for Co <= 1, above that x Co^-0.5; creep pushes unchanged. Co = 0: unchanged. */
double cohesie = 0;
int lawine_hoek = 1;   /* 24 Oct.: 0 = angle-of-repose rule in the avalanche routine off; the toppling rule (column >= 3 layers above its neighbour falls) stays (2D) */
int ter_delen = 15;   /* 23 Oct.: which parts of the terrain angle act (decomposition): 1 avalanches, 2 creep (stop rule, creep distance coarse), 4 wind-driven sliding, 8 gravity in the trajectories */
static inline double hel_d(double s, int d) { return (terrein != 0 && (ter_delen & d)) ? tan(atan(s) + terrein) : s; }   /* true slope, upward downwind */
int ter_lawrichting = 3;   /* 24 Oct.: terrain angle in the avalanche rule per direction: 1 = only downwind (s = +1, down the lee side), 2 = only upwind (s = -1, down the stoss side), 3 = both */
static inline double tr_hel(double tr, int s) { return (terrein != 0 && (ter_delen & 1) && (ter_lawrichting & (s > 0 ? 1 : 2))) ? tan(atan(tr) + s * terrein) : tr; }   /* angle of repose for an avalanche in direction s */
static int N, HM, fout;
static uint8_t *col; static int32_t *h;
/* 16 Oct.: QUASI-3D (NY adjacent cross-sections, one grain wide, dy = dx). The existing rules act per
   cross-section through a row pointer (zet_rij): h, col, windmod, shadow and the impact-rule arrays point to row r.
   Lateral coupling (each separately): lat_ej 1 = horizontal direction of ejections isotropic (Ammi et al. 2009), 2 = ejections
   as in 2D plus a lateral component |v cos th| sin(psi), psi uniform; lat_reb = spread (rad) of the lateral
   deviation on rebound; lat_kr 1 = creeping grains turn to a lower neighbouring row (probability |s_y| / tan pivot);
   lat_law 1 = avalanches also across (slope over Wl rows > angle of repose, or toppling). NY = 1: exactly the 2D model. */
static int NY = 1, rij = 0; static double dyr = 0.4e-3;
static int32_t *h0 = 0; static uint8_t *col0 = 0; static double *wm0 = 0; static uint8_t *sch0 = 0;
static double *py = 0, *pvy = 0;
int lat_ej = 0, lat_kr = 0, lat_law = 0; double lat_reb = 0;
/* 16 Oct.: LATERAL BOOKKEEPING (quasi-3D): lat_stat[((proc * 2 + q) * NY + r) * N + i], proc 0 jump (permanent landing in a
   cross-section other than where the grain left the bed), 1 creep (ending in another cross-section), 2 lateral avalanche; q 0 departure, 1 arrival. */
static double *lat_stat = 0; void zet_lat_stat(double *a) { lat_stat = a; }
static int *p_r0 = 0, *p_x0 = 0; static int l_r0 = 0, l_x0 = 0, p_rcap = 0;
static inline void lat_tel_bij(int proc, int r0, int i0, int r1, int i1) {
    if (!lat_stat) return;
    lat_stat[((int64_t)(proc * 2 + 0) * NY + r0) * N + i0] += 1; lat_stat[((int64_t)(proc * 2 + 1) * NY + r1) * N + i1] += 1;
}
static double lat_tel[6];   /* creep sideways, avalanche across, avalanche along, ejections, sum |vy|, sum |vx| of ejections */
void lat_uit(double *u) { for (int q = 0; q < 6; q++) { u[q] = lat_tel[q]; lat_tel[q] = 0; } }
void zet_3d(int ny, double dy, double *py_, double *pvy_) { NY = ny > 0 ? ny : 1; dyr = dy; py = py_; pvy = pvy_; }
/* TWO GRAIN SIZES (3 Oct.): class k has diameter dk[k]; nk = 1 = one size (P.d) */
int nk = 1; double dk[4] = {0, 0, 0, 0}, ustk[4] = {0, 0, 0, 0};
static double mk[4];
static inline double dkl(int k) { return nk > 1 ? dk[k] : P.d; }
static double W, m, Lx;                      /* slab width, grain mass, domain length */
static double *stroom_s = 0, *stroom_z = 0; static int stroom_nz = 0;   /* 13 Nov.: STROOM, slope of the streamlines [nz][N], heights (m) */
void zet_stroom(double *s, double *z, int nz) { stroom_s = s; stroom_z = z; stroom_nz = nz; }
static double *windmod;                         /* FLOW MODULATION (3 Oct.): factor on the wind profile per column, or NULL */

/* wind profile on a logarithmic grid above the bed */
#define NZ 200
static double zk[NZ], Uk[NZ], taup[NZ];
static double zmax = 0.5;
static double tp_last[NZ], tp_f[NZ];
static double mom_drag, t_tot;

static inline int md(int i) { i %= N; return i < 0 ? i + N : i; }
static inline double hb(int i) { return h[md(i)] * P.dz; }
/* 6 Oct.: CRUST (bonding after rest). korst[i] = height (number of layers) of column i at the
   end of the last rest period; the top grain is bonded as long as h[i] == korst[i]. A bonded grain
   is only knocked loose by an impact with v > korst_fac * xi_c * sqrt(g d); creep and aerodynamic
   entrainment leave it in place. If it is removed anyway, the crust is gone there (korst[i] = 0). Not used in the article. */
static int32_t *korst = 0; double korst_fac = 1.0;
/* 7 Oct.: HARD CRUST (korst_hard = 1): an impact on a bonded grain knocks nothing loose and pushes nothing;
   the grain rebounds if its normal speed suffices (as on a hard bed, Ho et al. 2011), otherwise
   it stays on the crust; with probability korst_pb per impact the crust crumbles at that spot. */
int korst_hard = 0; double korst_pb = 0;
/* 6 Oct.: OPEN BOUNDARY (C12b): grains leaving the domain on the right disappear; on the left saltation from
   upwind enters through injecteer(): n grains just inside x = 0, height ~ Exp(mean 5 mm) above the bed,
   vx ~ N(1.5; 0.5) m/s (>= 0.3), vz ~ N(0; 0.5) m/s, class according to the bed distribution (f). */
int open_rand = 0; long n_uit = 0, n_in = 0;
void zet_korst(int32_t *k_) { korst = k_; }
static inline int gehecht(int i) { return korst && korst[i] > 0 && h[i] == korst[i]; }
/* 7 Oct.: CRUST WITH MEMORY. stug[cell] = number of rest periods the grain in that cell
   has survived untouched (0..stug_max), for the WHOLE bed. A grain that is moved starts at 0.
   Effect on the top grain with level n (factor f = 1 + stug_a n): ejection only if the impact
   v > f xi_c sqrt(gd); creep pushes it with probability 1/f; aerodynamic entrainment only if u*_bed > f u*_ft. */
static uint8_t *stug = 0; double stug_a = 0; int stug_max = 5;
void zet_stug(uint8_t *s_) { stug = s_; }
static inline double stugf(int i) { return (stug && h[i] > 0) ? 1 + stug_a * stug[(int64_t)i * HM + h[i] - 1] : 1.0; }
/* 10 Oct.: COUNTERS OF COARSE GRAINS (diagnostics only, no influence on the physics): per column departure (out) and arrival
   (in) of coarse grains (class >= 1) per process ctx: 0 other, 1 creep, 2 avalanche, 3 landing from the air,
   4 launch (splash/aero). Layout gtel[(ctx * 2 + in) * N + i]. */
static int64_t *gtel = 0; static int ctx = 0;
static int gp_wr = 0; static int32_t *gp_kol = 0; void gp_zet(int32_t *a) { gp_kol = a; }   /* 27 Oct.: bed change per process and column (GPKOL) */
#define GP_K(op_, i_) do { if (gp_kol) { int c_ = ctx == 1 ? (gp_wr ? 2 : 1) : ctx == 2 ? 3 : ctx == 3 ? 4 : ctx == 4 ? 5 : 0; gp_kol[(int64_t)(c_ * 2 + (op_)) * N + (i_)] += 1; } } while (0)
static int ld_c = 0, ld_op = 0; double ld_tel[57];
#define REC_K 12
#define REC_S 4000
#define REC_W 151
int rec_aan = 0, rec_min = 30, rec_max = REC_K, rec_hw = 75, rec_n = 0; static int rec_cur = 0;   /* 27 Oct.: recording of avalanche chains (KETENREC) */
static int *rec_meta = 0, *rec_st = 0, *rec_h0 = 0, *rec_h1 = 0; static double *rec_d = 0;
void rec_zet(int *meta, int *st, int *h0, int *h1, double *d) { rec_meta = meta; rec_st = st; rec_h0 = h0; rec_h1 = h1; rec_d = d; }
static int ld_wr = 0; double lk_tel[44]; static double *lk_kol = 0;   /* 27 Oct.: counter of avalanche chains after creep */
void lk_uit(double *u) { for (int q = 0; q < 44; q++) { u[q] = lk_tel[q]; lk_tel[q] = 0; } }
void lk_zet(double *a) { lk_kol = a; }   /* 27 Oct.: avalanche diagnostics counter: last bed change outside the avalanche */
void ld_uit(double *u) { for (int q = 0; q < 57; q++) { u[q] = ld_tel[q]; ld_tel[q] = 0; } }
static double *kol_t = 0; static uint8_t *kol_c = 0; static int kol_n = 0; static double begr_t = 0; double begr_tel[7];   /* 26 Oct.: buried counter */
void begr_uit(double *u) { for (int q = 0; q < 7; q++) { u[q] = begr_tel[q]; begr_tel[q] = 0; } }
/* 19 Oct.: gtel_alle = 1: count all classes, fine and coarse separately: gtel[((ctx * 2 + in) * 2 + (k >= 1)) * N + i] (20 N). Diagnostics only. */
int gtel_alle = 0;
static inline void gtel_bij(int i, uint8_t k_, int in_) { if (gtel_alle) gtel[(int64_t)((ctx * 2 + in_) * 2 + (k_ >= 1)) * N + i]++; else if (k_ >= 1) gtel[(int64_t)(ctx * 2 + in_) * N + i]++; }
void zet_grof_tellers(int64_t *a) { gtel = a; }
static inline uint8_t pak(int i) { GP_K(0, i); if (ctx != 2) { ld_c = ctx; ld_op = 0; } if (korst && h[i] == korst[i]) korst[i] = 0; h[i]--; if (stug) stug[(int64_t)i * HM + h[i]] = 0; uint8_t k_ = col[(int64_t)i * HM + h[i]]; if (gtel) gtel_bij(i, k_, 0); return k_; }
static inline void leg0(int i, uint8_t k) { if (ctx != 2) { ld_c = ctx; ld_op = 1; } if (h[i] >= HM) { fout = 1; return; } GP_K(1, i); col[(int64_t)i * HM + h[i]] = k; if (stug) stug[(int64_t)i * HM + h[i]] = 0; h[i]++; if (kol_t) { kol_t[i] = begr_t; kol_c[i] = (uint8_t)ctx; } if (gtel) gtel_bij(i, k, 1); }
/* 21 Oct.: gat_alle > 0: a fine grain that arrives on top of a coarse grain by creep (ctx 1) or an avalanche (ctx 2) falls with probability
   gat_alle into the space below (the coarse one stays on top), like rule (d) for landings. Default 0: as before. */
double gat_alle = 0; long n_gat = 0;
static inline void leg(int i, uint8_t k) {
    if (gat_alle > 0 && k == 0 && (ctx == 1 || ctx == 2) && h[i] >= 1 && col[(int64_t)i * HM + h[i] - 1] >= 1 && U() < gat_alle) {
        h[i]--; uint8_t kg_ = col[(int64_t)i * HM + h[i]]; leg0(i, k); col[(int64_t)i * HM + h[i]] = kg_; h[i]++; n_gat++;
        if (gtel) { gtel_bij(i, kg_, 0); gtel_bij(i, kg_, 1); }
        return;
    }
    leg0(i, k);
}

double tanrk[4] = {-1, -1, -1, -1};
/* 30 Oct.: FLUXTEL: COUNTER OF TRANSPORT PER KIND AND PLACE (diagnostics; no random numbers). ft_kol[(2 c + q) N + i]: q 0 departure,
   1 arrival in column i; ft_som[c] sum of the displacement (m). Kind c: 0 creep, 1 sliding, 2 avalanche, 3-6 jump < 5, 5-20, 20-107, >= 107 mm. */
#define FT_NC 11   /* 31 Oct.: 7-10 saltating grain by largest height < 1, 1-3, 3-10, >= 10 mm */
static double *ft_kol = 0; static double ft_som[FT_NC]; static int *p_c0 = 0; static int l_c0 = -1;
int art_kr = 1, art_ej = 1; static double *rb_kol = 0; static uint8_t *p_rb = 0;   /* 4 Nov.: ART_KR, ART_EJ */
void zet_rb_kol(double *a) { rb_kol = a; }
static double *hk_kol = 0, *hk_s = 0; static uint8_t *hk_p = 0;   /* 7 Nov.: HKTEL */
void zet_hk(double *a, double *b) { hk_kol = a; hk_s = b; }
static double *rbt_kol = 0; static int *p_rbj = 0; double rbt_s[2] = {0, 0};   /* 6 Nov.: RBTEL */
void zet_rbt(double *a) { rbt_kol = a; }
static double *kt_kol = 0, *kt_hist = 0; double kt_s[8]; static int kt_stap = -1, kt_kl = 0;   /* 7 Nov.: KRTEL */
void zet_kt(double *a, double *b) { kt_kol = a; kt_hist = b; }
void kt_uit(double *u) { for (int q = 0; q < 8; q++) { u[q] = kt_s[q]; kt_s[q] = 0; } }
void rbt_uit(double *u) { u[0] = rbt_s[0]; u[1] = rbt_s[1]; rbt_s[0] = rbt_s[1] = 0; }
static double *p_h0 = 0; static double l_h0 = 0;   /* 31 Oct.: largest height over the whole displacement */
/* 31 Oct.: MOD_LH: acceleration that decreases with height (mod_lh in m; 0 = off). mh_jh = Jackson & Hunt part, mh_pl = ceiling factor. */
double mod_lh = 0; static double *mh_jh = 0, *mh_pl = 0;
double mod_zc = 0;   /* 1 Nov.: MOD_ZC: > 0 acceleration only below z = mod_zc, < 0 only above z = |mod_zc| (m) */
void zet_mod_lh(double *jh, double *pl) { mh_jh = jh; mh_pl = pl; }
void zet_ft_kol(double *a) { ft_kol = a; }
void ft_uit(double *u) { for (int c = 0; c < FT_NC; c++) { u[c] = ft_som[c]; ft_som[c] = 0; } }
static inline void ft_bij(int c, int i, int j, double d) { ft_kol[(2 * c) * N + i] += 1; ft_kol[(2 * c + 1) * N + j] += 1; ft_som[c] += d; }
static int64_t *kol_ins = 0, *kol_kr = 0;   /* 5 Oct.: counters per column: all impacts, creep departures */
void zet_tellers(int64_t *a, int64_t *b) { kol_ins = a; kol_kr = b; }
static int64_t *kol_kr_aan = 0, *kol_afz = 0;   /* 6 Oct.: creep arrivals, permanent landing of a saltating grain */
void zet_tellers2(int64_t *a, int64_t *b) { kol_kr_aan = a; kol_afz = b; }
   /* 5 Oct.: angle of repose per class of the top grain (-1 = P.tanr) */
static int stapel[1 << 16];
static void lawine3(int i0);
double law_tel[6]; void law_uit(double *u) { for (int q = 0; q < 6; q++) { u[q] = law_tel[q]; law_tel[q] = 0; } }   /* 27 Oct.: toppling rule counter */
static void lawine(int i0) {
    if (P.bevroren) return;
    if (NY > 1 && lat_law) { lawine3(i0); return; }
    int n = 0, Wl = P.Wl;
    for (int q = -Wl; q <= Wl; q++) stapel[n++] = md(i0 + q);
    const int lk_k = (ld_c == 1 && ld_op == 1), lk_w = lk_k && ld_wr; int lk_i0 = -1, lk_j1 = -1, lk_sb;   /* 27 Oct. */
    { double s_ = (double)(h[md(i0 - Wl)] - h[md(i0 + Wl)]) * P.dz / (2 * Wl * P.dx); lk_sb = s_ < -0.2 ? 0 : s_ < 0 ? 1 : s_ < 0.2 ? 2 : s_ < 0.4 ? 3 : s_ < 0.55 ? 4 : 5; }
    if (rec_aan && lk_k && rec_n < rec_max && rec_n < REC_K && rec_meta) { rec_cur = 1;   /* 27 Oct.: recording */
        for (int q = -rec_hw; q <= rec_hw; q++) rec_h0[rec_n * REC_W + q + rec_hw] = h[md(i0 + q)]; }
    long guard = 0; int ld_n = 0, ld_pi = -1, ld_pj = -1; const int ld_b = (ld_c == 1 ? 1 : ld_c == 3 ? 2 : ld_c == 4 ? 3 : 0) * 2 + ld_op;   /* 27 Oct. */
    /* limited number of steps per call: with different angles of repose per class a grain
       could otherwise keep going back and forth between two columns (5 Oct.) */
    while (n > 0 && guard++ < 20000) {
        int i = stapel[--n], beste = 0; double ex = 0, eb_ = 0, tb_ = 0, emax_ = -1e300;
        for (int s = -1; s <= 1; s += 2) {
            int kt_ = h[i] > 0 ? col[(int64_t)i * HM + h[i] - 1] : 0;
            double tr_ = lawine_hoek ? tr_hel((tanrk[kt_] > 0 ? tanrk[kt_] : P.tanr) * (1 + cohesie), s) : 1e9;   /* 24 Oct.: lawine_hoek 0 = angle-of-repose rule off (toppling rule stays) */
            double e = ((h[i] - h[md(i + s * Wl)]) * P.dz - tr_ * Wl * P.dx) / (Wl * P.dx);
            double t = (double)(h[i] - h[md(i + s)] - (P.toren - 1));
            double mm = e > t ? e : t;
            if (e > emax_) emax_ = e;
            if (mm > ex + 1e-12) { ex = mm; beste = s; eb_ = e; tb_ = t; }
        }
        if (beste && ex > 0 && h[i] > 1 && h[md(i + beste)] < h[i]) {   /* only to a lower neighbour */
            int j = md(i + beste);
            {   /* 27 Oct.: toppling rule counter, before the step */
                int nd_ = h[i] - h[md(i - 1)] >= P.toren && h[i] - h[md(i + 1)] >= P.toren;
                int pt_ = h[md(j - 1)] - h[j] >= P.toren && h[md(j + 1)] - h[j] >= P.toren;
                int k_ = nd_ ? (emax_ > 0 ? 2 : 1) : (eb_ >= tb_ ? 0 : (pt_ ? 3 : 4));
                law_tel[k_] += 1; law_tel[5] += 1;
                ld_tel[ld_b * 5 + k_] += 1; if (i == ld_pj && j == ld_pi) ld_tel[40 + k_] += 1; ld_pi = i; ld_pj = j; ld_n++;   /* 27 Oct. */
                if (rec_cur && ld_n <= REC_S) { int s_ = ld_n - 1, *r_ = rec_st + ((int64_t)rec_n * REC_S + s_) * 5; int di_ = i - i0, dj_ = j - i0;
                    if (di_ > N / 2) di_ -= N; if (di_ < -N / 2) di_ += N; if (dj_ > N / 2) dj_ -= N; if (dj_ < -N / 2) dj_ += N;
                    r_[0] = di_; r_[1] = dj_; r_[2] = k_; r_[3] = h[i]; r_[4] = h[j];
                    rec_d[((int64_t)rec_n * REC_S + s_) * 2] = eb_ * Wl * P.dx / P.dz; rec_d[((int64_t)rec_n * REC_S + s_) * 2 + 1] = tb_; }   /* 27 Oct.: before the step */
                if (lk_k) { if (lk_i0 < 0) lk_i0 = i; lk_j1 = j; lk_tel[19 * lk_w + 6 + lk_sb] += 1; if (lk_kol) lk_kol[i] += 1; } else lk_tel[38 + lk_sb] += 1;
                if (k_ == 0) { double ov_ = eb_ * P.Wl * P.dx / P.dz; ld_tel[45 + (ov_ <= 1 ? 0 : ov_ <= 2 ? 1 : ov_ <= 4 ? 2 : 3)] += 1; }
            }
            if (ft_kol) ft_bij(2, i, j, beste * P.dx);   /* 30 Oct.: FLUXTEL */
            int c0_ = ctx; ctx = 2; leg(j, pak(i)); ctx = c0_;
            if (n + 4 * Wl + 6 < (1 << 16)) for (int q = -Wl; q <= Wl; q++) { stapel[n++] = md(i + q); stapel[n++] = md(j + q); }
        }
    }
    if (rec_cur) { rec_cur = 0;   /* 27 Oct.: keep the recording if the chain is long enough */
        if (ld_n >= rec_min) { int *m_ = rec_meta + rec_n * 6; m_[0] = i0; m_[1] = ld_n; m_[2] = lk_w; m_[3] = N; m_[4] = (int)(begr_t * 1000); m_[5] = Wl;
            for (int q = -rec_hw; q <= rec_hw; q++) rec_h1[rec_n * REC_W + q + rec_hw] = h[md(i0 + q)]; rec_n++; } }
    if (lk_k) { lk_tel[19 * lk_w + lk_sb] += 1;
        if (ld_n > 0) { int dk_ = lk_j1 - lk_i0; if (dk_ > N / 2) dk_ -= N; if (dk_ < -N / 2) dk_ += N; double d_ = dk_ * P.dx * 1e3;
            lk_tel[19 * lk_w + 12 + (d_ < 0 ? 0 : d_ <= 0.5 ? 1 : d_ <= 1 ? 2 : d_ <= 2 ? 3 : d_ <= 4 ? 4 : d_ <= 8 ? 5 : 6)] += 1; } }   /* 27 Oct. */
    ld_tel[49 + (ld_n == 0 ? 0 : ld_n == 1 ? 1 : ld_n == 2 ? 2 : ld_n <= 4 ? 3 : ld_n <= 8 ? 4 : ld_n <= 16 ? 5 : ld_n <= 64 ? 6 : 7)] += 1;   /* 27 Oct. */
}

/* 6 Oct.: BUFFER LAYER (profielmodel = 1). Mixing length of Van Driest, l = kappa z' (1 - exp(-z' u_tau / (26 nu))),
   z' = z + dvd, with viscous shear stress: tf = nu dU/dz + l^2 (dU/dz)^2, so
   dU/dz = (-nu + sqrt(nu^2 + 4 l^2 tf)) / (2 l^2), u_tau = sqrt(tf) locally. Gives viscous sublayer,
   buffer layer and log layer (cf. 'Law of the Wall', Guo & Julien 2007, in Paehtz & Tholen 2021);
   dvd (shift) calibrated on their roughness. profielmodel = 0: pure log law as before. */
/* 12 Oct.: MOMENTUM PER MODEL GRAIN (impuls_cel = 1; diagnostics): a model grain stands for a bed cell of sand (mass
   rho_p 0.6 dx dz per m width, as flux() and the mass balance compute), but the momentum transfer to the air used
   the mass of a real grain of its class (fine 176 um: 0.68 x the cell; coarse 334/488 um: 4.7 x / 14.6 x).
   impuls_cel = 1 computes the momentum with the cell mass (m, reference diameter P.d). Default since 12 Oct.: 1.
   With KORRELMASSA=1 (29 Oct.; the article model) a cell is one grain of its class with mass m_k and impuls_cel = 0. */
int impuls_cel = 1;   /* 12 Oct.: NEW DEFAULT (consistent); runs up to C23 and the recalibration of 11 Oct. used 0 */
int profielmodel = 0; double dvd = 0, A_vd = 26;
static inline double dUdz(double zm, double tf) {
    if (!profielmodel) return sqrt(tf) / (P.kappa * zm);
    double zz = zm + dvd, lp = P.kappa * zz * (1 - exp(-zz * sqrt(tf) / (A_vd * P.nu))), l2 = lp * lp;
    if (l2 < 1e-30) return tf / P.nu;
    return (-P.nu + sqrt(P.nu * P.nu + 4 * l2 * tf)) / (2 * l2);
}
static void profiel_init(void) {
    for (int k = 0; k < NZ; k++) {
        zk[k] = P.z0 * exp(log(zmax / P.z0) * k / (NZ - 1));
        Uk[k] = P.ustar / P.kappa * log(zk[k] / P.z0);
        taup[k] = 0; tp_f[k] = 0;
    }
    if (profielmodel) {
        double us2 = P.ustar * P.ustar; Uk[0] = 0;
        for (int k = 1; k < NZ; k++) Uk[k] = Uk[k - 1] + dUdz(sqrt(zk[k] * zk[k - 1]), us2) * (zk[k] - zk[k - 1]);
    }
}
static inline double wind(double zr) {             /* zr = height above the local bed */
    if (zr <= P.z0) return 0.0;
    if (zr >= zmax) return Uk[NZ - 1] + P.ustar / P.kappa * log(zr / zmax);
    double f = log(zr / P.z0) / log(zmax / P.z0) * (NZ - 1);
    int k = (int)f; double w = f - k;
    return Uk[k] * (1 - w) + Uk[k + 1] * w;
}
/* 12 Oct.: WITHOUT A WIND LAYER PARALLEL TO THE BED (a boundary layer is a mental construct).
   (a) alfa_ref > 0: the height in the wind profile is measured from zref = alfa_ref * href + (1 - alfa_ref) * h_local,
       with href the bed smoothed over L_ref (from Python, zet_href); the imposed modulation counts with
       (1 - alfa_ref) (at alfa_ref = 1 the effect of the topography comes only from the geometry).
   (b) nblok > 0: the slowing down by the grains is computed per block of nblok columns (own wind profile per block,
       linearly interpolated between block centres) instead of averaged over the domain. The domain profile Uk stays for
       aerodynamic entrainment and output. Bed (height field) and creep (mobile layer) unchanged.
   Default alfa_ref = 0 and nblok = 0: as before (the article model). */
double alfa_ref = 0; int nblok = 0;
static double *href_arr = 0;
void zet_href(double *a) { href_arr = a; }
static double *Ukb = 0, *taupb = 0, *tpfb = 0; static int NB = 0, BW = 0, NBN = 0;
static inline double windp(const double *U, double zr) {
    if (zr <= P.z0) return 0.0;
    if (zr >= zmax) return U[NZ - 1] + P.ustar / P.kappa * log(zr / zmax);
    double f = log(zr / P.z0) / log(zmax / P.z0) * (NZ - 1);
    int k = (int)f; double w = f - k;
    return U[k] * (1 - w) + U[k + 1] * w;
}
static inline int blok(int i) { int b = i / BW; return b < NB ? b : NB - 1; }
static inline double windk(double zr, int i) {
    if (!Ukb) return wind(zr);
    double f = (i + 0.5) / BW - 0.5; int b0 = (int)floor(f); double w = f - b0;
    int c0 = ((b0 % NB) + NB) % NB, c1 = (c0 + 1) % NB;
    return (1 - w) * windp(Ukb + (int64_t)c0 * NZ, zr) + w * windp(Ukb + (int64_t)c1 * NZ, zr);
}
/* 18 Oct.: SEPARATION ENVELOPE (omh = 1; imposed wind shadow). omh_arr[i] = height (m) of the envelope above the
   local bed, from Python: e = max(0, Ze - Zs) with Ze = max_j (Zs(i - j) - j dx tan th_e), Zs the smoothed bed. The wind profile
   is measured from the envelope (hb + e): below the envelope no wind, above it the height above the envelope. Outside the
   shadow e = 0 and the model computes as before. Below the envelope (e > dz/2) there is no wind-driven sliding (pure wind effect);
   creep keeps the wind drag of the modulation there. Modulation and ceiling can be computed in Python on the envelope (OMH_MOD). */
int omh = 0; static double *omh_arr = 0;
void zet_omh(double *a) { omh_arr = a; }
static inline int md_(int i, int n) { int r = i % n; return r < 0 ? r + n : r; }
double stroom_hel(double zr, int i, int N_) {   /* 13 Nov.: STROOM; slope of the streamline at height zr (m) above column i, linear in z */
    if (!stroom_s) return 0.0; i = md_(i, N_); if (zr <= stroom_z[0]) return stroom_s[i];
    if (zr >= stroom_z[stroom_nz - 1]) return 0.0;
    int j = 0; while (j < stroom_nz - 2 && zr >= stroom_z[j + 1]) j++;
    double f = (zr - stroom_z[j]) / (stroom_z[j + 1] - stroom_z[j]);
    return (1 - f) * stroom_s[(int64_t)j * N_ + i] + f * stroom_s[(int64_t)(j + 1) * N_ + i];
}
static inline double zref(int i) { if (omh && omh_arr) return hb(i) + omh_arr[md(i)]; return (alfa_ref > 0 && href_arr) ? alfa_ref * href_arr[md(i)] + (1 - alfa_ref) * hb(i) : hb(i); }
static inline int omh_bed(int i) { return omh && omh_arr && omh_arr[md(i)] > 0.5 * P.dz; }
static inline double modf_(int i) { if (!windmod) return 1.0; return alfa_ref > 0 ? 1 + (1 - alfa_ref) * (windmod[i] - 1) : windmod[i]; }
/* new profile from the grain momentum flux taup, eq. 29. The momentum flux is filtered
   in time (exponentially, time scale P.relax in s) because it comes from countable crossings;
   the profile itself follows the filtered flux directly (the air responds in ms). */
long n_begrensd = 0;
int tf_begrens = 1;   /* 6 Nov.: TF_BEGRENS: 0 = no safeguard (cap) on the wind */
long bp_tel[8] = {0, 0, 0, 0, 0, 0, 0, 0};   /* 6 Nov.: counters of known limitations */
void beperk_uit(long *u) { for (int q = 0; q < 8; q++) { u[q] = bp_tel[q]; bp_tel[q] = 0; } }
long begrensd_uit(void) { long r = n_begrensd; n_begrensd = 0; return r; }
static void profiel_bij(double tint) {
    double Un[NZ]; Un[0] = 0;
    double us2 = P.ustar * P.ustar, r = tint / P.relax; if (r > 1) r = 1;
    for (int k = 0; k < NZ; k++) { double tp = taup[k] / (tint * Lx * W * NY); tp_f[k] = (1 - r) * tp_f[k] + r * tp; tp_last[k] = tp_f[k]; taup[k] = 0; }
    for (int k = 1; k < NZ; k++) {
        double tp = 0.5 * (tp_f[k] + tp_f[k - 1]);
        double tf = us2 - tp / P.rho_a; if (tf < 1e-6 * us2) tf = 1e-6 * us2;
        bp_tel[0]++;
        if (tf > us2) { n_begrensd++; bp_tel[1]++; if (tf_begrens) tf = us2; }   /* 5 Oct.: grains do not accelerate the air (domain-averaged tau_p >= 0); 6 Nov.: adjustable */
        double zm = sqrt(zk[k] * zk[k - 1]);
        Un[k] = Un[k - 1] + dUdz(zm, tf) * (zk[k] - zk[k - 1]);
    }
    for (int k = 0; k < NZ; k++) Uk[k] = Un[k];
    if (Ukb) for (int b = 0; b < NB; b++) {
        double Lb = (b < NB - 1 ? BW : N - (NB - 1) * BW) * P.dx;
        double *tq = taupb + (int64_t)b * NZ, *tf_ = tpfb + (int64_t)b * NZ, *U = Ukb + (int64_t)b * NZ;
        for (int k = 0; k < NZ; k++) { double tp = tq[k] / (tint * Lb * W); tf_[k] = (1 - r) * tf_[k] + r * tp; tq[k] = 0; }
        U[0] = 0;
        for (int k = 1; k < NZ; k++) {
            double tp = 0.5 * (tf_[k] + tf_[k - 1]);
            double tf = us2 - tp / P.rho_a; if (tf < 1e-6 * us2) tf = 1e-6 * us2;
            if (tf > us2 && tf_begrens) tf = us2;
            U[k] = U[k - 1] + dUdz(sqrt(zk[k] * zk[k - 1]), tf) * (zk[k] - zk[k - 1]);
        }
    }
}
void wind_bij(double *out, double zb) { for (int i = 0; i < N; i++) out[i] = windk(hb(i) + zb - zref(i), i) * modf_(i); }

/* ---- grains in the air ---- */
static double *px, *pz, *pvx, *pvz; static uint8_t *pk; static int M, Mcap;
static uint8_t *schaduw; static double *I_cnt = 0, *I_ema = 0, *I_r = 0; static double *Ic0 = 0, *Ie0 = 0, *Ir0 = 0;
static inline void zet_rij(int r) {
    rij = r; h = h0 + (int64_t)r * N; col = col0 + (int64_t)r * N * HM;
    if (wm0) windmod = wm0 + (int64_t)r * N;
    if (sch0) schaduw = sch0 + (int64_t)r * N;
    if (Ic0) { I_cnt = Ic0 + (int64_t)r * N; I_ema = Ie0 + (int64_t)r * N; I_r = Ir0 + (int64_t)r * N; }
}
static inline int rij_van(double y) { int r = (int)floor(y / dyr); r %= NY; return r < 0 ? r + NY : r; }
/* 16 Oct.: AVALANCHE IN TWO DIRECTIONS (quasi-3D, lat_law): as lawine(), but also across: slope over Wl rows above the
   angle of repose or toppling to the neighbouring row; the grain moves to the lower adjacent column or row. Stack of (row, column). */
static int stapel3[1 << 17]; static long law3_guard = 0, law3_calls = 0;
void law3_uit(long *u) { u[0] = law3_calls; u[1] = law3_guard; law3_calls = law3_guard = 0; }
static void lawine3(int i0) {
    int r0 = rij, n = 0, Wl = P.Wl;
    for (int q = -Wl; q <= Wl; q++) { stapel3[n++] = r0 * N + md(i0 + q); stapel3[n++] = ((r0 + q + NY * Wl) % NY) * N + i0; }
    long guard = 0;
    while (n > 0 && guard++ < 40000) {
        int g = stapel3[--n], r = g / N, i = g % N; zet_rij(r);
        int kt_ = h[i] > 0 ? col[(int64_t)i * HM + h[i] - 1] : 0;
        double tr_ = tanrk[kt_] > 0 ? tanrk[kt_] : P.tanr, ex = 0; int bx = 0, by = 0;
        for (int s = -1; s <= 1; s += 2) {
            double e = ((h[i] - h[md(i + s * Wl)]) * P.dz - tr_ * Wl * P.dx) / (Wl * P.dx);
            double t = (double)(h[i] - h[md(i + s)] - (P.toren - 1));
            double mm = e > t ? e : t; if (mm > ex + 1e-12) { ex = mm; bx = s; by = 0; }
            int rw = (r + s * Wl + NY * Wl) % NY, r1 = (r + s + NY) % NY;
            double ey = ((h[i] - h0[(int64_t)rw * N + i]) * P.dz - tr_ * Wl * dyr) / (Wl * dyr);
            double ty = (double)(h[i] - h0[(int64_t)r1 * N + i] - (P.toren - 1));
            mm = ey > ty ? ey : ty; if (mm > ex + 1e-12) { ex = mm; bx = 0; by = s; }
        }
        if (ex <= 0 || h[i] <= 1) continue;
        int rj = by ? (r + by + NY) % NY : r, ij = bx ? md(i + bx) : i;
        if (h0[(int64_t)rj * N + ij] >= h[i]) continue;               /* only to a lower neighbour */
        int c0_ = ctx; ctx = 2; uint8_t kk = pak(i); zet_rij(rj); leg(ij, kk); ctx = c0_; lat_tel[by ? 1 : 2] += 1; if (by) lat_tel_bij(2, r, i, rj, ij);
        if (n + 8 * Wl + 12 < (1 << 17))
            for (int q = -Wl; q <= Wl; q++) {
                stapel3[n++] = r * N + md(i + q); stapel3[n++] = ((r + q + NY * Wl) % NY) * N + i;
                stapel3[n++] = rj * N + md(ij + q); stapel3[n++] = ((rj + q + NY * Wl) % NY) * N + ij;
            }
    }
    law3_calls++; if (guard >= 40000) law3_guard++;
    zet_rij(r0);
}


/* 5 Oct.: JUMP MEASUREMENT. Per grain in the air: horizontal distance travelled and type of launch
   (0 ejection, 1 rebound, 2 aerodynamic). On impact the distance goes into a histogram per type
   (bins of 1 mm, 0-1999 mm, last bin = overflow). */
#define NSB 2000
static double *p_afst = 0, *p_hmax = 0; static uint8_t *p_typ = 0; static int l_typ = 0;
#define NHB 1000
long hoogte_hist[3][NHB];   /* highest point above the local bed per jump, bins of 0.5 mm */
void hoogte_uit(long *uit) { for (int t = 0; t < 3; t++) for (int b = 0; b < NHB; b++) { uit[t * NHB + b] = hoogte_hist[t][b]; hoogte_hist[t][b] = 0; } }
double ez_max = 1e9;   /* 5 Oct.: upper limit of the vertical restitution (1 = do not bounce higher than arrived) */
long sprong_hist[3][NSB];
void sprong_reset(void) { for (int t = 0; t < 3; t++) for (int b = 0; b < NSB; b++) sprong_hist[t][b] = 0; }
void sprong_uit(long *uit) { for (int t = 0; t < 3; t++) for (int b = 0; b < NSB; b++) uit[t * NSB + b] = sprong_hist[t][b]; }
/* 16 Oct.: COUPLING TEST (requires sprong_aan = 1). p_D = horizontal displacement since the grain left the bed
   (ejection or entrainment by the wind); carried over on rebound. On a permanent landing in column j:
     koppeling 0: deposit in j (physical);
     koppeling 1: PERTURBED DISPLACEMENT: departure x0 = x_j - D; deposit at x0 + D', with D' an earlier
                  measured displacement of the same class (ring buffer of KOP_NB, index from its own LCG: the
                  main sequence of random numbers is untouched); while the buffer holds < KOP_MIN: in j;
     koppeling 2: DECOUPLED IDENTITY: deposit in j, but with the class of the PREVIOUS permanent landing
                  (anywhere in the domain); the class of this grain waits in a slot (one grain 'in transit').
   Mass: 1 moves only the place; 2 conserves bed + air + slot per class exactly. kop_tel per class:
   [0] landings, [1] sum D, [2] sum D', [3] deposited of this class; kop_hist[0/1][class][mm] D and D'. */
static double *p_D = 0; static double l_D = 0;
/* 17 Oct.: JUMP DISTRIBUTION BY DEPARTURE POSITION (for the hybrid model; requires sprong_aan = 1). p_T = time since the grain left
   the bed (carried over on rebound). On each permanent landing: departure column x0 = x_land - D, folded with sp3_fold columns
   and binned in bins of sp3_vak columns; counter per class, departure bin and distance bin (SP3_DB m, from -SP3_DNEG), plus the sum of the flight time.
   bevr_top = 1 (diagnostics): on a frozen bed an ejection gets the class of the top bed grain (without changing the bed). */
static double *p_T = 0; static double l_T = 0;
int sp3_aan = 0, sp3_fold = 268, sp3_vak = 4, bevr_top = 0;
#define SP3_NV 270
#define SP3_ND 2501
#define SP3_DB 0.4e-3
#define SP3_DNEG 20e-3
static long sp3_n[3][SP3_NV][SP3_ND]; static double sp3_t[3][SP3_NV][SP3_ND]; static long sp3_ov[3][SP3_NV];   /* landing column (folded) of flights beyond the last distance bin */
void sp3_ov_uit(long *o) { for (int c = 0; c < 3; c++) for (int v = 0; v < SP3_NV; v++) { o[c * SP3_NV + v] = sp3_ov[c][v]; sp3_ov[c][v] = 0; } }
void sp3_reset(void) { for (int c = 0; c < 3; c++) for (int v = 0; v < SP3_NV; v++) for (int b = 0; b < SP3_ND; b++) { sp3_n[c][v][b] = 0; sp3_t[c][v][b] = 0; } }
void sp3_uit(long *n_, double *t_) { for (int c = 0; c < 3; c++) for (int v = 0; v < SP3_NV; v++) for (int b = 0; b < SP3_ND; b++) { n_[(c * SP3_NV + v) * SP3_ND + b] = sp3_n[c][v][b]; t_[(c * SP3_NV + v) * SP3_ND + b] = sp3_t[c][v][b]; } }
int koppeling = 0;
#define KOP_NB 4096
#define KOP_MIN 200
static double kop_buf[3][KOP_NB]; static long kop_n[3] = {0, 0, 0};
static unsigned long long kop_lcg = 88172645463325252ULL;
static int kop_slot = -1;
double kop_tel[3][4]; long kop_hist[2][3][NSB];
void kop_reset(void) { for (int c = 0; c < 3; c++) { for (int q = 0; q < 4; q++) kop_tel[c][q] = 0; for (int b = 0; b < NSB; b++) kop_hist[0][c][b] = kop_hist[1][c][b] = 0; } }
void kop_uit(double *t, long *hh) { for (int c = 0; c < 3; c++) for (int q = 0; q < 4; q++) t[c * 4 + q] = kop_tel[c][q];
    for (int a = 0; a < 2; a++) for (int c = 0; c < 3; c++) for (int b = 0; b < NSB; b++) hh[(a * 3 + c) * NSB + b] = kop_hist[a][c][b]; }
int kop_slot_uit(void) { return kop_slot; }
static int kop_bin(double D) { int b = (int)(D * 1e3); if (b < 0) b = 0; if (b >= NSB) b = NSB - 1; return b; }
int M_max = 0; long n_vol = 0;                     /* 5 Oct.: highest number in the air; refused launches */
static void lanceer(int i, double v, double th, uint8_t k) {
    if (M >= Mcap) { n_vol++; if (!P.bevroren) leg(i, k); return; }   /* no leak: grain back into the bed */
    if (M + 1 > M_max) M_max = M + 1;
    px[M] = (i + 0.5) * P.dx; pz[M] = hb(i) + 0.5 * dkl(k) + 1e-6;
    pvx[M] = v * cos(th); pvz[M] = v * sin(th); pk[M] = k;
    if (NY > 1) { py[M] = (rij + 0.5) * dyr; pvy[M] = 0; if (p_r0) { if (l_typ == 1) { p_r0[M] = l_r0; p_x0[M] = l_x0; } else { p_r0[M] = rij; p_x0[M] = i; } } }
    if (p_afst) { p_afst[M] = 0; p_hmax[M] = 0; p_typ[M] = (uint8_t)l_typ; p_D[M] = l_typ == 1 ? l_D : 0; p_T[M] = l_typ == 1 ? l_T : 0; }
    if (p_c0) p_c0[M] = l_typ == 1 ? l_c0 : i;   /* 30 Oct.: FLUXTEL */
    if (p_h0) p_h0[M] = l_typ == 1 ? l_h0 : 0;   /* 31 Oct.: MOD_LH */
    if (p_rb) p_rb[M] = 0; if (p_rbj) p_rbj[M] = -1;   /* 4 Nov.: ART_KR, ART_EJ */
    if (hk_p) hk_p[M] = (uint8_t)(2 * l_typ + 1 + (pvx[M] <= 0));   /* 7 Nov.: HKTEL */
    M++;
}

static long n_ins, n_reb, n_ej, n_aero;
static int32_t *h_ins, *h_ej;                      /* counts per column (frozen measurement) */

/* ANGLE-DEPENDENT COLLISION (3 Oct., hoekmodel = 1). The impact angle theta is measured relative to the
   LOCAL bed (slope over 5 columns). Collision experiments (Beladjine et al. 2007; parametrisation as
   used by Lammel et al. 2017): total restitution e = 0.87 - 0.72 sin theta, vertical
   restitution e_z = 0.30 / sin theta - 0.15. Rebound: COMSALT probability B(1 - exp(-gam v)) AND the
   grain must keep at least sqrt(2 g d) normal to the bed (Lammel et al. eq. 41). Rebound angle
   relative to the bed ~ Exp with mean asin(e_z sin theta / e). Number of ejections:
   COMSALT a v / sqrt(g D) times (1 - e^2) / C_vlak, with C_vlak = <1 - e^2> over the impacts on a
   flat bed (so that the flat-bed mean stays the same; energy to the bed ~ 1 - e^2, Lammel
   et al. eq. 39). Ejection angles relative to the local bed. */
/* CREEP (3 Oct.): an impact pushes on average p_cr grains (Poisson) from the column hit (and the next)
   downwind; the grain rolls on column by column with probability 1 - dx/l_cr per column and stops in front of a
   step of >= 2 layers upward. p_cr was first chosen so that creep is 25 % of the total transport on
   a flat bed (Bagnold 1941); l_cr = 6.2 mm (as in the first grain automaton). */
double p_cr = 0.0, l_cr = 6.2e-3, l_cr_grof = 16e-3;
double l_schaduw = -1;   /* 5 Oct.: creep distance in the shadow (m); -1 = one grain diameter */
static double som_kruip;                       /* sum of the creep distances (m) */
static double som_kruip_m = 0;                 /* 29 Oct.: the same, weighted with m_k / m_cel (KORRELMASSA) */
/* CREEP VERSION 2 (3 Oct., kruipmodel = 2): a push always moves the top grain one column;
   after that it rolls on per column with probability exp(-dx / l), with
     l = l_cr * tan(phi_ref) / max(tan(phi) + s, 0.1 tan(phi_ref))  in the impact zone,
     l = d_k (one grain diameter)                                     in the shadow (no impacts),
   phi = 61.5 degrees (D/K)^-0.3 the pivot angle of grain D on bed grain K (Miller & Byrne 1966, natural
   sand; via Buffington et al. 1992), phi_ref = phi(1), s = slope in the rolling direction (upward > 0).
   Shadow: horizon slope > tan 12 degrees (map from Python, every second). */
/* CREEP VERSION 3 (5 Oct., kruipmodel = 3): fine grains (class 0) as version 1 (creep distance l_cr),
   coarse grains (class >= 1) as version 2 (pivot angle, stop in the shadow) with l_cr_grof. */
int kruipmodel = 1;
int tilt_plus = 0;   /* 5 Nov.: TILT_PLUS: 1 = take-off angle relative to the horizontal = angle relative to the bed + tilt (sign correction) */
static inline double vertrek_h(double th_bed, double tilt, int ej) {
    if (!tilt_plus) { double g = th_bed - tilt; return g < 0.02 ? 0.02 : g; }
    if (th_bed < 0.02) th_bed = 0.02; if (ej && th_bed > 3.1216) th_bed = 3.1216;
    return th_bed + tilt;
}
double vertrekhoek(double th_bed, double tilt, int ej) { return vertrek_h(th_bed, tilt, ej); }   /* for the analytical test */
int hoek_hel = 1, kruip_hel = 1;   /* 15 Oct. (symmetry diagnostics): 0 = impact angle relative to the horizontal instead of the local bed; 0 = creep distance coarse without slope term */
int kruip_wind = 0;   /* 4 Oct.: wind drag on rolling grains (Manukyan & Prigozhin 2009): creep distance x (modulation factor)^2 */
int aero_lokaal = 0; double zeta_a = 1e5;
void zet_schaduw(uint8_t *s_) { schaduw = s_; sch0 = s_; }
static inline double tanpiv(double D, double K) { return tan(61.5 * 3.141592653589793 / 180 * pow(D / K, -0.3)); }
/* 6 Oct.: CREEP STOP RULE (kruip_stop = 1): a rolling grain stops where the smoothed slope
   upward (over 5 columns ahead of the grain) is greater than tan(angle of repose), instead of at a step of >= 2 layers. */
int kruip_stop = 0; double kruip_stop_tan = -1;   /* 8 Oct.: kruip_stop 3 = fine step rule, coarse slope rule; kruip_stop_tan > 0: own threshold */
/* 10 Oct.: IMPACT RULE FOR CREEP (kruip_inslag = 1): each creep distance (fine, coarse, also in the shadow) is multiplied per column
   by the local impact rate r = I(x) / <I>: I is the number of impacts per column per second,
   time-averaged (exponentially, time scale tau_I) and smoothed over glad_I columns; <I> is the domain mean;
   r capped at r_max. Hypothesis: a creeping grain moves only as long as it is pushed; because of the
   inertia of the saltating grains the impact rate decreases towards the crest and strongly in the lee. Off in the article model. */
int kruip_inslag = 0; double tau_I = 5.0, r_max = 3.0; int glad_I = 5;
int inslag_steil = 0;   /* 6 Nov.: INSLAG_STEIL: 1 = a grain that cannot hit the slope from above hits a flat step (tilt 0) */
int hoek_lsq = 0;   /* 6 Nov.: HOEK_LSQ_KOL: > 0 = slope at an impact as the least-squares slope over +-hoek_lsq columns */
int kb_lagen = 2, hoek_hw = 2;   /* 26 Oct.: creep blocking in layers, half width of the impact slope in columns (from Python, from mm) */   /* tau_I 5 s: noise per column ~15 %, ripple moves ~1 mm per 5 s */
double helling_bed(int j) {   /* 6 Nov.: slope relative to the x-axis at column j as botsing() uses it */
    if (hoek_lsq <= 0) return (hb(j + hoek_hw) - hb(j - hoek_hw)) / (2 * hoek_hw * P.dx);
    double s = 0, n2 = 0;
    for (int q = -hoek_lsq; q <= hoek_lsq; q++) { s += q * hb(j + q); n2 += (double)q * q; }
    return s / (n2 * P.dx);
}
static int I_N = 0, I_init = 0;
void inslag_r_uit(double *u) { for (int i = 0; i < I_N; i++) u[i] = I_r ? I_r[i] : 1.0; }
void inslag_reset(void) { I_init = 0; }
/* kruip_inslag = 2 (10 Oct., C19): only coarse grains (class >= 1) follow the impact rule */
static void inslag_bij1(double dt_);
static inline double r_ins(int i, int k) { return ((kruip_inslag == 1 || (kruip_inslag == 2 && k >= 1)) && I_r) ? I_r[i] : 1.0; }
static void inslag_bij(double dt_) {
    if (!kruip_inslag || !I_cnt) return;
    if (NY > 1) { int r0_ = rij, ini = I_init; for (int r = 0; r < NY; r++) { zet_rij(r); I_init = ini; inslag_bij1(dt_); } zet_rij(r0_); return; }
    inslag_bij1(dt_);
}
static void inslag_bij1(double dt_) {
    {
    double a = dt_ / tau_I; if (a > 1) a = 1;
    if (!I_init) { double s = 0; for (int i = 0; i < N; i++) s += I_cnt[i]; s /= N * dt_; for (int i = 0; i < N; i++) I_ema[i] = s; I_init = 1; }
    for (int i = 0; i < N; i++) { I_ema[i] = (1 - a) * I_ema[i] + a * I_cnt[i] / dt_; I_cnt[i] = 0; }
    double m = 0; for (int i = 0; i < N; i++) m += I_ema[i]; m /= N;
    int hw = glad_I / 2;
    for (int i = 0; i < N; i++) {
        double s = 0; for (int q = -hw; q <= hw; q++) s += I_ema[md(i + q)];
        double r = m > 0 ? s / (2 * hw + 1) / m : 1.0; I_r[i] = r > r_max ? r_max : r;
    }
    }
}
int kruip_terug = 0; static int kr_dir = 1;   /* 8 Nov.: KRUIP_TERUG: rolling direction of a creep push (+1 downwind, -1 upwind) */
static inline int kruip_blok(int jj, int nx, int k) {
    int m = kruip_stop; if (m == 3) m = k >= 1 ? 1 : 0;
    if (!m) return h[nx] - h[jj] >= kb_lagen;
    double sl = kr_dir > 0 ? (hb(nx + 2) - hb(nx - 3)) / (5 * P.dx) : (hb(nx - 2) - hb(nx + 3)) / (5 * P.dx);   /* 8 Nov.: in the rolling direction */
    return hel_d(sl, 2) > (kruip_stop_tan > 0 ? kruip_stop_tan : P.tanr);
}
double kb_tel[4]; void kb_uit(double *u) { for (int q = 0; q < 4; q++) { u[q] = kb_tel[q]; kb_tel[q] = 0; } }   /* 27 Oct.: creep blocking counter */
static inline int kbt(int r, int s) { kb_tel[2 * s] += 1; if (r) kb_tel[2 * s + 1] += 1; return r; }
double rol_mud = 0; int rol_blok = 0;   /* 27 Oct.: dynamic friction and block slope (ROL_MUD) */
int rolregel = 0, rol_ns = 2, rol_nmin = 1; double rol_lcr = 4.1e-3; int rt_n1 = 2; double rt_tel[10]; void rt_uit(double *u) { for (int q = 0; q < 10; q++) { u[q] = rt_tel[q]; rt_tel[q] = 0; } }   /* 27 Oct.: rolling rule in mm (ROLREGEL) */
int kb_n = 1;   /* 27 Oct.: creep blocking per distance (KB_AFST): test over kb_n columns, once per kb_n columns away */
static inline int kbn(int jj, int nx, int k, int off, int s) { if (kb_n <= 1) return kbt(kruip_blok(jj, nx, k), s); if (off % kb_n) return 0; return kbt(kruip_blok(jj, md(jj + kr_dir * kb_n), k), s); }
/* 11 Oct.: HORIZONTAL MOMENTUM BUDGET FOR CREEP (kruip_impuls = 2, C21; the vertical part of the impact
   goes into the bed/the earth). Budget Bx = m_in (vx_in - vx_back) of the impacting grain. Each push carries a fixed
   horizontal momentum J = m_fine * vej_m (as long as Bx lasts; the last push gets the rest). The pushed grain
   gets speed J / m_k; its creep distance is multiplied by kr_fac = J / (m_k vej_m) (= 1 for a fine
   grain with a full push; coarse 334 um ~0.15, 488 um ~0.05). Ejections keep their own (scalar) budget. */
extern int kruip_impuls;
static double kr_J = 0, kr_vej = 1, kr_fac = 1.0;
double som_fac[2] = {0, 0}; long n_fac[2] = {0, 0}, n_kr_rest = 0;
void fac_uit(double *u) { u[0] = som_fac[0]; u[1] = n_fac[0]; u[2] = som_fac[1]; u[3] = n_fac[1]; u[4] = n_kr_rest;
    som_fac[0] = som_fac[1] = 0; n_fac[0] = n_fac[1] = 0; n_kr_rest = 0; }
/* 16 Oct.: WIND-DRIVEN SLIDING AT THE BED (windrol). Per profile update and per column:
   local stress tau = phi rho_a u*_b^2 f_i^2 (f_i = windmod, imposed modulation x ceiling), with phi a turbulent
   peak factor (lognormal, mean 1, coefficient of variation wr_cv; drawn per column per update; correlation in
   time and space ignored). Threshold of the top grain k on bed grain kb at slope s (downwind, upward > 0):
   tau_c = rho_a ustk[k]^2 max(tanpiv(D_k, D_kb) + s, 0.1 tref) / tref  (on a flat bed of equal grains: the
   fluid threshold; coarse on fine moves more easily because of the smaller pivot angle). Rate: zeta_a (tau - tau_c) grains per m^2
   per s (as aero_lokaal). The grain moves column by column downwind as long as phi tau(nx) > tau_c
   (same phi: a gust), and stops in the shadow or in front of a step (kruip_blok). u*_b:
     windrol 1: u* at the bed from the model (profile); windrol 2: Owen (1964): u*_b = min(u*, 0.8 ustk[0]) (the article model).
   Mass: displaced only. Counters wr_stat[(q*3 + k)*N + i]: q 0 start, 1 stop (also on a frozen bed, then without
   displacement). */
int windrol = 0; double wr_cv = 0.4, wr_zeta = 1e5; long n_wr = 0; double som_wr = 0;
static double *wr_stat = 0;
void zet_wr_stat(double *a) { wr_stat = a; }
void wr_uit(double *u) { u[0] = n_wr; u[1] = som_wr; n_wr = 0; som_wr = 0; }
static double *wr_mod = 0; void zet_wr_mod(double *a) { wr_mod = a; }   /* 30 Oct.: MOD_FG, own wind factor for the sliding rule */
double wr_meng = 0.0;   /* 1 Nov.: WR_MENG; 0 = self-regulation direct (multiplying), 1 = slow (adding) */
static inline double wr_tau_bed(double phi, double us_b, double f) {
    if (wr_meng == 0.0) return phi * P.rho_a * us_b * us_b * f * f;   /* exactly the old expression */
    if (f <= 0.0) return 0.0;
    double tb0 = P.rho_a * us_b * us_b, tm = tb0 * f * f, ta = tb0 + P.rho_a * P.ustar * P.ustar * (f * f - 1.0);
    if (ta < 0.0) ta = 0.0;
    return phi * ((1.0 - wr_meng) * tm + wr_meng * ta);
}
double wr_tau_test(double phi, double us_b, double f) { return wr_tau_bed(phi, us_b, f); }
int wr_nomod = 0, wr_maxstap = 0; static double *gl_kol = 0; void zet_gl_kol(double *a) { gl_kol = a; }   /* 30 Oct.: GLIJTEL */
double wr_piv = 1.0;   /* 23 Oct.: threshold of wind-driven sliding: factor on tan(pivot angle); > 1 = more angular grains; 1 = unchanged */
static double wr_tau_c(int k, int kb, double sl) {
    double tref = tanpiv(1, 1), mu = wr_piv * tanpiv(dkl(k), dkl(kb)) + hel_d(sl, 4); if (mu < 0.1 * tref) mu = 0.1 * tref;
    return P.rho_a * ustk[k] * ustk[k] * mu / tref * (1 + 3 * cohesie);
}
static void wind_rollen(double us_b, double dtu) {
    double sg = sqrt(log(1 + wr_cv * wr_cv)), mu_l = -0.5 * sg * sg;
    for (int i = 0; i < N; i++) {
        if (h[i] <= 1 || gehecht(i)) continue;
        if (schaduw && schaduw[i]) continue;
        int nx = md(i + 1), k = col[(int64_t)i * HM + h[i] - 1];
        int kb = h[i] > 1 ? col[(int64_t)i * HM + h[i] - 2] : k;
        double f = omh_bed(i) ? 0.0 : (wr_mod ? wr_mod[i] : ((windmod && !wr_nomod) ? windmod[i] : 1.0)), phi = exp(mu_l + sg * nrm());
        double tau = wr_tau_bed(phi, us_b, f);   /* 1 Nov.: WR_MENG */
        double tc = wr_tau_c(k, kb, (h[nx] - h[i]) * P.dz / P.dx);
        if (tau <= tc) continue;
        double lam = wr_zeta * (tau - tc) * P.dx * W * dtu;   /* wr_zeta default = Anderson & Haff (1e5 per N per s); the article model uses 1e9 per m^2 per s (WR_ZETA) */
        int nr_ = 0; { double L_ = exp(-lam), p_ = 1; do { nr_++; p_ *= U(); } while (p_ > L_); nr_--; }
        if (nr_ == 0) continue;   /* at most one sliding grain per column per update */
        int jj = i, stap = 0;
        while (stap < 2000) {
            if (wr_maxstap > 0 && stap >= wr_maxstap) break;   /* 30 Oct.: A2 */
            int n2 = md(jj + 1);
            if (kbn(jj, n2, k, stap, 1)) break;
            if (schaduw && schaduw[n2]) { jj = n2; stap++; break; }       /* slides into the shadow and stays */
            double f2 = omh_bed(n2) ? 0.0 : (wr_mod ? wr_mod[n2] : ((windmod && !wr_nomod) ? windmod[n2] : 1.0));
            int kb2 = h[n2] > 0 ? col[(int64_t)n2 * HM + h[n2] - 1] : k;
            if (wr_tau_bed(phi, us_b, f2) <= wr_tau_c(k, kb2, (h[n2] - h[jj]) * P.dz / P.dx)) break;   /* 1 Nov. */
            jj = n2; stap++;
        }
        if (stap == 0) continue;
        n_wr++; som_wr += stap * P.dx; if (ft_kol) ft_bij(1, i, jj, stap * P.dx);   /* 30 Oct.: FLUXTEL */
        if (gl_kol) { gl_kol[i] += 1; gl_kol[N + i] += stap * P.dx * 1e3; gl_kol[2 * N + jj] += 1; }   /* 30 Oct. */
        if (kam_hist && kam_vlag[i]) kam_bij(1, stap);
        if (wr_stat) { wr_stat[k * N + i] += 1; wr_stat[(3 + k) * N + jj] += 1; }
        if (!P.bevroren) { ctx = 1; gp_wr = 1; uint8_t kk = pak(i); gp_wr = 0; ctx = 0; lawine(i); ctx = 1; ld_wr = 1; gp_wr = 1; leg(jj, kk); gp_wr = 0; ctx = 0; lawine(jj); ld_wr = 0; }   /* 27 Oct.: ld_wr (counter) */
    }
}
/* 16 Oct.: lateral deflection (lat_kr): to the lower neighbouring row if it lies lower, with probability |s_y| / tan pivot
   (capped at 1); switches the row pointer. Only in quasi-3D. */
static inline void kruip_opzij(int nx, int k) {
    int ru = (rij + 1) % NY, rd = (rij + NY - 1) % NY;
    int hu = h0[(int64_t)ru * N + nx], hd = h0[(int64_t)rd * N + nx];
    int rn = hu < hd ? ru : (hd < hu ? rd : (U() < 0.5 ? ru : rd)); int hn = hu < hd ? hu : hd;
    if (hn >= h[nx]) return;
    double sy = (h[nx] - hn) * P.dz / dyr;
    int kb = h[nx] > 0 ? col[(int64_t)nx * HM + h[nx] - 1] : k;
    double p_ = sy / tanpiv(dkl(k), dkl(kb)); if (p_ > 1) p_ = 1;
    if (U() < p_) { zet_rij(rn); lat_tel[0] += 1; }
}
static void lawine3(int i0);
static void kruip(int j) {
    if (h[j] <= 1) return;
    if (gehecht(j)) return;
    if (stug && U() * stugf(j) > 1.0) return;
    int k = col[(int64_t)j * HM + h[j] - 1];
    kt_kl = k;   /* 7 Nov.: KRTEL */
    if (lanceer_duw && kr_u > 0 && !P.bevroren && NY == 1 && kr_dir > 0) {
        n_duw_u++;
        double m0_ = 0, m1_ = 0; for (int q_ = -2; q_ <= 2; q_++) { m0_ += h[md(j + q_)]; m1_ += h[md(j + 5 + q_)]; }   /* bed smoothed over 2 mm: shape, no grain roughness */
        double drop = (m0_ - m1_) / 5.0 * P.dz, s_ = 5 * P.dx, gz_ = terrein != 0 ? P.g * cos(terrein) : P.g;
        if (drop > 0 && drop * 2 * kr_u * kr_u > gz_ * s_ * s_) {          /* the bed drops away faster than the grain falls */
            double u_ = kr_u; kr_u = 0;
            ctx = 4; uint8_t kk = pak(j); ctx = 0; lawine(j);
            l_typ = 3; lanceer(j, u_, 0.02, kk); l_typ = 0; n_lduw++;
            return;
        }
    }
    int r_start = rij; kt_stap = 0;   /* 7 Nov.: KRTEL: picked up from here (0 = blocked) */
    kr_fac = 1.0;
    if (kruip_impuls == 2) { kr_fac = kr_J / (mk[k] * kr_vej); int c_ = k >= 1; som_fac[c_] += kr_fac; n_fac[c_]++; }
    int jj, stap;
    if (kruipmodel == 2 || (kruipmodel == 3 && k >= 1) || rolregel) {
        double D = dkl(k), tref = tanpiv(1, 1), lc = (rolregel && k == 0) ? rol_lcr : (kruipmodel == 3 ? l_cr_grof : l_cr);
        if (rolregel) { jj = md(j + rol_nmin); stap = rol_nmin; } else { jj = md(j + kr_dir); stap = 1; }   /* 27 Oct.: first displacement in mm */
        if (kbn(j, jj, k, 0, 0)) return;
        while (stap < 100000) {
            int nx = md(jj + kr_dir);
            if (kbn(jj, nx, k, stap, 0)) break;
            double l;
            if (schaduw && schaduw[nx]) l = l_schaduw > 0 ? l_schaduw : D;
            else {
                int kb = h[nx] > 0 ? col[(int64_t)nx * HM + h[nx] - 1] : k;
                double sl;
                if (rolregel && rol_blok) { double a_ = 0, b_ = 0; for (int q_ = 0; q_ < rol_ns; q_++) { a_ += h[md(jj + q_)]; b_ += h[md(jj + rol_ns + q_)]; }   /* 27 Oct.: block slope */
                    sl = hel_d((b_ - a_) / rol_ns * P.dz / (rol_ns * P.dx), 2); }
                else sl = rolregel ? hel_d((h[md(jj + rol_ns)] - h[jj]) * P.dz / (rol_ns * P.dx), 2) : hel_d((h[nx] - h[jj]) * P.dz / P.dx, 2);   /* 27 Oct.: slope over ROL_S_MM */
                if (rolregel && rol_mud > 0) { double mu = rol_mud + sl; if (mu < 0.05 * rol_mud) mu = 0.05 * rol_mud; l = lc * rol_mud / mu; }   /* 27 Oct.: dynamic friction */
                else { double mu = tanpiv(D, dkl(kb)) + ((kruip_hel || rolregel) ? sl : 0.0); if (mu < 0.1 * tref) mu = 0.1 * tref;
                l = lc * tref / mu; }
                if (kruip_wind && windmod && kr_dir > 0) l *= modf_(nx) * modf_(nx);
            }
            l *= r_ins(nx, k) * kr_fac;
            if (U() >= exp(-P.dx / l)) break;
            jj = nx; stap++;
            if (NY > 1 && lat_kr) kruip_opzij(jj, k);
        }
    } else {
        jj = j; stap = 0;
        while (stap < 100000) {
            int nx = md(jj + kr_dir);
            if (kbn(jj, nx, k, stap, 0)) break;
            double le = l_cr;
            if (kruip_wind && windmod && kr_dir > 0) le *= modf_(nx) * modf_(nx);
            le *= r_ins(nx, k) * kr_fac;
            double q = 1.0 - P.dx / le; if (q < 0) q = 0;
            if (U() >= q) break;
            jj = nx; stap++;
            if (NY > 1 && lat_kr) kruip_opzij(jj, k);
        }
        jj = md(j + kr_dir * stap);
        if (stap == 0) { if (NY > 1) zet_rij(r_start); return; }
    }
    kt_stap = stap;   /* 7 Nov.: KRTEL */
    som_kruip += kr_dir * stap * P.dx; if (ft_kol) ft_bij(0, j, jj, kr_dir * stap * P.dx);   /* 30 Oct.: FLUXTEL */
    som_kruip_m += kr_dir * stap * P.dx * (mk[k] / (P.rho_p * 3.141592653589793 / 6 * P.d * P.d * P.d));   /* 29 Oct. */
    { int n1_ = rt_n1 > 0 ? rt_n1 : 1; double s_ = (h[md(j + n1_)] - h[j]) * P.dz / (n1_ * P.dx); int c_ = s_ < -0.2 ? 0 : s_ < -0.05 ? 1 : s_ <= 0.05 ? 2 : s_ <= 0.2 ? 3 : 4;
      rt_tel[2 * c_] += 1; rt_tel[2 * c_ + 1] += stap * P.dx; }   /* 27 Oct.: counter of creep distance per initial slope (ROLREGEL) */
    if (kam_hist && kam_vlag[j]) kam_bij(0, stap);
    if (kol_kr) kol_kr[j]++;
    if (h_ej) h_ej[j]++;
    if (h_ins) h_ins[jj]++;
    if (kol_kr_aan) kol_kr_aan[jj]++;
    if (NY > 1) {
        int r_eind = rij;
        if (r_eind != r_start) lat_tel_bij(1, r_start, j, r_eind, jj);
        if (!P.bevroren) { zet_rij(r_start); ctx = 1; uint8_t kk = pak(j); ctx = 0; lawine(j); zet_rij(r_eind); ctx = 1; leg(jj, kk); ctx = 0; lawine(jj); }
        zet_rij(r_start);
        return;
    }
    if (!P.bevroren) { ctx = 1; uint8_t kk = pak(j); ctx = 0; lawine(j); ctx = 1; leg(jj, kk); ctx = 0; lawine(jj); }
}
void kruip_toets(int j, int richting) { kr_dir = richting; kruip(j); kr_dir = 1; }   /* 8 Nov.: KRUIP_TERUG, analytical test */
double kruip_uit(void) { double r = som_kruip; som_kruip = 0; return r; }
double kruip_m_uit(void) { double r = som_kruip_m; som_kruip_m = 0; return r; }   /* 29 Oct.: KORRELMASSA */
double mk_uit(int k) { return mk[k]; }   /* 29 Oct.: mass per class (set on each call of the grain loop) */
int hoekmodel = 0;
double C_vlak = 1.0;
static double som_f, n_f;                     /* measure <1 - e^2> */
long hoek_hist[2][180];   /* 5 Oct.: impact angle relative to the horizontal, bins of 0.5 deg; [0] all, [1] v > 1 m/s */
void hoek_uit(long *u) { for (int a = 0; a < 2; a++) for (int b = 0; b < 180; b++) { u[a * 180 + b] = hoek_hist[a][b]; hoek_hist[a][b] = 0; } }
/* 6 Oct.: SPLASH FOLLOWING BELADJINE ET AL. 2007 / VALANCE ET AL. 2015 (splashmodel = 2): number of ejections
   ~ Poisson with mean s_ej (1 - e^2) max(xi/sqrt(gd) - xi_c, 0) (slope 0.325 = 13/40 from Fig. 10b
   of Valance et al.: n_ej/(1-e^2) ~ 13 (xi/xi_c - 1) with xi_c = 40 sqrt(gd)); ejection angle circular-normal
   around 85 deg with s = pi/8. Ejection speed and momentum budget unchanged (COMSALT). splashmodel = 1: COMSALT. */
int splashmodel = 1; double xi_c = 40.0, s_ej = 0.325;
/* 11 Oct.: launch angle of ejections (splashmodel 2): circular-normal around ej_hoek with spread ej_s (rad).
   Default 85 deg / pi/8 (Beladjine, beads); C22: 50 deg (sand measurements Rice et al. 1995, Willetts & Rice 1989). */
double ej_hoek = 85 * 3.141592653589793 / 180, ej_s = 3.141592653589793 / 8;
/* 12 Oct.: DIRECTION OF THE MOMENTUM IN EJECTIONS (ej_hor = 1, C23): the vertical part of an ejection comes from
   the bed (the earth); only the horizontal part, m_e v_e cos(theta), is taken from the horizontal momentum loss
   m_in (vx_in - vx_back) of the impacting grain. Creep stays as in C19 (not paid). Default 0: scalar
   COMSALT budget (1 - e) v m. */
int ej_hor = 0;
double ej_vfac = 1.0;
/* 17 Oct.: EJECTIONS (symmetry diagnostics). ej_min = 1: an ejection is launched only if its speed normal to the bed
   is at least sqrt(2 g D) (D = diameter of the top bed grain at the impact): it must be able to pass over its neighbours; otherwise
   the grain stays and costs no momentum budget. ej_logn > 0: launch speed lognormal with the same mean and
   sigma(ln v) = ej_logn (Beladjine et al. 2007: lognormal) instead of exponential. Default 0: as before. */
int ej_min = 0; double ej_logn = 0.0; long n_ej_onder = 0;
/* 18 Oct.: DOSE-RESPONSE. ej_min_n: threshold as a multiple of the grain diameter (launch only if v_n >= sqrt(2 g n D)); ej_onder_kruip = 1:
   an ejection below the threshold becomes a creep push on the top grain (column j or j+1, like the pushes), with the existing creep rules. */
double ej_min_n = 1.0; int ej_onder_kruip = 0;
/* 18 Oct.: ej_lmin (m) > 0: launch only if the ballistic jump distance 2 v^2 sin(th) |cos(th)| / g (th relative to the bed) >= ej_lmin;
   otherwise as above (creep push or stay). */
double ej_lmin = 0.0;
/* 18 Oct.: WALL RULE (wand >= 1; wand = 2: also first the avalanche after removing an ejected grain, then launch, and only a DESCENDING grain can land): a rising grain (v_z >= 0) that enters a higher neighbouring column with its centre below the top
   of that column (+ D/2) hits the side: it loses its horizontal speed, stays in its own column and falls back there (no
   impact, no splash). Repairs the artefact that such a grain 'landed' on top of the higher column.
   MOMENTUM ALONG THE SURFACE (impuls_raak = 1): the bed absorbs the momentum component normal to the LOCAL surface, not the
   vertical one. Budget for ejections = m_in (v cos th_in - v_out cos th_out), angles relative to the local bed; an ejection costs m_e v_e cos th_e
   with th_e relative to the bed. Without slope equal to the horizontal budget of C23. */
int wand = 0, impuls_raak = 0; long n_wand = 0;
int begr_aan = 0;
double zij_tel[2][4]; void zij_uit(double *u) { for (int a = 0; a < 2; a++) for (int b = 0; b < 4; b++) { u[a * 4 + b] = zij_tel[a][b]; zij_tel[a][b] = 0; } }   /* 26 Oct.: top/side impact */
int bijval_eerst = 0, alleen_dalend = 0;   /* 26 Oct.: alleen_dalend: only a descending grain impacts */   /* 26 Oct.: let the hole fill first, then launch (without wall) */
/* 18 Oct.: EDGE COLLISION (wand = 3, the article model; otherwise as wand = 2): a grain that moves sideways into a higher neighbouring column while rising (centre below
   top + D/2) hits the edge of the step: contact normal from the corner to the
   centre, n = (-sign(vx) sqrt(1 - s^2), s), s = (z - top)/(D/2) capped at [-1, 1]; the normal speed is mirrored with
   restitution rand_e (chosen 0.5), the tangential one stays. The grain stays in its own column; no impact, no splash. */
double rand_e = 0.5;
long wand_uit(void) { long r = n_wand; n_wand = 0; return r; }
/* 18 Oct.: diagnostics: impacts of a still rising grain (v_z >= 0), per kind of jump (0 after ejection, 1 after rebound), with their jump length < 2 mm */
static double *stijg_kol = 0; void zet_stijg_kol(double *a) { stijg_kol = a; }   /* per column: [0] landings after ejection, [1] of which rising, [2] rising and on a higher column than the departure */
long stijg_tel[2][4]; void stijg_uit(long *u) { for (int a = 0; a < 2; a++) for (int b = 0; b < 4; b++) { u[a * 4 + b] = stijg_tel[a][b]; stijg_tel[a][b] = 0; } }   /* [3]: rising and in its own departure column */
long ej_onder_uit(void) { long r = n_ej_onder; n_ej_onder = 0; return r; }
double som_vi = 0; long n_vi = 0;
void inslag_uit(double *u) { u[0] = som_vi; u[1] = (double)n_vi; som_vi = 0; n_vi = 0; }
static int inj_n = 0; static double inj_f1 = 0, inj_f2 = 0;
void injecteer_zet(int n, double f1, double f2) { inj_n = n; inj_f1 = f1; inj_f2 = f2; }   /* executed at the start of draai() */
static void injecteer(int n, double f1, double f2) {
    for (int q = 0; q < n && M < Mcap; q++) {
        double r = U(); uint8_t k = r < f2 ? 2 : (r < f1 + f2 ? 1 : 0); if (nk == 1) k = 0;
        px[M] = U() * P.dx; pz[M] = hb(0) + 0.5 * dkl(k) - 5e-3 * log(1 - U());
        double vx = 1.5 + 0.5 * nrm(); if (vx < 0.3) vx = 0.3;
        pvx[M] = vx; pvz[M] = 0.5 * nrm(); pk[M] = k;
        if (p_afst) { p_afst[M] = 0; p_hmax[M] = 0; p_typ[M] = 2; p_D[M] = 0; p_T[M] = 0; }
        if (p_c0) p_c0[M] = -1;
        if (p_h0) p_h0[M] = 0;
        if (p_rb) p_rb[M] = 0; if (p_rbj) p_rbj[M] = -1;
        if (hk_p) hk_p[M] = 0;   /* 7 Nov.: HKTEL */
        M++; n_in++;
    }
}
void open_uit(long *u) { u[0] = n_in; u[1] = n_uit; }
int sprong_aan = 0; int diag_aan = 0; static double som_vin = 0; static long i0, r0, e0, a0;
/* 11 Oct.: IMPACT STATISTICS PER COLUMN (diagnostics only): [0] number, [1] sum of the descent angle relative to the horizontal (rad),
   [2] sum of the angle relative to the local bed (rad), [3] sum of the speed (m/s), [4] sum of (1 - e^2), [5] sum of the expected number of
   ejections (lam), [6] number of rebounds. Layout ins_stat[q * N + j]. */
static double *ins_stat = 0;
void zet_ins_stat(double *a) { ins_stat = a; }
/* 15 Oct.: JUMP STATISTICS PER COLUMN (diagnostics; requires sprong_aan = 1). Per class of the jump length of the
   impacting grain (0-2, 2-5, 5-20, 20-50, 50-107, > 107 mm) and per kind of jump (0 after ejection, 1 after rebound):
   impacts and permanent landings on the impact column, executed ejections on the ejection column. No random numbers.
   Layout hop_stat[(q * HOP_NC + c) * N + j], q 0 impact, 1 landing, 2 ejection; c 0-5 length, 6-7 kind. */
#define HOP_NC 8
static double *hop_stat = 0;
void zet_hop_stat(double *a) { hop_stat = a; }
static int hop_klasse(double afst) { double mm = afst * 1e3; return mm < 2 ? 0 : mm < 5 ? 1 : mm < 20 ? 2 : mm < 50 ? 3 : mm < 107.2 ? 4 : 5; }
int kruip_impuls = 0; long n_kr_geweigerd = 0, n_ej_geweigerd = 0; double som_kr_impuls = 0;
/* 24 Oct.: IMPULSE PUSH (kruip_impuls = 3; hypothesis: grains grazing the crest abrade the top). The number of pushes
   per impact is Poisson with mean p_cr * Bx / kr_bxref, with Bx = m_in (vx_in - vx_back) the momentum loss of the impacting grain
   along the x-axis (on a slope: along the terrain, downwind; Bx < 0 counts as 0) and kr_bxref the mean of Bx over all
   impacts on a flat bed of the best model (so the mean number of pushes stays the same). The push itself (creep distance) is unchanged.
   som_bx / n_bx: counters for Bx (all modes; no random numbers). */
double kr_bxref = 0; double som_bx = 0, som_nc = 0; long n_bx = 0;
void bx_uit(double *u) { u[0] = som_bx; u[1] = n_bx; u[2] = som_nc; som_bx = 0; n_bx = 0; som_nc = 0; }
void impuls_uit(double *u) { u[0] = n_kr_geweigerd; u[1] = n_ej_geweigerd; u[2] = som_kr_impuls; n_kr_geweigerd = n_ej_geweigerd = 0; som_kr_impuls = 0; }
static inline void draai_horizontaal(int b, double psi) {   /* horizontal speed (now along x) to direction psi */
    double vh = pvx[b]; pvx[b] = vh * cos(psi); pvy[b] = vh * sin(psi);
}
/* 20 Oct.: SUPPORT AND SORTING (hypothesis). steun_arr[i] = O, the missing support of a bed grain by
   downwind neighbours (0 in the middle of the stoss side, 1 where the bed no longer rises downwind; from Python, smoothed bed). An impact
   pushes with its part along the bed (v cos theta) towards that neighbour; where it is missing, fine bed grains shoot away more easily:
   ejections of fine grains x (1 + steun_a O cos theta), of coarse unchanged (thinning of candidates, without using budget).
   steun_a = 0: as before. */
double steun_a = 0; static double *steun_arr = 0; long n_steun_grof = 0;
/* 20 Oct.: LANDING RULE FINE ON COARSE: a fine grain (class 0) that would stay on a column with a coarse top
   grain, (a) rebounds after all with probability land_a (ordinary rebound rules), and of the rest (c) rolls with probability land_c to the lower
   neighbouring column (downwind first), (d) falls with probability land_d into the space between the coarse grains (ends up below the coarse grain), otherwise
   (b) it stays on top. All probabilities 0: as before. */
double land_a = 0, land_c = 0, land_d = 0; long n_land[4] = {0, 0, 0, 0};
/* 21 Oct.: PURE SORTING RULE (sort_a > 0): in an ejection with a coarse top grain, with probability min(1, sort_a O cos theta)
   the grain directly below shoots away if it is fine; the coarse one stays on top. Number of ejections and momentum budget unchanged (O from steun_arr). */
double sort_a = 0; long n_sort = 0;
void land_uit(long *o) { for (int q = 0; q < 4; q++) { o[q] = n_land[q]; n_land[q] = 0; } }
static inline int top_grof(int i) { return h[md(i)] > 0 && col[(int64_t)md(i) * HM + h[md(i)] - 1] >= 1; }
void zet_steun(double *a_) { steun_arr = a_; }
int kruip_tegen = 0;   /* 7 Nov.: KRUIP_TEGEN: 1 = no creep pushes after an impact against the wind */
int kaats_terug = 0; int bt_reb = 0;   /* 9 Nov.: KAATS_TERUG */
static void botsing(int a, int j) {
    double vx = pvx[a], vz = pvz[a];
    double psi_in = 0;                             /* 16 Oct.: quasi-3D: compute in the plane of the incoming trajectory */
    int r0_in = rij, x0_in = j;
    if (NY > 1) { psi_in = atan2(pvy[a], pvx[a]); vx = sqrt(pvx[a] * pvx[a] + pvy[a] * pvy[a]); if (p_r0) { r0_in = p_r0[a]; x0_in = p_x0[a]; } }
    double v = sqrt(vx * vx + vz * vz);
    double vx_terug = 0, vt_terug = 0;   /* horizontal speed / speed along the bed after rebound (0 if the grain stays) */
    som_vin += v; som_vi += v; n_vi++;
    { double al = atan2(-vz, vx > 1e-9 ? vx : 1e-9) * 180 / 3.141592653589793; int b = (int)(al * 2); if (b < 0) b = 0; if (b > 179) b = 179;
      hoek_hist[0][b]++; if (v > 1.0) hoek_hist[1][b]++; }
    if (kol_ins) kol_ins[j]++;
    const int kt_r = vx > 0 ? 0 : 1;   /* 7 Nov.: KRTEL */
    const int sr_ = (kaats_terug && NY == 1 && kt_r == 1) ? -1 : 1;   /* 9 Nov.: KAATS_TERUG: compute in the direction of motion */
    if (sr_ < 0) vx = -vx;
    if (hk_p) { int c_ = hk_p[a] & 15, z_ = (hk_p[a] & 16) ? 1 : 0; if (c_ > 8) c_ = 8;   /* 7 Nov.: HKTEL */
        hk_kol[((2 * kt_r + z_) * 9 + c_) * N + j] += 1; hk_s[2 * (hk_p[a] & 63) + kt_r] += 1; hk_p[a] = hk_p[M - 1]; }
    if (kt_kol) { kt_kol[kt_r * N + j] += 1; int b_ = (int)(v / 0.05); if (b_ > 39) b_ = 39; kt_hist[512 + 40 * kt_r + b_] += 1; kt_s[6 + kt_r] += v; }
    if (kruip_inslag && I_cnt) I_cnt[j] += 1;
    uint8_t k = pk[a];
    int art_ = (wand == 0 && vz >= 0);   /* 4 Nov.: ART_KR, ART_EJ: an artefact impact */
    if (p_rb) { if (p_rb[a] && rb_kol) rb_kol[N + j] += 1; p_rb[a] = p_rb[M - 1]; }
    if (p_rbj) p_rbj[a] = p_rbj[M - 1];   /* 6 Nov.: RBTEL */
    int hc = -1, ht = -1; double D_in = 0, T_in = 0, x_land = px[a]; int c0_in = -1; double h0_in = 0;
    if (kam_hist && p_afst && (p_typ[a] <= 1 || p_typ[a] == 3)) { int c0_ = md((int)floor((px[a] - p_afst[a]) / P.dx)); if (kam_vlag[c0_]) kam_bij(p_typ[a] == 3 ? 6 : 2 + p_typ[a], p_afst[a] / P.dx); }
    if (hop_stat && p_afst) { hc = hop_klasse(p_afst[a]); ht = 6 + (p_typ[a] == 1);   /* 6: after ejection or entrainment by the wind, 7: after rebound */ hop_stat[hc * N + j] += 1; hop_stat[ht * N + j] += 1; }
    if (p_afst) { int ty_ = p_typ[a] == 1; stijg_tel[ty_][0]++; if (vz >= 0) { stijg_tel[ty_][1]++; if (p_afst[a] < 2e-3) stijg_tel[ty_][2]++; if (md((int)floor((px[a] - p_afst[a]) / P.dx)) == j) stijg_tel[ty_][3]++; }
        if (stijg_kol && ty_ == 0) { stijg_kol[j] += 1; if (vz >= 0) { stijg_kol[N + j] += 1; int c0_ = md((int)floor((px[a] - p_afst[a]) / P.dx)); if (h[j] > h[c0_]) stijg_kol[2 * N + j] += 1; } } }
    if (p_afst) {
        int b = (int)(p_afst[a] * 1e3); if (b < 0) b = 0; if (b >= NSB) b = NSB - 1;
        sprong_hist[p_typ[a] > 2 ? 0 : p_typ[a]][b]++;   /* type 3 (launch by a push) counts here as ejection */
        int bh = (int)(p_hmax[a] * 2e3); if (bh < 0) bh = 0; if (bh >= NHB) bh = NHB - 1;
        hoogte_hist[p_typ[a] > 2 ? 0 : p_typ[a]][bh]++;
        D_in = p_D[a]; T_in = p_T[a]; if (p_c0) { c0_in = p_c0[a]; p_c0[a] = p_c0[M - 1]; } if (p_h0) { h0_in = p_h0[a]; p_h0[a] = p_h0[M - 1]; }   /* 30 Oct.: FLUXTEL */
        p_afst[a] = p_afst[M - 1]; p_hmax[a] = p_hmax[M - 1]; p_typ[a] = p_typ[M - 1]; p_D[a] = p_D[M - 1]; p_T[a] = p_T[M - 1];
    }
    px[a] = px[M - 1]; pz[a] = pz[M - 1]; pvx[a] = pvx[M - 1]; pvz[a] = pvz[M - 1]; pk[a] = pk[M - 1];
    if (NY > 1) { py[a] = py[M - 1]; pvy[a] = pvy[M - 1]; if (p_r0) { p_r0[a] = p_r0[M - 1]; p_x0[a] = p_x0[M - 1]; } }
    M--;
    n_ins++;
    int kt = h[j] > 0 ? col[(int64_t)j * HM + h[j] - 1] : k;     /* top bed grain before the impact */
    int hard = korst_hard && gehecht(j);
    if (hard && U() < korst_pb) { korst[j] = 0; hard = 0; }   /* crust crumbles with probability korst_pb per impact */
    double sg = sqrt(P.g * P.Dsp);
    double areb = 0.0, fN = 1.0, tilt = 0.0, al_s = 0, th_s = 0, ee2 = 0;
    int reb;
    if (hoekmodel) {
        double sl = sr_ * helling_bed(j);   /* 6 Nov.: HOEK_LSQ_KOL; 9 Nov.: in the direction of motion */
        tilt = hoek_hel ? atan(sl) : 0.0;
        double alpha = atan2(-vz, vx > 1e-9 ? vx : 1e-9);          /* descent angle relative to the horizontal */
        bp_tel[2]++; if (alpha + tilt < 0.02) { bp_tel[3]++; if (inslag_steil) tilt = 0.0; }   /* 6 Nov.: INSLAG_STEIL */
        double th = alpha + tilt; if (th < 0.02) th = 0.02; if (th > 1.55) th = 1.55;
        double st = sin(th);
        double e = 0.87 - 0.72 * st; if (e < 0.05) e = 0.05;
        al_s = alpha; th_s = th; ee2 = 1 - e * e;
        if (kam_hist && kam_vlag[j] && v > 1.0 && vz < 0) { kam_bij(4, th * 57.29577951308232); kam_bij(5, alpha * 57.29577951308232); }   /* only fast, descending (saltating) grains */
        double ez = 0.30 / st - 0.15; if (ez < 0) ez = 0; if (ez > ez_max) ez = ez_max;
        som_f += 1 - e * e; n_f += 1;
        fN = (1 - e * e) / C_vlak;
        double vn = ez * v * st;                                   /* speed normal to the bed after rebound */
        reb = (U() < P.B * (1 - exp(-P.gam * v))) && (vn >= sqrt(2 * P.g * dkl(k)));
        if (!reb && land_a > 0 && k == 0 && top_grof(j) && U() < land_a) { reb = 1; n_land[0]++; }   /* 20 Oct.: (a) */
        if (hard) reb = vn >= sqrt(2 * P.g * dkl(k));          /* hard crust: rebound without splash */
        if (reb) {
            double ee = e * (1 + 0.25 * nrm()); if (ee < 0.05) ee = 0.05; if (ee > 1) ee = 1;
            double sm = ez * st / e; if (sm > 0.999) sm = 0.999;
            double thm = asin(sm);
            double thr = -thm * log(1 - U()); if (thr > 1.5) thr = 1.5;
            double g_ = vertrek_h(thr, tilt, 0);        /* to the x-axis (5 Nov.: TILT_PLUS) */
            areb = ee;
            vx_terug = ee * v * cos(g_); vt_terug = ee * v * cos(thr);
            l_c0 = c0_in; l_h0 = h0_in; l_typ = 1; l_D = D_in + (sp3_aan ? (j + 0.5) * P.dx - x_land : 0.0); l_T = T_in; l_r0 = r0_in; l_x0 = x0_in; { int M0_ = M; lanceer(j, ee * v, sr_ > 0 ? g_ : 3.141592653589793 - g_, k); if (NY > 1 && M > M0_) draai_horizontaal(M - 1, psi_in + (lat_reb > 0 ? lat_reb * nrm() : 0)); }
            n_reb++;
        }
    } else {
        reb = U() < P.B * (1 - exp(-P.gam * v));
        if (reb) {
            double e = P.ereb_m + P.ereb_s * nrm(); if (e < 0.01) e = 0.01; if (e > 1) e = 1;
            double th = -P.threb * log(1 - U()); if (th > 1.5) th = 1.5;
            areb = sqrt(e);
            vx_terug = areb * v * cos(th);
            l_c0 = c0_in; l_h0 = h0_in; l_typ = 1; l_D = D_in + (sp3_aan ? (j + 0.5) * P.dx - x_land : 0.0); l_T = T_in; l_r0 = r0_in; l_x0 = x0_in; { int M0_ = M; lanceer(j, areb * v, sr_ > 0 ? th : 3.141592653589793 - th, k); if (NY > 1 && M > M0_) draai_horizontaal(M - 1, psi_in + (lat_reb > 0 ? lat_reb * nrm() : 0)); }
            n_reb++;
        }
    }
    bt_reb = reb;   /* 9 Nov.: KAATS_TERUG (test) */
    if (!reb) {
        if (hc >= 0) { hop_stat[(HOP_NC + hc) * N + j] += 1; hop_stat[(HOP_NC + ht) * N + j] += 1; }
        if (h_ins) h_ins[j]++;
        if (kol_afz) kol_afz[j]++;
        int jd = j; uint8_t kd = k;
        if (sp3_aan && p_afst && k < 3) {             /* 17 Oct.: jump distribution by departure position */
            double x0 = x_land - D_in; int c0 = (int)floor(x0 / P.dx);   /* exact: D also counts the shift on rebound */ c0 = ((c0 % sp3_fold) + sp3_fold) % sp3_fold;
            int v = c0 / sp3_vak; if (v >= SP3_NV) v = SP3_NV - 1;
            int b = (int)floor((D_in + SP3_DNEG) / SP3_DB); if (b < 0) b = 0; if (b >= SP3_ND) b = SP3_ND - 1;   /* bin 0 starts at -SP3_DNEG (backward jumps) */
            sp3_n[k][v][b]++; sp3_t[k][v][b] += T_in;
            if (b == SP3_ND - 1) { int cl = ((j % sp3_fold) + sp3_fold) % sp3_fold; if (cl < SP3_NV) sp3_ov[k][cl]++; }
        }
        if (p_afst && k < 3) {                       /* 16 Oct.: coupling test (count only for koppeling 0) */
            int c = k; double Dp = D_in;
            if (koppeling == 1 && kop_n[c] >= KOP_MIN) {
                kop_lcg = kop_lcg * 6364136223846793005ULL + 1442695040888963407ULL;
                long nb = kop_n[c] < KOP_NB ? kop_n[c] : KOP_NB;
                Dp = kop_buf[c][(kop_lcg >> 33) % (unsigned long long)nb];
                double xn = (j + 0.5) * P.dx - D_in + Dp;
                jd = md((int)floor(xn / P.dx));
            }
            if (koppeling == 2) { if (kop_slot >= 0) kd = (uint8_t)kop_slot; else jd = -1; kop_slot = k; }   /* first landing: grain to the slot */
            kop_buf[c][kop_n[c] % KOP_NB] = D_in; kop_n[c]++;
            kop_tel[c][0] += 1; kop_tel[c][1] += D_in; kop_tel[c][2] += Dp; if (jd >= 0) kop_tel[kd][3] += 1;
            kop_hist[0][c][kop_bin(D_in)]++; kop_hist[1][c][kop_bin(Dp)]++;
        }
        if (ft_kol && p_c0 && c0_in >= 0) {   /* 30 Oct.: FLUXTEL; displacement in whole columns, periodically as close as possible to D_in */
            double dc = j - c0_in; dc += N * round((D_in / P.dx - dc) / N); double d_ = dc * P.dx;
            ft_bij(d_ < 5e-3 ? 3 : d_ < 20e-3 ? 4 : d_ < 107.2e-3 ? 5 : 6, c0_in, j, d_);
            if (p_h0) ft_bij(h0_in < 1e-3 ? 7 : h0_in < 3e-3 ? 8 : h0_in < 10e-3 ? 9 : 10, c0_in, j, d_); }   /* 31 Oct.: MOD_LH */
        if (NY > 1 && r0_in != rij && jd >= 0) lat_tel_bij(0, r0_in, x0_in, rij, jd);
        if (!P.bevroren && jd >= 0) {
            if ((land_c > 0 || land_d > 0) && kd == 0 && top_grof(jd)) {           /* 20 Oct.: fine on coarse, (b)-(d) */
                double r_ = U();
                if (r_ < land_c) {                                                  /* (c) rolls to the lower neighbour */
                    int n1 = md(jd + 1), n0 = md(jd - 1);
                    if (h[n1] <= h[jd]) jd = n1; else if (h[n0] <= h[jd]) jd = n0;
                    n_land[2]++;
                } else if (r_ < land_c + land_d && h[jd] >= 1) {                    /* (d) into the space below the coarse grain */
                    ctx = 3; uint8_t kg_ = pak(jd); leg(jd, kd); leg(jd, kg_); ctx = 0; lawine(jd); n_land[3]++;
                    jd = -2;
                } else n_land[1]++;                                                 /* (b) stays on top */
            }
            if (jd >= 0) { ctx = 3; leg(jd, kd); ctx = 0; lawine(jd); }
        }
    }
    if (hard) return;                              /* hard crust: no creep, no ejections */
    /* 11 Oct.: MOMENTUM BUDGET ALSO FOR CREEP (kruip_impuls = 1): each creep push costs the momentum of a mean
       ejection, m_k * vej_m (m_k = mass of the pushed top grain), and is taken before the ejections from the budget
       (1 - e) v m of the impacting grain. If it no longer fits, the pushes stop. No extra random numbers. */
    double vej_m = P.alpha_ej / P.a * sg * (1 - exp(-v / (40 * sg)));
    double budget = (1 - areb) * v * mk[k];        /* momentum budget */
    double bx = mk[k] * (vx - vx_terug);            /* horizontal momentum loss of the impacting grain (C21) */
    double bx_ej = bx;                               /* the same, budget for ejections with ej_hor (C23) */
    if (impuls_raak && hoekmodel) bx_ej = mk[k] * (v * cos(th_s) - vt_terug);   /* 18 Oct.: along the local surface */
    som_bx += bx; n_bx++;
    if (p_cr > 0 && (art_kr || !art_)) {                                /* number of pushed grains ~ Poisson(p_cr); impulse push: Poisson(p_cr Bx / Bx_ref) */
        double lam_c = (kruip_impuls == 3 && kr_bxref > 0) ? p_cr * (bx > 0 ? bx : 0) / kr_bxref : p_cr;
        int nc = 0; double L = exp(-lam_c), pp = 1; do { nc++; pp *= U(); } while (pp > L); nc--;
        if (kruip_tegen && kt_r == 1) nc = 0;   /* 7 Nov.: KRUIP_TEGEN */
        if (kruip_terug && kt_r == 1) kr_dir = -1;   /* 8 Nov.: KRUIP_TERUG */
        som_nc += nc; bp_tel[4] += nc;
        for (int q = 0; q < nc; q++) {
            int jq = md(j + kr_dir * (q % 2));
            if (kruip_impuls == 2) {
                double J = mk[0] * vej_m;
                if (bx <= 0) break;
                if (J > bx) { J = bx; n_kr_rest++; }
                bx -= J; kr_J = J; kr_vej = vej_m;
                kruip(jq); kr_fac = 1.0;
                continue;
            }
            if (kruip_impuls == 1) {
                double kost = vej_m * mk[h[jq] > 0 ? col[(int64_t)jq * HM + h[jq] - 1] : 0];
                if (kost > budget) { n_kr_geweigerd += nc - q; break; }
                budget -= kost; som_kr_impuls += kost;
            }
            if (kruip_impuls == 4) {   /* 6 Nov.: KRUIP_IMPULS=4: horizontal budget shared with the ejections */
                double kost = vej_m * mk[h[jq] > 0 ? col[(int64_t)jq * HM + h[jq] - 1] : 0];
                bp_tel[6]++; if (kost > bx_ej) { bp_tel[5] += nc - q; break; }
                bx_ej -= kost;
            }
            if (lanceer_duw) { int kt2_ = h[jq] > 0 ? col[(int64_t)jq * HM + h[jq] - 1] : 0; kr_u = (bx > 0 ? bx : 0) / (nc * mk[kt2_]); }
            kt_stap = -1; kruip(jq); kr_u = 0;
            if (kt_kol && kt_stap >= 0) { int n_ = kt_stap < 127 ? kt_stap : 127; kt_hist[(2 * kt_r + (kt_kl >= 1)) * 128 + n_] += 1;   /* 7 Nov.: KRTEL */
                kt_kol[(2 + kt_r) * N + jq] += 1; kt_kol[(4 + kt_r) * N + jq] += kt_stap * P.dx; }
        }
        kr_dir = 1;   /* 8 Nov.: KRUIP_TERUG */
    }
    /* ejections: number ~ a v / sqrt(g D) * (m_impact / m_bedgrain) (COMSALT eq. 21) */
    double mratio = nk > 1 ? mk[k] / mk[kt] : 1.0;
    double lam = P.a * v / sg * fN * mratio * (cohesie > 1 ? pow(cohesie, -0.5) : 1.0);
    if (splashmodel == 2) { double x_ = v / sg - xi_c; lam = x_ > 0 ? s_ej * fN * C_vlak * x_ * mratio * (cohesie > 1 ? pow(cohesie, -0.5) : 1.0) : 0; }
    if (ins_stat) { ins_stat[j] += 1; ins_stat[N + j] += al_s; ins_stat[2 * N + j] += th_s; ins_stat[3 * N + j] += v;
                    ins_stat[4 * N + j] += ee2; ins_stat[5 * N + j] += lam; ins_stat[6 * N + j] += reb; }
    double fst = 1.0; int steun_aan = steun_a > 0 && steun_arr;
    if (steun_aan) { fst = 1 + steun_a * steun_arr[md(j)] * (hoekmodel ? cos(th_s) : 1.0); lam *= fst; }
    int n = 0; if (art_ej || !art_) { double L = exp(-lam), p = 1; do { n++; p *= U(); } while (p > L); n--; }
    if (kt_kol) { kt_s[kt_r] += reb; kt_s[2 + kt_r] += n; kt_s[4 + kt_r] += lam; }   /* 7 Nov.: KRTEL */
    for (int q = 0; q < n; q++) {
        int xe_pre = -1;
        if (steun_aan) {                            /* choose the spot first; keep a coarse candidate with probability 1/fst, otherwise nothing is used */
            double eps_ = -P.eps_d * P.d * log(1 - U()); xe_pre = md(j + sr_ * (int)floor(eps_ / P.dx + U()));
            if (h[xe_pre] > 1 && col[(int64_t)xe_pre * HM + h[xe_pre] - 1] >= 1 && U() * fst >= 1.0) { n_steun_grof++; continue; }
        }
        double ve = (ej_logn > 0 ? vej_m * exp(ej_logn * nrm() - 0.5 * ej_logn * ej_logn) : -vej_m * log(1 - U())) * ej_vfac;   /* 16 Oct.: ej_vfac; 17 Oct.: ej_logn */
        int xe0 = j; (void)xe0;
        double me = mk[kt];
        if (!ej_hor) {
            if (ve * me > budget) { n_ej_geweigerd += n - q; break; }
            budget -= ve * me;
        }
        double th = -P.thej * log(1 - U()); if (th > 1.5) th = 1.5;
        if (splashmodel == 2) { th = ej_hoek + ej_s * nrm(); if (th > 3.12) th = 3.12; }
        if ((ej_min && ve * sin(th) < sqrt(2 * P.g * ej_min_n * dkl(kt))) || (ej_lmin > 0 && 2 * ve * ve * sin(th) * fabs(cos(th)) / P.g < ej_lmin)) { n_ej_onder++; if (ej_onder_kruip) kruip(md(j + q % 2)); continue; }   /* 17 Oct.: does not pass over the neighbours; 18 Oct.: n diameters, possibly as creep */
        double th_bed = th;
        if (hoekmodel) th = vertrek_h(th, tilt, 1);   /* 5 Nov.: TILT_PLUS */
        if (ej_hor) {                              /* 12 Oct.: only the horizontal part from the impact (C23) */
            double ct_ = impuls_raak ? cos(th_bed) : cos(th);   /* 18 Oct.: with impuls_raak the component along the bed */
            double kost = ve * me * (ct_ > 0 ? ct_ : 0);
            if (kost > bx_ej) { n_ej_geweigerd += n - q; bp_tel[7] += n - q; break; }
            bx_ej -= kost;
        }
        int xe;
        if (xe_pre >= 0) xe = xe_pre;
        else { double eps = -P.eps_d * P.d * log(1 - U()); xe = md(j + sr_ * (int)floor(eps / P.dx + U())); }
        if (h[xe] <= 1) continue;
        if (gehecht(xe) && v < korst_fac * xi_c * sg) continue;
        if (stug && v < stugf(xe) * xi_c * sg) continue;
        if (h_ej) h_ej[xe]++;
        if (hc >= 0) { hop_stat[(2 * HOP_NC + hc) * N + xe] += 1; hop_stat[(2 * HOP_NC + ht) * N + xe] += 1; }
        uint8_t ke;
        if (!P.bevroren && sort_a > 0 && steun_arr && h[xe] >= 3 && col[(int64_t)xe * HM + h[xe] - 1] >= 1 && col[(int64_t)xe * HM + h[xe] - 2] == 0
            && U() < sort_a * steun_arr[xe] * (hoekmodel ? cos(th_s) : 1.0)) {
            ctx = 4; uint8_t kg_ = pak(xe); ke = pak(xe); leg(xe, kg_); ctx = 0; n_sort++;      /* fine grain away from below the coarse one */
        } else ke = P.bevroren ? (bevr_top && h[xe] > 0 ? col[(int64_t)xe * HM + h[xe] - 1] : 0) : (ctx = 4, pak(xe));
        ctx = 0;
        if ((wand >= 2 || bijval_eerst) && !P.bevroren) lawine(xe);   /* 18 Oct.: first let the hole fill, then launch from the actual surface (26 Oct.: also bijval_eerst) */
        { int M0_ = M; l_typ = 0; lanceer(xe, ve, sr_ > 0 ? th : 3.141592653589793 - th, ke);
          if (NY > 1 && M > M0_) {                   /* 16 Oct.: lateral direction of the ejection */
              int b_ = M - 1;
              if (lat_ej == 1) { double vh = fabs(ve * cos(th)), ps = 6.283185307179586 * U(); pvx[b_] = vh * cos(ps); pvy[b_] = vh * sin(ps); }
              else if (lat_ej == 2) { double vf = ve * cos(th), vl = fabs(ve * cos(th)) * sin(6.283185307179586 * U());
                                      pvx[b_] = vf * cos(psi_in) - vl * sin(psi_in); pvy[b_] = vf * sin(psi_in) + vl * cos(psi_in); }
              else draai_horizontaal(b_, psi_in);
              lat_tel[3] += 1; lat_tel[4] += fabs(pvy[b_]); lat_tel[5] += fabs(pvx[b_]);
          } }
        if (!P.bevroren && wand < 2 && !bijval_eerst) lawine(xe);
        n_ej++;
    }
}
#ifdef BOTSING_TOETS   /* 9 Nov.: only in the test build: a second call of botsing() keeps the compiler from inlining, and that changes the rounding */
int botsing_toets(double x, double vx, double vz, int k) {   /* 9 Nov.: KAATS_TERUG, analytical test (without counters, frozen bed) */
    M = 0; px[0] = x; pz[0] = 0; pvx[0] = vx; pvz[0] = vz; pk[0] = (uint8_t)k; M = 1; bt_reb = 0;
    botsing(0, md((int)floor(x / P.dx))); return M; }
#endif
double f_vlak_uit(void) { double r = n_f > 0 ? som_f / n_f : 0; som_f = 0; n_f = 0; return r; }

static inline double Cd(double Re) { double a = sqrt(0.5) + sqrt(24.0 / (Re > 1e-9 ? Re : 1e-9)); return a * a; }

/* main loop: T seconds */
int draai(int n_, int hm, uint8_t *col_, int32_t *h_, Par *par, double T,
          double *px_, double *pz_, double *pvx_, double *pvz_, uint8_t *pk_, int *M_, int Mcap_,
          double *Uk_out, long *tel, int32_t *hins, int32_t *hej, double *mod_) {
    windmod = mod_; wm0 = mod_;
    N = n_ / NY; HM = hm; col = col_; h = h_; h0 = h_; col0 = col_; P = *par; fout = 0; rij = 0;
    if (NY > 1 && (h_ins || h_ej || kol_ins || kol_kr || kol_kr_aan || kol_afz || gtel || korst || stug || ins_stat || hop_stat || wr_stat
                   || nblok > 0 || alfa_ref > 0 || open_rand || inj_n > 0 || !py || !pvy)) return -2;   /* not supported in quasi-3D */
    px = px_; pz = pz_; pvx = pvx_; pvz = pvz_; pk = pk_; M = *M_; Mcap = Mcap_;
    h_ins = hins; h_ej = hej;
    if (hk_kol && !hk_p) hk_p = calloc(Mcap, 1);   /* 7 Nov.: HKTEL */
    if (rbt_kol && !p_rbj) { p_rbj = malloc(Mcap * sizeof(int)); for (int q = 0; q < Mcap; q++) p_rbj[q] = -1; }   /* 6 Nov.: RBTEL */
    if (rb_kol && !p_rb) p_rb = calloc(Mcap, 1);   /* 4 Nov.: ART_KR, ART_EJ */
    if (sprong_aan && ft_kol && !p_h0) { p_h0 = malloc(Mcap * sizeof(double)); for (int q = 0; q < Mcap; q++) p_h0[q] = 0; }   /* 31 Oct.: MOD_LH */
    if (sprong_aan && ft_kol && !p_c0) { p_c0 = malloc(Mcap * sizeof(int)); for (int q = 0; q < Mcap; q++) p_c0[q] = -1; }   /* 30 Oct.: FLUXTEL */
    if (sprong_aan && !p_afst) { p_T = calloc(Mcap, sizeof(double)); p_D = calloc(Mcap, sizeof(double)); p_afst = calloc(Mcap, sizeof(double)); p_typ = calloc(Mcap, 1); p_hmax = calloc(Mcap, sizeof(double)); }
    if (NY > 1 && p_rcap != Mcap) { free(p_r0); free(p_x0); p_r0 = calloc(Mcap, sizeof(int)); p_x0 = calloc(Mcap, sizeof(int)); p_rcap = Mcap; }
    if (inj_n > 0) { injecteer(inj_n, inj_f1, inj_f2); inj_n = 0; }
    if (begr_aan && kol_n != N) { free(kol_t); free(kol_c); kol_t = calloc(N, sizeof(double)); kol_c = calloc(N, 1); kol_n = N; for (int q = 0; q < N; q++) kol_t[q] = -1e9; }   /* 26 Oct. */
    if (nblok > 0 && (NBN != N || BW != nblok)) {
        free(Ukb); free(taupb); free(tpfb); BW = nblok; NB = N / BW; if (NB < 1) NB = 1; NBN = N;
        Ukb = calloc((size_t)NB * NZ, sizeof(double)); taupb = calloc((size_t)NB * NZ, sizeof(double)); tpfb = calloc((size_t)NB * NZ, sizeof(double));
        for (int b = 0; b < NB; b++) for (int k = 0; k < NZ; k++) Ukb[(int64_t)b * NZ + k] = Uk_out[0] < 0 ? P.ustar / P.kappa * log(zk[k] / P.z0) : Uk_out[k];
    }
    if (nblok <= 0 && Ukb) { free(Ukb); free(taupb); free(tpfb); Ukb = taupb = tpfb = 0; NB = BW = NBN = 0; }
    if (kruip_inslag && I_N != N * NY) {
        free(Ic0); free(Ie0); free(Ir0); Ic0 = calloc((size_t)N * NY, sizeof(double)); Ie0 = calloc((size_t)N * NY, sizeof(double));
        Ir0 = malloc((size_t)N * NY * sizeof(double)); for (int i = 0; i < N * NY; i++) Ir0[i] = 1.0; I_N = N * NY; I_init = 0;
    }
    if (Ic0) { I_cnt = Ic0; I_ema = Ie0; I_r = Ir0; }
    Lx = N * P.dx;
    m = P.rho_p * 3.141592653589793 / 6 * P.d * P.d * P.d;
    for (int q = 0; q < 4; q++) { double dd = dkl(q); mk[q] = P.rho_p * 3.141592653589793 / 6 * dd * dd * dd; }
    W = (3.141592653589793 / 6 * P.d * P.d * P.d) / (P.dx * P.dz * 0.6);
    static int init = 0;
    if (!init || Uk_out[0] < 0) { profiel_init(); init = 1; }
    else for (int k = 0; k < NZ; k++) Uk[k] = Uk_out[k];
    n_ins = n_reb = n_ej = n_aero = 0; i0 = r0 = e0 = a0 = 0; som_vin = 0;
    double t = 0;
    static double tupd = 0;   /* 6 Oct.: kept between calls; otherwise the remaining momentum flux
                                of the previous call counted over too short a time (flux too low with short calls) */
    while (t < T) {
        for (int a = 0; a < M; a++) {
            if (NY > 1) zet_rij(rij_van(py[a]));
            int i = md((int)floor(px[a] / P.dx));
            int kl = pk[a]; double da = dkl(kl), ma = mk[kl];
            double c = 0.75 * P.rho_a / P.rho_p / da;
            double zr = pz[a] - zref(i) - 0.5 * da;         /* height of the grain centre above the contact height (or the reference level) */
            double u = windk(zr, i) * ((mod_zc != 0 && mh_jh) ? mh_pl[i] * (((mod_zc > 0) == (zr < fabs(mod_zc))) ? mh_jh[i] : 1)   /* 1 Nov.: MOD_ZC */
                                      : (mod_lh > 0 && mh_jh) ? mh_pl[i] * (1 + (mh_jh[i] - 1) * exp(-(zr > 0 ? zr : 0) / mod_lh)) : modf_(i));   /* 31 Oct.: MOD_LH */
            double wv = (stroom_s && NY == 1) ? u * stroom_hel(zr, i, N) : 0.0;   /* 13 Nov.: STROOM, vertical wind */
            double rx = pvx[a] - u, rz = (stroom_s && NY == 1) ? pvz[a] - wv : pvz[a], vr = sqrt(rx * rx + rz * rz);
            double k = c * Cd(vr * da / P.nu) * vr;
            /* semi-implicit for the drag */
            double z_oud = pz[a], vx_oud = pvx[a], x_oud = px[a];
            pvx[a] = (terrein != 0 && (ter_delen & 8)) ? (pvx[a] + P.dt * k * u - P.dt * P.g * sin(terrein)) / (1 + P.dt * k) : (pvx[a] + P.dt * k * u) / (1 + P.dt * k);
            mom_drag += ma * (pvx[a] - vx_oud);
            pvz[a] = (stroom_s && NY == 1) ? (pvz[a] - P.dt * ((terrein != 0 && (ter_delen & 8)) ? P.g * cos(terrein) : P.g) + P.dt * k * wv) / (1 + P.dt * k)   /* 13 Nov.: STROOM */
                                           : (pvz[a] - P.dt * ((terrein != 0 && (ter_delen & 8)) ? P.g * cos(terrein) : P.g)) / (1 + P.dt * k);
            if (NY > 1) { pvy[a] = pvy[a] / (1 + P.dt * k); py[a] += P.dt * pvy[a]; double Ly_ = NY * dyr; if (py[a] < 0) py[a] += Ly_; if (py[a] >= Ly_) py[a] -= Ly_; }
            px[a] += P.dt * pvx[a]; pz[a] += P.dt * pvz[a]; if (p_afst) { p_afst[a] += P.dt * pvx[a]; p_D[a] += P.dt * pvx[a]; p_T[a] += P.dt; double zz_ = pz[a] - hb(md((int)floor(px[a] / P.dx))) - 0.5 * da; if (zz_ > p_hmax[a]) p_hmax[a] = zz_; if (p_h0 && zz_ > p_h0[a]) p_h0[a] = zz_; }
            if (px[a] < 0) px[a] += Lx;
            if (px[a] >= Lx) {
                if (open_rand) {           /* 6 Oct.: open boundary downwind: grain leaves the domain */
                    px[a] = px[M - 1]; pz[a] = pz[M - 1]; pvx[a] = pvx[M - 1]; pvz[a] = pvz[M - 1]; pk[a] = pk[M - 1];
                    if (p_afst) { p_afst[a] = p_afst[M - 1]; p_hmax[a] = p_hmax[M - 1]; p_typ[a] = p_typ[M - 1]; p_D[a] = p_D[M - 1]; p_T[a] = p_T[M - 1]; }
                    if (p_c0) p_c0[a] = p_c0[M - 1];
                    if (p_h0) p_h0[a] = p_h0[M - 1];
                    if (p_rb) p_rb[a] = p_rb[M - 1];
                    if (p_rbj) p_rbj[a] = p_rbj[M - 1];
                    if (hk_p) hk_p[a] = hk_p[M - 1];   /* 7 Nov.: HKTEL */
                    M--; a--; n_uit++; continue;
                }
                px[a] -= Lx;
            }
            if (wand == 3) {                             /* 18 Oct.: edge collision */
                int jn = md((int)floor(px[a] / P.dx));
                double top = hb(jn), rr = 0.5 * da;
                if (jn != i && h[jn] > h[i] && pvz[a] >= 0 && pz[a] <= top + rr) {   /* only rising grains; a descending grain makes an ordinary impact */
                    double grens = vx_oud >= 0 ? (i + 1) * P.dx - 1e-9 : i * P.dx + 1e-9;
                    double schuif = grens - (x_oud + P.dt * pvx[a]);
                    if (p_afst) { p_afst[a] += schuif; p_D[a] += schuif; }
                    px[a] = grens; if (px[a] >= Lx) px[a] -= Lx; if (px[a] < 0) px[a] += Lx;
                    double sn = (pz[a] - top) / rr; if (sn > 1) sn = 1; if (sn < -1) sn = -1;
                    double nx = -(vx_oud >= 0 ? 1.0 : -1.0) * sqrt(1 - sn * sn), nz = sn;
                    double vn = pvx[a] * nx + pvz[a] * nz;
                    if (vn < 0) { double hkx_ = pvx[a]; pvx[a] -= (1 + rand_e) * vn * nx; pvz[a] -= (1 + rand_e) * vn * nz;
                        if (hk_p) hk_p[a] |= (uint8_t)(16 | ((hkx_ > 0 && pvx[a] <= 0) ? 32 : 0)); }   /* 7 Nov.: HKTEL */
                    n_wand++;
                    if (rbt_kol) { int c_ = sn >= 0 ? 0 : (sn > -1 ? 1 : 2); rbt_kol[c_ * N + jn] += 1; if (p_rbj) p_rbj[a] = sn < 0 ? jn : -1; }   /* 6 Nov.: RBTEL */
                    if (rb_kol) { rb_kol[jn] += 1; if (p_rb) p_rb[a] = 1; }   /* 4 Nov.: ART_KR, ART_EJ */
                }
            } else if (wand && pvz[a] >= 0) {            /* 18 Oct.: wall rule */
                int jn = md((int)floor(px[a] / P.dx));
                if (jn != i && h[jn] > h[i] && pz[a] <= hb(jn) + 0.5 * da) {
                    double grens = vx_oud >= 0 ? (i + 1) * P.dx - 1e-9 : i * P.dx + 1e-9;
                    double schuif = grens - (x_oud + P.dt * pvx[a]);
                    if (p_afst) { p_afst[a] += schuif; p_D[a] += schuif; }
                    px[a] = grens; if (px[a] >= Lx) px[a] -= Lx; if (px[a] < 0) px[a] += Lx;
                    pvx[a] = 0; n_wand++;
                }
            }
            /* momentum flux through the levels zk (above the bed of the old column) */
            double z1 = z_oud - zref(i) - 0.5 * da, z2 = pz[a] - zref(i) - 0.5 * da;
            if (z1 > P.z0 || z2 > P.z0) {
                double lo = z1 < z2 ? z1 : z2, hi = z1 < z2 ? z2 : z1;
                if (hi > P.z0) {
                    int k0 = lo <= P.z0 ? 0 : (int)ceil(log(lo / P.z0) / log(zmax / P.z0) * (NZ - 1));
                    int k1 = hi >= zmax ? NZ - 1 : (int)floor(log(hi / P.z0) / log(zmax / P.z0) * (NZ - 1));
                    double s = (z2 < z1) ? 1.0 : -1.0;          /* descending +, rising - */
                    for (int q = k0; q <= k1 && q < NZ; q++) { taup[q] += s * (impuls_cel ? m : ma) * 0.5 * (vx_oud + pvx[a]); if (taupb) taupb[(int64_t)blok(i) * NZ + q] += s * (impuls_cel ? m : ma) * 0.5 * (vx_oud + pvx[a]); }   /* same expression: same rounding (FMA) as before */
                }
            }
            /* collision: grain centre at or below the bed of the (new) column */
            if (NY > 1) zet_rij(rij_van(py[a]));
            int j = md((int)floor(px[a] / P.dx));
            if (pz[a] <= hb(j) + 0.5 * da && pvz[a] < ((wand >= 2 || alleen_dalend) ? 0.0 : 0.5)) {   /* 18 Oct.: with wand >= 2 only a descending grain lands */
                if (kol_t) { double d_ = hb(j) + 0.5 * da - pz[a]; begr_tel[6] += 1;   /* 26 Oct.: buried counter */
                    if (d_ > 1e-4) { double vxa = fabs(pvx[a]) > 1e-6 ? fabs(pvx[a]) : 1e-6, xl = j * P.dx, dist = pvx[a] >= 0 ? px[a] - xl : xl + P.dx - px[a];
                        double tin = begr_t - dist / vxa; if (p_afst && p_typ[a] != 1 && begr_t - p_T[a] > tin) tin = begr_t - p_T[a];
                        int c_ = kol_c[j] == 1 ? 2 : kol_c[j] == 2 ? 3 : kol_c[j] == 3 ? 4 : 1; begr_tel[0] += 1; begr_tel[kol_t[j] >= tin ? c_ : 5] += 1; } }
                { int zt_ = (p_afst && p_typ[a] == 1) ? 1 : 0, zz_ = (j != i && h[j] > h[i]) ? 1 : 0; zij_tel[zt_][zz_] += 1; zij_tel[zt_][2 + zz_] += hb(j) + 0.5 * da - pz[a]; }   /* 26 Oct.: counter */
                if (rbt_kol) { rbt_kol[4 * N + j] += 1;   /* 6 Nov.: RBTEL */
                    if (j != i && h[j] > h[i] && pz[a] < hb(j)) { rbt_kol[3 * N + j] += 1; rbt_s[0] += hb(j) - pz[a]; rbt_s[1] += 1; }
                    if (p_rbj && p_rbj[a] == j) rbt_kol[5 * N + j] += 1; }
                /* impact: the grain gives its momentum to the bed; levels below its current
                   height (relative to the old column) that it has not yet passed count as descending */
                double zr_c = pz[a] - zref(i) - 0.5 * da;
                for (int q = 0; q < NZ && zk[q] < zr_c; q++) { taup[q] += (impuls_cel ? m : ma) * pvx[a]; if (taupb) taupb[(int64_t)blok(i) * NZ + q] += (impuls_cel ? m : ma) * pvx[a]; }
                botsing(a, j); a--; if (fout) { *M_ = M; return -1; } }
            else if (pz[a] > 2.0) { /* out of the domain upward: let it come down again */ pz[a] = 2.0; pvz[a] = -fabs(pvz[a]); }
        }
        t += P.dt; tupd += P.dt; t_tot += P.dt; begr_t += P.dt;
        if (tupd >= P.t_upd) {
            profiel_bij(tupd);
            inslag_bij(tupd);
            if (diag_aan) {                     /* 5 Oct.: diagnostics of runaway saltation */
                double vs = 0, vi = 0, zm_ = 0; int nh = 0;
                for (int a = 0; a < M; a++) { vs += pvx[a]; zm_ += pz[a] - hb(md((int)floor(px[a] / P.dx))); if (pz[a] - hb(md((int)floor(px[a] / P.dx))) > 0.05) nh++; }
                double mm = 0; if (windmod) for (int i = 0; i < N; i++) if (windmod[i] > mm) mm = windmod[i];
                vi = som_vin / (n_ins - i0 > 0 ? n_ins - i0 : 1); som_vin = 0;
                fprintf(stderr, "DIAG t %.3f M %d ins %ld reb %ld ej %ld aero %ld v_in %.2f <vx> %.2f <z> %.4f n>5cm %d U(1mm) %.2f U(1cm) %.2f U(10cm) %.2f tp0 %.3f max_mod %.2f\n",
                    t, M, n_ins - i0, n_reb - r0, n_ej - e0, n_aero - a0, vi, M ? vs / M : 0, M ? zm_ / M : 0, nh,
                    wind(1e-3), wind(1e-2), wind(0.1), tp_f[0], mm);
                i0 = n_ins; r0 = n_reb; e0 = n_ej; a0 = n_aero;
            }
            /* aerodynamic entrainment when the fluid threshold at the bed is exceeded */
            double us_b = (Uk[1] - Uk[0]) / (log(zk[1] / zk[0]) / P.kappa);   /* local u* at the bed */
            double ex = us_b * us_b / (P.ust_fl * P.ust_fl) - 1;
            if (aero_lokaal) {
                /* LOCAL ENTRAINMENT (4 Oct.): per column the local shear stress (with flow modulation)
                   against the fluid threshold of the top grain; rate zeta (tau - tau_ft) grains per
                   m^2 per s (Anderson & Haff 1988, 1991: zeta ~ 1e5 per N per s). */
                for (int ii = 0; ii < N * NY; ii++) {
                    int i = ii % N; if (NY > 1 && i == 0) zet_rij(ii / N);
                    if (h[i] <= 1 || gehecht(i)) continue;
                    double f = windmod ? windmod[i] : 1.0;
                    double tau = P.rho_a * us_b * us_b * f * f;
                    int kk2 = col[(int64_t)i * HM + h[i] - 1];
                    double uf = nk > 1 ? ustk[kk2] : P.ust_fl;
                    double tft = P.rho_a * uf * uf;
                    if (tau <= tft) continue;
                    double lamA = zeta_a * (tau - tft) * P.dx * W * tupd;
                    if (U() < lamA) {                     /* lamA << 1: at most one grain per step */
                        uint8_t ke = P.bevroren ? 0 : (ctx = 4, pak(i)); ctx = 0;
                        if ((wand >= 2 || bijval_eerst) && !P.bevroren) lawine(i); l_typ = 2; lanceer(i, 0.3 + 0.2 * U(), 1.0, ke); if (!P.bevroren && wand < 2 && !bijval_eerst) lawine(i); n_aero++;
                    }
                }
                ex = 0;
            }
            if (ex > 0) {
                double lamA = P.aero_rate * ex * Lx * tupd * NY;
                int nA = 0; { double L = exp(-lamA), p = 1; if (lamA > 30) nA = (int)(lamA + sqrt(lamA) * nrm()); else { do { nA++; p *= U(); } while (p > L); nA--; } }
                for (int q = 0; q < nA; q++) {
                    if (NY > 1) zet_rij((int)(U() * NY));
                    int i = (int)(U() * N); if (h[i] <= 1 || gehecht(i)) continue;
                    if (nk > 1) { int kk2 = col[(int64_t)i * HM + h[i] - 1]; if (us_b < ustk[kk2]) continue; }
                    if (stug && us_b < (nk > 1 ? ustk[col[(int64_t)i * HM + h[i] - 1]] : P.ust_fl) * stugf(i)) continue;
                    uint8_t ke = P.bevroren ? 0 : (ctx = 4, pak(i)); ctx = 0;
                    if ((wand >= 2 || bijval_eerst) && !P.bevroren) lawine(i); l_typ = 2; lanceer(i, 0.3 + 0.2 * U(), 1.0, ke); if (!P.bevroren && wand < 2 && !bijval_eerst) lawine(i); n_aero++;
                }
            }
            if (windrol && nk > 1) {                     /* 16 Oct.: wind-driven sliding at the bed */
                double ub = windrol == 2 ? fmin(P.ustar, 0.8 * ustk[0]) : us_b;
                if (NY > 1) { for (int r = 0; r < NY; r++) { zet_rij(r); wind_rollen(ub, tupd); } }
                else wind_rollen(ub, tupd);
            }
            tupd = 0;
        }
    }
    if (NY > 1) zet_rij(0);
    for (int k = 0; k < NZ; k++) Uk_out[k] = Uk[k];
    tel[0] = n_ins; tel[1] = n_reb; tel[2] = n_ej; tel[3] = n_aero;
    *M_ = M;
    return 0;
}
double drag_uit(void) { double r = mom_drag / (t_tot * Lx * W); mom_drag = 0; t_tot = 0; return r; }
void tp_uit(double *o) { for (int k = 0; k < NZ; k++) o[k] = tp_last[k]; }
void profiel(double *z_out) { for (int k = 0; k < NZ; k++) z_out[k] = zk[k]; }
void par_echo(Par *q, double *o) { o[0] = q->dt; o[1] = q->t_upd; o[2] = q->relax; o[3] = q->Wl; o[4] = q->bevroren; o[5] = q->tanr; }

int mmax_uit(void) { int r = M_max; M_max = 0; return r; }
long nvol_uit(void) { return n_vol; }

/* 7 Oct.: SLOPE DIFFUSION BY SALTATION (Duran, Claudin & Andreotti 2014): the saltation flux has a
   downslope component, q = q_s - D dh/dx with D = b q_s,vol / phi (m^2/s). Applied to the bed:
   per column boundary top grains are moved from high to low, expected number D dt |dh/dx| / (dx dz)
   (stochastically rounded), in substeps with D dt_sub / dx^2 <= 0.4 (stable). Classes are conserved. Off in the article model. */
long n_diff = 0;
void diffusie(int n_, int hm, uint8_t *col_, int32_t *h_, Par *par, double D, double T) {
    N = n_; HM = hm; col = col_; h = h_; P = *par;
    if (D <= 0) return;
    int ns = (int)ceil(D * T / (0.4 * P.dx * P.dx)); if (ns < 1) ns = 1;
    double dts = T / ns;
    static int *nmv = 0, *rich = 0; static int ncap = 0;
    if (ncap < N) { free(nmv); free(rich); nmv = malloc(N * sizeof(int)); rich = malloc(N * sizeof(int)); ncap = N; }
    for (int s = 0; s < ns; s++) {
        for (int i = 0; i < N; i++) {            /* first all fluxes from the state at the start of the substep */
            int j = md(i + 1);
            double sl = (h[j] - h[i]) * P.dz / P.dx;
            double nexp = D * dts * fabs(sl) / (P.dx * P.dz);
            int nm = (int)nexp; if (U() < nexp - nm) nm++;
            nmv[i] = nm; rich[i] = h[j] > h[i] ? -1 : 1;   /* -1: from j to i, +1: from i to j */
        }
        for (int i = 0; i < N; i++) {            /* then apply */
            int j = md(i + 1), van = rich[i] > 0 ? i : j, naar = rich[i] > 0 ? j : i;
            for (int m2 = 0; m2 < nmv[i] && h[van] > 1; m2++) { uint8_t k = pak(van); leg(naar, k); n_diff++; }
        }
    }
}
long diff_uit(void) { long r = n_diff; n_diff = 0; return r; }

/* rest period over: all grains in the bed one level stiffer (up to stug_max) */
void stug_rust(int n_, int hm, int32_t *h_) {
    if (!stug) return;
    for (int i = 0; i < n_; i++) for (int z = 0; z < h_[i]; z++) { uint8_t *c = &stug[(int64_t)i * hm + z]; if (*c < stug_max) (*c)++; }
}
