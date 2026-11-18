"""Model run: grows a bed of 2140 mm from a flat bed with noise and records it (started by run_model.py).

Usage: python3 run_fetch.py ref <seed>   (settings as environment variables; see run_model.py and series.py)
Every OPNAME_DT seconds (default 30) it stores: the bed height per column, the number of impacts and of creep departures
per column in that window (counters in wind_korrel.c), wind-driven sliding and the wind profile. Output: <name>.npz in the
current folder. Most switches read below belong to variants or diagnostics from the development of the model; with
the switch off (the default) the result is bit-identical to the run without it. The article model is series SR0.
(Originally the 'fetch' experiment of 5 Oct. 2026; the variants 'hoek' and 'rol' of that sensitivity test are kept.)
"""
import os, sys, time, ctypes
os.environ.setdefault("PCR", "0.627")
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "ref"
ZAAD = int(sys.argv[2]) if len(sys.argv) > 2 else 261
CVAR = os.environ.get("CVAR", "C8")
sys.argv = [sys.argv[0], CVAR]
import numpy as np
import run_factor as RF          # sets all C8 settings (and the later variants, CVAR)
WK, L = RF.WK, RF.L
TANG = np.tan(np.deg2rad(20.0 if VARIANT == "hoek" else 12.0))
# 22 Oct.: SEPARATION ENVELOPE (imposed wind shadow) AND WIND-DRIVEN SLIDING in the growth runs (default off, then bit-identical)
OMH_GR = float(os.environ["OMH_GR"]) if os.environ.get("OMH_GR") else None
OMH_MOD = int(os.environ.get("OMH_MOD", "1")); OMH_GLAD = int(os.environ.get("OMH_GLAD", str(int(round(10.0 / RF.DX))))); OMH_SOM = [0.0, 0.0, 0]   # 26 Oct.: OMH_GLAD 10 mm
WR = int(os.environ.get("WINDROL", "0")); ctypes.c_int.in_dll(L, "windrol").value = WR
if os.environ.get("WR_ZETA"): ctypes.c_double.in_dll(L, "wr_zeta").value = float(os.environ["WR_ZETA"])
if os.environ.get("WR_CV"): ctypes.c_double.in_dll(L, "wr_cv").value = float(os.environ["WR_CV"])
if os.environ.get("WR_MENG"): ctypes.c_double.in_dll(L, "wr_meng").value = float(os.environ["WR_MENG"])   # 1 Nov.: WR_MENG
if os.environ.get("WR_PIV"): ctypes.c_double.in_dll(L, "wr_piv").value = float(os.environ["WR_PIV"])   # 23 Oct.: rolling threshold (dashboard)
if VARIANT == "rol": ctypes.c_double.in_dll(L, "l_schaduw").value = 1.0e-3
NAAM = (f"fetch_z{ZAAD}.npz" if VARIANT == "ref" else f"fetch_{VARIANT}_z{ZAAD}.npz")
if CVAR != "C8": NAAM = f"{CVAR.lower()}_z{ZAAD}.npz"
zaad, T, DX, GL = ZAAD, 5400.0, RF.DX, RF.GL
LX_MM = float(os.environ.get("LX_MM", "2140"))   # 23 Oct.: domain length (dashboard)
w = WK.WindKorrel(Lx_mm=LX_MM, seed=zaad, ruis=1, f0=RF.F0, dx=DX * 1e-3, Wl=GL, **RF.CELKW)   # 26 Oct.: depth, HM, noise, toppling, dz from mm
if os.environ.get("LAWINE_WL"): w.p.Wl = int(os.environ["LAWINE_WL"])   # 25 Oct.: window of the angle-of-repose rule in columns (default GL = 5, 2 mm); only avalanching, not smoothing (sensitivity test)
if os.environ.get("TOREN"): w.p.toren = int(os.environ["TOREN"])          # 25 Oct.: toppling rule in layers (default 3; sensitivity test)
GRAV = os.environ.get("GRAVEER")   # cut-in ripples (Andreotti et al. 2006): wavelength in mm
if GRAV:
    lam = float(GRAV); Hg = lam / 20.0                 # RI 20
    xx = (w.x % lam) / lam                              # 0..1 within a ripple; stoss 0-0.75, lee 0.75-1
    prof = np.where(xx < 0.75, xx / 0.75, (1 - xx) / 0.25) * Hg - Hg / 2
    rngG = np.random.default_rng(ZAAD + 77); basis = w.h.copy()
    for i in range(w.N):
        doel = int(round(basis[i] + prof[i] / (w.p.dz * 1e3)))
        if doel > w.h[i]:
            w.col[i * w.HM + w.h[i]: i * w.HM + doel] = rngG.choice(3, size=doel - w.h[i], p=RF.F0) if RF.F0 is not None else 0
        w.h[i] = doel
    w.korrels0 = int(w.h.sum())
SINUS = os.environ.get("BED_SINUS")   # 25 Oct.: imposed sine "n,a": wavelength LX_MM / n, amplitude a mm (sensitivity per wavelength; ripplets in series RA/RB)
if SINUS:
    n_s, a_s = int(SINUS.split(",")[0]), float(SINUS.split(",")[1]); rngS = np.random.default_rng(ZAAD + 78); basis = w.h.copy()
    prof = a_s * np.sin(2 * np.pi * n_s * np.arange(w.N) / w.N)
    for i in range(w.N):
        doel = int(round(basis[i] + prof[i] / (w.p.dz * 1e3)))
        if doel > w.h[i]:
            w.col[i * w.HM + w.h[i]: i * w.HM + doel] = rngS.choice(3, size=doel - w.h[i], p=RF.F0) if RF.F0 is not None else 0
        w.h[i] = doel
    w.korrels0 = int(w.h.sum())
BEVR = int(os.environ.get("BEVROREN", "0") or 0); HEJ_ = []; _hin = np.zeros(w.N, np.int32); _hej = np.zeros(w.N, np.int32)   # 25 Oct.: frozen bed (measurement)
if BEVR: w.p.bevroren = 1
tel0 = w.telling()
E_ = None
if OMH_GR is not None:
    E_ = np.zeros(w.N); L.zet_omh(E_.ctypes.data_as(ctypes.c_void_p)); ctypes.c_int.in_dll(L, "omh").value = 1
def omhullende():
    """envelope with slope tan(OMH_GR) from each crest downwind (bed smoothed over 10 mm);
    e = Ze - Zs to C; modulation and height on bed + e (OMH_MOD = 1)."""
    hz = w.h.astype(float) * w.p.dz
    if GL > 1:
        k_ = np.ones(GL) / GL
        hz = np.convolve(np.concatenate([hz[-GL:], hz, hz[:GL]]), k_, "same")[GL:-GL]
    G2 = OMH_GLAD; k2 = np.ones(G2) / G2; h2 = w.h.astype(float) * w.p.dz
    h2 = np.convolve(np.concatenate([h2[-G2:], h2, h2[:G2]]), k2, "same")[G2:-G2]
    T_ = np.tan(np.radians(OMH_GR)); J = int(min(w.N - 1, np.ceil((h2.max() - h2.min()) / (w.p.dx * T_)) + 2))
    ze = h2.copy()
    for j in range(1, J + 1): ze = np.maximum(ze, np.roll(h2, j) - j * w.p.dx * T_)
    E_[:] = ze - h2
    ze = hz + E_
    OMH_SOM[0] += float(np.mean(E_ > 0.5 * w.p.dz)); OMH_SOM[1] += float(E_.mean()); OMH_SOM[2] += 1
    ZE_[0] = ze
    if not OMH_MOD: return w.modulatie(3.0 * MOD_F, 1.0 * MOD_F, glad=GL), w.hoogte(glad=GL)
    m = omh_mod(MOD_F)
    e_mm = E_ * 1e3
    return m, w.hoogte(glad=GL) + (e_mm - e_mm.mean())
ZE_ = [None]
def omh_mod(F):
    """30 Oct. (MOD_FG): modulation with strength F on the bed + envelope of the last call of omhullende()"""
    ze = ZE_[0]; k = 2 * np.pi * np.fft.rfftfreq(w.N, d=w.p.dx); filt = np.exp(-0.5 * (k * 2.0e-3) ** 2)
    d = np.fft.irfft(WK.storing_kern(k, 3.0 * F, 1.0 * F, w.p.z0) * filt * np.fft.rfft(ze - ze.mean()), n=w.N)   # 9 Nov.: MOD_AB
    return np.sqrt(np.maximum(1.0 + d, 0.0))
ins = np.zeros(w.N, np.int64); kr = np.zeros(w.N, np.int64)
L.zet_tellers(ins.ctypes.data_as(ctypes.c_void_p), kr.ctypes.data_as(ctypes.c_void_p))
kra = np.zeros(w.N, np.int64); afz = np.zeros(w.N, np.int64)
L.zet_tellers2(kra.ctypes.data_as(ctypes.c_void_p), afz.ctypes.data_as(ctypes.c_void_p))
KA_, AF_, SK_, KS_ = [], [], [], []
WR_ = []; _wr = np.zeros(2); _wrs = np.zeros(2)
KAMSTAT = int(os.environ.get("KAMSTAT", "0")); KV = np.zeros(w.N, np.uint8); KH = np.zeros(7 * 400); KH_ = []; IS = np.zeros(7 * w.N); IS_ = []; KV_ = []   # 24 Oct.: diagnostics at the crest
if KAMSTAT:
    ctypes.c_int.in_dll(L, "sprong_aan").value = 1                 # needed for the jump length (bookkeeping only, no random numbers)
    L.zet_kam(KV.ctypes.data_as(ctypes.c_void_p), KH.ctypes.data_as(ctypes.c_void_p)); L.zet_ins_stat(IS.ctypes.data_as(ctypes.c_void_p))
def kam_zet():
    """columns within 2 mm of a crest (bed smoothed over 2 mm, ripples as in the analyses)"""
    hz = w.h.astype(float) * w.p.dz * 1e3; hz = np.convolve(np.concatenate([hz[-5:], hz, hz[:5]]), np.ones(5) / 5, "same")[5:-5]; hz -= hz.mean()
    rib, _, _ = RF.MM.ripples(np.arange(w.N) * DX, hz); KV[:] = 0
    for r in rib:
        for q_ in range(-5, 6): KV[(r["c"] + q_) % w.N] = 1
WRSTAT = int(os.environ.get("WRSTAT", "0")); WS = np.zeros(6 * w.N); WS_ = []   # 24 Oct.: start and stop positions of wind-driven sliding (diagnostics)
if os.environ.get("WR_NOMOD"): ctypes.c_int.in_dll(L, "wr_nomod").value = int(os.environ["WR_NOMOD"])   # 30 Oct.: GLIJTEL
if os.environ.get("WR_MAXSTAP"): ctypes.c_int.in_dll(L, "wr_maxstap").value = int(os.environ["WR_MAXSTAP"])
FLUXTEL = int(os.environ.get("FLUXTEL", "0") or 0); FT_ = []; FTS_ = []; _FTK = np.zeros(22 * w.N); _FTS = np.zeros(11)   # 30 Oct.: FLUXTEL; 31 Oct.: 11 kinds
MOD_LH = float(os.environ.get("MOD_LH", "0") or 0); MJH_ = np.ones(w.N); MPL_ = np.ones(w.N)   # 31 Oct.: MOD_LH (mm)
if MOD_LH > 0: ctypes.c_double.in_dll(L, "mod_lh").value = MOD_LH * 1e-3; L.zet_mod_lh(MJH_.ctypes.data_as(ctypes.c_void_p), MPL_.ctypes.data_as(ctypes.c_void_p))
MOD_ZC = float(os.environ.get("MOD_ZC", "0") or 0)   # 1 Nov.: MOD_ZC (mm)
if MOD_ZC != 0: ctypes.c_double.in_dll(L, "mod_zc").value = MOD_ZC * 1e-3; L.zet_mod_lh(MJH_.ctypes.data_as(ctypes.c_void_p), MPL_.ctypes.data_as(ctypes.c_void_p))
if FLUXTEL: ctypes.c_int.in_dll(L, "sprong_aan").value = 1; L.zet_ft_kol(_FTK.ctypes.data_as(ctypes.c_void_p))
GLIJTEL = int(os.environ.get("GLIJTEL", "0") or 0); GL_ = []; _GLK = np.zeros(3 * w.N)
if GLIJTEL: L.zet_gl_kol(_GLK.ctypes.data_as(ctypes.c_void_p))
if WRSTAT: L.zet_wr_stat(WS.ctypes.data_as(ctypes.c_void_p))   # 23 Oct.: wind-driven sliding per window (number, sum of sliding distance in m)
GROFTEL = int(os.environ.get("GROFTEL", "0") or 0)   # 10 Oct.: counters of coarse grains per process (diagnostics only); 19 Oct.: 2 = all classes, fine and coarse separately
NGT = 20 if GROFTEL == 2 else 10; gt = np.zeros(NGT * w.N, np.int64); GT_ = []
if GROFTEL == 2: ctypes.c_int.in_dll(L, "gtel_alle").value = 1
if os.environ.get("NAAMSUF"): NAAM = NAAM.replace(".npz", os.environ["NAAMSUF"] + ".npz")   # 11 Oct.: own name suffix
if GROFTEL: L.zet_grof_tellers(gt.ctypes.data_as(ctypes.c_void_p)); NAAM = NAAM.replace(".npz", "_groftel.npz")
T = float(os.environ.get("TEIND", T))
USTAR = float(os.environ.get("USTAR", "0.35"))
LDUW = int(os.environ.get("LANCEER_DUW", "0")); LD_ = []; _ld = np.zeros(2, np.int64)   # 24 Oct.: launching by a push
if LDUW: ctypes.c_int.in_dll(L, "lanceer_duw").value = 1; ctypes.c_int.in_dll(L, "sprong_aan").value = 1
SPRONGHIST = int(os.environ.get("SPRONGHIST", "0") or 0); SH_ = []; _sh = np.zeros(3 * 2000, np.int64)   # 25 Oct.: distribution of jump length per type (0 ejection, 1 rebound, 2 wind), cumulative per record
if SPRONGHIST: ctypes.c_int.in_dll(L, "sprong_aan").value = 1; L.sprong_reset()
if os.environ.get("ALLEEN_DALEND"): ctypes.c_int.in_dll(L, "alleen_dalend").value = int(os.environ["ALLEEN_DALEND"])   # 26 Oct.: only descending grains impact
if os.environ.get("BIJVAL_EERST"): ctypes.c_int.in_dll(L, "bijval_eerst").value = int(os.environ["BIJVAL_EERST"])   # 26 Oct.: fill the hole first, then launch
_LAW = np.zeros(6); LAW_ = []; _LD = np.zeros(57); LD2_ = []; KETENREC = int(os.environ.get("KETENREC", "0") or 0); REC_T0 = float(os.environ.get("REC_T0", "0"))   # 27 Oct.: KETENREC
if KETENREC:
    ctypes.c_int.in_dll(L, "rec_min").value = int(os.environ.get("REC_MIN", "30")); ctypes.c_int.in_dll(L, "rec_max").value = int(os.environ.get("REC_MAX", "12"))
    ctypes.c_int.in_dll(L, "rec_hw").value = int(round(15.0 / DX))
    _RM = np.zeros(12 * 6, np.int32); _RS = np.zeros(12 * 4000 * 5, np.int32); _RH0 = np.zeros(12 * 151, np.int32); _RH1 = np.zeros(12 * 151, np.int32); _RD = np.zeros(12 * 4000 * 2)
    L.rec_zet(*[a.ctypes.data_as(ctypes.c_void_p) for a in (_RM, _RS, _RH0, _RH1, _RD)])
GPKOL = int(os.environ.get("GPKOL", "0") or 0); GP_ = []   # 27 Oct.: GPKOL
STIJGKOL = int(os.environ.get("STIJGKOL", "0") or 0); SKK_ = []   # 28 Oct.: STIJGKOL
RBKOL = int(os.environ.get("RBKOL", "0") or 0); RBK_ = []   # 4 Nov.: ART_KR, ART_EJ
SKM_ = []   # 29 Oct.: KORRELMASSA
if STIJGKOL: assert SPRONGHIST, "STIJGKOL requires SPRONGHIST=1"; _SKK = np.zeros(3 * w.N); L.zet_stijg_kol(_SKK.ctypes.data_as(ctypes.c_void_p))
if GPKOL: _GP = np.zeros(12 * w.N, np.int32); L.gp_zet(_GP.ctypes.data_as(ctypes.c_void_p))
if RBKOL: _RBK = np.zeros(2 * w.N); L.zet_rb_kol(_RBK.ctypes.data_as(ctypes.c_void_p))
RBTEL = int(os.environ.get("RBTEL", "0") or 0); RBT_ = []; RBS_ = []; _RBS = np.zeros(2)   # 6 Nov.: RBTEL
if RBTEL: _RBT = np.zeros(6 * w.N); L.zet_rbt(_RBT.ctypes.data_as(ctypes.c_void_p))
HKTEL = int(os.environ.get("HKTEL", "0") or 0); HKK_ = []; HKS_ = []   # 7 Nov.: HKTEL
if HKTEL: _HKK = np.zeros(36 * w.N); _HKS = np.zeros(128); L.zet_hk(_HKK.ctypes.data_as(ctypes.c_void_p), _HKS.ctypes.data_as(ctypes.c_void_p))
KRTEL = int(os.environ.get("KRTEL", "0") or 0); KTK_ = []; KTH_ = []; KTS_ = []; _KTS = np.zeros(8)   # 7 Nov.: KRTEL
if KRTEL: _KTK = np.zeros(6 * w.N); _KTH = np.zeros(592); L.zet_kt(_KTK.ctypes.data_as(ctypes.c_void_p), _KTH.ctypes.data_as(ctypes.c_void_p))
_KB = np.zeros(4); KB_ = []   # 27 Oct.: KB_MM
ctypes.c_int.in_dll(L, "rt_n1").value = max(1, int(round(0.8 / DX))); _RT = np.zeros(10); RT_ = []   # 27 Oct.: counter of creep distance per initial slope
if int(os.environ.get("ROLREGEL", "0") or 0):   # 27 Oct.: ROLREGEL
    ctypes.c_double.in_dll(L, "rol_mud").value = float(os.environ.get("ROL_MUD", "0")); ctypes.c_int.in_dll(L, "rol_blok").value = int(os.environ.get("ROL_BLOK", "0") or 0)   # 27 Oct.: ROL_MUD
    ctypes.c_int.in_dll(L, "rolregel").value = 1; ctypes.c_double.in_dll(L, "rol_lcr").value = float(os.environ.get("ROL_LCR", "4.1")) * 1e-3
    ctypes.c_int.in_dll(L, "rol_ns").value = max(1, int(round(float(os.environ.get("ROL_S_MM", "1.0")) / DX)))
    ctypes.c_int.in_dll(L, "rol_nmin").value = max(1, int(round(float(os.environ.get("ROL_MIN_MM", "0.4")) / DX)))
if os.environ.get("KB_AFST"): ctypes.c_int.in_dll(L, "kb_n").value = max(1, int(round(float(os.environ["KB_AFST"]) / DX)))   # 27 Oct.: KB_AFST
_LK = np.zeros(44); LK_ = []; _LKK = np.zeros(w.N); LKK_ = []; L.lk_zet(_LKK.ctypes.data_as(ctypes.c_void_p))   # 27 Oct.: avalanche chains   # 27 Oct.: avalanche diagnostics counter   # 27 Oct.: toppling rule counter
_ZIJ = np.zeros(8); ZIJ_ = []; _BEGR = np.zeros(7); BEGR_ = []
if os.environ.get("BEGRAVEN"): ctypes.c_int.in_dll(L, "begr_aan").value = 1   # 26 Oct.: buried counter   # 26 Oct.: top/side impact
_TEL = np.zeros(4, np.int64); TEL_ = []; _ST = np.zeros(8, np.int64); ST_ = []   # 26 Oct. (cell size diagnostics)
if os.environ.get("EJ_VFAC"): ctypes.c_double.in_dll(L, "ej_vfac").value = float(os.environ["EJ_VFAC"])   # 25 Oct.: ejection speed x factor (reptation jump test)
if os.environ.get("IMPULSDUW"):                                  # 24 Oct.: impulse push (number of pushes from the momentum loss along the x-axis)
    ctypes.c_int.in_dll(L, "kruip_impuls").value = 3; ctypes.c_double.in_dll(L, "kr_bxref").value = float(os.environ["IMPULSDUW"])
BX_ = []; _bx = np.zeros(3)
_BP = np.zeros(8, np.int64); BP_ = []   # 6 Nov.: TF_BEGRENS, KRUIP_IMPULS, INSLAG_STEIL
if os.environ.get("COHESIE"): ctypes.c_double.in_dll(L, "cohesie").value = float(os.environ["COHESIE"])   # 24 Oct.: cohesion number Co in the ripple body
if os.environ.get("L_SCHADUW"): ctypes.c_double.in_dll(L, "l_schaduw").value = float(os.environ["L_SCHADUW"]) * 1e-3 if float(os.environ["L_SCHADUW"]) > 0 else -1.0   # 25 Oct.: creep distance of coarse grains in the shadow (mm; <= 0: one grain diameter, as C4)
if os.environ.get("LAWINE_HOEK"): ctypes.c_int.in_dll(L, "lawine_hoek").value = int(os.environ["LAWINE_HOEK"])   # 24 Oct.: 0 = angle-of-repose rule off
if os.environ.get("TERREIN_LAWINE"): ctypes.c_int.in_dll(L, "ter_lawrichting").value = int(os.environ["TERREIN_LAWINE"])   # 24 Oct.: 1 down the lee side, 2 down the stoss side, 3 both
if os.environ.get("TERREIN_DELEN"): ctypes.c_int.in_dll(L, "ter_delen").value = int(os.environ["TERREIN_DELEN"])   # 23 Oct.: decomposition of the terrain angle
if float(os.environ.get("TERREIN", "0")): ctypes.c_double.in_dll(L, "terrein").value = float(os.environ["TERREIN"]) * np.pi / 180   # 23 Oct.: terrain angle (degrees)
PLAFOND = int(os.environ.get("PLAFOND", "1"))
STROOM = int(os.environ.get("STROOM", "0") or 0)   # 13 Nov.: STROOM, streamlines follow the bed
TPTEL = int(os.environ.get("TPTEL", "0") or 0); TP_, TPU_ = [], []; _tp1 = np.zeros(WK.NZ); _tps = np.zeros(WK.NZ); _tpn = [0]   # 11 Nov.: TPTEL
MOD_F = float(os.environ.get("MOD_F", "1") or 1); MODTEL = int(os.environ.get("MODTEL", "0") or 0); MD_ = []   # 30 Oct.: MOD_F
MOD_FG = float(os.environ["MOD_FG"]) if os.environ.get("MOD_FG") else None; WRM_ = np.ones(w.N); MDG_ = []   # 30 Oct.: MOD_FG
if MOD_FG is not None: L.zet_wr_mod(WRM_.ctypes.data_as(ctypes.c_void_p))
ODT = int(os.environ.get("OPNAME_DT", "30"))   # 23 Oct.: seconds per record (default 30); h0 = bed at t = 0
H0 = w.h.copy()
RUST = [float(v) for v in os.environ["RUST"].split(",")] if os.environ.get("RUST") else None
KORST = float(os.environ["KORST"]) if os.environ.get("KORST") else None
korst = np.zeros(w.N, np.int32); in_rust = False
OPEN = bool(os.environ.get("OPEN"))
DROOG = float(os.environ.get("DROOG", "0"))
BDIFF = float(os.environ.get("BDIFF", "0"))
STUG = float(os.environ.get("STUG", "0"))
HARD_PB = os.environ.get("HARD_PB")
if HARD_PB:
    ctypes.c_int.in_dll(L, "korst_hard").value = 1; ctypes.c_double.in_dll(L, "korst_pb").value = float(HARD_PB)
stug = np.zeros(w.N * w.HM, np.uint8)
if STUG > 0:
    ctypes.c_double.in_dll(L, "stug_a").value = STUG; L.zet_stug(stug.ctypes.data_as(ctypes.c_void_p))
ND_ = []
if OPEN: ctypes.c_int.in_dll(L, "open_rand").value = 1
open_tel = np.zeros(2, np.int64); rngI = np.random.default_rng(ZAAD + 900)
if KORST:
    ctypes.c_double.in_dll(L, "korst_fac").value = KORST; L.zet_korst(korst.ctypes.data_as(ctypes.c_void_p))
rngU = np.random.default_rng(zaad + 500); Uw = 5.0
LAAG_ALFA = float(os.environ.get("LAAG_ALFA", "0")); LAAG_BLOK = float(os.environ.get("LAAG_BLOK", "0")); LAAG_LREF = float(os.environ.get("LAAG_LREF", "100"))
ctypes.c_double.in_dll(L, "alfa_ref").value = LAAG_ALFA; ctypes.c_int.in_dll(L, "nblok").value = int(round(LAAG_BLOK / DX))   # 12 Oct.: without a wind layer parallel to the bed
H_, I_, K_, t_, U_, G_ = [], [], [], [], [], []
t0 = time.time()
while w.t < T - 1e-9:
    us = []
    for _ in range(ODT):
        if RF.VLAGEN:                                   # 19 Oct.: VLAGEN=0 = constant wind (u* 0.35 m/s)
            Uw += -(Uw - 5.0) / 20.0 + 1.75 * np.sqrt(2.0 / 20.0) * rngU.standard_normal()
            Uw = min(max(Uw, 1.0), 8.0)
        w.p.ustar = USTAR * Uw / 5.0                     # 23 Oct.: USTAR (default 0.35 m/s)
        if RUST and (w.t % (RUST[0] + RUST[1])) >= RUST[0]: w.p.ustar = 0.02   # rest period: no wind
        nu_rust = bool(RUST) and (w.t % (RUST[0] + RUST[1])) >= RUST[0]
        if STUG > 0 and in_rust and not nu_rust: L.stug_rust(w.N, w.HM, w.h)
        if KORST and in_rust and not nu_rust:
            korst[:] = w.h   # end of rest: surface bonded
            korst[:int(DROOG * w.N)] = 0   # dry spot: never bonds (source of saltation after rest)
        in_rust = nu_rust
        if OMH_GR is not None: m, hb_ = omhullende()
        else:
            m = w.modulatie(3.0 * MOD_F, 1.0 * MOD_F, glad=GL)
            hb_ = w.hoogte(glad=GL)
        w.mod = np.ascontiguousarray(m * 107.0 / (107.0 - hb_) if PLAFOND else m)   # 23 Oct.: PLAFOND=0 without acceleration under the ceiling (diagnostics)
        if MOD_LH > 0 or MOD_ZC != 0: MJH_[:] = m; MPL_[:] = 107.0 / (107.0 - hb_) if PLAFOND else 1.0   # 31 Oct.: MOD_LH
        if MOD_FG is not None:   # 30 Oct.: MOD_FG, own wind factor for the sliding rule
            mg = omh_mod(MOD_FG) if (OMH_GR is not None and OMH_MOD) else w.modulatie(3.0 * MOD_FG, 1.0 * MOD_FG, glad=GL)
            if not (OMH_GR is not None and OMH_MOD): w.mod = np.ascontiguousarray(m * 107.0 / (107.0 - hb_) if PLAFOND else m)
            WRM_[:] = mg * 107.0 / (107.0 - hb_) if PLAFOND else mg
        if STROOM: w.stroomlijn(glad=GL)   # 13 Nov.: STROOM
        w.schaduwkaart(tan_g=TANG, glad=GL); RF.steun_zet(w)
        if LAAG_ALFA > 0: w.zet_referentie(LAAG_LREF)
        if OPEN and not (bool(RUST) and (w.t % (RUST[0] + RUST[1])) >= RUST[0]):
            S_ = w.p.ustar ** 2 / ((2650 / 1.2 - 1) * 9.81 * 0.2e-3); Sd_ = 0.2 ** 2 / ((2650 / 1.2 - 1) * 9.81 * 0.2e-3)
            q_in = 1.3 * 28.5 * 2650 * 0.2e-3 * np.sqrt(9.81 * 0.2e-3) * max(S_ - Sd_, 0)   # saturated flux (bel20: 1.3 x formula)
            Wm = (np.pi / 6 * w.p.d ** 3) / (w.p.dx * w.p.dz * 0.6); mg = 2650 * np.pi / 6 * w.p.d ** 3
            lam_ = q_in * Wm / mg                       # model grains per second through x = 0
            L.injecteer_zet(int(rngI.poisson(lam_)), 0.088, 0.010)
        if KAMSTAT and int(w.t) % 5 == 0: kam_zet()
        if KETENREC and w.t >= REC_T0 - 1e-9: ctypes.c_int.in_dll(L, "rec_aan").value = 1   # 27 Oct.
        (w.draai(1.0, _hin, _hej) if BEVR else w.draai(1.0)); us.append(Uw); _TEL += w.tel   # 26 Oct. (cell size diagnostics): counters per record
        if TPTEL: L.tp_uit(_tp1); _tps += _tp1; _tpn[0] += 1   # 11 Nov.: TPTEL
        if WR: L.wr_uit(_wr.ctypes.data_as(ctypes.c_void_p)); _wrs += _wr
        if BDIFF > 0:
            L.diffusie(w.N, w.HM, w.col, w.h, ctypes.byref(w.p), BDIFF * w.flux() / (2650 * 0.6), 1.0)
    H_.append(w.h.copy()); I_.append(ins.copy()); K_.append(kr.copy()); t_.append(w.t); U_.append(np.mean(us)); G_.append(w.col[np.arange(w.N) * w.HM + w.h - 1].copy())
    KS_.append(float(np.mean(stug[np.arange(w.N) * w.HM + w.h - 1])) if STUG > 0 else float(np.mean((korst > 0) & (korst == w.h))));
    KA_.append(kra.copy()); AF_.append(afz.copy()); SK_.append(L.kruip_uit()); SKM_.append(L.kruip_m_uit()); kra[:] = 0; afz[:] = 0; WR_.append(_wrs.copy()); _wrs[:] = 0
    if WRSTAT: WS_.append(WS.reshape(6, w.N).copy()); WS[:] = 0
    if GLIJTEL: GL_.append(_GLK.reshape(3, w.N).astype(np.float32)); _GLK[:] = 0   # 30 Oct.
    if FLUXTEL: FT_.append(_FTK.reshape(22, w.N).astype(np.float32)); _FTK[:] = 0; L.ft_uit(_FTS.ctypes.data_as(ctypes.c_void_p)); FTS_.append(_FTS.copy())   # 30 Oct.
    if MODTEL: MD_.append(np.asarray(w.mod, np.float32).copy()); MDG_.append(WRM_.astype(np.float32))   # 30 Oct.: MOD_F, MOD_FG
    if TPTEL: TP_.append(_tps / max(_tpn[0], 1)); TPU_.append(w.p.ustar); _tps[:] = 0; _tpn[0] = 0   # 11 Nov.: TPTEL
    L.bx_uit(_bx.ctypes.data_as(ctypes.c_void_p)); BX_.append(_bx.copy())
    L.beperk_uit(_BP.ctypes.data_as(ctypes.c_void_p)); BP_.append(_BP.copy())   # 6 Nov.
    L.lduw_uit(_ld.ctypes.data_as(ctypes.c_void_p)); LD_.append(_ld.copy())   # sum Bx, number of impacts, sum of pushes
    if SPRONGHIST: L.sprong_uit(_sh); SH_.append(_sh.copy()); L.stijg_uit(_ST.ctypes.data_as(ctypes.c_void_p)); ST_.append(_ST.copy())
    TEL_.append(_TEL.copy()); _TEL[:] = 0; L.zij_uit(_ZIJ.ctypes.data_as(ctypes.c_void_p)); ZIJ_.append(_ZIJ.copy()); L.begr_uit(_BEGR.ctypes.data_as(ctypes.c_void_p)); BEGR_.append(_BEGR.copy()); L.law_uit(_LAW.ctypes.data_as(ctypes.c_void_p)); LAW_.append(_LAW.copy()); L.ld_uit(_LD.ctypes.data_as(ctypes.c_void_p)); LD2_.append(_LD.copy()); L.lk_uit(_LK.ctypes.data_as(ctypes.c_void_p)); LK_.append(_LK.copy()); LKK_.append(_LKK.astype(np.float32)); _LKK[:] = 0; L.kb_uit(_KB.ctypes.data_as(ctypes.c_void_p)); KB_.append(_KB.copy()); L.rt_uit(_RT.ctypes.data_as(ctypes.c_void_p)); RT_.append(_RT.copy())
    if GPKOL: GP_.append(_GP.reshape(12, w.N).copy()); _GP[:] = 0   # 27 Oct.
    if STIJGKOL: SKK_.append(_SKK.reshape(3, w.N).astype(np.float32)); _SKK[:] = 0   # 28 Oct.
    if RBKOL: RBK_.append(_RBK.reshape(2, w.N).astype(np.float32)); _RBK[:] = 0   # 4 Nov.
    if KRTEL: KTK_.append(_KTK.reshape(6, w.N).astype(np.float32)); _KTK[:] = 0; KTH_.append(_KTH.copy()); _KTH[:] = 0; L.kt_uit(_KTS.ctypes.data_as(ctypes.c_void_p)); KTS_.append(_KTS.copy())   # 7 Nov.
    if HKTEL: HKK_.append(_HKK.reshape(36, w.N).astype(np.float32)); _HKK[:] = 0; HKS_.append(_HKS.copy()); _HKS[:] = 0   # 7 Nov.: HKTEL
    if RBTEL: RBT_.append(_RBT.reshape(6, w.N).astype(np.float32)); _RBT[:] = 0; L.rbt_uit(_RBS.ctypes.data_as(ctypes.c_void_p)); RBS_.append(_RBS.copy())   # 6 Nov.
    if BEVR: HEJ_.append(_hej.copy()); _hej[:] = 0; _hin[:] = 0   # ejections + creep departures per column (ejections = hej - kr)
    if KAMSTAT: KH_.append(KH.reshape(7, 400).copy()); KH[:] = 0; IS_.append(IS.reshape(7, w.N).copy()); IS[:] = 0; KV_.append(KV.copy())
    ins[:] = 0; kr[:] = 0
    if GROFTEL: GT_.append(gt.reshape(NGT, w.N).copy()); gt[:] = 0
    if int(w.t) % 300 == 0:
        h = w.hoogte(glad=GL)
        print(f"t {w.t:5.0f}  L_piek {RF.piek(h, DX, Lmax=800):4.0f}  rms {h.std():.2f} mm  ({time.time()-t0:.0f} s)", flush=True)
        np.savez_compressed(NAAM, dz=w.p.dz * 1e3, h=np.array(H_), ins=np.array(I_), kr=np.array(K_), t=np.array(t_), U=np.array(U_), top=np.array(G_), kra=np.array(KA_), afz=np.array(AF_), skr=np.array(SK_), korst=np.array(KS_), gt=np.array(GT_), dx=DX, h0=H0, omh_frac=OMH_SOM[0] / max(OMH_SOM[2], 1), omh_gem=OMH_SOM[1] / max(OMH_SOM[2], 1), wr=np.array(WR_), ws=np.array(WS_), kh=np.array(KH_), ist=np.array(IS_), kv=np.array(KV_), bx=np.array(BX_), ld=np.array(LD_))
hmx_ = int(w.h.max()); BED = w.col.reshape(w.N, w.HM)[:, :hmx_].copy()   # 24 Oct.: complete bed structure at the end (grain class per cell)
np.savez_compressed(NAAM, dz=w.p.dz * 1e3, h=np.array(H_), ins=np.array(I_), kr=np.array(K_), t=np.array(t_), U=np.array(U_), top=np.array(G_), kra=np.array(KA_), afz=np.array(AF_), skr=np.array(SK_), korst=np.array(KS_), gt=np.array(GT_), dx=DX, h0=H0, omh_frac=OMH_SOM[0] / max(OMH_SOM[2], 1), omh_gem=OMH_SOM[1] / max(OMH_SOM[2], 1), wr=np.array(WR_), ws=np.array(WS_), kh=np.array(KH_), ist=np.array(IS_), kv=np.array(KV_), bx=np.array(BX_), ld=np.array(LD_), bed=BED, bed_h=w.h.copy())
if SPRONGHIST: _d = dict(np.load(NAAM)); _d["sh"] = np.array(SH_).reshape(len(SH_), 3, 2000); _d["stijg"] = np.array(ST_); np.savez_compressed(NAAM, **_d)
_d = dict(np.load(NAAM)); _d["tel"] = np.array(TEL_); _d["zij"] = np.array(ZIJ_); _d["begr"] = np.array(BEGR_); _d["lawtel"] = np.array(LAW_); _d["lawdiag"] = np.array(LD2_); _d["lawketen"] = np.array(LK_); _d["lk_kol"] = np.array(LKK_); _d["kb"] = np.array(KB_); _d["roltel"] = np.array(RT_); (_d.update(gp=np.array(GP_)) if GPKOL else None); (_d.update(rec_n=ctypes.c_int.in_dll(L, "rec_n").value, rec_hw=ctypes.c_int.in_dll(L, "rec_hw").value, rec_meta=_RM.reshape(12, 6), rec_st=_RS.reshape(12, 4000, 5), rec_h0=_RH0.reshape(12, 151), rec_h1=_RH1.reshape(12, 151), rec_d=_RD.reshape(12, 4000, 2)) if KETENREC else None); np.savez_compressed(NAAM, **_d)   # 26 Oct. (cell size diagnostics)
if BEVR: _d = dict(np.load(NAAM)); _d["hej"] = np.array(HEJ_); np.savez_compressed(NAAM, **_d)
if STIJGKOL: _d = dict(np.load(NAAM)); _d["sk"] = np.array(SKK_); np.savez_compressed(NAAM, **_d)   # 28 Oct.
if RBKOL: _d = dict(np.load(NAAM)); _d["rb"] = np.array(RBK_); np.savez_compressed(NAAM, **_d)   # 4 Nov.
if KRTEL: _d = dict(np.load(NAAM)); _d["ktk"] = np.array(KTK_); _d["kth"] = np.array(KTH_); _d["kts"] = np.array(KTS_); np.savez_compressed(NAAM, **_d)   # 7 Nov.: KRTEL
if HKTEL: _d = dict(np.load(NAAM)); _d["hkk"] = np.array(HKK_); _d["hks"] = np.array(HKS_); np.savez_compressed(NAAM, **_d)   # 7 Nov.: HKTEL
if RBTEL: _d = dict(np.load(NAAM)); _d["rbt"] = np.array(RBT_); _d["rbs"] = np.array(RBS_); np.savez_compressed(NAAM, **_d)   # 6 Nov.: RBTEL
_d = dict(np.load(NAAM)); _d["skm"] = np.array(SKM_); np.savez_compressed(NAAM, **_d)   # 29 Oct.: creep weighted with m_k
if GLIJTEL: _d = dict(np.load(NAAM)); _d["gl"] = np.array(GL_); np.savez_compressed(NAAM, **_d)   # 30 Oct.: GLIJTEL
if FLUXTEL: _d = dict(np.load(NAAM)); _d["ft"] = np.array(FT_); _d["fts"] = np.array(FTS_); np.savez_compressed(NAAM, **_d)   # 30 Oct.: FLUXTEL
if MODTEL: _d = dict(np.load(NAAM)); _d["md"] = np.array(MD_); (_d.update(mdg=np.array(MDG_)) if MOD_FG is not None else None); np.savez_compressed(NAAM, **_d)   # 30 Oct.
if BEVR: assert int(w.h.sum()) == w.korrels0   # frozen: bed unchanged
elif not OPEN: assert w.massa_ok() and np.array_equal(w.telling(), tel0)
else: L.open_uit(open_tel); print('OPEN: inside', open_tel[0], 'out', open_tel[1])
_d = dict(np.load(NAAM)); _d["beperk"] = np.array(BP_); np.savez_compressed(NAAM, **_d)   # 6 Nov.
_bp = np.array(BP_).sum(0); print(f"CAPPED: air shear stress capped at {_bp[1] / max(_bp[0], 1):.3f} of the profile levels; too steep {_bp[3] / max(_bp[2], 1):.4f} of the impacts; creep pushes refused {_bp[5] / max(_bp[4], 1):.3f}", flush=True)
if TPTEL: _d = dict(np.load(NAAM)); _d["tp"] = np.array(TP_); _d["tp_u"] = np.array(TPU_); _d["tp_z"] = w.profiel()[0]; np.savez_compressed(NAAM, **_d)   # 11 Nov.: TPTEL
print("END", flush=True)
