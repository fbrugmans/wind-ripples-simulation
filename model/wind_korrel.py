"""Python wrapper around wind_korrel.c: wind-coupled grain model (3 Oct. 2026).

Default values, fixed in advance:
  grain    d = 0.2 mm (Hulshorster Zand trough median ~187 um, crest median 227 um), rho_p 2650 kg/m3
  air      rho_a 1.2 kg/m3, nu 1.5e-5 m2/s, kappa 0.4, z0 = d/30
  wind     u* = 0.35 m/s. Estimate for the 1983 field: 5-8 m/s at 10 cm gives, with the saltation roughness
           of Owen (z0s ~ 0.1 mm), u* ~ 0.3-0.45 m/s; chosen: the middle.
  threshold fluid threshold u*_ft = 0.26 m/s (Shao & Lu 2000, d = 0.2 mm)
  splash   COMSALT Table 1 (Kok & Renno 2009): B 0.96, gamma 1 s/m, e_reb 0.45 +- 0.22,
           rebound angle Exp(40 deg), a 0.02, alpha_ej 0.15, ejection angle Exp(50 deg), D = d
           (in the article model the collision rules of Beladjine et al. 2007 replace part of this; see run_factor.py)
  delay    eps ~ Exp(2 d) (Lester et al. 2025)
  bed      dx 0.4 mm, dz 0.2 mm, avalanche if the slope over 5 columns > tan 32 deg, toppling >= 3 layers
  numerics dt 5e-5 s; wind profile every 0.01 s from the grain momentum flux, filtered over 0.1 s
"""
import ctypes
import os
import subprocess

import numpy as np

HIER = os.path.dirname(os.path.abspath(__file__))
_BIB = os.path.join(HIER, "wind_korrel.dylib")


class Par(ctypes.Structure):
    _fields_ = [(n, ctypes.c_double) for n in
                ("d", "rho_p", "rho_a", "nu", "g", "ustar", "kappa", "z0", "ust_fl", "aero_rate",
                 "B", "gam", "ereb_m", "ereb_s", "threb", "a", "alpha_ej", "thej", "Dsp", "eps_d",
                 "dx", "dz", "tanr")] + [("Wl", ctypes.c_int), ("toren", ctypes.c_int)] + \
               [(n, ctypes.c_double) for n in ("dt", "t_upd", "relax")] + [("bevroren", ctypes.c_int)]


def _bouw():
    src = os.path.join(HIER, "wind_korrel.c")
    if (not os.path.exists(_BIB)) or os.path.getmtime(_BIB) < os.path.getmtime(src):
        subprocess.check_call(["clang", "-O3", "-shared", "-fPIC", src, "-o", _BIB])
    lib = ctypes.CDLL(_BIB)
    Pn = np.ctypeslib.ndpointer
    i, d = ctypes.c_int, ctypes.c_double
    lib.draai.restype = i
    lib.draai.argtypes = [i, i, Pn(np.uint8, flags="C"), Pn(np.int32, flags="C"), ctypes.POINTER(Par), d,
                          Pn(np.float64, flags="C"), Pn(np.float64, flags="C"), Pn(np.float64, flags="C"),
                          Pn(np.float64, flags="C"), Pn(np.uint8, flags="C"), ctypes.POINTER(i), i,
                          Pn(np.float64, flags="C"), Pn(np.int64, flags="C"), ctypes.c_void_p, ctypes.c_void_p,
                          ctypes.c_void_p]
    lib.zaai.argtypes = [ctypes.c_uint64]
    lib.profiel.argtypes = [Pn(np.float64, flags="C")]
    lib.tp_uit.argtypes = [Pn(np.float64, flags="C")]
    lib.drag_uit.restype = d
    lib.f_vlak_uit.restype = d
    lib.kruip_uit.restype = d
    lib.mmax_uit.restype = i
    lib.inslag_uit.argtypes = [Pn(np.float64, flags="C")]
    lib.hoek_uit.argtypes = [Pn(np.int64, flags="C")]
    lib.injecteer_zet.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_double]
    lib.open_uit.argtypes = [Pn(np.int64, flags="C")]
    lib.diffusie.argtypes = [ctypes.c_int, ctypes.c_int, Pn(np.uint8, flags="C"), Pn(np.int32, flags="C"), ctypes.POINTER(Par), ctypes.c_double, ctypes.c_double]
    lib.diff_uit.restype = ctypes.c_long
    lib.zet_stug.argtypes = [ctypes.c_void_p]
    lib.stug_rust.argtypes = [ctypes.c_int, ctypes.c_int, Pn(np.int32, flags="C")]
    lib.zet_korst.argtypes = [ctypes.c_void_p]
    lib.zet_tellers2.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.zet_href.argtypes = [ctypes.c_void_p]
    lib.zet_omh.argtypes = [ctypes.c_void_p]   # 18 Oct.: separation envelope (imposed wind shadow)
    lib.zet_stroom.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]   # 13 Nov.: STROOM
    lib.stroom_hel.argtypes = [ctypes.c_double, ctypes.c_int, ctypes.c_int]; lib.stroom_hel.restype = ctypes.c_double
    lib.zet_steun.argtypes = [ctypes.c_void_p]   # 20 Oct.: support and sorting
    lib.wind_bij.argtypes = [ctypes.c_void_p, ctypes.c_double]
    lib.zet_tellers.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.hoogte_uit.argtypes = [Pn(np.int64, flags="C")]
    lib.sprong_uit.argtypes = [Pn(np.int64, flags="C")]
    lib.begrensd_uit.restype = ctypes.c_long
    lib.nvol_uit.restype = ctypes.c_long
    lib.zet_schaduw.argtypes = [ctypes.c_void_p]
    return lib


LIB = _bouw()
# 17 Oct.: ejection rules via the environment (default off), the same for all scripts
if os.environ.get("EJ_MIN"): ctypes.c_int.in_dll(LIB, "ej_min").value = int(os.environ["EJ_MIN"])
if os.environ.get("WAND"): ctypes.c_int.in_dll(LIB, "wand").value = int(os.environ["WAND"])   # 18 Oct.
if os.environ.get("TF_BEGRENS"): ctypes.c_int.in_dll(LIB, "tf_begrens").value = int(os.environ["TF_BEGRENS"])   # 6 Nov.: TF_BEGRENS, KRUIP_IMPULS, INSLAG_STEIL
if os.environ.get("INSLAG_STEIL"): ctypes.c_int.in_dll(LIB, "inslag_steil").value = int(os.environ["INSLAG_STEIL"])
if os.environ.get("KRUIP_TEGEN"): ctypes.c_int.in_dll(LIB, "kruip_tegen").value = int(os.environ["KRUIP_TEGEN"])   # 7 Nov.: KRUIP_TEGEN
if os.environ.get("KRUIP_TERUG"): ctypes.c_int.in_dll(LIB, "kruip_terug").value = int(os.environ["KRUIP_TERUG"])   # 8 Nov.: KRUIP_TERUG
if os.environ.get("KAATS_TERUG"): ctypes.c_int.in_dll(LIB, "kaats_terug").value = int(os.environ["KAATS_TERUG"])   # 9 Nov.: KAATS_TERUG
if os.environ.get("KRUIP_IMPULS"): ctypes.c_int.in_dll(LIB, "kruip_impuls").value = int(os.environ["KRUIP_IMPULS"])
if os.environ.get("HOEK_LSQ_KOL"): ctypes.c_int.in_dll(LIB, "hoek_lsq").value = int(os.environ["HOEK_LSQ_KOL"])   # 6 Nov.: HOEK_LSQ_KOL
if os.environ.get("HOEK_HEL"): ctypes.c_int.in_dll(LIB, "hoek_hel").value = int(os.environ["HOEK_HEL"])   # 6 Nov.: HOEK_HEL
if os.environ.get("TILT_PLUS"): ctypes.c_int.in_dll(LIB, "tilt_plus").value = int(os.environ["TILT_PLUS"])   # 5 Nov.: TILT_PLUS
if os.environ.get("ART_KR"): ctypes.c_int.in_dll(LIB, "art_kr").value = int(os.environ["ART_KR"])   # 4 Nov.: ART_KR, ART_EJ
if os.environ.get("ART_EJ"): ctypes.c_int.in_dll(LIB, "art_ej").value = int(os.environ["ART_EJ"])
if os.environ.get("RAND_E"): ctypes.c_double.in_dll(LIB, "rand_e").value = float(os.environ["RAND_E"])   # 18 Oct.
if os.environ.get("IMPULS_RAAK"): ctypes.c_int.in_dll(LIB, "impuls_raak").value = int(os.environ["IMPULS_RAAK"])
LIB.wand_uit.restype = ctypes.c_long
if os.environ.get("EJ_LMIN"): ctypes.c_double.in_dll(LIB, "ej_lmin").value = float(os.environ["EJ_LMIN"])   # 18 Oct.: in m
if os.environ.get("EJ_MIN_N"): ctypes.c_double.in_dll(LIB, "ej_min_n").value = float(os.environ["EJ_MIN_N"])   # 18 Oct.
if os.environ.get("EJ_ONDER_KRUIP"): ctypes.c_int.in_dll(LIB, "ej_onder_kruip").value = int(os.environ["EJ_ONDER_KRUIP"])
if os.environ.get("EJ_LOGN"): ctypes.c_double.in_dll(LIB, "ej_logn").value = float(os.environ["EJ_LOGN"])
LIB.ej_onder_uit.restype = ctypes.c_long
LIB.kruip_m_uit.restype = ctypes.c_double; LIB.mk_uit.restype = ctypes.c_double; LIB.mk_uit.argtypes = [ctypes.c_int]   # 29 Oct.
KORRELMASSA = int(os.environ.get("KORRELMASSA", "0") or 0)   # 29 Oct.: KORRELMASSA (a filled cell = one grain, mass m_k)
if KORRELMASSA: ctypes.c_int.in_dll(LIB, "impuls_cel").value = 0
NZ = 200


MOD_AB = int(os.environ.get("MOD_AB", "0") or 0)   # 9 Nov.: MOD_AB
MOD_Z0 = float(os.environ["MOD_Z0"]) * 1e-3 if os.environ.get("MOD_Z0") else None
_FA = (1.0702, 0.093069, 0.10838, 0.024835, 0.041603, 0.0010625); _FB = (0.036989, 0.15765, 0.11518, 0.0020249, 0.0028725, 0.00053483)
def ab_fit(eta0):
    """9 Nov. (MOD_AB=1): A and B as a function of eta0 = k z0 (Fourriere, Claudin & Andreotti 2010, eq. 4.6)."""
    R = np.log(2 * np.pi / np.asarray(eta0, float)); a, b = _FA, _FB
    A = 2 + (a[0] + a[1] * R + a[2] * R ** 2 + a[3] * R ** 3) / (1 + a[4] * R ** 2 + a[5] * R ** 4)
    B = (b[0] + b[1] * R + b[2] * R ** 2 + b[3] * R ** 3) / (1 + b[4] * R ** 2 + b[5] * R ** 4)
    return A, B
MOD_UB = os.environ.get("MOD_UB", "")   # 10 Nov.: MOD_UB (u* at the bed, m/s)
MOD_KROM = os.environ.get("MOD_KROM", "")   # 10 Nov.: MOD_KROM (beta_c of the curvature correction; empty = without)
_VD = None
def vd_tabel():
    """10 Nov. (MOD_AB=2): table A(k), B(k) of the van Driest model at u* at the bed MOD_UB."""
    global _VD
    if _VD is None:
        kr_ = f"_b{round(10 * float(MOD_KROM)):03d}" if MOD_KROM else ""   # 10 Nov.: MOD_KROM
        f = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"tabel_vandriest_u{round(100 * float(MOD_UB)):03d}{kr_}.npz")
        d = np.load(f); _VD = (d["lnk"], d["A"], d["B"])
    return _VD
def storing_kern(k, A, B, z0):
    """9 Nov.: A |k| + i B k per Fourier component. Without MOD_AB exactly as before; with MOD_AB A(k) (A / 3), B(k) (B / 1)."""
    if not MOD_AB: return A * np.abs(k) + 1j * B * k
    if MOD_AB == 2:   # 10 Nov.: MOD_AB=2
        lnk, Ta, Tb = vd_tabel(); kk = np.abs(k); lk = np.log(np.where(kk > 0, kk, 1.0))
        Av, Bv = np.interp(lk, lnk, Ta), np.interp(lk, lnk, Tb)
        return np.where(kk > 0, (A / 3.0) * Av * kk + 1j * (B / 1.0) * Bv * k, 0.0)
    zz = MOD_Z0 if MOD_Z0 is not None else z0
    kk = np.abs(k); e0 = np.where(kk > 0, kk * zz, 1.0); Af, Bf = ab_fit(e0)
    return np.where(kk > 0, (A / 3.0) * Af * kk + 1j * (B / 1.0) * Bf * k, 0.0)

class WindKorrel:
    def __init__(self, Lx_mm=1070.0, d_mm=0.2, ustar=0.35, diepte=80, HM=600, seed=1, ruis=1, f0=None, ruis_stap=1, ruis_blok=1, **kw):
        p = dict(d=d_mm * 1e-3, rho_p=2650.0, rho_a=1.2, nu=1.5e-5, g=9.81, ustar=ustar, kappa=0.4,
                 z0=d_mm * 1e-3 / 30, ust_fl=0.26, aero_rate=1000.0,
                 B=0.96, gam=1.0, ereb_m=0.45, ereb_s=0.22, threb=np.deg2rad(40), a=0.02, alpha_ej=0.15,
                 thej=np.deg2rad(50), Dsp=d_mm * 1e-3, eps_d=2.0, dx=0.4e-3, dz=0.2e-3,
                 tanr=np.tan(np.deg2rad(32)), Wl=5, toren=3, dt=5e-5, t_upd=0.01, relax=0.1, bevroren=0)
        if os.environ.get("REB_B"): p["B"] = float(os.environ["REB_B"])          # 17 Oct.: rebound probability B (1 - exp(-gam v))
        if os.environ.get("REB_GAM"): p["gam"] = float(os.environ["REB_GAM"])
        p.update(kw)
        self.p = Par(**p)
        self.N = int(round(Lx_mm * 1e-3 / self.p.dx))
        self.HM = int(HM)
        rng = np.random.default_rng(seed)
        LIB.zaai(int(seed))
        self.h = np.full(self.N, diepte, np.int32)
        if ruis:
            nb_ = max(1, int(ruis_blok)); nbl_ = -(-self.N // nb_)   # 28 Oct.: initial noise per block of nb_ columns (RUIS_BLOK_MM); nb_ = 1: as before
            self.h += np.repeat(rng.integers(0, ruis + 1, nbl_).astype(np.int32), nb_)[:self.N] * np.int32(ruis_stap)   # 26 Oct.: ruis_stap layers per step
        self.col = np.zeros(self.N * self.HM, np.uint8)
        if f0 is not None:                      # grain classes in the bed (two sizes)
            f0 = np.asarray(f0, float)
            for i in range(self.N):
                self.col[i * self.HM: i * self.HM + self.h[i]] = rng.choice(len(f0), size=self.h[i], p=f0)
        self.Mcap = 200000
        self.px = np.zeros(self.Mcap); self.pz = np.zeros(self.Mcap)
        self.pvx = np.zeros(self.Mcap); self.pvz = np.zeros(self.Mcap)
        self.pk = np.zeros(self.Mcap, np.uint8)
        self.M = ctypes.c_int(0)
        self.Uk = np.full(NZ, -1.0)
        self.tel = np.zeros(4, np.int64)
        self.t = 0.0
        self.korrels0 = int(self.h.sum())

    def modulatie(self, A=3.0, B=1.0, glad=5, sigma_mm=2.0):
        """Flow modulation (Jackson & Hunt 1975; Charru, Andreotti & Claudin 2013):
        shear stress tau = tau0 (1 + d), d^ = (A |k| + i B k) h^ (wind in +x); the
        shear velocity and thus the whole wind profile scale with sqrt(1 + d) (simplification:
        uniform over height). Bed first smoothed over 'glad' columns."""
        hz = self.h.astype(float) * self.p.dz
        if glad > 1:
            k_ = np.ones(glad) / glad
            hz = np.convolve(np.concatenate([hz[-glad:], hz, hz[:glad]]), k_, "same")[glad:-glad]
        k = 2 * np.pi * np.fft.rfftfreq(self.N, d=self.p.dx)
        filt = np.exp(-0.5 * (k * sigma_mm * 1e-3) ** 2)
        d = np.fft.irfft(storing_kern(k, A, B, self.p.z0) * filt * np.fft.rfft(hz - hz.mean()), n=self.N)   # 9 Nov.: MOD_AB
        self.mod = np.ascontiguousarray(np.sqrt(np.maximum(1.0 + d, 0.0)))
        return self.mod

    STROOM_Z_MM = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
    def stroomlijn(self, glad=5, sigma_mm=2.0):
        """13 Nov. (STROOM): slope d eta / dx of the streamlines at the heights STROOM_Z_MM above the bed; eta^ = h^ exp(-|k| z), with the same
        smoothing and filter as modulatie(). At the bed = slope of the smoothed bed. Passed to C (zet_stroom)."""
        hz = self.h.astype(float) * self.p.dz
        if glad > 1:
            k_ = np.ones(glad) / glad
            hz = np.convolve(np.concatenate([hz[-glad:], hz, hz[:glad]]), k_, "same")[glad:-glad]
        k = 2 * np.pi * np.fft.rfftfreq(self.N, d=self.p.dx); filt = np.exp(-0.5 * (k * sigma_mm * 1e-3) ** 2); H_ = np.fft.rfft(hz - hz.mean()) * filt
        z = np.array(self.STROOM_Z_MM) * 1e-3
        self.stroom_s = np.ascontiguousarray(np.array([np.fft.irfft(1j * k * np.exp(-k * zz) * H_, n=self.N) for zz in z]))
        self.stroom_z = np.ascontiguousarray(z)
        LIB.zet_stroom(self.stroom_s.ctypes.data_as(ctypes.c_void_p), self.stroom_z.ctypes.data_as(ctypes.c_void_p), len(z))
        return self.stroom_s

    def zet_referentie(self, L_mm=100.0):
        """12 Oct.: reference level for the wind profile: bed smoothed over L_mm (running mean, periodic)."""
        hz = self.h.astype(float) * self.p.dz
        n = max(1, int(round(L_mm * 1e-3 / self.p.dx))); k_ = np.ones(n) / n
        ext = np.concatenate([hz[-n:], hz, hz[:n]])
        self.href = np.ascontiguousarray(np.convolve(ext, k_, "same")[n:-n])
        LIB.zet_href(self.href.ctypes.data_as(ctypes.c_void_p))
        return self.href

    def schaduwkaart(self, tan_g=np.tan(np.deg2rad(12.0)), glad=5):
        """1 where the horizon slope (smoothed bed) is greater than tan_g: no impacts."""
        hz = self.h.astype(float) * self.p.dz
        if glad > 1:
            k_ = np.ones(glad) / glad
            hz = np.convolve(np.concatenate([hz[-glad:], hz, hz[:glad]]), k_, "same")[glad:-glad]
        span = hz.max() - hz.min()
        J = int(min(self.N - 1, np.ceil(span / (self.p.dx * tan_g)) + 2))
        Th = np.full(self.N, -np.inf)
        for j in range(1, J + 1):
            Th = np.maximum(Th, (np.roll(hz, j) - hz) / (j * self.p.dx))
        self.sch = np.ascontiguousarray((Th > tan_g).astype(np.uint8))
        LIB.zet_schaduw(self.sch.ctypes.data_as(ctypes.c_void_p))
        return self.sch

    def draai(self, T, hins=None, hej=None):
        mod = getattr(self, "mod", None)
        rc = LIB.draai(self.N, self.HM, self.col, self.h, ctypes.byref(self.p), T,
                       self.px, self.pz, self.pvx, self.pvz, self.pk, ctypes.byref(self.M), self.Mcap,
                       self.Uk, self.tel,
                       None if hins is None else hins.ctypes.data_as(ctypes.c_void_p),
                       None if hej is None else hej.ctypes.data_as(ctypes.c_void_p),
                       None if mod is None else mod.ctypes.data_as(ctypes.c_void_p))
        if rc != 0:
            raise RuntimeError("column full")
        self.t += T

    @property
    def x(self):
        return np.arange(self.N) * self.p.dx * 1e3                   # mm

    def hoogte(self, glad=5):
        z = self.h.astype(float) * self.p.dz * 1e3
        if glad > 1:
            k = np.ones(glad) / glad
            z = np.convolve(np.concatenate([z[-glad:], z, z[:glad]]), k, "same")[glad:-glad]
        return z - z.mean()

    def flux(self):
        """Horizontal mass flux q (kg/m/s) of the grains in the air, per metre width."""
        p = self.p
        m = p.rho_p * np.pi / 6 * p.d ** 3
        W = (np.pi / 6 * p.d ** 3) / (p.dx * p.dz * 0.6)
        Lx = self.N * p.dx
        return float(np.sum(self.pvx[:self.M.value]) * m / (Lx * W))

    def flux_k(self):
        """29 Oct. (KORRELMASSA): saltation flux with the real mass per class, sum m_k v_x (kg/m/s per metre width)."""
        p = self.p; M = self.M.value
        mk = np.array([LIB.mk_uit(k) for k in range(4)])
        W = (np.pi / 6 * p.d ** 3) / (p.dx * p.dz * 0.6)
        return float(np.sum(self.pvx[:M] * mk[self.pk[:M]]) / (self.N * p.dx * W))

    def profiel(self):
        z = np.zeros(NZ); LIB.profiel(z)
        return z, self.Uk.copy()

    def telling(self):
        uit = np.zeros(4, np.int64)
        for i in range(self.N):
            uit += np.bincount(self.col[i * self.HM: i * self.HM + self.h[i]], minlength=4)
        uit += np.bincount(self.pk[:self.M.value], minlength=4)
        return uit

    def oppervlak(self, lagen=2, glad=5, k=1):
        """Fraction of class k in the top 'lagen' grains per column, smoothed over 'glad' columns."""
        f = np.zeros(self.N)
        for j in range(lagen):
            idx = np.arange(self.N) * self.HM + np.maximum(self.h - 1 - j, 0)
            f += (self.col[idx] == k)
        f /= lagen
        if glad > 1:
            kk = np.ones(glad) / glad
            f = np.convolve(np.concatenate([f[-glad:], f, f[:glad]]), kk, "same")[glad:-glad]
        return f

    def massa_ok(self):
        return int(self.h.sum()) + self.M.value == self.korrels0
