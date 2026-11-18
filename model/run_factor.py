"""Grain and bed rules of the model variant VAR (first argument of run_fetch.py's caller; the article uses C23).

Sets the parameters of the C library (wind_korrel.c) for the variant: grain classes, creep model, ejection rules,
angle of repose and the cell size, and reads the switches given as environment variables. The variants C0-C23 are
successive versions from the development of the model (October 2026); each later variant changes one rule of an
earlier one. C23: three grain classes 176/334/488 um (90.2/8.8/1.0 %); splash after Beladjine et al. (2007) with an
ejection threshold of 20 sqrt(gd); ejection angle around 50 degrees; only the horizontal part of ejections is paid
from the horizontal momentum loss of the impacting grain; creep with p_cr pushed grains per impact (PCR).
"""
import ctypes
import sys
import json
import time

import numpy as np

import wind_korrel as WK
import ripple_measure as MM

L = WK.LIB
VAR = sys.argv[1]
DRIE = VAR in ("C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C16", "C17", "C18", "C19", "C20", "C21", "C22", "C23")
for n, v in (("hoekmodel", 1), ("kruipmodel", 2 if VAR == "C1" else (3 if VAR in ("C4", "C5", "C6", "C7", "C8", "C9", "C10", "C16", "C17", "C18", "C19", "C20", "C21", "C22", "C23") else 1)), ("nk", 3 if DRIE else 1), ("kruip_wind", 1)):
    ctypes.c_int.in_dll(L, n).value = v
ctypes.c_double.in_dll(L, "C_vlak").value = 0.453
import os
LAAG_ALFA = float(os.environ.get("LAAG_ALFA", "0")); ctypes.c_double.in_dll(L, "alfa_ref").value = LAAG_ALFA   # 12 Oct.
ctypes.c_double.in_dll(L, "p_cr").value = float(os.environ.get("PCR", "5.0")) if VAR in ("C8", "C9", "C10", "C16", "C17", "C18", "C19", "C20", "C21", "C22", "C23") else 5.0
ctypes.c_double.in_dll(L, "l_cr").value = 16e-3 if VAR == "C1" else 21e-3
DK = (176e-6, 334e-6, 488e-6)
dk = (ctypes.c_double * 4).in_dll(L, "dk"); us = (ctypes.c_double * 4).in_dll(L, "ustk")
shao = lambda d: np.sqrt(0.0123 * (2650 * 9.81 * d / 1.2 + 3e-4 / (1.2 * d)))
if DRIE:
    for i, d in enumerate(DK):
        dk[i] = d; us[i] = shao(d)
tr = (ctypes.c_double * 4).in_dll(L, "tanrk")
ctypes.c_double.in_dll(L, "l_cr_grof").value = 16e-3
ctypes.c_double.in_dll(L, "l_schaduw").value = 3.5e-3 if VAR in ("C6", "C7", "C8", "C9", "C10", "C16", "C17", "C18", "C19", "C20", "C21", "C22", "C23") else -1
ctypes.c_double.in_dll(L, "ez_max").value = 1.0 if VAR in ("C7", "C8") else 1e9
if VAR in ("C9", "C10", "C16", "C17", "C18", "C19", "C20", "C21", "C22", "C23"):
    ctypes.c_int.in_dll(L, "splashmodel").value = 2; ctypes.c_double.in_dll(L, "xi_c").value = 20.0
ctypes.c_int.in_dll(L, "kruip_stop").value = {"C10": 1, "C16": 1, "C17": 3, "C18": 1}.get(VAR, 0)
ctypes.c_double.in_dll(L, "kruip_stop_tan").value = np.tan(np.radians(20)) if VAR == "C16" else -1
ctypes.c_int.in_dll(L, "kruip_inslag").value = {"C18": 1, "C19": 2, "C20": 2, "C21": 2, "C22": 2, "C23": 2}.get(VAR, 0)   # 10 Oct.: C18 = C10 + impact rule for creep
DX = float(os.environ.get("CEL_DX", "0.4")); DZ = float(os.environ.get("CEL_DZ", "0.2"))   # 26 Oct.: cell size in mm
if VAR in ("C18", "C19", "C20", "C21", "C22", "C23"): ctypes.c_double.in_dll(L, "tau_I").value = 10.0; ctypes.c_int.in_dll(L, "glad_I").value = 2 * int(round(1.2 / DX)) + 1   # 10 Oct.: less noise (26 Oct.: 7 columns = own column + 2 x 1.2 mm)
ctypes.c_int.in_dll(L, "kb_lagen").value = int(round(float(os.environ.get("KB_MM", "0.4")) / DZ)); ctypes.c_int.in_dll(L, "hoek_hw").value = int(round(0.8 / DX))   # 26 Oct.: 0.4 mm, 2 x 0.8 mm; 27 Oct.: KB_MM (creep blocking step)
if os.environ.get("KRUIP_INSLAG"): ctypes.c_int.in_dll(L, "kruip_inslag").value = int(os.environ["KRUIP_INSLAG"])   # 18 Oct.: 1 = impact rule for all classes
VLAGEN = int(os.environ.get("VLAGEN", "1"))   # 18 Oct.: 0 = constant wind u* 0.35 m/s
ctypes.c_int.in_dll(L, "kruip_impuls").value = int(os.environ["KRUIP_IMPULS"]) if os.environ.get("KRUIP_IMPULS") else {"C20": 1, "C21": 2}.get(VAR, 0)   # 6 Nov.: KRUIP_IMPULS   # 11 Oct.: C20 = C19 + creep pushes paid from the momentum budget
if VAR in ("C22", "C23"): ctypes.c_double.in_dll(L, "ej_hoek").value = np.radians(50.0)   # 11 Oct.: C22 = C19 + ejection angle 50 degrees (sand)
ctypes.c_int.in_dll(L, "ej_hor").value = 1 if VAR == "C23" else 0   # 12 Oct.: C23 = C22 + horizontal momentum budget for ejections
if VAR in ("C3", "C5"):
    tr[0] = np.tan(np.radians(32)); tr[1] = tr[2] = np.tan(np.radians(34))
F0 = [0.902, 0.088, 0.010] if DRIE else None
if os.environ.get("EEN_KORREL"): F0 = [1.0, 0.0, 0.0]   # 19 Oct.: test P1, only the fine class (176 um)
SPLASH_FAC = float(os.environ.get("SPLASH_FAC", "1"))   # 19 Oct.: test P2, factor on the number of ejections per impact (s_ej)
if SPLASH_FAC != 1: ctypes.c_double.in_dll(L, "s_ej").value *= SPLASH_FAC
# 20 Oct.: SUPPORT AND SORTING (STEUN_A > 0): O = clip(1 - s_down / STEUN_SREF, 0, 1), s_down the slope over STEUN_L mm downwind on the
# bed smoothed over STEUN_GLAD mm; ejections of fine bed grains x (1 + STEUN_A O cos theta) (see wind_korrel.c). Not used in the article.
STEUN_A = float(os.environ.get("STEUN_A", "0")); STEUN_SREF = float(os.environ.get("STEUN_SREF", "0.05"))
STEUN_L = float(os.environ.get("STEUN_L", "2.0")); STEUN_GLAD = float(os.environ.get("STEUN_GLAD", "10.0"))
if STEUN_A > 0: ctypes.c_double.in_dll(L, "steun_a").value = STEUN_A
# 20 Oct.: LANDING RULE fine on coarse (LAND_A rebound, LAND_C roll off, LAND_D into the space between; see wind_korrel.c). Not used in the article.
for n_ in ("LAND_A", "LAND_C", "LAND_D"):
    if os.environ.get(n_): ctypes.c_double.in_dll(L, n_.lower()).value = float(os.environ[n_])
SORT_A = float(os.environ.get("SORT_A", "0"))   # 21 Oct.: pure sorting rule (see wind_korrel.c); uses the same measure O
if SORT_A > 0: ctypes.c_double.in_dll(L, "sort_a").value = SORT_A
if os.environ.get("GAT_ALLE"): ctypes.c_double.in_dll(L, "gat_alle").value = float(os.environ["GAT_ALLE"])   # 21 Oct.: (d) also for creep and avalanches
# 21 Oct.: SUPPORT MEASURE 2 (default): support is missing only in the crest area where the crest passes into the lee side: the column lies
# at most STEUN_DL layers below the highest column within STEUN_D mm upwind, AND the column STEUN_D mm downwind lies at least
# STEUN_DL layers lower (bed smoothed over STEUN_G columns). Then O = 1, otherwise 0 (stoss side, trough, flat bed, small pits: O = 0).
# STEUN_MAAT=1: the old measure of 20 Oct. (O = clip(1 - s_down / STEUN_SREF, 0, 1); also gave a flat bed O = 1).
STEUN_MAAT = int(os.environ.get("STEUN_MAAT", "2")); STEUN_D = float(os.environ.get("STEUN_D", "4.0"))
STEUN_DL = float(os.environ.get("STEUN_DL", "2")); STEUN_G = int(os.environ.get("STEUN_G", "3"))
def steun_maat2(hl):
    """hl: column heights in layers; returns O (0/1) per column"""
    G = STEUN_G; z = hl.astype(float)
    if G > 1: z = np.convolve(np.concatenate([z[-G:], z, z[:G]]), np.ones(G) / G, "same")[G:-G]
    D = max(1, int(round(STEUN_D / DX)))
    neer = z - np.roll(z, -D) >= STEUN_DL
    boven = np.max(np.stack([np.roll(z, j) for j in range(0, D + 1)]), 0)
    return (neer & (z >= boven - STEUN_DL)).astype(float)
def steun_zet(w):
    if STEUN_A <= 0 and SORT_A <= 0: return None
    if STEUN_MAAT == 2:
        w._steun = np.ascontiguousarray(steun_maat2(w.h)); L.zet_steun(w._steun.ctypes.data_as(ctypes.c_void_p)); return w._steun
    G = max(1, int(round(STEUN_GLAD / DX))); k_ = np.ones(G) / G; z = w.h.astype(float) * w.p.dz
    z = np.convolve(np.concatenate([z[-G:], z, z[:G]]), k_, "same")[G:-G]
    n = max(1, int(round(STEUN_L / DX))); sd = (np.roll(z, -n) - z) / (n * w.p.dx)
    w._steun = np.ascontiguousarray(np.clip(1.0 - sd / STEUN_SREF, 0.0, 1.0)); L.zet_steun(w._steun.ctypes.data_as(ctypes.c_void_p))
    return w._steun
GL = int(round(2.0 / DX))   # 26 Oct.: smoothing and window 2.0 mm (5 columns at dx 0.4)
TOREN_L = int(round(1.5 * DX / DZ)) if os.environ.get("CEL_TOREN", "tan") == "tan" else int(round(0.6 / DZ))   # 26 Oct.: toppling rule (3 layers)
DIEPTE_L, HM_L, RUIS_STAP = int(round(16.0 / DZ)), int(round(120.0 / DZ)), int(round(float(os.environ.get("CEL_RUIS", "0.2")) / DZ))   # 26 Oct.: 16 mm, 120 mm, noise 0 or 0.2 mm (CEL_RUIS 0: flat)
CELKW = dict(diepte=DIEPTE_L, HM=HM_L, ruis_stap=RUIS_STAP, ruis_blok=max(1, int(round(float(os.environ.get("RUIS_BLOK_MM", "0") or 0) / DX))), toren=TOREN_L, **({"dz": DZ * 1e-3} if os.environ.get("CEL_DZ") else {}))


def piek(h, dx, Lmin=8.0, Lmax=400.0):
    F = np.abs(np.fft.rfft(h - h.mean())) ** 2
    Lg = len(h) * dx / np.maximum(np.arange(len(F)), 1)
    m = (Lg >= Lmin) & (Lg <= Lmax)
    return float(Lg[m][np.argmax(F[m])])


