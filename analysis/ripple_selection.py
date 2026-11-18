"""Selection test ('The choice of ripples'; fixed in advance on 14 Nov. 2026): can choosing clearly recognisable ripples explain the short lee side in the field?

SR0, seeds 271-274, 210-240 min; measuring rule of the article (prominence 3 mm, H >= 2.5); per seed the upper third by K1 H, K2 break of slope
at the foot of the lee side, K3 steepest lee face, K4 H/S (upper bound, circular). Criterion: mean S <= 35 mm in >= 3 of 4 seeds.
Usage (from the folder with the runs):  python3 ripple_selection.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import measure as M, ripple_measure as RM
SEEDS = (271, 272, 273, 274); SERIES = "SR0"; T0, T1 = 210 * 60, 240 * 60; NK = 5.0   # mm for the break of slope


def ripples(z):
    d = np.load(f"c23_z{z}_{SERIES}.npz"); H, T, dx = d["h"], d["t"], float(d["dx"]); dz = float(d["dz"]) if "dz" in d.files else 0.2
    N = H.shape[1]; x = np.arange(N) * dx; n = int(round(NK / dx)); out = []
    for k in range(len(T)):
        if not T0 < T[k] <= T1: continue
        h = M.smooth(H[k] * dz); h -= h.mean(); rip, _, _ = RM.ripples(x, h, prom_frac=0.0, prom_min=3.0)
        for r in rip:
            if r["H"] < 2.5: continue
            t = r["t"]; after = (h[(t + n) % N] - h[t]) / NK; before = (h[t] - h[(t - n) % N]) / NK
            out.append((r["L"], r["H"], r["I"], r["S"], after - before, r["max_lee"]))
    return np.array(out)


def row(name, A):
    L, H, I, S = A[:, 0].mean(), A[:, 1].mean(), A[:, 2].mean(), A[:, 3].mean()
    return f"{name:30s} n {len(A):4d}  L {L:5.0f}  H {H:4.1f}  I {I:4.0f}  S {S:4.0f}  SI {I / S:4.2f}  stoss {H / I:.3f}  lee {H / S:.3f}", S


SELECTION = [("A  all ripples", None), ("K1 highest H", 1), ("K2 sharpest break foot lee", 4), ("K3 steepest lee face", 5), ("K4 largest H/S (upper bound)", "HS")]
print("Field: L 107, H 5, I 80, S 27 -> SI 2.96, stoss 0.063, lee 0.185. Reading: S <= 35 in >= 3/4 seeds = largely explains; >= 45 = does not.\n")
everything = {nm: [] for nm, _ in SELECTION}; Sper = {nm: [] for nm, _ in SELECTION}
for z in SEEDS:
    R = ripples(z); print(f"seed {z}:")
    for nm, c in SELECTION:
        if c is None: A = R
        else:
            key = R[:, 1] / R[:, 3] if c == "HS" else R[:, c]
            A = R[key >= np.quantile(key, 2 / 3)]
        s, S = row("  " + nm, A); print(s); everything[nm].append(A); Sper[nm].append(S)
print("\nAll seeds together:")
for nm, c in SELECTION:
    s, S = row("  " + nm, np.vstack(everything[nm])); Sz = np.array(Sper[nm])
    verdict = "" if c in (None, "HS") else ("  -> largely" if (Sz <= 35).sum() >= 3 else "  -> no" if (Sz >= 45).sum() >= 3 else "  -> partly")
    print(s + f"  (S per seed {' / '.join(f'{v:.0f}' for v in Sz)})" + verdict)
