"""Imposed wind shadow (section 'An imposed wind shadow'; fixed in advance on 13 Nov. 2026).

Basis: rough log theory (MOD_AB=1, MOD_Z0=0.1); imposed wind shadow OMH_GR 8 / 5 / 3 degrees (SQ8/SQ5/SQ3) and without (SQ0), 4 hours,
seeds 271-274; also streamlines that follow the bed (STROOM=1) without / with OMH_GR 5 (SR0/SR5), paired per seed with SQ0/SQ5 (descriptive).
Measuring rule as in measure.py, per seed mean L, H, I, S; RI, SI, stoss and lee from these.
Question D (diagnostic): does a shadow angle at 210-240 min reach SI >= 2.0, L 86-128 mm and H 3.5-6.5 mm together in >= 3 of 4 seeds?
Usage (from the folder with the runs):  python3 wind_shadow.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import measure as M
FIELD = dict(L=(107, 21), H=(5, 1.5), I=(80, 15), S=(27, 8), RI=(21.4, 5), SI=(2.96, 1), stoss=(0.063, 0.01), lee=(0.185, 0.06))
VARIANTS = (("SQ0", "no wind shadow"), ("SQ8", "wind shadow 8 deg"), ("SQ5", "wind shadow 5 deg"), ("SQ3", "wind shadow 3 deg"),
            ("SR0", "streamlines follow the bed, no wind shadow"), ("SR5", "streamlines follow the bed, wind shadow 5 deg"))


def per_seed(A):
    out = {}
    for z in np.unique(A[:, 4]):
        L, H, I, S = A[A[:, 4] == z, :4].T
        out[int(z)] = dict(L=L.mean(), H=H.mean(), I=I.mean(), S=S.mean(), RI=L.mean() / H.mean(), SI=I.mean() / S.mean(), stoss=H.mean() / I.mean(), lee=H.mean() / S.mean())
    return out


print("Field: L 107, H 5, I 80, S 27 -> SI 2.96, RI 21.4, stoss 0.063, lee 0.185.")
for series, name in VARIANTS:
    print(f"\n=== {series}: {name} ===")
    for a, b in ((25, 30), (55, 60), (110, 120), (210, 240)): print(M.line(f"{a}-{b} min", *M.measure(".", series, a * 60, b * 60)))
    A, nrec, Lb = M.measure(".", series, 12600, 14400)
    if nrec == 0: print("  runs missing"); continue
    P = per_seed(A); d = sum(P[z]["SI"] >= 2.0 and 86 <= P[z]["L"] <= 128 and 3.5 <= P[z]["H"] <= 6.5 for z in P)
    for z in P: print(f"  seed {z}: " + "  ".join(f"{m} {P[z][m]:.3g}" for m in FIELD) + f"  -> {sum(abs(P[z][m] - g) <= s for m, (g, s) in FIELD.items())} of 8 within the field")
    print(f"  D: {d} of {len(P)} seeds SI >= 2.0 with L 86-128 and H 3.5-6.5 -> {'MET' if d >= 3 else 'NOT MET'}")
    if series in ("SR0", "SR5"):
        Q = per_seed(M.measure(".", "SQ" + series[2], 12600, 14400)[0])
        print("  paired with SQ" + series[2] + " (streamlines - without): " + "; ".join(f"{m} " + " / ".join(f"{P[z][m] - Q[z][m]:+.3g}" for z in P if z in Q) for m in ("L", "H", "SI", "stoss", "lee")))
