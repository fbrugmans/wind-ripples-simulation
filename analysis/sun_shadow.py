"""Shadow test ('The lee side'; fixed in advance on 14 Nov. 2026): does the eye see the break of slope at the boundary of light and shadow?

Light from the upwind side (sun and wind from the same side), solar elevation alpha in the plane of the profile. SR0, seeds 271-274, 210-240 min.
The trough is placed where the shadow behind the crest ends. Criterion: S <= 35 mm in >= 3 of 4 seeds at one solar elevation.
Usage (from the folder with the runs):  python3 sun_shadow.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import measure as M, ripple_measure as RM
SEEDS = (271, 272, 273, 274); SERIES = "SR0"; T0, T1 = 210 * 60, 240 * 60; ALPHA = (5, 10, 15, 20, 30); SMIN = 5.0


def shadow(h, dx, ta):
    """True where the bed lies in the shadow; light from -x at slope ta (periodic: the profile is traversed twice)."""
    N = len(h); s = np.zeros(N, bool); top = -np.inf   # height of the highest light ray above the current point
    for _ in range(2):
        for i in range(N):
            top -= ta * dx
            if h[i] < top: s[i] = True
            else: top = h[i]
    return s


def measure_shadow(z, ta):
    d = np.load(f"c23_z{z}_{SERIES}.npz"); Hs, T, dx = d["h"], d["t"], float(d["dx"]); dz = float(d["dz"]) if "dz" in d.files else 0.2
    N = Hs.shape[1]; x = np.arange(N) * dx; out = []; ntot = nnone = nshort = 0
    for k in range(len(T)):
        if not T0 < T[k] <= T1: continue
        h = M.smooth(Hs[k] * dz); h -= h.mean(); rip, _, _ = RM.ripples(x, h, prom_frac=0.0, prom_min=3.0)
        if len(rip) < 2: continue
        sh = shadow(h, dx, ta); end = []
        for r in rip:   # end of the shadow behind each crest (all ripples, also small ones: they set the stoss side of the next)
            c, c2 = r["c"], r["c2"]; n = int((c2 - c) % N) or N; j = 1
            while j < n and sh[(c + j) % N]: j += 1
            end.append(j if j < n else None)
        for q, r in enumerate(rip):
            if r["H"] < 2.5: continue
            ntot += 1; e = end[q]; ev = end[q - 1]; rv = rip[q - 1]
            if e is None: nnone += 1; continue
            if e * dx < SMIN: nshort += 1; continue
            if ev is None or rv["c2"] != r["c"]: continue
            Sz = e * dx; Iz = ((r["c"] - rv["c"]) % N - ev) * dx
            if Iz <= 0: continue
            Hz = h[r["c"]] - h[(r["c"] + e) % N]
            out.append((Iz + Sz, Hz, r["H"], Iz, Sz))
    return np.array(out).reshape(-1, 5), ntot, nnone, nshort


print("Field: L 107, H 5, I 80, S 27 -> SI 2.96, stoss 0.063, lee 0.185. Without shadow (measuring rule, to the trough): L 127, H 4.9, I 71, S 56, SI 1.27.")
print("Reading: S_z <= 35 in >= 3/4 seeds at one solar elevation = largely explains; >= 45 at all solar elevations = does not.\n")
verdicts = []
for a in ALPHA:
    ta = np.tan(np.radians(a)); Sz = []; everything = []
    print(f"solar elevation {a} deg:")
    for z in SEEDS:
        A, nt, nn, ns = measure_shadow(z, ta); everything.append(A); L, Hz, H, I, S = A.mean(0)
        Sz.append(S)
        print(f"  seed {z}: visible {len(A) / nt:4.0%} (no break {nn / nt:4.0%}, shadow < 5 mm {ns / nt:4.0%})  L {L:4.0f}  H_z {Hz:4.1f} (to trough {H:4.1f})"
              f"  I {I:4.0f}  S {S:4.0f}  SI {I / S:4.2f}  stoss {Hz / I:.3f}  lee {Hz / S:.3f}")
    A = np.vstack(everything); L, Hz, H, I, S = A.mean(0); Sz = np.array(Sz)
    print(f"  together:  n {len(A):5d}  L {L:4.0f}  H_z {Hz:4.1f} (to trough {H:4.1f})  I {I:4.0f}  S {S:4.0f}  SI {I / S:4.2f}  stoss {Hz / I:.3f}  lee {Hz / S:.3f}")
    verdicts.append(((Sz <= 35).sum() >= 3, (Sz >= 45).sum() >= 3))
g = [a for a, o in zip(ALPHA, verdicts) if o[0]]
print("\nReading:", f"largely explains at {g} deg" if g else ("does not" if all(o[1] for o in verdicts) else "partly"))
