"""Does the asymmetry grow with the crest spacing? Ripple measures over time in the 16-hour series SL16 (descriptive).

Measuring rule as in measure.py (prominence 3 mm, H >= 2.5 mm; variant H >= 1.5 mm for young ripples).
Usage (from the folder with the runs):  python3 long_run.py <folder with the runs> <series, e.g. SR0 or SL16> [folder with SR0, to check the first 4 hours]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import measure as M
FOLDER, SERIES = sys.argv[1], sys.argv[2]
SEEDS = (271, 272, 273, 274)
tmax = min(float(np.load(os.path.join(FOLDER, f"c23_z{z}_{SERIES}.npz"))["t"][-1]) for z in SEEDS) / 60
WINDOWS = [(2, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 45), (45, 60)] + [(a, a + 30) for a in range(60, 240, 30)] + [(a, a + 60) for a in range(240, int(tmax), 60)]
WINDOWS = [v for v in WINDOWS if v[1] <= tmax + 1e-6]


def line(A):
    if len(A) < 8: return f"n {len(A):5d}  (too few ripples)"
    L, H, I, S, Z = A.T; si = I / S
    pz = " / ".join(f"{I[Z == z].mean() / S[Z == z].mean():.2f}" if (Z == z).sum() > 2 else "  - " for z in SEEDS)
    lz = " / ".join(f"{L[Z == z].mean():.0f}" if (Z == z).sum() > 2 else " - " for z in SEEDS)
    return (f"n {len(A):5d}  L {L.mean():5.0f}  H {H.mean():4.1f}  I {I.mean():4.0f}  S {S.mean():4.0f}  S/L {S.mean() / L.mean():.2f}  RI {L.mean() / H.mean():4.1f}  tan i {H.mean() / I.mean():.3f}  tan s {H.mean() / S.mean():.3f}  SI from means {I.mean() / S.mean():.2f}"
            f"  SI per ripple {si.mean():.2f} +- {si.std():.2f} (med {np.median(si):.2f})   per seed L {lz}; SI {pz}")


print("1983 field, ratios from the means: RI 21.4, tan i 0.063, tan s 0.185, SI 2.96 (computed per ripple: RI 22, tan i 0.05, tan s 0.20, SI 3).")
print(f"Symmetry index against crest spacing over time, runs {SERIES} (seeds 271-274), up to {tmax:.0f} min. 1983 field: L 107, S 27 (S/L 0.25), SI 3; photo 21 Sept. 2026: L 64, S 23 (S/L 0.36), SI per ripple 1.85.")
for hmin in (2.5, 1.5):
    print(f"\n(1) windows in time (min), ripples with H >= {hmin} mm")
    for a, b in WINDOWS: print(f"  {a:4d}-{b:<4d} " + line(M.measure(FOLDER, SERIES, a * 60, b * 60, hmin=hmin)[0]))
print("\n(2) per ripple within a window: is the asymmetry related to the crest spacing? (H >= 2.5 mm; thirds by L; r = correlation of SI per ripple with L)")
last = [(20, 30), (45, 60), (int(tmax) - 30, int(tmax))] if tmax <= 240 else [(45, 60), (210, 240), (int(tmax) - 60, int(tmax))]
for a, b in last:
    A = M.measure(FOLDER, SERIES, a * 60, b * 60)[0]
    if len(A) < 30: continue
    L, H, I, S, Z = A.T; q = np.percentile(L, [100 / 3, 200 / 3]); kl = np.digitize(L, q)
    print(f"  {a}-{b} min: r(SI, L) {np.corrcoef(I / S, L)[0, 1]:+.2f}; r(S/L, L) {np.corrcoef(S / L, L)[0, 1]:+.2f}; " + "; ".join(
        f"L {L[kl == k].mean():.0f}: S {S[kl == k].mean():.0f}, SI from means {I[kl == k].mean() / S[kl == k].mean():.2f}, per ripple {np.mean(I[kl == k] / S[kl == k]):.2f}" for k in range(3)))
if len(sys.argv) > 3:
    print("\n(3) check: first 4 hours bit-identical to SR0 (h, top, ins, kr, wr)?")
    for z in SEEDS:
        a = np.load(os.path.join(FOLDER, f"c23_z{z}_{SERIES}.npz")); s = np.load(os.path.join(sys.argv[3], f"c23_z{z}_SR0.npz")); n = len(s["t"])
        print(f"  seed {z}: " + ("YES" if all(np.array_equal(a[k][:n], s[k]) for k in ("h", "top", "ins", "kr", "wr")) else "NO"))
