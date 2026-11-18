"""Symmetry index per ripple for the runs of Tables 7 and 8 (computed per ripple and averaged, as in 1983; descriptive).

Measuring rule as in measure.py, 210-240 min. Per seed: SI from the mean I and S, and SI computed per ripple and averaged;
also RI, stoss and lee slope per ripple. Paired differences as in contributions_from_means.py, plus SG0 (sliding off) against SR0.
Usage (from the folder with the runs):  python3 contributions_per_ripple.py [folder with the runs]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, measure as M
import contributions_from_means as B


def ratios(series):
    m = B.find(series)
    if m is None: return None
    A = M.measure(m, series, 12600, 14400)[0]; Z = A[:, 4]; out = {}
    for z in np.unique(Z):
        L, H, I, S = A[Z == z, :4].T
        out[int(z)] = dict(SIm=I.mean() / S.mean(), SIr=np.mean(I / S), RIr=np.mean(L / H), stossr=100 * np.mean(H / I), leer=100 * np.mean(H / S),
                           RIm=L.mean() / H.mean(), stossm=100 * H.mean() / I.mean(), leem=100 * H.mean() / S.mean(), n=len(L))
    return out


if __name__ == "__main__":
    B.FOLDER = sys.argv[1] if len(sys.argv) > 1 else B.FOLDER
    SERIES = list(B.SERIES) + [("SG0", "SR0 without wind-driven sliding")]
    PAIRS = list(B.PAIRS) + [("SR0", "SG0", "sliding on instead of off (article model)")]
    print("Symmetry index per seed, 210-240 min: SI from the means (m) and computed per ripple and averaged (r, as 1983); mean over the seeds.")
    print("Per ripple also RI, stoss and lee slope (%). 1983 field per ripple: RI 22, SI 3, stoss 5 %, lee 20 %.\n")
    R = {}
    for series, name in SERIES:
        P = ratios(series)
        if P is None: print(f"  {series:5s} {name}: runs missing"); continue
        R[series] = P; g = lambda k: np.mean([P[z][k] for z in P])
        print(f"  {series:5s} {name:46s} SI m {g('SIm'):.2f} / r {g('SIr'):.2f} (seeds r " + " / ".join(f"{P[z]['SIr']:.2f}" for z in P) + ")"
              f"   per ripple RI {g('RIr'):.1f} stoss {g('stossr'):.1f} lee {g('leer'):.1f}   from means RI {g('RIm'):.1f} stoss {g('stossm'):.1f} lee {g('leem'):.1f}")
    print("\nPaired per seed, difference in SI: from the means | per ripple (smallest .. largest of the seeds)")
    for a, b, name in PAIRS:
        if a in R and b in R:
            zs = [z for z in R[a] if z in R[b]]
            dm = [R[a][z]['SIm'] - R[b][z]['SIm'] for z in zs]; dr = [R[a][z]['SIr'] - R[b][z]['SIr'] for z in zs]
            print(f"  {name:52s} {a} - {b}:  m {min(dm):+.2f} .. {max(dm):+.2f}  |  r {min(dr):+.2f} .. {max(dr):+.2f}   (r per seed " + " / ".join(f"{x:+.2f}" for x in dr) + ")")
