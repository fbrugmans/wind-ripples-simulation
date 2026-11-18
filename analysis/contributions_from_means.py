"""Contribution of the processes examined to the asymmetry, symmetry index from the means (Table 8, second column; descriptive).

Measuring rule as in measure.py, 210-240 min; SI per seed from the mean I and S; paired differences between series that differ
in one thing. Usage (from the folder with the runs):  python3 contributions_from_means.py [folder with the runs]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, measure as M, article_numbers as A
FOLDER = sys.argv[1] if len(sys.argv) > 1 else "."


def find(series):
    return FOLDER if os.path.exists(os.path.join(FOLDER, f"c23_z271_{series}.npz")) else None


def si(series):
    m = find(series)
    if m is None: return None
    B = M.measure(m, series, 12600, 14400)[0]; Z = B[:, 4]
    return {int(z): (B[Z == z, 2].mean() / B[Z == z, 3].mean(), B[Z == z, 0].mean(), B[Z == z, 1].mean()) for z in np.unique(Z)}, A.sliding_share(m, series, 12600, 14400)


SERIES = (("NMO", "fixed A,B; no wind shadow"), ("NMS", "fixed A,B; wind shadow 5 deg"), ("KAO", "fixed A,B + mirrored rebound; none"), ("KAS", "same; wind shadow 5 deg"),
          ("ABO", "rough z0 d/30; none"), ("ABS", "rough z0 d/30; wind shadow 5 deg"), ("AZO", "rough z0 0.1, KRUIP_INSLAG=1; none"), ("AZS", "same; wind shadow 5 deg"),
          ("SQ0", "rough z0 0.1, KRUIP_INSLAG=0; none"), ("SQ5", "same; wind shadow 5 deg"), ("SR0", "same + streamlines; none"), ("SR5", "same + streamlines; wind shadow 5 deg"),
          ("VLO", "van Driest; none"), ("VLS", "van Driest; wind shadow"), ("BRLO", "van Driest + curvature, KRUIP_INSLAG=1; none"), ("BRLS", "same; wind shadow"),
          ("PKO", "van Driest + curvature, KRUIP_INSLAG=0; none"), ("PKS", "same; wind shadow"), ("CO10", "PKO + cohesion 0.1"), ("CO25", "PKO + cohesion 0.25"), ("CO50", "PKO + cohesion 0.5"))
PAIRS = (("NMS", "NMO", "wind shadow with fixed A,B"), ("KAS", "KAO", "wind shadow with fixed A,B + mirrored rebound"), ("ABS", "ABO", "wind shadow with rough z0 d/30 (KRUIP_INSLAG=1)"),
         ("AZS", "AZO", "wind shadow with rough z0 0.1 (KRUIP_INSLAG=1)"), ("SQ5", "SQ0", "wind shadow with rough z0 0.1 (KRUIP_INSLAG=0)"), ("SR5", "SR0", "wind shadow with streamlines"),
         ("ABO", "KAO", "rough log theory instead of fixed A,B"), ("AZO", "ABO", "z0 0.1 instead of d/30"), ("VLO", "ABO", "viscous sublayer instead of rough"),
         ("SQ0", "AZO", "KRUIP_INSLAG off (no wind shadow, rough)"), ("SQ5", "AZS", "KRUIP_INSLAG off (with wind shadow, rough)"), ("PKO", "BRLO", "KRUIP_INSLAG off (van Driest + curvature)"),
         ("SR0", "SQ0", "streamlines"), ("CO10", "PKO", "cohesion 0.1 (sliding 0.36 -> 0.21)"), ("CO25", "PKO", "cohesion 0.25 (-> 0.09)"), ("CO50", "PKO", "cohesion 0.5 (-> 0.02)"))

if __name__ == "__main__":
    print("Symmetry index (SI from the mean I and S per seed), L and H, share of sliding; 210-240 min"); R = {}
    for series, name in SERIES:
        r = si(series)
        if r is None: print(f"  {series:5s} {name}: runs missing"); continue
        P, g = r; R[series] = P
        print(f"  {series:5s} {name:46s} SI " + " / ".join(f"{P[z][0]:.2f}" for z in P) + f"  mean {np.mean([P[z][0] for z in P]):.2f}   L {np.mean([P[z][1] for z in P]):.0f}  H {np.mean([P[z][2] for z in P]):.1f}"
              + "   sliding " + (" / ".join(f"{x:.2f}" for x in g) if g else "-"))
    print("\nPaired per seed (difference in SI):")
    for a, b, name in PAIRS:
        if a in R and b in R: print(f"  {name:52s} {a} - {b}: " + " / ".join(f"{R[a][z][0] - R[b][z][0]:+.2f}" for z in R[a] if z in R[b]))
