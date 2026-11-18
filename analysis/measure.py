"""Shared tools for the analyses: smoothing, the ripple measure of the article, and the ripple shape of Brugmans (1983).

measure(): the measuring rule of the article. In each record of the bed (smoothed over 2 mm) crests and troughs are found with a
fixed prominence of 3 mm (model/ripple_measure.py); ripples lower than 2.5 mm are not counted. Returns per ripple L, H, I, S and
the seed. The typical ripple is the mean of L, H, I and S; SI, RI and the slopes are computed from these means, or per ripple
and then averaged as in 1983.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model"))
import numpy as np
import ripple_measure as RM

SEEDS = (271, 272, 273, 274)


def smooth(a, k=5):
    """Periodic running mean over k columns (k = 5: 2 mm at a cell width of 0.4 mm)."""
    return np.convolve(np.concatenate([a[-k:], a, a[:k]]), np.ones(k) / k, "same")[k:-k]


def measure(folder, series, t0, t1, prominence=3.0, hmin=2.5):
    """Ripples of all records with t0 < t <= t1 of the four seeds of a series: array (L, H, I, S, seed), number of records, bed length."""
    A = []; nrec = 0; Lb = None
    for z in SEEDS:
        f = os.path.join(folder, f"c23_z{z}_{series}.npz")
        if not os.path.exists(f): continue
        d = np.load(f); H, T, dx = d["h"], d["t"], float(d["dx"]); dz = float(d["dz"]) if "dz" in d.files else 0.2
        x = np.arange(H.shape[1]) * dx; Lb = x[-1] + dx
        for k in range(len(T)):
            if not t0 < T[k] <= t1: continue
            h = smooth(H[k] * dz); h -= h.mean(); rip, _, _ = RM.ripples(x, h, prom_frac=0.0, prom_min=prominence); nrec += 1
            A += [(r["L"], r["H"], r["I"], r["S"], z) for r in rip if r["H"] >= hmin]
    return np.array(A).reshape(-1, 5), nrec, Lb


def tmax(folder, series):
    f = os.path.join(folder, f"c23_z271_{series}.npz"); return float(np.load(f)["t"][-1]) if os.path.exists(f) else None


def line(name, A, nrec, Lb):
    """One line of output: ripples per metre, mean L, H, I, S and the ratios from the means."""
    if nrec == 0: return f"  {name:52s} no records"
    if len(A) == 0: return f"  {name:52s} no ripples >= 2.5 mm ({nrec} records)"
    L, H, I, S, Z = A.T; per = " / ".join(f"{I[Z == z].mean() / S[Z == z].mean():.2f}" for z in np.unique(Z))
    return (f"  {name:52s} {len(L) / nrec / (Lb / 1000):4.1f} per m  L {L.mean():5.0f}  H {H.mean():4.1f}  I {I.mean():4.0f}  S {S.mean():4.0f}"
            f"  -> SI {I.mean() / S.mean():.2f} (seeds {per})  RI {L.mean() / H.mean():4.1f}  stoss {H.mean() / I.mean():.3f}  lee {H.mean() / S.mean():.3f}")


# The ripple of Figure 1 of Brugmans (1983), digitised from the scan (pixel coordinates; y downward). Used as the template.
F1_BASE = 363.5                                   # line through the troughs
F1_STOSS = [(22, 363.5), (30, 362), (50, 357.5), (70, 353), (90, 348), (110, 343), (130, 337.5), (150, 332.5), (170, 328), (190, 323), (210, 316.5), (230, 310.5),
           (250, 305), (270, 298), (290, 290), (310, 280.5), (330, 271), (350, 260.5), (370, 251.5), (390, 240.5), (410, 229), (430, 217.5), (450, 208),
           (470, 199), (490, 190.5), (510, 182.5), (530, 176), (550, 172), (570, 169.5), (590, 168.2), (605, 167.5)]
F1_LEE = [(605, 167.5), (616, 169), (620, 170), (630, 173.5), (650, 185), (670, 207), (690, 237.5), (700, 252), (710, 266), (720, 278.5), (730, 291),
          (750, 315), (770, 335), (790, 350), (810, 359), (830, 362.5), (843, 363.5)]
