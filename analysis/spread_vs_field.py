"""Comparison as a spread (Tables 5 and 6; measures fixed in advance on 15 Nov. 2026, outcome descriptive).

Ripple measures per ripple in the article model (SR0, 210-240 min, measuring rule of the article) against Table IA of Brugmans (1983) (n = 325):
(A) distribution per measure, unweighted and weighted by L (coin throw); location, spread, share within the field range, overlap;
(B) reproduced field survey of 13 sample areas x 25 ripples; (C) correlations between sample-area means (Table IIA) and per ripple; (D) SG0 for comparison.
Usage (from the folder with the runs):  python3 spread_vs_field.py [folder with the runs]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from math import erf, sqrt
import measure as M, ripple_measure as RM
FOLDER = sys.argv[1] if len(sys.argv) > 1 else "."
SEEDS = (271, 272, 273, 274)
NAMES = ("L", "H", "I", "S", "RI", "SI", "tan s", "tan i")
FIELD = {"L": (107, 21, 58, 170), "H": (5, 1.5, 1.5, 8.5), "I": (80, 15, 40, 117), "S": (27, 8, 10, 63),          # m, s, range (Table IA)
         "RI": (22, 5, 12, 39), "SI": (3, 1, 1, 6), "tan s": (0.20, 0.06, 0.08, 0.36), "tan i": (0.05, 0.01, 0.03, 0.11)}
PAIRS = [("L", "I", 95), ("L", "S", 84), ("L", "H", 67), ("H", "I", 73), ("H", "S", 38), ("S", "I", 63),              # Table IIA, r x 100
         ("SI", "S", -82), ("SI", "tan s", 69), ("tan s", "S", -54),
         ("RI", "tan i", -92), ("RI", "tan s", -88), ("RI", "H", -82), ("H", "tan i", 74), ("H", "tan s", 55), ("tan s", "tan i", 74)]


def read(series, t0=210 * 60, t1=240 * 60, prominence=3.0, hmin=2.5):
    """Per seed a table (n, 8): L, H, I, S, RI, SI, tan s, tan i per ripple."""
    out = {}
    for z in SEEDS:
        d = np.load(os.path.join(FOLDER, f"c23_z{z}_{series}.npz")); H, T, dx = d["h"], d["t"], float(d["dx"]); dz = float(d["dz"]); x = np.arange(H.shape[1]) * dx; A = []
        for k in range(len(T)):
            if not t0 < T[k] <= t1: continue
            h = M.smooth(H[k] * dz); h -= h.mean(); rip, _, _ = RM.ripples(x, h, prom_frac=0.0, prom_min=prominence)
            A += [(r["L"], r["H"], r["I"], r["S"]) for r in rip if r["H"] >= hmin]
        L, Hh, I, S = np.array(A).T
        out[z] = np.column_stack([L, Hh, I, S, L / Hh, I / S, Hh / S, Hh / I])
    return out


def wmean(v, w): return float(np.sum(v * w) / np.sum(w))
def wsd(v, w): m = wmean(v, w); return float(np.sqrt(np.sum(w * (v - m) ** 2) / np.sum(w)))
def wquant(v, w, q):
    o = np.argsort(v); c = np.cumsum(w[o]) / np.sum(w); return [float(v[o][min(np.searchsorted(c, p), len(v) - 1)]) for p in q]
def Phi(x): return 0.5 * (1 + erf(x / sqrt(2)))


def overlap(v, w, m, s, lo, hi, nk=40):
    """Common area of the (weighted) model distribution and the normal distribution of the field truncated to [lo, hi]."""
    a = min(lo, float(v.min())); b = max(hi, float(v.max())); r = np.linspace(a, b, nk + 1)
    pm = np.histogram(v, r, weights=w)[0] / np.sum(w)
    norm = Phi((hi - m) / s) - Phi((lo - m) / s)
    pv = np.array([max(0.0, Phi((min(r[i + 1], hi) - m) / s) - Phi((max(r[i], lo) - m) / s)) if r[i + 1] > lo and r[i] < hi else 0.0 for i in range(nk)]) / norm
    return float(np.minimum(pm, pv).sum())


def label(o): return "comparable" if o >= 0.7 else ("partly" if o >= 0.4 else "different")


def part_A(D, title):
    A = np.vstack([D[z] for z in SEEDS]); print(f"\n{title}: {len(A)} ripple measurements (all records 210-240 min, 4 seeds)")
    for wn, w in (("unweighted", np.ones(len(A))), ("weighted by L (coin throw)", A[:, 0].copy())):
        print(f"  {wn}:")
        print(f"    {'measure':7s} {'field m':>7s} {'s':>6s} {'range':>11s} | {'model mean':>10s} {'sd':>7s} {'median':>8s} {'2.5-97.5 %':>15s} | {'location':>8s} {'spread':>7s} {'in range':>8s} {'overlap':>7s}")
        for j, nm in enumerate(NAMES):
            m, s, lo, hi = FIELD[nm]; v = A[:, j]; g = wmean(v, w); sd = wsd(v, w); q = wquant(v, w, (0.025, 0.5, 0.975)); o = overlap(v, w, m, s, lo, hi)
            print(f"    {nm:7s} {m:7.3g} {s:6.3g} {f'{lo:g}-{hi:g}':>11s} | {g:10.3g} {sd:7.3g} {q[1]:8.3g} {f'{q[0]:.3g}-{q[2]:.3g}':>15s} | {(g - m) / s:+8.2f} {sd / s:7.2f} "
                  f"{wmean(((v >= lo) & (v <= hi)).astype(float), w):8.2f} {o:7.2f}  {label(o)}")
    return A


def parts_BC(D, nh=2000):
    rng = np.random.default_rng(1983); Mn = np.zeros((nh, 8)); Sd = np.zeros((nh, 8)); R = np.zeros((nh, len(PAIRS))); idx = {n: i for i, n in enumerate(NAMES)}
    for h in range(nh):
        vl = []
        for a in range(13):
            T = D[SEEDS[a % 4]]; p = T[:, 0] / T[:, 0].sum(); vl.append(T[rng.choice(len(T), 25, p=p)])
        al = np.vstack(vl); Mn[h] = al.mean(0); Sd[h] = al.std(0, ddof=1); vm = np.array([v.mean(0) for v in vl])
        R[h] = [np.corrcoef(vm[:, idx[a]], vm[:, idx[b]])[0, 1] for a, b, _ in PAIRS]
    print("\n(B) Reproduced field survey: 13 sample areas x 25 ripples, drawn with probability proportional to L; 2000 repetitions")
    print(f"    {'measure':7s} {'field m':>7s} | {'model m: median':>16s} {'2.5-97.5 %':>15s} | {'field s':>7s} | {'model s: median':>16s} {'2.5-97.5 %':>15s}")
    for j, nm in enumerate(NAMES):
        m, s = FIELD[nm][:2]; a = np.percentile(Mn[:, j], (2.5, 50, 97.5)); b = np.percentile(Sd[:, j], (2.5, 50, 97.5))
        print(f"    {nm:7s} {m:7.3g} | {a[1]:16.3g} {f'{a[0]:.3g}-{a[2]:.3g}':>15s} | {s:7.3g} | {b[1]:16.3g} {f'{b[0]:.3g}-{b[2]:.3g}':>15s}")
    print("\n(C) Correlations (r x 100) between the means of the 13 sample areas; field = Table IIA. Note: in the model the sample areas differ only by chance.")
    al = np.vstack([D[z] for z in SEEDS])
    print(f"    {'pair':16s} {'field':>5s} | {'model median':>13s} {'2.5-97.5 %':>12s} | {'per ripple (model)':>18s}")
    for k, (a, b, rv) in enumerate(PAIRS):
        q = np.percentile(R[:, k], (2.5, 50, 97.5)) * 100; rr = np.corrcoef(al[:, idx[a]], al[:, idx[b]])[0, 1] * 100
        print(f"    {a + ' - ' + b:16s} {rv:+5d} | {q[1]:+13.0f} {f'{q[0]:+.0f}..{q[2]:+.0f}':>12s} | {rr:+18.0f}")


if __name__ == "__main__":
    print("Comparison as a spread: article model against Brugmans 1983, Table IA (common sample areas, n = 325). Measuring rule of the article; H = depth of the trough below the line through the crests.")
    print("location = (model mean - field mean) / field sd; spread = model sd / field sd; overlap = common area of the distributions (0-1).")
    D = read("SR0"); print("\n(A) SR0 = article model"); part_A(D, "SR0")
    for z in SEEDS: print(f"    seed {z}: {len(D[z])} measurements; per ripple mean SI {D[z][:, 5].mean():.2f}, weighted by L {wmean(D[z][:, 5], D[z][:, 0]):.2f}; I/S from the means {D[z][:, 2].mean() / D[z][:, 3].mean():.2f}")
    parts_BC(D)
    if os.path.exists(os.path.join(FOLDER, "c23_z271_SG0.npz")):
        print("\n(D) SG0 = article model without wind-driven sliding"); part_A(read("SG0"), "SG0")
