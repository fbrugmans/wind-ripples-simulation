"""Numbers for the article from the runs of the article model (series SR0) and the comparison series (descriptive).

Measuring rule as in measure.py (prominence 3 mm, H >= 2.5 mm). Prints:
  (1) SR0 over time; (2) distribution per ripple at 210-240 min (Tables 4 and 5); (3) migration rate; (4) share of wind-driven
  sliding; (5) initiation runs RN31L/RN50L (Table 3) and the descriptions of the wind (Table 7); (6) mean profile around the crest.
Usage (from the folder with the runs):  python3 article_numbers.py [folder with the runs]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import measure as M
FOLDER = sys.argv[1] if len(sys.argv) > 1 else "."
SEEDS = (271, 272, 273, 274)
FIELD = dict(L=(107, 21, 58, 170), H=(5, 1.5, 1.5, 8.5), I=(80, 15, 40, 117), S=(27, 8, 10, 63))          # 1983: mean, sd, smallest, largest
FIELD_RATIOS = dict(RI=(22, 5, 12, 39), SI=(3, 1, 1, 6), tan_s=(0.20, 0.06, 0.08, 0.36), tan_i=(0.05, 0.01, 0.03, 0.11))


def find(series):
    """Folder that holds the runs of a series (the runs folder itself), or None."""
    return FOLDER if os.path.exists(os.path.join(FOLDER, f"c23_z271_{series}.npz")) else None


def typical(A):
    L, H, I, S = A[:, :4].T
    return dict(n=len(L), L=L.mean(), H=H.mean(), I=I.mean(), S=S.mean(), RI=L.mean() / H.mean(), SI=I.mean() / S.mean(), stoss=H.mean() / I.mean(), lee=H.mean() / S.mean())


def line(name, A):
    if len(A) == 0: return f"  {name:34s} no ripples"
    m = typical(A); Z = A[:, 4]; ps = [typical(A[Z == z]) for z in np.unique(Z)]
    return (f"  {name:34s} n {m['n']:5d}  L {m['L']:5.0f}  H {m['H']:4.1f}  I {m['I']:4.0f}  S {m['S']:4.0f}  SI {m['SI']:.2f}  RI {m['RI']:4.1f}  stoss {m['stoss']:.3f}  lee {m['lee']:.3f}"
            + "   per seed L " + " / ".join(f"{p['L']:.0f}" for p in ps) + "; H " + " / ".join(f"{p['H']:.1f}" for p in ps) + "; SI " + " / ".join(f"{p['SI']:.2f}" for p in ps))


def migration(folder, series, t0, t1):
    """mm/min per seed: median of the shift between successive records (cross-correlation, periodic, parabola through the peak)."""
    out = []
    for z in SEEDS:
        f = os.path.join(folder, f"c23_z{z}_{series}.npz")
        if not os.path.exists(f): continue
        d = np.load(f); H, T, dx = d["h"], d["t"], float(d["dx"]); dz = float(d["dz"]); k = np.where((T > t0) & (T <= t1))[0]; v = []
        for a, b in zip(k[:-1], k[1:]):
            h0 = M.smooth(H[a] * dz); h1 = M.smooth(H[b] * dz); h0 = h0 - h0.mean(); h1 = h1 - h1.mean(); N = len(h0)
            F = np.fft.irfft(np.conj(np.fft.rfft(h0)) * np.fft.rfft(h1), n=N); j = int(np.argmax(F)); a_, b_, c_ = F[(j - 1) % N], F[j], F[(j + 1) % N]
            s = 0.5 * (a_ - c_) / (a_ - 2 * b_ + c_) if (a_ - 2 * b_ + c_) != 0 else 0.0; j = j if j < N // 2 else j - N
            v.append((j + s) * dx / (T[b] - T[a]) * 60)
        out.append(float(np.median(v)))
    return out


def sliding_share(folder, series, t0, t1):
    """Share of wind-driven sliding in the distance travelled by bed grains (creep + sliding), per seed."""
    out = []
    for z in SEEDS:
        f = os.path.join(folder, f"c23_z{z}_{series}.npz")
        if not os.path.exists(f): continue
        d = np.load(f); k = (d["t"] > t0) & (d["t"] <= t1)
        if "wr" not in d.files or "skr" not in d.files or d["wr"].ndim != 2: continue
        g = float(d["wr"][k, 1].sum()); s = float(np.asarray(d["skr"])[k].sum()); out.append(g / (g + s) if g + s > 0 else np.nan)
    return out


if __name__ == "__main__":
    print("Numbers for the article, article model SR0 (descriptive; measuring rule of the article). Field: L 107, H 5, I 80, S 27 -> SI 2.96, RI 21.4, stoss 0.063, lee 0.185.")
    m = find("SR0")
    if m is None: print("runs SR0 missing"); sys.exit(1)
    print("\n(1) SR0 over time (windows in min)")
    for b in range(30, 241, 30): print(line(f"{b - 5}-{b}", M.measure(m, "SR0", (b - 5) * 60, b * 60)[0]))
    A = M.measure(m, "SR0", 12600, 14400)[0]; print(line("210-240", A))
    print("\n(2) SR0, 210-240 min: distribution per ripple (all seeds together)")
    L, H, I, S, Z = A.T
    for name, v in (("L", L), ("H", H), ("I", I), ("S", S)):
        g, sd, lo, hi = FIELD[name]
        print(f"  {name}: mean {v.mean():6.1f}  sd {v.std():5.1f} (field {g} +- {sd})  5-95 % {np.percentile(v, 5):.1f}-{np.percentile(v, 95):.1f}  smallest-largest {v.min():.1f}-{v.max():.1f}"
              f"  share within the field range {lo}-{hi}: {np.mean((v >= lo) & (v <= hi)):.2f}   sd per seed " + " / ".join(f"{v[Z == z].std():.1f}" for z in np.unique(Z)))
    for name, v in (("RI", L / H), ("SI", I / S), ("tan_s", H / S), ("tan_i", H / I)):
        g, sd, lo, hi = FIELD_RATIOS[name]
        print(f"  {name} computed per ripple and averaged: mean {v.mean():.3g}  sd {v.std():.3g}  median {np.median(v):.3g} (field {g} +- {sd})  share within the field range {lo}-{hi}: {np.mean((v >= lo) & (v <= hi)):.2f}"
              + "   per seed " + " / ".join(f"{v[Z == z].mean():.3g}" for z in np.unique(Z)))
    print(f"  share of ripples with I < S: {np.mean(I < S):.2f}; with SI >= 2: {np.mean(I / S >= 2):.2f}; ripples per record per metre: {len(L) / (4 * 60) / 2.14:.1f}")
    print("\n(3) migration (mm/min, per seed): SR0 210-240 min " + " / ".join(f"{v:.1f}" for v in migration(m, "SR0", 12600, 14400))
          + "; 25-30 min " + " / ".join(f"{v:.1f}" for v in migration(m, "SR0", 1500, 1800)))
    print("(4) share of wind-driven sliding in creep + sliding (distance), SR0 210-240 min, per seed: " + " / ".join(f"{v:.3f}" for v in sliding_share(m, "SR0", 12600, 14400)))
    print("\n(5) initiation runs with this model (flat bed, 30 min) and the descriptions of the wind")
    for series, name, t0, t1 in (("RN31L", "u* 0.31, 25-30 min", 1500, 1800), ("RN50L", "u* 0.50, 25-30 min", 1500, 1800), ("RN31L", "u* 0.31, 10-15 min", 600, 900), ("RN50L", "u* 0.50, 10-15 min", 600, 900),
                                 ("RN31L", "u* 0.31, 4-5 min", 240, 300), ("RN50L", "u* 0.50, 4-5 min", 240, 300)):
        mm = find(series)
        if mm is None: print(f"  {name}: runs missing"); continue
        print(line(f"{series} {name}", M.measure(mm, series, t0, t1)[0]))
    for series in ("RN31L", "RN50L"):
        mm = find(series)
        if mm: print(f"  migration {series} 25-30 min (mm/min, per seed): " + " / ".join(f"{v:.1f}" for v in migration(mm, series, 1500, 1800)))
    for series, name in (("NMO", "fixed A 3, B 1"), ("KAO", "fixed A 3, B 1 (mirrored rebound)"), ("ABO", "rough log theory, z0 d/30"), ("AZO", "rough log theory, z0 0.1 mm (KRUIP_INSLAG=1)"),
                         ("VLO", "van Driest u*_b 0.20"), ("BRLO", "van Driest + curvature 7 (KRUIP_INSLAG=1)"), ("PKO", "van Driest + curvature 7"), ("SQ0", "rough log theory, z0 0.1 mm"), ("SR0", "same + streamlines")):
        mm = find(series)
        if mm is None: print(f"  {series} ({name}): runs missing"); continue
        print(line(f"{series} {name}"[:34], M.measure(mm, series, 12600, 14400)[0]) + "   migration " + " / ".join(f"{v:.1f}" for v in migration(mm, series, 12600, 14400))
              + "   sliding " + " / ".join(f"{v:.2f}" for v in sliding_share(mm, series, 12600, 14400)))
    # (6) mean profile around the crest (the numbers in 'The lee side' of the article)
    import ripple_measure as RM
    mm = find("SR0")
    if mm:
        P = []
        for z in SEEDS:
            d = np.load(os.path.join(mm, f"c23_z{z}_SR0.npz")); dx, dz = float(d["dx"]), float(d["dz"]); x = np.arange(d["h"].shape[1]) * dx; w = int(110 / dx)
            for kk in np.where((d["t"] > 12600) & (d["t"] <= 14400))[0]:
                hh = M.smooth(d["h"][kk] * dz); hh = hh - hh.mean(); rr, _, _ = RM.ripples(x, hh, prom_frac=0.0, prom_min=3.0)
                for r in rr:
                    if r["H"] < 2.5: continue
                    c = int(r["c"]); P.append(hh[(c + np.arange(-w, w + 1)) % len(hh)] - hh[c])
        p = np.mean(P, 0); xs = np.arange(-w, w + 1) * dx; at = lambda v: float(np.interp(v, xs, p))
        print(f"\n(6) mean profile around the crest, SR0 210-240 min, {len(P)} ripples (aligned on the highest point of the smoothed bed; height relative to the crest, mm):")
        print("   stoss: " + "  ".join(f"{-v} mm: {at(-v):.2f}" for v in (2, 5, 10, 20, 40, 60, 80)))
        print("   lee:   " + "  ".join(f"+{v} mm: {at(v):.2f}" for v in (2, 5, 10, 20, 30, 40, 50, 60)))
        li, lo = (xs > 0) & (xs < 100), (xs < 0) & (xs > -100); il, io = np.argmin(np.where(li, p, 9)), np.argmin(np.where(lo, p, 9))
        print(f"   lee slope 0-10 mm {-at(10) / 10:.3f}, 10-45 mm {(at(10) - at(45)) / 35:.3f}; stoss slope 0-10 mm {-at(-10) / 10:.3f}, 10-60 mm {(at(-10) - at(-60)) / 50:.3f};"
              f" lowest point lee side at +{xs[il]:.0f} mm ({p[il]:.2f} mm), stoss side at {xs[io]:.0f} mm ({p[io]:.2f} mm)")
