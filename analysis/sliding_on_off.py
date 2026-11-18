"""Wind-driven sliding on and off in the article model (fixed in advance on 15 Nov. 2026).

ON = SR0, OFF = SG0 (same settings + WINDROL=0). Measuring rule as in measure.py; paired per seed SR0 - SG0.
Criterion G1 (fixed in advance): sliding contributes to the asymmetry if the difference in SI is > 0 in all four seeds.
Usage (from the folder with the runs):  python3 sliding_on_off.py <folder ON> <folder OFF> [series ON] [series OFF]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import measure as M, article_numbers as A
FON, FOFF = sys.argv[1], sys.argv[2]; SON = sys.argv[3] if len(sys.argv) > 3 else "SR0"; SOFF = sys.argv[4] if len(sys.argv) > 4 else "SG0"
SEEDS = (271, 272, 273, 274)
FIELD = dict(L=(107, 21), H=(5, 1.5), I=(80, 15), S=(27, 8), RI=(21.4, 5), SI=(2.96, 1), stoss=(0.063, 0.01), lee=(0.185, 0.06))


def per_seed(folder, series, t0, t1):
    B, nrec, Lb = M.measure(folder, series, t0, t1); out = {}
    for z in SEEDS:
        q = B[B[:, 4] == z]
        if len(q) == 0: out[z] = None; continue
        L, H, I, S = (q[:, i].mean() for i in range(4))
        out[z] = dict(n=len(q), L=L, H=H, I=I, S=S, RI=L / H, SI=I / S, stoss=H / I, lee=H / S)
    return out, nrec, Lb


def creep(folder, series, t0, t1):
    out = []
    for z in SEEDS:
        d = np.load(os.path.join(folder, f"c23_z{z}_{series}.npz")); k = (d["t"] > t0) & (d["t"] <= t1); out.append((float(np.asarray(d["skr"])[k].sum()), float(d["wr"][k, 1].sum())))
    return out


def spectrum(folder, series, t0, t1):
    out = []
    for z in SEEDS:
        d = np.load(os.path.join(folder, f"c23_z{z}_{series}.npz")); dx, dz = float(d["dx"]), float(d["dz"]); k = np.where((d["t"] > t0) & (d["t"] <= t1))[0]; N = d["h"].shape[1]
        F = np.mean([np.abs(np.fft.rfft(d["h"][i] * dz - (d["h"][i] * dz).mean())[1:N // 2 + 1]) for i in k], 0); lam = N * dx / np.arange(1, N // 2 + 1); sel = (lam > 4) & (lam < 400)
        g = np.exp(np.convolve(np.log(F + 1e-12), np.ones(9) / 9, "same")); out.append((float(lam[sel][np.argmax(g[sel])]), float(np.mean([M.smooth(d["h"][i] * dz).std() for i in k]))))
    return out


def row(v, f="{:.2f}"): return " / ".join("  -" if x is None or not np.isfinite(x) else f.format(x) for x in v)


if __name__ == "__main__":
    for m, s in ((FON, SON), (FOFF, SOFF)):
        if not all(os.path.exists(os.path.join(m, f"c23_z{z}_{s}.npz")) for z in SEEDS): print(f"runs {s} missing in {m}"); sys.exit(1)
    print(f"Wind-driven sliding on ({SON}) and off ({SOFF}); measuring rule of the article; per seed 271 / 272 / 273 / 274. Field: L 107, H 5, I 80, S 27 -> SI 2.96, RI 21.4, stoss 0.063, lee 0.185.")
    PA, nrec, Lb = per_seed(FON, SON, 12600, 14400); PU, nrecu, _ = per_seed(FOFF, SOFF, 12600, 14400)
    dens = [None if PU[z] is None else PU[z]["n"] / (nrecu / 4) / (Lb / 1000) for z in SEEDS]
    print("\n(1) 210-240 min")
    print(f"   ripples per metre of bed per record, {SOFF}: " + row(dens, "{:.1f}") + f"   ({SON}: " + row([PA[z]["n"] / (nrec / 4) / (Lb / 1000) for z in SEEDS], "{:.1f}") + ")")
    valid = all(d is not None and d >= 2 for d in dens)
    for name, P in ((f"{SON} on", PA), (f"{SOFF} off", PU)):
        for m in FIELD: print(f"   {name:10s} {m:5s} " + row([None if P[z] is None else P[z][m] for z in SEEDS], "{:.3g}"))
        print(f"   {name:10s} number of the 8 field measures within field +- 1 sd: " + " / ".join("-" if P[z] is None else str(sum(abs(P[z][m] - g) <= s for m, (g, s) in FIELD.items())) for z in SEEDS))
    print(f"   paired {SON} - {SOFF}:")
    D = {}
    for m in FIELD:
        D[m] = [None if PU[z] is None else PA[z][m] - PU[z][m] for z in SEEDS]; print(f"      {m:5s} " + row(D[m], "{:+.3g}"))
    if valid:
        d = np.array(D["SI"], float); verdict = "sliding CONTRIBUTES to the asymmetry" if np.all(d > 0) else "sliding WORKS AGAINST the asymmetry" if np.all(d < 0) else "NO DEMONSTRABLE CONTRIBUTION"
        print(f"   G1: difference in SI > 0 in {int(np.sum(d > 0))} of 4 seeds, < 0 in {int(np.sum(d < 0))} of 4 -> {verdict} (mean {d.mean():+.2f})")
    else: print("   G1: NO TEST (condition not met: fewer than 2 ripples per metre of bed per record in one or more seeds)")
    print("\n(2) over time (windows in min): L, H, SI per seed")
    for b in range(30, 241, 30):
        for name, m, s in ((SON, FON, SON), (SOFF, FOFF, SOFF)):
            P, _, _ = per_seed(m, s, (b - 5) * 60, b * 60)
            print(f"   {b - 5:3d}-{b:3d} {name:5s} L " + row([None if P[z] is None else P[z]["L"] for z in SEEDS], "{:.0f}") + "   H " + row([None if P[z] is None else P[z]["H"] for z in SEEDS], "{:.1f}")
                  + "   SI " + row([None if P[z] is None else P[z]["SI"] for z in SEEDS]))
    print("\n(3) migration, creep distance, spectrum (210-240 min)")
    ma, mu = A.migration(FON, SON, 12600, 14400), A.migration(FOFF, SOFF, 12600, 14400)
    print(f"   migration (mm/min): {SON} " + row(ma, "{:.1f}") + f"; {SOFF} " + row(mu, "{:.1f}"))
    ka, ku = creep(FON, SON, 12600, 14400), creep(FOFF, SOFF, 12600, 14400)
    print(f"   creep distance {SOFF} / {SON}: " + row([u[0] / a[0] for a, u in zip(ka, ku)]) + f"; sliding distance {SON} (share of creep + sliding): " + row([a[1] / (a[0] + a[1]) for a in ka])
          + f"; {SOFF}: " + row([u[1] / (u[0] + u[1]) if u[0] + u[1] > 0 else 0.0 for u in ku])
          + f"; (creep + sliding) {SOFF} / {SON}: " + row([(u[0] + u[1]) / (a[0] + a[1]) for a, u in zip(ka, ku)]))
    sa, su = spectrum(FON, SON, 12600, 14400), spectrum(FOFF, SOFF, 12600, 14400)
    print(f"   peak of the bed spectrum (mm): {SON} " + row([x[0] for x in sa], "{:.0f}") + f"; {SOFF} " + row([x[0] for x in su], "{:.0f}")
          + f";  rms height (mm): {SON} " + row([x[1] for x in sa]) + f"; {SOFF} " + row([x[1] for x in su]))
