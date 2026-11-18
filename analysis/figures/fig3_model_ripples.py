"""Figure 3: the ripples of the article model (SR0; descriptive).

(a) part of the bed profile after 4 hours with the crests and troughs of the measuring rule; (b) typical ripple of model and field as a
triangle from the mean I, S and H, with the mean model profile around the crest; (c) crest spacing and height over time; (d) symmetry index over time.
Usage (from the folder with the runs):  python3 fig3_model_ripples.py [folder with the runs] [output.png] [folder of the long run] [series of the long run]
With the long run (SL16, 16 hours, first 4 hours bit-identical to SR0) panels (c) and (d) run to its end; (a) and (b) stay at 4 hours.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import measure as M, ripple_measure as RM
FOLDER = sys.argv[1] if len(sys.argv) > 1 else "."; OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "fig3_model_ripples.png")
FOLDER_L = sys.argv[3] if len(sys.argv) > 3 else FOLDER; SERIES_L = sys.argv[4] if len(sys.argv) > 4 else "SR0"
SEEDS = (271, 272, 273, 274); FIELD = dict(L=(107, 21), H=(5, 1.5), I=80, S=27)
KM, KV, KZ = "#1f4e79", "#b5651d", "#9aa5b1"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
fig = plt.figure(figsize=(11, 10.5)); gs = fig.add_gridspec(3, 2, height_ratios=[0.75, 1.2, 1.0], hspace=0.75, wspace=0.3)

# (a) profile
ax = fig.add_subplot(gs[0, :]); d = np.load(os.path.join(FOLDER, "c23_z271_SR0.npz")); dx, dz = float(d["dx"]), float(d["dz"]); x = np.arange(d["h"].shape[1]) * dx
h = M.smooth(d["h"][-1] * dz); h = h - h.mean(); rip, _, _ = RM.ripples(x, h, prom_frac=0.0, prom_min=3.0)
x0, x1 = 300, 1100; k = (x >= x0) & (x <= x1); ax.fill_between(x[k] - x0, h[k], h[k].min() - 1.5, color="#e9dcc3"); ax.plot(x[k] - x0, h[k], color="#6b5a3a", lw=1.2)
for r in [r for r in rip if r["H"] >= 2.5]:
    c = int(r["c"]); t = int(r["t"])
    if x0 <= x[c] <= x1: ax.plot(x[c] - x0, h[c], "v", color=KM, ms=6, mec="none")
    if x0 <= x[t] <= x1: ax.plot(x[t] - x0, h[t], "^", color=KV, ms=6, mec="none")
ax.set_aspect(10); ax.set_xlim(0, x1 - x0); ax.set_yticks([-3, 0, 3]); ax.set_xlabel("distance along the x-axis (mm); wind from the left"); ax.set_ylabel("height (mm)")
ax.set_title("(a) Part of the bed after 4 hours (one seed; height exaggerated 10 ×); triangles: crests and troughs according to the measuring rule", loc="left", fontsize=12)

# (b) typical ripple
ax = fig.add_subplot(gs[1, :]); A = M.measure(FOLDER, "SR0", 12600, 14400)[0]; L, H, I, S = (A[:, i].mean() for i in range(4))
P = []; w = int(110 / dx)
for z in SEEDS:
    d = np.load(os.path.join(FOLDER, f"c23_z{z}_SR0.npz")); T = d["t"]
    for kk in np.where((T > 12600) & (T <= 14400))[0][::6]:
        hh = M.smooth(d["h"][kk] * dz); hh = hh - hh.mean(); rr, _, _ = RM.ripples(x, hh, prom_frac=0.0, prom_min=3.0)
        for r in rr:
            if r["H"] < 2.5: continue
            c = int(r["c"]); P.append(hh[(c + np.arange(-w, w + 1)) % len(hh)] - hh[c])
P = np.mean(P, 0); xs = np.arange(-w, w + 1) * dx
ax.plot(xs, P, color=KZ, lw=2.2, label="model: mean profile around the crest")
ax.plot([-I, 0, S], [-H, 0, -H], color=KM, lw=2.4, label=f"model: triangle from the means (I {I:.0f}, S {S:.0f}, H {H:.1f} mm)")
ax.plot([-FIELD["I"], 0, FIELD["S"]], [-5, 0, -5], color=KV, lw=2.4, ls="--", label="1983 field: triangle from the means (I 80, S 27, H 5 mm)")
ax.set_aspect(5); ax.set_xlim(-105, 105); ax.set_ylim(-5.6, 0.6); ax.set_xlabel("distance from the crest (mm); stoss side left, lee side right"); ax.set_ylabel("height relative to the crest (mm)")
ax.legend(frameon=False, fontsize=10.5, loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=1)
ax.set_title("(b) The typical ripple of model and field (height exaggerated 5 ×)", loc="left", fontsize=12)

# (c), (d) time
HOURS = SERIES_L != "SR0"; tt = np.arange(30, 961, 30) if HOURS else np.arange(30, 241, 30); tx = tt / 60 if HOURS else tt; ms = 3 if HOURS else 6
R = {z: [] for z in SEEDS}
for b in tt:
    B = M.measure(FOLDER_L, SERIES_L, (b - 5) * 60, b * 60)[0]
    for z in SEEDS:
        q = B[B[:, 4] == z]; R[z].append((q[:, 0].mean(), q[:, 1].mean(), q[:, 2].mean() / q[:, 3].mean()))
R = {z: np.array(v) for z, v in R.items()}; G = np.mean([R[z] for z in SEEDS], 0)
ax = fig.add_subplot(gs[2, 0]); ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
ax.axhspan(107 - 21, 107 + 21, color=KM, alpha=0.10, lw=0); ax.axhline(107, color=KM, lw=1, ls=":")
XL = "time (hours)" if HOURS else "time (min)"; XT = np.arange(0, 17, 2) if HOURS else tt; xa = tx[0] * 1.13
for z in SEEDS: ax.plot(tx, R[z][:, 0], color=KM, lw=0.8, alpha=0.5); ax2.plot(tx, R[z][:, 1], color=KV, lw=0.8, alpha=0.5)
ax.plot(tx, G[:, 0], "o-", ms=ms, color=KM, lw=2); ax2.plot(tx, G[:, 1], "s-", ms=ms, color=KV, lw=2); ax2.axhline(5, color=KV, lw=1, ls=":")
ax.set_ylim(0, 180); ax2.set_ylim(0, 9); ax.set_xlabel(XL); ax.set_ylabel("crest spacing L (mm)", color=KM); ax2.set_ylabel("height H (mm)", color=KV); ax.set_xticks(XT)
ax.set_title("(c) Crest spacing and height over time", loc="left", fontsize=12); ax.text(xa, 165, "grey band: field 107 ± 21 mm", color=KM, fontsize=10); ax2.text(tx[-1] * 0.83, 5.35, "field 5", color=KV, fontsize=10)
ax = fig.add_subplot(gs[2, 1])
for z in SEEDS: ax.plot(tx, R[z][:, 2], color=KM, lw=0.8, alpha=0.5)
ax.plot(tx, G[:, 2], "o-", ms=ms, color=KM, lw=2, label="model"); ax.axhline(2.96, color=KV, lw=1.6, ls="--"); ax.axhline(1, color="0.6", lw=1, ls=":")
ax.text(xa, 3.05, "field 2.96", color=KV, fontsize=10); ax.text(xa, 0.78, "symmetric", color="0.4", fontsize=10)
ax.set_ylim(0, 3.5); ax.set_xlabel(XL); ax.set_ylabel("symmetry index SI = I : S"); ax.set_xticks(XT); ax.set_title("(d) Symmetry index over time", loc="left", fontsize=12)
fig.savefig(OUT, dpi=170, bbox_inches="tight"); print("written:", OUT, "| typical ripple", f"L {L:.0f} H {H:.2f} I {I:.0f} S {S:.0f}")
