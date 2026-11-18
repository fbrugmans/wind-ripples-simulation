"""Figure 4: test of the measuring method on a field photograph (Hulshorster Zand, 21 Sept. 2026; photo_method.py).

(a) the scaled photograph with the drawn crest lines, a number of reproduced field measurements (coin, 50 cm scale perpendicular to the
crests) and two fixed lines; (b) the brightness along a fixed line with crest and trough.
Usage:  python3 fig4_photo_test.py <photo.HEIC>   ->  fig4_photo_test.png next to this script
"""
import os, sys
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import photo_method as m
G = m.G; plt.rcParams.update({"font.size": 14})
RED, YELLOW, GREEN, BLUE = "#e5352b", "#ffd60a", "#3ddc5a", "#3aa0ff"
R, Rg = m.to_scale(*m.read_photo(sys.argv[1])); mt = m.Measurement(R, Rg)
P = m.coins(mt, np.random.default_rng(m.SEED)); D, _ = m.coin_perpendicular(mt, P, scale=True)
v = np.clip(np.nan_to_num(R) / np.nanpercentile(R, 99.5), 0, 1) ** (1 / 2.2); ny, nx = R.shape
fig, (a, b) = plt.subplots(2, 1, figsize=(14, 14.2), gridspec_kw=dict(height_ratios=[3.1, 1]))
a.imshow(v, cmap="gray", extent=(0, nx * G, ny * G, 0), vmin=0.15, vmax=1.0)
a.plot(mt.kx * G, mt.ky * G, ".", color=RED, ms=1.6)
# reproduced field measurements: the first throws that lie at least 85 mm apart
chosen = []
for p, r in zip(P, D):
    if r is None or np.isnan(r[2]) or (p[1] > 640 and p[0] < 260) or any(np.hypot(*(p - q)) < 85 for q in chosen): continue
    chosen.append(p)
    yi, xi = int(round(p[1] / G)), int(round(p[0] / G)); q = np.array([mt.idx[1][yi, xi] * G, mt.idx[0][yi, xi] * G]); u = mt.normal(q[0] / G, q[1] / G)
    # same direction as in the measurement: perpendicular to the crests under the 50 cm scale
    for _ in range(3):
        s, inn, hit, prof = mt.line(q, u, -m.SCALE_LEN / 2, m.SCALE_LEN / 2); k = mt.crossings(s, hit & inn)
        if len(k) < 2: break
        wv = np.mean([mt.normal(*((q + c * u) / G)) for c in k], 0); u = wv / np.linalg.norm(wv)
    e = q + r[0] * u; d = q + r[2] * u
    a.plot([q[0], e[0]], [q[1], e[1]], color=GREEN, lw=2.2); a.plot(p[0], p[1], "x", color="cyan", ms=10, mew=2.4)
    a.plot([q[0], e[0]], [q[1], e[1]], "s", color=RED, ms=7, mec="w", mew=1); a.plot(d[0], d[1], "o", color=YELLOW, ms=10, mec="k", mew=1.2)
    if len(chosen) >= 22: break
# two fixed lines along the tape measure
u = mt.tape; n = np.array([-u[1], u[0]]); c0 = np.array([nx * G / 2, ny * G / 2]); prof_line = None
for t in (-170, 40):
    p = c0 + t * n; s, inn, hit, prof = mt.line(p, u, -700, 700); k = mt.crossings(s, hit)
    a.plot(p[0] + s[inn] * u[0], p[1] + s[inn] * u[1], color=BLUE, lw=1.6, ls=(0, (5, 3)))
    if t == 40: prof_line = (p, s, inn, prof, k)
a.set_xlim(45, 1030); a.set_ylim(745, 55); a.set_xticks([]); a.set_yticks([])
a.plot([70, 170], [722, 722], color="w", lw=5, solid_capstyle="butt"); a.text(120, 714, "100 mm", color="w", ha="center", va="bottom", fontsize=14, fontweight="bold")
a.annotate("", (340, 82), (230, 82), arrowprops=dict(arrowstyle="-|>", color="w", lw=3)); a.text(222, 82, "wind and sun", color="w", ha="right", va="center", fontsize=14, fontweight="bold")
h = [Line2D([], [], color=RED, lw=2, label="crest line"), Line2D([], [], marker="x", color="cyan", ls="", ms=10, mew=2.4, label="coin"),
     Line2D([], [], color=GREEN, lw=2.2, marker="s", mfc=RED, mec="w", ms=7, label="scale from crest to crest"),
     Line2D([], [], marker="o", color=YELLOW, mec="k", ls="", ms=10, label="trough"), Line2D([], [], color=BLUE, lw=1.6, ls=(0, (5, 3)), label="fixed line")]
a.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 1.065), ncol=5, frameon=False, fontsize=13, handletextpad=0.5, columnspacing=1.6)
a.text(0.008, 0.985, "(a)", transform=a.transAxes, color="w", fontsize=16, fontweight="bold", va="top")
# (b) brightness along the lowest fixed line
p, s, inn, prof, k = prof_line; s0 = s[inn][0]
b.plot(s - s0, np.where(inn, prof, np.nan), color="#0b0b0b", lw=1.4)
first = True
for ka, kb in zip(k[:-1], k[1:]):
    if not inn[(s >= ka) & (s <= kb)].all(): continue
    d = mt.trough(s, prof, ka, kb, m.F_TROUGH)
    for kk in (ka, kb): b.axvline(kk - s0, color=RED, lw=1.4, label="crest" if first and kk == ka else None)
    if not np.isnan(d): b.axvline(d - s0, color="#c79a00", lw=1.4, ls="--", label="trough" if first else None)
    first = False
b.set_xlim(0, s[inn][-1] - s0); b.set_xlabel("distance along the lowest fixed line (mm), downwind"); b.set_ylabel("brightness\n(mean = 1)")
b.legend(loc="upper right", ncol=2, frameon=True, fontsize=12); b.text(0.008, 0.95, "(b)", transform=b.transAxes, fontsize=16, fontweight="bold", va="top")
for sp in ("top", "right"): b.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig4_photo_test.png"), dpi=130, facecolor="white")
print(len(chosen), "measurements drawn")
