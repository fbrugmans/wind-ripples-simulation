"""Figure 5: the bed of the model at true scale (vertically not exaggerated).

Top, NOT selected: the article model (SR0, four seeds) after 4 hours and seed 271 after 16 hours (SL16), each the first 300 mm of the bed.
Below, SELECTED by eye: per seed after 4 hours one stretch of 140 mm with a clear ripple, chosen from overviews of the whole bed at true scale.
All panels at the same scale; profile smoothed as for all measures (measure.smooth). At the bottom for comparison the mean ripple of the
1983 field (L 107, H 5, I 80, S 27 mm; shape of Figure 1 of 1983), repeated: schematic, not a measurement.
Usage (from the folder with the runs):  python3 fig5_true_scale.py [folder with the runs]  ->  fig5_true_scale.png next to this script
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import measure as M
from template_fit import template

FOLDER = sys.argv[1] if len(sys.argv) > 1 else "."   # folder with the runs (default: current folder)
DX, XV, XK, ZR = 0.4, 300.0, 140.0, 6.0
SAND, EDGE, FIELD = "#e8d9b0", "#7a5c1e", "#b5651d"
CHOSEN = [(274, 55), (272, 1115), (273, 1050), (271, 1450)]   # (seed, start in mm), chosen by eye


def profile(f, k=-1):
    d = np.load(f); h = M.smooth(d["h"][k] * float(d["dz"])); return h - h.mean(), float(d["t"][k])


W, H_FIG = 12.0, 9.4; L0, WIDTH = 0.07, 0.91                    # full width = 300 mm
s = WIDTH * W / XV                                               # inch per mm, the same for all panels
ah = 2 * ZR * s / H_FIG; aw_k = XK * s / W; gap = WIDTH - 2 * aw_k
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig = plt.figure(figsize=(W, H_FIG))


def panel(rect, x, z, title, col=EDGE, fill=SAND, x0=0.0, width=XV, tcol="black"):
    ax = fig.add_axes(rect); ax.fill_between(x, z, -ZR, color=fill, lw=0); ax.plot(x, z, color=col, lw=1.0)
    ax.set_xlim(x0, x0 + width); ax.set_ylim(-ZR, ZR); ax.set_yticks([-5, 0, 5]); ax.set_ylabel("mm", fontsize=9); ax.tick_params(labelsize=9)
    ax.set_title(title, loc="left", fontsize=10.5, color=tcol); return ax


y = 1.0 - 0.055; step = ah + 0.052
fig.text(L0, y + 0.02, "Not selected: the first 300 mm of each bed", fontsize=12, weight="bold")
rows = [(os.path.join(FOLDER, f"c23_z{z}_SR0.npz"), f"seed {z}") for z in (271, 272, 273, 274)] + [(os.path.join(FOLDER, "c23_z271_SL16.npz"), "seed 271")]
n = int(XV / DX) + 1; x = np.arange(n) * DX
for i, (f, lab) in enumerate(rows):
    h, t = profile(f); y -= step
    ax = panel([L0, y, WIDTH, ah], x, h[:n], f"({chr(97 + i)}) {lab}, after {t / 3600:.0f} hours")
    if i == 0:
        ax.annotate("", xy=(297, 9.5), xytext=(267, 9.5), arrowprops=dict(arrowstyle="-|>", color="0.3", lw=1.2), annotation_clip=False)
        ax.text(264, 9.5, "wind", ha="right", va="center", fontsize=10, color="0.3")
y -= 0.045; fig.text(L0, y, "Selected by eye: stretches of 140 mm with a clear ripple, after 4 hours", fontsize=12, weight="bold")
for j, (z, a) in enumerate(CHOSEN):
    h, t = profile(os.path.join(FOLDER, f"c23_z{z}_SR0.npz")); xx = np.arange(len(h)) * DX; m = (xx >= a) & (xx <= a + XK)
    if j % 2 == 0: y -= step
    panel([L0 + (j % 2) * (aw_k + gap), y, aw_k, ah], xx[m], h[m], f"({chr(102 + j)}) seed {z}, {a}–{a + XK:.0f} mm", x0=a, width=XK)
y -= step + 0.03
nI, nS = int(round(80 / DX)), int(round(27 / DX)); g = 5.0 * template(nI, nS)[:-1]; zf = np.tile(g, int(np.ceil(n / len(g))) + 1)[:n]; zf = zf - zf.mean()
ax = panel([L0, y, WIDTH, ah], x, zf, "(j) For comparison: the mean ripple of the 1983 field (L 107, H 5 mm, symmetry index 3), repeated; schematic",
           col=FIELD, fill="#f3e3d3", tcol=FIELD)
ax.set_xlabel("distance along the bed (mm); all panels at the same scale 1 : 1, vertically not exaggerated")
out = os.path.join(HERE, "fig5_true_scale.png"); fig.savefig(out, dpi=200, bbox_inches="tight"); print("written:", out)
