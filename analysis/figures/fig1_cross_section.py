"""Figure 1 (sketch): cross-section of wind ripples with the measures used, after the scheme in Brugmans (1983).

Field values: crest spacing = ripple length 107 mm, ripple height 5 mm, stoss length 80 mm, lee length 27 mm; vertical exaggeration 5.
Stoss side straight at the bottom and slightly convex at the top, lee side steep at the top and slightly concave at the bottom. Crest lip
(short, steep, straight, coarse edge of the crest on the lee side) as a thick dark line. Not a model result.
Usage:  python3 fig1_cross_section.py   ->  fig1_cross_section.png next to this script
"""
import os
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
INK, INK2, BLUE, ORANGE = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 12})
L, H, I = 107.0, 5.0, 80.0; S = L - I; EX = 5


def profile(x):
    xm = np.mod(x, L)
    s = np.clip(xm / I, 0, 1); stoss = H * (s + 0.3 * s**2 * (1 - s))
    t = np.clip((xm - I) / S, 0, 1); lee = H * (1 - t) ** 1.4
    return np.where(xm <= I, stoss, lee)


fig, a = plt.subplots(figsize=(12.5, 5.6))
x = np.linspace(-14, 2 * L + 12, 3000)
a.fill_between(x, profile(x), -1.0, color="#e8d9b0", lw=0); a.plot(x, profile(x), color="#7a5c1e", lw=1.8)


def dim(x0, x1, z, text, colour=INK, dz=0.45):
    a.annotate("", (x1, z), (x0, z), arrowprops=dict(arrowstyle="<->", color=colour, lw=1.2, shrinkA=0, shrinkB=0))
    a.text((x0 + x1) / 2, z + dz, text, ha="center", va="bottom", color=colour, fontsize=12)


def guide(xp, z0, z1):
    a.plot([xp, xp], [z0, z1], color=INK2, lw=0.7, ls=":")


k1, k2, d1, d2 = I, I + L, 0.0, L
# crest spacing at the top, from crest to crest
for k in (k1, k2): guide(k, H, 9.6)
dim(k1, k2, 9.2, "crest spacing (crest to crest)", BLUE)
# ripple length below, from trough to trough; stoss and lee length below that
for d in (d1, d2): guide(d, 0, -5.2)
guide(k1, H, -3.2)
dim(d1, d2, -2.6, "ripple length (trough to trough)", ORANGE, dz=-1.55)
dim(d1, k1, -5.0, "stoss length", INK, dz=-1.55)
dim(k1, d2, -5.0, "lee length", INK, dz=-1.55)
# ripple height along the z-axis
xh = 2 * L + 18
a.plot([2 * L, xh + 3], [0, 0], color=INK2, lw=0.7, ls=":"); a.plot([k2, xh + 3], [H, H], color=INK2, lw=0.7, ls=":")
a.annotate("", (xh, H), (xh, 0), arrowprops=dict(arrowstyle="<->", color=INK, lw=1.2, shrinkA=0, shrinkB=0))
a.text(xh + 2, H / 2, "ripple\nheight\n(along z-axis)", va="center", ha="left", fontsize=12)
# slope on the lower stoss side: height difference along the z-axis per distance along the x-axis
xa, xb = 12.0, 36.0; za, zb = profile(np.array([xa, xb]))
a.plot([xa, xb, xb], [za, za, zb], color=INK2, lw=1.0)
a.text((xa + xb) / 2, za - 0.3, "distance along x-axis", ha="center", va="top", fontsize=10, color=INK2)
a.text(xb + 1, (za + zb) / 2, "height difference\nalong z-axis", ha="left", va="center", fontsize=10, color=INK2)
a.text(xa - 10, 4.7, "slope = height difference / distance", fontsize=10, color=INK2, va="bottom")
# names
a.text(k1, H + 0.35, "crest", ha="center", va="bottom", fontsize=12)
a.text(k2, H + 0.35, "crest", ha="center", va="bottom", fontsize=12)
a.text(d2, -0.35, "trough", ha="center", va="top", fontsize=12); a.text(d1, -0.35, "trough", ha="center", va="top", fontsize=12)
a.text(d2 + 40, 4.4, "stoss side", ha="center", fontsize=13, style="italic", color="#7a5c1e")
# crest lip: short, steep, straight, coarse edge of the crest on the lee side
xl = np.linspace(k1, k1 + 4, 40); a.plot(xl, profile(xl), color="#3d2a08", lw=4.5, solid_capstyle="butt")
a.annotate("crest lip", (k1 + 2.5, profile(np.array([k1 + 2.5]))[0]), (k1 + 9, 8.0), fontsize=12, ha="left", va="center",
           arrowprops=dict(arrowstyle="->", color=INK, lw=1.0, shrinkA=2, shrinkB=2))
a.text(k1 + 17, 4.3, "lee side", ha="center", fontsize=13, style="italic", color="#7a5c1e")
# wind and axes
a.annotate("wind", (24, 11.3), (-6, 11.3), arrowprops=dict(arrowstyle="->", color=INK, lw=1.6), fontsize=13, va="center")
a.annotate("", (12, -6.2), (-10, -6.2), arrowprops=dict(arrowstyle="->", color=INK, lw=1.2)); a.text(13, -6.2, "x", va="center", fontsize=11)
a.annotate("", (-10, -2.0), (-10, -6.2), arrowprops=dict(arrowstyle="->", color=INK, lw=1.2)); a.text(-10, -1.7, "z", ha="center", va="bottom", fontsize=11)
a.set_xlim(-16, 2 * L + 42); a.set_ylim(-7, 12.4); a.set_aspect(EX); a.axis("off")
a.text(2 * L + 42, -7, f"vertical exaggeration {EX} ×", ha="right", va="bottom", fontsize=10, color=INK2)
fig.savefig(os.path.join(HERE, "fig1_cross_section.png"), dpi=200, bbox_inches="tight", facecolor="white")
print("fig1_cross_section.png")
