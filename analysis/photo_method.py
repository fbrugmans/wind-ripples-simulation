"""Measuring method on a field photograph ('The choice of ripples', Figure 4; descriptive; settings fixed on 15 Nov. 2026 before the coin throws).

Question: what does the ripple pattern in plan view (sinuous, branching crests) do to the field survey of 1983 (throw a coin, lay the
scale perpendicular to the nearest crest) compared with straight lines in a fixed direction, as the model measures?

Photograph: Hulshorster Zand, 21 Sept. 2026 17:03, from above; solar elevation 21.9 degrees, sun and wind from the left, tape measure at
the bottom (not included in the repository). No height: only the positions of crest and trough, so crest spacing L, lee length S,
stoss length I = L - S and SI = I / S. The HEIC photograph is converted with 'sips' (macOS).

Usage:  python3 photo_method.py <photo.HEIC> check   (only images to check crest and trough, no numbers)
        python3 photo_method.py <photo.HEIC>         (measurement; writes output/photo_method.txt)
"""
import os, sys, subprocess, tempfile
import numpy as np
from scipy import ndimage as ndi
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------- settings fixed in advance ----------------
G = 0.5                       # grid spacing of the scaled image (mm)
# scale: period of the millimetre marks of the tape measure (pixels along the image x), measured in windows of 800 pixels
TAPE_X = np.array([647, 1047, 1447, 1847, 2247, 2647, 3047, 3447, 3847, 4247, 4647.])
TAPE_P = np.array([5.121, 5.132, 5.2195, 5.264, 5.3375, 5.3885, 5.435, 5.476, 5.573, 5.5605, 5.5345])
TAPE_SLOPE = 0.1195           # slope of the tape measure in the image
SIG_CREST = 4.0               # smoothing for the crest line (mm)
CREST_HI, CREST_LO, CREST_MIN = 0.012, 0.006, 40.0   # thresholds of the brightness drop per grid point (strong, weak), minimum length of a crest line (mm)
                              # (first measurement 15 Nov.: 0.018 / 0.010; lowered after weak crests had been missed)
SIG_PROF = 2.0                # smoothing for the brightness profile (mm)
SUN = 21.9                    # solar elevation (degrees)
TAN_I = 0.05                  # slope of the stoss side (1983 field, tan i)
F_PHYS = 1.0 / (1.0 + TAN_I / np.tan(np.radians(SUN)))   # 0.89: first measurement; the trough then lies ~1 cm too far from the crest
F_TROUGH = 0.50               # trough halfway between darkest and lightest: on average 10 mm closer to the crest than at 0.89 (calibrated on the observer)
F_VAR = (0.40, F_TROUGH, 0.60)   # 13 and 8 mm closer to the crest than at 0.89
N_COINS, SEED = 2000, 1983
LINE_SPACING = 10.0           # distance between the fixed lines (mm)
SCALE_LEN = 500.0             # length of the measuring scale (mm)
L_MIN = 6.0                   # two crest crossings within this distance count as one (mm)


def read_photo(path):
    png = os.path.join(tempfile.gettempdir(), "photo_method.png")
    subprocess.run(["sips", "-s", "format", "png", path, "--out", png], check=True, capture_output=True)
    im = np.asarray(Image.open(png).convert("RGB")).astype(float) / 255
    lin = np.where(im <= 0.04045, im / 12.92, ((im + 0.055) / 1.055) ** 2.4)
    Y = 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]
    yellow = ((im[..., 0] - im[..., 2]) > 0.25).astype(float)
    return Y, yellow


def to_scale(Y, yellow):
    """Image to millimetres: the scale increases along the image x (camera not exactly perpendicular); taken equal across it."""
    h, w = Y.shape
    pf = np.polyfit(TAPE_X, TAPE_P / np.cos(np.arctan(TAPE_SLOPE)), 2)
    x = np.arange(w); mmpx = 1 / np.polyval(pf, x); X = np.cumsum(mmpx); X -= X[0]
    Xg = np.arange(0, X[-1], G); xg = np.interp(Xg, X, x); sg = np.interp(Xg, X, mmpx)
    Yg = np.arange(-(h / 2) * mmpx.max(), (h / 2) * mmpx.max(), G)
    yy = h / 2 + Yg[:, None] / sg[None, :]; xx = np.broadcast_to(xg[None, :], yy.shape)
    R = ndi.map_coordinates(ndi.gaussian_filter(Y, 1.1), [yy, xx], order=1, cval=np.nan)
    Rg = ndi.map_coordinates(ndi.gaussian_filter(yellow, 1.1), [yy, xx], order=1, cval=0)
    return R, Rg


def crest_lines(R):
    """Crest line = where the brightness drops fastest downwind (upwind edge of the dark crest lip)."""
    A = np.where(np.isnan(R), np.nanmedian(R), R)
    A = A / ndi.gaussian_filter(A, 150 / G)
    s = SIG_CREST / G
    gx = ndi.gaussian_filter(A, s, order=(0, 1)); gy = ndi.gaussian_filter(A, s, order=(1, 0))
    m = np.hypot(gx, gy); ux = gx / (m + 1e-12); uy = gy / (m + 1e-12)
    yy, xx = np.mgrid[0:A.shape[0], 0:A.shape[1]].astype(float)
    m1 = ndi.map_coordinates(m, [yy + uy, xx + ux], order=1); m2 = ndi.map_coordinates(m, [yy - uy, xx - ux], order=1)
    top = (m >= m1) & (m >= m2) & (gx < 0) & (np.abs(ux) > 0.5)
    lab, _ = ndi.label(top & (m > CREST_LO), structure=np.ones((3, 3)))
    keep = np.unique(lab[top & (m > CREST_HI)]); keep = keep[keep > 0]
    gr = ndi.sum(top & (m > CREST_LO), lab, keep); keep = keep[gr >= CREST_MIN / G]
    K = np.isin(lab, keep)
    return K, ndi.gaussian_filter(A, SIG_PROF / G), gx, gy


def area(R, Rg):
    """Measured area (mm): above the tape measure, without grass, straws, twig, small pit and footprints."""
    ny, nx = R.shape; Y, X = np.mgrid[0:ny, 0:nx] * G
    ys, xs = np.where(Rg > 0.5); a, b = np.polyfit(xs * G, ys * G, 1)          # axis of the tape measure
    upper = np.where(X < 400, 113, np.where(X < 613, 220, 60))
    M = (X > 45) & (X < 1026) & (Y > upper) & (Y < a * X + b - 35) & ~np.isnan(R)
    M &= ~((X > 853) & (X < 1000) & (Y > 347) & (Y < 507))                     # twig
    M &= ~((X > 946) & (Y > 573))                                              # case of the tape measure
    M &= np.hypot(X - 833, Y - 230) > 17                                       # small pit
    M &= np.hypot(X - 70, Y - 705) > 20                                        # footprint
    return M, np.array([1.0, a]) / np.hypot(1, a)                              # direction of the tape measure (unit vector x, y)


class Measurement:
    def __init__(self, R, Rg):
        self.R = R; self.K, self.A, self.gx, self.gy = crest_lines(R)
        self.M, self.tape = area(R, Rg)
        self.K &= self.M
        self.Kd = ndi.binary_dilation(self.K, structure=np.ones((3, 3)))
        self.dist, self.idx = ndi.distance_transform_edt(~self.K, return_indices=True)
        self.ky, self.kx = np.where(self.K)
        nrm = np.stack([-self.gx[self.K], -self.gy[self.K]], 1); nrm /= np.linalg.norm(nrm, axis=1)[:, None]
        self.knorm = nrm; self.wind = nrm.mean(0) / np.linalg.norm(nrm.mean(0))   # mean direction perpendicular to the crests

    def normal(self, qx, qy, radius=10.0):
        """Direction perpendicular to the crest at (qx, qy) in grid points, averaged over the crest within 10 mm; points downwind."""
        d = np.hypot(self.kx - qx, self.ky - qy) * G
        v = self.knorm[d <= radius].mean(0)
        return v / np.linalg.norm(v)

    def line(self, p, u, s0, s1):
        """Samples along the line p + s u (p in mm). Returns s, inside the area, crest hit, brightness."""
        s = np.arange(s0, s1 + G / 2, G); x = (p[0] + s * u[0]) / G; y = (p[1] + s * u[1]) / G
        ok = (x >= 0) & (y >= 0) & (x <= self.R.shape[1] - 1) & (y <= self.R.shape[0] - 1)
        xi = np.clip(np.rint(x).astype(int), 0, self.R.shape[1] - 1); yi = np.clip(np.rint(y).astype(int), 0, self.R.shape[0] - 1)
        inn = ok & self.M[yi, xi]; hit = inn & self.Kd[yi, xi]
        prof = ndi.map_coordinates(self.A, [y, x], order=1, mode="nearest")
        return s, inn, hit, prof

    @staticmethod
    def crossings(s, hit):
        """Positions (s) where the line crosses a crest line; crossings within L_MIN taken together."""
        i = np.where(hit)[0]
        if len(i) == 0: return np.array([])
        gr = np.split(i, np.where(np.diff(s[i]) > L_MIN)[0] + 1)
        return np.array([s[g].mean() for g in gr])

    @staticmethod
    def trough(s, prof, c0, c1, f):
        """Trough between the crests c0 and c1: first position after the darkest point where the brightness reaches the fraction f from darkest to lightest."""
        m = (s >= c0) & (s <= c1); ss, pp = s[m], prof[m]
        if len(ss) < 5: return np.nan
        n0 = max(2, int(0.5 * len(ss))); i0 = int(np.argmin(pp[:n0]))
        flat = np.percentile(pp[i0:], 90); level = pp[i0] + f * (flat - pp[i0])
        j = np.where(pp[i0:] >= level)[0]
        return ss[i0 + j[0]] if len(j) else np.nan

    def from_crest(self, q, u, f=F_TROUGH):
        """Measure from the crest at q (mm) along u downwind: L to the next crest, S to the trough. None if the line leaves the area earlier."""
        s, inn, hit, prof = self.line(q, u, -2.0, 400.0)
        out = np.where(~inn & (s > 0))[0]; smax = s[out[0]] if len(out) else s[-1]
        k = self.crossings(s, hit); k = k[(k > L_MIN) & (k < smax)]
        if len(k) == 0: return None
        L = k[0]; d = [self.trough(s, prof, 0.0, L, ff) for ff in F_VAR]
        return (L, *d)


def fixed_lines(mt, u):
    """All ripples along parallel lines in direction u, LINE_SPACING apart. Returns rows (L, S at F_VAR..., angle with the crest normal)."""
    n = np.array([-u[1], u[0]]); ny, nx = mt.R.shape; out = []
    for t in np.arange(-1200, 1200, LINE_SPACING):
        p = np.array([nx * G / 2, ny * G / 2]) + t * n
        s, inn, hit, prof = mt.line(p, u, -900, 900)
        if inn.sum() < 100: continue
        k = mt.crossings(s, hit)
        for c0, c1 in zip(k[:-1], k[1:]):
            m = (s >= c0) & (s <= c1)
            if not inn[m].all(): continue
            q = p + c0 * u; nq = mt.normal(q[0] / G, q[1] / G)
            angle = np.degrees(np.arccos(np.clip(abs(nq @ u), 0, 1)))
            out.append((c1 - c0, *[mt.trough(s, prof, c0, c1, f) - c0 for f in F_VAR], angle))
    return np.array(out)


def coins(mt, rng):
    ys, xs = np.where(mt.M); i = rng.integers(0, len(ys), N_COINS)
    return np.stack([xs[i] * G, ys[i] * G], 1)


def coin_fixed(mt, P, u):
    """Coin, but the scale in the fixed direction u: the crest nearest to the coin along the line."""
    out = []
    for p in P:
        s, inn, hit, prof = mt.line(p, u, -400, 400)
        k = mt.crossings(s, hit)
        if len(k) == 0: out.append(None); continue
        c = k[np.argmin(abs(k))]
        out.append(mt.from_crest(p + c * u, u))
    return out


def coin_perpendicular(mt, P, scale=False):
    """Field method: scale through the nearest point of a crest line, perpendicular to that crest (or, with scale, perpendicular to the
    mean of the crests under a scale of 50 cm with that point in the middle)."""
    out, angles = [], []
    for p in P:
        yi, xi = int(round(p[1] / G)), int(round(p[0] / G))
        qy, qx = mt.idx[0][yi, xi], mt.idx[1][yi, xi]; q = np.array([qx * G, qy * G])
        u = mt.normal(qx, qy)
        if scale:
            for _ in range(3):
                s, inn, hit, prof = mt.line(q, u, -SCALE_LEN / 2, SCALE_LEN / 2)
                k = mt.crossings(s, hit & inn)
                if len(k) < 2: break
                v = np.mean([mt.normal(*((q + c * u) / G)) for c in k], 0); u = v / np.linalg.norm(v)
        out.append(mt.from_crest(q, u)); angles.append(np.degrees(np.arctan2(u[1], u[0])))
    return out, np.array(angles)


def table(name, rows, f=1, rng=None):
    """rows: list of (L, S40, S50, S60) or None."""
    n0 = len(rows); a = np.array([r[:4] for r in rows if r is not None and not np.isnan(r[1 + f])], float)
    L, S = a[:, 0], a[:, 1 + f]; I = L - S; SI = I / S
    t = (f"{name:<46s} n {len(a):4d}/{n0:<4d} L {L.mean():6.1f} +- {L.std(ddof=1):5.1f} (med {np.median(L):5.1f})  "
         f"S {S.mean():5.1f} +- {S.std(ddof=1):4.1f}  I {I.mean():5.1f} +- {I.std(ddof=1):4.1f}  "
         f"SI per ripple {SI.mean():4.2f} +- {SI.std(ddof=1):4.2f} (med {np.median(SI):4.2f})  SI from the means {I.mean() / S.mean():4.2f}")
    if rng is not None:                                      # spread of a sample area of 25 throws
        g = np.array([a[rng.integers(0, len(a), 25)] for _ in range(2000)]); Lm = g[:, :, 0].mean(1); Sm = g[:, :, 1 + f].mean(1)
        t += (f"\n{'':<46s} sample area of 25: L {np.percentile(Lm, 2.5):5.1f}-{np.percentile(Lm, 97.5):5.1f}  "
              f"S {np.percentile(Sm, 2.5):4.1f}-{np.percentile(Sm, 97.5):4.1f}")
    return t, (L.mean(), S.mean(), I.mean(), SI.mean())


def image(mt, path, lines=()):
    v = np.clip(np.nan_to_num(mt.R) / np.nanpercentile(mt.R, 99.5), 0, 1) ** (1 / 2.2)
    fig, a = plt.subplots(figsize=(21.1, 17.2)); ny, nx = mt.R.shape
    a.imshow(v, cmap="gray", extent=(0, nx * G, ny * G, 0)); a.contour(np.arange(nx) * G, np.arange(ny) * G, mt.M, [0.5], colors="deepskyblue", linewidths=1)
    a.plot(mt.kx * G, mt.ky * G, ",", color="red")
    for p, u, c0, c1, d, col in lines:
        a.plot([p[0] + c0 * u[0], p[0] + c1 * u[0]], [p[1] + c0 * u[1], p[1] + c1 * u[1]], color=col, lw=1)
        if not np.isnan(d): a.plot(p[0] + d * u[0], p[1] + d * u[1], "o", color="yellow", ms=4, mec="k", mew=0.5)
    a.set_xlabel("mm"); a.set_ylabel("mm"); fig.tight_layout(); fig.savefig(path, dpi=75); plt.close(fig)


def main():
    photo = sys.argv[1]; check = len(sys.argv) > 2 and sys.argv[2] == "check"
    os.makedirs(os.path.join(HERE, "output"), exist_ok=True)
    R, Rg = to_scale(*read_photo(photo)); mt = Measurement(R, Rg)
    rng = np.random.default_rng(SEED)
    if check:                                       # crest and trough along a few lines parallel to the tape measure, to check by eye
        u = mt.tape; n = np.array([-u[1], u[0]]); ny, nx = R.shape; lines = []; fig, ax = plt.subplots(4, 1, figsize=(18, 11))
        for a, t in zip(ax, (-150, -50, 50, 150)):
            p = np.array([nx * G / 2, ny * G / 2]) + t * n; s, inn, hit, prof = mt.line(p, u, -700, 700); k = mt.crossings(s, hit)
            a.plot(s[inn], prof[inn], "k", lw=0.8)
            for c0, c1 in zip(k[:-1], k[1:]):
                if not inn[(s >= c0) & (s <= c1)].all(): continue
                d = mt.trough(s, prof, c0, c1, F_TROUGH); lines.append((p, u, c0, c1, d, "lime"))
                a.axvline(c0, color="r", lw=0.8); a.axvline(c1, color="r", lw=0.8)
                for f, col in zip(F_VAR, ("0.6", "b", "0.6")): a.axvline(mt.trough(s, prof, c0, c1, f), color=col, lw=0.8, ls="--")
            a.set_title(f"line parallel to the tape measure, {t:+d} mm across: red = crest, blue dashed = trough (grey: variants)", fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(HERE, "output", "photo_method_profile.png"), dpi=70)
        image(mt, os.path.join(HERE, "output", "photo_method_check.png"), lines)
        print("tape", mt.tape, "wind (mean crest normal)", mt.wind, "F_TROUGH %.3f" % F_TROUGH, "area m2 %.3f" % (mt.M.sum() * G * G / 1e6)); return

    out = []; w = out.append
    ht = np.degrees(np.arctan2(mt.tape[1], mt.tape[0])); hw = np.degrees(np.arctan2(mt.wind[1], mt.wind[0]))
    w("MEASURING METHOD ON A FIELD PHOTOGRAPH (descriptive), photograph %s" % os.path.basename(photo))
    w(f"measured area {mt.M.sum() * G * G / 1e6:.3f} m2; crest line {mt.K.sum() * G / 1000:.1f} m; direction of the tape measure {ht:+.1f} deg, mean direction perpendicular to the crests {hw:+.1f} deg (in the image)")
    w(f"trough: fraction {F_TROUGH:.2f} from darkest to lightest (variants {F_VAR[0]:.2f} and {F_VAR[2]:.2f}); crest thresholds {CREST_HI}/{CREST_LO}; coins {N_COINS}, seed {SEED}\n")
    A = fixed_lines(mt, mt.tape); A2 = fixed_lines(mt, mt.wind)
    P = coins(mt, rng); B = coin_fixed(mt, P, mt.tape); C, hC = coin_perpendicular(mt, P); D, hD = coin_perpendicular(mt, P, scale=True)
    sets = (("A  fixed lines along the tape measure, all ripples", [tuple(r[:4]) for r in A], None),
            ("A' fixed lines along the mean wind direction", [tuple(r[:4]) for r in A2], None),
            ("B  coin, scale along the tape measure", B, rng),
            ("C  coin, perpendicular to the nearest crest", C, rng),
            ("D  coin, perpendicular to crests under 50 cm scale", D, rng))
    for f, ff in enumerate(F_VAR):
        w(f"--- trough at fraction {ff:.2f}{'  (main result)' if f == 1 else ''} ---")
        for name, rows, r in sets: w(table(name, rows, f, r if f == 1 else None)[0])
        w("")
    w("angle between the fixed line and the direction perpendicular to the crest crossed (A): mean %.1f deg, median %.1f, 90th percentile %.1f; mean 1/cos %.3f"
      % (A[:, 4].mean(), np.median(A[:, 4]), np.percentile(A[:, 4], 90), (1 / np.cos(np.radians(A[:, 4]))).mean()))
    w("direction of the scale: C spread %.1f deg around %+.1f deg; D spread %.1f deg around %+.1f deg" % (hC.std(), hC.mean(), hD.std(), hD.mean()))
    # paired per coin: C against B
    par = [(b, c) for b, c in zip(B, C) if b is not None and c is not None and not np.isnan(b[2]) and not np.isnan(c[2])]
    b = np.array([x[0] for x in par]); c = np.array([x[1] for x in par])
    w("paired per coin (n %d): L perpendicular - L along the tape measure %+.1f mm (median %+.1f); S %+.1f (median %+.1f); share with shorter L perpendicular %.2f"
      % (len(par), (c[:, 0] - b[:, 0]).mean(), np.median(c[:, 0] - b[:, 0]), (c[:, 2] - b[:, 2]).mean(), np.median(c[:, 2] - b[:, 2]), (c[:, 0] < b[:, 0] - 1).mean()))
    for name, rows, _ in sets:
        Lr = np.array([r[0] for r in rows if r is not None]); w(f"{name:<46s} share L > 170 mm (above the 1983 field range): {(Lr > 170).mean():.3f}; L < 58 mm: {(Lr < 58).mean():.3f}")
    text = "\n".join(out); print(text)
    open(os.path.join(HERE, "output", "photo_method.txt"), "w").write(text + "\n")
    v = np.clip(np.nan_to_num(mt.R) / np.nanpercentile(mt.R, 99.5), 0, 1) ** (1 / 2.2); ny, nx = mt.R.shape
    for name, (xa, xb) in (("left", (40, 560)), ("right", (520, 1040))):
        fig, a = plt.subplots(figsize=(20, 26)); a.imshow(v, cmap="gray", extent=(0, nx * G, ny * G, 0)); a.plot(mt.kx * G, mt.ky * G, ".", color="red", ms=1.5)
        for p, r in list(zip(P, C))[:60]:
            if r is None: continue
            yi, xi = int(round(p[1] / G)), int(round(p[0] / G)); q = np.array([mt.idx[1][yi, xi] * G, mt.idx[0][yi, xi] * G]); u = mt.normal(q[0] / G, q[1] / G)
            e = q + r[0] * u; d = q + r[2] * u
            a.plot([q[0], e[0]], [q[1], e[1]], color="lime", lw=1.5); a.plot(p[0], p[1], "x", color="cyan", ms=9, mew=2)
            a.plot([q[0], e[0]], [q[1], e[1]], "s", color="red", ms=8, mec="w", mew=1)
            if not np.isnan(r[2]): a.plot(d[0], d[1], "o", color="yellow", ms=13, mec="k", mew=1.5)
        a.set_xlim(xa, xb); a.set_ylim(720, 50); a.set_xlabel("mm"); a.set_ylabel("mm")
        a.set_title("First 60 coin throws, scale perpendicular to the nearest crest (wind from the left). Blue cross = coin; red squares = crest and next crest; yellow dot = trough", fontsize=12)
        fig.tight_layout(); fig.savefig(os.path.join(HERE, "output", f"photo_method_dots_{name}.png"), dpi=70); plt.close(fig)


if __name__ == "__main__":
    main()
