"""Does the impact model hold on to a regular first ripplet? (Discussion, 'The observations of 1983 reconsidered'; fixed in advance on 14 Nov. 2026.)

Runs (article model, seeds 271-274): u* 0.31 and 0.50; RN flat bed, RA ripplet of 35.1 mm (mode 61), RB ripplet of 101.9 mm (mode 21);
K = 120 s with a record every 1 s, L = 1800 s with a record every 5 s.
Measures, paired per seed with the run without a ripplet:
  (1) strength  = amplitude (mm) of the Fourier component with the imposed wavelength in the raw bed profile;
  (2) position  = shift of that component along the x-axis (mm; phase unwrapped in time; + = downwind);
  (3) regularity = spread (sd / mean) of the distance between crests with prominence >= 0.2 mm in the smoothed profile (Gauss sigma 3 mm, trend > 400 mm removed);
  (4) crest spacing = mean distance between those crests; peak of the smoothed spectrum (4-400 mm, 9 modes);
  (5) rms height of the smoothed profile (mm).
Criteria R1-R4 (RA31, RA50) and S1 (RB50) are printed below. Usage (from the folder with the runs):  python3 imposed_ripplets.py
"""
import os
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
SEEDS = (271, 272, 273, 274); TK = (1, 2, 3, 5, 10, 20, 30, 60, 120); TL = (300, 600, 1200, 1800); MODE = {"RA": 61, "RB": 21}


def smooth(z, dx):
    z = z - z.mean(); return gaussian_filter1d(z, 3 / dx, mode="wrap") - gaussian_filter1d(z, 400 / 2.355 / dx, mode="wrap")


def comp(z, n):
    F = np.fft.rfft(z - z.mean()); return 2 * abs(F[n]) / len(z), float(np.angle(F[n]))


def crests(z, dx):
    g = smooth(z, dx); kp, _ = find_peaks(g, prominence=0.2); d = np.diff(kp) * dx
    return (len(kp), float(d.mean()) if len(d) > 2 else np.nan, float(d.std() / d.mean()) if len(d) > 2 else np.nan, float(g.std()))


def peak(z, dx, kk=9):
    N = len(z); lam = N * dx / np.arange(1, N // 2 + 1); sel = (lam > 4) & (lam < 400); F = np.abs(np.fft.rfft(z - z.mean())[1:N // 2 + 1])
    return float(lam[sel][np.argmax(np.exp(np.convolve(np.log(F + 1e-12), np.ones(kk) / kk, "same"))[sel])])


class Run:
    """Short and long run of a variant and seed joined: profiles (mm) at each record time; t = 0 is the initial bed h0."""
    def __init__(self, name, z):
        self.ok = all(os.path.exists(f"c23_z{z}_{name}{s}.npz") for s in "KL")
        if not self.ok: return
        k = np.load(f"c23_z{z}_{name}K.npz"); l = np.load(f"c23_z{z}_{name}L.npz"); self.dx = float(k["dx"]); dz = float(k["dz"])
        il = np.where(l["t"] > k["t"][-1] + 1e-9)[0]
        self.t = np.concatenate(([0.0], k["t"], l["t"][il])); self.h = np.concatenate((k["h0"][None], k["h"], l["h"][il])).astype(float) * dz
        sam = [(int(np.argmin(np.abs(k["t"] - t))), int(np.argmin(np.abs(l["t"] - t)))) for t in range(5, 121, 5)]
        self.prefix = bool(np.array_equal(k["h0"], l["h0"]) and all(np.array_equal(k["h"][a], l["h"][b]) for a, b in sam)); self.N = self.h.shape[1]

    def at(self, t):
        i = int(np.argmin(np.abs(self.t - t))); assert abs(self.t[i] - t) < 0.5, (t, self.t[i]); return self.h[i]

    def track(self, n):
        """Strength and unwrapped shift (mm) of mode n at all record times."""
        c = [comp(h, n) for h in self.h]; A = np.array([x[0] for x in c]); ph = np.unwrap(np.array([x[1] for x in c]))
        return A, (ph[0] - ph) / (2 * np.pi) * self.N * self.dx / n


def row(v, f="{:.3f}"): return " / ".join("  nan" if not np.isfinite(x) else f.format(x) for x in v)


def count(b): return int(np.sum(b))


def block(u):
    R = {v: [Run(f"{v}{u}", z) for z in SEEDS] for v in ("RN", "RA", "RB")}
    if not all(r.ok for v in R for r in R[v]): print(f"\nu* 0.{u}: runs missing"); return
    print(f"\n================ u* 0.{u} m/s ================")
    print("short run = start of the long run (h0 and h at 5, 10, ..., 120 s): " + ", ".join(f"{v} {count([r.prefix for r in R[v]])}/4" for v in R))
    dx = R["RN"][0].dx
    for v, lab in (("RA", "ripplet 35.1 mm (mode 61)"), ("RB", "ripplet 101.9 mm (mode 21)")):
        n = MODE[v]; S = [r.track(n) for r in R[v]]; S0 = [r.track(n) for r in R["RN"]]
        print(f"\n{v}{u}: {lab}; per seed 271 / 272 / 273 / 274")
        print("    t(s)   strength with ripplet (mm)        same mode without ripplet (mm)     shift (mm, + downwind)")
        for t in (0,) + TK + TL:
            i = [int(np.argmin(np.abs(r.t - t))) for r in R[v]]; j = [int(np.argmin(np.abs(r.t - t))) for r in R["RN"]]
            print(f"   {t:5d}   {row([s[0][k] for s, k in zip(S, i)])}     {row([s[0][k] for s, k in zip(S0, j)])}     {row([s[1][k] for s, k in zip(S, i)], '{:+.1f}')}")
        for a, b in ((30, 120), (300, 1800)):
            sn = []
            for r, s in zip(R[v], S):
                ia, ib = int(np.argmin(np.abs(r.t - a))), int(np.argmin(np.abs(r.t - b))); sn.append((s[1][ib] - s[1][ia]) / (b - a) * 60)
            print(f"   speed of the ripplet {a}-{b} s (mm/min): {row(sn, '{:+.1f}')}")
    print(f"\nRegularity and size (smoothed profile), per seed 271 / 272 / 273 / 274")
    print("    t(s)  variant   spread of crest spacing       mean crest spacing (mm)     number of crests    spectrum peak (mm)        rms height (mm)")
    K = {}
    for t in (5, 10, 20, 30, 60, 120) + TL:
        for v in ("RN", "RA", "RB"):
            k = [crests(r.at(t), dx) for r in R[v]]; p = [peak(r.at(t), dx) for r in R[v]]; K[(v, t)] = (k, p)
            print(f"   {t:5d}  {v}{u}      {row([x[2] for x in k], '{:.2f}')}       {row([x[1] for x in k], '{:5.1f}')}     {row([x[0] for x in k], '{:3.0f}')}     {row(p, '{:5.0f}')}     {row([x[3] for x in k], '{:.2f}')}")
    # criteria
    n = MODE["RA"]; S = [r.track(n) for r in R["RA"]]; S0 = [r.track(n) for r in R["RN"]]
    def A(s, r, t): return s[0][int(np.argmin(np.abs(r.t - t)))]
    a0 = [A(s, r, 0) for s, r in zip(S, R["RA"])]; a60 = [A(s, r, 60) for s, r in zip(S, R["RA"])]; a120 = [A(s, r, 120) for s, r in zip(S, R["RA"])]
    u60 = [A(s, r, 60) for s, r in zip(S0, R["RN"])]
    r1 = [x60 >= 0.5 * x0 and x60 >= 3 * y for x0, x60, y in zip(a0, a60, u60)]; r2 = [x120 > x0 for x0, x120 in zip(a0, a120)]
    sp = lambda v, t: np.array([x[2] for x in K[(v, t)][0]])
    r3a = (sp("RA", 60) <= 0.25) & (sp("RA", 120) <= 0.25); r3b = (sp("RA", 60) < sp("RN", 60)) & (sp("RA", 120) < sp("RN", 120))
    r4 = (sp("RA", 600) < sp("RN", 600)) & (sp("RA", 1800) < sp("RN", 1800))
    print(f"\nCRITERIA RA{u} (ripplet 35.1 mm), u* 0.{u}")
    print(f"   R1 the start persists (60 s: >= 0.5 x initial and >= 3 x without ripplet): {count(r1)}/4 seeds -> {'MET' if count(r1) >= 3 else 'NOT MET'}"
          f"   [ratio 60 s / initial: {row(np.array(a60) / np.array(a0), '{:.2f}')}; with / without ripplet at 60 s: {row(np.array(a60) / np.array(u60), '{:.1f}')}]")
    print(f"   R2 impacts build it up (120 s > initial): {count(r2)}/4 seeds -> {'MET' if count(r2) >= 3 else 'NOT MET'}   [ratio 120 s / initial: {row(np.array(a120) / np.array(a0), '{:.2f}')}]")
    print(f"   R3 regular (spread <= 0.25 at 60 and 120 s in >= 3/4: {count(r3a)}/4; smaller than without ripplet in 4/4: {count(r3b)}/4) -> {'MET' if count(r3a) >= 3 and count(r3b) == 4 else 'NOT MET'}")
    print(f"   R4 regularity persists (spread at 600 and 1800 s smaller than without ripplet): {count(r4)}/4 seeds -> {'MET' if count(r4) == 4 else 'NOT MET'}"
          f"   [difference 600 s: {row(sp('RA', 600) - sp('RN', 600), '{:+.2f}')}; 1800 s: {row(sp('RA', 1800) - sp('RN', 1800), '{:+.2f}')}]")
    if u == "50":
        pk = np.array([[peak(r.at(t), dx) for t in (60, 300, 600, 1800)] for r in R["RB"]]); rm = np.array([[crests(r.at(t), dx)[3] for t in (60, 300, 1800)] for r in R["RB"]])
        s1a = np.all((pk >= 85) & (pk <= 120), 1); s1b = (rm[:, 2] > rm[:, 1]) & (rm[:, 1] > rm[:, 0])
        print(f"CRITERION S1 RB50 (strong wind, ripplet 101.9 mm): peak 85-120 mm at 60, 300, 600, 1800 s: {count(s1a)}/4 seeds; rms height 1800 > 300 > 60 s: {count(s1b)}/4 seeds"
              f" -> {'MET' if count(s1a) >= 3 and count(s1b) >= 3 else 'NOT MET'}")
        print("   peak per seed (60 / 300 / 600 / 1800 s): " + " | ".join(row(x, "{:.0f}") for x in pk))
        pn = np.array([[peak(r.at(t), dx) for t in (60, 300, 600, 1800)] for r in R["RN"]])
        print("   for comparison RN50 (flat bed), peak per seed: " + " | ".join(row(x, "{:.0f}") for x in pn))


if __name__ == "__main__":
    print("Does the impact model hold on to a regular first ripplet? (fixed in advance on 14 Nov. 2026)")
    for u in ("31", "50"): block(u)
