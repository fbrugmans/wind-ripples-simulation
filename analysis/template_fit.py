"""TEMPLATE FIT ('Recognising the ripple shape', Figures 6 and 7; settings fixed in advance on 16 Nov. 2026; descriptive, no criteria).
The typical ripple shape (Figure 1 of Brugmans 1983, stoss and lee sides each normalised separately) is fitted to the model profile step by step
by stretching (I, S, H) and shifting (position, level) with least squares. Each time the fit is chosen that most lowers the sum of squares of the
whole profile, without overlap with ripples chosen earlier (troughs may be shared), until no fit with H > 0 fits any more. SI = I / S comes from
the fit and is thus not imposed. Per ripple R^2 = share of the variation within the window trough-crest-trough explained by the template.
Check (option 1): the existing measuring rule (prominence 3 mm, H >= 2.5) on the same records; matched if the crests lie within 10 mm.
Noise reference: each profile with random phases (same spectrum, no shape).
Three ways of laying the template: free (fit_free), fixed windows (fit_fixed), chain (fit_chain); mirrored template (fit_mirror).
Usage (from the folder with the runs SR0 and SL16; fits are written to analysis/output):
  python3 template_fit.py test_free | test_fixed | test_chain | test_mirror     artificial profiles: I, S, H must be recovered
  python3 template_fit.py fit_free . ; fit_fixed . ; fit_chain . ; fit_mirror .   fits (fit_mirror needs the free and chain fits)
  python3 template_fit.py table_free | table_fixed | table_chain | table_mirror  tables -> output/template*.txt
  python3 template_fit.py fig6 . ; fig7                                           Figures 6 and 7 -> figures/"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
from scipy.interpolate import UnivariateSpline
import measure as M, ripple_measure as RM

DX = 0.4
I_MM = np.arange(10, 251, 2.0); S_MM = np.arange(4, 151, 2.0)
SEEDS = (271, 272, 273, 274)
RCUT = (0.0, 0.5, 0.7, 0.8, 0.9)
SHAPE = int(os.environ.get("SJ_VORM", "2"))   # 1: first template (one trough level, lee side of 1983); 2: two trough levels + lee tail (adjusted template, 16 Nov.; used in the article)
QS = (1.0,) if SHAPE == 1 else (1.0, 0.75, 0.5, 0.35)
SUF = "" if SHAPE == 1 else "2"
OUT_FREE = os.path.join(HERE, "output", f"template{SUF}_ripples.npz")
CHOICE = os.environ.get("SJ_KEUZE", "R2")   # R2 (default; deviation from the advance registration after the code test); E: decrease of the sum of squares (registered); En: per cell


# ------------------------------------------------------------------ template (Figure 1 of 1983)
def _shape():
    x0 = 22.0
    def P(pts): a = np.array(pts, float); return a[:, 0] - x0, M.F1_BASE - a[:, 1]
    xl, yl = P(M.F1_STOSS); xj, yj = P(M.F1_LEE)
    w = np.ones(len(xl)); w[[0, -1]] = 30; loef = UnivariateSpline(xl, yl, w=w, k=3, s=len(xl) * 0.7)
    w = np.ones(len(xj)); w[[0, -1]] = 30; lij = UnivariateSpline(xj, yj, w=w, k=3, s=len(xj) * 0.5)
    xk, xe = xl[-1], xj[-1]; u = np.linspace(0, 1, 2001)
    gl = loef(u * xk); gl = (gl - gl[0]) / (gl[-1] - gl[0])
    gj = lij(xk + u * (xe - xk)); gj = (gj - gj[-1]) / (gj[0] - gj[-1])
    return u, np.clip(gl, 0, 1.2), np.clip(gj, 0, 1.2)
U, GL, GJ = _shape()


def template(nI, nS, q=1.0):
    """Template on the grid: points 0..nI+nS (both troughs included), height 0 in the troughs and 1 at the crest. Lee side g_j(v^q): q < 1 = longer gentle tail."""
    k = np.arange(nI + nS + 1); v = np.clip((k - nI) / nS, 0, 1)
    return np.where(k <= nI, np.interp(k / nI, U, GL), np.interp(v ** q if q != 1 else v, U, GJ))


TPL = None
def _init():
    global TPL
    if TPL is not None: return
    nI = np.rint(I_MM / DX).astype(int); nS = np.rint(S_MM / DX).astype(int)
    TPL = [(a, b, template(a, b)) for a in nI for b in nS]


# ------------------------------------------------------------------ fit
def fit(z, ruis_fasen=None):
    """Stepwise fit on a periodic profile z (mm). Returns a list (x0 cell, nI, nS, H, z0, R2, E[, q, b])."""
    if SHAPE == 2: return fit2(z, ruis_fasen)
    _init(); N = len(z); z = np.asarray(z, float)
    if ruis_fasen is not None:
        Zf = np.fft.rfft(z); ph = np.exp(1j * ruis_fasen[:len(Zf)]); ph[0] = 1
        if N % 2 == 0: ph[-1] = 1
        z = np.fft.irfft(np.abs(Zf) * ph, N)
    zz = np.concatenate([z, z]); c1 = np.concatenate([[0], np.cumsum(zz)]); c2 = np.concatenate([[0], np.cumsum(zz * zz)])
    Zf = np.fft.rfft(z); J = len(TPL)
    E = np.full((J, N), -1.0, np.float32); x0 = np.arange(N)
    for j0 in range(0, J, 400):
        blok = TPL[j0:j0 + 400]; G = np.zeros((len(blok), N))
        for i, (_, _, g) in enumerate(blok): G[i, :len(g)] = g
        Szg = np.fft.irfft(Zf[None, :] * np.conj(np.fft.rfft(G, axis=1)), N, axis=1)
        for i, (_, _, g) in enumerate(blok):
            n = len(g); Sz = c1[x0 + n] - c1[x0]; Sg = g.sum(); Sgg_c = (g * g).sum() - Sg * Sg / n
            Szg_c = Szg[i] - Sz * Sg / n
            e = Szg_c * Szg_c / Sgg_c
            if CHOICE == "En": e = e / n                     # explained variation per cell (= H^2 x variance of the template)
            elif CHOICE == "R2": szz = c2[x0 + n] - c2[x0] - Sz * Sz / n; e = np.where(szz > 1e-9, e / np.maximum(szz, 1e-9), -1)
            E[j0 + i] = np.where(Szg_c > 0, e, -1)
    bezet = np.zeros(N, bool)   # occupied cells (interior of chosen windows)
    uit = []
    while True:
        k = int(np.argmax(E)); j, a = divmod(k, N)
        if E[j, a] <= 1e-9: break   # numerical zero (flat stretch), not a ripple criterion
        nI, nS, g = TPL[j]; n = len(g); w = zz[a:a + n]; Sg = g.sum(); Sgg_c = (g * g).sum() - Sg * Sg / n
        Szg_c = float((w * g).sum() - w.sum() * Sg / n); H = Szg_c / Sgg_c; z0 = float(w.mean() - H * g.mean()); R2 = 1 - float(((w - z0 - H * g) ** 2).sum() / ((w - w.mean()) ** 2).sum())
        nI, nS, g = TPL[j]; n = len(g); uit.append((a, nI, nS, H, z0, R2, H * Szg_c))
        b = a + n - 1; bezet[(a + np.arange(1, n - 1)) % N] = True
        # invalid: every window [x, x+m-1] whose interior touches an occupied cell or that encloses an occupied interior
        for jj, (_, _, gg) in enumerate(TPL):
            m = len(gg); lo = a - m + 2; L = b - 1 - lo + 1
            if L >= N: E[jj] = -1
            else: E[jj, (lo + np.arange(L)) % N] = -1
        # windows lying across a stretch occupied earlier are thus also excluded (each choice excludes its own surroundings)
    return uit


TPL2 = None
def _init2():
    """Windows (nI, nS) sorted by length; per window the templates for each q, with the constants of the partial regression on [1, k]."""
    global TPL2
    if TPL2 is not None: return
    nI = np.rint(I_MM / DX).astype(int); nS = np.rint(S_MM / DX).astype(int)
    paren = sorted([(a, b) for a in nI for b in nS], key=lambda p: (p[0] + p[1], p[0]))
    L = []
    for a, b in paren:
        n = a + b + 1; r = np.arange(n, dtype=float); X = np.c_[np.ones(n), r]; per_q = []
        for q in QS:
            g = template(a, b, q); co = np.linalg.lstsq(X, g, rcond=None)[0]; gp = g - X @ co
            per_q.append((g, co[0], co[1], float((gp * gp).sum())))
        L.append((a, b, n, per_q))
    ns = np.array([t[2] for t in L]); grens = np.r_[0, np.where(np.diff(ns))[0] + 1, len(ns)]
    TPL2 = (L, [(int(ns[grens[i]]), int(grens[i]), int(grens[i + 1])) for i in range(len(grens) - 1)])


def fit2(z, ruis_fasen=None):
    """As fit(), with two trough levels (straight line below the template) and lee tail q. Choice on R^2 above the line."""
    _init2(); L, groepen = TPL2; N = len(z); z = np.asarray(z, float)
    if ruis_fasen is not None:
        Zf = np.fft.rfft(z); ph = np.exp(1j * ruis_fasen[:len(Zf)]); ph[0] = 1
        if N % 2 == 0: ph[-1] = 1
        z = np.fft.irfft(np.abs(Zf) * ph, N)
    zz = np.concatenate([z, z]); ii = np.arange(2 * N, dtype=float)
    c1 = np.concatenate([[0], np.cumsum(zz)]); c2 = np.concatenate([[0], np.cumsum(zz * zz)]); c3 = np.concatenate([[0], np.cumsum(zz * ii)])
    Zf = np.fft.rfft(z); J = len(L); nq = len(QS); x0 = np.arange(N)
    E = np.full((J, N), -1.0, np.float32); QI = np.zeros((J, N), np.int8)
    for j0 in range(0, J, 100):
        blok = L[j0:j0 + 100]; G = np.zeros((len(blok) * nq, N))
        for i, t in enumerate(blok):
            for iq, (g, *_) in enumerate(t[3]): G[i * nq + iq, :len(g)] = g
        Szg = np.fft.irfft(Zf[None, :] * np.conj(np.fft.rfft(G, axis=1)), N, axis=1)
        for i, (a, b, n, per_q) in enumerate(blok):
            Sz = c1[x0 + n] - c1[x0]; Szz = c2[x0 + n] - c2[x0]; Szr = c3[x0 + n] - c3[x0] - x0 * Sz
            rm = (n - 1) / 2; Srr_c = n * (n * n - 1) / 12.0; Szr_c = Szr - Sz * rm
            res_lijn = Szz - Sz * Sz / n - Szr_c * Szr_c / Srr_c       # what remains after the straight line
            best = np.full(N, -1.0); bq = np.zeros(N, np.int8)
            for iq, (g, al, be, gpp) in enumerate(per_q):
                s_zgp = Szg[i * nq + iq] - al * Sz - be * Szr
                e = np.where((s_zgp > 0) & (res_lijn > 1e-9), s_zgp * s_zgp / gpp / np.maximum(res_lijn, 1e-9), -1)
                beter = e > best; best = np.where(beter, e, best); bq = np.where(beter, iq, bq)
            E[j0 + i] = best; QI[j0 + i] = bq
    uit = []
    while True:
        k = int(np.argmax(E)); j, a = divmod(k, N)
        if E[j, a] <= 1e-9: break   # numerical zero, not a ripple criterion
        nI, nS, n, per_q = L[j]; iq = int(QI[j, a]); g = per_q[iq][0]; w = zz[a:a + n]
        X = np.c_[np.ones(n), np.arange(n) / n, g]; (z0, bb, H) = np.linalg.lstsq(X, w, rcond=None)[0]
        Xl = X[:, :2]; rl = w - Xl @ np.linalg.lstsq(Xl, w, rcond=None)[0]; R2 = 1 - float(((w - X @ np.array([z0, bb, H])) ** 2).sum() / (rl * rl).sum())
        uit.append((a, nI, nS, float(H), float(z0), R2, float(E[j, a]), QS[iq], float(bb)))
        b = a + n - 1
        for (m, r0, r1) in groepen:                # windows whose interior touches the chosen interior
            lo = a - m + 2; Lr = b - 1 - lo + 1
            if Lr >= N: E[r0:r1] = -1; continue
            lo %= N
            if lo + Lr <= N: E[r0:r1, lo:lo + Lr] = -1
            else: E[r0:r1, lo:] = -1; E[r0:r1, :lo + Lr - N] = -1
    return uit


def _existing(x, h):
    rib, _, _ = RM.ripples(x, h, prom_frac=0.0, prom_min=3.0)
    return [(r["c"], r["I"], r["S"], r["H"]) for r in rib if r["H"] >= 2.5]


def _work(arg):
    f, k, z, t, ruis = arg
    d = np.load(f); dz = float(d["dz"]); h = M.smooth(d["h"][k] * dz); h = h - h.mean(); N = len(h); x = np.arange(N) * DX
    rf = np.random.default_rng(1000 * z + k).uniform(0, 2 * np.pi, N // 2 + 1) if ruis else None
    P = fit(h, rf); B = _existing(x, h) if not ruis else []
    rows = []
    for p in P:
        (a, nI, nS, H, z0, r2, e), extra = p[:7], (p[7:] if len(p) > 7 else (1.0, 0.0))
        kam = (a + nI) % N; s_old = i_old = h_old = np.nan
        if B:
            dd = np.array([min((c - kam) % N, (kam - c) % N) for c, *_ in B]); i = int(np.argmin(dd))
            if dd[i] * DX <= 10: _, i_old, s_old, h_old = B[i]
        rows.append((z, t, a * DX, nI * DX, nS * DX, H, r2, e, i_old, s_old, h_old) + tuple(extra))
    return rows, (len(B), [(z, t) + tuple(b[1:]) for b in B])


def fit_free(projmap):
    from multiprocessing import Pool
    taken = {}
    def voeg(naam, map_, suf, t0, t1, stap, ruis=False):
        L = []
        for z in SEEDS:
            f = os.path.join(map_, f"c23_z{z}_{suf}.npz"); T = np.load(f)["t"]
            ks = [k for k in np.where((T > t0) & (T <= t1))[0]][::stap]
            L += [(f, k, z, float(T[k]), ruis) for k in ks]
        taken[naam] = L
    st = 1 if SHAPE == 1 else 2
    voeg("SR0 210-240", projmap, "SR0", 12600, 14400, st)
    voeg("ruis SR0 210-240", projmap, "SR0", 12600, 14400, st, ruis=True)
    for a, b in (((210, 240), (450, 480), (690, 720), (930, 960)) if SHAPE == 1 else ((450, 480), (690, 720), (930, 960))):
        voeg(f"SL16 {a}-{b}", projmap, "SL16", a * 60, b * 60, 2 * st)
    res = {}
    with Pool(6, initializer=_init if SHAPE == 1 else _init2) as p:
        for naam, L in taken.items():
            out = p.map(_work, L, chunksize=2)
            res[naam] = (np.array([r for o, _ in out for r in o]), np.array([b for _, (_, bb) in out for b in bb]).reshape(-1, 5), len(L))
            print(naam, len(L), "records", len(res[naam][0]), "fits", flush=True)
            np.savez(OUT_FREE, **{f"{k}|pas": v[0] for k, v in res.items()}, **{f"{k}|oud": v[1] for k, v in res.items()}, **{f"{k}|n": v[2] for k, v in res.items()})
    np.savez(OUT_FREE, **{f"{k}|pas": v[0] for k, v in res.items()}, **{f"{k}|oud": v[1] for k, v in res.items()}, **{f"{k}|n": v[2] for k, v in res.items()})


# ------------------------------------------------------------------ tables
def _measures(I, S, H):
    if len(I) == 0: return None
    si = I / S
    return dict(n=len(I), L=(I + S).mean(), H=H.mean(), I=I.mean(), S=S.mean(), SI=I.mean() / S.mean(), RI=(I + S).mean() / H.mean(),
                ti=H.mean() / I.mean(), ts=H.mean() / S.mean(), SIr=si.mean(), SIsd=si.std(), SImed=np.median(si), RIr=((I + S) / H).mean(),
                tir=(H / I).mean(), tsr=(H / S).mean())


def table_free():
    D = np.load(OUT_FREE); namen = sorted({k.split("|")[0] for k in D.files}, key=lambda s: (s.startswith("ruis"), s.startswith("SL16"), s))
    LB = 5350 * DX / 1000; o = []
    o.append("TEMPLATE FIT (template_fit.py; fixed in advance on 16 Nov. 2026; descriptive). Template = shape of Figure 1 of 1983, stoss and lee sides stretched separately.")
    if SHAPE == 2: o.append("ADJUSTED TEMPLATE (SJ_VORM=2): two trough levels (straight line below the template, H = crest above that line) and lee tail q (1 / 0.75 / 0.5 / 0.35); R2 = share of what remains after the line.")
    o.append("1983 field (per ripple averaged): RI 22, SI 3 +- 1, tan i 0.05, tan s 0.20; L 107, H 5, I 80, S 27. From the means: SI 2.96, tan i 0.063, tan s 0.185.")
    o.append("Columns: n/m = ripples per metre of bed per record; bed = share of the bed length within chosen windows; 'mean' = typical ripple from the means (measuring rule);")
    o.append("'per ripple' = mean of the ratios per ripple (as 1983), SI also median and sd.")
    for nm in namen:
        P = D[f"{nm}|pas"]; nop = int(D[f"{nm}|n"]); OUD = D[f"{nm}|oud"]
        o.append(f"\n=== {nm} ({nop} records, {len(P)} fits) ===")
        if len(OUD):
            m = _measures(OUD[:, 2], OUD[:, 3], OUD[:, 4])
            o.append(f"  existing measuring rule (prominence 3 mm, H >= 2.5): {m['n'] / nop / LB:4.1f}/m  mean: SI {m['SI']:.2f} RI {m['RI']:4.1f} tan i {m['ti']:.3f} tan s {m['ts']:.3f}"
                     f" | per ripple: SI {m['SIr']:.2f} +- {m['SIsd']:.2f} (med. {m['SImed']:.2f})  | L {m['L']:.0f} H {m['H']:.1f} I {m['I']:.0f} S {m['S']:.0f}")
        for hmin, kop in ((0.0, "all fits"), (2.5, "added afterwards (descriptive): only H >= 2.5 mm")):
            o.append(f"  -- {kop}")
            o.append(f"  {'R2 >=':6s} {'n/m':>5s} {'bed':>5s} | {'mean: SI':>8s} {'RI':>5s} {'tan i':>6s} {'tan s':>6s} | {'per ripple: SI (sd, med.)':>26s} {'RI':>5s} {'tan i':>6s} {'tan s':>6s} | {'L':>4s} {'H':>4s} {'I':>4s} {'S':>4s}")
            for rc in RCUT:
                q = P[(P[:, 6] >= rc) & (P[:, 5] >= hmin)]; m = _measures(q[:, 3], q[:, 4], q[:, 5])
                if m is None: o.append(f"  {rc:6.1f}  none"); continue
                bed = (q[:, 3] + q[:, 4]).sum() / (nop * LB * 1000)
                o.append(f"  {rc:6.1f} {m['n'] / nop / LB:5.1f} {bed:5.2f} | {m['SI']:8.2f} {m['RI']:5.1f} {m['ti']:6.3f} {m['ts']:6.3f} | {m['SIr']:5.2f} ({m['SIsd']:.2f}, {m['SImed']:.2f}){'':>9s} {m['RIr']:5.1f} {m['tir']:6.3f} {m['tsr']:6.3f}"
                         f" | {m['L']:4.0f} {m['H']:4.1f} {m['I']:4.0f} {m['S']:4.0f}")
        if SHAPE == 2:
            for hmin in (0.0, 2.5):
                q = P[(P[:, 6] >= 0.8) & (P[:, 5] >= hmin)]
                o.append(f"  -- R2 >= 0.8{', H >= 2.5' if hmin else ''}: share per lee tail q " + "  ".join(f"{qq}: {np.mean(np.isclose(q[:, 11], qq)):.2f}" for qq in QS)
                         + f" | height difference of the troughs (downwind - upwind) mean {q[:, 12].mean():+.2f}, mean |b| {np.abs(q[:, 12]).mean():.2f} mm")
            f1 = os.path.join(HERE, "output", "template_ripples.npz")
            if os.path.exists(f1) and f"{nm}|pas" in np.load(f1).files:
                P1 = np.load(f1)[f"{nm}|pas"]; tz = set(map(tuple, P[:, :2])); P1 = P1[[tuple(r) in tz for r in P1[:, :2]]]
                for hmin in (0.0, 2.5):
                    a = [P1[(P1[:, 6] >= 0.8) & (P1[:, 5] >= hmin)], P[(P[:, 6] >= 0.8) & (P[:, 5] >= hmin)]]
                    o.append(f"  -- comparison with the first template, same records, R2 >= 0.8{', H >= 2.5' if hmin else ''}: first / adjusted:  n/m {len(a[0]) / nop / LB:.1f} / {len(a[1]) / nop / LB:.1f}"
                             + "  " + "  ".join(f"{k} " + " / ".join(f"{v:.{d}f}" for v in vv) for k, vv, d in (
                                 ("SI", [x[:, 3].mean() / x[:, 4].mean() for x in a], 2), ("tan i", [x[:, 5].mean() / x[:, 3].mean() for x in a], 3),
                                 ("tan s", [x[:, 5].mean() / x[:, 4].mean() for x in a], 3), ("L", [(x[:, 3] + x[:, 4]).mean() for x in a], 0),
                                 ("H", [x[:, 5].mean() for x in a], 1), ("I", [x[:, 3].mean() for x in a], 0), ("S", [x[:, 4].mean() for x in a], 0))))
        if nm.startswith("ruis"): continue
        o.append("  -- check (option 1): the same ripples measured with the existing rule (crests within 10 mm), SI from the means: template / existing rule")
        for rc in RCUT:
            q = P[(P[:, 6] >= rc) & np.isfinite(P[:, 8])]
            if len(q) == 0: continue
            o.append(f"  R2 >= {rc:.1f}: {len(q):5d} matched ({len(q) / max(len(OUD), 1):.0%} of the existing)  SI {q[:, 3].mean() / q[:, 4].mean():.2f} / {q[:, 8].mean() / q[:, 9].mean():.2f}"
                     f"  I {q[:, 3].mean():.0f} / {q[:, 8].mean():.0f}  S {q[:, 4].mean():.0f} / {q[:, 9].mean():.0f}  H {q[:, 5].mean():.1f} / {q[:, 10].mean():.1f}")
        o.append("  -- distribution of R2 (share of the fits; ripples with H >= 2.5 in brackets): " + "  ".join(
            f"{a:.1f}-{b:.1f} {np.mean((P[:, 6] >= a) & (P[:, 6] < b)):.2f} ({np.mean((P[:, 6] >= a) & (P[:, 6] < b) & (P[:, 5] >= 2.5)):.2f})" for a, b in ((-1, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01))))
    s = "\n".join(o); print(s)
    with open(os.path.join(HERE, "output", f"template{SUF}.txt"), "w") as f: f.write(s + "\n")


# ------------------------------------------------------------------ test_free
def test_free():
    rng = np.random.default_rng(5); N = 5350; z = np.zeros(N); ware = []; a = 0; niv = 0.0; qs = []
    while True:
        I = float(rng.choice(np.arange(40, 141, 2))); S = float(rng.choice(np.arange(20, 71, 2))); H = float(rng.uniform(2, 8))
        nI, nS = int(round(I / DX)), int(round(S / DX)); n = nI + nS + 1
        if a + nI + nS >= N: break
        q = float(rng.choice(QS)); dn = float(rng.uniform(-2, 2)) if SHAPE == 2 else 0.0   # shape 2: lee tail and troughs at varying height
        z[a:a + n] = niv + dn * np.arange(n) / n + H * template(nI, nS, q); z[a + n - 1:] = niv + dn * (n - 1) / n
        ware.append((a, I, S, H)); qs.append((q, dn * (n - 1) / n)); niv += dn * (n - 1) / n; a += nI + nS
    z -= z.mean(); ok = True
    for ruis in (0.0, 0.3):
        zr = z + (M.smooth(np.random.default_rng(7).normal(0, ruis * 2.2, N)) if ruis else 0)   # sd after smoothing ~noise
        P = fit(zr); fout = []
        for (a0, I, S, H) in ware:
            dd = [min((p[0] - a0) % N, (a0 - p[0]) % N) * DX for p in P]; i = int(np.argmin(dd))
            if dd[i] <= 15: p = P[i]; fout.append((dd[i], p[1] * DX - I, p[2] * DX - S, p[3] - H, p[5]))
        fout = np.array(fout).reshape(-1, 5); kl = [p for p in P if p[5] < 0.7]
        print(f"noise sd ~{ruis} mm: {len(ware)} true ripples (H 2-8 mm), {len(P)} fits, {len(fout)} recovered (trough within 15 mm); trough error mean {fout[:, 0].mean():.1f} max {fout[:, 0].max():.1f} mm,"
              f" I mean |{np.abs(fout[:, 1]).mean():.1f}| max {np.abs(fout[:, 1]).max():.1f}, S mean |{np.abs(fout[:, 2]).mean():.1f}| max {np.abs(fout[:, 2]).max():.1f}, H max {np.abs(fout[:, 3]).max():.2f} mm;"
              f" R2 of the true ones {fout[:, 4].min():.3f}-{fout[:, 4].max():.3f}; other fits {len(P) - len(fout)} (R2 < 0.7: {len(kl)}, H max {max([p[3] for p in kl], default=0):.2f})")
        if SHAPE == 2:
            qg = [(P[int(np.argmin([min((p[0] - w[0]) % N, (w[0] - p[0]) % N) for p in P]))][7:9]) for w in ware]
            print(f"   lee tail q correct in {sum(abs(g[0] - w[0]) < 1e-9 for g, w in zip(qg, qs))} of {len(ware)}; height difference of the troughs max error"
                  f" {max(abs(g[1] * (1) - w[1]) for g, w in zip(qg, qs)):.2f} mm (b per window, n/(n-1) neglected)")
        if ruis == 0 and not (len(fout) == len(ware) and np.abs(fout[:, :3]).max() < 1e-9 and np.abs(fout[:, 3]).max() < 1e-6): ok = False
    print("TEST", "MET" if ok else "NOT MET")


# ------------------------------------------------------------------ fixed windows
QV = (1.0, 0.75, 0.5, 0.35)
OUT_FIXED = os.path.join(HERE, "output", "template_fixed_ripples.npz")


def fit_window(w, vorm):
    """Fit the template in one fixed window w (points trough..trough). shape 1: one level, q = 1; shape 2: straight line below the template, q free.
    Returns (R2, kI, H, q, b) with kI = index of the crest; None if nothing with H > 0 fits."""
    n = len(w) - 1; r = np.arange(n + 1, dtype=float)
    X = np.c_[np.ones(n + 1)] if vorm == 1 else np.c_[np.ones(n + 1), r / n]
    proj = X @ np.linalg.pinv(X); wl = w - proj @ w; sww = float(wl @ wl)
    if sww <= 1e-12 or n < 5: return None
    best = None
    for q in ((1.0,) if vorm == 1 else QV):
        kI = np.arange(2, n - 1); G = np.array([template(k, n - k, q) for k in kI]); Gp = G - G @ proj.T
        sg = Gp @ wl; gg = (Gp * Gp).sum(1); R2 = np.where(sg > 0, sg * sg / gg / sww, -1); i = int(np.argmax(R2))
        if R2[i] > 0 and (best is None or R2[i] > best[0]):
            H = float(sg[i] / gg[i]); co = np.linalg.lstsq(X, w - H * G[i], rcond=None)[0]
            best = (float(R2[i]), int(kI[i]), H, q, float(co[1]) if vorm == 2 else 0.0)
    return best


def _windows(h):
    """Troughs of the existing rule (prominence 3 mm) and per window trough-trough the highest point and the H of the existing rule."""
    N = len(h); x = np.arange(N) * DX; tr = RM.troughs(h, 3.0)
    if len(tr) < 2: return []
    rib, _, _ = RM.ripples(x, h, prom_frac=0.0, prom_min=3.0); Hc = {r["c"]: r["H"] for r in rib}
    uit = []
    for j, t0 in enumerate(tr):
        t1 = tr[(j + 1) % len(tr)]; n = int((t1 - t0) % N) or N; idx = (t0 + np.arange(n + 1)) % N
        c = int(np.argmax(h[idx[1:-1]])) + 1
        uit.append((int(t0), n, c, Hc.get(int(idx[c]), np.nan), idx))
    return uit


def _work_fixed(arg):
    f, k, z, t, ruis = arg
    d = np.load(f); h = M.smooth(d["h"][k] * float(d["dz"])); h = h - h.mean(); N = len(h)
    if ruis:
        rf = np.random.default_rng(1000 * z + k).uniform(0, 2 * np.pi, N // 2 + 1); Zf = np.fft.rfft(h); ph = np.exp(1j * rf[:len(Zf)]); ph[0] = 1; ph[-1] = 1
        h = np.fft.irfft(np.abs(Zf) * ph, N)
    rows = []
    for (t0, n, c, Hold, idx) in _windows(h):
        w = h[idx]; uit = [z, t, t0 * DX, n * DX, c * DX, (n - c) * DX, Hold]
        for vorm in (1, 2):
            p = fit_window(w, vorm)
            if p is None: uit += [np.nan] * (5 if vorm == 1 else 6); continue
            R2, kI, H, q, bb = p
            uit += [R2, kI * DX, (n - kI) * DX, H, 0.0] if vorm == 1 else [R2, kI * DX, (n - kI) * DX, H, q, bb]
        rows.append(uit)
    return rows
# columns: 0 seed, 1 t, 2 trough (mm), 3 window L, 4 I old, 5 S old, 6 H old | 7-11 first template: R2, I, S, H, (b=0) | 12-16 adjusted: R2, I, S, H, q | 17 b


def fit_fixed(projmap):
    from multiprocessing import Pool
    taken = {}
    def voeg(naam, map_, suf, t0, t1, stap, ruis=False):
        L = []
        for z in SEEDS:
            f = os.path.join(map_, f"c23_z{z}_{suf}.npz"); T = np.load(f)["t"]
            L += [(f, k, z, float(T[k]), ruis) for k in np.where((T > t0) & (T <= t1))[0][::stap]]
        taken[naam] = L
    voeg("SR0 210-240", projmap, "SR0", 12600, 14400, 1); voeg("ruis SR0 210-240", projmap, "SR0", 12600, 14400, 1, ruis=True)
    for a, b in ((450, 480), (690, 720), (930, 960)): voeg(f"SL16 {a}-{b}", projmap, "SL16", a * 60, b * 60, 2)
    res = {}
    with Pool(6) as p:
        for naam, L in taken.items():
            out = p.map(_work_fixed, L, chunksize=4); res[naam] = (np.array([r for o in out for r in o], float), len(L))
            print(naam, len(L), "records", len(res[naam][0]), "windows", flush=True)
    np.savez(OUT_FIXED, **{f"{k}|v": v[0] for k, v in res.items()}, **{f"{k}|n": v[1] for k, v in res.items()})


def table_fixed():
    D = np.load(OUT_FIXED); namen = sorted({k.split("|")[0] for k in D.files}, key=lambda s: (s.startswith("ruis"), s.startswith("SL16"), s)); LB = 5350 * DX / 1000
    o = ["TEMPLATE IN FIXED WINDOWS (template_fit.py fit_fixed; fixed in advance on 16 Nov. 2026; descriptive). Windows = trough to trough by the existing rule",
         "(prominence 3 mm); within each window only the shape is fitted. 'old' = highest point as crest; 'first' = template with one level, q 1; 'adjusted' = two levels + lee tail.",
         "Ratios from the means (measuring rule); SI per ripple as mean (median). H old = H of the existing rule (plexiglass), H template = crest above the trough line.",
         "1983 field: SI 2.96 (per ripple 3 +- 1), tan i 0.063, tan s 0.185, RI 21-22."]
    for nm in namen:
        V = D[f"{nm}|v"]; nop = int(D[f"{nm}|n"]); o.append(f"\n=== {nm} ({nop} records, {len(V)} windows, {len(V) / nop / LB:.1f} per m) ===")
        for kop, m in (("all windows", np.isfinite(V[:, 12])), ("H old >= 2.5 mm", np.isfinite(V[:, 12]) & (V[:, 6] >= 2.5))):
            q = V[m]; o.append(f"  -- {kop}: {len(q) / nop / LB:.1f} per m")
            for lab, I, S, H in (("old (highest point)", q[:, 4], q[:, 5], q[:, 6]), ("first template", q[:, 8], q[:, 9], q[:, 10]), ("adjusted template", q[:, 13], q[:, 14], q[:, 15])):
                ok = np.isfinite(I) & np.isfinite(H); I, S, H = I[ok], S[ok], H[ok]
                o.append(f"     {lab:21s} SI {I.mean() / S.mean():.2f}  tan i {H.mean() / I.mean():.3f}  tan s {H.mean() / S.mean():.3f}  RI {(I + S).mean() / H.mean():5.1f}"
                         f" | per ripple SI {np.mean(I / S):.2f} (med. {np.median(I / S):.2f}) | I {I.mean():.0f} S {S.mean():.0f} H {H.mean():.1f}")
            o.append("     R2 adjusted: " + "  ".join(f">= {c}: {np.mean(q[:, 12] >= c):.2f}" for c in (0.5, 0.7, 0.8, 0.9)) + f"  (median {np.median(q[:, 12]):.2f});"
                     f" first: median {np.nanmedian(q[:, 7]):.2f}")
            for c in (0.8, 0.9):
                r = q[q[:, 12] >= c]
                if len(r): o.append(f"     adjusted, R2 >= {c}: {len(r) / nop / LB:.1f} per m  SI {r[:, 13].mean() / r[:, 14].mean():.2f}  tan i {r[:, 15].mean() / r[:, 13].mean():.3f}"
                                    f"  tan s {r[:, 15].mean() / r[:, 14].mean():.3f}  (old on the same windows SI {r[:, 4].mean() / r[:, 5].mean():.2f})")
            o.append("     lee tail q " + "  ".join(f"{qq}: {np.mean(np.isclose(q[:, 16], qq)):.2f}" for qq in QV)
                     + f" | trough downwind - upwind mean {np.nanmean(q[:, 17]):+.2f} mm | fitted crest - highest point mean {np.mean(q[:, 13] - q[:, 4]):+.1f} mm (|{np.mean(np.abs(q[:, 13] - q[:, 4])):.1f}|)")
    s = "\n".join(o); print(s)
    with open(os.path.join(HERE, "output", "template_fixed.txt"), "w") as f: f.write(s + "\n")


def test_fixed():
    rng = np.random.default_rng(5); ok = True; fout = []
    for _ in range(40):
        I = float(rng.choice(np.arange(40, 141, 2))); S = float(rng.choice(np.arange(20, 71, 2))); H = float(rng.uniform(2, 8)); q = float(rng.choice(QV)); bb = float(rng.uniform(-2, 2))
        nI, nS = int(round(I / DX)), int(round(S / DX)); n = nI + nS; w = 0.7 + bb * np.arange(n + 1) / n + H * template(nI, nS, q)
        p = fit_window(w, 2); fout.append((p[1] - nI, p[3] - q, p[2] - H, p[4] - bb, p[0]))
    f = np.array(fout); ok = np.abs(f[:, :4]).max() < 1e-6
    print(f"fixed windows, without noise, 40 artificial ripples: max error crest {np.abs(f[:, 0]).max()} cellen, q {np.abs(f[:, 1]).max():.2g}, H {np.abs(f[:, 2]).max():.2g} mm, b {np.abs(f[:, 3]).max():.2g} mm; R2 min {f[:, 4].min():.6f}")
    print("TEST", "MET" if ok else "NOT MET")


# ------------------------------------------------------------------ figures: see fig6() and fig7() below
# ------------------------------------------------------------------ chain fit
PROMS = (1.0, 1.5, 2.0, 3.0, 4.0, 5.0)
OUT_CHAIN = os.path.join(HERE, "output", "template_chain_ripples.npz")


from functools import lru_cache
@lru_cache(maxsize=3000)   # capped: 200000 filled the memory of an 8 GB Mac
def _sjc(nI, nS, q): return template(int(nI), int(nS), float(q))


class Chain:
    """Contiguous series of template ripples on a periodic profile z. T: troughs (increasing, T[M] = T[0] + N); C: crests (absolute); Q: lee tail per ripple."""
    def __init__(self, z, T):
        self.z = np.asarray(z, float); self.N = len(z); self.zz = np.concatenate([self.z, self.z, self.z])
        T = np.sort(np.asarray(T, int)); self.T = np.r_[T, T[0] + self.N]; M = len(T); self.C = np.zeros(M, int); self.Q = np.ones(M)
        for j in range(M):
            w = self.zz[self.T[j] + self.N:self.T[j + 1] + 1 + self.N]; p = fit_window(w, 2) if len(w) > 6 else None
            self.C[j] = self.T[j] + (p[1] if p else max(2, min(len(w) - 3, int(np.argmax(w))))); self.Q[j] = p[3] if p else 1.0
        self.los()

    @property
    def M(self): return len(self.T) - 1

    def g(self, j, t0=None, t1=None, c=None, q=None):
        t0 = self.T[j] if t0 is None else t0; t1 = self.T[j + 1] if t1 is None else t1; c = self.C[j] if c is None else c; q = self.Q[j] if q is None else q
        n = t1 - t0; return _sjc(c - t0, n - (c - t0), q)[:n]

    def los(self):
        """All trough heights and H in one linear least-squares step."""
        M, N = self.M, self.N; A = np.zeros((N, 2 * M)); b = np.zeros(N)
        for j in range(M):
            t0, t1 = self.T[j], self.T[j + 1]; n = t1 - t0; k = np.arange(n); idx = (t0 + k) % N
            A[idx, j] += 1 - k / n; A[idx, (j + 1) % M] += k / n; A[idx, M + j] = self.g(j)
        co = np.linalg.lstsq(A, self.z, rcond=None)[0]; self.Y, self.H = co[:M], co[M:]; self.res = self.z - A @ co; self.SS = float(self.res @ self.res)

    def lijn(self, j, t0, t1, yl, yr):
        n = t1 - t0; k = np.arange(n); return yl * (1 - k / n) + yr * k / n

    def stap_kam(self):
        for j in range(self.M):
            t0, t1 = self.T[j], self.T[j + 1]; n = t1 - t0
            if n < 6: continue
            r = self.zz[t0 + self.N:t1 + self.N] - self.lijn(j, t0, t1, self.Y[j], self.Y[(j + 1) % self.M]); best = (-1, None, None)
            for q in QV:
                kI = np.arange(2, n - 1); G = np.array([_sjc(k, n - k, q)[:n] for k in kI]); sg = G @ r; gg = (G * G).sum(1)
                e = np.where(sg > 0, sg * sg / gg, -1); i = int(np.argmax(e))
                if e[i] > best[0]: best = (e[i], kI[i], q)
            if best[1] is not None: self.C[j] = t0 + best[1]; self.Q[j] = best[2]
        self.los()

    def _ss_dal(self, t, tl, cl, cr, tr, yl, yr, ql, qr, rhs0):
        gl = _sjc(cl - tl, t - cl, ql)[:t - tl]; gr = _sjc(cr - t, tr - cr, qr)[:tr - t]
        kl = np.arange(t - tl) / (t - tl); kr = np.arange(tr - t) / (tr - t)
        rl = rhs0[:t - tl] - yl * (1 - kl); rr = rhs0[t - tl:] - yr * kr; al = kl; ar = 1 - kr   # unknowns: height of trough j, H left, H right
        Nm = np.array([[al @ al + ar @ ar, al @ gl, ar @ gr], [al @ gl, gl @ gl, 0.0], [ar @ gr, 0.0, gr @ gr]]); v = np.array([al @ rl + ar @ rr, gl @ rl, gr @ rr])
        try: co = np.linalg.solve(Nm, v)
        except np.linalg.LinAlgError: return np.inf
        return float(rl @ rl + rr @ rr - co @ v)

    def stap_dal(self, gezamenlijk=False):
        """Shift the position per trough (crests fixed). gezamenlijk: also the lee tail of both neighbours, within 40 cells of the current position."""
        M = self.M
        for j in range(M):
            jl = (j - 1) % M; cl = self.C[jl] - (self.N if j == 0 else 0); cr = self.C[j]; tl = self.T[j - 1] if j > 0 else self.T[M - 1] - self.N; tr = self.T[j + 1]
            yl, yr = self.Y[jl], self.Y[(j + 1) % M]; rhs0 = self.zz[tl + self.N:tr + self.N]
            if gezamenlijk:
                tt = range(max(cl + 2, self.T[j] - 40), min(cr - 1, self.T[j] + 41)); qq = [(a, b) for a in QV for b in QV]
            else:
                tt = range(cl + 2, cr - 1); qq = [(self.Q[jl], self.Q[j])]
            best = (self._ss_dal(self.T[j], tl, cl, cr, tr, yl, yr, self.Q[jl], self.Q[j], rhs0), self.T[j], self.Q[jl], self.Q[j])
            for t in tt:
                for ql, qr in qq:
                    ss = self._ss_dal(t, tl, cl, cr, tr, yl, yr, ql, qr, rhs0)
                    if ss < best[0] - 1e-12: best = (ss, t, ql, qr)
            _, t, self.Q[jl], self.Q[j] = best; self.T[j] = t
            if j == 0: self.T[M] = t + self.N          # T[0] may become negative; all windows read zz with shift N
        self.los()

    def fit(self, rondes=10):
        for _ in range(rondes):
            ss0 = self.SS; self.stap_kam(); self.stap_dal(); self.stap_dal(gezamenlijk=True)
            if ss0 - self.SS <= 1e-9 * max(ss0, 1e-12): break
        return self

    def model(self, T, C, Y, H, Q):
        """Chain on the grid with trough and crest positions as real numbers (in cells). T has M+1 values, T[M] = T[0] + N."""
        N, M = self.N, len(C); i = np.arange(N, dtype=float); x = (i - T[0]) % N + T[0]; j = np.clip(np.searchsorted(T, x, "right") - 1, 0, M - 1)
        t0, t1, c = T[j], T[j + 1], C[j]; Yr = np.r_[Y, Y[0]]
        u = np.clip((x - t0) / (c - t0), 0, 1); v = np.clip((x - c) / (t1 - c), 0, 1)
        g = np.where(x <= c, np.interp(u, U, GL), np.interp(v ** Q[j], U, GJ))
        return Yr[j] + (Yr[j + 1] - Yr[j]) * (x - t0) / (t1 - t0) + H[j] * g

    def verfijn(self, rondes=4):
        """Simultaneous fit: all trough and crest positions (real), heights, H and the lee tail q (real, 0.35-1) at once with least squares.
        (Deviation from the advance registration: q continuous in the last step instead of four values; otherwise the fit gets stuck where trough and q must shift together.)"""
        from scipy.optimize import least_squares
        M, N = self.M, self.N; T = self.T.astype(float); C = self.C.astype(float); Q = self.Q.astype(float)
        for _ in range(rondes):
            Tm = T[:-1]; Cl = np.r_[C[-1] - N, C[:-1]]; Tn = T[1:]
            lo = np.r_[(Cl + Tm) / 2 + 0.01, (Tm + C) / 2 + 0.01, np.full(2 * M, -np.inf), np.full(M, 0.35)]
            hi = np.r_[(Tm + C) / 2 - 0.01, (C + Tn) / 2 - 0.01, np.full(2 * M, np.inf), np.full(M, 1.0)]
            x0 = np.clip(np.r_[Tm, C, self.Y, self.H, Q], lo + 1e-9, hi - 1e-9)
            f = lambda p: self.model(np.r_[p[:M], p[0] + N], p[M:2 * M], p[2 * M:3 * M], p[3 * M:4 * M], p[4 * M:]) - self.z
            r = least_squares(f, x0, bounds=(lo, hi), x_scale="jac", xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=300)
            p = r.x; ss0 = self.SS
            T = np.r_[p[:M], p[0] + N]; C = p[M:2 * M]; self.Y, self.H, Q = p[2 * M:3 * M], p[3 * M:4 * M], p[4 * M:]; self.SS = float(r.fun @ r.fun)
            if ss0 - self.SS <= 1e-9 * max(ss0, 1e-12): break
        self.Tf, self.Cf, self.Q = T, C, Q
        return self

    def ribbels(self):
        """Per ripple: trough (mm), I, S, H, q, R2 above the line within the window (refined, real positions)."""
        T, C = (self.Tf, self.Cf) if hasattr(self, "Tf") else (self.T.astype(float), self.C.astype(float)); uit = []
        fit = self.model(T, C, self.Y, self.H, self.Q); i = np.arange(self.N, dtype=float); x = (i - T[0]) % self.N + T[0]
        for j in range(self.M):
            m = (x >= T[j]) & (x < T[j + 1]); w = self.z[m]; xx = x[m]; X = np.c_[np.ones(m.sum()), xx]
            wl = w - X @ np.linalg.lstsq(X, w, rcond=None)[0]; R2 = 1 - float(((w - fit[m]) ** 2).sum() / max((wl * wl).sum(), 1e-12))
            uit.append(((T[j] % self.N) * DX, (C[j] - T[j]) * DX, (T[j + 1] - C[j]) * DX, float(self.H[j]), float(self.Q[j]), R2))
        return uit


def bic(SS, M, N): ne = N / 5; return ne * np.log(SS / ne) + 5 * M * np.log(ne)


def _work_chain(arg):
    f, k, z, t, ruis = arg; _sjc.cache_clear()
    d = np.load(f); h = M.smooth(d["h"][k] * float(d["dz"])); h = h - h.mean(); N = len(h)
    if ruis:
        rf = np.random.default_rng(1000 * z + k).uniform(0, 2 * np.pi, N // 2 + 1); Zf = np.fft.rfft(h); ph = np.exp(1j * rf[:len(Zf)]); ph[0] = 1; ph[-1] = 1
        h = np.fft.irfft(np.abs(Zf) * ph, N)
    per = {}; rows = []; som = []
    for pr in PROMS:
        tr = RM.troughs(h, pr)
        if len(tr) < 2: continue
        K = Chain(h, tr).fit().verfijn(); per[pr] = K; som.append((z, t, pr, K.M, K.SS, bic(K.SS, K.M, N), 1 - K.SS / float(h @ h)))
    kies = min(per, key=lambda pr: bic(per[pr].SS, per[pr].M, N))
    for pr, K in per.items():
        for r in K.ribbels(): rows.append((z, t, pr, float(pr == kies)) + r)
    return rows, som
# columns ripples: 0 seed, 1 t, 2 starting prominence, 3 chosen by BIC, 4 trough (mm), 5 I, 6 S, 7 H, 8 q, 9 R2 | sum: seed, t, prom, M, SS, BIC, R2 chain


def fit_chain(projmap):
    from multiprocessing import Pool
    taken = {}
    def voeg(naam, map_, suf, t0, t1, stap, ruis=False):
        L = []
        for z in SEEDS:
            f = os.path.join(map_, f"c23_z{z}_{suf}.npz"); T = np.load(f)["t"]
            L += [(f, k, z, float(T[k]), ruis) for k in np.where((T > t0) & (T <= t1))[0][::stap]]
        taken[naam] = L
    voeg("SR0 210-240", projmap, "SR0", 12600, 14400, 2); voeg("ruis SR0 210-240", projmap, "SR0", 12600, 14400, 2, ruis=True)
    for a, b in ((450, 480), (690, 720), (930, 960)): voeg(f"SL16 {a}-{b}", projmap, "SL16", a * 60, b * 60, 4)
    res = {}
    if os.path.exists(OUT_CHAIN):                         # resume: series already done are not redone
        D = np.load(OUT_CHAIN); res = {k.split("|")[0]: (D[k.split("|")[0] + "|r"], D[k.split("|")[0] + "|s"], int(D[k.split("|")[0] + "|n"])) for k in D.files}
    with Pool(3, maxtasksperchild=10) as p:          # 3 processes: ~0.8 GB peak each (an 8 GB Mac crashed with 6)
        for naam, L in taken.items():
            if naam in res: print(naam, "already done", flush=True); continue
            out = p.map(_work_chain, L, chunksize=2)
            res[naam] = (np.array([r for o, _ in out for r in o], float), np.array([r for _, s in out for r in s], float), len(L))
            print(naam, len(L), "records", len(res[naam][0]), "ripples (all prominences)", flush=True)
            np.savez(OUT_CHAIN, **{f"{k}|r": v[0] for k, v in res.items()}, **{f"{k}|s": v[1] for k, v in res.items()}, **{f"{k}|n": v[2] for k, v in res.items()})


def test_chain():
    rng = np.random.default_rng(11); N = 5350; ok = True
    for poging in range(3):
        L = []; a = 0
        while N - a > 300:
            nI = int(rng.integers(100, 351)); nS = int(rng.integers(50, 176)); L.append((a, nI, nS, float(rng.choice(QV)), float(rng.uniform(3, 8)))); a += nI + nS
        a0, nI, nS, q, H = L[-1]; rest = N - a0
        if rest < 150: L.pop(); a0, nI, nS, q, H = L[-1]; rest = N - a0
        nI = int(rest * 0.6); nS = rest - nI; L[-1] = (a0, nI, nS, q, H)
        Y = rng.uniform(-2, 2, len(L)); z = np.zeros(N)
        for j, (a0, nI, nS, q, H) in enumerate(L):
            n = nI + nS; k = np.arange(n); z[a0:a0 + n] = Y[j] * (1 - k / n) + Y[(j + 1) % len(L)] * k / n + H * template(nI, nS, q)[:n]
        per = {}
        for pr in PROMS:
            tr = RM.troughs(z, pr)
            if len(tr) >= 2: per[pr] = Chain(z, tr).fit(20).verfijn()
        kies = min(per, key=lambda pr: bic(max(per[pr].SS, 1e-20), per[pr].M, N)); K = per[kies]
        cd = lambda a, b: abs((a - b + N / 2) % N - N / 2); gev = list(zip(K.Tf[:-1], K.Cf, K.Q, K.H)); afw = []
        for a0, nI, nS, q, H in L:
            g = min(gev, key=lambda g: cd(g[0], a0)); afw.append((cd(g[0], a0), cd(g[1], a0 + nI), abs(g[2] - q), abs(g[3] - H)))
        afw = np.array(afw); goed = K.M == len(L) and afw[:, :2].max() < 0.01 and afw[:, 2].max() < 1e-3 and afw[:, 3].max() < 1e-3
        print(f"   max deviation trough {afw[:, 0].max():.2g} cell, crest {afw[:, 1].max():.2g} cell, q {afw[:, 2].max():.2g}, H {afw[:, 3].max():.2g} mm")
        print(f"artificial profile {poging + 1}: {len(L)} ripples; BIC chooses prominence {kies} with {K.M} ripples; SS {K.SS:.2e}; all exact: {goed}"
              + "".join(f" | prom {pr}: M {per[pr].M} SS {per[pr].SS:.1e}" for pr in per))
        ok &= goed
    print("TEST", "MET" if ok else "NOT MET")


def _km(R):
    I, S, H = R[:, 5], R[:, 6], R[:, 7]
    return dict(SI=I.mean() / S.mean(), ti=H.mean() / I.mean(), ts=H.mean() / S.mean(), RI=(I + S).mean() / H.mean(), SIr=np.mean(I / S), SImed=np.median(I / S),
                I=I.mean(), S=S.mean(), H=H.mean(), q=R[:, 8].mean(), qs=np.mean(R[:, 8] < 0.9), R2=np.median(R[:, 9]))


def table_chain():
    D = np.load(OUT_CHAIN); namen = sorted({k.split("|")[0] for k in D.files}, key=lambda s: (s.startswith("ruis"), s.startswith("SL16"), s)); LB = 5350 * DX / 1000
    o = ["CHAIN FIT (template_fit.py fit_chain; fixed in advance on 16 Nov. 2026; descriptive). Contiguous template ripples (shared troughs, lee tail q).",
         "A = starting troughs of the existing rule (prominence 3 mm); B = number chosen with the BIC (N_eff = N/5) from starting troughs with prominence 1-5 mm.",
         "Ratios from the means; SI per ripple mean (median); H = crest above the trough line; q < 0.9 = share with lee tail; R2 = median per ripple above the line.",
         "1983 field: SI 2.96 (per ripple 3 +- 1), tan i 0.063, tan s 0.185, RI 21-22."]
    for nm in namen:
        R = D[f"{nm}|r"]; S_ = D[f"{nm}|s"]; nop = int(D[f"{nm}|n"]); o.append(f"\n=== {nm} ({nop} records) ===")
        kies = R[R[:, 3] == 1]; gek = np.unique(kies[:, :3], axis=0)[:, 2]
        o.append("  BIC chooses starting troughs with prominence: " + "  ".join(f"{pr}: {np.mean(gek == pr):.2f}" for pr in PROMS))
        o.append(f"  {'':24s} {'n/m':>5s} {'chain R2':>10s} | {'SI':>5s} {'tan i':>6s} {'tan s':>6s} {'RI':>5s} | {'SI/ripple (med.)':>16s} | {'I':>4s} {'S':>4s} {'H':>4s} | {'q<0.9':>5s} {'R2 med.':>7s}")
        def regel(lab, X, nr):
            if len(X) == 0: return f"  {lab:24s} none"
            m = _km(X); return (f"  {lab:24s} {len(X) / nop / LB:5.1f} {nr:10s} | {m['SI']:5.2f} {m['ti']:6.3f} {m['ts']:6.3f} {m['RI']:5.1f} | {m['SIr']:6.2f} ({m['SImed']:.2f}){'':>4s} |"
                                f" {m['I']:4.0f} {m['S']:4.0f} {m['H']:4.1f} | {m['qs']:5.2f} {m['R2']:7.2f}")
        for pr in PROMS:
            X = R[R[:, 2] == pr]; r2 = S_[S_[:, 2] == pr][:, 6].mean() if np.any(S_[:, 2] == pr) else np.nan
            o.append(regel(f"prominence {pr}" + (" (A)" if pr == 3.0 else ""), X, f"{r2:.3f}"))
        o.append(regel("B (chosen by BIC)", kies, ""))
        o.append("  -- only H >= 2.5 mm (H of the template):")
        for pr in (1.0, 2.0, 3.0, 5.0):
            X = R[(R[:, 2] == pr) & (R[:, 7] >= 2.5)]; o.append(regel(f"prominence {pr}" + (" (A)" if pr == 3.0 else ""), X, ""))
        o.append(regel("B (chosen by BIC)", kies[kies[:, 7] >= 2.5], ""))
    s = "\n".join(o); print(s)
    with open(os.path.join(HERE, "output", "template_chain.txt"), "w") as f: f.write(s + "\n")


def fig6(projmap):
    """Figure 6: comparison of the three fits with the adjusted template (free, fixed windows, chain with starting troughs of prominence 1 mm).
    Only SR0 210-240 min (common to all three); run with SJ_VORM=2 (free fit with lee tail; the default)."""
    assert SHAPE == 2, "run with SJ_VORM=2"
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11.5, "axes.spines.top": False, "axes.spines.right": False})
    KL = {"vrij": "#6a4c93", "vast": "#2a7f62", "ket": "#1f4e79"}; VELD = "#b5651d"; GRIJS = "#8c8c8c"
    NAAM = {"vrij": "free fit", "vast": "fixed windows", "ket": "chain (prominence 1 mm)"}
    # ---- data: per method and series (model / noise) the columns I, S, H, R2
    D2 = np.load(OUT_FREE); DV = np.load(OUT_FIXED); DK = np.load(OUT_CHAIN); dat = {}
    for wie, nm in (("model", "SR0 210-240"), ("ruis", "ruis SR0 210-240")):
        P = D2[f"{nm}|pas"]; dat[("vrij", wie)] = P[:, [3, 4, 5, 6]]
        V = DV[f"{nm}|v"]; V = V[np.isfinite(V[:, 12])]; dat[("vast", wie)] = V[:, [13, 14, 15, 12]]; dat[("oud", wie)] = V[:, [4, 5, 6, 12]]
        R = DK[f"{nm}|r"]; R = R[R[:, 2] == 1.0]; dat[("ket", wie)] = R[:, [5, 6, 7, 9]]
    def maat(X, hmin=-np.inf):          # all ripples: also H < 0 (chain), as in table_chain
        X = X[X[:, 2] >= hmin]; I, S, H = X[:, 0], X[:, 1], X[:, 2]
        return dict(SI=I.mean() / S.mean(), ti=H.mean() / I.mean(), ts=H.mean() / S.mean(), n=len(X))
    fig = plt.figure(figsize=(12, 13.5)); gs = fig.add_gridspec(4, 3, height_ratios=[0.42, 0.42, 0.42, 1], hspace=0.42, wspace=0.32)
    # ---- (a-c) the same stretch of bed, three fits
    d = np.load(os.path.join(projmap, "c23_z271_SR0.npz")); h = M.smooth(d["h"][-1] * float(d["dz"])); h -= h.mean(); N = len(h); x = np.arange(N) * DX
    x0, x1 = 300, 700; k = (x >= x0) & (x <= x1); ylo, yhi = h[k].min() - 1.0, h[k].max() + 0.8
    def strook(ax, letter, wie, titel):
        ax.fill_between(x[k] - x0, h[k], ylo, color="#e9dcc3"); ax.plot(x[k] - x0, h[k], color="#6b5a3a", lw=1.0)
        ax.set_aspect(10); ax.set_xlim(0, x1 - x0); ax.set_ylim(ylo, yhi); ax.set_ylabel("height (mm)")
        ax.set_title(f"({letter}) {titel}", loc="left", fontsize=11.5, color=KL[wie])
    def tik(ax, xm, zm, kl):
        if 0 <= xm <= x1 - x0: ax.plot([xm, xm], [zm - 0.5, zm + 0.5], color=kl, lw=1)
    ax = fig.add_subplot(gs[0, :]); strook(ax, "a", "vrij", "Free fit: each window chosen separately on the best R$^2$ (gaps and small ripples possible)")
    for (a, nI, nS, H, z0, r2, e, qq, bb) in fit2(h):
        g = template(nI, nS, qq); n = len(g); zf = z0 + bb * np.arange(n) / n + H * g; xx = (a + np.arange(n)) * DX
        for sh in (0, N * DX):
            xs = xx - sh - x0
            if xs[-1] < 0 or xs[0] > x1 - x0: continue
            ax.plot(xs, zf, color=KL["vrij"], lw=1.8); tik(ax, xs[0], zf[0], KL["vrij"]); tik(ax, xs[-1], zf[-1], KL["vrij"])
    ax = fig.add_subplot(gs[1, :]); strook(ax, "b", "vast", "Fixed windows: trough to trough by the existing rule (prominence 3 mm), the shape fitted within each window")
    for (t0, n, c, Hold, idx) in _windows(h):
        p = fit_window(h[idx], 2)
        if p is None: continue
        R2, kI, H, q, bb = p; g = template(kI, n - kI, q); w = h[idx]; X = np.c_[np.ones(n + 1), np.arange(n + 1) / n]
        zf = X @ np.linalg.lstsq(X, w - H * g, rcond=None)[0] + H * g; xx = (t0 + np.arange(n + 1)) * DX
        for sh in (0, N * DX):
            xs = xx - sh - x0
            if xs[-1] < 0 or xs[0] > x1 - x0: continue
            ax.plot(xs, zf, color=KL["vast"], lw=1.8); tik(ax, xs[0], zf[0], KL["vast"])
    ax = fig.add_subplot(gs[2, :]); strook(ax, "c", "ket", "Chain: contiguous ripples with shared troughs; starting troughs prominence 1 mm")
    K = Chain(h, RM.troughs(h, 1.0)).fit().verfijn(); f = K.model(K.Tf, K.Cf, K.Y, K.H, K.Q)
    ax.plot(x[k] - x0, f[k], color=KL["ket"], lw=1.8)
    for t in K.Tf[:-1]: tik(ax, (t % N) * DX - x0, f[int(round(t)) % N], KL["ket"])
    ax.set_xlabel("distance along the x-axis (mm); wind from the left; seed 271 after 4 hours, height exaggerated 10 ×; ticks: troughs")
    # ---- (d) symmetry index
    ax = fig.add_subplot(gs[3, 0]); meth = ["vrij", "vast", "ket"]; xs = np.arange(3)
    for j, (hmin, dx_, lab) in enumerate(((-np.inf, -0.17, "all ripples"), (2.5, 0.17, "H ≥ 2.5 mm"))):
        for i, m in enumerate(meth):
            mm_, rr = maat(dat[(m, "model")], hmin)["SI"], maat(dat[(m, "ruis")], hmin)["SI"]
            ax.plot([i + dx_] * 2, [rr, mm_], color=KL[m], lw=1, alpha=0.6)
            ax.plot(i + dx_, mm_, "o" if hmin < 0 else "s", color=KL[m], ms=8); ax.plot(i + dx_, rr, "o" if hmin < 0 else "s", mfc="white", mec=KL[m], mew=1.5, ms=8)
            ax.text(i + dx_ + (0.1 if dx_ > 0 else -0.1), mm_, f"{mm_:.2f}", fontsize=8.5, va="center", ha="left" if dx_ > 0 else "right", color=KL[m])
            ax.text(i + dx_ + (0.1 if dx_ > 0 else -0.1), rr, f"{rr:.2f}", fontsize=8, va="center", ha="left" if dx_ > 0 else "right", color=KL[m], alpha=0.8)
    o_m, o_r = maat(dat[("oud", "model")], 2.5)["SI"], maat(dat[("oud", "ruis")], 2.5)["SI"]
    ax.axhline(o_m, color=GRIJS, lw=1.2, ls="-.", label=f"existing rule, H ≥ 2.5: {o_m:.2f}")
    ax.axhline(2.96, color=VELD, ls="--", lw=1.4); ax.text(-0.35, 3.03, "1983 field 2.96", color=VELD, fontsize=9.5); ax.axhline(1, color="0.6", ls=":", lw=1)
    ax.plot([], [], "o", color="0.3", label="model, all ripples"); ax.plot([], [], "s", color="0.3", label="model, H ≥ 2.5 mm"); ax.plot([], [], "o", mfc="white", mec="0.3", label="noise (open)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", bbox_to_anchor=(1, 0.84))
    ax.set_xticks(xs); ax.set_xticklabels(["free", "fixed\nwindows", "chain\n1 mm"]); ax.set_xlim(-0.65, 2.65); ax.set_ylim(0, 3.4)
    ax.set_ylabel("symmetry index SI = I : S (from the means)"); ax.set_title("(d) Symmetry index", loc="left", fontsize=12)
    # ---- (e) slopes
    ax = fig.add_subplot(gs[3, 1]); t = np.linspace(0, 0.2, 2)
    for si, ls in ((1, ":"), (2, "--"), (3, "-.")):
        ax.plot(t, si * t, ls, color="0.75", lw=1); xe = {1: 0.1, 2: 0.095, 3: 0.048}[si]; ax.text(xe + 0.002, si * xe, f"SI {si}", fontsize=8.5, color="0.55", ha="left" if si > 1 else "right", va="top" if si > 1 else "bottom")
    for m in meth:
        a, b = maat(dat[(m, "model")], 2.5), maat(dat[(m, "ruis")], 2.5)
        ax.plot([b["ti"], a["ti"]], [b["ts"], a["ts"]], color=KL[m], lw=1, alpha=0.6)
        ax.plot(a["ti"], a["ts"], "s", color=KL[m], ms=8); ax.plot(b["ti"], b["ts"], "s", mfc="white", mec=KL[m], mew=1.5, ms=8)
    a = maat(dat[("oud", "model")], 2.5); ax.plot(a["ti"], a["ts"], "D", color=GRIJS, ms=7)
    ax.plot(0.063, 0.185, "*", color=VELD, ms=15); ax.text(0.066, 0.185, "1983 field", color=VELD, fontsize=9.5, va="center")
    ax.annotate("existing rule", (a["ti"], a["ts"]), xytext=(0.03, 0.125), color=GRIJS, fontsize=8.5, arrowprops=dict(arrowstyle="-", color=GRIJS, lw=0.8))
    ax.set_xlim(0, 0.1); ax.set_ylim(0, 0.2); ax.set_xlabel("tan i = H / I (stoss)"); ax.set_ylabel("tan s = H / S (lee)")
    ax.set_title("(e) Slopes, H ≥ 2.5 mm (open: noise)", loc="left", fontsize=12)
    # ---- (f) R2 per ripple
    ax = fig.add_subplot(gs[3, 2]); rr = np.linspace(0, 1, 201)
    for m in meth:
        for wie, ls in (("model", "-"), ("ruis", "--")):
            X = dat[(m, wie)]; X = X[X[:, 2] >= 2.5]; r2 = np.clip(X[:, 3], 0, 1)
            ax.plot(rr, [np.mean(r2 >= v) for v in rr], ls, color=KL[m], lw=2 if wie == "model" else 1.3,
                    label=f"{NAAM[m]}:\nmedian {np.median(r2):.2f} (noise {np.median(np.clip(dat[(m, 'ruis')][dat[(m, 'ruis')][:, 2] >= 2.5][:, 3], 0, 1)):.2f})" if wie == "model" else None)
    ax.plot([], [], "--", color="0.4", label="noise (same spectrum)")
    ax.set_xlabel("R$^2$ per ripple (threshold)"); ax.set_ylabel("fraction of ripples with R$^2$ ≥ threshold"); ax.set_ylim(0, 1.02); ax.set_xlim(0, 1)
    ax.legend(frameon=False, fontsize=8.5, loc="center left", bbox_to_anchor=(0, 0.42), labelspacing=0.9); ax.set_title("(f) Fit per ripple, H ≥ 2.5 mm", loc="left", fontsize=12)
    uit = os.path.join(HERE, "figures", "fig6_template_fits.png"); fig.savefig(uit, dpi=160, bbox_inches="tight"); print("geschreven:", uit)
    for m in meth + ["oud"]:
        for hmin in (-np.inf, 2.5):
            a, b = maat(dat[(m, "model")], hmin), maat(dat[(m, "ruis")], hmin)
            print(f"{m:5s} H>={hmin}: model SI {a['SI']:.2f} tan i {a['ti']:.3f} tan s {a['ts']:.3f} (n {a['n']}) | noise SI {b['SI']:.2f} tan i {b['ti']:.3f} tan s {b['ts']:.3f} (n {b['n']})")


# ------------------------------------------------------------------ test with the mirrored template (Figure 7)
OUT_MIRROR = os.path.join(HERE, "output", "template_mirror_ripples.npz")
METH = ("V", "K", "F")   # fixed windows, chain (prominence 1), free fit


def mirror_window(w):
    """R2 of the adjusted template as it is and mirrored (= fit on the reversed window); no fit with H > 0 counts as R2 0.
    Returns (R2_right, R2_mirror, kI_right, kI_mirror); kI_mirror counted in the reversed window."""
    a, b = fit_window(w, 2), fit_window(w[::-1], 2)
    return (a[0] if a else 0.0), (b[0] if b else 0.0), (a[1] if a else -1), (b[1] if b else -1)


def _hv(w):
    n = len(w) - 1; return float((w - (w[0] + (w[-1] - w[0]) * np.arange(n + 1) / n)).max())


def _work_mirror(arg):
    f, k, z, t, ruis, KF = arg
    d = np.load(f); h = M.smooth(d["h"][k] * float(d["dz"])); h = h - h.mean(); N = len(h)
    if ruis:
        rf = np.random.default_rng(1000 * z + k).uniform(0, 2 * np.pi, N // 2 + 1); Zf = np.fft.rfft(h); ph = np.exp(1j * rf[:len(Zf)]); ph[0] = 1; ph[-1] = 1
        h = np.fft.irfft(np.abs(Zf) * ph, N)
    vensters = [(0, t0, n) for (t0, n, c, Hold, idx) in _windows(h)] + [(mi, t0, n) for mi, L in ((1, KF[0]), (2, KF[1])) for (t0, n) in L]
    rows = []
    for mi, t0, n in vensters:
        if n < 8: continue
        w = h[(t0 + np.arange(n + 1)) % N]; g, s, kg, ks = mirror_window(w)
        rows.append((mi, z, t, n * DX, _hv(w), g, s, kg * DX, ks * DX))
    return rows
# columns: 0 method (0 V, 1 K, 2 F), 1 seed, 2 t, 3 L window (mm), 4 H_v, 5 R2 right, 6 R2 mirror, 7 crest right (mm from the upwind trough; -1 none), 8 crest mirror (in the reversed window)


def fit_mirror(projmap):
    from multiprocessing import Pool
    DK, DF = np.load(OUT_CHAIN), np.load(os.path.join(HERE, "output", "template2_ripples.npz")); res = {}
    sets = [("SR0 210-240", projmap, "SR0", False), ("ruis SR0 210-240", projmap, "SR0", True)] + [(f"SL16 {a}-{b}", projmap, "SL16", False) for a, b in ((450, 480), (690, 720), (930, 960))]
    for naam, map_, suf, ruis in sets:
        R = DK[f"{naam}|r"]; R = R[R[:, 2] == 1.0]; P = DF[f"{naam}|pas"]; taken = []
        for z in SEEDS:
            f = os.path.join(map_, f"c23_z{z}_{suf}.npz"); T = np.load(f)["t"]
            for t in np.unique(R[R[:, 0] == z][:, 1]):
                k = int(np.argmin(np.abs(T - t))); assert abs(T[k] - t) < 1e-6
                r = R[(R[:, 0] == z) & (R[:, 1] == t)]; p = P[(P[:, 0] == z) & (P[:, 1] == t)]
                K = [(int(round(a / DX)), int(round((a + i + s) / DX)) - int(round(a / DX))) for a, i, s in r[:, 4:7]]
                F = [(int(round(a / DX)), int(round(i / DX)) + int(round(s / DX))) for a, i, s in p[:, 2:5]]
                if len(p) == 0: print("  note: no free fit for", naam, z, t)
                taken.append((f, k, z, float(t), ruis, (K, F)))
        with Pool(3, maxtasksperchild=20) as pl: out = pl.map(_work_mirror, taken, chunksize=2)
        res[naam] = (np.array([r for o in out for r in o], float), len(taken))
        print(naam, len(taken), "records", len(res[naam][0]), "windows", flush=True)
        np.savez(OUT_MIRROR, **{f"{k}|v": v[0] for k, v in res.items()}, **{f"{k}|n": v[1] for k, v in res.items()})


def test_mirror():
    rng = np.random.default_rng(7); ok = True
    fa = []
    for _ in range(30):
        nI = int(rng.integers(60, 300)); nS = int(rng.integers(30, 150)); q = float(rng.choice(QV)); n = nI + nS
        w = 0.5 + rng.uniform(-2, 2) * np.arange(n + 1) / n + rng.uniform(2, 8) * template(nI, nS, q); g, s, kg, ks = mirror_window(w); fa.append((g, s))
    fa = np.array(fa); a_ok = np.all(np.abs(fa[:, 0] - 1) < 1e-9) and np.all(fa[:, 1] < 1 - 1e-6)
    print(f"(a) 30 artificial windows without noise: R2 right min {fa[:, 0].min():.12f}, R2 mirror max {fa[:, 1].max():.4f} (mean {fa[:, 1].mean():.3f}): {a_ok}")
    fb = []
    for _ in range(30):
        n = int(rng.integers(80, 400)); w = np.cumsum(rng.normal(size=n + 1)) * 0.1 + 2 * np.sin(np.pi * np.arange(n + 1) / n) * rng.uniform(0.5, 2)
        g, s, *_ = mirror_window(w); g2, s2, *_ = mirror_window(w[::-1]); fb.append(abs((g - s) + (g2 - s2)))
    b_ok = max(fb) < 1e-12; print(f"(b) mirroring exact on 30 noise windows: max |dR2(w) + dR2(w reversed)| {max(fb):.2g}: {b_ok}")
    fc = []
    for _ in range(10):
        m = int(rng.integers(40, 200)); half = rng.uniform(1, 5) * template(m, m, 1.0)[:m + 1] + 0.05 * rng.normal(size=m + 1)
        w = np.r_[half, half[::-1][1:]]; g, s, *_ = mirror_window(w); fc.append(abs(g - s))
    c_ok = max(fc) < 1e-12; print(f"(c) 10 symmetric windows: max |dR2| {max(fc):.2g}: {c_ok}")
    ok = a_ok and b_ok and c_ok; print("TEST", "MET" if ok else "NOT MET")


def table_mirror():
    D = np.load(OUT_MIRROR); LB = 5350 * DX / 1000; NM = {0: "V fixed windows", 1: "K chain (prom. 1)", 2: "F free fit"}
    o = ["TEST WITH THE MIRRORED TEMPLATE (template_fit.py fit_mirror; fixed in advance on 16 Nov. 2026). dR2 = R2 template as it is - R2 mirrored (steep side facing the wind).",
         "Per window both fits with the adjusted template; H_v = highest point above the trough line (independent of the direction). Main group H_v >= 2.5 mm.",
         "T1 (per method): median dR2 (H_v >= 2.5) model > noise in 4/4 seeds, SR0 210-240 min."]
    def regel(lab, X, nop):
        if len(X) == 0: return f"    {lab:22s} none"
        dr = X[:, 5] - X[:, 6]
        return (f"    {lab:22s} {len(X) / nop / LB:5.1f}/m  dR2 median {np.median(dr):+.3f}  mean {dr.mean():+.3f}  share > 0 {np.mean(dr > 0):.2f}  share < 0 {np.mean(dr < 0):.2f}"
                f" | R2 right med. {np.median(X[:, 5]):.3f}  spiegel {np.median(X[:, 6]):.3f}")
    for nm in sorted({k.split("|")[0] for k in D.files}, key=lambda s: (s.startswith("ruis"), s.startswith("SL16"), s)):
        V = D[f"{nm}|v"]; nop = int(D[f"{nm}|n"]); o.append(f"\n=== {nm} ({nop} records) ===")
        for mi in (0, 1, 2):
            X = V[V[:, 0] == mi]; o.append(f"  {NM[mi]}"); o.append(regel("all windows", X, nop)); o.append(regel("H_v >= 2.5 mm", X[X[:, 4] >= 2.5], nop))
            for a, b in ((2.5, 4), (4, 6), (6, 99)): o.append(regel(f"  H_v {a}-{b if b < 99 else ''} mm", X[(X[:, 4] >= a) & (X[:, 4] < b)], nop))
            Y = X[(X[:, 4] >= 2.5) & (X[:, 7] > 0)]
            if len(Y) > 5:
                from scipy.stats import spearmanr
                si = Y[:, 7] / (Y[:, 3] - Y[:, 7]); o.append(f"    relation of dR2 with SI of the fit (Spearman, H_v >= 2.5): {spearmanr(Y[:, 5] - Y[:, 6], si)[0]:+.2f}")
    o.append("\n=== T1: median dR2 (H_v >= 2.5) per seed, SR0 210-240: model / noise ===")
    M, Rr = D["SR0 210-240|v"], D["ruis SR0 210-240|v"]
    for mi in (0, 1, 2):
        cel = []; n_ok = 0
        for z in SEEDS:
            a = M[(M[:, 0] == mi) & (M[:, 1] == z) & (M[:, 4] >= 2.5)]; b = Rr[(Rr[:, 0] == mi) & (Rr[:, 1] == z) & (Rr[:, 4] >= 2.5)]
            ma, mb = np.median(a[:, 5] - a[:, 6]), np.median(b[:, 5] - b[:, 6]); n_ok += ma > mb; cel.append(f"z{z} {ma:+.3f} / {mb:+.3f}")
        o.append(f"  {NM[mi]:22s} " + "  ".join(cel) + f"  -> {n_ok}/4: T1 {'MET' if n_ok == 4 else 'NOT MET'}")
    s = "\n".join(o); print(s)
    with open(os.path.join(HERE, "output", "template_mirror.txt"), "w") as f: f.write(s + "\n")


def fig7():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11.5, "axes.spines.top": False, "axes.spines.right": False})
    D = np.load(OUT_MIRROR); M, Rr = D["SR0 210-240|v"], D["ruis SR0 210-240|v"]
    NM = {0: "(a) Fixed windows", 1: "(b) Chain (prominence 1 mm)", 2: "(c) Free fit"}
    fig, axs = plt.subplots(1, 4, figsize=(16, 4.6), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.1], wspace=0.35)); bins = np.linspace(-0.15, 0.15, 61)
    for mi in (0, 1, 2):
        ax = axs[mi]
        for X, kl, lab in ((M, "#1f4e79", "model"), (Rr, "#b5651d", "noise (same spectrum)")):
            X = X[(X[:, 0] == mi) & (X[:, 4] >= 2.5)]; dr = X[:, 5] - X[:, 6]
            ax.hist(np.clip(dr, -0.149, 0.149), bins, weights=np.full(len(dr), 1 / len(dr)), histtype="step", lw=2, color=kl, label=f"{lab}:\nmedian {np.median(dr):+.3f}, ΔR² > 0 in {np.mean(dr > 0):.0%}")
        ax.axvline(0, color="0.5", lw=1, ls=":"); ax.set_xlabel("ΔR² = R² as it is − R² mirrored\n(extreme values at the edge)"); ax.set_title(NM[mi], loc="left", fontsize=12)
        ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.27)); ax.set_ylabel("fraction of windows" if mi == 0 else "")
    ax = axs[3]
    for mi, kl in ((0, "#2a7f62"), (1, "#1f4e79"), (2, "#6a4c93")):
        for X, mk in ((M, "o"), (Rr, "o")):
            y = [np.median((X[(X[:, 0] == mi) & (X[:, 1] == z) & (X[:, 4] >= 2.5)][:, 5] - X[(X[:, 0] == mi) & (X[:, 1] == z) & (X[:, 4] >= 2.5)][:, 6])) for z in SEEDS]
            ax.plot(np.arange(4) + 0.12 * (mi - 1), y, mk, color=kl, mfc=kl if X is M else "white", mew=1.5, ms=7)
    ax.axhline(0, color="0.5", lw=1, ls=":"); ax.set_xticks(range(4)); ax.set_xticklabels([str(z) for z in SEEDS]); ax.set_xlabel("seed")
    for mi, kl, lab in ((0, "#2a7f62", "fixed windows"), (1, "#1f4e79", "chain"), (2, "#6a4c93", "free")): ax.plot([], [], "o", color=kl, label=lab)
    ax.plot([], [], "o", mfc="white", mec="0.3", label="noise (open)"); ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2)
    ax.set_ylabel("median ΔR²"); ax.set_title("(d) Per seed (H ≥ 2.5 mm)", loc="left", fontsize=12)
    fig.suptitle("Does the template win in the right direction? SR0 210-240 min, windows with H ≥ 2.5 mm", x=0.07, ha="left", fontsize=12.5, y=1.02)
    uit = os.path.join(HERE, "figures", "fig7_mirrored_template.png"); fig.savefig(uit, dpi=160, bbox_inches="tight"); print("geschreven:", uit)


if __name__ == "__main__":
    PROJ = sys.argv[2] if len(sys.argv) > 2 else "."   # folder with the runs (default: current folder)
    {"test_free": test_free, "fit_free": lambda: fit_free(PROJ), "table_free": table_free, "test_fixed": test_fixed, "fit_fixed": lambda: fit_fixed(PROJ),
     "table_fixed": table_fixed, "test_chain": test_chain, "fit_chain": lambda: fit_chain(PROJ),
     "table_chain": lambda: table_chain(), "fig6": lambda: fig6(PROJ), "test_mirror": test_mirror, "fit_mirror": lambda: fit_mirror(PROJ), "table_mirror": table_mirror, "fig7": fig7}[sys.argv[1]]()
