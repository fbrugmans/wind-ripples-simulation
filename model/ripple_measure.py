"""Ripple geometry from a bed profile, following the field method of Brugmans (1983).

Per ripple (as with a plexiglass plate laid over the crests):
  L  = crest to the next crest downwind
  S  = crest to the first trough downwind (chord, along the x-axis)
  I  = L - S
  H  = depth of the trough below the chord between the two crests
  SI = I/S, RI = L/H, tan s = H/S, tan i = H/I
  plus the MAXIMUM local lee slope (the steep upper part), which exists in the field but is not measured with the chord.
"""
import numpy as np
from scipy.signal import find_peaks


def troughs(h, prom):
    """Troughs of a periodic profile h with at least the given prominence (indices, sorted)."""
    N = len(h)
    hh = np.concatenate([h, h, h])
    idx, _ = find_peaks(-hh, prominence=prom)
    idx = idx[(idx >= N) & (idx < 2 * N)] - N
    return np.sort(idx)


def ripples(x, h, s=+1, prom_frac=0.10, prom_min=0.05):
    """List of ripples with the indices of crest (c), trough (t) and next crest (c2) and L, S, I, H. Periodic.
    With s = -1 the profile is mirrored so that downwind is always +x."""
    if s < 0:
        h = h[::-1]
    N = len(h)
    dx = float(x[1] - x[0])
    prom = max(prom_frac * float(np.ptp(h)), prom_min)
    tr = troughs(h, prom)
    if len(tr) < 2:
        return [], h, dx
    crests = []
    for j, t0 in enumerate(tr):
        t1 = tr[(j + 1) % len(tr)]
        n = int((t1 - t0) % N) or N
        seg = (t0 + np.arange(1, n)) % N
        crests.append(int(seg[int(np.argmax(h[seg]))]))
    crests = np.array(sorted(set(crests)), dtype=int)
    out = []
    for ci, c in enumerate(crests):
        c2 = crests[(ci + 1) % len(crests)]
        d = (tr - c) % N
        d[d == 0] = N
        t = int(tr[int(np.argmin(d))])
        S = int((t - c) % N)
        Lc = int((c2 - c) % N) or N
        I = Lc - S
        if S <= 0 or I <= 0:
            continue
        w = S / Lc
        H = (1 - w) * h[c] + w * h[c2] - h[t]
        if H <= 0:
            continue
        lee = (c + np.arange(0, S + 1)) % N
        hx = np.diff(h[lee]) / dx
        out.append(dict(c=c, t=t, c2=int(c2), L=Lc * dx, S=S * dx, I=I * dx, H=H,
                        max_lee=float(-hx.min()) if len(hx) else np.nan))
    return out, h, dx


def geometry(x, h, s=+1, **kw):
    """Medians (and the mean SI) over the ripples of one profile."""
    rip, _, _ = ripples(x, h, s=s, **kw)
    if not rip:
        return dict(n=0)
    a = {k: np.array([r[k] for r in rip]) for k in ("L", "S", "I", "H", "max_lee")}
    return dict(n=len(rip), L_med=float(np.median(a["L"])), H_med=float(np.median(a["H"])),
                SI_med=float(np.median(a["I"] / a["S"])),
                SI_mean=float(np.mean(a["I"] / a["S"])),
                RI_med=float(np.median(a["L"] / a["H"])),
                S_over_L=float(np.median(a["S"] / a["L"])),
                tan_s_med=float(np.median(a["H"] / a["S"])),
                tan_i_med=float(np.median(a["H"] / a["I"])),
                max_lee_med=float(np.median(a["max_lee"])))
