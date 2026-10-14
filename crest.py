"""Crest signal h'(y, t): the alongshore perturbation of the bed at the barrier crest.

For each frame, the alongshore-mean profile is removed from zb and the remaining
perturbation is averaged in x with a Gaussian centred on the crest:

    h'(y, t) = sum_x w(x - xc(t)) [zb(x, y, t) - mean_y zb(x, :, t)]

The crest position xc(t) is tracked inside a search window and smoothed in time,
so the Gaussian follows the crest as it migrates landward during the storm.
"""

import numpy as np


def _local_maxima(p):
    """Indices of the interior local maxima of a 1D profile."""
    return np.flatnonzero((p[1:-1] >= p[:-2]) & (p[1:-1] > p[2:])) + 1


def track_crest(base, x, win=(1550.0, 1650.0), step_max=5.0, margin=0.02):
    """Crest position xc (nt,) in m from the alongshore-mean profiles base (nx, nt).

    The tracked crest is the highest peak in the window, but the tracker only
    switches to another peak if it is higher by more than `margin` (m), or if the
    tracked one has moved more than `step_max` (m) or disappeared.
    """
    iw = np.flatnonzero((x >= win[0]) & (x <= win[1]))
    nt = base.shape[1]
    xc = np.full(nt, np.nan)
    current = None                      # index (within iw) of the tracked crest
    for k in range(nt):
        p = base[iw, k]
        if current is None:
            i = int(np.nanargmax(p))
            found = 0 < i < iw.size - 1   # a maximum on the window edge is not a crest
        else:
            peaks = _local_maxima(p)
            found = peaks.size > 0
            if found:
                near = peaks[np.argmin(np.abs(peaks - current))]
                same = abs(x[iw[near]] - x[iw[current]]) <= step_max
                high = peaks[np.argmax(p[peaks])]
                # hysteresis: crests of almost equal height would make the tracker
                # jump back and forth between the eroding crest and the new one
                i = high if (not same or p[high] > p[near] + margin) else near
        if found:
            xc[k] = x[iw[i]]
            current = i
        elif k > 0 and np.isfinite(xc[k - 1]):
            xc[k] = xc[k - 1]
        else:
            xc[k] = np.mean(win)
    return xc


def rlowess(y, n, iterations=5):
    """Robust local linear smoothing of y over windows of n points.

    Same as MATLAB smoothdata(y, 'rlowess', n): tricube weights, bisquare
    robustness weights (cut at 6 MAD), window shifted inwards at the edges.
    """
    y = np.asarray(y, dtype=float)
    N = y.size
    if n <= 1 or N < 3:
        return y.copy()
    n = min(n, N)
    before = n // 2
    after = n // 2 if n % 2 else n // 2 - 1
    t = np.arange(N, dtype=float)
    robust = np.ones(N)
    out = y.copy()
    for it in range(iterations + 1):
        for i in range(N):
            a, b = i - before, i + after + 1
            if a < 0:
                a, b = 0, b - a
            if b > N:
                a, b = a - (b - N), N
            d = np.abs(t[a:b] - t[i])
            w = (1 - (d / d.max()) ** 3) ** 3 * robust[a:b]
            if w.sum() <= 0:
                out[i] = y[i]
                continue
            if np.count_nonzero(w) == 1:
                out[i] = y[a:b][w > 0][0]
                continue
            X = np.column_stack([np.ones(b - a), t[a:b] - t[i]])
            sw = np.sqrt(w)
            coef, *_ = np.linalg.lstsq(X * sw[:, None], y[a:b] * sw, rcond=None)
            out[i] = coef[0]
        if it == iterations:
            break
        r = y - out
        mad = np.median(np.abs(r))
        if mad == 0:
            break
        u = r / (6 * mad)
        robust = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)
    return out


def gaussian_collapse(zb_frame, x, xc, L_x=20.0):
    """h'(y) (ny,) of one frame zb_frame (nx, ny), with a Gaussian of half-width L_x (m) at xc."""
    zp = zb_frame - zb_frame.mean(axis=1, keepdims=True)
    w = np.exp(-(((x - xc) / L_x) ** 2))
    w /= w.sum()
    use = w > 1e-3
    return w[use] @ zp[use, :]


def crest_signal(zb, x, L_x=20.0, win=(1550.0, 1650.0), smooth_frac=0.10,
                 step_max=5.0, margin=0.02):
    """Crest signal hp (ny, nt) in m and smoothed crest position xc (nt,) from zb (nt, nx, ny).

    The smoothing window is a fraction of the number of frames (at least 3),
    so it covers the same part of the storm whatever the output interval.
    """
    nt = zb.shape[0]
    base = zb.mean(axis=2).T                          # alongshore-mean profile (nx, nt)
    xc = track_crest(base, x, win, step_max, margin)
    n_smooth = max(3, int(np.floor(smooth_frac * nt + 0.5)))   # round half up, as MATLAB
    xc = rlowess(xc, n_smooth)
    hp = np.stack([gaussian_collapse(zb[k], x, xc[k], L_x) for k in range(nt)], axis=1)
    return hp, xc
