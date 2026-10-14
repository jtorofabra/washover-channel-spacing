"""Morphological clocks and the equal-clock test.

A clock is a cumulative quantity that grows during the storm: the storm time t,
the incident wave energy E_inc, the overwash volume V_ov or the deposited volume
Sigma+. Runs that differ in one parameter are compared at the same value c* of a
clock instead of at the same time. The residual spread says how much of the
final difference in spacing is left at equal clock (0: the clock accounts for
all of it; 1: for none).
"""

import numpy as np


def _cell_widths(x):
    """Trapezoidal quadrature weights (m) on a non-uniform grid x."""
    w = np.zeros(x.size)
    w[0] = (x[1] - x[0]) / 2
    w[1:-1] = (x[2:] - x[:-2]) / 2
    w[-1] = (x[-1] - x[-2]) / 2
    return w


def incident_energy(t, hm0, rt=3600.0):
    """E_inc(t) = integral of Hm0^2 dt (m^2 s) at the output times t (s).

    hm0 is the first column of jonswap.txt, one value per rt seconds from t = 0.
    It comes from the forcing, so it does not depend on the output interval.
    """
    tj = np.arange(hm0.size + 1) * rt
    cum = np.concatenate([[0.0], np.cumsum(hm0 ** 2 * rt)])
    return np.interp(np.minimum(t, tj[-1]), tj, cum)


def overwash_discharge(zb, zs, u, x, dy, win=(1550.0, 1650.0)):
    """Landward discharge Q_ov (nt,) in m^3/s through the control section near the crest.

    The control section is, for each y, the point of minimum water depth inside
    the crest window. Dry cells are NaN in zs and u and count as zero flow.
    """
    iw = np.flatnonzero((x >= win[0]) & (x <= win[1]))
    cols = np.arange(zb.shape[2])
    Q = np.zeros(zb.shape[0])
    for k in range(zb.shape[0]):
        depth = zs[k][iw, :] - zb[k][iw, :]
        j = np.argmin(np.where(np.isnan(depth), np.inf, depth), axis=0)
        h = np.fmax(0.0, depth[j, cols])                # all-NaN column -> 0
        Q[k] = np.sum(h * np.fmax(0.0, u[k][iw[j], cols])) * dy
    return Q


def overwash_volume(t, Q):
    """V_ov(t) in m^3: cumulative trapezoidal integral of Q_ov over t (s)."""
    return np.concatenate([[0.0], np.cumsum(0.5 * (Q[1:] + Q[:-1]) * np.diff(t))])


def deposited_volume(zb, x, dy, band=(1450.0, 1750.0)):
    """Sigma+(t) in m^3: cumulative sum of the bed rises between frames in the band (x, m).

    Only increases of zb count, so erosion elsewhere does not cancel deposition.
    """
    iz = np.flatnonzero((x >= band[0]) & (x <= band[1]))
    wx = _cell_widths(x)[iz][:, None]
    sigma = np.zeros(zb.shape[0])
    for k in range(1, zb.shape[0]):
        rise = np.maximum(0.0, zb[k][iz, :] - zb[k - 1][iz, :])
        sigma[k] = sigma[k - 1] + np.sum(rise * wx) * dy
    return sigma


def smooth(lam, n=5):
    """Centred moving average over n frames (shorter windows at the ends)."""
    lam = np.asarray(lam, dtype=float)
    h = n // 2
    return np.array([np.nanmean(lam[max(0, i - h):i + n - h]) for i in range(lam.size)])


def lambda_at_clock(clock, lam, c, n_smooth=5):
    """Smoothed spacing when the clock reaches c (linear interpolation); NaN if it never does."""
    lam = smooth(lam, n_smooth)
    k = np.flatnonzero(clock >= c)
    if k.size == 0:
        return np.nan
    k = k[0]
    if k == 0 or clock[k] == clock[k - 1]:
        return lam[k]
    f = (c - clock[k - 1]) / (clock[k] - clock[k - 1])
    return lam[k - 1] + f * (lam[k] - lam[k - 1])


def reference_clock(clocks, lams, lam_inf, progress=0.6, n_grid=200, n_smooth=5):
    """Reference value c*: the first clock value at which the median run has covered
    `progress` of the way from its initial spacing to lam_inf.

    clocks and lams are lists with one series per run (raw lambda, not smoothed);
    lam_inf the late spacing of each run. Runs that barely grow (< 1 m) are ignored.
    """
    end = [float(c[-1]) for c in clocks]

    def median_progress(c):
        p = [(lambda_at_clock(ck, lk, c, n_smooth) - lk[0]) / (li - lk[0])
             for ck, lk, li, e in zip(clocks, lams, lam_inf, end) if e >= c and li - lk[0] > 1]
        return np.median(p) if p else np.nan

    grid = np.linspace(min(float(c[0]) for c in clocks), max(end), n_grid)[1:]
    return next((c for c in grid if median_progress(c) >= progress), max(end))


def residual_spread(level, lam_c, lam_inf):
    """Range of log lambda at c* over the range of log lam_inf, across parameter levels.

    Use it on the runs of one seed. Spacings are averaged per level first; levels
    that never reach c* (NaN in lam_c) are left out of both ranges.
    """
    level, lam_c, lam_inf = (np.asarray(v, dtype=float) for v in (level, lam_c, lam_inf))
    ok = np.isfinite(lam_c)
    levels = np.unique(level[ok])
    if levels.size < 2:
        return np.nan
    at_c = np.log([lam_c[ok & (level == v)].mean() for v in levels])
    late = np.log([lam_inf[level == v].mean() for v in levels])
    if np.ptp(late) == 0:
        return np.nan
    return np.ptp(at_c) / np.ptp(late)
