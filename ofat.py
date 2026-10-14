"""Late spacing and one-factor-at-a-time (OFAT) sensitivities.

The sensitivity to a parameter X is the slope b of log lambda_inf = a + b log X,
fitted along one OFAT axis (all other parameters at their reference values).
Runs that share a seed start from the same initial bed, so the line is fitted
seed by seed and the slopes are averaged: the seed-to-seed noise cancels within
each fit. The 95 % interval is a Student t over the per-seed slopes
(2 degrees of freedom with three seeds).
"""

import numpy as np
from scipy import stats


def late_spacing(t, lam, window=(30.0, 70.0)):
    """lambda_inf: mean of lambda over the plateau window (hours of storm time).

    t in s. The default 30-70 h is the main window; 50-70 h is the control.
    """
    th = np.asarray(t) / 3600
    m = (th >= window[0]) & (th <= window[1])
    return float(np.mean(lam[m])) if m.any() else np.nan


def paired_slope(x, lam_inf, seed, level=0.95):
    """OFAT slope b, its confidence interval (lo, hi) and the per-seed slopes.

    x, lam_inf and seed have one entry per run of the axis. Seeds with fewer
    than two runs are skipped.
    """
    x, lam_inf, seed = np.asarray(x, float), np.asarray(lam_inf, float), np.asarray(seed)
    ok = np.isfinite(x) & np.isfinite(lam_inf) & (x > 0) & (lam_inf > 0)
    lx, ll, seed = np.log(x[ok]), np.log(lam_inf[ok]), seed[ok]
    slopes = np.array([np.polyfit(lx[seed == s], ll[seed == s], 1)[0]
                       for s in np.unique(seed) if np.sum(seed == s) >= 2])
    if slopes.size == 0:
        return np.nan, (np.nan, np.nan), slopes
    b = slopes.mean()
    if slopes.size < 2:
        return b, (np.nan, np.nan), slopes
    half = stats.t.ppf(0.5 + level / 2, slopes.size - 1) * slopes.std(ddof=1) / np.sqrt(slopes.size)
    return b, (b - half, b + half), slopes
