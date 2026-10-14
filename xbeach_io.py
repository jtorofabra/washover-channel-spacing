"""Read the global output of an XBeach 2DH run (xboutput.nc).

XBeach stores 2D variables as (time, ny, nx). Here they are returned as
(nt, nx, ny), so that zb[k] is one frame with x (cross-shore) on the first axis
and y (alongshore) on the second, as ncread gives them in MATLAB.
"""

import numpy as np
import netCDF4


def _as_float(a):
    # Dry cells of zs, u, ... carry the _FillValue (-3.4e38): turn them into NaN.
    return np.ma.filled(np.ma.asarray(a, dtype=float), np.nan)


def read_grid(path):
    """Cross-shore x (nx,) and alongshore y (ny,) axes in m, and output times t (nt,) in s.

    x is usually non-uniform; y is uniform.
    """
    with netCDF4.Dataset(path) as nc:
        x = _as_float(nc["globalx"][0, :])
        y = _as_float(nc["globaly"][:, 0])
        t = _as_float(nc["globaltime"][:])
    return x, y, t


def read_field(path, name="zb"):
    """A 2D output variable as an array (nt, nx, ny), with NaN in dry cells.

    zb of a 506 x 601 grid with 65 frames takes about 160 MB.
    """
    with netCDF4.Dataset(path) as nc:
        a = _as_float(nc[name][:])
    return np.transpose(a, (0, 2, 1))


def alongshore_length(y):
    """Length Ly of a periodic alongshore domain (cyclic = 1): one cell more than y[-1] - y[0]."""
    dy = float(np.mean(np.diff(y)))
    return float(y[-1] - y[0] + dy), dy
