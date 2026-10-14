"""From one xboutput.nc to the crest signal h'(y, t), lambda_F(t), lambda_W(t) and lambda_inf.

    python example.py path/to/xboutput.nc [figure.png]
"""

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from xbeach_io import read_grid, read_field, alongshore_length
from crest import crest_signal
from estimators import lambda_fourier, lambda_wavelet
from ofat import late_spacing


def main(path, figure="example.png"):
    x, y, t = read_grid(path)
    zb = read_field(path, "zb")
    Ly, dy = alongshore_length(y)

    hp, xc = crest_signal(zb, x)
    lam_F = lambda_fourier(hp, Ly)
    lam_W = lambda_wavelet(hp, dy)

    print(f"Ly = {Ly:.0f} m, {t.size} frames, crest from x = {xc[0]:.0f} to {xc[-1]:.0f} m")
    for name, lam in (("lambda_F", lam_F), ("lambda_W", lam_W)):
        print(f"{name}: lambda_inf = {late_spacing(t, lam):.1f} m (30-70 h), "
              f"{late_spacing(t, lam, (50, 70)):.1f} m (50-70 h)")

    th = t / 3600
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    v = np.nanmax(np.abs(hp))
    im = a1.pcolormesh(th, y, hp, cmap="RdBu_r", vmin=-v, vmax=v, shading="nearest")
    fig.colorbar(im, ax=a1, label="h' (m)")
    a1.set_ylabel("y (m)")
    a2.plot(th, lam_F, label=r"$\lambda_F$ (Fourier centroid)")
    a2.plot(th, lam_W, "--", label=r"$\lambda_W$ (wavelet)")
    a2.axvspan(30, 70, color="0.9", zorder=0)
    a2.set_xlabel("storm time (h)")
    a2.set_ylabel(r"$\lambda$ (m)")
    a2.legend()
    fig.tight_layout()
    fig.savefig(figure, dpi=150)
    print(f"-> {figure}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(*sys.argv[1:3])
