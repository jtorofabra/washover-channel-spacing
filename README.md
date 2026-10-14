# Washover channel spacing

Methods to measure the alongshore spacing λ between washover channels in XBeach 2DH
simulations of barrier overwash, and to test what sets it. They were written for a
research internship in the MORFO group (UPC) on the question: which physical mechanism
selects the channel spacing during an overwash storm? Each script holds one method and
works on any `xboutput.nc` with a global `zb` output.

## Files

- `xbeach_io.py`: reads the grid (non-uniform x, uniform y), output times and 2D fields of `xboutput.nc`.
- `crest.py`: crest tracking with hysteresis and Gaussian collapse of `zb` into the crest signal h'(y, t).
- `estimators.py`: spacing from h': Fourier spectral centroid λ_F and corrected wavelet centroid λ_W.
- `clocks.py`: morphological clocks (E_inc, V_ov, Σ⁺), reference value c*, λ at equal clock and residual spread.
- `ofat.py`: late spacing λ∞ (plateau mean) and OFAT slopes with paired seeds and Student-t intervals.
- `example.py`: from one `xboutput.nc` to h'(y, t), λ_F(t), λ_W(t), λ∞ and a figure.
- `test_estimators.py`: checks on synthetic signals (`pytest test_estimators.py`).



## Conventions

- XBeach writes 2D variables as (time, ny, nx). `read_field` returns (nt, nx, ny): in each
  frame the first axis is cross-shore and the second alongshore, as `ncread` in MATLAB.
- The cross-shore grid is not uniform (1-10 m here): integrals in x use trapezoidal cell widths.
- The alongshore boundaries are periodic (`cyclic = 1`), so Ly = ny·dy and the FFT along y is exact.
- With `morfacopt = 1` the output time is morphological time; λ∞ windows (30-70 h, control
  50-70 h) are in that time.
- The defaults (crest window 1550-1650 m, deposit band 1450-1750 m, modes n = 2-15 for Ly = 1202 m) are those of the Castelldefels P3 profile, change them for other domains.

## References


- Carrión-Bertrán, N., Calvete, D., & Ribas, F. (2026). Role of the initial topobathymetry in
  washover deposit formation during extreme events. *Journal of Geophysical Research: Earth
  Surface*, 131, e2025JF008755. https://doi.org/10.1029/2025JF008755


