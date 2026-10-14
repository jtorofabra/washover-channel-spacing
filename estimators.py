"""Two estimators of the alongshore channel spacing lambda(t) from the crest signal h'(y, t).

lambda_fourier: Ly over the centroid mode number of the alongshore Fourier spectrum.
lambda_wavelet: local scale centroid of a Morlet CWT, corrected so that it neither
                saturates at the domain scale nor locks onto the Fourier modes Ly/n.

Both assume h' is periodic in y (XBeach with cyclic = 1).
"""

import numpy as np


def lambda_fourier(hp, Ly, modes=range(2, 16)):
    """Spectral-centroid spacing (nt,) in m of hp (ny, nt).

    Mode n means n channels in Ly. The default band n = 2..15 (80-600 m for
    Ly = 1202 m) leaves out the mean, the single domain-wide undulation and grid noise.
    """
    ny = hp.shape[0]
    A = 2 * np.fft.fft(hp, axis=0) / ny
    n = np.asarray(list(modes))
    E = np.abs(A[n, :]) ** 2
    n_mean = (n[:, None] * E).sum(axis=0) / E.sum(axis=0)
    return Ly / n_mean


def _wavenumbers(ny, dy):
    """Angular wavenumber (rad/m) of each fft coefficient; Nyquist counted as positive."""
    k = np.arange(ny)
    npos = ny // 2 + 1
    kap = np.empty(ny)
    kap[:npos] = 2 * np.pi * k[:npos] / (ny * dy)
    kap[npos:] = -2 * np.pi * (ny - k[npos:]) / (ny * dy)
    return kap


def morlet_power(h, dy, scales, omega0=6.0):
    """Morlet power |W(y, s)|^2, shape (ny, ns), of one periodic series h (ny,).

    Analytic Morlet wavelet with unit energy at every scale (Torrence & Compo 1998),
    evaluated by circular convolution in Fourier space.
    """
    kap = _wavenumbers(h.size, dy)
    H = np.fft.fft(h - np.nanmean(h))
    s = np.asarray(scales, dtype=float)[None, :]
    psi = (np.pi ** -0.25 * np.sqrt(2 * np.pi * s / dy)
           * np.exp(-0.5 * (s * kap[:, None] - omega0) ** 2) * (kap > 0)[:, None])
    W = np.fft.ifft(H[:, None] * psi, axis=0)
    return np.abs(W) ** 2


def wavelet_power(h, dy, lam, lam_cut=600.0, omega0=6.0):
    """Rectified power P/s, shape (ny, nl), of h (ny,) on the wavelength grid lam (m).

    Fourier modes longer than lam_cut are removed first: slow undulations of the
    crest are not channels, but they would dominate the large scales.
    """
    ny = h.size
    H = np.fft.fft(h - h.mean())
    n = np.abs(np.fft.fftfreq(ny, 1.0 / ny))
    H[(n > 0) & (ny * dy / np.maximum(n, 1) > lam_cut)] = 0
    # Fourier wavelength of a Morlet scale: lam = 4 pi s / (omega0 + sqrt(2 + omega0^2))
    s = np.asarray(lam) * (omega0 + np.sqrt(2 + omega0 ** 2)) / (4 * np.pi)
    return morlet_power(np.fft.ifft(H).real, dy, s, omega0) / s[None, :]


def lambda_wavelet(hp, dy, lam_min=80.0, lam_max=800.0, lam_cut=600.0, n_scales=80, omega0=6.0):
    """Wavelet spacing (nt,) in m of hp (ny, nt).

    At each y the local spacing is the power-weighted centroid in scale; these are
    then averaged along y weighted by the total power at each y.
    """
    hp = np.asarray(hp, dtype=float)
    lam = np.geomspace(lam_min, lam_max, n_scales)
    out = np.full(hp.shape[1], np.nan)
    for k in range(hp.shape[1]):
        P = wavelet_power(hp[:, k], dy, lam, lam_cut, omega0)
        w = P.sum(axis=1)
        if w.sum() > 0:
            lam_local = (P * lam).sum(axis=1) / np.where(w > 0, w, 1)
            out[k] = np.sum(w * lam_local) / w.sum()
    return out
