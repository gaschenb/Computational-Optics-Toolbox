"""Wavefront sensing and fringe analysis.

Shack–Hartmann: physical-optics spot simulation (FFT per lenslet), centroiding (thresholded,
iteratively windowed, reference cross-correlation), zonal (Southwell least squares) and modal
(Zernike) reconstruction.
Fringe projection profilometry: Fourier-transform profilometry (Takeda), N-step phase shifting with
spatial unwrapping, and multi-frequency temporal phase unwrapping."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
from scipy.sparse.linalg import lsqr

from .fields import TWO_PI, wrap
from .waveoptics import zernike


# ============================================================================ Shack–Hartmann
def sh_pupil_grid(n_lens, s):
    """Normalised pupil coordinates on the full sensor (n_lens·s pixels), unit circle = aperture."""
    N = n_lens * s
    c = (np.arange(N) + 0.5 - N / 2) / (N / 2)
    X, Y = np.meshgrid(c, c)
    return X, Y, np.hypot(X, Y), np.arctan2(Y, X)


def wavefront_waves(coeffs, rho, theta):
    W = np.zeros_like(rho)
    for j, c in coeffs.items():
        if c:
            W = W + c * zernike(int(j), np.minimum(rho, 1.0), theta)
    return W


def sh_simulate(coeffs, n_lens=16, s=16, wl=0.633, pitch=150.0, focal=5000.0, photons=2000, read_noise=2.0,
                rng=None, fill=0.5):
    """Spot image of a lenslet array (n_lens² lenslets, s×s pixels each). The focal plane of every
    lenslet is computed with an FFT of its sub-aperture field. Returns (image, reference image,
    valid-lenslet mask, focal-plane pixel pitch, true wavefront in waves (not masked), aperture)."""
    rng = np.random.default_rng(0) if rng is None else rng
    X, Y, R, T = sh_pupil_grid(n_lens, s)
    ap = R <= 1.0
    W = wavefront_waves(coeffs, R, T)
    Q = wl * focal * s / pitch ** 2
    M = max(s, int(round(Q * s)))
    pf = wl * focal / (M * pitch / s)              # focal-plane sampling (µm)
    img = np.zeros_like(W); ref = np.zeros_like(W)
    valid = np.zeros((n_lens, n_lens), bool)
    for i in range(n_lens):
        for j in range(n_lens):
            sl = (slice(i * s, (i + 1) * s), slice(j * s, (j + 1) * s))
            a = ap[sl]
            if a.mean() < fill:
                continue
            valid[i, j] = True
            for w, out in ((W[sl], img), (np.zeros_like(W[sl]), ref)):
                u = np.zeros((M, M), complex)
                u[:s, :s] = a * np.exp(2j * np.pi * w * a)
                F = np.fft.fftshift(np.abs(np.fft.fft2(u)) ** 2)
                c0 = M // 2 - s // 2
                out[sl] = F[c0:c0 + s, c0:c0 + s]
    for out in (img, ref):
        out /= out.max() + 1e-20
    if photons:
        k = photons / max(img[img > 0.05].mean() if np.any(img > 0.05) else 1, 1e-12)
        img[:] = rng.poisson(img * k) / k + read_noise / k * rng.standard_normal(img.shape)
    return img, ref, valid, pf, W, ap


CENTROID_METHODS = ("thresholded centre of gravity", "windowed CoG (iterative)", "cross-correlation")


def centroids(img, n_lens, s, valid, method="windowed CoG (iterative)", thresh=0.2, ref=None):
    """Spot positions (pixels, relative to the lenslet centre) for every valid lenslet."""
    out = np.full((n_lens, n_lens, 2), np.nan)
    yy, xx = np.indices((s, s)) - (s - 1) / 2
    for i in range(n_lens):
        for j in range(n_lens):
            if not valid[i, j]:
                continue
            sub = img[i * s:(i + 1) * s, j * s:(j + 1) * s].astype(float)
            if method == "cross-correlation" and ref is not None:
                r = ref[i * s:(i + 1) * s, j * s:(j + 1) * s]
                xc = np.real(np.fft.ifft2(np.fft.fft2(sub) * np.conj(np.fft.fft2(r))))
                xc = np.fft.fftshift(xc)
                k = np.unravel_index(np.argmax(xc), xc.shape)
                y0, x0 = k
                win = xc[max(y0 - 1, 0):y0 + 2, max(x0 - 1, 0):x0 + 2]
                wy, wx = np.indices(win.shape)
                wsum = win.sum() + 1e-20
                dy = (wy * win).sum() / wsum + max(y0 - 1, 0) - s // 2
                dx = (wx * win).sum() / wsum + max(x0 - 1, 0) - s // 2
                out[i, j] = (dy, dx)
                continue
            w = np.clip(sub - thresh * sub.max(), 0, None)
            cy = (w * yy).sum() / (w.sum() + 1e-20); cx = (w * xx).sum() / (w.sum() + 1e-20)
            if method.startswith("windowed"):
                r = s / 2.5
                for _ in range(4):
                    win = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (r / 2) ** 2))
                    ww = np.clip(sub, 0, None) * win
                    cy = (ww * yy).sum() / (ww.sum() + 1e-20); cx = (ww * xx).sum() / (ww.sum() + 1e-20)
                    r = max(r * 0.75, 2.0)
            out[i, j] = (cy, cx)
    return out


def slopes(cent, cent_ref, pf, focal, pitch, s, wl):
    """Wavefront gradient in waves per lenslet pitch from spot displacements."""
    d = cent - cent_ref
    # displacement (µm) = f · θ, θ = dW/dx (µm/µm); per pitch in waves: θ·pitch/λ
    return d * pf / focal * pitch / wl            # (..., 2) = (sy, sx)


def zonal_southwell(sl, valid):
    """Southwell least squares: W_{i,j+1} − W_{i,j} = (s_x,ij + s_x,ij+1)/2 (and y), in waves."""
    n = valid.shape[0]
    idx = -np.ones((n, n), int)
    idx[valid] = np.arange(valid.sum())
    rows, cols, vals, b = [], [], [], []
    r = 0
    for i in range(n):
        for j in range(n):
            if not valid[i, j]:
                continue
            if j + 1 < n and valid[i, j + 1]:
                rows += [r, r]; cols += [idx[i, j + 1], idx[i, j]]; vals += [1, -1]
                b.append(0.5 * (sl[i, j, 1] + sl[i, j + 1, 1])); r += 1
            if i + 1 < n and valid[i + 1, j]:
                rows += [r, r]; cols += [idx[i + 1, j], idx[i, j]]; vals += [1, -1]
                b.append(0.5 * (sl[i, j, 0] + sl[i + 1, j, 0])); r += 1
    rows.append(r); cols.append(0); vals.append(1e-3); b.append(0.0)      # fix the piston
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(r + 1, valid.sum()))
    w = lsqr(A, np.array(b), atol=1e-12, btol=1e-12, iter_lim=20000)[0]
    W = np.full((n, n), np.nan)
    W[valid] = w - w.mean()
    return W


def zernike_slope_matrix(n_lens, s, valid, J=15):
    """Average Zernike gradients over each lenslet (waves per pitch), columns j = 2..J."""
    X, Y, R, T = sh_pupil_grid(n_lens, s)
    rows = []
    for j in range(2, J + 1):
        Z = zernike(j, np.minimum(R, 1.0), T)
        gy, gx = np.gradient(Z)
        gy = gy * s; gx = gx * s                   # per pitch
        Gy = gy.reshape(n_lens, s, n_lens, s).mean(axis=(1, 3))
        Gx = gx.reshape(n_lens, s, n_lens, s).mean(axis=(1, 3))
        rows.append(np.concatenate([Gy[valid], Gx[valid]]))
    return np.array(rows).T


def modal_zernike(sl, valid, n_lens, s, J=15):
    G = zernike_slope_matrix(n_lens, s, valid, J)
    b = np.concatenate([sl[..., 0][valid], sl[..., 1][valid]])
    c, *_ = np.linalg.lstsq(G, b, rcond=None)
    return {j: float(v) for j, v in zip(range(2, J + 1), c)}


# ============================================================================ fringe projection
def fpp_phase_to_height(dphi, period, angle_deg):
    """Crossed-optical-axes approximation: h = Δφ p / (2π tan θ)."""
    return dphi * period / (TWO_PI * np.tan(np.deg2rad(angle_deg)))


def fpp_fringes(h, period, angle_deg, shifts, a=0.5, b=0.4, albedo=None, noise=0.0, rng=None, gamma=1.0):
    """Camera images of sinusoidal fringes (period in pixels) on a surface of height h (pixels units)."""
    rng = np.random.default_rng(0) if rng is None else rng
    ny, nx = h.shape
    x = np.arange(nx)[None, :]
    phi = TWO_PI * x / period + TWO_PI * h * np.tan(np.deg2rad(angle_deg)) / period
    alb = 1.0 if albedo is None else albedo
    ims = np.array([alb * (a + b * np.cos(phi + d)) for d in shifts])
    ims = np.clip(ims, 0, None) ** gamma
    if noise:
        ims = ims + noise * rng.standard_normal(ims.shape)
    return ims


FPP_METHODS = ("Fourier-transform profilometry", "N-step phase shifting + spatial unwrap",
               "multi-frequency temporal unwrap")


def ftp_phase(img, period):
    """Takeda: isolate the +1 order around 1/period with a Hann window and demodulate it; returns the
    wrapped phase with the carrier already removed."""
    from .holography import offaxis_filter
    ny, nx = img.shape
    order = (ny // 2, nx // 2 + nx / period)
    u = offaxis_filter(img, order=order, radius=0.8 * nx / period, window="hann")
    return np.angle(u)


def psi_phase(ims, shifts):
    s = np.asarray(shifts)
    num = -np.tensordot(np.sin(s), ims, 1)
    den = np.tensordot(np.cos(s), ims, 1)
    return np.arctan2(num, den)


def temporal_unwrap(phases, periods):
    """Hierarchical temporal unwrapping: periods sorted coarse → fine; the coarsest must be ≥ the
    field width (no ambiguity). Φ_k = φ_k + 2π round((Φ_{k−1}·p_{k−1}/p_k − φ_k)/2π)."""
    Phi = phases[0]
    for k in range(1, len(periods)):
        scale = periods[k - 1] / periods[k]
        Phi = phases[k] + TWO_PI * np.rint((Phi * scale - phases[k]) / TWO_PI)
    return Phi
