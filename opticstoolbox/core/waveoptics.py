"""Pupils, Zernike polynomials (Noll ordering), aberrated coherent / incoherent PSF, OTF, MTF."""
from __future__ import annotations

from math import factorial

import numpy as np
import scipy.fft as sfft

from .fields import fft2, freq_grid, ifft2


def noll_to_nm(j):
    """Noll index j ≥ 1 → (n, m)."""
    n = 0
    while (n + 1) * (n + 2) // 2 < j:
        n += 1
    k = j - n * (n + 1) // 2 - 1          # 0..n
    ms = [m for m in range(-n, n + 1, 2)]
    mabs = sorted({abs(m) for m in ms})
    # Noll: within an order, |m| increases; even j → cos (m>0), odd j → sin (m<0)
    order = []
    for a in mabs:
        if a == 0:
            order.append(0)
        else:
            order += [a, -a] if (len(order) + n * (n + 1) // 2 + 1) % 2 == 0 else [-a, a]
    m = order[k]
    return n, m


def zernike_radial(n, m, rho):
    m = abs(m)
    R = np.zeros_like(rho)
    for s in range((n - m) // 2 + 1):
        c = (-1) ** s * factorial(n - s) / (factorial(s) * factorial((n + m) // 2 - s) * factorial((n - m) // 2 - s))
        R = R + c * rho ** (n - 2 * s)
    return R


def zernike(j, rho, theta):
    """Orthonormal Zernike Z_j over the unit disk (RMS 1), Noll convention."""
    n, m = noll_to_nm(j)
    norm = np.sqrt(n + 1) if m == 0 else np.sqrt(2 * (n + 1))
    R = zernike_radial(n, m, rho)
    if m > 0:
        return norm * R * np.cos(m * theta)
    if m < 0:
        return norm * R * np.sin(-m * theta)
    return norm * R


ZERNIKE_NAMES = {1: "piston", 2: "tilt x", 3: "tilt y", 4: "defocus", 5: "astig 45°", 6: "astig 0°",
                 7: "coma y", 8: "coma x", 9: "trefoil y", 10: "trefoil x", 11: "spherical"}


def zernike_phase(coeffs, rho, theta):
    """Σ c_j Z_j (coeffs in waves RMS → returns radians)."""
    W = np.zeros_like(rho)
    for j, c in coeffs.items():
        if c:
            W = W + c * zernike(int(j), rho, theta)
    return 2 * np.pi * W


def pupil(shape, wl, dx, na, coeffs=None, defocus_um=0.0, n_medium=1.0):
    """Coherent transfer function P(f) = circ(|f| λ/NA) exp(iW) on the FFT grid of an image with pitch dx
    (object-space units). defocus adds the exact angular-spectrum phase 2π z √((n/λ)² − f²)."""
    FX, FY = freq_grid(shape, dx)
    fc = na / wl
    rho = np.hypot(FX, FY) / fc
    theta = np.arctan2(FY, FX)
    P = (rho <= 1.0).astype(complex)
    if coeffs:
        P = P * np.exp(1j * zernike_phase(coeffs, np.minimum(rho, 1), theta))
    if defocus_um:
        kz = np.sqrt(np.clip((n_medium / wl) ** 2 - FX ** 2 - FY ** 2, 0, None))
        P = P * np.exp(2j * np.pi * defocus_um * kz)
    return P


def coherent_psf(P):
    return sfft.fftshift(ifft2(P))


def incoherent_psf(P):
    h = np.abs(coherent_psf(P)) ** 2
    return h / h.sum()


def otf(psf):
    """OTF of a centred, normalised PSF (FFT layout)."""
    H = fft2(sfft.ifftshift(psf))
    return H / H[0, 0]


def strehl(P):
    P0 = np.abs(P)
    return float(np.abs(coherent_psf(P)).max() ** 2 / max(np.abs(coherent_psf(P0)).max() ** 2, 1e-20))


def image_incoherent(obj, psf):
    """Periodic convolution of an intensity object with a centred PSF."""
    return np.real(ifft2(fft2(obj) * fft2(sfft.ifftshift(psf))))


def image_coherent(field, P):
    return ifft2(fft2(field) * P)


def radial_profile(img, center=None, nbins=None):
    ny, nx = img.shape
    cy, cx = (ny // 2, nx // 2) if center is None else center
    Y, X = np.indices(img.shape)
    r = np.hypot(X - cx, Y - cy).astype(int)
    nb = nbins or min(cx, cy)
    s = np.bincount(r.ravel(), img.ravel())[:nb]
    c = np.bincount(r.ravel())[:nb]
    return s / np.maximum(c, 1)
