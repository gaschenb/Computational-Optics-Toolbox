"""Scalar diffraction: angular spectrum (band-limited), Fresnel (transfer function, impulse response,
single FFT), Fraunhofer, Rayleigh–Sommerfeld convolution, chirp-z zoom, sampling diagnostics.

All lengths share one unit (the toolbox uses micrometres). Fields are 2-D complex arrays sampled
on a square grid of pitch dx; stacks (..., ny, nx) propagate in one call."""
from __future__ import annotations

import numpy as np
import scipy.fft as sfft

from .fields import crop_to, fft2, freq_grid, grid, ifft2, pad_to


def asm_kernel(shape, wl, dx, z, n_medium=1.0, bandlimit=True, evanescent=False):
    """Transfer function H(fx, fy) = exp(i 2π z √((n/λ)² − fx² − fy²)) with Matsushima's band limit
    |f| < 1 / (λ √((2 Δf z)² + 1)), which removes the aliasing of the kernel's chirp."""
    ny, nx = shape
    FX, FY = freq_grid(shape, dx)
    k2 = (n_medium / wl) ** 2 - FX ** 2 - FY ** 2
    prop = k2 >= 0
    kz = np.sqrt(np.abs(k2))
    z = np.asarray(z, float)
    zz = z.reshape(z.shape + (1, 1)) if z.ndim else z
    H = np.where(prop, np.exp(2j * np.pi * zz * kz), np.exp(-2 * np.pi * np.abs(zz) * kz) if evanescent else 0)
    if bandlimit:
        dfx, dfy = 1.0 / (nx * dx), 1.0 / (ny * dx)
        lx = 1.0 / (wl / n_medium * np.sqrt((2 * dfx * np.abs(zz)) ** 2 + 1))
        ly = 1.0 / (wl / n_medium * np.sqrt((2 * dfy * np.abs(zz)) ** 2 + 1))
        H = H * ((np.abs(FX) < lx) & (np.abs(FY) < ly))
    return H


def angular_spectrum(u, wl, dx, z, n_medium=1.0, bandlimit=True, pad=False):
    """Exact (non-paraxial) propagation by distance z; z may be a 1-D array → stack of planes."""
    u = np.asarray(u, complex)
    shape0 = u.shape[-2:]
    if pad:
        u = pad_to(u, (2 * shape0[0], 2 * shape0[1]))
    H = asm_kernel(u.shape[-2:], wl, dx, z, n_medium, bandlimit)
    out = ifft2(fft2(u) * H)
    return crop_to(out, shape0) if pad else out


def fresnel_tf(u, wl, dx, z, pad=False):
    """Fresnel transfer function H = e^{ikz} exp(−iπλz(fx²+fy²)); accurate when dx ≥ λz/L."""
    u = np.asarray(u, complex)
    shape0 = u.shape[-2:]
    if pad:
        u = pad_to(u, (2 * shape0[0], 2 * shape0[1]))
    FX, FY = freq_grid(u.shape[-2:], dx)
    H = np.exp(2j * np.pi * z / wl) * np.exp(-1j * np.pi * wl * z * (FX ** 2 + FY ** 2))
    out = ifft2(fft2(u) * H)
    return crop_to(out, shape0) if pad else out


def fresnel_ir(u, wl, dx, z, pad=True):
    """Fresnel impulse response h = e^{ikz}/(iλz) exp(iπ(x²+y²)/(λz)), sampled and FFT-convolved;
    accurate when dx ≤ λz/L (long distances)."""
    u = np.asarray(u, complex)
    shape0 = u.shape[-2:]
    if pad:
        u = pad_to(u, (2 * shape0[0], 2 * shape0[1]))
    X, Y = grid(u.shape[-2:], dx)
    h = np.exp(2j * np.pi * z / wl) / (1j * wl * z) * np.exp(1j * np.pi * (X ** 2 + Y ** 2) / (wl * z)) * dx * dx
    H = fft2(sfft.ifftshift(h))
    out = ifft2(fft2(u) * H)
    return crop_to(out, shape0) if pad else out


def fresnel_single_fft(u, wl, dx, z):
    """One-FFT Fresnel transform. Returns (field, output pitch λz/(N dx))."""
    u = np.asarray(u, complex)
    ny, nx = u.shape[-2:]
    X, Y = grid((ny, nx), dx)
    dx2 = wl * z / (nx * dx)
    X2, Y2 = grid((ny, nx), dx2)
    U = sfft.fftshift(fft2(sfft.ifftshift(u * np.exp(1j * np.pi * (X ** 2 + Y ** 2) / (wl * z)))))
    pref = np.exp(2j * np.pi * z / wl) / (1j * wl * z) * np.exp(1j * np.pi * (X2 ** 2 + Y2 ** 2) / (wl * z))
    return pref * U * dx * dx, dx2


def fraunhofer(u, wl, dx, z):
    """Far field. Returns (field, output pitch λz/(N dx))."""
    u = np.asarray(u, complex)
    ny, nx = u.shape[-2:]
    dx2 = wl * z / (nx * dx)
    X2, Y2 = grid((ny, nx), dx2)
    U = sfft.fftshift(fft2(sfft.ifftshift(u)))
    pref = np.exp(2j * np.pi * z / wl) / (1j * wl * z) * np.exp(1j * np.pi * (X2 ** 2 + Y2 ** 2) / (wl * z))
    return pref * U * dx * dx, dx2


def rayleigh_sommerfeld(u, wl, dx, z):
    """First Rayleigh–Sommerfeld solution by FFT convolution with the exact impulse response
    h = z/(iλ) e^{ikr}/r² (1 + i/(kr))·(−i... ) on a 2× padded grid (the reference method)."""
    u = np.asarray(u, complex)
    ny, nx = u.shape[-2:]
    N = (2 * ny, 2 * nx)
    X, Y = grid(N, dx)
    r = np.sqrt(X ** 2 + Y ** 2 + z ** 2)
    k = 2 * np.pi / wl
    h = z / (2 * np.pi) * np.exp(1j * k * r) / r ** 3 * (1 - 1j * k * r) * dx * dx
    out = ifft2(fft2(pad_to(u, N)) * fft2(sfft.ifftshift(h)))
    return crop_to(out, (ny, nx))


def zoom_dft_1d(x, f0, f1, m, axis=-1, dx=1.0):
    """DFT of x evaluated at m frequencies in [f0, f1) (cycles per unit) via matrix product —
    an exact chirp-z/zoom transform, cheap for the image sizes used here."""
    x = np.moveaxis(np.asarray(x, complex), axis, -1)
    n = x.shape[-1]
    t = (np.arange(n) - n // 2) * dx
    f = f0 + (f1 - f0) * np.arange(m) / m
    W = np.exp(-2j * np.pi * np.outer(t, f))
    return np.moveaxis(x @ W, -1, axis), f


def fraunhofer_zoom(u, wl, dx, z, window, m=None):
    """Far field on an arbitrary output window [-w/2, w/2]² with m×m samples (zoom DFT)."""
    ny, nx = u.shape[-2:]
    m = m or nx
    fmax = window / (2 * wl * z)
    U, _ = zoom_dft_1d(u, -fmax, fmax, m, axis=-1, dx=dx)
    U, _ = zoom_dft_1d(U, -fmax, fmax, m, axis=-2, dx=dx)
    dx2 = window / m
    X2, Y2 = grid((m, m), dx2)
    pref = np.exp(2j * np.pi * z / wl) / (1j * wl * z) * np.exp(1j * np.pi * (X2 ** 2 + Y2 ** 2) / (wl * z))
    return pref * U * dx * dx, dx2


METHODS = {
    "angular spectrum": angular_spectrum,
    "angular spectrum (no band limit)": lambda u, wl, dx, z: angular_spectrum(u, wl, dx, z, bandlimit=False),
    "Fresnel transfer function": fresnel_tf,
    "Fresnel impulse response": fresnel_ir,
    "Rayleigh–Sommerfeld": rayleigh_sommerfeld,
}


def propagate(u, wl, dx, z, method="angular spectrum", **kw):
    if method in ("Fresnel single FFT", "Fraunhofer"):
        f = fresnel_single_fft if method == "Fresnel single FFT" else fraunhofer
        return f(u, wl, dx, z)
    return METHODS[method](u, wl, dx, z, **kw)


def sampling_report(n, dx, wl, z, na=None):
    """Critical distance z_c = N dx² / λ: below it use the transfer function / ASM, above it the
    impulse response or single-FFT form. Also the Nyquist-limited NA = λ / (2 dx)."""
    zc = n * dx * dx / wl
    rep = {"critical distance z_c": zc, "Fresnel number N_F (L²/λz)": (n * dx) ** 2 / (wl * max(abs(z), 1e-12)),
           "max NA sampled (λ/2dx)": min(1.0, wl / (2 * dx)),
           "recommended": "transfer function / ASM" if abs(z) < zc else "impulse response / single FFT"}
    if na is not None:
        rep["coherent cutoff NA/λ"] = na / wl
    return rep
