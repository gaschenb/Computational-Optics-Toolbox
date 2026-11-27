"""Snapshot hyperspectral imaging with a single-disperser coded aperture (SD-CASSI, Wagadarikar 2008).

y(i, j) = Σ_ℓ M(i, j − ℓ) X(i, j − ℓ, ℓ): the coded cube is sheared by one pixel per band and summed
on the detector. Reconstructions: back-projection (Aᵀy normalised), GAP-TV (generalised alternating
projection, Yuan 2016) and ADMM-TV."""
from __future__ import annotations

import numpy as np

from .regularizers import tv_prox


def wavelengths(L, lo=450.0, hi=650.0):
    return np.linspace(lo, hi, L)


def _gauss(wl, mu, s):
    return np.exp(-(wl - mu) ** 2 / (2 * s ** 2))


def spectral_scene(n=64, L=24, rng=None, layout=None, wl_range=None):
    """Hyperspectral cube (n, n, L): piecewise-constant regions with distinct reflectance spectra and a
    mild texture. Returns (cube, wavelengths, labels, spectra)."""
    from .fields import blobs, natural_texture
    rng = np.random.default_rng(12) if rng is None else rng
    wl = wavelengths(L) if wl_range is None else wavelengths(L, *wl_range)
    spectra = [0.25 + 0.0 * wl,
               0.15 + 0.8 * _gauss(wl, 470, 20),
               0.1 + 0.85 * _gauss(wl, 540, 25),
               0.1 + 0.9 * _gauss(wl, 610, 22),
               0.2 + 0.6 * _gauss(wl, 500, 15) + 0.5 * _gauss(wl, 630, 15)]
    lab = np.zeros((n, n), int)
    y, x = np.indices((n, n)) / n
    if layout is not None:          # regions from a grey-level object: five levels → five materials
        from .fields import normalize, resize, test_object
        o = normalize(np.clip(resize(test_object(layout, max(n, 128), rng), (n, n)), 0, None))
        lab = np.clip((o * 4.999).astype(int), 0, 4)
        tex = 0.85 + 0.3 * natural_texture(n, rng)
        cube = np.stack([np.choose(lab, [s[k] for s in spectra]) for k in range(L)], -1) * tex[..., None]
        return cube, wl, lab, np.array(spectra)
    lab[(x - 0.3) ** 2 + (y - 0.3) ** 2 < 0.04] = 1
    lab[(np.abs(x - 0.72) < 0.17) & (np.abs(y - 0.3) < 0.15)] = 2
    lab[(x - 0.3) ** 2 / 0.05 + (y - 0.75) ** 2 / 0.02 < 1] = 3
    lab[(np.abs(x - 0.72) < 0.15) & (np.abs(y - 0.75) < 0.17) & ((x + y) % 0.12 < 0.07)] = 4
    tex = 0.85 + 0.3 * natural_texture(n, rng)
    cube = np.stack([np.choose(lab, [s[k] for s in spectra]) for k in range(L)], -1) * tex[..., None]
    return cube, wl, lab, np.array(spectra)


def coded_mask(n, density=0.5, rng=None):
    rng = np.random.default_rng(3) if rng is None else rng
    return (rng.random((n, n)) < density).astype(float)


def forward(cube, mask, shift=1):
    n, m, L = cube.shape
    y = np.zeros((n, m + shift * (L - 1)))
    for l in range(L):
        y[:, l * shift:l * shift + m] += mask * cube[..., l]
    return y


def adjoint(y, mask, L, shift=1):
    n, m = mask.shape
    return np.stack([mask * y[:, l * shift:l * shift + m] for l in range(L)], -1)


def phi_diag(mask, L, shift=1):
    """Diagonal of A Aᵀ (each detector pixel collects Σ_ℓ M² of the sheared masks)."""
    return forward(np.ones(mask.shape + (L,)), mask ** 2, shift)


CASSI_METHODS = ("back-projection", "GAP-TV", "ADMM-TV")


def reconstruct(y, mask, L, method="GAP-TV", lam=0.02, iters=80, shift=1, monitor=None, accelerate=True):
    Phi = np.maximum(phi_diag(mask, L, shift), 1e-6)
    x = adjoint(y / Phi, mask, L, shift)
    if method == "back-projection":
        return x, np.array([])
    hist = []
    if method == "GAP-TV":
        yk = y.copy()
        for k in range(iters):
            r = y - forward(x, mask, shift)
            if accelerate:
                yk = yk + r
                x = x + adjoint((yk - forward(x, mask, shift)) / Phi, mask, L, shift)
            else:
                x = x + adjoint(r / Phi, mask, L, shift)
            x = np.moveaxis(tv_prox(np.moveaxis(x, -1, 0), lam, 6), 0, -1)
            x = np.maximum(x, 0)
            hist.append(float(np.linalg.norm(r) / np.linalg.norm(y)))
            if monitor is not None:
                monitor.step(k + 1, iters)
        return x, np.array(hist)
    # ADMM-TV: min ½‖y − Ax‖² + λ TV(z), x = z
    rho = 1.0
    z = x.copy(); u = np.zeros_like(x)
    for k in range(iters):
        v = z - u
        # x = (AᵀA + ρI)⁻¹(Aᵀy + ρv) via the diagonal A Aᵀ (matrix inversion lemma)
        rhs = adjoint(y, mask, L, shift) + rho * v
        x = (rhs - adjoint(forward(rhs, mask, shift) / (Phi + rho), mask, L, shift)) / rho
        z = np.moveaxis(tv_prox(np.moveaxis(x + u, -1, 0), lam / rho, 6), 0, -1)
        z = np.maximum(z, 0)
        u = u + x - z
        hist.append(float(np.linalg.norm(y - forward(z, mask, shift)) / np.linalg.norm(y)))
        if monitor is not None:
            monitor.step(k + 1, iters)
    return z, np.array(hist)


def cmf(wl):
    """Analytic CIE 1931 colour-matching fits (Wyman, Sloan & Shirley 2013)."""
    def g(x, mu, s1, s2):
        return np.where(x < mu, np.exp(-0.5 * ((x - mu) / s1) ** 2), np.exp(-0.5 * ((x - mu) / s2) ** 2))
    xb = 1.056 * g(wl, 599.8, 37.9, 31.0) + 0.362 * g(wl, 442.0, 16.0, 26.7) - 0.065 * g(wl, 501.1, 20.4, 26.2)
    yb = 0.821 * g(wl, 568.8, 46.9, 40.5) + 0.286 * g(wl, 530.9, 16.3, 31.1)
    zb = 1.217 * g(wl, 437.0, 11.8, 36.0) + 0.681 * g(wl, 459.0, 26.0, 13.8)
    return np.stack([xb, yb, zb], 1)


def to_rgb(cube, wl):
    XYZ = cube @ cmf(wl)
    M = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
    rgb = XYZ @ M.T
    rgb = np.clip(rgb / max(np.quantile(rgb, 0.995), 1e-9), 0, 1)
    return rgb ** (1 / 2.2)
