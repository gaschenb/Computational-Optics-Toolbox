"""Confocal non-line-of-sight (NLOS) imaging: seeing around a corner from time-resolved light
bounced off a relay wall.

Forward model (confocal scanning, O'Toole et al. 2018):
    τ(x′, y′, t) = ∭ ρ(x, y, z) / r⁴ · δ(2r − ct) dx dy dz,  r = √((x′−x)² + (y′−y)² + z²)
Reconstructions: filtered back-projection (Velten et al. 2012, Laplacian-filtered), the light-cone
transform (LCT: resampling t → v = (ct/2)² turns the model into a 3-D convolution, inverted by
Wiener deconvolution) and f–k (Stolt) migration (Lindell et al. 2019)."""
from __future__ import annotations

import numpy as np
import scipy.fft as sfft
from scipy import ndimage as ndi

C = 3e8  # m/s


def hidden_scene(n=32, nz=64, wall=1.0, z_range=(0.4, 1.2), kind="letters", rng=None):
    """Albedo volume ρ(z, y, x) on the wall grid (x, y ∈ [−wall/2, wall/2]) and depths z_range (m)."""
    from .fields import resize, text_image
    rng = np.random.default_rng(1) if rng is None else rng
    vol = np.zeros((nz, n, n))
    zs = np.linspace(0, z_range[1] * 1.25, nz)
    if kind == "letters":
        items = [("N", 0.55), ("L", 0.8), ("S", 1.05)]
        from .scattering import _glyph
        q = n // 3
        h = int(0.55 * n)
        for k, (ch, z) in enumerate(items):
            g = resize(_glyph(ch), (h, q - 2))
            m = np.zeros((n, n))
            y0 = (n - h) // 2
            m[y0:y0 + h, k * q + 1:k * q + q - 1] = g
            iz = int(np.argmin(np.abs(zs - z * z_range[1] / 1.2)))
            vol[iz] = np.maximum(vol[iz], (m > 0.4).astype(float))
    else:  # random patches
        for _ in range(4):
            iz = int(np.argmin(np.abs(zs - rng.uniform(*z_range))))
            y0, x0 = rng.integers(2, n - 10, 2)
            vol[iz, y0:y0 + rng.integers(4, 9), x0:x0 + rng.integers(4, 9)] = 1
    return vol, zs


def simulate_transient(vol, zs, wall=1.0, bins=256, bin_ps=32.0, jitter_ps=50.0, photons=None, bg=0.0, rng=None):
    """Confocal transients τ(t, y′, x′) (time first) from the albedo volume; 1/r⁴ fall-off, SPAD jitter,
    optional Poisson noise (photons = expected total signal counts) and uniform background."""
    rng = np.random.default_rng(0) if rng is None else rng
    nz, n, _ = vol.shape
    xw = (np.arange(n) - (n - 1) / 2) / n * wall
    Yw, Xw = np.meshgrid(xw, xw, indexing="ij")
    tau = np.zeros((bins, n, n))
    dt = bin_ps * 1e-12
    for iz, iy, ix in np.argwhere(vol > 0):
        r = np.sqrt((Xw - xw[ix]) ** 2 + (Yw - xw[iy]) ** 2 + zs[iz] ** 2)
        tb = 2 * r / C / dt
        i0 = np.floor(tb).astype(int); f = tb - i0
        w = vol[iz, iy, ix] / np.maximum(r, 1e-3) ** 4
        ok = i0 + 1 < bins
        np.add.at(tau, (i0[ok], np.nonzero(ok)[0], np.nonzero(ok)[1]), (w * (1 - f))[ok])
        np.add.at(tau, (i0[ok] + 1, np.nonzero(ok)[0], np.nonzero(ok)[1]), (w * f)[ok])
    if jitter_ps:
        tau = ndi.gaussian_filter1d(tau, jitter_ps / bin_ps, axis=0)
    if photons:
        tau = tau * photons / max(tau.sum(), 1e-20) + bg / bins
        tau = rng.poisson(tau).astype(float)
    return tau


NLOS_METHODS = ("light-cone transform (LCT)", "f–k migration", "filtered back-projection")


def _zgrid(bins, bin_ps):
    return (np.arange(bins) + 0.5) * bin_ps * 1e-12 * C / 2


def lct(tau, wall=1.0, bin_ps=32.0, snr=1e-1):
    """O'Toole 2018: v^{3/2} τ(2√v/c) = h ∗ R{ρ}, h = δ(x² + y² − v); Wiener-inverted in 3-D."""
    T, n, _ = tau.shape
    z = _zgrid(T, bin_ps)
    zmax = z[-1]
    M = T
    v = (np.arange(M) + 0.5) / M * zmax ** 2
    # resample t → v (linear interpolation along time), with the v^{3/2} = z³ radiometric factor
    zq = np.sqrt(v)
    idx = np.clip(zq / (zmax / T) - 0.5, 0, T - 1.001)
    i0 = idx.astype(int); f = idx - i0
    R = (tau[i0] * (1 - f)[:, None, None] + tau[i0 + 1] * f[:, None, None]) * (zq ** 3)[:, None, None] * 1e6
    # kernel on the (2M, 2n, 2n) padded grid
    dx = wall / n
    off = (np.arange(2 * n) - n) * dx
    Yo, Xo = np.meshgrid(off, off, indexing="ij")
    vk = (Xo ** 2 + Yo ** 2) / (zmax ** 2 / M)
    K = np.zeros((2 * M, 2 * n, 2 * n))
    iv = np.floor(vk).astype(int); fv = vk - iv
    ok = iv + 1 < M
    yy, xx = np.nonzero(ok)
    K[iv[ok], yy, xx] += 1 - fv[ok]
    K[iv[ok] + 1, yy, xx] += fv[ok]
    K = np.roll(K, (-0, -n, -n), axis=(0, 1, 2))      # kernel origin at index 0 (x, y centred)
    Fk = sfft.fftn(K, workers=-1)
    Rp = np.zeros((2 * M, 2 * n, 2 * n)); Rp[:M, :n, :n] = R
    Fr = sfft.fftn(Rp, workers=-1)
    W = np.conj(Fk) / (np.abs(Fk) ** 2 + 1.0 / snr * np.mean(np.abs(Fk) ** 2) * 1e-2)
    rho_v = np.real(sfft.ifftn(Fr * W, workers=-1))[:M, :n, :n]
    # back from u = z² to z (uniform), ρ = 2√u · R
    u = v
    out = np.zeros((T, n, n))
    zz = z
    uq = zz ** 2
    j = np.clip(uq / (zmax ** 2 / M) - 0.5, 0, M - 1.001)
    j0 = j.astype(int); g = j - j0
    out = (rho_v[j0] * (1 - g)[:, None, None] + rho_v[j0 + 1] * g[:, None, None]) * (2 * zz)[:, None, None]
    return np.maximum(out, 0)


def fk_migration(tau, wall=1.0, bin_ps=32.0):
    """Lindell 2019: treat √(τ)·z as a wavefield recorded at z = 0 with velocity c/2 and apply Stolt
    interpolation  Φ(k_x, k_y, k_z) = |k_z|/√(k_x²+k_y²+k_z²) · Ψ(k_x, k_y, √(k_x²+k_y²+k_z²))."""
    T, n, _ = tau.shape
    z = _zgrid(T, bin_ps)
    zmax = z[-1]
    psi = np.sqrt(np.maximum(tau, 0)) * z[:, None, None]
    P = np.zeros((2 * T, 2 * n, 2 * n)); P[:T, :n, :n] = psi
    F = sfft.fftshift(sfft.fftn(P, workers=-1))
    kz = (np.arange(2 * T) - T) / (2 * T)          # cycles per bin (depth units: zmax/T)
    ky = (np.arange(2 * n) - n) / (2 * n)
    scale = (wall / n) / (zmax / T)                 # lateral pixel / depth bin
    KZ, KY, KX = np.meshgrid(kz, ky / scale, ky / scale, indexing="ij")
    kt = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2) * np.sign(KZ)
    # linear interpolation of F along its first axis at kt
    pos = kt * (2 * T) + T
    i0 = np.floor(pos).astype(int); f = pos - i0
    valid = (i0 >= 0) & (i0 + 1 < 2 * T)
    i0c = np.clip(i0, 0, 2 * T - 2)
    jj, kk = np.meshgrid(np.arange(2 * n), np.arange(2 * n), indexing="ij")
    F0 = F[i0c, jj[None], kk[None]]
    F1 = F[i0c + 1, jj[None], kk[None]]
    G = (F0 * (1 - f) + F1 * f) * valid * (KZ > 0) * np.abs(KZ) / np.maximum(np.abs(kt), 1e-9)
    out = sfft.ifftn(sfft.ifftshift(G), workers=-1)[:T, :n, :n]
    return np.abs(out) ** 2


def backprojection(tau, wall=1.0, bin_ps=32.0, laplacian=True):
    """Σ_wall τ(2r/c) r⁴ over every voxel (r⁴ compensates the fall-off), then a −∂²/∂z² filter."""
    T, n, _ = tau.shape
    z = _zgrid(T, bin_ps)
    xw = (np.arange(n) - (n - 1) / 2) / n * wall
    Yw, Xw = np.meshgrid(xw, xw, indexing="ij")
    dt = bin_ps * 1e-12
    out = np.zeros((T, n, n))
    tf = tau.reshape(T, -1)
    for iz, zz in enumerate(z):
        for iy in range(n):
            r = np.sqrt((Xw[None] - xw[:, None, None]) ** 2 + (Yw[None] - xw[iy]) ** 2 + zz ** 2)  # (x, wy, wx)
            tb = np.clip(np.rint(2 * r / C / dt).astype(int), 0, T - 1).reshape(n, -1)
            vals = tf[tb, np.arange(n * n)[None, :]] * (r.reshape(n, -1) ** 4)
            out[iz, iy] = vals.sum(1)
    if laplacian:
        out = -ndi.laplace(out)
    return np.maximum(out, 0)


def nlos_reconstruct(tau, method, wall=1.0, bin_ps=32.0, snr=0.1):
    if method.startswith("light-cone"):
        return lct(tau, wall, bin_ps, snr)
    if method.startswith("f–k"):
        return fk_migration(tau, wall, bin_ps)
    return backprojection(tau, wall, bin_ps)


def projections(vol):
    """Front-view maximum projection, depth (index of the max) and top view."""
    front = vol.max(0)
    depth = np.argmax(vol, 0).astype(float)
    depth[front < 0.15 * front.max()] = np.nan
    top = vol.max(1)
    return front, depth, top
