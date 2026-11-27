"""Deconvolution and convolutional computational cameras.

Deconvolution: Wiener, Richardson–Lucy (optionally TV-regularised, Dey et al. 2006), TV-ADMM,
blind Richardson–Lucy (Fish et al. 1995).
Lensless DiffuserCam (Antipa et al. 2018): caustic PSF, cropped-convolution model, ADMM with
TV + non-negativity, and FISTA baseline.
Coded exposure ('flutter shutter', Raskar et al. 2006): box vs. random vs. optimised binary codes,
1-D motion blur, Tikhonov / Wiener inversion."""
from __future__ import annotations

import numpy as np
import scipy.fft as sfft

from .fields import crop_to, fft2, ifft2, pad_to
from .regularizers import fista, grad, div, tv_prox


def psf_otf(psf, shape):
    """OTF (FFT layout) of a centred PSF, zero-padded to `shape`."""
    p = pad_to(psf, shape) if psf.shape != tuple(shape) else psf
    return fft2(sfft.ifftshift(p))


def convolve(x, H):
    return np.real(ifft2(fft2(x) * H))


# ----------------------------------------------------------------------------- PSFs
def gaussian_psf(n, sigma):
    y, x = np.indices((n, n)) - n // 2
    p = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return p / p.sum()


def motion_psf(n, length, angle_deg):
    p = np.zeros((n, n))
    c = n // 2
    t = np.linspace(-length / 2, length / 2, max(int(4 * length), 2))
    a = np.deg2rad(angle_deg)
    ys = np.rint(c + t * np.sin(a)).astype(int)
    xs = np.rint(c + t * np.cos(a)).astype(int)
    np.add.at(p, (ys, xs), 1)
    return p / p.sum()


def disk_psf(n, radius):
    y, x = np.indices((n, n)) - n // 2
    p = (x ** 2 + y ** 2 <= radius ** 2).astype(float)
    return p / p.sum()


# ----------------------------------------------------------------------------- deconvolution
DECONV_METHODS = ("Wiener", "Richardson–Lucy", "Richardson–Lucy + TV", "TV (ADMM)", "blind Richardson–Lucy")


def wiener(b, H, nsr=1e-2):
    return np.real(ifft2(fft2(b) * np.conj(H) / (np.abs(H) ** 2 + nsr)))


def richardson_lucy(b, H, iters=50, tv=0.0, x0=None, monitor=None):
    """x ← x · Hᵀ(b / Hx) / (1 − λ div(∇x/|∇x|)) (TV term optional)."""
    b = np.maximum(b, 0)
    x = np.full_like(b, b.mean()) if x0 is None else x0.copy()
    Hc = np.conj(H)
    hist = []
    for k in range(iters):
        est = np.maximum(convolve(x, H), 1e-12)
        corr = convolve(b / est, Hc)
        if tv > 0:
            gx, gy = grad(x)
            nrm = np.sqrt(gx ** 2 + gy ** 2) + 1e-3
            corr = corr / np.maximum(1 - tv * div(gx / nrm, gy / nrm), 0.2)
        x = x * corr
        hist.append(float(np.linalg.norm(est - b) / np.linalg.norm(b)))
        if monitor is not None:
            monitor.step(k + 1, iters, (lambda x=x, h=list(hist): (x, h)) if monitor.want_preview() else None)
    return x, np.array(hist)


def tv_admm(b, H, lam=1e-3, rho=None, iters=60, nonneg=True, monitor=None):
    """min ½‖h∗x − b‖² + λ‖∇x‖₁ (+ x ≥ 0) by ADMM with z = ∇x, closed-form x-update in Fourier."""
    n = b.shape
    Dx = np.zeros(n); Dx[0, 0] = -1; Dx[0, 1] = 1
    Dy = np.zeros(n); Dy[0, 0] = -1; Dy[1, 0] = 1
    FDx, FDy = fft2(Dx), fft2(Dy)
    rho = 10 * lam if rho is None else rho
    den = np.abs(H) ** 2 + rho * (np.abs(FDx) ** 2 + np.abs(FDy) ** 2) + 1e-12
    Hb = np.conj(H) * fft2(b)
    x = b.copy()
    zx = np.zeros(n); zy = np.zeros(n); ux = zx.copy(); uy = zy.copy()
    hist = []
    for k in range(iters):
        rhs = Hb + rho * (np.conj(FDx) * fft2(zx - ux) + np.conj(FDy) * fft2(zy - uy))
        x = np.real(ifft2(rhs / den))
        if nonneg:
            x = np.maximum(x, 0)
        Fx = fft2(x)
        gx, gy = np.real(ifft2(FDx * Fx)), np.real(ifft2(FDy * Fx))
        vx, vy = gx + ux, gy + uy
        mag = np.sqrt(vx ** 2 + vy ** 2)
        s = np.maximum(1 - lam / rho / np.maximum(mag, 1e-12), 0)
        zx, zy = s * vx, s * vy
        ux += gx - zx; uy += gy - zy
        hist.append(float(np.linalg.norm(convolve(x, H) - b) / np.linalg.norm(b)))
        if monitor is not None:
            monitor.step(k + 1, iters)
    return x, np.array(hist)


def blind_rl(b, psf_size, iters=30, inner=3, sigma0=2.0, monitor=None):
    """Blind RL (Fish et al. 1995): alternate RL updates of the image and of a compactly supported PSF."""
    n = b.shape[0]
    psf = gaussian_psf(n, sigma0)
    sup = np.zeros((n, n), bool)
    c = n // 2
    h = psf_size // 2
    sup[c - h:c + h + 1, c - h:c + h + 1] = True
    x = np.full_like(b, b.mean())
    b = np.maximum(b, 0)
    hist = []
    for k in range(iters):
        H = psf_otf(psf, b.shape)
        for _ in range(inner):
            est = np.maximum(convolve(x, H), 1e-12)
            x = x * convolve(b / est, np.conj(H))
        X = fft2(x)
        for _ in range(inner):
            H = psf_otf(psf, b.shape)
            est = np.maximum(convolve(x, H), 1e-12)
            r = b / est
            upd = np.real(ifft2(fft2(r) * np.conj(X)))
            psf = psf * sfft.fftshift(upd) / max(x.sum(), 1e-12)
            psf = np.where(sup, np.maximum(psf, 0), 0)
            psf /= psf.sum() + 1e-20
        hist.append(float(np.linalg.norm(est - b) / np.linalg.norm(b)))
        if monitor is not None:
            monitor.step(k + 1, iters)
    return x, psf, np.array(hist)


# ----------------------------------------------------------------------------- DiffuserCam
def caustic_psf(n, rng=None, smooth=10.0, strength=40.0, z_rel=1.0):
    """Caustic pattern of a smooth random phase plate (diffuser) at a short distance: the PSF of a
    lensless camera. Intensity normalised to unit sum."""
    rng = np.random.default_rng(4) if rng is None else rng
    from scipy import ndimage as ndi
    N = 2 * n
    ph = ndi.gaussian_filter(rng.standard_normal((N, N)), smooth, mode="wrap")
    ph = strength * ph / np.abs(ph).max()
    y, x = (np.indices((N, N)) - N / 2) / N
    ap = (np.hypot(x, y) < 0.45).astype(float)
    u = ap * np.exp(1j * ph)
    FX, FY = np.meshgrid(sfft.fftfreq(N), sfft.fftfreq(N))
    Hz = np.exp(-1j * np.pi * z_rel * 60.0 * (FX ** 2 + FY ** 2) * N)
    I = np.abs(ifft2(fft2(u) * Hz)) ** 2
    I = crop_to(I, (n, n))
    return I / I.sum()


class CropConv:
    """b = C(h ∗ x): x lives on a 2n×2n grid (scene larger than the sensor), C crops the n×n sensor."""

    def __init__(self, psf):
        self.n = psf.shape[0]
        self.N = 2 * self.n
        self.H = psf_otf(psf, (self.N, self.N))

    def A(self, x):
        return crop_to(convolve(x, self.H), (self.n, self.n))

    def At(self, r):
        return convolve(pad_to(r, (self.N, self.N)), np.conj(self.H))


LENSLESS_METHODS = ("ADMM (TV + non-negativity)", "FISTA (non-negativity)", "Wiener (no crop model)")


def diffusercam_admm(b, psf, lam=1e-5, mu1=3e-3, mu2=3e-4, mu3=3e-3, iters=100, monitor=None):
    """Antipa et al. 2018 ADMM for min ½‖C(h∗v) − b‖² + τ‖∇v‖₁ + 1₊(v) with splits
    ξ = h∗v (handles the crop exactly), u = ∇v, w = v. Each update is closed form."""
    op = CropConv(psf)
    N, n = op.N, op.n
    hscale = np.abs(op.H).max()
    H = op.H / hscale                    # normalised so max|H| = 1; undone at the end
    Hc = np.conj(H)
    Dx = np.zeros((N, N)); Dx[0, 0] = -1; Dx[0, 1] = 1
    Dy = np.zeros((N, N)); Dy[0, 0] = -1; Dy[1, 0] = 1
    FDx, FDy = fft2(Dx), fft2(Dy)
    s = b.max()
    b = b / s
    CtC = pad_to(np.ones((n, n)), (N, N))
    Ctb = pad_to(b, (N, N))
    v = np.zeros((N, N))
    alpha1 = np.zeros((N, N)); alpha2x = np.zeros((N, N)); alpha2y = np.zeros((N, N)); alpha3 = np.zeros((N, N))
    tau = lam
    den = mu1 * np.abs(H) ** 2 + mu2 * (np.abs(FDx) ** 2 + np.abs(FDy) ** 2) + mu3
    hist = []
    Hv = convolve(v, H)
    for k in range(iters):
        # u (TV) update
        Fv = fft2(v)
        gx, gy = np.real(ifft2(FDx * Fv)), np.real(ifft2(FDy * Fv))
        vx, vy = gx + alpha2x / mu2, gy + alpha2y / mu2
        mag = np.sqrt(vx ** 2 + vy ** 2)
        sh = np.maximum(1 - tau / mu2 / np.maximum(mag, 1e-12), 0)
        ux, uy = sh * vx, sh * vy
        # ξ (crop) update
        xi = (alpha1 + mu1 * Hv + Ctb) / (CtC + mu1)
        # w (non-negativity)
        w = np.maximum(alpha3 / mu3 + v, 0)
        # v update
        r = mu1 * xi - alpha1
        rhs = Hc * fft2(r) + np.conj(FDx) * fft2(mu2 * ux - alpha2x) + np.conj(FDy) * fft2(mu2 * uy - alpha2y) \
            + fft2(mu3 * w - alpha3)
        v = np.real(ifft2(rhs / den))
        Hv = convolve(v, H)
        Fv = fft2(v)
        gx, gy = np.real(ifft2(FDx * Fv)), np.real(ifft2(FDy * Fv))
        alpha1 += mu1 * (Hv - xi)
        alpha2x += mu2 * (gx - ux); alpha2y += mu2 * (gy - uy)
        alpha3 += mu3 * (v - w)
        hist.append(float(np.linalg.norm(crop_to(Hv, (n, n)) - b) / np.linalg.norm(b)))
        if monitor is not None:
            monitor.step(k + 1, iters, (lambda v=v, h=list(hist): (np.maximum(v, 0) * s / hscale, h))
                         if monitor.want_preview() else None)
    return np.maximum(v, 0) * s / hscale, np.array(hist)


def diffusercam_fista(b, psf, iters=100, monitor=None):
    op = CropConv(psf)
    x, h = fista(op.A, op.At, b, (op.N, op.N), lambda z, s: z, iters=iters, nonneg=True, monitor=monitor)
    return x, h


# ----------------------------------------------------------------------------- coded exposure
def code(kind, L, rng=None, trials=4000):
    """Shutter code of L chips: box (open for all chips), random (50 %), or optimised — the random
    binary code with the largest minimum |DFT| (zero-padded) found among `trials` candidates."""
    rng = np.random.default_rng(0) if rng is None else rng
    if kind == "box":
        return np.ones(L)
    if kind == "random":
        c = (rng.random(L) < 0.5).astype(float)
        c[0] = c[-1] = 1
        return c
    best, score = None, -1
    for _ in range(trials):
        c = (rng.random(L) < 0.5).astype(float)
        c[0] = c[-1] = 1
        m = np.abs(np.fft.rfft(c, 4 * L)).min() / c.sum()
        if m > score:
            best, score = c, m
    return best


def motion_blur_code(img, c, length_px):
    """Linear horizontal motion over length_px pixels during an exposure gated by code c."""
    n = img.shape[1]
    L = len(c)
    k = np.zeros(n)
    pos = np.linspace(0, length_px, L, endpoint=False)
    for p, v in zip(pos, c):
        i = int(np.floor(p)); f = p - i
        k[i % n] += v * (1 - f); k[(i + 1) % n] += v * f
    k /= k.sum()
    K = np.fft.fft(k)
    return np.real(np.fft.ifft(np.fft.fft(img, axis=1) * K[None, :], axis=1)), K


def deblur_1d(b, K, reg=1e-3):
    return np.real(np.fft.ifft(np.fft.fft(b, axis=1) * np.conj(K)[None] / (np.abs(K)[None] ** 2 + reg), axis=1))
