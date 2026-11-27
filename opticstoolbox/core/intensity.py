"""Non-interferometric quantitative phase: transport of intensity (TIE) and differential phase
contrast (DPC) with partially coherent weak-object transfer functions."""
from __future__ import annotations

import numpy as np
import scipy.fft as sfft

from .fields import TWO_PI, fft2, freq_grid, ifft2
from .waveoptics import pupil


# ============================================================================ TIE
def axial_derivative(stack, zs, order=2):
    """∂I/∂z at z = 0 from planes at zs by least-squares polynomial fit (order 1 = two-plane / central
    difference for symmetric planes; higher orders suppress the nonlinearity error, Waller 2010)."""
    I = np.asarray(stack, float)
    zs = np.asarray(zs, float)
    order = min(order, len(zs) - 1)
    V = np.vander(zs, order + 1, increasing=True)            # [1, z, z², …]
    coef = np.linalg.lstsq(V, I.reshape(len(zs), -1), rcond=None)[0]
    return coef[1].reshape(I.shape[1:]), coef[0].reshape(I.shape[1:])


def _inv_laplacian_fft(f, dx, reg):
    FX, FY = freq_grid(f.shape, dx)
    q = 4 * np.pi ** 2 * (FX ** 2 + FY ** 2)
    G = -q / (q ** 2 + reg ** 2)          # Tikhonov-regularised −1/q
    G[0, 0] = 0
    return np.real(ifft2(fft2(f) * G))


def _inv_laplacian_dct(f, dx, reg):
    ny, nx = f.shape
    F = sfft.dctn(f, type=2, norm="ortho", workers=-1)
    ky = np.pi * np.arange(ny) / (ny * dx); kx = np.pi * np.arange(nx) / (nx * dx)
    q = ky[:, None] ** 2 + kx[None, :] ** 2
    G = -q / (q ** 2 + reg ** 2)
    G[0, 0] = 0
    return sfft.idctn(F * G, type=2, norm="ortho", workers=-1)


def _grad(f, dx):
    gy, gx = np.gradient(f, dx)
    return gx, gy


def _div(gx, gy, dx):
    return np.gradient(gx, dx, axis=1) + np.gradient(gy, dx, axis=0)


TIE_METHODS = ("FFT Poisson (uniform I)", "DCT Poisson (uniform I)", "Teague (non-uniform I)")


def tie(dIdz, I0, wl, dx, method="DCT Poisson (uniform I)", reg=0.0, n_medium=1.0):
    """Solve ∇·(I ∇φ) = −k ∂I/∂z (k = 2πn/λ).
    uniform I:  ∇²φ = −k ∂zI / Ī
    Teague:     ∇²ψ = −k ∂zI,  φ = ∇⁻² ∇·(∇ψ / I)
    reg is a Tikhonov constant (rad/µm)² that damps the low-frequency noise blow-up."""
    k = TWO_PI * n_medium / wl
    inv = _inv_laplacian_dct if "DCT" in method or "Teague" in method else _inv_laplacian_fft
    rhs = -k * dIdz
    if method.startswith("Teague"):
        psi = inv(rhs, dx, reg)
        gx, gy = _grad(psi, dx)
        Is = np.maximum(I0, 0.05 * I0.max())
        return inv(_div(gx / Is, gy / Is, dx), dx, reg)
    return inv(rhs / max(float(np.mean(I0)), 1e-12), dx, reg)


# ============================================================================ DPC
def source_points(shape, dx, wl, na_ill, kind="top", step=3, na_inner=0.0):
    """Integer frequency-grid points of a half-circle (top/bottom/left/right), full disc or annulus."""
    ny, nx = shape
    FX, FY = freq_grid(shape, dx, shift=True)
    r = np.hypot(FX, FY) * wl
    m = (r <= na_ill) & (r >= na_inner)
    if kind == "top":
        m &= FY > 0
    elif kind == "bottom":
        m &= FY < 0
    elif kind == "left":
        m &= FX < 0
    elif kind == "right":
        m &= FX > 0
    sub = np.zeros(shape, bool)
    sub[::step, ::step] = True
    # keep the grid symmetric about the centre
    sub = np.roll(sub, ((ny // 2) % step, (nx // 2) % step), axis=(0, 1))
    pts = np.argwhere(m & sub) - np.array([ny // 2, nx // 2])
    return pts            # (k, 2) integer offsets (dy, dx) in FFT bins


def abbe_image(t, P, pts):
    """Partially coherent image: Σ_s |F⁻¹{P(u) T(u − u_s)}|² over source points (Abbe)."""
    T = fft2(t)
    acc = np.zeros(t.shape)
    for dy, dx in pts:
        acc += np.abs(ifft2(P * np.roll(T, (dy, dx), axis=(0, 1)))) ** 2
    return acc / max(len(pts), 1)


def _corr_direct(A, B):
    """C(v) = Σ_u A(u) B(u + v) on the (periodic) FFT frequency grid, by the correlation theorem."""
    n = A.size
    a = sfft.ifft2(np.conj(A), workers=-1)
    b = sfft.ifft2(B, workers=-1)
    return sfft.fft2(np.conj(a) * b, workers=-1) * n


def wotf(P, S):
    """Weak-object transfer functions of a partially coherent system (Tian & Waller 2015) for
    source S(u) and pupil P(u) on the FFT grid:  Î = B δ + H_μ μ̂ + H_φ φ̂ with t ≈ 1 − μ + iφ,
    H_φ = i[Σ S P*(u') P(u'+v) − Σ S P(u') P*(u'−v)],  H_μ = −[… + …],  B = Σ S |P|²."""
    A = S * np.conj(P)
    C1 = _corr_direct(A, P)                       # Σ S P*(u') P(u'+v)
    # Σ_u S P(u) P*(u − v) = conj(Σ_u S P*(u) P(u − v)) and Σ_u f(u) g(u − v) = (f ⋆ g)(−v)
    C2 = np.conj(np.roll(np.flip(_corr_direct(S * np.conj(P), P)), 1, axis=(0, 1)))
    B = float(np.sum(S * np.abs(P) ** 2))
    Hphi = 1j * (C1 - C2) / B
    Hmu = -(C1 + C2) / B
    return Hmu, Hphi


def source_map(shape, pts):
    S = np.zeros(shape)
    for dy, dx in pts:
        S[dy % shape[0], dx % shape[1]] += 1
    return S


def dpc_images(t, wl, dx, na, na_ill=None, axes=("top", "bottom", "left", "right"), step=3,
               coeffs=None, photons=None, rng=None):
    """Simulate the four half-circle DPC images of transmission t."""
    na_ill = na if na_ill is None else na_ill
    P = pupil(t.shape, wl, dx, na, coeffs)
    out, srcs = [], []
    for a in axes:
        pts = source_points(t.shape, dx, wl, na_ill, a, step)
        I = abbe_image(t, P, pts)
        if photons and rng is not None:
            k = photons / I.mean()
            I = rng.poisson(I * k) / k
        out.append(I)
        srcs.append(pts)
    return np.array(out), srcs, P


DPC_METHODS = ("Tikhonov (phase only)", "Tikhonov (phase + absorption)", "TV (ADMM)")


def dpc_transfer(shape, wl, dx, na, srcs):
    """Transfer functions of the normalised DPC signals (I_a − I_b)/(I_a + I_b) for the pairs
    (0,1), (2,3), …; returns lists (H_μ, H_φ) per pair."""
    P = pupil(shape, wl, dx, na)
    Hs = []
    for i in range(0, len(srcs), 2):
        Sa, Sb = source_map(shape, srcs[i]), source_map(shape, srcs[i + 1])
        ma, pa = wotf(P, Sa)
        mb, pb = wotf(P, Sb)
        Hs.append(((ma - mb) / 2, (pa - pb) / 2))
    return Hs


def dpc_signals(images):
    return np.array([(images[i] - images[i + 1]) / (images[i] + images[i + 1] + 1e-12)
                     for i in range(0, len(images), 2)])


def dpc_reconstruct(images, wl, dx, na, srcs, method="Tikhonov (phase only)", reg=1e-3, tv=1e-3,
                    iters=60, rho=1.0, monitor=None):
    sig = dpc_signals(images)
    Hs = dpc_transfer(images.shape[-2:], wl, dx, na, srcs)
    D = [fft2(s) for s in sig]
    if method.startswith("Tikhonov (phase + absorption)"):
        a11 = sum(np.abs(hm) ** 2 for hm, _ in Hs) + reg
        a22 = sum(np.abs(hp) ** 2 for _, hp in Hs) + reg
        a12 = sum(np.conj(hm) * hp for hm, hp in Hs)
        b1 = sum(np.conj(hm) * d for (hm, _), d in zip(Hs, D))
        b2 = sum(np.conj(hp) * d for (_, hp), d in zip(Hs, D))
        det = a11 * a22 - np.abs(a12) ** 2
        mu = (a22 * b1 - a12 * b2) / det
        ph = (a11 * b2 - np.conj(a12) * b1) / det
        return np.real(ifft2(ph)), np.real(ifft2(mu))
    num = sum(np.conj(hp) * d for (_, hp), d in zip(Hs, D))
    den = sum(np.abs(hp) ** 2 for _, hp in Hs)
    if method.startswith("Tikhonov"):
        return np.real(ifft2(num / (den + reg))), None
    # TV-regularised: min Σ‖H_k φ − d_k‖² + τ‖∇φ‖₁ by ADMM (split z = ∇φ)
    ny, nx = images.shape[-2:]
    Dx = np.zeros((ny, nx)); Dx[0, 0] = -1; Dx[0, 1] = 1
    Dy = np.zeros((ny, nx)); Dy[0, 0] = -1; Dy[1, 0] = 1
    FDx, FDy = fft2(Dx), fft2(Dy)
    Lap = np.abs(FDx) ** 2 + np.abs(FDy) ** 2
    phi = np.real(ifft2(num / (den + reg)))
    zx = np.zeros_like(phi); zy = np.zeros_like(phi); ux = zx.copy(); uy = zy.copy()
    for it in range(iters):
        rhs = num + rho * (np.conj(FDx) * fft2(zx - ux) + np.conj(FDy) * fft2(zy - uy))
        phi = np.real(ifft2(rhs / (den + rho * Lap + 1e-9)))
        Fp = fft2(phi)
        gx, gy = np.real(ifft2(FDx * Fp)), np.real(ifft2(FDy * Fp))
        vx, vy = gx + ux, gy + uy
        mag = np.sqrt(vx ** 2 + vy ** 2)
        shrink = np.maximum(1 - tv / rho / np.maximum(mag, 1e-12), 0)
        zx, zy = shrink * vx, shrink * vy
        ux += gx - zx; uy += gy - zy
        if monitor is not None:
            monitor.step(it + 1, iters)
    return phi, None
