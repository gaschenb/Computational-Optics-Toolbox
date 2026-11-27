"""Ghost imaging and single-pixel imaging.

Correlation ghost imaging (thermal speckle or random binary patterns): classical GI, differential
GI (Ferri 2010), normalised GI, pseudo-inverse and compressive (TV / FISTA) reconstruction.
Single-pixel imaging: differential Hadamard with sequency / 'cake-cutting' ordering and
sub-sampling, four-step Fourier single-pixel imaging (Zhang 2015), compressive TV recovery."""
from __future__ import annotations

import numpy as np
from scipy.linalg import hadamard

from .fields import fft2, freq_grid, ifft2
from .regularizers import fista, tv_prox


# ----------------------------------------------------------------------------- patterns
def speckle_patterns(M, n, grain=2.0, rng=None):
    """Pseudo-thermal speckle (rotating ground glass): |F{random phase · aperture}|²; grain ≈ pixels."""
    rng = np.random.default_rng(0) if rng is None else rng
    FX, FY = freq_grid((n, n))
    ap = np.hypot(FX, FY) < 0.5 / grain
    out = np.empty((M, n, n))
    for i0 in range(0, M, 64):
        k = min(64, M - i0)
        out[i0:i0 + k] = np.abs(ifft2(ap * np.exp(2j * np.pi * rng.random((k, n, n))))) ** 2
    return out / out.mean()


def binary_patterns(M, n, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    return (rng.random((M, n, n)) < 0.5).astype(float)


def _sequency(H):
    return np.sum(np.abs(np.diff(H, axis=1)) > 0, axis=1)


def hadamard_order(n, ordering="cake-cutting"):
    """Order of the n² 2-D Walsh patterns as (row, column) index pairs.
    natural | sequency (sum of row/column sequencies) | cake-cutting (number of connected
    regions, Yu et al. 2019): the low-'complexity' patterns first, which concentrates energy."""
    H1 = hadamard(n)
    seq = _sequency(H1)
    I, J = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    if ordering == "natural":
        return np.stack([I.ravel(), J.ravel()], 1)
    if ordering == "sequency":
        key = (seq[I] + seq[J]).ravel() * n * n + np.maximum(seq[I], seq[J]).ravel()
    else:
        key = ((seq[I] + 1) * (seq[J] + 1)).ravel() * n * n + (seq[I] + seq[J]).ravel()
    order = np.argsort(key, kind="stable")
    return np.stack([I.ravel()[order], J.ravel()[order]], 1)


def hadamard_pattern(H1, i, j):
    return np.outer(H1[i], H1[j]).astype(float)


# ----------------------------------------------------------------------------- measurement
def bucket(patterns, obj, noise=0.0, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    B = np.tensordot(patterns, obj, axes=([1, 2], [0, 1]))
    if noise:
        B = B + noise * np.std(B) * rng.standard_normal(B.shape)
    return B


GI_METHODS = ("classical GI", "differential GI", "normalised GI", "pseudo-inverse", "compressive (TV)")


def ghost_reconstruct(patterns, B, method="differential GI", lam=0.02, iters=150, monitor=None):
    """G(x) = ⟨B P(x)⟩ − ⟨B⟩⟨P(x)⟩ (classical); DGI replaces B by B − ⟨B⟩ S/⟨S⟩, S = Σ_x P;
    NGI normalises by ⟨B⟩; pseudo-inverse solves min ‖P̃x − B̃‖ (damped LSQR on mean-removed
    data); compressive adds isotropic TV (FISTA)."""
    M, ny, nx = patterns.shape
    P = patterns.reshape(M, -1)
    if method == "classical GI":
        G = (B - B.mean()) @ (P - P.mean(0)) / M
    elif method == "differential GI":
        S = P.sum(1)
        Bd = B - B.mean() / S.mean() * S
        G = Bd @ (P - P.mean(0)) / M
    elif method == "normalised GI":
        G = (B / B.mean()) @ P / M - P.mean(0)
    elif method == "pseudo-inverse":
        from scipy.sparse.linalg import lsqr
        Pc = P - P.mean(0)
        G = lsqr(Pc, B - B.mean(), damp=1e-3 * np.linalg.norm(Pc) / np.sqrt(P.shape[1]), iter_lim=200)[0]
    else:
        Pc = P - P.mean(0)
        Bc = B - B.mean()
        A = lambda x: Pc @ x.ravel()
        At = lambda r: (Pc.T @ r).reshape(ny, nx)
        L = np.linalg.norm(Pc, 2) ** 2 * 1.01
        scale = np.abs(At(Bc)).max() / L
        x, _ = fista(A, At, Bc, (ny, nx), lambda z, s: tv_prox(z, lam * scale * s * L, 15), iters=iters,
                     L=L, monitor=monitor)
        return x
    return G.reshape(ny, nx)


# ----------------------------------------------------------------------------- single-pixel imaging
SPI_METHODS = ("Hadamard (linear)", "Hadamard + TV", "Fourier four-step", "Fourier + TV")


def hadamard_spi(obj, ratio, ordering="cake-cutting", noise=0.0, rng=None):
    """Differential Hadamard measurements y_k = ⟨P⁺_k, x⟩ − ⟨P⁻_k, x⟩ = ⟨H_i ⊗ H_j, x⟩ for the
    first ratio·n² patterns (2 patterns each on a DMD). Returns (y, index pairs, H1)."""
    rng = np.random.default_rng(0) if rng is None else rng
    n = obj.shape[0]
    H1 = hadamard(n).astype(float)
    order = hadamard_order(n, ordering)
    K = max(1, int(round(ratio * n * n)))
    idx = order[:K]
    C = H1 @ obj @ H1.T
    y = C[idx[:, 0], idx[:, 1]]
    if noise:
        y = y + noise * np.std(y) * rng.standard_normal(y.shape)
    return y, idx, H1


def hadamard_recon(y, idx, H1, method="Hadamard (linear)", lam=0.01, iters=120, monitor=None):
    n = H1.shape[0]
    C = np.zeros((n, n))
    C[idx[:, 0], idx[:, 1]] = y
    x_lin = H1.T @ C @ H1 / (n * n)
    if method == "Hadamard (linear)":
        return x_lin
    ii, jj = idx[:, 0], idx[:, 1]
    A = lambda x: (H1 @ x @ H1.T)[ii, jj] / n          # ‖A‖ = 1

    def At(r):
        Cm = np.zeros((n, n))
        Cm[ii, jj] = r
        return H1.T @ Cm @ H1 / n
    x, _ = fista(A, At, y / n, (n, n), lambda z, s: tv_prox(z, lam * s, 15), iters=iters, L=1.01,
                 x0=x_lin, monitor=monitor)
    return x


def fourier_spi(obj, ratio, noise=0.0, rng=None):
    """Four-step Fourier SPI: for each sampled frequency f the patterns ½(1 + cos(2πf·r + φ_k)),
    φ_k = 0, π/2, π, 3π/2 give D(f) = (I₀ − I_π) + i(I_{π/2} − I_{3π/2}) ∝ the DFT coefficient.
    Frequencies are taken in order of |f|; Hermitian symmetry halves the measurement count.
    Returns (sampled spectrum, mask)."""
    rng = np.random.default_rng(0) if rng is None else rng
    n = obj.shape[0]
    F = fft2(obj)
    FX, FY = freq_grid((n, n))
    r = np.hypot(FX, FY)
    K = max(1, int(round(ratio * n * n)))
    thr = np.sort(r.ravel())[min(K, n * n - 1)]
    mask = r <= thr
    Fs = np.where(mask, F, 0)
    if noise:
        sig = noise * np.std(np.abs(F[mask]))
        Fs = Fs + mask * sig * (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
        Fs = 0.5 * (Fs + np.conj(np.roll(np.flip(Fs), 1, axis=(0, 1))))
    return Fs, mask


def fourier_recon(Fs, mask, method="Fourier four-step", lam=0.01, iters=120, monitor=None):
    x_lin = np.real(ifft2(Fs))
    if method == "Fourier four-step":
        return x_lin
    n = Fs.shape[0]
    k = int(mask.sum())

    def A(x):
        v = fft2(x)[mask] / n
        return np.concatenate([v.real, v.imag])

    def At(r):
        Cm = np.zeros((n, n), complex)
        Cm[mask] = r[:k] + 1j * r[k:]
        return np.real(ifft2(Cm)) * n
    b = Fs[mask] / n
    x, _ = fista(A, At, np.concatenate([b.real, b.imag]), (n, n), lambda z, s: tv_prox(z, lam * s, 15),
                 iters=iters, L=1.01, x0=x_lin, monitor=monitor)
    return x
