"""Shared proximal operators and first-order solvers (TV, ℓ1, FISTA)."""
from __future__ import annotations

import numpy as np


def grad(x):
    """Forward differences with periodic boundaries on the last two axes → (gx, gy)."""
    return np.roll(x, -1, -1) - x, np.roll(x, -1, -2) - x


def div(px, py):
    """Negative adjoint of grad."""
    return px - np.roll(px, 1, -1) + py - np.roll(py, 1, -2)


def tv_norm(x):
    gx, gy = grad(x)
    return float(np.sum(np.sqrt(gx ** 2 + gy ** 2)))


def tv_prox(x, lam, iters=20):
    """argmin_u ½‖u − x‖² + λ TV(u) (isotropic) by Chambolle's dual projection; works on stacks."""
    if lam <= 0:
        return x
    px = np.zeros_like(x)
    py = np.zeros_like(x)
    tau = 0.249
    for _ in range(iters):
        u = div(px, py) - x / lam
        gx, gy = grad(u)
        nrm = 1 + tau * np.sqrt(gx ** 2 + gy ** 2)
        px = (px + tau * gx) / nrm
        py = (py + tau * gy) / nrm
    return x - lam * div(px, py)


def soft(x, t):
    if np.iscomplexobj(x):
        a = np.abs(x)
        return x * np.maximum(1 - t / np.maximum(a, 1e-20), 0)
    return np.sign(x) * np.maximum(np.abs(x) - t, 0)


def power_norm(A, At, shape, iters=15, rng=None):
    """‖A‖² by power iteration."""
    rng = np.random.default_rng(0) if rng is None else rng
    v = rng.standard_normal(shape)
    for _ in range(iters):
        v = At(A(v))
        v /= np.linalg.norm(v) + 1e-20
    return float(np.linalg.norm(At(A(v))))


def fista(A, At, b, shape, prox, iters=100, L=None, x0=None, monitor=None, nonneg=False, callback=None):
    """min ½‖Ax − b‖² + g(x) with prox(z, step) = prox_{step·g}(z). Returns (x, relative residuals)."""
    L = power_norm(A, At, shape) * 1.05 if L is None else L
    x = np.zeros(shape) if x0 is None else x0.copy()
    y = x.copy()
    t = 1.0
    hist = []
    nb = np.linalg.norm(b) + 1e-20
    for k in range(iters):
        r = A(y) - b
        z = prox(y - At(r) / L, 1.0 / L)
        if nonneg:
            z = np.maximum(z, 0)
        t2 = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        y = z + (t - 1) / t2 * (z - x)
        x, t = z, t2
        hist.append(float(np.linalg.norm(r) / nb))
        if callback is not None:
            callback(k, x)
        if monitor is not None:
            monitor.step(k + 1, iters, (lambda x=x, h=list(hist): (x, h)) if monitor.want_preview() else None)
    return x, np.array(hist)
