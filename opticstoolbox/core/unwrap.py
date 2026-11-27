"""2-D phase unwrapping: Itoh line integration, Goldstein branch cuts, quality-guided flood fill,
unweighted least squares (DCT Poisson), weighted least squares (PCG with a DCT preconditioner) and
PUMA (graph-cut energy minimisation, Bioucas-Dias & Valadão 2007)."""
from __future__ import annotations

import heapq
from collections import deque

import numpy as np
import scipy.fft as sfft
from scipy import ndimage as ndi
from scipy import sparse
from scipy.sparse import csgraph

from .fields import TWO_PI, wrap


def _w(x):
    return (x + np.pi) % TWO_PI - np.pi


# ----------------------------------------------------------------------------- diagnostics
def residues(psi):
    """Residue charge (+1 / −1 / 0) of every 2×2 loop, shape (ny−1, nx−1)."""
    d1 = _w(psi[:-1, 1:] - psi[:-1, :-1])
    d2 = _w(psi[1:, 1:] - psi[:-1, 1:])
    d3 = _w(psi[1:, :-1] - psi[1:, 1:])
    d4 = _w(psi[:-1, :-1] - psi[1:, :-1])
    return np.rint((d1 + d2 + d3 + d4) / TWO_PI).astype(int)


def quality_pdv(psi, k=3):
    """Phase-derivative-variance quality (higher = better), Ghiglia & Pritt §3.4."""
    dx = np.zeros_like(psi); dy = np.zeros_like(psi)
    dx[:, :-1] = _w(np.diff(psi, axis=1)); dx[:, -1] = dx[:, -2]
    dy[:-1] = _w(np.diff(psi, axis=0)); dy[-1] = dy[-2]
    v = lambda g: np.sqrt(np.clip(ndi.uniform_filter(g * g, k) - ndi.uniform_filter(g, k) ** 2, 0, None))
    return -(v(dx) + v(dy))


def quality_from_amplitude(amp):
    a = np.asarray(amp, float)
    return a / (a.max() + 1e-12)


# ----------------------------------------------------------------------------- path following
def itoh(psi):
    """Row-then-column 1-D unwrapping (exact only without residues)."""
    out = np.unwrap(psi, axis=0)
    return np.unwrap(out, axis=1) if out.ndim > 1 else out


def _flood(psi, blocked, seed_order=None):
    """BFS integration of wrapped differences over unblocked pixels; blocked pixels filled last."""
    ny, nx = psi.shape
    out = np.zeros_like(psi)
    done = np.zeros(psi.shape, bool)
    seeds = seed_order if seed_order is not None else np.flatnonzero(~blocked.ravel())
    for s in seeds:
        if done.flat[s] or blocked.flat[s]:
            continue
        sy, sx = divmod(int(s), nx)
        out[sy, sx] = psi[sy, sx]; done[sy, sx] = True
        q = deque([(sy, sx)])
        while q:
            y, x = q.popleft()
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < ny and 0 <= xx < nx and not done[yy, xx] and not blocked[yy, xx]:
                    out[yy, xx] = out[y, x] + _w(psi[yy, xx] - psi[y, x])
                    done[yy, xx] = True
                    q.append((yy, xx))
    # cut pixels: take any unwrapped neighbour (repeat until filled)
    for _ in range(4):
        rem = np.argwhere(~done)
        if not len(rem):
            break
        for y, x in rem:
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < ny and 0 <= xx < nx and done[yy, xx]:
                    out[y, x] = out[yy, xx] + _w(psi[y, x] - psi[yy, xx]); done[y, x] = True
                    break
    out[~done] = psi[~done]
    return out


def _line(mask, y0, x0, y1, x1):
    n = int(max(abs(y1 - y0), abs(x1 - x0))) + 1
    ys = np.rint(np.linspace(y0, y1, n)).astype(int)
    xs = np.rint(np.linspace(x0, x1, n)).astype(int)
    mask[np.clip(ys, 0, mask.shape[0] - 1), np.clip(xs, 0, mask.shape[1] - 1)] = True


def goldstein(psi, max_box=None):
    """Goldstein–Zebker–Werner branch cuts: residues are paired (or tied to the border) by growing
    boxes until the net charge is zero; integration then avoids the cuts."""
    ny, nx = psi.shape
    R = residues(psi)
    cuts = np.zeros(psi.shape, bool)
    pos = {tuple(p): int(R[tuple(p)]) for p in np.argwhere(R != 0)}
    if not pos:
        return itoh(psi), cuts
    max_box = max_box or max(ny, nx) // 2
    balanced = set()
    occupied = np.zeros(R.shape, bool)
    for p in pos:
        occupied[p] = True
    for p0 in list(pos):
        if p0 in balanced:
            continue
        charge = pos[p0]
        tree = [p0]
        balanced.add(p0)
        box = 1
        while charge != 0 and box <= max_box:
            found_border = False
            for c in list(tree):
                cy, cx = c
                y0, y1 = max(cy - box, 0), min(cy + box, R.shape[0] - 1)
                x0, x1 = max(cx - box, 0), min(cx + box, R.shape[1] - 1)
                # border inside the box: connect and neutralise
                if cy - box < 0 or cx - box < 0 or cy + box > R.shape[0] - 1 or cx + box > R.shape[1] - 1:
                    by = 0 if cy - box < 0 else (ny - 1 if cy + box > R.shape[0] - 1 else cy)
                    bx = 0 if cx - box < 0 else (nx - 1 if cx + box > R.shape[1] - 1 else cx)
                    if by == cy and bx == cx:
                        bx = 0 if cx < nx / 2 else nx - 1
                    _line(cuts, cy, cx, by, bx)
                    charge = 0
                    found_border = True
                    break
                sub = np.argwhere(occupied[y0:y1 + 1, x0:x1 + 1])
                for dy, dx in sub:
                    q = (y0 + dy, x0 + dx)
                    if q in tree:
                        continue
                    _line(cuts, cy, cx, q[0], q[1])
                    tree.append(q)
                    if q not in balanced:
                        charge += pos[q]
                        balanced.add(q)
                    if charge == 0:
                        break
                if charge == 0:
                    break
            if found_border:
                break
            box += 1
    return _flood(psi, cuts), cuts


def quality_guided(psi, quality=None):
    """Flood fill in order of decreasing quality (default: phase-derivative variance)."""
    ny, nx = psi.shape
    q = quality_pdv(psi) if quality is None else np.asarray(quality, float)
    out = psi.copy()
    done = np.zeros(psi.shape, bool)
    s = int(np.argmax(q))
    sy, sx = divmod(s, nx)
    done[sy, sx] = True
    heap = []

    def push(y, x):
        for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= yy < ny and 0 <= xx < nx and not done[yy, xx]:
                heapq.heappush(heap, (-q[yy, xx], yy, xx, y, x))
    push(sy, sx)
    while heap:
        _, y, x, py, px = heapq.heappop(heap)
        if done[y, x]:
            continue
        out[y, x] = out[py, px] + _w(psi[y, x] - psi[py, px])
        done[y, x] = True
        push(y, x)
    return out


# ----------------------------------------------------------------------------- least squares
def _wrapped_grads(psi):
    gx = np.zeros_like(psi); gy = np.zeros_like(psi)
    gx[:, :-1] = _w(np.diff(psi, axis=1))
    gy[:-1] = _w(np.diff(psi, axis=0))
    return gx, gy


def _div(gx, gy):
    d = gx.copy(); d[:, 1:] -= gx[:, :-1]
    e = gy.copy(); e[1:] -= gy[:-1]
    return d + e


def _poisson_dct(rho):
    ny, nx = rho.shape
    R = sfft.dctn(rho, type=2, norm="ortho", workers=-1)
    i = np.arange(ny)[:, None]; j = np.arange(nx)[None, :]
    den = 2 * np.cos(np.pi * i / ny) + 2 * np.cos(np.pi * j / nx) - 4
    den[0, 0] = 1
    P = R / den
    P[0, 0] = 0
    return sfft.idctn(P, type=2, norm="ortho", workers=-1)


def least_squares(psi):
    """Unweighted LS (Ghiglia & Romero 1994): ∇²φ = ∇·W(∇ψ) with Neumann boundaries, solved by DCT."""
    gx, gy = _wrapped_grads(psi)
    return _poisson_dct(_div(gx, gy))


def weighted_least_squares(psi, weights=None, iters=60, tol=1e-6, monitor=None):
    """Weighted LS by preconditioned conjugate gradients (DCT Poisson preconditioner)."""
    w = quality_pdv(psi) if weights is None else np.asarray(weights, float)
    w = w - w.min(); w = (w / (w.max() + 1e-12)) ** 2 + 1e-3
    wx = np.zeros_like(psi); wy = np.zeros_like(psi)
    wx[:, :-1] = np.minimum(w[:, 1:], w[:, :-1])
    wy[:-1] = np.minimum(w[1:], w[:-1])
    gx, gy = _wrapped_grads(psi)
    b = _div(wx * gx, wy * gy)

    def A(p):
        px = np.zeros_like(p); py = np.zeros_like(p)
        px[:, :-1] = np.diff(p, axis=1); py[:-1] = np.diff(p, axis=0)
        return _div(wx * px, wy * py)
    x = _poisson_dct(b)
    r = b - A(x)
    z = _poisson_dct(r); p = z.copy()
    rz = np.vdot(r, z).real
    nb = np.linalg.norm(b) + 1e-20
    for k in range(iters):
        Ap = A(p)
        alpha = rz / (np.vdot(p, Ap).real + 1e-30)
        x += alpha * p
        r -= alpha * Ap
        if monitor is not None:
            monitor.step(k + 1, iters)
        if np.linalg.norm(r) / nb < tol:
            break
        z = _poisson_dct(r)
        rz_new = np.vdot(r, z).real
        p = z + (rz_new / (rz + 1e-30)) * p
        rz = rz_new
    return x


def congruent(phi_smooth, psi):
    """Add the integer multiple of 2π that makes a smooth LS solution agree with the wrapped data."""
    return phi_smooth + _w(psi - phi_smooth)


# ----------------------------------------------------------------------------- PUMA
def puma(psi, p=1.0, weights=None, max_iter=60, monitor=None):
    """PUMA: minimise Σ_e w_e |Δψ_e + 2π(k_j − k_i)|^p over integer k by repeated binary graph cuts
    (exact for convex potentials p ≥ 1). Returns (phi, energy history)."""
    ny, nx = psi.shape
    N = ny * nx
    idx = np.arange(N).reshape(ny, nx)
    ei = np.concatenate([idx[:, :-1].ravel(), idx[:-1].ravel()])
    ej = np.concatenate([idx[:, 1:].ravel(), idx[1:].ravel()])
    flat = psi.ravel()
    D = flat[ej] - flat[ei]
    if weights is None:
        we = np.ones(len(ei))
    else:
        w = np.asarray(weights, float).ravel()
        we = np.minimum(w[ei], w[ej]) + 1e-6
    V = lambda x: np.abs(x) ** p
    k = np.zeros(N)
    S, T = N, N + 1
    hist = []
    for it in range(max_iter):
        d = D + TWO_PI * (k[ej] - k[ei])
        E = float(np.sum(we * V(d)))
        hist.append(E)
        A = we * V(d); Dd = A
        B = we * V(d + TWO_PI)        # δi=0, δj=1
        C = we * V(d - TWO_PI)        # δi=1, δj=0
        pair = B + C - A - Dd
        unary = np.zeros(N)
        np.add.at(unary, ei, C - A)
        np.add.at(unary, ej, Dd - C)
        scale = 1e4 / max(np.max(np.abs(pair)), np.max(np.abs(unary)), 1e-12)
        cp = np.rint(pair * scale).astype(np.int64)
        cu = np.rint(unary * scale).astype(np.int64)
        rows = [ei, np.full(np.sum(cu > 0), S), np.flatnonzero(cu < 0)]
        cols = [ej, np.flatnonzero(cu > 0), np.full(np.sum(cu < 0), T)]
        caps = [cp, cu[cu > 0], -cu[cu < 0]]
        r = np.concatenate(rows); c = np.concatenate(cols); cap = np.concatenate(caps)
        keep = cap > 0
        G = sparse.csr_matrix((cap[keep].astype(np.int32), (r[keep], c[keep])), shape=(N + 2, N + 2))
        G.sum_duplicates()
        # add zero-capacity reverse edges so the residual graph is complete
        Gr = G + sparse.csr_matrix((np.zeros(G.nnz, np.int32), (G.tocoo().col, G.tocoo().row)), shape=G.shape)
        res = csgraph.maximum_flow(Gr.astype(np.int32), S, T, method="dinic")
        Rg = (Gr - res.flow).tocsr()
        Rg.data[Rg.data <= 0] = 0
        Rg.eliminate_zeros()
        reach = csgraph.breadth_first_order(Rg, S, directed=True, return_predecessors=False)
        delta = np.ones(N + 2, bool); delta[reach] = False
        dk = delta[:N].astype(float)
        if res.flow_value == 0 or not dk.any() or dk.all():
            break
        k_new = k + dk
        d2 = D + TWO_PI * (k_new[ej] - k_new[ei])
        if np.sum(we * V(d2)) >= E - 1e-9:
            break
        k = k_new
        if monitor is not None:
            monitor.step(it + 1, max_iter)
    return (flat + TWO_PI * k).reshape(ny, nx), hist


METHODS = ("Itoh (row/column)", "Goldstein branch cuts", "quality-guided", "least squares (DCT)",
           "weighted least squares (PCG)", "PUMA (graph cuts)")


def unwrap(psi, method="quality-guided", quality=None, monitor=None, **kw):
    if method.startswith("Itoh"):
        return itoh(psi)
    if method.startswith("Goldstein"):
        return goldstein(psi)[0]
    if method.startswith("quality"):
        return quality_guided(psi, quality)
    if method.startswith("least squares"):
        return least_squares(psi)
    if method.startswith("weighted"):
        return weighted_least_squares(psi, quality, monitor=monitor, **kw)
    if method.startswith("PUMA"):
        return puma(psi, weights=None if quality is None else quality_from_amplitude(quality - np.min(quality)),
                    monitor=monitor, **kw)[0]
    raise ValueError(method)
