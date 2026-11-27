"""Digital holographic microscopy: off-axis Fourier filtering with automatic order detection,
aberration compensation (reference hologram, polynomial fit, PCA), phase-shifting (N-step,
unknown-shift AIA, PCA), Gabor in-line with iterative twin-image removal, multi-height lensless
retrieval, Kramers–Kronig single-shot reconstruction, slightly off-axis two-frame holography,
compressive holography and autofocus."""
from __future__ import annotations

import numpy as np
import scipy.fft as sfft
from scipy import ndimage as ndi

from .fields import TWO_PI, fft2, freq_grid, grid, ifft2, remove_plane, wrap
from .propagation import angular_spectrum
from .unwrap import least_squares as ls_unwrap


# ----------------------------------------------------------------------------- forward models
def reference_wave(shape, dx, wl, fx, fy, curvature_um=0.0, amp=1.0):
    """Tilted plane wave with carrier (fx, fy) in cycles/µm, optionally spherical with radius R."""
    X, Y = grid(shape, dx)
    R = amp * np.exp(2j * np.pi * (fx * X + fy * Y))
    if curvature_um:
        R = R * np.exp(1j * np.pi * (X ** 2 + Y ** 2) / (wl * curvature_um))
    return R


def hologram(obj_field, ref, photons=None, rng=None, bits=None):
    """Recorded intensity |O + R|², optionally Poisson-limited and quantised."""
    I = np.abs(obj_field + ref) ** 2
    if photons and rng is not None:
        k = photons / max(I.mean(), 1e-12)
        I = rng.poisson(I * k) / k
    if bits:
        q = (2 ** bits - 1) / I.max()
        I = np.round(I * q) / q
    return I


# ----------------------------------------------------------------------------- off-axis
def find_order(I, exclude=0.08, half="auto"):
    """Locate the +1 order: brightest spectral peak outside a DC disc of radius `exclude`×N.
    Returns (ky, kx) in FFT-shifted pixel coordinates, refined to sub-pixel by a centroid."""
    S = np.abs(sfft.fftshift(fft2(I - I.mean())))
    ny, nx = S.shape
    Yy, Xx = np.indices(S.shape)
    r = np.hypot(Yy - ny // 2, Xx - nx // 2)
    S2 = ndi.gaussian_filter(S, 1.0) * (r > exclude * min(nx, ny))
    if half == "auto":       # keep the half-plane containing the global maximum
        iy, ix = np.unravel_index(np.argmax(S2), S2.shape)
    else:
        S2 = S2 * (Xx > nx // 2) if half == "right" else S2 * (Yy > ny // 2)
        iy, ix = np.unravel_index(np.argmax(S2), S2.shape)
    # sub-pixel refinement: upsampled DFT (×20) of the hologram over ±1 bin around the peak
    up = 41
    fy = (iy - ny // 2 + np.linspace(-1, 1, up)) / ny
    fx = (ix - nx // 2 + np.linspace(-1, 1, up)) / nx
    y, x = np.arange(ny) - ny // 2, np.arange(nx) - nx // 2
    Ey = np.exp(-2j * np.pi * np.outer(fy, y))
    Ex = np.exp(-2j * np.pi * np.outer(x, fx))
    Z = np.abs(Ey @ (I - I.mean()) @ Ex)
    jy, jx = np.unravel_index(np.argmax(Z), Z.shape)
    cy, cx = ny // 2 + fy[jy] * ny, nx // 2 + fx[jx] * nx
    return float(cy), float(cx)


def offaxis_filter(I, order=None, radius=None, window="hann", dc_suppress=True, return_info=False):
    """Fourier-filter the +1 order, re-centre it (integer shift + residual linear phase) and return
    the complex field at the hologram plane. radius defaults to 1/3 of the carrier distance."""
    ny, nx = I.shape
    H = I - I.mean() if dc_suppress else I
    F = sfft.fftshift(fft2(H))
    cy, cx = find_order(I) if order is None else order
    dist = np.hypot(cy - ny // 2, cx - nx // 2)
    radius = dist / 3.0 if radius is None else radius
    Yy, Xx = np.indices(F.shape)
    rr = np.hypot(Yy - cy, Xx - cx) / radius
    if window == "hann":
        W = np.where(rr < 1, 0.5 * (1 + np.cos(np.pi * rr)), 0)
    elif window == "tukey":
        W = np.where(rr < 0.7, 1.0, np.where(rr < 1, 0.5 * (1 + np.cos(np.pi * (rr - 0.7) / 0.3)), 0))
    else:
        W = (rr < 1).astype(float)
    iy, ix = int(round(cy)), int(round(cx))
    G = np.roll(F * W, (ny // 2 - iy, nx // 2 - ix), axis=(0, 1))
    u = ifft2(sfft.ifftshift(G))
    fy, fx = (cy - iy) / ny, (cx - ix) / nx                     # residual carrier (cycles/pixel)
    Yp, Xp = np.indices(I.shape)
    u = u * np.exp(-2j * np.pi * (fx * Xp + fy * Yp))
    if return_info:
        return u, dict(order=(cy, cx), radius=radius, window=W, spectrum=np.log1p(np.abs(F)))
    return u


def compensate(u, method="polynomial fit", reference=None, order=2, mask=None, iters=3):
    """Remove the residual carrier and curvature (the 'numerical parametric lens').

    polynomial fit — fit a 2-D polynomial to the unwrapped phase over a background mask, re-fit
                     after excluding the specimen (robust, a few rounds);
    PCA            — Zuo et al. 2013: rank-1 SVD of exp(iφ); fit polynomials to the unwrapped
                     singular vectors (separable aberrations);
    reference      — divide by a blank (object-free) reconstructed field."""
    if method == "none":
        return u
    if method.startswith("reference"):
        if reference is None:
            raise ValueError("reference compensation needs a blank hologram")
        return u * np.exp(-1j * np.angle(reference))
    ph = np.angle(u)
    ny, nx = ph.shape
    if method.startswith("PCA"):
        E = np.exp(1j * ph)
        Uu, s, Vh = np.linalg.svd(E, full_matrices=False)
        uy = np.unwrap(np.angle(Uu[:, 0])); vx = np.unwrap(np.angle(Vh[0]))
        y = np.arange(ny) / ny - 0.5; x = np.arange(nx) / nx - 0.5
        py = np.polyval(np.polyfit(y, uy, order), y)
        px = np.polyval(np.polyfit(x, vx, order), x)
        ab = py[:, None] + px[None, :]
        out = u * np.exp(-1j * ab)
        return out * np.exp(-1j * np.angle(np.mean(out)))
    # polynomial fit on the unwrapped phase
    unw = ls_unwrap(ph)
    m = np.ones(ph.shape, bool) if mask is None else mask.copy()
    res = unw
    for _ in range(iters):
        res = remove_plane(unw, m, order)
        if mask is None:       # exclude the specimen: keep the flattest 60 %
            g = np.hypot(*np.gradient(ndi.gaussian_filter(res, 2)))
            dev = np.abs(res - np.median(res[m])) + 4 * g
            m = dev < np.quantile(dev, 0.6)
    ab = unw - res
    return u * np.exp(-1j * ab)


# ----------------------------------------------------------------------------- phase shifting
def phase_shifting(frames, shifts=None):
    """Least-squares N-step phase shifting: O·R* ∝ Σ_k I_k e^{iδ_k} (equal steps 2π/N by default);
    arbitrary known shifts use the full 3-unknown LS system per pixel."""
    I = np.asarray(frames, float)
    N = len(I)
    d = TWO_PI * np.arange(N) / N if shifts is None else np.asarray(shifts, float)
    A = np.stack([np.ones(N), np.cos(d), np.sin(d)], 1)          # I = a + b cos(φ+...) form
    coef = np.linalg.lstsq(A, I.reshape(N, -1), rcond=None)[0]
    # I_k = |O + R e^{iδ_k}|² = a + c cos δ_k + s sin δ_k with c + i s = 2 O R*
    field = (coef[1] + 1j * coef[2]).reshape(I.shape[1:])
    return field / 2


def aia(frames, iters=30, tol=1e-5, monitor=None):
    """Advanced iterative algorithm (Wang & Han 2004): alternate per-pixel LS for (a, b cos φ, b sin φ)
    given the shifts, and per-frame LS for the shifts given φ. Returns (field, shifts)."""
    I = np.asarray(frames, float)
    N = len(I)
    Iv = I.reshape(N, -1)
    d = TWO_PI * np.arange(N) / N + 0.3 * np.sin(np.arange(N))   # deliberately imperfect start
    for it in range(iters):
        A = np.stack([np.ones(N), np.cos(d), np.sin(d)], 1)
        c = np.linalg.lstsq(A, Iv, rcond=None)[0]                   # (3, P)
        B = np.stack([c[1], c[2]], 1)                               # per-pixel design for shifts
        g = np.linalg.lstsq(B, (Iv - c[0]).T, rcond=None)[0]        # (2, N): [cos δ, sin δ]
        dn = np.arctan2(g[1], g[0])
        dn = dn - dn[0]
        if monitor is not None:
            monitor.step(it + 1, iters)
        if np.max(np.abs(wrap(dn - (d - d[0])))) < tol:
            d = dn
            break
        d = dn
    A = np.stack([np.ones(N), np.cos(d), np.sin(d)], 1)
    c = np.linalg.lstsq(A, Iv, rcond=None)[0]
    return ((c[1] + 1j * c[2]).reshape(I.shape[1:]) / 2), wrap(d)


def pca_phase(frames):
    """PCA demodulation (Vargas et al. 2011): the first two principal components of the
    mean-subtracted frames are the quadrature signals. Global sign/offset is unobservable."""
    I = np.asarray(frames, float)
    N = len(I)
    X = I.reshape(N, -1)
    X = X - X.mean(0)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    c, si = Vt[0] * s[0], Vt[1] * s[1]
    return (c + 1j * si).reshape(I.shape[1:])


def slightly_offaxis(I1, I2, R_int, O_int, order=None, radius=None):
    """Two-frame slightly off-axis holography (Shaked et al. 2009): subtracting the separately
    measured |R|² + |O|² and combining frames shifted by π/2 cancels the DC and −1 orders, so a
    carrier of ~1× the object bandwidth (not 3×) suffices."""
    D1 = I1 - R_int - O_int
    D2 = I2 - R_int - O_int
    C = D1 + 1j * D2                      # = 2 O R*  (the −1 order cancels)
    return offaxis_filter_complex(C, order, radius)


def offaxis_filter_complex(C, order=None, radius=None):
    ny, nx = C.shape
    F = sfft.fftshift(fft2(C))
    S = np.abs(F)
    Yy, Xx = np.indices(F.shape)
    if order is None:
        r = np.hypot(Yy - ny // 2, Xx - nx // 2)
        S2 = ndi.gaussian_filter(S, 2) * (r > 3)
        order = np.unravel_index(np.argmax(S2), S2.shape)
    cy, cx = order
    dist = np.hypot(cy - ny // 2, cx - nx // 2)
    radius = dist if radius is None else radius
    W = (np.hypot(Yy - cy, Xx - cx) < radius)
    G = np.roll(F * W, (ny // 2 - int(round(cy)), nx // 2 - int(round(cx))), axis=(0, 1))
    return ifft2(sfft.ifftshift(G))


# ----------------------------------------------------------------------------- Kramers–Kronig
def kramers_kronig(I, ref_amp, carrier, axis=1):
    """Single-shot KK holography (Baek et al., Nat. Photon. 2019). With the reference at the edge of
    the object pupil the signal s = S/R has a one-sided spectrum, so ln(1+s) is analytic and its
    imaginary part is the Hilbert transform of ½ ln(I/|R|²). carrier = (fy, fx) cycles/pixel of R."""
    ny, nx = I.shape
    Yp, Xp = np.indices(I.shape)
    fy, fx = carrier
    re = 0.5 * np.log(np.clip(I, 1e-12, None) / ref_amp ** 2)
    FY, FX = np.meshgrid(sfft.fftfreq(ny), sfft.fftfreq(nx), indexing="ij")
    nvec = -np.array([fy, fx]) / (np.hypot(fy, fx) + 1e-12)   # s = S/R is centred at −f_R
    proj = FY * nvec[0] + FX * nvec[1]
    step = np.where(proj > 0, 2.0, np.where(proj == 0, 1.0, 0.0))
    chi = ifft2(fft2(re) * step)
    s = np.exp(chi) - 1.0
    R = ref_amp * np.exp(2j * np.pi * (fx * Xp + fy * Yp))
    return s * R, s


# ----------------------------------------------------------------------------- in-line holography
def gabor_backprop(I, wl, dx, z, n_medium=1.0):
    """Back-propagate √I from the sensor to the object plane (twin image included)."""
    return angular_spectrum(np.sqrt(np.clip(I, 0, None)).astype(complex), wl, dx, -z, n_medium)


def gabor_iterative(I, wl, dx, z, iters=50, constraint="absorption", support=None, n_medium=1.0,
                    monitor=None, beta=0.9):
    """Twin-image removal by alternating projections (Latychevskaia & Fink 2007): enforce the measured
    amplitude at the sensor and a physical transmission in the object plane:
    absorption  |t| ≤ 1 (no gain) and phase kept;  phase  |t| = 1;  support  t = 1 outside the mask."""
    amp = np.sqrt(np.clip(I / max(np.median(I), 1e-12), 0, None))
    t = angular_spectrum(amp.astype(complex), wl, dx, -z, n_medium)
    err = []
    for k in range(iters):
        if constraint == "phase":
            t = np.exp(1j * np.angle(t))
        else:
            a = np.abs(t)
            t = np.where(a > 1, t / np.maximum(a, 1e-12), t)
        if support is not None:
            t = np.where(support, t, beta * t + (1 - beta))
        u = angular_spectrum(t, wl, dx, z, n_medium)
        err.append(float(np.linalg.norm(np.abs(u) - amp) / np.linalg.norm(amp)))
        u = amp * np.exp(1j * np.angle(u))
        t = angular_spectrum(u, wl, dx, -z, n_medium)
        if monitor is not None:
            monitor.step(k + 1, iters, (lambda t=t, e=list(err): (t, e)) if monitor.want_preview() else None)
    return t, np.array(err)


def multi_height(Is, zs, wl, dx, iters=30, n_medium=1.0, monitor=None):
    """Multi-height lensless phase retrieval (Greenbaum & Ozcan 2012): cycle through the holograms
    recorded at sample-to-sensor distances z_k, replacing the amplitude at each plane.
    Returns (object-plane field at z_0, error history)."""
    amps = [np.sqrt(np.clip(I / max(np.median(I), 1e-12), 0, None)) for I in Is]
    u = amps[0].astype(complex)          # field at plane 0
    err = []
    K = len(zs)
    for it in range(iters):
        e = 0.0
        for k in list(range(1, K)) + [0]:
            prev = (k - 1) % K
            u = angular_spectrum(u, wl, dx, zs[k] - zs[prev], n_medium)
            e += np.linalg.norm(np.abs(u) - amps[k]) / np.linalg.norm(amps[k])
            u = amps[k] * np.exp(1j * np.angle(u))
        err.append(e / K)
        if monitor is not None:
            monitor.step(it + 1, iters)
    obj = angular_spectrum(u, wl, dx, -zs[0], n_medium)
    return obj, np.array(err)


# ----------------------------------------------------------------------------- autofocus
def focus_metrics():
    return ("Tamura coefficient", "amplitude variance", "gradient energy (Tenengrad)", "spectral L1 norm")


def focus_curve(u, wl, dx, zs, metric="Tamura coefficient", object_type="phase", n_medium=1.0):
    """Sharpness vs refocus distance. For phase objects the in-focus amplitude is *least* structured,
    so the curve is negated: the best focus is always the maximum."""
    stack = angular_spectrum(u, wl, dx, np.asarray(zs, float), n_medium)
    A = np.abs(stack)
    if metric.startswith("Tamura"):
        v = np.sqrt(A.std(axis=(-2, -1)) / A.mean(axis=(-2, -1)))
    elif metric.startswith("amplitude variance"):
        v = A.var(axis=(-2, -1)) / A.mean(axis=(-2, -1)) ** 2
    elif metric.startswith("gradient"):
        gy, gx = np.gradient(A, axis=(-2, -1))
        v = np.mean(gx ** 2 + gy ** 2, axis=(-2, -1)) / A.mean(axis=(-2, -1)) ** 2
    else:
        v = np.sum(np.abs(fft2(A)), axis=(-2, -1)) / np.abs(fft2(A))[..., 0, 0]
    if object_type == "phase":
        v = -v
    return np.asarray(v), stack


def best_focus(zs, v):
    i = int(np.argmax(v))
    if 0 < i < len(zs) - 1:
        a, b, c = v[i - 1], v[i], v[i + 1]
        den = a - 2 * b + c
        off = 0.5 * (a - c) / den if den != 0 else 0.0
        return float(zs[i] + off * (zs[i + 1] - zs[i]))
    return float(zs[i])


# ----------------------------------------------------------------------------- compressive holography
def _tv_prox(x, lam, iters=20):
    """Chambolle's projection for min ½‖u−x‖² + lam·TV(u) on real 2-D images (vectorised over a stack)."""
    px = np.zeros_like(x); py = np.zeros_like(x)
    tau = 0.249
    for _ in range(iters):
        div = px - np.roll(px, 1, -1) + py - np.roll(py, 1, -2)
        u = div - x / lam
        gx = np.roll(u, -1, -1) - u
        gy = np.roll(u, -1, -2) - u
        nrm = 1 + tau * np.sqrt(gx ** 2 + gy ** 2)
        px = (px + tau * gx) / nrm
        py = (py + tau * gy) / nrm
    return x - lam * (px - np.roll(px, 1, -1) + py - np.roll(py, 1, -2))


def compressive_forward(f, wl, dx, zs, n_medium=1.0):
    """Linearised Gabor model: g = 2 Re{Σ_z H_z f_z} (|R|² removed, |E|² neglected)."""
    E = np.sum(angular_spectrum(f.astype(complex), wl, dx, zs, n_medium), axis=0)
    return 2 * np.real(E)


def compressive_adjoint(g, wl, dx, zs, n_medium=1.0):
    return 2 * np.real(angular_spectrum(np.broadcast_to(g.astype(complex), (len(zs),) + g.shape),
                                        wl, dx, -np.asarray(zs), n_medium))


def compressive_holography(g, wl, dx, zs, reg="TV", lam=0.02, iters=60, nonneg=True, n_medium=1.0,
                           monitor=None):
    """Brady et al. 2009: recover a 3-D (multi-plane) real scattering density from ONE 2-D hologram by
    min ½‖g − 2Re(Hf)‖² + λ R(f), solved with FISTA; R = TV per plane or ℓ1 (sparse points)."""
    Z = len(zs)
    A = lambda f: compressive_forward(f, wl, dx, zs, n_medium)
    At = lambda r: compressive_adjoint(r, wl, dx, zs, n_medium)
    # Lipschitz constant by power iteration
    v = np.random.default_rng(0).standard_normal((Z,) + g.shape)
    for _ in range(8):
        v = At(A(v)); v /= np.linalg.norm(v)
    L = float(np.linalg.norm(At(A(v)))) * 1.05
    x = np.zeros((Z,) + g.shape); y = x.copy(); t = 1.0
    hist = []
    scale = lam * np.abs(At(g)).max()
    for k in range(iters):
        r = A(y) - g
        z = y - At(r) / L
        if reg == "TV":
            z = _tv_prox(z, scale / L)
        else:
            z = np.sign(z) * np.maximum(np.abs(z) - scale / L, 0)
        if nonneg:
            z = np.maximum(z, 0)
        t2 = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        y = z + (t - 1) / t2 * (z - x)
        x, t = z, t2
        hist.append(float(np.linalg.norm(r) / np.linalg.norm(g)))
        if monitor is not None:
            monitor.step(k + 1, iters)
    return x, np.array(hist)
