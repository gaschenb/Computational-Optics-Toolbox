"""Phase retrieval.

Fourier (CDI) problems — recover x from |F x| and a support / positivity constraint:
    Error Reduction, Hybrid Input–Output, Difference Map, RAAR, ASR, shrink-wrap support update,
    Gerchberg–Saxton (two intensities), multi-plane (multi-distance) Gerchberg–Saxton.
Generic problems — recover x ∈ ℂⁿ from y = |A x|² (Gaussian or coded-diffraction A):
    Wirtinger flow, truncated WF, reshaped (amplitude) WF, alternating minimisation (GS),
    PhaseCut (block-coordinate SDP) and PhaseLift (proximal gradient on the PSD cone)."""
from __future__ import annotations

import numpy as np
import scipy.fft as sfft
from scipy import ndimage as ndi

from .fields import fft2, ifft2
from .propagation import angular_spectrum


# ============================================================================ Fourier CDI
def project_modulus(x, mag):
    X = fft2(x)
    return ifft2(mag * np.exp(1j * np.angle(X)))


def project_support(x, S, real=False, positive=False):
    y = np.where(S, x, 0)
    if real or positive:
        y = np.real(y).astype(complex)
    if positive:
        y = np.where(np.real(y) > 0, y, 0)
    return y


def fourier_error(x, mag):
    return float(np.linalg.norm(np.abs(fft2(x)) - mag) / (np.linalg.norm(mag) + 1e-20))


def shrinkwrap(x, sigma, thresh=0.12):
    a = ndi.gaussian_filter(np.abs(x), sigma)
    return a > thresh * a.max()


CDI_METHODS = ("ER", "HIO", "HIO + ER", "Difference Map", "RAAR", "ASR", "OSS")


def cdi(mag, support, method="HIO + ER", iters=300, beta=0.9, real=False, positive=False,
        shrink=False, shrink_every=20, sigma0=3.0, sigma_min=1.0, x0=None, rng=None, monitor=None,
        er_tail=0.1, avg_tail=0.0, shrink_start=0.0):
    """Iterative projection algorithms for |Fx| = mag, supp(x) ⊆ S.

    ER    x ← P_S P_M x
    HIO   x ← P_M x on S (where constraints hold), x − β P_M x elsewhere  (Fienup 1982)
    DM    x ← x + β [P_S((1+γ_M)P_M x − γ_M x) − P_M((1+γ_S)P_S x − γ_S x)],  γ_S = −1/β, γ_M = 1/β
    RAAR  x ← β/2 (R_S R_M + I) x + (1−β) P_M x        (Luke 2005)
    ASR   x ← ½ (R_S R_M + I) x                        (averaged successive reflections = DM, β=1)
    'HIO + ER' finishes with ER for the last 10 %. With `avg_tail` > 0 the constrained iterate P_S P_M x is
    averaged over that fraction of the HIO phase (before the ER tail) and the ER tail starts from the
    average: with noisy magnitudes HIO keeps wandering near the solution, and the average of one run is
    much less sensitive to the random start than the last iterate. Returns (x, support, error history)."""
    rng = np.random.default_rng(0) if rng is None else rng
    if method == "OSS":
        return oss(mag, support, iters, beta, real, positive, rng=rng, monitor=monitor, x0=x0)
    S = np.asarray(support, bool).copy()
    x = (np.abs(ifft2(mag * np.exp(2j * np.pi * rng.random(mag.shape))))).astype(complex) if x0 is None else x0
    x = x * S
    PS = lambda z: project_support(z, S, real, positive)
    PM = lambda z: project_modulus(z, mag)
    err = []
    sigma = sigma0
    n_er = int(er_tail * iters) if method == "HIO + ER" else 0
    n_avg = int(avg_tail * iters) if method == "HIO + ER" else 0
    acc, n_acc = None, 0
    for k in range(iters):
        m = method
        if method == "HIO + ER":
            m = "ER" if k >= iters - n_er else "HIO"
            if n_avg and k == iters - n_er and n_acc:
                x = acc / n_acc
        if m == "ER":
            x = PS(PM(x))
        elif m == "HIO":
            y = PM(x)
            ok = S.copy()
            if real or positive:
                ok = ok & (np.real(y) > 0 if positive else ok)
                y = np.real(y).astype(complex) if real or positive else y
            x = np.where(ok, y, x - beta * y)
        elif m == "Difference Map":
            gs, gm = -1.0 / beta, 1.0 / beta
            fs = (1 + gs) * PS(x) - gs * x
            fm = (1 + gm) * PM(x) - gm * x
            x = x + beta * (PS(fm) - PM(fs))
        elif m == "RAAR":
            pm = PM(x)
            rm = 2 * pm - x
            rs_rm = 2 * PS(rm) - rm
            x = 0.5 * beta * (rs_rm + x) + (1 - beta) * pm
        elif m == "ASR":
            pm = PM(x)
            rm = 2 * pm - x
            x = 0.5 * (2 * PS(rm) - rm + x)
        if shrink and k > shrink_start * iters and k % shrink_every == 0 and k < iters - n_er:
            est = PS(PM(x)) if m in ("Difference Map", "RAAR", "ASR") else x
            S = shrinkwrap(est, sigma)
            sigma = max(sigma_min, sigma * 0.95)
        est = PS(PM(x)) if m in ("Difference Map", "RAAR", "ASR") else PS(x)
        if n_avg and iters - n_er - n_avg <= k < iters - n_er:
            a = PS(PM(x))
            acc = a if acc is None else acc + a
            n_acc += 1
        err.append(fourier_error(est, mag))
        if monitor is not None:
            monitor.step(k + 1, iters, (lambda e=est, er=list(err), S=S: (e, er, S)) if monitor.want_preview() else None)
    final = PS(PM(x)) if method in ("Difference Map", "RAAR", "ASR") else PS(x)
    return final, S, np.array(err)


def oss(mag, support, iters=1000, beta=0.9, real=False, positive=False, stages=10, rng=None, monitor=None, x0=None):
    """Oversampling smoothness (Rodriguez et al. 2013): HIO whose values outside the support are smoothed
    by a Fourier-space Gaussian W = exp(−½ (f/α)²), with α shrinking from the full band to ~1 frequency
    sample over `stages` stages. The smoothing damps the high-frequency oscillations outside the support
    that let HIO wander between twin-mixed and noise-driven states, so one run converges reproducibly with
    noisy magnitudes. Returns the lowest-error estimate of the last stage."""
    rng = np.random.default_rng(0) if rng is None else rng
    S = np.asarray(support, bool)
    ny, nx = mag.shape
    x = (np.abs(ifft2(mag * np.exp(2j * np.pi * rng.random(mag.shape))))).astype(complex) if x0 is None else x0
    x = x * S
    FY, FX = np.meshgrid(sfft.fftfreq(ny) * ny, sfft.fftfreq(nx) * nx, indexing="ij")
    R2 = FX ** 2 + FY ** 2
    alphas = np.linspace(max(ny, nx), 1.0, stages)
    per = max(1, iters // stages)
    PS = lambda z: project_support(z, S, real, positive)
    err, best, best_err = [], None, np.inf
    for k in range(iters):
        st = min(stages - 1, k // per)
        y = project_modulus(x, mag)
        ok = S.copy()
        if real or positive:
            y = np.real(y).astype(complex)
            if positive:
                ok &= np.real(y) > 0
        x = np.where(ok, y, x - beta * y)
        W = np.exp(-0.5 * R2 / alphas[st] ** 2)
        x = np.where(S, x, ifft2(fft2(x) * W))
        est = PS(x)
        e = fourier_error(est, mag)
        err.append(e)
        if st == stages - 1 and e < best_err:
            best, best_err = est.copy(), e
        if monitor is not None:
            monitor.step(k + 1, iters, (lambda e=est, er=list(err): (e, er, S)) if monitor.want_preview() else None)
    return (best if best is not None else PS(x)), S, np.array(err)


def gerchberg_saxton(amp_obj, amp_fourier, iters=200, rng=None, monitor=None):
    """Classic GS: known amplitudes in two Fourier-conjugate planes, recover the phase."""
    rng = np.random.default_rng(0) if rng is None else rng
    x = amp_obj * np.exp(2j * np.pi * rng.random(amp_obj.shape))
    err = []
    for k in range(iters):
        X = fft2(x)
        err.append(float(np.linalg.norm(np.abs(X) - amp_fourier) / np.linalg.norm(amp_fourier)))
        x = ifft2(amp_fourier * np.exp(1j * np.angle(X)))
        x = amp_obj * np.exp(1j * np.angle(x))
        if monitor is not None:
            monitor.step(k + 1, iters)
    return x, np.array(err)


def multiplane_gs(amps, zs, wl, dx, iters=50, n_medium=1.0, monitor=None, x0=None):
    """Multi-distance GS (single-beam multiple-intensity reconstruction, Pedrini 2005): propagate
    between defocused planes z_k replacing the amplitude each time. Returns field at z_0."""
    K = len(zs)
    u = amps[0].astype(complex) if x0 is None else x0
    err = []
    for it in range(iters):
        e = 0.0
        for k in list(range(1, K)) + [0]:
            u = angular_spectrum(u, wl, dx, zs[k] - zs[(k - 1) % K], n_medium)
            e += np.linalg.norm(np.abs(u) - amps[k]) / np.linalg.norm(amps[k])
            u = amps[k] * np.exp(1j * np.angle(u))
        err.append(e / K)
        if monitor is not None:
            monitor.step(it + 1, iters)
    return u, np.array(err)


# ============================================================================ generic |Ax|²
class GaussianOp:
    """y = |A x|² with A an m×n complex Gaussian matrix (entries ~ CN(0, 1))."""

    def __init__(self, n, m, rng):
        self.n, self.m = n, m
        self.A = (rng.standard_normal((m, n)) + 1j * rng.standard_normal((m, n))) / np.sqrt(2)
        self.gram = float(m)            # AᴴA ≈ gram·I

    def __call__(self, x):
        return self.A @ x

    def H(self, z):
        return self.A.conj().T @ z

    def dense(self):
        return self.A


class CDPOp:
    """Coded diffraction patterns: A = [F D_1; …; F D_L], D_l random phase masks (octanary),
    the measurement model of Candès, Li & Soltanolkotabi 2015. x is an image of shape (ny, nx)."""

    def __init__(self, shape, L, rng):
        self.shape, self.L = shape, L
        b1 = rng.choice([1, -1, 1j, -1j], size=(L,) + shape)
        b2 = rng.choice([np.sqrt(0.5), np.sqrt(3)], p=[0.8, 0.2], size=(L,) + shape)
        self.D = b1 * b2
        self.n = int(np.prod(shape)); self.m = L * self.n
        self.gram = float(np.mean(np.sum(np.abs(self.D) ** 2, axis=0)))

    def __call__(self, x):
        return fft2(self.D * x.reshape(self.shape)).ravel() / np.sqrt(self.n)

    def H(self, z):
        Z = z.reshape((self.L,) + self.shape)
        return (np.sum(np.conj(self.D) * ifft2(Z), axis=0) * np.sqrt(self.n)).ravel()

    def dense(self):
        raise ValueError("CDP operator is matrix-free")


def spectral_init(A, y, iters=50, rng=None, truncate=None):
    """Leading eigenvector of (1/m) Σ y_i a_i a_iᴴ (optionally truncated), scaled to √(mean y)."""
    rng = np.random.default_rng(0) if rng is None else rng
    w = y if truncate is None else y * (y <= truncate ** 2 * np.mean(y))
    z = rng.standard_normal(A.n) + 1j * rng.standard_normal(A.n)
    z /= np.linalg.norm(z)
    for _ in range(iters):
        z = A.H(w * A(z))
        z /= np.linalg.norm(z)
    lam = np.sqrt(np.sum(y) / A.gram)
    return lam * z


def wirtinger_flow(A, y, iters=500, mu_max=0.4, t0=330.0, x0=None, monitor=None, truth=None):
    """WF (Candès, Li & Soltanolkotabi 2015): z ← z − (μ/‖z₀‖²)·(1/m) Aᴴ[(|Az|² − y) ⊙ Az]
    (normalised by the frame constant of A so the same μ works for Gaussian and CDP models)."""
    z = spectral_init(A, y) if x0 is None else x0.copy()
    scale = A.gram * np.mean(y)
    err = []
    for k in range(iters):
        Az = A(z)
        g = A.H((np.abs(Az) ** 2 - y) * Az)
        mu = min(1 - np.exp(-(k + 1) / t0), mu_max)
        z = z - mu / scale * g
        err.append(_dist(z, truth) if truth is not None else float(np.linalg.norm(np.abs(A(z)) ** 2 - y) / np.linalg.norm(y)))
        if monitor is not None:
            monitor.step(k + 1, iters)
    return z, np.array(err)


def truncated_wf(A, y, iters=500, mu=0.2, a_lb=0.3, a_ub=5.0, a_h=5.0, a_y=3.0, monitor=None, truth=None):
    """TWF (Chen & Candès 2015): Poisson loss Σ |aᴴz|² − y log|aᴴz|², truncated spectral init and
    gradients that keep only terms with α_lb ≤ |aᴴz|/s ≤ α_ub and |y − |aᴴz|²| ≤ α_h K_t |aᴴz|/s
    (s = RMS of |Az|, which makes the rule scale-free for any A)."""
    z = spectral_init(A, y, truncate=a_y)
    err = []
    for k in range(iters):
        Az = A(z)
        az = np.abs(Az)
        s = np.sqrt(np.mean(az ** 2)) + 1e-20
        r = az / s
        Kt = np.mean(np.abs(y - az ** 2))
        keep = (r >= a_lb) & (r <= a_ub) & (np.abs(y - az ** 2) <= a_h * Kt * r)
        g = A.H(np.where(keep, 2 * (az ** 2 - y) / np.maximum(az ** 2, 1e-20) * Az, 0)) / A.gram
        z = z - mu * g
        err.append(_dist(z, truth) if truth is not None else float(np.linalg.norm(np.abs(A(z)) ** 2 - y) / np.linalg.norm(y)))
        if monitor is not None:
            monitor.step(k + 1, iters)
    return z, np.array(err)


def reshaped_wf(A, y, iters=500, mu=0.8, monitor=None, truth=None):
    """Reshaped / amplitude-based WF (Zhang & Liang 2016): gradient of ½ Σ (|a_iᴴz| − √y_i)²."""
    z = spectral_init(A, y)
    sy = np.sqrt(np.clip(y, 0, None))
    err = []
    for k in range(iters):
        Az = A(z)
        g = A.H(Az - sy * np.exp(1j * np.angle(Az))) / A.gram
        z = z - mu * g
        err.append(_dist(z, truth) if truth is not None else float(np.linalg.norm(np.abs(A(z)) - sy) / np.linalg.norm(sy)))
        if monitor is not None:
            monitor.step(k + 1, iters)
    return z, np.array(err)


def alt_min(A, y, iters=200, monitor=None, truth=None):
    """Alternating minimisation / GS for generic A (Netrapalli et al. 2013): spectral init, then
    z ← argmin ‖Az − √y ⊙ phase(Az)‖ (LS solve with the dense A or CG for operators)."""
    z = spectral_init(A, y)
    sy = np.sqrt(np.clip(y, 0, None))
    err = []
    dense = None
    try:
        dense = A.dense()
        pinv = np.linalg.pinv(dense)
    except ValueError:
        pinv = None
    for k in range(iters):
        b = sy * np.exp(1j * np.angle(A(z)))
        if pinv is not None:
            z = pinv @ b
        else:          # CDP: A is a tight frame scaled by Σ|D|² → AᴴA = diag(Σ_l |D_l|²)
            z = A.H(b) / np.sum(np.abs(A.D) ** 2, axis=0).ravel()
        err.append(_dist(z, truth) if truth is not None else float(np.linalg.norm(np.abs(A(z)) - sy) / np.linalg.norm(sy)))
        if monitor is not None:
            monitor.step(k + 1, iters)
    return z, np.array(err)


def phasecut(A, y, iters=60, nu=1e-8, monitor=None, truth=None):
    """PhaseCut (Waldspurger, d'Aspremont & Mallat 2015): with b = √y, solve the MaxCut-like SDP
    min Tr(U M) s.t. diag U = 1, U ⪰ 0, M = diag(b)(I − A A⁺)diag(b), by block-coordinate descent;
    x = A⁺ diag(b) u with u the leading eigenvector. Dense, O(m²) memory — small problems only."""
    Ad = A.dense()
    m = Ad.shape[0]
    b = np.sqrt(np.clip(y, 0, None))
    P = Ad @ np.linalg.pinv(Ad)
    M = (b[:, None] * (np.eye(m) - P)) * b[None, :]
    U = np.eye(m, dtype=complex)
    err = []
    for k in range(iters):
        for i in range(m):
            idx = np.r_[0:i, i + 1:m]
            uc = U[np.ix_(idx, idx)] @ M[idx, i]
            gamma = np.real(np.vdot(uc, M[idx, i]))
            if gamma > 0:
                v = -np.sqrt((1 - nu) / gamma) * uc
            else:
                v = np.zeros(m - 1, complex)
            U[idx, i] = v
            U[i, idx] = np.conj(v)
        w, V = np.linalg.eigh(U)
        u = V[:, -1] * np.exp(-1j * 0)
        x = np.linalg.pinv(Ad) @ (b * np.exp(1j * np.angle(u)))
        err.append(_dist(x, truth) if truth is not None else float(np.linalg.norm(np.abs(Ad @ x) - b) / np.linalg.norm(b)))
        if monitor is not None:
            monitor.step(k + 1, iters)
    return x, np.array(err)


def phaselift(A, y, iters=300, lam=None, monitor=None, truth=None):
    """PhaseLift (Candès, Strohmer & Voroninski 2013): lift X = xxᴴ, solve
    min ½ Σ (a_iᴴ X a_i − y_i)² + λ Tr X, X ⪰ 0, by accelerated projected (prox) gradient;
    the estimate is the scaled leading eigenvector. O(n²) unknowns — small n only."""
    Ad = A.dense()
    m, n = Ad.shape
    lam = 1e-3 * np.mean(y) if lam is None else lam
    AX = lambda X: np.real(np.einsum("ij,jk,ik->i", Ad, X, Ad.conj()))
    AtR = lambda r: Ad.conj().T @ (r[:, None] * Ad)
    # step: 1/L with L ≈ ‖Σ |a_i|⁴‖ ~ m n²; estimate by power method
    X = np.zeros((n, n), complex)
    V = np.eye(n) / n
    for _ in range(10):
        V = AtR(AX(V)); V /= np.linalg.norm(V)
    L = np.linalg.norm(AtR(AX(V))) * 1.1
    Y = X.copy(); t = 1.0
    err = []
    for k in range(iters):
        G = AtR(AX(Y) - y)
        Z = Y - G / L
        Z = 0.5 * (Z + Z.conj().T)
        w, Q = np.linalg.eigh(Z)
        w = np.maximum(w - lam / L, 0)
        Xn = (Q * w) @ Q.conj().T
        t2 = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        Y = Xn + (t - 1) / t2 * (Xn - X)
        X, t = Xn, t2
        w, Q = np.linalg.eigh(X)
        x = Q[:, -1] * np.sqrt(max(w[-1], 0))
        err.append(_dist(x, truth) if truth is not None else float(np.linalg.norm(AX(X) - y) / np.linalg.norm(y)))
        if monitor is not None:
            monitor.step(k + 1, iters)
    return x, np.array(err)


def _dist(z, x):
    """dist(z, x) = min_φ ‖z − e^{iφ}x‖ / ‖x‖."""
    c = np.vdot(z, x)
    ph = c / abs(c) if abs(c) > 0 else 1
    return float(np.linalg.norm(x - z * ph) / (np.linalg.norm(x) + 1e-20))


GENERIC_METHODS = {"Wirtinger flow": wirtinger_flow, "truncated Wirtinger flow": truncated_wf,
                   "reshaped Wirtinger flow": reshaped_wf, "alternating minimisation (GS)": alt_min,
                   "PhaseCut (SDP)": phasecut, "PhaseLift (SDP)": phaselift}
