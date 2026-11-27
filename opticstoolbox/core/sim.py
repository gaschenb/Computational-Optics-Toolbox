"""Two-dimensional linear structured illumination microscopy (Gustafsson 2000).

Forward model: D_{a,p} = h ∗ [s · (1 + m cos(2π k_a·r + φ_p))] + noise for 3 angles × 3 phases.
Reconstruction: per-angle band separation (3×3 phase matrix), automatic estimation of the pattern
wave-vector (upsampled-DFT peak search), phases and modulation, sub-pixel band shifting in real
space, generalised Wiener combination with a triangular apodisation."""
from __future__ import annotations

import numpy as np
import scipy.fft as sfft

from .fields import fft2, freq_grid, grid, ifft2
from .waveoptics import incoherent_psf, otf, pupil


def sim_otf(shape, wl, dx, na):
    psf = incoherent_psf(pupil(shape, wl, dx, na))
    return psf, otf(psf)


def sim_patterns(shape, dx, k_mag, angles_deg, phases, m=0.9):
    X, Y = grid(shape, dx)
    pats = []
    for a in np.deg2rad(angles_deg):
        kx, ky = k_mag * np.cos(a), k_mag * np.sin(a)
        for p in phases:
            pats.append(1 + m * np.cos(2 * np.pi * (kx * X + ky * Y) + p))
    return np.array(pats)


def simulate_sim(obj, wl, dx, na, k_frac=0.9, angles_deg=(7, 67, 127), n_phases=3, m=0.9, photons=None,
                 rng=None, background=0.0):
    rng = np.random.default_rng(0) if rng is None else rng
    psf, H = sim_otf(obj.shape, wl, dx, na)
    fc = 2 * na / wl
    phases = 2 * np.pi * np.arange(n_phases) / n_phases
    pats = sim_patterns(obj.shape, dx, k_frac * fc, angles_deg, phases, m)
    raw = np.real(ifft2(fft2(obj[None] * pats) * H)) + background
    raw = np.clip(raw, 0, None)
    if photons:
        k = photons / raw.mean()
        raw = rng.poisson(raw * k) / k
    return raw.reshape(len(angles_deg), n_phases, *obj.shape), psf, H


def _peak(R, approx_k, dx, search=0.15):
    """Sub-pixel location (cycles/µm) of the spectral peak of image R near ±approx_k."""
    ny, nx = R.shape
    FX, FY = freq_grid(R.shape, dx)
    S = np.abs(fft2(R - R.mean()))
    kk = np.hypot(FX, FY)
    ka = np.hypot(*approx_k)
    win = (np.abs(kk - ka) < search * ka) & ((FX * approx_k[0] + FY * approx_k[1]) > 0)
    i = np.argmax(S * win)
    fy0, fx0 = FY.flat[i], FX.flat[i]
    # upsampled DFT refinement over ±1 bin
    up = 31
    dfx, dfy = 1 / (nx * dx), 1 / (ny * dx)
    fxs = fx0 + np.linspace(-1, 1, up) * dfx
    fys = fy0 + np.linspace(-1, 1, up) * dfy
    y = (np.arange(ny) - ny // 2) * dx; x = (np.arange(nx) - nx // 2) * dx
    Ey = np.exp(-2j * np.pi * np.outer(fys, y)); Ex = np.exp(-2j * np.pi * np.outer(x, fxs))
    Z = Ey @ (R - R.mean()) @ Ex
    j = np.unravel_index(np.argmax(np.abs(Z)), Z.shape)
    return np.array([fxs[j[1]], fys[j[0]]]), Z[j]


def estimate_parameters(raw, dx, wl, na, angles_guess_deg=None, k_frac_guess=0.9):
    """Pattern vector k_a (from the strongest off-DC peak of the phase-summed |FFT|), phases φ_p
    (argument of each raw spectrum at k_a) per angle."""
    A, P = raw.shape[:2]
    fc = 2 * na / wl
    if angles_guess_deg is None:
        angles_guess_deg = (7, 67, 127)[:A]
    ks, phs = [], []
    for a in range(A):
        g = np.deg2rad(angles_guess_deg[a])
        ka = k_frac_guess * fc * np.array([np.cos(g), np.sin(g)])
        k, _ = _peak(raw[a, 0], ka, dx, search=0.25)
        ks.append(k)
        X, Y = grid(raw.shape[-2:], dx)
        e = np.exp(-2j * np.pi * (k[0] * X + k[1] * Y))
        phs.append([np.angle(np.sum((raw[a, p] - raw[a, p].mean()) * e)) for p in range(P)])
    return np.array(ks), np.array(phs)


SIM_METHODS = ("Wiener SIM (estimated parameters)", "Wiener SIM (true parameters)", "widefield", "widefield + Wiener")


def reconstruct_sim(raw, dx, wl, na, ks=None, phases=None, m=None, w=0.05, apod=1.0, H=None):
    """Generalised Wiener SIM. Returns (image, dict of diagnostics)."""
    A, P, ny, nx = raw.shape
    if H is None:
        _, H = sim_otf((ny, nx), wl, dx, na)
    if ks is None or phases is None:
        ks_e, ph_e = estimate_parameters(raw, dx, wl, na)
        ks = ks_e if ks is None else ks
        phases = ph_e if phases is None else phases
    X, Y = grid((ny, nx), dx)
    psf = np.real(sfft.fftshift(ifft2(H)))
    num = np.zeros((ny, nx), complex)
    den = np.zeros((ny, nx))
    m_est = []
    D0H = None
    for a in range(A):
        R = np.array([fft2(raw[a, p]) for p in range(P)])
        ph = np.asarray(phases[a])
        M = np.stack([np.ones(P), 0.5 * np.exp(1j * ph), 0.5 * np.exp(-1j * ph)], 1)
        Minv = np.linalg.pinv(M)
        bands = np.tensordot(Minv, R, axes=(1, 0))          # D0, D+, D−
        k = ks[a]
        # modulation estimate from the overlap of D0 and the shifted D+
        e = np.exp(-2j * np.pi * (k[0] * X + k[1] * Y))
        Dp = fft2(ifft2(bands[1]) * e)                       # S(f) H(f + k)
        Hp = fft2(sfft.ifftshift(psf * e))                    # H(f + k)
        if m is None:
            ov = (np.abs(H) > 0.05) & (np.abs(Hp) > 0.05)
            num_m = np.sum(Dp[ov] * np.conj(bands[0][ov]) * H[ov] * np.conj(Hp[ov]))
            den_m = np.sum(np.abs(bands[0][ov]) ** 2 * np.abs(Hp[ov]) ** 2) + 1e-20
            ma = float(np.clip(np.abs(num_m / den_m), 0.05, 1.0))
        else:
            ma = m
        m_est.append(ma)
        Dm = fft2(ifft2(bands[2]) * np.conj(e))
        Hm = fft2(sfft.ifftshift(psf * np.conj(e)))
        for D, Hb, mb in ((bands[0], H, 1.0), (Dp / ma, Hp, 1.0), (Dm / ma, Hm, 1.0)):
            num += np.conj(Hb) * D
            den += np.abs(Hb) ** 2
        if a == 0:
            D0H = bands[0]
    FX, FY = freq_grid((ny, nx), dx)
    fc = 2 * na / wl
    kmax = np.max(np.hypot(ks[:, 0], ks[:, 1]))
    fe = fc + kmax
    Ap = np.clip(1 - np.hypot(FX, FY) / fe, 0, 1) ** apod
    S = num / (den + w ** 2 * A) * Ap
    img = np.real(ifft2(S))
    return img, dict(k=ks, phases=phases, modulation=np.array(m_est), spectrum=np.log1p(np.abs(sfft.fftshift(S))),
                     widefield_spectrum=np.log1p(np.abs(sfft.fftshift(D0H))))


def widefield(raw):
    return raw.reshape(-1, *raw.shape[-2:]).mean(0)


def wiener_widefield(raw, H, w=0.05, dx=None, wl=None, na=None):
    wf = widefield(raw)
    W = fft2(wf) * np.conj(H) / (np.abs(H) ** 2 + w ** 2)
    if dx is not None:
        FX, FY = freq_grid(wf.shape, dx)
        W *= np.clip(1 - np.hypot(FX, FY) / (2 * na / wl), 0, 1)
    return np.real(ifft2(W))
