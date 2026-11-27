"""Single-photon and two-photon imaging.

Single-photon:
  * photon-limited (Poisson) imaging — Anscombe + TV, and Poisson-likelihood TV by primal–dual
    (Chambolle–Pock with the closed-form KL conjugate prox);
  * first-photon imaging (Kirmani et al., Science 2014) — reflectivity from the number of pulses
    until the first detection (geometric likelihood + TV) and depth from the first arrival time with
    background censoring;
  * single-photon lidar — SPAD time-correlated histograms with background and pile-up (Coates
    correction), peak / log-matched-filter / censored ML + TV depth estimation.
Two-photon:
  * two-photon excitation microscopy — 3-D PSFs of 1P widefield-scanning, 1P confocal and 2P
    excitation, optical sectioning curves, depth attenuation, image formation and 3-D RL deconvolution;
  * biphoton (SPDC) quantum ghost imaging — Monte-Carlo photon pairs with position correlation,
    heralded coincidence imaging and accidental-coincidence subtraction."""
from __future__ import annotations

import numpy as np
import scipy.fft as sfft
from scipy import ndimage as ndi

from .fields import fft2, freq_grid, ifft2
from .regularizers import div, grad, tv_prox

C_UM_PER_PS = 299.792458   # µm / ps  (c)


# ============================================================================ photon-limited imaging
def anscombe(y):
    return 2 * np.sqrt(np.clip(y, 0, None) + 3 / 8)


def inverse_anscombe(z):
    """Closed-form approximation of the exact unbiased inverse (Mäkitalo & Foi 2011)."""
    z = np.maximum(z, 1e-6)
    return 0.25 * z ** 2 + 0.25 * np.sqrt(1.5) / z - 11 / 8 / z ** 2 + 0.625 * np.sqrt(1.5) / z ** 3 - 1 / 8


POISSON_METHODS = ("raw counts", "Gaussian smoothing", "Anscombe + TV", "Poisson TV (primal–dual)")


def poisson_tv(y, H=None, lam=0.05, iters=200, b=0.0, monitor=None):
    """min_x Σ (Ax + b − y log(Ax + b)) + λ TV(x), x ≥ 0, A = blur (H in FFT layout) or identity.
    Chambolle–Pock with K = [A; ∇]; prox of the KL conjugate is closed form."""
    A = (lambda x: x) if H is None else (lambda x: np.real(ifft2(fft2(x) * H)))
    At = (lambda x: x) if H is None else (lambda x: np.real(ifft2(fft2(x) * np.conj(H))))
    L = np.sqrt(1 + 8)                 # ‖K‖ ≤ √(‖A‖² + ‖∇‖²)
    tau = sigma = 0.99 / L
    x = np.maximum(y, 0.1).astype(float)
    xb = x.copy()
    p = np.zeros_like(x)               # dual for KL
    qx = np.zeros_like(x); qy = np.zeros_like(x)
    hist = []
    for k in range(iters):
        v = p + sigma * (A(xb) + b)
        p = 0.5 * (v + 1 - np.sqrt((v - 1) ** 2 + 4 * sigma * y))
        gx, gy = grad(xb)
        qx += sigma * gx; qy += sigma * gy
        nrm = np.maximum(1, np.sqrt(qx ** 2 + qy ** 2) / lam)
        qx /= nrm; qy /= nrm
        x_old = x
        x = np.maximum(x - tau * (At(p) - div(qx, qy)), 0)
        xb = 2 * x - x_old
        if monitor is not None:
            monitor.step(k + 1, iters)
        hist.append(float(np.linalg.norm(x - x_old) / (np.linalg.norm(x) + 1e-12)))
    return x, np.array(hist)


def photon_limited(y, method, lam=0.05, sigma=1.5, iters=200, H=None, monitor=None):
    if method == "raw counts":
        return y.astype(float)
    if method == "Gaussian smoothing":
        return ndi.gaussian_filter(y.astype(float), sigma)
    if method == "Anscombe + TV":
        z = anscombe(y)
        return np.maximum(inverse_anscombe(tv_prox(z, lam * 8, 40)), 0)
    return poisson_tv(y, H, lam, iters, monitor=monitor)[0]


# ============================================================================ first-photon imaging
def simulate_first_photon(refl, depth_um, eta=0.05, bg=0.002, jitter_ps=60.0, gate_ps=20000.0,
                          max_pulses=2000, rng=None):
    """Per pixel: pulses until the first detection n ~ Geometric(1 − e^{−(η a + B)}); that detection is
    a signal photon with probability η a/(η a + B), arriving at 2z/c + jitter, else uniform in the gate.
    Returns (n_pulses, arrival_ps, is_signal)."""
    rng = np.random.default_rng(0) if rng is None else rng
    S = eta * refl
    p = 1 - np.exp(-(S + bg))
    n = rng.geometric(np.clip(p, 1e-9, 1))
    n = np.minimum(n, max_pulses)
    sig = rng.random(refl.shape) < S / (S + bg + 1e-20)
    t_sig = 2 * depth_um / C_UM_PER_PS + jitter_ps * rng.standard_normal(refl.shape)
    t_bg = rng.uniform(0, gate_ps, refl.shape)
    t = np.where(sig, t_sig, t_bg)
    return n, t, sig


FIRST_PHOTON_METHODS = ("pixel-wise ML", "first-photon imaging (TV + censoring)")


def fpi_ml_reflectivity(n, eta, bg):
    """Pixel-wise ML from one geometric sample: p̂ = 1/n → â = (−log(1 − 1/n) − B)/η. n = 1 has an
    unbounded ML; it is replaced by the value for n = 1.5 (the estimate saturates)."""
    nn = np.maximum(np.asarray(n, float), 1.5)
    return np.maximum((-np.log(1 - 1 / nn) - bg) / eta, 0)


def fpi_reflectivity(n, eta, bg, lam=0.3, iters=80):
    """Regularised reflectivity: the ML estimate's spread scales with a (geometric statistics), so the
    log map has near-constant variance; it is TV-denoised there and mapped back."""
    a0 = fpi_ml_reflectivity(n, eta, bg)
    floor = np.quantile(a0[a0 > 0], 0.02) if np.any(a0 > 0) else 1e-3
    la = np.log(np.maximum(a0, floor))
    return np.exp(tv_prox(la, lam, iters))


def censor_and_regularise(t_ps, a_hat, eta, bg, gate_ps, jitter_ps, lam=0.3, window=3):
    """Reject arrivals far from the local median (likely background, more so where reflectivity is
    low), fill with the neighbourhood median, then TV-denoise the time-of-flight map."""
    med = ndi.median_filter(t_ps, size=window * 2 + 1)
    pb = bg / (eta * a_hat + bg + 1e-12)                 # probability that a detection is background
    thr = jitter_ps * (2 + 6 * (1 - pb))
    keep = np.abs(t_ps - med) < thr
    tc = np.where(keep, t_ps, med)
    tc = ndi.median_filter(tc, 3)
    scale = gate_ps
    return tv_prox(tc / scale, lam * jitter_ps / scale, 50) * scale, keep


# ============================================================================ single-photon lidar
def simulate_lidar(refl, depth_um, signal=5.0, background=20.0, bins=256, bin_ps=40.0, pulse_ps=80.0,
                   cycles=None, rng=None):
    """SPAD histograms. Default: low-flux (no pile-up) Poisson counts per bin. If `cycles` is given the
    detector records only the first photon of each laser cycle (synchronous mode, pile-up)."""
    rng = np.random.default_rng(0) if rng is None else rng
    tb = (np.arange(bins) + 0.5) * bin_ps
    t0 = 2 * depth_um / C_UM_PER_PS
    pulse = np.exp(-(tb[None, None, :] - t0[..., None]) ** 2 / (2 * pulse_ps ** 2))
    pulse /= pulse.sum(-1, keepdims=True) + 1e-20
    rate = signal * refl[..., None] * pulse + background / bins
    if not cycles:
        return rng.poisson(rate).astype(float), tb
    r = rate / cycles                                      # mean photons per bin per cycle
    cum = np.cumsum(r, -1)
    p_first = (1 - np.exp(-r)) * np.exp(-(cum - r))
    p_none = np.exp(-cum[..., -1:])
    probs = np.concatenate([p_first, p_none], -1)
    flat = probs.reshape(-1, bins + 1)
    flat = flat / flat.sum(1, keepdims=True)
    H = np.array([rng.multinomial(cycles, f) for f in flat])[:, :bins]
    return H.reshape(refl.shape + (bins,)).astype(float), tb


def coates(h, cycles):
    """Coates pile-up correction: λ̂_t = −log(1 − h_t / (N − Σ_{s<t} h_s))."""
    before = np.cumsum(h, -1) - h
    rem = np.maximum(cycles - before, 1)
    return -np.log(np.clip(1 - h / rem, 1e-9, 1)) * cycles


LIDAR_METHODS = ("histogram peak", "log-matched filter (ML)", "censored ML + TV")


def lidar_depth(H, tb, pulse_ps, method="log-matched filter (ML)", lam=0.5, background=None):
    """Depth (µm) and intensity estimates from TCSPC histograms."""
    bins = H.shape[-1]
    bin_ps = tb[1] - tb[0]
    k = np.arange(bins) - bins // 2
    kern = np.exp(-(k * bin_ps) ** 2 / (2 * pulse_ps ** 2))
    kern /= kern.sum()
    counts = H.sum(-1)
    if method == "histogram peak":
        idx = np.argmax(H, -1).astype(float)
    else:
        bgl = (np.median(H, -1, keepdims=True) if background is None else background / bins)
        # log-matched filter: Σ h_t log(S·g(t − τ) + b) maximised over τ (Poisson ML for unknown τ)
        S = np.maximum(counts[..., None] - bgl * bins, 0.5)
        logk = np.log(S * np.maximum(kern, 1e-12)[None, None, :] + np.maximum(bgl, 1e-3))
        F = np.real(sfft.ifft(sfft.fft(H, axis=-1) * np.conj(sfft.fft(sfft.ifftshift(logk[0, 0]))), axis=-1)) \
            if np.ndim(bgl) == 0 else None
        if F is None:
            Kf = sfft.fft(sfft.ifftshift(np.log(np.maximum(kern, 1e-12) + 1e-3)))
            F = np.real(sfft.ifft(sfft.fft(H, axis=-1) * np.conj(Kf)[None, None, :], axis=-1))
        idx = np.argmax(F, -1).astype(float)
        # sub-bin parabolic refinement
        i0 = idx.astype(int)
        fm = np.take_along_axis(F, ((i0 - 1) % bins)[..., None], -1)[..., 0]
        f0 = np.take_along_axis(F, i0[..., None], -1)[..., 0]
        fp = np.take_along_axis(F, ((i0 + 1) % bins)[..., None], -1)[..., 0]
        den = fm - 2 * f0 + fp
        idx = idx + np.where(den < 0, 0.5 * (fm - fp) / np.where(den == 0, -1, den), 0)
    t = (idx + 0.5) * bin_ps
    depth = t * C_UM_PER_PS / 2
    if method.startswith("censored"):
        med = ndi.median_filter(depth, 5)
        tol = 4 * pulse_ps * C_UM_PER_PS / 2
        keep = np.abs(depth - med) < tol
        depth = np.where(keep, depth, med)
        span = depth.max() - depth.min() + 1e-9
        depth = tv_prox((depth - depth.min()) / span, lam * 0.02, 60) * span + depth.min()
    bg_est = np.median(H, -1)
    intensity = np.maximum(counts - bg_est * bins, 0)
    return depth, intensity


# ============================================================================ two-photon microscopy
def psf3d(n, nz, dx, dz, wl, na, n_medium=1.33):
    """|field|² of a focused beam (scalar, angular-spectrum defocus) on an (nz, n, n) grid."""
    FX, FY = freq_grid((n, n), dx)
    P = (np.hypot(FX, FY) <= na / wl).astype(complex)
    kz = np.sqrt(np.clip((n_medium / wl) ** 2 - FX ** 2 - FY ** 2, 0, None))
    z = (np.arange(nz) - nz // 2) * dz
    U = ifft2(P[None] * np.exp(2j * np.pi * z[:, None, None] * kz[None]))
    I = np.abs(sfft.fftshift(U, axes=(-2, -1))) ** 2
    return I / I[nz // 2].sum()


TWO_PHOTON_MODES = ("1P scanning (no pinhole)", "1P confocal", "2P excitation")


def effective_psf(mode, n, nz, dx, dz, wl_1p=0.488, wl_2p=0.92, wl_em=0.52, na=1.0, n_medium=1.33,
                  pinhole_au=1.0):
    if mode.startswith("2P"):
        h = psf3d(n, nz, dx, dz, wl_2p, na, n_medium) ** 2
    elif mode.endswith("confocal"):
        hex_ = psf3d(n, nz, dx, dz, wl_1p, na, n_medium)
        hdet = psf3d(n, nz, dx, dz, wl_em, na, n_medium)
        r_au = 0.61 * wl_em / na * pinhole_au
        y, x = (np.indices((n, n)) - n // 2) * dx
        pin = (np.hypot(x, y) <= r_au).astype(float)
        hdet = np.real(ifft2(fft2(hdet) * fft2(sfft.ifftshift(pin))[None]))
        h = hex_ * np.maximum(hdet, 0)
    else:
        h = psf3d(n, nz, dx, dz, wl_1p, na, n_medium)
    return h / h.sum()


def sectioning_curve(h):
    s = h.sum(axis=(1, 2))
    return s / s.max()


def neuron_volume(n=96, nz=48, rng=None):
    """Somata (spheres) and dendrite-like filaments, axes (z, y, x), values in [0, 1]."""
    rng = np.random.default_rng(8) if rng is None else rng
    v = np.zeros((nz, n, n))
    Z, Y, X = np.meshgrid(np.arange(nz), np.arange(n), np.arange(n), indexing="ij")
    for _ in range(5):
        c = rng.uniform([0.2 * nz, 0.15 * n, 0.15 * n], [0.8 * nz, 0.85 * n, 0.85 * n])
        r = rng.uniform(3, 6)
        v[((Z - c[0]) * 2) ** 2 + (Y - c[1]) ** 2 + (X - c[2]) ** 2 < r ** 2] = 1.0
    for _ in range(14):
        p = rng.uniform([0, 0, 0], [nz, n, n])
        d = rng.standard_normal(3); d[0] *= 0.3; d /= np.linalg.norm(d)
        for s in range(0, 3 * n):
            q = p + 0.5 * s * d
            iz, iy, ix = np.rint(q).astype(int)
            if 0 <= iz < nz and 0 <= iy < n and 0 <= ix < n:
                v[iz, iy, ix] = max(v[iz, iy, ix], 0.7)
            d = d + 0.05 * rng.standard_normal(3) * np.array([0.3, 1, 1]); d /= np.linalg.norm(d)
    return np.clip(ndi.gaussian_filter(v, 0.6) * 1.5, 0, 1)


def image_3d(vol, h, photons=None, rng=None, ls_um=None, wl_factor=1.0, dz=0.25, exponent=1):
    """3-D convolution (FFT, h centred) with optional depth attenuation exp(−exponent·z/ℓ_s)."""
    rng = np.random.default_rng(0) if rng is None else rng
    v = vol
    if ls_um:
        z = np.arange(vol.shape[0])[:, None, None] * dz
        v = vol * np.exp(-exponent * z / (ls_um * wl_factor))
    img = np.real(sfft.ifftn(sfft.fftn(v) * sfft.fftn(sfft.ifftshift(h))))
    img = np.clip(img, 0, None)
    if photons:
        k = photons / max(img.max(), 1e-12)
        img = rng.poisson(img * k) / k
    return img


def rl3d(img, h, iters=20, monitor=None):
    H = sfft.fftn(sfft.ifftshift(h))
    x = np.full_like(img, img.mean())
    for k in range(iters):
        est = np.maximum(np.real(sfft.ifftn(sfft.fftn(x) * H)), 1e-12)
        x = x * np.real(sfft.ifftn(sfft.fftn(img / est) * np.conj(H)))
        if monitor is not None:
            monitor.step(k + 1, iters)
    return x


# ============================================================================ biphoton ghost imaging
def simulate_biphoton(obj, frames=200000, mu=0.2, sigma_corr=1.0, eta_s=0.5, eta_i=0.6, dark_bucket=1e-4,
                      dark_cam=1e-5, beam_sigma=None, rng=None):
    """SPDC photon pairs: per frame N ~ Poisson(μ) pairs; signal position uniform (or Gaussian pump),
    idler at the correlated position x_s + δ, δ ~ N(0, σ_corr²). The signal passes the object
    (transmission T) to a bucket detector; the idler hits a photon-counting camera. A frame is heralded
    when the bucket clicks. Returns dict with coincidence, accidental and raw camera images."""
    rng = np.random.default_rng(0) if rng is None else rng
    n = obj.shape[0]
    N = rng.poisson(mu * frames)
    fid = rng.integers(0, frames, N)
    if beam_sigma:
        xs = rng.normal(n / 2, beam_sigma, (N, 2))
    else:
        xs = rng.uniform(0, n, (N, 2))
    xi = xs + sigma_corr * rng.standard_normal((N, 2))
    iy_s = np.clip(xs[:, 0].astype(int), 0, n - 1); ix_s = np.clip(xs[:, 1].astype(int), 0, n - 1)
    sig_det = rng.random(N) < eta_s * obj[iy_s, ix_s]
    idl_det = (rng.random(N) < eta_i) & (xi >= 0).all(1) & (xi < n).all(1)
    herald = np.zeros(frames, bool)
    herald[fid[sig_det]] = True
    herald |= rng.random(frames) < dark_bucket
    # camera clicks: idlers + dark counts (random frames / pixels)
    cy = np.clip(xi[idl_det, 0].astype(int), 0, n - 1); cx = np.clip(xi[idl_det, 1].astype(int), 0, n - 1)
    cf = fid[idl_det]
    nd = rng.poisson(dark_cam * n * n * frames)
    df = rng.integers(0, frames, nd); dy = rng.integers(0, n, nd); dxx = rng.integers(0, n, nd)
    f_all = np.concatenate([cf, df]); y_all = np.concatenate([cy, dy]); x_all = np.concatenate([cx, dxx])
    coinc = np.zeros((n, n)); acc = np.zeros((n, n)); raw = np.zeros((n, n))
    h = herald[f_all]
    np.add.at(coinc, (y_all[h], x_all[h]), 1)
    # accidentals: herald of frame k with camera frame k+1 (uncorrelated by construction)
    h_shift = herald[(f_all - 1) % frames]
    np.add.at(acc, (y_all[h_shift], x_all[h_shift]), 1)
    np.add.at(raw, (y_all, x_all), 1)
    return dict(coincidences=coinc, accidentals=acc, raw=raw, heralds=int(herald.sum()), pairs=int(N))


BIPHOTON_METHODS = ("coincidence image", "accidental-subtracted", "unheralded camera (classical)")
