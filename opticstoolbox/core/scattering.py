"""Imaging through scattering media: speckle-correlation imaging within the optical memory effect
(Bertolotti 2012, Katz 2014). The camera sees I = O ∗ S (speckle PSF); its autocorrelation is
≈ O ⋆ O + const, so |Ô| is known and O follows from Fourier phase retrieval."""
from __future__ import annotations

import numpy as np
import scipy.fft as sfft
from scipy import ndimage as ndi

from .fields import fft2, freq_grid, ifft2, pad_to
from .phaseretrieval import cdi


def speckle_psf(n, grain=3.0, rng=None):
    """Idealised speckle intensity (unit mean, grain ≈ `grain` px): |F{aperture · random phase}|². Used by
    the ghost-imaging pattern generator; the scattering experiment uses `diffuser_psf` (ASM) instead."""
    rng = np.random.default_rng(9) if rng is None else rng
    FX, FY = freq_grid((n, n))
    ap = np.hypot(FX, FY) < 0.5 / grain
    S = np.abs(ifft2(ap * np.exp(2j * np.pi * rng.random((n, n))))) ** 2
    return S / S.mean()


# ----------------------------------------------------------------------------- physical forward model
# Geometry (all lengths in µm):  incoherent object --u--> thin diffuser + iris (diameter D) --v--> sensor.
# A point source at x_o illuminates the diffuser with a spherical wave; its linear phase term is a tilt,
# so behind a thin diffuser the camera PSF only shifts by −x_o·v/u (memory effect, magnification M = v/u).
# The PSF is the ASM propagation of (spherical wave × iris × diffuser phase) over v, computed on a grid
# finer than the camera pixel; the camera then integrates intensity over each pixel's active area.

def diffuser_phase(shape, dx, corr, sigma_phi, rng):
    """Gaussian-correlated random phase screen: ⟨φ(r)φ(r+ρ)⟩ = σφ² exp(−ρ²/ℓ²) with ℓ = `corr`."""
    w = rng.standard_normal(shape)
    w = ndi.gaussian_filter(w, corr / (2.0 * dx), mode="wrap")
    return sigma_phi * w / (w.std() + 1e-30)


def speckle_geometry(n=256, pixel=6.5, wl=0.532, D=1000.0, v=30000.0, u=300000.0, angle_deg=2.0,
                     obj_mm=2.5, fill=1.0, oversample=None, sigma_phi=3 * np.pi):
    """Derived sampling quantities for the diffuser → camera system, and the fine simulation grid.

    Sub-pixel samples q are chosen so that (i) the intensity |U|², whose spectrum extends to D/(λv), is
    Nyquist-sampled (dx ≤ λv/2D); (ii) the phase screen is resolved (dx ≤ ℓ/3); (iii) the pixel area and
    fill factor are resolved (q ≥ 2, ≥ 4 if fill < 1). The grid covers the iris, the sensor and the object's
    image, and the ASM is zero-padded 2× with Matsushima's band limit, so the propagation is a linear (not
    circular) convolution and the kernel's chirp is never aliased."""
    theta = np.deg2rad(angle_deg)
    corr = wl * sigma_phi * np.sqrt(2) / (2 * np.pi * theta)           # rms scattering angle θ per axis
    dx_need = min(wl * v / (2 * D), corr / 3)
    q = int(oversample) if oversample else max(2 if fill >= 0.999 else 4, int(np.ceil(pixel / dx_need)))
    dx = pixel / q
    M = v / u
    obj_px = obj_mm * 1e3 * M / pixel                                   # image of the object on the sensor
    W = int(np.ceil(obj_px * q))
    # steepest ray that reaches the sensor: iris edge → opposite sensor corner (+ the object's image shift)
    theta_need = np.arctan((D / 2 + n * pixel / np.sqrt(2) + obj_px * pixel / 2) / v)
    # Matsushima's band limit on the 2×-padded grid passes angles up to ≈ atan(L·dx / v): make it cover θ_need
    need = max(n * q + W + 8, int(np.ceil(1.1 * D / dx)) + 8, int(np.ceil(v * np.tan(theta_need) / dx * 1.05)))
    L = sfft.next_fast_len(need)
    dfx = 1.0 / (2 * L * dx)
    theta_bl = np.arcsin(min(1.0, 1.0 / np.sqrt((2 * dfx * v) ** 2 + 1)))
    grain = wl * v / D / pixel
    return dict(n=n, pixel=pixel, wl=wl, D=D, v=v, u=u, theta=theta, corr=corr, sigma_phi=sigma_phi, q=q, dx=dx,
                L=L, M=M, obj_px=obj_px, W=W, fill=fill, grain=grain, theta_need=theta_need, theta_bl=theta_bl,
                fresnel_number=(D / 2) ** 2 / (wl * v), band_limit_ok=theta_bl >= theta_need,
                intensity_nyquist=dx <= wl * v / (2 * D) * 1.0001, screen_resolved=dx <= corr / 3 * 1.0001,
                sensor_nyquist=grain >= 2.0)


def diffuser_psf(g, rng, source=(0.0, 0.0)):
    """Point-source PSF on the fine grid (L × L, pitch dx), intensity, plus diagnostics. `source` = (y, x)
    of the point in the object plane (µm); the simulation uses the on-axis PSF and the memory effect, and
    the tests check that an off-axis source gives the same pattern shifted by −source·v/u."""
    from .propagation import angular_spectrum, asm_kernel
    L, dx, wl = g["L"], g["dx"], g["wl"]
    x = (np.arange(L) - L // 2) * dx
    X, Y = np.meshgrid(x, x)
    R2 = X ** 2 + Y ** 2
    iris = R2 <= (g["D"] / 2) ** 2
    k = 2 * np.pi / wl
    sph = np.exp(1j * k * (np.sqrt((X - source[1]) ** 2 + (Y - source[0]) ** 2 + g["u"] ** 2) - g["u"]))              # exact spherical wave from the point
    phi = diffuser_phase((L, L), dx, g["corr"], g["sigma_phi"], rng)
    U0 = iris * sph * np.exp(1j * phi)
    U = angular_spectrum(U0, wl, dx, g["v"], bandlimit=True, pad=True)
    H = asm_kernel((2 * L, 2 * L), wl, dx, g["v"], bandlimit=True)
    spec = np.abs(fft2(pad_to(U0, (2 * L, 2 * L)))) ** 2
    kept = float((spec * (np.abs(H) > 0)).sum() / spec.sum())
    S = np.abs(U) ** 2
    return S, {"power kept by band limit": kept}


def pixel_integrate(I_fine, q, fill=1.0):
    """Camera sampling: integrate intensity over each pixel's active area (square, side √fill·pitch,
    centred), with fractional weights for partially covered sub-samples; returns one value per pixel."""
    n = I_fine.shape[0] // q
    a = np.sqrt(fill) * q                                                 # active width in sub-samples
    edges = np.arange(q + 1) - q / 2.0
    w1 = np.clip(np.minimum(edges[1:], a / 2) - np.maximum(edges[:-1], -a / 2), 0, None)
    B = I_fine[:n * q, :n * q].reshape(n, q, n, q)
    return np.einsum("iajb,a,b->ij", B, w1, w1)


def camera_sample(I_px, photons=None, rng=None, read_noise=0.0, bits=None, background=0.0):
    """Expected photo-electrons per pixel → Poisson shot noise → Gaussian read noise → ADC quantisation."""
    rng = np.random.default_rng(0) if rng is None else rng
    I = I_px / I_px.mean() * (photons or 1.0)
    I = I + background * I.mean()
    if photons:
        I = rng.poisson(I).astype(float)
        if read_noise:
            I = I + read_noise * rng.standard_normal(I.shape)
        if bits:
            full = max(I.max(), 1.0)
            step = full / (2 ** bits - 1)
            I = np.round(np.clip(I, 0, None) / step) * step
    return np.clip(I, 0, None)


def simulate_scattering(obj, g, photons=None, rng=None, read_noise=0.0, bits=None, background=0.0, frames=1):
    """Full forward model. `obj` is the object intensity (any size); it is resampled to its image on the
    fine grid (W × W sub-samples, inverted by the geometry), convolved with the ASM PSF (exact within the
    memory effect), cropped to the sensor and pixel-integrated. Returns (camera, truth, diagnostics) where
    truth is the inverted object image integrated by the same pixels. With `frames` > 1 the camera is a
    stack of exposures through independent diffuser realisations (e.g. a translated or rotated diffuser)."""
    from .fields import crop_to, resize
    rng = np.random.default_rng(0) if rng is None else rng
    L, q, n, W = g["L"], g["q"], g["n"], g["W"]
    o = np.clip(resize(np.asarray(obj, float), (W, W)), 0, None)[::-1, ::-1]
    O = pad_to(o, (L, L))
    FO = fft2(sfft.ifftshift(O))                                          # O centred at the origin → no shift
    cams = []
    for _ in range(max(1, int(frames))):
        S, diag = diffuser_psf(g, rng)
        I = np.clip(crop_to(np.real(ifft2(FO * fft2(S))), (n * q, n * q)), 0, None)
        cams.append(camera_sample(pixel_integrate(I, q, g["fill"]), photons, rng, read_noise, bits, background))
    camera = cams[0] if len(cams) == 1 else np.stack(cams)
    m = int(np.ceil(W / q)) * q
    truth = pad_to(pixel_integrate(pad_to(o, (m, m)), q, g["fill"]), (n, n))
    Sc = crop_to(S, (n * q, n * q))
    Sp = pixel_integrate(Sc, q, g["fill"])
    diag["speckle contrast (camera)"] = float(Sp.std() / Sp.mean())
    diag["speckle contrast (fine grid)"] = float(Sc.std() / Sc.mean())
    diag["measured grain (px)"] = speckle_grain(Sp)
    return camera, truth, diag


def speckle_grain(S):
    """Speckle grain = FWHM of the normalised intensity autocorrelation peak (px)."""
    J = S - S.mean()
    A = np.real(sfft.fftshift(ifft2(np.abs(fft2(J)) ** 2)))
    c = A.shape[0] // 2
    row = A[c] / A[c, c]
    above = np.nonzero(row[c:] < 0.5)[0]
    if not len(above):
        return float("nan")
    i = above[0]
    f = (row[c + i - 1] - 0.5) / (row[c + i - 1] - row[c + i] + 1e-30)
    return float(2 * (i - 1 + f))


def _glyph(ch, n=128):
    """One character, cropped to its bounding box and centred in a square."""
    from .fields import pad_to, text_image
    g = text_image(ch, n, size=0.8 * n)
    ys, xs = np.nonzero(g > 0.3)
    g = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    m = max(g.shape)
    return pad_to(g, (m, m))


def hidden_object(n, size=40, kind="letter", rng=None):
    from .fields import pad_to, resize, test_object
    small = _glyph("F") if kind == "letter" else test_object(kind, 128, rng)
    small = resize(small, (size, size))
    small = np.clip(small, 0, None)
    return pad_to(small, (n, n))


def camera_image(obj, S, photons=None, rng=None, background=0.0):
    rng = np.random.default_rng(0) if rng is None else rng
    I = np.real(ifft2(fft2(obj) * fft2(S))) + background
    I = np.clip(I, 0, None)
    if photons:
        k = photons / I.mean()
        I = rng.poisson(I * k) / k
    return I


def autocorrelation(I, window=None, envelope=None):
    """Mean-removed autocorrelation (centred) with an optional Tukey-like window radius. A slowly varying
    illumination envelope is removed with a Gaussian high-pass of width `envelope` (px; it must be
    well above the object size, else it eats the object's own autocorrelation)."""
    sig = I.shape[0] / 16 if envelope is None else envelope
    J = I - ndi.gaussian_filter(I, sig, mode="wrap")
    A = np.real(sfft.fftshift(ifft2(np.abs(fft2(J)) ** 2)))
    A -= np.median(A)
    if window:
        y, x = np.indices(A.shape) - np.array(A.shape)[:, None, None] // 2
        r = np.hypot(x, y) / window
        A *= np.where(r < 0.7, 1, np.where(r < 1, 0.5 * (1 + np.cos(np.pi * (r - 0.7) / 0.3)), 0))
    return A


SPECKLE_METHODS = ("HIO + ER", "OSS", "RAAR", "Difference Map", "ER")


def speckle_reconstruct(I, obj_size, method="HIO + ER", iters=1500, rng=None, monitor=None, window=1.5,
                        shrink=True, avg_tail=0.6, **cdi_kw):
    """Autocorrelation → Fourier magnitude → one phase-retrieval run with positivity and a box support.

    The measured magnitude is noisy (finite number of speckle grains per object-sized patch), so plain
    HIO never settles and its last iterate depends on the random start. Shrink-wrap tightens the support
    to the object, and the constrained iterate is averaged over the last `avg_tail` of the HIO phase before
    the ER tail; with the exact magnitude every start recovers the object exactly, and with measured data
    a single run is reproducible across starts once the autocorrelation is averaged over a few exposures
    (I may be a stack of exposures through independent diffuser realisations)."""
    rng = np.random.default_rng(0) if rng is None else rng
    I = np.asarray(I, float)
    stack = I.reshape((-1,) + I.shape[-2:])
    n = stack.shape[-1]
    A = np.mean([autocorrelation(f, window=window * obj_size, envelope=max(n / 16, 2.0 * obj_size)) for f in stack], axis=0)
    # A vanishes outside the window, so the phase retrieval runs on a crop of A that still oversamples the
    # object ≥ 2× per axis: same magnitudes, sampled more coarsely in frequency, far fewer pixels.
    from .fields import crop_to
    m = min(n, sfft.next_fast_len(int(np.ceil(max(4 * obj_size, 2 * window * obj_size + 4)))))
    Ac = crop_to(A, (m, m))
    mag = np.sqrt(np.clip(np.real(fft2(sfft.ifftshift(Ac))), 0, None))
    S = np.zeros((m, m), bool)
    h = int(np.ceil(0.6 * obj_size))
    S[m // 2 - h:m // 2 + h, m // 2 - h:m // 2 + h] = True
    x, Sx, err = cdi(mag, S, method, iters=iters, positive=True, beta=0.9, rng=rng, monitor=monitor,
                     shrink=shrink, avg_tail=avg_tail, **cdi_kw)
    x = pad_to(np.real(x), (n, n))
    return x, A, err
