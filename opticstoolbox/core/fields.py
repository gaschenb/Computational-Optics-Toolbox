"""Grids, FFT helpers, test objects (phantoms) and image-quality metrics."""
from __future__ import annotations

import numpy as np
import scipy.fft as sfft
from scipy import ndimage as ndi

TWO_PI = 2.0 * np.pi


# ----------------------------------------------------------------------------- FFT helpers
def fft2(x, axes=(-2, -1)):
    return sfft.fft2(x, axes=axes, workers=-1)


def ifft2(x, axes=(-2, -1)):
    return sfft.ifft2(x, axes=axes, workers=-1)


def fft2c(x, axes=(-2, -1)):
    """Centred, unitary 2-D DFT (zero frequency in the middle)."""
    return sfft.fftshift(sfft.fft2(sfft.ifftshift(x, axes=axes), axes=axes, norm="ortho", workers=-1), axes=axes)


def ifft2c(x, axes=(-2, -1)):
    return sfft.fftshift(sfft.ifft2(sfft.ifftshift(x, axes=axes), axes=axes, norm="ortho", workers=-1), axes=axes)


def freqs(n, dx=1.0, shift=False):
    f = sfft.fftfreq(n, dx)
    return sfft.fftshift(f) if shift else f


def freq_grid(shape, dx=1.0, shift=False):
    ny, nx = shape
    fy, fx = freqs(ny, dx, shift), freqs(nx, dx, shift)
    return np.meshgrid(fx, fy)          # FX, FY


def coords(n, dx=1.0):
    """Centred coordinates (pixel n//2 at 0)."""
    return (np.arange(n) - n // 2) * dx


def grid(shape, dx=1.0):
    ny, nx = shape if np.ndim(shape) else (shape, shape)
    return np.meshgrid(coords(nx, dx), coords(ny, dx))   # X, Y


def pad_to(x, shape, value=0):
    ny, nx = x.shape[-2:]
    py, px = shape[0] - ny, shape[1] - nx
    pads = [(0, 0)] * (x.ndim - 2) + [(py // 2, py - py // 2), (px // 2, px - px // 2)]
    return np.pad(x, pads, constant_values=value)


def crop_to(x, shape):
    ny, nx = x.shape[-2:]
    y0, x0 = (ny - shape[0]) // 2, (nx - shape[1]) // 2
    return x[..., y0:y0 + shape[0], x0:x0 + shape[1]]


def wrap(phi):
    return np.angle(np.exp(1j * phi))


def normalize(x, lo=None, hi=None):
    x = np.asarray(x, float)
    lo = np.nanmin(x) if lo is None else lo
    hi = np.nanmax(x) if hi is None else hi
    return (x - lo) / (hi - lo + 1e-12)


def resize(img, shape, order=1):
    img = np.asarray(img)
    if img.shape[-2:] == tuple(shape):
        return img
    z = [1.0] * (img.ndim - 2) + [shape[0] / img.shape[-2], shape[1] / img.shape[-1]]
    if np.iscomplexobj(img):
        return ndi.zoom(img.real, z, order=order) + 1j * ndi.zoom(img.imag, z, order=order)
    out = ndi.zoom(img, z, order=order)
    return crop_to(pad_to(out, shape, out.mean()), shape) if out.shape[-2:] != tuple(shape) else out


def gaussian_blur(x, sigma):
    if np.iscomplexobj(x):
        return ndi.gaussian_filter(x.real, sigma) + 1j * ndi.gaussian_filter(x.imag, sigma)
    return ndi.gaussian_filter(x, sigma)


def shift_subpixel(x, dy, dx):
    """Fourier shift (periodic) by a fractional number of pixels."""
    FX, FY = freq_grid(x.shape[-2:])
    return ifft2(fft2(x) * np.exp(-2j * np.pi * (FX * dx + FY * dy)))


def register_translation(a, b, upsample=10):
    """Shift (dy, dx) that best maps b onto a, by cross-correlation with local upsampling."""
    A, B = fft2(a), fft2(b)
    xc = np.abs(ifft2(A * np.conj(B)))
    iy, ix = np.unravel_index(np.argmax(xc), xc.shape)
    ny, nx = a.shape
    dy = iy if iy <= ny // 2 else iy - ny
    dx = ix if ix <= nx // 2 else ix - nx
    if upsample > 1:   # parabolic refinement
        def par(c_m, c_0, c_p):
            den = c_m - 2 * c_0 + c_p
            return 0.0 if den == 0 else 0.5 * (c_m - c_p) / den
        dy += par(xc[(iy - 1) % ny, ix], xc[iy, ix], xc[(iy + 1) % ny, ix])
        dx += par(xc[iy, (ix - 1) % nx], xc[iy, ix], xc[iy, (ix + 1) % nx])
    return float(dy), float(dx)


# ----------------------------------------------------------------------------- test objects
_TEXT_CACHE: dict = {}


def text_image(text="OPTICS", n=256, size=None, weight="bold"):
    """Rasterised text (white on black), rendered with matplotlib's Agg backend."""
    key = (text, n, size, weight)
    if key in _TEXT_CACHE:
        return _TEXT_CACHE[key].copy()
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(1, 1), dpi=n)
    FigureCanvasAgg(fig)
    fig.patch.set_facecolor("black")
    lines = text.split("\n")
    fs = size or (0.62 * 72 / max(len(max(lines, key=len)), 1) * 1.6)
    fig.text(0.5, 0.5, text, color="white", ha="center", va="center", fontsize=fs, weight=weight,
             family="DejaVu Sans")
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., 0].astype(float) / 255.0
    img = resize(img, (n, n))
    _TEXT_CACHE[key] = img
    return img.copy()


def siemens_star(n=256, spokes=24, r=0.45):
    X, Y = grid(n, 1.0 / n)
    th, rr = np.arctan2(Y, X), np.hypot(X, Y)
    s = (np.cos(spokes * th) > 0).astype(float)
    return s * (rr < r)


def usaf_like(n=256):
    """USAF-1951-style resolution target (positive: bright bars on a dark field), anti-aliased.

    Each element is a set of three horizontal bars beside a set of three vertical bars, bars 5w long and
    w wide with w gaps; the bar width shrinks by 2^(1/6) per element and by 2 per group. As on the real
    chart, the two coarsest groups run down the left and right sides and the finer groups are nested,
    at a quarter of the size, in the middle."""
    ss = 4                                            # supersampling for anti-aliasing
    N = n * ss
    img = np.zeros((N, N))

    def rect(x0, y0, x1, y1):
        a, b = int(round(y0 * N)), int(round(y1 * N))
        c, d = int(round(x0 * N)), int(round(x1 * N))
        if b > a and d > c:
            img[max(a, 0):min(b, N), max(c, 0):min(d, N)] = 1.0

    def element(x, y, w):                             # top-left corner, bar width w (fractions of the field)
        for k in range(3):                            # horizontal bars (resolve vertical detail)
            rect(x, y + 2 * k * w, x + 5 * w, y + 2 * k * w + w)
        for k in range(3):                            # vertical bars
            rect(x + 7 * w + 2 * k * w, y, x + 7 * w + 2 * k * w + w, y + 5 * w)

    def column(x, y, w0, right=False):
        """Six elements of one group stacked from (x, y); right=True aligns them on their right edge."""
        for e in range(6):
            w = w0 * 2 ** (-e / 6)
            element(x - 12 * w if right else x, y, w)
            y += 7 * w

    def pair(x0, y0, S, depth):
        w0 = 0.92 * S / 32.1                          # 6 elements with 2w gaps fill 0.92 S
        column(x0 + 0.02 * S, y0 + 0.04 * S, w0)                  # group g: left side
        column(x0 + 0.98 * S, y0 + 0.04 * S, w0 / 2, right=True)  # group g+1: right side, top
        if depth > 1:
            pair(x0 + 0.42 * S, y0 + 0.5 * S, 0.4 * S, depth - 1)  # groups g+2, g+3 nested

    pair(0.0, 0.0, 1.0, 3)
    return img.reshape(n, ss, n, ss).mean(axis=(1, 3))


def beads(n=256, k=40, rng=None, sigma=1.0):
    """Sub-resolution fluorescent beads: random points blurred by `sigma` px (unit peak)."""
    rng = np.random.default_rng(5) if rng is None else rng
    img = np.zeros((n, n))
    m = int(0.1 * n)
    ys, xs = rng.integers(m, n - m, k), rng.integers(m, n - m, k)
    img[ys, xs] = rng.uniform(0.5, 1.0, k)
    img = ndi.gaussian_filter(img, sigma)
    return img / max(img.max(), 1e-12)


def blobs(n=256, k=14, rng=None, smooth=2.0):
    rng = np.random.default_rng(0) if rng is None else rng
    X, Y = grid(n, 1.0 / n)
    img = np.zeros((n, n))
    for _ in range(k):
        cx, cy = rng.uniform(-0.35, 0.35, 2)
        a, b = rng.uniform(0.03, 0.12, 2)
        t = rng.uniform(0, np.pi)
        xr = (X - cx) * np.cos(t) + (Y - cy) * np.sin(t)
        yr = -(X - cx) * np.sin(t) + (Y - cy) * np.cos(t)
        img += rng.uniform(0.4, 1.0) * ((xr / a) ** 2 + (yr / b) ** 2 < 1)
    return ndi.gaussian_filter(np.clip(img, 0, 1.5), smooth)


def cell_phantom(n=256, rng=None, cells=5, radius=0.16):
    """Optical thickness map of a few cells: ellipsoidal bodies with a nucleus and granules (0..1)."""
    rng = np.random.default_rng(3) if rng is None else rng
    X, Y = grid(n, 1.0 / n)
    t = np.zeros((n, n))
    cs = [(0.0, 0.0)] if cells == 1 else []
    while len(cs) < cells:
        c = rng.uniform(-0.3, 0.3, 2)
        if all(np.hypot(*(c - np.array(o))) > 1.4 * radius for o in cs) or len(cs) > 40:
            cs.append(tuple(c))
    for cx, cy in cs[:cells]:
        a, b = radius * rng.uniform(0.7, 1.1), radius * rng.uniform(0.55, 0.9)
        th = rng.uniform(0, np.pi)
        xr = (X - cx) * np.cos(th) + (Y - cy) * np.sin(th)
        yr = -(X - cx) * np.sin(th) + (Y - cy) * np.cos(th)
        q = 1 - (xr / a) ** 2 - (yr / b) ** 2
        t += 0.6 * np.sqrt(np.clip(q, 0, None))
        qn = 1 - ((xr - 0.2 * a) / (0.4 * a)) ** 2 - (yr / (0.45 * b)) ** 2
        t += 0.4 * np.sqrt(np.clip(qn, 0, None))
        for _ in range(6):
            gx, gy = rng.uniform(-0.6, 0.6) * a, rng.uniform(-0.6, 0.6) * b
            t += 0.12 * np.exp(-((xr - gx) ** 2 + (yr - gy) ** 2) / (2 * (0.012) ** 2)) * (q > 0)
    t = ndi.gaussian_filter(t, 0.8)
    return t / max(t.max(), 1e-12)


def shepp_logan(n=256):
    """Modified Shepp–Logan head phantom (Toft), values in [0, 1]."""
    E = [(1.0, .69, .92, 0, 0, 0), (-.8, .6624, .874, 0, -.0184, 0), (-.2, .11, .31, .22, 0, -18),
         (-.2, .16, .41, -.22, 0, 18), (.1, .21, .25, 0, .35, 0), (.1, .046, .046, 0, .1, 0),
         (.1, .046, .046, 0, -.1, 0), (.1, .046, .023, -.08, -.605, 0), (.1, .023, .023, 0, -.606, 0),
         (.1, .023, .046, .06, -.605, 0)]
    x = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, -x)
    img = np.zeros((n, n))
    for A, a, b, x0, y0, phi in E:
        p = np.deg2rad(phi)
        xr = (X - x0) * np.cos(p) + (Y - y0) * np.sin(p)
        yr = -(X - x0) * np.sin(p) + (Y - y0) * np.cos(p)
        img[(xr / a) ** 2 + (yr / b) ** 2 <= 1] += A
    return np.clip(img, 0, None)


def smooth_random(n=256, corr=12.0, rng=None):
    rng = np.random.default_rng(1) if rng is None else rng
    z = ndi.gaussian_filter(rng.standard_normal((n, n)), corr, mode="wrap")
    return z / (np.abs(z).max() + 1e-12)


def natural_texture(n=256, rng=None):
    """Band-limited 1/f texture in [0, 1] (a stand-in for a natural photograph)."""
    rng = np.random.default_rng(7) if rng is None else rng
    FX, FY = freq_grid((n, n))
    f = np.hypot(FX, FY)
    f[0, 0] = 1
    spec = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / f ** 1.25
    spec[0, 0] = 0
    img = np.real(ifft2(spec))
    return normalize(img)


OBJECTS = ("usaf", "siemens", "text", "letter", "cells", "blobs", "beads", "shepp-logan", "texture")


def objects(first=None, exclude=()):
    """All sample objects, with `first` as the default (first entry)."""
    names = [o for o in OBJECTS if o not in exclude]
    if first is not None:
        names.remove(first)
        names.insert(0, first)
    return tuple(names)


def test_object(name, n=256, rng=None):
    rng = np.random.default_rng(3) if rng is None else rng
    if name == "cells":
        return cell_phantom(n, rng)
    if name == "text":
        return ndi.gaussian_filter(text_image("OPTICS", n), 0.6)
    if name == "usaf":
        return usaf_like(n)
    if name == "siemens":
        return siemens_star(n)
    if name == "blobs":
        return blobs(n, rng=rng)
    if name == "shepp-logan":
        return shepp_logan(n)
    if name == "texture":
        return natural_texture(n, rng)
    if name == "beads":
        return beads(n, rng=rng)
    if name == "letter":
        g = text_image("F", n, size=54)
        return ndi.gaussian_filter(g, 0.5)
    raise ValueError(f"unknown object {name!r}; choose from {OBJECTS}")


def complex_object(amp_name="blobs", phase_name="cells", n=256, phase_max=np.pi, amp_contrast=0.3,
                   rng=None):
    """Transmission t = a·exp(iφ) with a ∈ [1-c, 1], φ ∈ [0, phase_max]."""
    rng = np.random.default_rng(3) if rng is None else rng
    a = 1.0 - amp_contrast * normalize(test_object(amp_name, n, rng)) if amp_contrast > 0 else np.ones((n, n))
    ph = phase_max * normalize(test_object(phase_name, n, rng))
    return a * np.exp(1j * ph)


# ----------------------------------------------------------------------------- noise
def poisson_noise(intensity, photons, rng):
    """Scale so the mean pixel receives `photons` photons, draw Poisson counts, scale back."""
    if photons is None or photons <= 0 or not np.isfinite(photons):
        return intensity
    m = max(float(np.mean(intensity)), 1e-12)
    k = photons / m
    return rng.poisson(np.clip(intensity, 0, None) * k) / k


def gaussian_noise(x, rel_sigma, rng):
    if rel_sigma <= 0:
        return x
    return x + rel_sigma * np.std(x) * rng.standard_normal(x.shape)


# ----------------------------------------------------------------------------- metrics
def psnr(x, ref, peak=None):
    x, ref = np.asarray(x, float), np.asarray(ref, float)
    peak = (ref.max() - ref.min()) if peak is None else peak
    mse = np.mean((x - ref) ** 2)
    return float(10 * np.log10(peak ** 2 / max(mse, 1e-20)))


def ssim(x, ref, sigma=1.5):
    x, ref = np.asarray(x, float), np.asarray(ref, float)
    L = ref.max() - ref.min() + 1e-12
    C1, C2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    mx, my = ndi.gaussian_filter(x, sigma), ndi.gaussian_filter(ref, sigma)
    sxx = ndi.gaussian_filter(x * x, sigma) - mx ** 2
    syy = ndi.gaussian_filter(ref * ref, sigma) - my ** 2
    sxy = ndi.gaussian_filter(x * ref, sigma) - mx * my
    s = ((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx ** 2 + my ** 2 + C1) * (sxx + syy + C2))
    return float(np.mean(s))


def fit_scale(x, ref, offset=True):
    """Least-squares a·x (+b) ≈ ref; returns the fitted image."""
    shape = np.shape(ref)
    x, ref = np.ravel(np.asarray(x, float)), np.ravel(np.asarray(ref, float))
    A = np.stack([x, np.ones_like(x)], 1) if offset else x[:, None]
    c, *_ = np.linalg.lstsq(A, ref, rcond=None)
    return (A @ c).reshape(shape)


def nrmse(x, ref, fit=True):
    ref = np.asarray(ref, float)
    xx = fit_scale(x, ref) if fit else np.asarray(x, float)
    return float(np.linalg.norm(xx - ref) / (np.linalg.norm(ref - (ref.mean() if fit else 0)) + 1e-12))


def remove_plane(phi, mask=None, order=1):
    """Subtract a least-squares polynomial (piston, tilt, optionally curvature) from a phase map."""
    ny, nx = phi.shape
    Y, X = np.mgrid[0:ny, 0:nx]
    X, Y = (X - nx / 2) / nx, (Y - ny / 2) / ny
    terms = [np.ones_like(X)]
    if order >= 1:
        terms += [X, Y]
    if order >= 2:
        terms += [X * X, Y * Y, X * Y]
    A = np.stack([t.ravel() for t in terms], 1)
    m = np.ones(phi.size, bool) if mask is None else mask.ravel()
    c, *_ = np.linalg.lstsq(A[m], phi.ravel()[m], rcond=None)
    return phi - (A @ c).reshape(phi.shape)


def phase_rmse(phi, ref, mask=None, tilt=False):
    """RMS phase error after removing the unobservable piston (and optionally tilt)."""
    d = wrap(np.asarray(phi) - np.asarray(ref))
    m = np.ones(d.shape, bool) if mask is None else mask
    if tilt:
        d = remove_plane(d, m, 1)
    else:
        d = wrap(d - np.angle(np.mean(np.exp(1j * d[m]))))
    return float(np.sqrt(np.mean(d[m] ** 2)))


def align_complex(x, ref, mask=None):
    """Multiply x by the global phase (and scale) that best matches ref."""
    m = np.ones(np.shape(ref), bool) if mask is None else mask
    c = np.vdot(x[m], ref[m]) / max(np.vdot(x[m], x[m]).real, 1e-20)
    return x * c


def complex_error(x, ref, mask=None):
    """min_c ‖c·x − ref‖ / ‖ref‖ (global phase and scale removed)."""
    m = np.ones(np.shape(ref), bool) if mask is None else mask
    xa = align_complex(x, ref, m)
    return float(np.linalg.norm(xa[m] - ref[m]) / (np.linalg.norm(ref[m]) + 1e-20))


def align_translation_and_phase(x, ref, mask=None):
    """Remove the translation / twin (conjugate-flip) ambiguity of Fourier phase retrieval, then the
    global phase; returns the aligned estimate and the error."""
    best = (np.inf, x)
    for cand in (x, np.conj(x[::-1, ::-1])):
        dy, dx = register_translation(np.abs(ref), np.abs(cand), upsample=1)
        c = np.roll(cand, (int(round(dy)), int(round(dx))), axis=(0, 1))
        e = complex_error(c, ref, mask) if np.iscomplexobj(ref) or np.iscomplexobj(c) else nrmse(np.real(c), ref)
        if e < best[0]:
            best = (e, align_complex(c, ref, mask) if np.iscomplexobj(c) else c)
    return best[1], best[0]
