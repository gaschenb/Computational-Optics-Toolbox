"""3-D shape from optical cues.

Structure from defocus: layered thin-lens focal-stack synthesis; depth from focus (SML, Tenengrad,
local variance, with Gaussian peak interpolation and all-in-focus fusion); two-image depth from
defocus by relative-blur matching.
Light fields: 4-D synthesis of a layered Lambertian scene, shift-and-add refocusing, depth from
angular-variance (correspondence) and refocus-contrast (defocus) cues.
Structure from polarisation: diffuse / specular Fresnel degree-of-polarisation models, Stokes
estimation from polariser images, zenith inversion, azimuth-ambiguity resolution (convexity prior,
shading cue), and normal integration (Frankot–Chellappa, Poisson DCT)."""
from __future__ import annotations

import numpy as np
import scipy.fft as sfft
from scipy import ndimage as ndi

from .fields import freq_grid, grid, normalize


# ============================================================================ scenes
def depth_scene(n=256, kind="ramp + bumps", rng=None):
    """Depth map in [0, 1] (0 = near, 1 = far)."""
    rng = np.random.default_rng(2) if rng is None else rng
    X, Y = grid(n, 1.0 / n)
    if kind == "ramp + bumps":
        d = 0.5 + 0.6 * X
        for _ in range(5):
            c = rng.uniform(-0.35, 0.35, 2)
            d -= 0.25 * np.exp(-((X - c[0]) ** 2 + (Y - c[1]) ** 2) / (2 * 0.08 ** 2))
    elif kind == "steps":
        d = np.digitize(X + 0.3 * Y, np.linspace(-0.6, 0.6, 5)) / 5.0
    elif kind == "sphere":
        r2 = X ** 2 + Y ** 2
        d = np.where(r2 < 0.16, 1 - np.sqrt(np.clip(0.16 - r2, 0, None)) / 0.4, 1.0)
    else:
        d = 0.5 + 0.4 * np.sin(6 * X) * np.cos(4 * Y)
    return np.clip(d, 0, 1)


def blur_sigma(depth, focus, kappa):
    """Thin-lens blur ∝ |1/z − 1/z_f|; in the normalised (inverse-depth) units used here σ = κ|d − f|."""
    return kappa * np.abs(depth - focus) + 0.3


def layered_blur(texture, depth, focus, kappa, levels=16):
    """Depth-dependent Gaussian blur by soft layer compositing (each level blurred with its σ)."""
    out = np.zeros_like(texture)
    wsum = np.zeros_like(texture)
    for c in np.linspace(0, 1, levels):
        w = np.clip(1 - np.abs(depth - c) * (levels - 1), 0, 1)
        if not w.any():
            continue
        s = float(blur_sigma(c, focus, kappa))
        out += ndi.gaussian_filter(texture * w, s)
        wsum += ndi.gaussian_filter(w, s)
    return out / np.maximum(wsum, 1e-6)


def focal_stack(texture, depth, focuses, kappa, noise=0.0, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    st = np.array([layered_blur(texture, depth, f, kappa) for f in focuses])
    if noise:
        st = st + noise * rng.standard_normal(st.shape)
    return st


# ============================================================================ depth from focus / defocus
FOCUS_MEASURES = ("sum-modified Laplacian", "Tenengrad", "local variance")


def focus_measure(img, kind="sum-modified Laplacian", window=9):
    if kind.startswith("sum"):
        ml = np.abs(2 * img - np.roll(img, 1, 0) - np.roll(img, -1, 0)) + \
            np.abs(2 * img - np.roll(img, 1, 1) - np.roll(img, -1, 1))
        return ndi.uniform_filter(ml, window)
    if kind == "Tenengrad":
        gx, gy = ndi.sobel(img, 1), ndi.sobel(img, 0)
        return ndi.uniform_filter(gx ** 2 + gy ** 2, window)
    m = ndi.uniform_filter(img, window)
    return ndi.uniform_filter(img ** 2, window) - m ** 2


def depth_from_focus(stack, focuses, kind="sum-modified Laplacian", window=9):
    """argmax_k FM_k(x) with three-point Gaussian interpolation; returns (depth, all-in-focus, FM stack)."""
    FM = np.array([focus_measure(s, kind, window) for s in stack])
    k = np.argmax(FM, 0)
    K = len(focuses)
    km = np.clip(k - 1, 0, K - 1); kp = np.clip(k + 1, 0, K - 1)
    take = lambda a, i: np.take_along_axis(a, i[None], 0)[0]
    f0, fm, fp = np.log(take(FM, k) + 1e-12), np.log(take(FM, km) + 1e-12), np.log(take(FM, kp) + 1e-12)
    den = fm - 2 * f0 + fp
    off = np.where((k > 0) & (k < K - 1) & (den < 0), 0.5 * (fm - fp) / np.where(den == 0, -1, den), 0.0)
    fz = np.asarray(focuses, float)
    step = np.gradient(fz)
    depth = fz[k] + np.clip(off, -1, 1) * step[k]
    aif = take(stack, k)
    return depth, aif, FM


def depth_from_defocus(I1, I2, f1, f2, kappa, candidates=64, window=11):
    """Two-image DfD by relative blur: for each candidate depth d, the sharper image is blurred by
    √|σ₂² − σ₁²| and compared to the other one in a local window; depth = argmin error."""
    ds = np.linspace(0, 1, candidates)
    best = np.full(I1.shape, np.inf)
    depth = np.zeros(I1.shape)
    cache = {}
    for d in ds:
        s1, s2 = float(blur_sigma(d, f1, kappa)), float(blur_sigma(d, f2, kappa))
        r = np.sqrt(abs(s2 ** 2 - s1 ** 2))
        key = (round(r, 2), s2 > s1)
        if key not in cache:
            if s2 > s1:
                e = (ndi.gaussian_filter(I1, r) - I2) if r > 0.05 else (I1 - I2)
            else:
                e = (ndi.gaussian_filter(I2, r) - I1) if r > 0.05 else (I2 - I1)
            cache[key] = ndi.uniform_filter(e ** 2, window)
        E = cache[key]
        upd = E < best
        best[upd] = E[upd]
        depth[upd] = d
    return depth, best


# ============================================================================ light fields
def light_field(texture_layers, depths, n_views=7, max_disp=2.0):
    """L[u, v, y, x] of a layered scene (front-to-back occlusion). texture_layers: list of (img, alpha).
    Disparity per unit view offset = max_disp · (depth − 0.5) · 2."""
    n = texture_layers[0][0].shape[0]
    c = (n_views - 1) / 2
    L = np.zeros((n_views, n_views, n, n))
    order = np.argsort(depths)[::-1]                 # far first
    for iu in range(n_views):
        for iv in range(n_views):
            img = np.zeros((n, n))
            for li in order:
                tex, alpha = texture_layers[li]
                disp = max_disp * (depths[li] - 0.5) * 2
                sh = ((iu - c) * disp, (iv - c) * disp)
                t = ndi.shift(tex, sh, order=1, mode="wrap")
                a = ndi.shift(alpha, sh, order=1, mode="wrap")
                img = img * (1 - a) + t * a
            L[iu, iv] = img
    return L


def refocus(L, slope):
    """Shift-and-add: Ē(x) = mean_{u,v} L(u, v, x + slope·(u − c, v − c))."""
    nu = L.shape[0]
    c = (nu - 1) / 2
    acc = np.zeros(L.shape[2:])
    for iu in range(nu):
        for iv in range(L.shape[1]):
            acc += ndi.shift(L[iu, iv], (-(iu - c) * slope, -(iv - c) * slope), order=1, mode="wrap")
    return acc / (nu * L.shape[1])


def refocus_stack(L, slopes):
    return np.array([refocus(L, s) for s in slopes])


LF_METHODS = ("angular variance (correspondence)", "refocus contrast (defocus)", "combined")


def lf_depth(L, slopes, method="angular variance (correspondence)", window=7):
    nu = L.shape[0]
    c = (nu - 1) / 2
    V, C = [], []
    for s in slopes:
        views = np.array([ndi.shift(L[iu, iv], (-(iu - c) * s, -(iv - c) * s), order=1, mode="wrap")
                          for iu in range(nu) for iv in range(L.shape[1])])
        mean = views.mean(0)
        V.append(ndi.uniform_filter(views.var(0), window))
        C.append(focus_measure(mean, "sum-modified Laplacian", window))
    V, C = np.array(V), np.array(C)
    if method.startswith("angular"):
        k = np.argmin(V, 0)
    elif method.startswith("refocus"):
        k = np.argmax(C, 0)
    else:
        score = -V / (V.mean(0, keepdims=True) + 1e-12) + C / (C.mean(0, keepdims=True) + 1e-12)
        k = np.argmax(score, 0)
    return np.asarray(slopes)[k], V, C


# ============================================================================ polarisation
def dop_diffuse(theta, n):
    s, c = np.sin(theta), np.cos(theta)
    num = (n - 1 / n) ** 2 * s ** 2
    den = 2 + 2 * n ** 2 - (n + 1 / n) ** 2 * s ** 2 + 4 * c * np.sqrt(np.clip(n ** 2 - s ** 2, 0, None))
    return num / den


def dop_specular(theta, n):
    s, c = np.sin(theta), np.cos(theta)
    num = 2 * s ** 2 * c * np.sqrt(np.clip(n ** 2 - s ** 2, 0, None))
    den = n ** 2 - s ** 2 - n ** 2 * s ** 2 + 2 * s ** 4
    return num / np.maximum(den, 1e-12)


def surface_normals(depth_like):
    """Unit normals (nx, ny, nz) of a height map h(x, y) (pixel units), pointing to the viewer."""
    gy, gx = np.gradient(depth_like)
    nrm = np.sqrt(gx ** 2 + gy ** 2 + 1)
    return np.stack([-gx / nrm, -gy / nrm, 1 / nrm], -1)


def sfp_scene(n=256, kind="sphere + bumps", rng=None):
    """Height map (pixels) and object mask."""
    rng = np.random.default_rng(6) if rng is None else rng
    X, Y = grid(n, 1.0)
    R = 0.42 * n
    r2 = X ** 2 + Y ** 2
    mask = r2 < R ** 2
    h = np.sqrt(np.clip(R ** 2 - r2, 0, None))
    if kind == "sphere + bumps":
        for _ in range(6):
            c = rng.uniform(-0.5, 0.5, 2) * R
            h += 0.12 * R * np.exp(-((X - c[0]) ** 2 + (Y - c[1]) ** 2) / (2 * (0.12 * R) ** 2))
    elif kind == "vase":
        h = np.sqrt(np.clip(R ** 2 - X ** 2 * (1 + 0.3 * np.cos(4 * np.pi * Y / n)), 0, None))
        mask = (np.abs(X) < R / np.sqrt(1 + 0.3 * np.cos(4 * np.pi * Y / n))) & (np.abs(Y) < 0.45 * n)
    h = np.where(mask, h, 0)
    return ndi.gaussian_filter(h, 1.0), ndi.binary_erosion(mask, iterations=2)


def polarisation_images(normals, mask, n_ref=1.5, light=(0.3, -0.3, 1.0), albedo=0.8, model="diffuse",
                        angles_deg=(0, 45, 90, 135), noise=0.0, rng=None, ambient=0.15):
    rng = np.random.default_rng(0) if rng is None else rng
    l = np.asarray(light, float); l /= np.linalg.norm(l)
    theta = np.arccos(np.clip(normals[..., 2], -1, 1))
    az = np.arctan2(normals[..., 1], normals[..., 0])
    shade = albedo * (np.clip(normals @ l, 0, None) + ambient)
    if model == "diffuse":
        rho, phi = dop_diffuse(theta, n_ref), az
    else:
        rho, phi = dop_specular(theta, n_ref), az + np.pi / 2
    out = []
    for a in np.deg2rad(angles_deg):
        I = 0.5 * shade * (1 + rho * np.cos(2 * a - 2 * phi))
        out.append(np.where(mask, I, 0))
    out = np.array(out)
    if noise:
        out = out + noise * rng.standard_normal(out.shape) * mask
    return out


def stokes(ims, angles_deg=(0, 45, 90, 135)):
    """Linear Stokes parameters by least squares from any ≥3 polariser angles."""
    a = np.deg2rad(np.asarray(angles_deg))
    A = np.stack([np.ones_like(a), np.cos(2 * a), np.sin(2 * a)], 1) * 0.5
    coef = np.linalg.lstsq(A, ims.reshape(len(a), -1), rcond=None)[0]
    S0, S1, S2 = (c.reshape(ims.shape[1:]) for c in coef)
    dolp = np.sqrt(S1 ** 2 + S2 ** 2) / np.maximum(S0, 1e-9)
    aolp = 0.5 * np.arctan2(S2, S1)
    return S0, dolp, aolp


def invert_dop(rho, n, model="diffuse"):
    th = np.linspace(0, np.pi / 2 - 1e-3, 2000)
    if model == "diffuse":
        r = dop_diffuse(th, n)
        return np.interp(np.clip(rho, 0, r.max()), r, th)
    r = dop_specular(th, n)
    k = np.argmax(r)            # take the branch below the Brewster angle
    return np.interp(np.clip(rho, 0, r[k]), r[:k + 1], th[:k + 1])


SFP_DISAMBIG = ("convexity prior", "shading cue", "none (raw AoLP)")


def normals_from_polarisation(ims, mask, n_ref, model="diffuse", disambig="convexity prior", light=(0.3, -0.3, 1.0),
                              albedo=None, angles_deg=(0, 45, 90, 135)):
    S0, dolp, aolp = stokes(ims, angles_deg)
    theta = invert_dop(dolp, n_ref, model)
    phi = aolp if model == "diffuse" else aolp - np.pi / 2
    cand = []
    for p in (phi, phi + np.pi):
        cand.append(np.stack([np.sin(theta) * np.cos(p), np.sin(theta) * np.sin(p), np.cos(theta)], -1))
    if disambig.startswith("none"):
        choose = np.zeros(mask.shape, bool)
    elif disambig.startswith("convexity"):
        ys, xs = np.nonzero(mask)
        cy, cx = ys.mean(), xs.mean()
        Y, X = np.indices(mask.shape)
        out = np.stack([X - cx, Y - cy], -1)
        choose = np.sum(cand[1][..., :2] * out, -1) > np.sum(cand[0][..., :2] * out, -1)
    else:
        l = np.asarray(light, float); l /= np.linalg.norm(l)
        alb = S0[mask].max() if albedo is None else albedo
        obs = S0 / max(alb, 1e-9)
        e0 = np.abs(np.clip(cand[0] @ l, 0, None) - obs)
        e1 = np.abs(np.clip(cand[1] @ l, 0, None) - obs)
        choose = e1 < e0
        # resolve the remaining (shading-symmetric) pixels with the convexity prior
        sym = np.abs(e1 - e0) < 0.02
        if sym.any():
            ys, xs = np.nonzero(mask)
            Y, X = np.indices(mask.shape)
            out = np.stack([X - xs.mean(), Y - ys.mean()], -1)
            cv = np.sum(cand[1][..., :2] * out, -1) > np.sum(cand[0][..., :2] * out, -1)
            choose = np.where(sym, cv, choose)
    N = np.where(choose[..., None], cand[1], cand[0])
    N[~mask] = (0, 0, 1)
    return N, dict(S0=S0, dolp=dolp, aolp=aolp, theta=theta)


def angular_error_deg(N, Nt, mask):
    d = np.clip(np.sum(N * Nt, -1), -1, 1)
    return np.degrees(np.arccos(d)) * mask


INTEGRATORS = ("Frankot–Chellappa (FFT)", "least squares (DCT Poisson)")


def integrate_normals(N, mask, method="least squares (DCT Poisson)"):
    nz = np.clip(N[..., 2], 0.05, None)
    p = np.where(mask, -N[..., 0] / nz, 0)
    q = np.where(mask, -N[..., 1] / nz, 0)
    if method.startswith("Frankot"):
        FX, FY = freq_grid(p.shape)
        wx, wy = 2 * np.pi * FX, 2 * np.pi * FY
        P, Q = sfft.fft2(p), sfft.fft2(q)
        den = wx ** 2 + wy ** 2
        den[0, 0] = 1
        Z = (-1j * wx * P - 1j * wy * Q) / den
        Z[0, 0] = 0
        z = np.real(sfft.ifft2(Z))
    else:
        from .unwrap import _div, _poisson_dct
        gx = np.zeros_like(p); gy = np.zeros_like(q)
        gx[:, :-1] = 0.5 * (p[:, :-1] + p[:, 1:])
        gy[:-1] = 0.5 * (q[:-1] + q[1:])
        z = _poisson_dct(_div(gx, gy))
    z = z - np.median(z[~mask]) if (~mask).any() else z - z.min()
    return np.where(mask, z, 0)


def normals_rgb(N):
    return np.clip((N + 1) / 2, 0, 1)
