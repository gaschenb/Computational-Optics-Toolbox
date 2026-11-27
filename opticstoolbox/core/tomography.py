"""Tomography.

Projection (ray) tomography: matched pixel-driven projector / back-projector pair, filtered
back-projection with selectable filters, SIRT and OS-SART, slice-by-slice 3-D reconstruction.
Optical diffraction tomography (3-D refractive index): multislice beam-propagation forward model,
illumination-scanning or sample-rotation geometries, Fourier diffraction theorem with the first
Born or Rytov approximation, and Gerchberg–Papoulis non-negativity iterations that fill the
missing cone."""
from __future__ import annotations

import numpy as np
import scipy.fft as sfft
from scipy import ndimage as ndi

from .fields import TWO_PI
from .propagation import asm_kernel


# ============================================================================ projection tomography
_PROJ: dict = {}


def projector(n, angles_deg):
    """Sparse pixel-driven parallel-beam system matrix (linear splatting), cached; shape (A·n, n²)."""
    key = (n, tuple(np.round(np.asarray(angles_deg, float), 6)))
    if key in _PROJ:
        return _PROJ[key]
    from scipy import sparse
    c = np.arange(n) - (n - 1) / 2
    X, Y = np.meshgrid(c, -c)
    X, Y = X.ravel(), Y.ravel()
    pix = np.arange(n * n)
    rows, cols, vals = [], [], []
    for a, th in enumerate(np.deg2rad(angles_deg)):
        t = X * np.cos(th) + Y * np.sin(th) + (n - 1) / 2
        i0 = np.floor(t).astype(int)
        w1 = t - i0
        for ii, ww in ((i0, 1 - w1), (i0 + 1, w1)):
            ok = (ii >= 0) & (ii < n) & (ww > 0)
            rows.append(a * n + ii[ok]); cols.append(pix[ok]); vals.append(ww[ok])
    M = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(len(angles_deg) * n, n * n))
    if len(_PROJ) > 8:
        _PROJ.clear()
    _PROJ[key] = (M, M.T.tocsr())
    return _PROJ[key]


def radon(img, angles_deg):
    """Parallel-beam projection; img (..., n, n) → sinogram (..., A, n)."""
    img = np.asarray(img, float)
    n = img.shape[-1]
    M, _ = projector(n, angles_deg)
    flat = img.reshape((-1, n * n)).T
    return (M @ flat).T.reshape(img.shape[:-2] + (len(angles_deg), n))


def backproject(sino, angles_deg):
    """Exact transpose of radon; sino (..., A, n) → (..., n, n)."""
    sino = np.asarray(sino, float)
    n = sino.shape[-1]
    _, MT = projector(n, angles_deg)
    flat = sino.reshape((-1, len(angles_deg) * n)).T
    return (MT @ flat).T.reshape(sino.shape[:-2] + (n, n))


FBP_FILTERS = ("Ram-Lak", "Shepp-Logan", "cosine", "Hann", "Hamming")


def fbp(sino, angles_deg, filt="Shepp-Logan", cutoff=1.0):
    """Filtered back-projection: ramp |ω| times a window, zero-padded to avoid wrap-around."""
    sino = np.asarray(sino, float)
    n = sino.shape[-1]
    N = int(2 ** np.ceil(np.log2(2 * n)))
    f = sfft.fftfreq(N)
    # spatial-domain ramp (Kak & Slaney) avoids the DC bias of the sampled |ω|
    k = np.arange(N); k[k > N // 2] -= N
    h = np.zeros(N); h[0] = 0.25
    odd = k % 2 == 1
    h[odd] = -1.0 / (np.pi * k[odd]) ** 2
    H = np.real(sfft.fft(h)) * 2
    w = np.abs(f) / 0.5 / cutoff
    win = {"Ram-Lak": np.ones(N), "Shepp-Logan": np.sinc(w / 2), "cosine": np.cos(np.pi * w / 2),
           "Hann": 0.5 + 0.5 * np.cos(np.pi * w), "Hamming": 0.54 + 0.46 * np.cos(np.pi * w)}[filt]
    H = H * win * (w <= 1)
    P = sfft.fft(sino, n=N, axis=-1, workers=-1) * H
    q = np.real(sfft.ifft(P, axis=-1, workers=-1))[..., :n]
    return backproject(q, angles_deg) * np.pi / (2 * len(angles_deg))


def sirt(sino, angles_deg, iters=50, nonneg=True, x0=None, monitor=None, subsets=1, relax=1.0):
    """SIRT (subsets = 1) / OS-SART (subsets > 1): x ← x + λ C Aᵀ R (b − A x) with R, C the inverse
    row / column sums of the (sub-)system matrix; non-negativity optional. Returns (x, residuals)."""
    sino = np.asarray(sino, float)
    n = sino.shape[-1]
    A = len(angles_deg)
    M, _ = projector(n, angles_deg)
    batch = sino.shape[:-2]
    b = sino.reshape((-1, A * n)).T                       # (A n, B)
    x = np.zeros((n * n, b.shape[1])) if x0 is None else x0.reshape((-1, n * n)).T.copy()
    groups = []
    for s_ in range(subsets):
        rows = np.concatenate([np.arange(a_ * n, (a_ + 1) * n) for a_ in range(s_, A, subsets)])
        Ms = M[rows]
        Mt = Ms.T.tocsr()
        R = np.asarray(Ms.sum(1)).ravel(); R = np.where(R > 1e-6, 1 / np.maximum(R, 1e-6), 0)
        C = np.asarray(Ms.sum(0)).ravel(); C = np.where(C > 1e-6, 1 / np.maximum(C, 1e-6), 0)
        groups.append((rows, Ms, Mt, R[:, None], C[:, None]))
    hist = []
    nb = np.linalg.norm(b) + 1e-20
    for it in range(iters):
        for rows, Ms, Mt, R, C in groups:
            x = x + relax * C * (Mt @ (R * (b[rows] - Ms @ x)))
            if nonneg:
                np.maximum(x, 0, out=x)
        hist.append(float(np.linalg.norm(b - M @ x) / nb))
        if monitor is not None:
            monitor.step(it + 1, iters, (lambda x=x, h=list(hist): (x.T.reshape(batch + (n, n)), h))
                         if monitor.want_preview() else None)
    return x.T.reshape(batch + (n, n)), np.array(hist)


# ============================================================================ phantoms
def cell_volume(n=64, rng=None, dn_cell=0.015, dn_nuc=0.03, beads=4, dn_bead=0.04):
    """Δn(x, y, z) of a cell (ellipsoid), nucleus, nucleolus and a few beads; axes (z, y, x)."""
    rng = np.random.default_rng(5) if rng is None else rng
    c = (np.arange(n) - n / 2 + 0.5) / n
    Z, Y, X = np.meshgrid(c, c, c, indexing="ij")
    v = np.zeros((n, n, n))
    cell = (X / 0.36) ** 2 + (Y / 0.28) ** 2 + (Z / 0.2) ** 2 < 1
    nuc = ((X - 0.06) / 0.14) ** 2 + ((Y + 0.03) / 0.11) ** 2 + (Z / 0.1) ** 2 < 1
    nucl = ((X - 0.09) / 0.04) ** 2 + ((Y + 0.01) / 0.04) ** 2 + (Z / 0.04) ** 2 < 1
    v[cell] = dn_cell
    v[nuc] = dn_nuc
    v[nucl] = dn_nuc * 1.4
    for _ in range(beads):
        p = rng.uniform(-0.22, 0.22, 3) * np.array([0.6, 1, 1])
        r = rng.uniform(0.025, 0.045)
        v[(X - p[2]) ** 2 + (Y - p[1]) ** 2 + (Z - p[0]) ** 2 < r ** 2] = dn_bead
    return ndi.gaussian_filter(v, 0.6)


# ============================================================================ ODT forward model
def illumination_angles(count, na_ill, n_medium, wl, dx, n, pattern="spiral"):
    """Transverse illumination frequencies (cycles/µm), snapped to the periodic grid."""
    fmax = na_ill / wl
    df = 1.0 / (n * dx)
    if pattern == "ring":
        t = np.linspace(0, TWO_PI, count, endpoint=False)
        f = np.stack([fmax * np.sin(t), fmax * np.cos(t)], 1)
        f = np.vstack([[0, 0], f])
    else:            # Fermat spiral filling the disc
        i = np.arange(count)
        r = fmax * np.sqrt((i + 0.5) / count)
        t = i * np.pi * (3 - np.sqrt(5))
        f = np.stack([r * np.sin(t), r * np.cos(t)], 1)
    return np.rint(f / df) * df


def bpm(dn, wl, dx, n_medium, fin=(0.0, 0.0), na_obj=None):
    """Multislice beam propagation through Δn(z, y, x) (paraxial-free: exact ASM steps in the medium).
    Returns (exit field at the last slice, incident field there)."""
    nz, ny, nx = dn.shape
    y = (np.arange(ny) - ny // 2) * dx; x = (np.arange(nx) - nx // 2) * dx
    Yg, Xg = np.meshgrid(y, x, indexing="ij")
    u = np.exp(2j * np.pi * (fin[0] * Yg + fin[1] * Xg))
    k0 = TWO_PI / wl
    H = asm_kernel((ny, nx), wl, dx, dx, n_medium, bandlimit=False)
    for z in range(nz):
        u = u * np.exp(1j * k0 * dn[z] * dx)
        u = sfft.ifft2(sfft.fft2(u, workers=-1) * H, workers=-1)
    w = np.sqrt(max((n_medium / wl) ** 2 - fin[0] ** 2 - fin[1] ** 2, 0))
    uin = np.exp(2j * np.pi * (fin[0] * Yg + fin[1] * Xg + w * nz * dx))
    if na_obj is not None:
        FY, FX = np.meshgrid(sfft.fftfreq(ny, dx), sfft.fftfreq(nx, dx), indexing="ij")
        P = np.hypot(FX, FY) <= na_obj / wl
        u = sfft.ifft2(sfft.fft2(u) * P)
    return u, uin


def refocus(u, wl, dx, z, n_medium):
    return sfft.ifft2(sfft.fft2(u, workers=-1) * asm_kernel(u.shape, wl, dx, z, n_medium, bandlimit=False), workers=-1)


def simulate_odt(dn, wl, dx, n_medium, fins, na_obj, geometry="illumination scanning", angles_deg=None,
                 noise=0.0, rng=None):
    """Fields at the volume's central plane (z = 0) after back-propagating the exit field."""
    rng = np.random.default_rng(0) if rng is None else rng
    nz = dn.shape[0]
    us, uins = [], []
    if geometry == "illumination scanning":
        for f in fins:
            u, uin = bpm(dn, wl, dx, n_medium, tuple(f), na_obj)
            us.append(refocus(u, wl, dx, -(nz / 2) * dx, n_medium))
            uins.append(refocus(uin, wl, dx, -(nz / 2) * dx, n_medium))
    else:
        for a in angles_deg:
            rot = ndi.rotate(dn, a, axes=(0, 2), reshape=False, order=1)
            u, uin = bpm(rot, wl, dx, n_medium, (0.0, 0.0), na_obj)
            us.append(refocus(u, wl, dx, -(nz / 2) * dx, n_medium))
            uins.append(refocus(uin, wl, dx, -(nz / 2) * dx, n_medium))
    us = np.array(us)
    if noise:
        us = us + noise * (rng.standard_normal(us.shape) + 1j * rng.standard_normal(us.shape)) / np.sqrt(2)
    return us, np.array(uins)


# ============================================================================ ODT inversion
def scattered_field(u, uin, approx="Rytov"):
    """Born: u_s = u − u_in.  Rytov: u_s = u_in · ln(u/u_in) with the phase unwrapped (2-D LS)."""
    if approx == "Born":
        return u - uin
    from .unwrap import least_squares
    r = u / uin
    ph = np.angle(r)
    phu = least_squares(ph)
    phu = phu + np.angle(np.exp(1j * (ph - phu)))       # congruent (exact where no wraps)
    phu -= TWO_PI * np.round(np.median(phu) / TWO_PI)
    return uin * (np.log(np.abs(r) + 1e-12) + 1j * phu)


def odt_reconstruct(us, uins, wl, dx, n_medium, fins=None, angles_deg=None, approx="Rytov", na_obj=None,
                    geometry="illumination scanning", gp_iters=0, nonneg=True, monitor=None):
    """Fourier diffraction theorem: F̃(K − k_in) = −2iγ Ũ_s(α, β; z=0), γ = √(k_m² − α² − β²),
    with f = k₀²(n² − n_m²) the scattering potential. Samples are gridded (nearest neighbour,
    averaged) into the 3-D spectrum; optional Gerchberg–Papoulis iterations enforce Δn ≥ 0 and
    keep the measured samples. Returns (Δn volume (z, y, x), sample-count map)."""
    K, ny, nx = us.shape
    nz = nx
    df = 1.0 / (nx * dx)
    FY, FX = np.meshgrid(sfft.fftfreq(ny, dx), sfft.fftfreq(nx, dx), indexing="ij")
    km = n_medium / wl
    w = np.sqrt(np.clip(km ** 2 - FX ** 2 - FY ** 2, 0, None))
    ok = (km ** 2 - FX ** 2 - FY ** 2) > 0
    if na_obj is not None:
        ok &= np.hypot(FX, FY) <= na_obj / wl * 0.999
    acc = np.zeros((nz, ny, nx), complex)
    cnt = np.zeros((nz, ny, nx))
    k0 = TWO_PI / wl
    gamma = TWO_PI * w
    for i in range(K):
        us_ = scattered_field(us[i], uins[i], approx)
        Us = sfft.fft2(sfft.ifftshift(us_)) * dx * dx       # Ũ_s on the centred grid
        c0 = uins[i][ny // 2, nx // 2]                       # incident phase at the origin
        Fv = -2j * gamma * Us / c0
        if geometry == "illumination scanning":
            fy0, fx0 = fins[i]
            w0 = np.sqrt(km ** 2 - fy0 ** 2 - fx0 ** 2)
            qy, qx, qz = FY - fy0, FX - fx0, w - w0
        else:
            a = np.deg2rad(angles_deg[i])
            qy, qx0, qz0 = FY, FX, w - km
            # undo the sample rotation (rotation about y by −a)
            qx = np.cos(a) * qx0 + np.sin(a) * qz0
            qz = -np.sin(a) * qx0 + np.cos(a) * qz0
        iy = np.rint(qy / df).astype(int) % ny
        ix = np.rint(qx / df).astype(int) % nx
        iz = np.rint(qz / df).astype(int)
        good = ok & (np.abs(iz) < nz // 2)
        np.add.at(acc, (iz[good] % nz, iy[good], ix[good]), Fv[good])
        np.add.at(cnt, (iz[good] % nz, iy[good], ix[good]), 1)
        if monitor is not None:
            monitor.step(i + 1, K + gp_iters)
    meas = cnt > 0
    Fhat = np.where(meas, acc / np.maximum(cnt, 1), 0)
    vol = 1.0 / dx ** 3
    f = sfft.fftshift(sfft.ifftn(Fhat, workers=-1)) * vol
    def to_dn(f):
        n2 = n_medium ** 2 + f / k0 ** 2
        return np.real(np.sqrt(n2.astype(complex))) - n_medium
    dn = to_dn(f)
    for it in range(gp_iters):
        d = np.real(dn)
        if nonneg:
            d = np.maximum(d, 0)
        fpot = k0 ** 2 * ((n_medium + d) ** 2 - n_medium ** 2)
        Fc = sfft.fftn(sfft.ifftshift(fpot), workers=-1) / vol
        Fc[meas] = Fhat[meas]
        f = sfft.fftshift(sfft.ifftn(Fc, workers=-1)) * vol
        dn = to_dn(f)
        if monitor is not None:
            monitor.step(K + it + 1, K + gp_iters)
    if nonneg and gp_iters:
        dn = np.maximum(dn, 0)
    return dn, sfft.fftshift(cnt)


# ----------------------------------------------------------------------------- more 3-D phantoms
def shepp_logan_3d(n=64):
    """3-D Shepp–Logan head phantom (Kak & Slaney ellipsoids, modified contrast), axes (z, y, x), in [0, 1]."""
    E = [(1.0, .69, .92, .81, 0, 0, 0, 0), (-.8, .6624, .874, .78, 0, -.0184, 0, 0),
         (-.2, .11, .31, .22, .22, 0, 0, -18), (-.2, .16, .41, .28, -.22, 0, 0, 18),
         (.1, .21, .25, .41, 0, .35, -.15, 0), (.1, .046, .046, .05, 0, .1, .25, 0),
         (.1, .046, .046, .05, 0, -.1, .25, 0), (.1, .046, .023, .05, -.08, -.605, 0, 0),
         (.1, .023, .023, .02, 0, -.606, 0, 0), (.1, .023, .046, .02, .06, -.605, 0, 0)]
    c = np.linspace(-1, 1, n)
    Z, Y, X = np.meshgrid(c, -c, c, indexing="ij")
    v = np.zeros((n, n, n))
    for A, a, b, cc, x0, y0, z0, phi in E:
        p = np.deg2rad(phi)
        xr = (X - x0) * np.cos(p) + (Y - y0) * np.sin(p)
        yr = -(X - x0) * np.sin(p) + (Y - y0) * np.cos(p)
        v[(xr / a) ** 2 + (yr / b) ** 2 + ((Z - z0) / cc) ** 2 <= 1] += A
    v = np.clip(v, 0, None)
    return v / max(v.max(), 1e-12)


def object_volume(img2d, n=64):
    """A 2-D sample object extruded into a sphere: slice z is the object cropped to the sphere's disc at
    that depth and scaled by a smooth axial profile, so each depth differs. Axes (z, y, x), in [0, 1]."""
    from .fields import resize
    a = np.clip(resize(np.asarray(img2d, float), (n, n)), 0, None)
    a = a / max(a.max(), 1e-12)
    c = (np.arange(n) - n / 2 + 0.5) / (n / 2)
    Z, Y, X = np.meshgrid(c, c, c, indexing="ij")
    ball = (X ** 2 + Y ** 2 + Z ** 2) <= 0.85 ** 2
    v = np.where(ball, a[None], 0.0) * (0.6 + 0.4 * np.cos(np.pi * Z / 2)) + 0.15 * ball
    return v / max(v.max(), 1e-12)


PHANTOMS_3D = ("cell + beads", "3-D Shepp–Logan", "sample object in a sphere")


def phantom_3d(kind, n, rng=None, obj="usaf", **cell_kw):
    from .fields import test_object
    if kind == "3-D Shepp–Logan":
        return shepp_logan_3d(n)
    if kind == "sample object in a sphere":
        return object_volume(test_object(obj, max(n, 128), rng), n)
    v = cell_volume(n, rng, **cell_kw)
    return v / max(v.max(), 1e-12)
