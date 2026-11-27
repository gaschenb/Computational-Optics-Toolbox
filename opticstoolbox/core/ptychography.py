"""Ptychography (ePIE, rPIE, mPIE with probe recovery and annealing position correction) and
Fourier ptychographic microscopy (sequential Gauss–Newton / EPRY with pupil recovery and LED
position correction)."""
from __future__ import annotations

import numpy as np
import scipy.fft as sfft

from .fields import fft2, freq_grid, grid, ifft2, shift_subpixel
from .propagation import angular_spectrum


# ============================================================================ real-space ptychography
def raster_positions(n_obj, n_probe, step, jitter=0.0, rng=None, pattern="raster"):
    """Top-left scan positions (y, x) in pixels. pattern: raster | spiral (Fermat)."""
    rng = np.random.default_rng(0) if rng is None else rng
    span = n_obj - n_probe
    if pattern == "spiral":
        k = int((span / step + 1) ** 2)
        i = np.arange(k)
        r = step * 0.55 * np.sqrt(i)
        t = i * np.pi * (3 - np.sqrt(5))
        pos = np.stack([span / 2 + r * np.sin(t), span / 2 + r * np.cos(t)], 1)
        pos = pos[(pos >= 0).all(1) & (pos <= span).all(1)]
    else:
        g = np.arange(0, span + 1e-9, step)
        Y, X = np.meshgrid(g, g, indexing="ij")
        pos = np.stack([Y.ravel(), X.ravel()], 1).astype(float)
    if jitter:
        pos = np.clip(pos + rng.uniform(-jitter, jitter, pos.shape), 0, span)
    return pos


def make_probe(n, wl, dx, radius_um, defocus_um=0.0, kind="defocused pinhole"):
    """Illumination: pinhole (top hat) propagated by defocus, or a focused beam with a diffuser."""
    X, Y = grid(n, dx)
    if kind == "focused (diffuser)":
        rng = np.random.default_rng(11)
        ph = np.exp(1j * 2 * np.pi * rng.random((n, n)))
        ap = (np.hypot(*freq_grid((n, n), dx)) < 0.5 * radius_um / (wl * 4)).astype(complex)
        P = ifft2(fft2(ph) * ap)
        P *= np.exp(-(X ** 2 + Y ** 2) / (2 * (radius_um) ** 2))
    else:
        P = (np.hypot(X, Y) < radius_um).astype(complex)
        if defocus_um:
            P = angular_spectrum(P, wl, dx, defocus_um, pad=True)
    return P / np.sqrt(np.mean(np.abs(P) ** 2))


def simulate_ptycho(obj, probe, positions, photons=None, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    n = probe.shape[0]
    pats = []
    for y, x in np.rint(positions).astype(int):
        ex = probe * obj[y:y + n, x:x + n]
        pats.append(np.abs(fft2(ex)) ** 2 / (n * n))
    I = np.array(pats)
    if photons:
        k = photons / I.sum(axis=(1, 2)).mean()
        I = rng.poisson(I * k) / k
    return I


PTYCHO_METHODS = ("ePIE", "rPIE", "mPIE")


def ptycho(I, positions, n_obj, probe0, method="rPIE", iters=100, alpha=None, beta=None,
           probe_update=True, probe_start=2, position_correction=False, pc_start=10, pc_amp=0.5, pc_gain=0.5,
           eta=0.7, T=3, rng=None, monitor=None, obj0=None):
    """ePIE (Maiden & Rodenburg 2009), rPIE and mPIE (Maiden, Johnson & Li, Optica 2017).

    ePIE  O ← O + α P*(ψ'−ψ)/max|P|²
    rPIE  O ← O + P*(ψ'−ψ)/((1−α)|P|² + α max|P|²)            (α ≈ 0.1: much more robust)
    mPIE  rPIE + Nesterov-type momentum on O and P every T sweeps (defaults T = 3, η = 0.7)
    Position correction: sub-pixel Gauss–Newton update of every scan position from the exit-wave
    residual (gradient-based, cf. Dwivedi et al. 2018), with sub-pixel probe shifts and the scan
    centroid held fixed (removes the global-translation ambiguity).
    Returns (object, probe, positions, error history)."""
    rng = np.random.default_rng(0) if rng is None else rng
    n = probe0.shape[0]
    J = len(I)
    a_def = {"ePIE": 1.0, "rPIE": 0.15, "mPIE": 0.1}[method]
    b_def = {"ePIE": 1.0, "rPIE": 1.0, "mPIE": 1.0}[method]
    alpha = a_def if alpha is None else alpha
    beta = b_def if beta is None else beta
    O = np.ones((n_obj, n_obj), complex) if obj0 is None else obj0.copy()
    P = probe0.astype(complex).copy()
    pos = np.array(positions, float).copy()
    pos0 = pos.copy()
    span = n_obj - n
    amp = np.sqrt(np.clip(I, 0, None) * n * n)
    vO = np.zeros_like(O); vP = np.zeros_like(P)
    O_old, P_old = O.copy(), P.copy()
    err = []
    for it in range(iters):
        order = rng.permutation(J)
        e = 0.0
        corr = position_correction and it >= pc_start
        for j in order:
            if corr:
                yi, xi = np.floor(pos[j]).astype(int)
                fy, fx = pos[j] - (yi, xi)
                y, x = min(max(yi, 0), span), min(max(xi, 0), span)
                Pj = shift_subpixel(P, fy, fx) if (fy or fx) else P
            else:
                y, x = np.rint(pos[j]).astype(int)
                Pj = P
            Oj = O[y:y + n, x:x + n]
            psi = Pj * Oj
            Psi = fft2(psi)
            aP = np.abs(Psi)
            e += np.sum((aP - amp[j]) ** 2)
            psi2 = ifft2(amp[j] * Psi / np.maximum(aP, 1e-12))
            d = psi2 - psi
            if corr:
                # Gauss–Newton step on the position: d ≈ P ∂O · Δr  →  Δr = Re⟨P∂O, d⟩ / ‖P∂O‖²
                gy, gx = np.gradient(Oj)
                step_ = []
                for g in (gy, gx):
                    Pg = Pj * g
                    step_.append(np.real(np.vdot(Pg, d)) / (np.vdot(Pg, Pg).real + 1e-12))
                pos[j] = np.clip(pos[j] + np.clip(pc_gain * np.array(step_), -pc_amp, pc_amp), 0, span)
            P2 = np.abs(Pj) ** 2
            if method == "ePIE":
                Oj_new = Oj + alpha * np.conj(Pj) * d / P2.max()
            else:
                Oj_new = Oj + np.conj(Pj) * d / ((1 - alpha) * P2 + alpha * P2.max())
            if probe_update and it >= probe_start:
                O2 = np.abs(Oj) ** 2
                if method == "ePIE":
                    dP = beta * np.conj(Oj) * d / O2.max()
                else:
                    dP = np.conj(Oj) * d / ((1 - beta) * O2 + beta * O2.max())
                P = P + (shift_subpixel(dP, -fy, -fx) if corr and (fy or fx) else dP)
            O[y:y + n, x:x + n] = Oj_new
        if corr:                                   # no global drift: keep the scan centroid fixed
            pos = np.clip(pos - (pos - pos0).mean(0), 0, span)
        if method == "mPIE" and (it + 1) % T == 0:
            vO = eta * vO + (O - O_old); O = O_old + vO + (O - O_old) * 0  # O ← O + η v
            O = O_old + vO
            if probe_update and it >= probe_start:
                vP = eta * vP + (P - P_old); P = P_old + vP
            O_old, P_old = O.copy(), P.copy()
        err.append(float(np.sqrt(e / np.sum(amp ** 2))))
        if monitor is not None:
            monitor.step(it + 1, iters, (lambda O=O, P=P, e=list(err), p=pos.copy(): (O, P, e, p))
                         if monitor.want_preview() else None)
    return O, P, pos, np.array(err)


def scanned_mask(positions, n_obj, n_probe, probe=None, frac=0.1):
    """Pixels illuminated with at least `frac` of the peak cumulative probe intensity."""
    acc = np.zeros((n_obj, n_obj))
    w = np.ones((n_probe, n_probe)) if probe is None else np.abs(probe) ** 2
    for y, x in np.rint(positions).astype(int):
        acc[y:y + n_probe, x:x + n_probe] += w
    return acc > frac * acc.max()


# ============================================================================ Fourier ptychography
def led_wavevectors(n_side, pitch_mm, height_mm, wl, shift_mm=(0.0, 0.0), rot_deg=0.0):
    """Illumination spatial frequencies (cycles/µm) of a square LED grid; returns (k, NA_ill, ij)."""
    c = (np.arange(n_side) - (n_side - 1) / 2) * pitch_mm
    Yl, Xl = np.meshgrid(c, c, indexing="ij")
    ij = np.stack(np.meshgrid(np.arange(n_side), np.arange(n_side), indexing="ij"), -1).reshape(-1, 2)
    th = np.deg2rad(rot_deg)
    X = Xl * np.cos(th) - Yl * np.sin(th) + shift_mm[1]
    Y = Xl * np.sin(th) + Yl * np.cos(th) + shift_mm[0]
    R = np.sqrt(X ** 2 + Y ** 2 + height_mm ** 2)
    kx, ky = X / R / wl, Y / R / wl
    k = np.stack([ky.ravel(), kx.ravel()], 1)
    return k, np.hypot(kx, ky).ravel() * wl, ij


def fpm_pupil(n_lr, dx_lr, wl, na, coeffs=None):
    from .waveoptics import pupil
    return sfft.fftshift(pupil((n_lr, n_lr), wl, dx_lr, na, coeffs))


def simulate_fpm(obj_hr, dx_hr, n_lr, wl, na, k, pupil_c, photons=None, rng=None):
    """Low-resolution intensities I_ℓ = |F⁻¹{P · Ô(· − k_ℓ)}|² (spectrum cropped to the camera grid)."""
    rng = np.random.default_rng(0) if rng is None else rng
    N = obj_hr.shape[0]
    O = sfft.fftshift(fft2(obj_hr))
    df = 1.0 / (N * dx_hr)
    c = N // 2
    h = n_lr // 2
    out = []
    for ky, kx in k:
        cy, cx = c + int(round(ky / df)), c + int(round(kx / df))
        sub = O[cy - h:cy + h, cx - h:cx + h]
        if sub.shape != (n_lr, n_lr):
            out.append(np.zeros((n_lr, n_lr)))
            continue
        u = ifft2(sfft.ifftshift(sub * pupil_c)) * (n_lr / N) ** 2
        out.append(np.abs(u) ** 2)
    I = np.array(out)
    if photons:
        k_ = photons / max(I[np.argmin(np.hypot(*k.T))].mean(), 1e-12)
        I = rng.poisson(I * k_) / k_
    return I


def fpm(I, k, N, dx_hr, wl, na, iters=20, pupil_update=True, method="Gauss–Newton (EPRY)",
        led_correction=False, lc_start=2, alpha=1.0, beta=1.0, delta1=1.0, delta2=1e3, monitor=None):
    """Sequential FPM recovery, LEDs processed from the centre outwards.

    Gauss–Newton / EPRY (Tian 2014; Ou 2014): spectrum and pupil updated jointly,
        Ô ← Ô + |P| P* Δ / (|P|max (|P|² + δ₁)),   P ← P + |Ô| Ô* Δ / (|Ô|max (|Ô|² + δ₂));
    'alternating projections' is the original Zheng 2013 update (no pupil recovery).
    LED correction: each LED's spectral offset is refined by ±1-pixel search (simulated-annealing
    style, Sun et al. 2016). Returns (object, pupil, offsets used, error history)."""
    n_lr = I.shape[-1]
    df = 1.0 / (N * dx_hr)
    c, h = N // 2, n_lr // 2
    dx_lr = dx_hr * N / n_lr
    P0 = fpm_pupil(n_lr, dx_lr, wl, na)
    support = np.abs(P0) > 0
    P = P0.astype(complex).copy()
    order = np.argsort(np.hypot(*k.T))
    cen = [(c + int(round(ky / df)), c + int(round(kx / df))) for ky, kx in k]
    cen = np.array(cen)
    i0 = order[0]
    up = np.sqrt(np.clip(I[i0], 0, None))
    from scipy import ndimage as ndi
    o = ndi.zoom(up, N / n_lr, order=1).astype(complex)
    O = sfft.fftshift(fft2(o))
    scale = (n_lr / N) ** 2
    err = []
    amp = np.sqrt(np.clip(I, 0, None))
    use_gn = method.startswith("Gauss")
    for it in range(iters):
        e = 0.0
        for l in order:
            cy, cx = cen[l]
            if led_correction and it >= lc_start:
                best = None
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        yy, xx = cy + dy, cx + dx
                        sub = O[yy - h:yy + h, xx - h:xx + h]
                        if sub.shape != (n_lr, n_lr):
                            continue
                        u = ifft2(sfft.ifftshift(sub * P)) * scale
                        ee = np.sum((np.abs(u) - amp[l]) ** 2)
                        if best is None or ee < best[0]:
                            best = (ee, yy, xx)
                _, cy, cx = best
                cen[l] = (cy, cx)
            sl = (slice(cy - h, cy + h), slice(cx - h, cx + h))
            sub = O[sl]
            if sub.shape != (n_lr, n_lr):
                continue
            psi_hat = sub * P
            u = ifft2(sfft.ifftshift(psi_hat)) * scale
            au = np.abs(u)
            e += np.sum((au - amp[l]) ** 2)
            u2 = amp[l] * u / np.maximum(au, 1e-12)
            Psi2 = sfft.fftshift(fft2(u2)) / scale
            D = Psi2 - psi_hat
            if use_gn:
                Pa = np.abs(P)
                O[sl] = sub + alpha * Pa * np.conj(P) * D / (Pa.max() * (Pa ** 2 + delta1))
                if pupil_update and it >= 1:
                    Sa = np.abs(sub)
                    P = P + beta * Sa * np.conj(sub) * D / (Sa.max() * (Sa ** 2 + delta2)) * support
            else:
                O[sl] = sub + np.conj(P) * D / (np.abs(P) ** 2).max()
        err.append(float(np.sqrt(e / np.sum(amp ** 2))))
        if monitor is not None:
            monitor.step(it + 1, iters, (lambda O=O, P=P, e=list(err): (ifft2(sfft.ifftshift(O)), P, e))
                         if monitor.want_preview() else None)
    obj = ifft2(sfft.ifftshift(O))
    return obj, P, cen, np.array(err)
