"""Small helpers shared by the experiment definitions."""
from __future__ import annotations

import numpy as np

from ..core import fields as F
from ..core.registry import Curve, Experiment, Input, Panel, Param, Result, register  # noqa: F401

# ----------------------------------------------------------------------------- common parameters
WL = lambda d=0.532, lo=0.2, hi=2.0: Param("wl", "wavelength", d, lo, hi, unit="µm", tip="Vacuum wavelength (µm)")
DX = lambda d=0.2, lo=0.02, hi=20.0: Param("dx", "pixel pitch", d, lo, hi, log=True, unit="µm",
                                          tip="Sample-plane pixel pitch (camera pixel / magnification)")
N = lambda d=256, lo=64, hi=1024: Param("n", "grid size", d, lo, hi, kind="int", tip="Pixels per side")
NOISE = lambda d=1e4, lo=10, hi=1e7: Param("photons", "photons / pixel", d, lo, hi, log=True,
                                           tip="Mean detected photons per pixel (Poisson noise); large = noiseless")
SEED = Param("seed", "random seed", 0, 0, 9999, kind="int")
ITERS = lambda d=100, lo=1, hi=5000: Param("iters", "iterations", d, lo, hi, kind="int")


def method(choices, default=None, label="method"):
    choices = tuple(choices)
    return Param("method", label, default or choices[0], choices=choices)


def choice(key, label, choices, default=None, tip=""):
    choices = tuple(choices)
    return Param(key, label, default or choices[0], choices=choices, tip=tip)


def flag(key, label, default=True, tip=""):
    return Param(key, label, bool(default), kind="bool", tip=tip)


# ----------------------------------------------------------------------------- panels
def amp(x, title="amplitude", **kw):
    return Panel(np.abs(x), "amplitude", title, **kw)


def phase(x, title="phase", wrapped=True, **kw):
    a = np.angle(x) if np.iscomplexobj(x) else np.asarray(x, float)
    return Panel(a, "phase" if wrapped else "signed", title, **kw)


def inten(x, title="intensity", **kw):
    return Panel(np.asarray(x, float), "intensity", title, **kw)


def signed(x, title, **kw):
    return Panel(np.asarray(x, float), "signed", title, **kw)


def depth(x, title="depth", **kw):
    return Panel(np.asarray(x, float), "depth", title, **kw)


def logspec(x, title="spectrum (log)"):
    S = np.abs(np.fft.fftshift(np.fft.fft2(x))) if not np.iscomplexobj(x) or x.ndim == 2 else x
    return Panel(np.log1p(S), "intensity", title)


def err(x, title="error", **kw):
    return Panel(np.asarray(x, float), "signed", title, **kw)


def extent(n, dx):
    h = n * dx / 2
    return (-h, h, -h, h)


def conv(title, ys, xlabel="iteration", ylabel="error", logy=True, x=None):
    """Curve from {name: y-array}."""
    series = {}
    for k, y in ys.items():
        y = np.asarray(y, float)
        if y.size == 0:
            continue
        series[k] = (np.arange(1, len(y) + 1) if x is None else np.asarray(x), y)
    return Curve(title, series, xlabel, ylabel, logy=logy)


def profile_curve(title, images, row=None, xlabel="x (px)", ylabel=""):
    series = {}
    for k, im in images.items():
        im = np.asarray(im, float)
        r = im.shape[0] // 2 if row is None else row
        series[k] = (np.arange(im.shape[1]), im[r])
    return Curve(title, series, xlabel, ylabel)


def rng_from(p):
    return np.random.default_rng(int(p.get("seed", 0)))


def stack_input(arr, k=None):
    a = np.asarray(arr, float)
    if a.ndim == 2:
        a = a[None]
    return a if k is None else a[:k]


def table_values(t):
    return np.asarray(t, float).ravel()


def bandlimit(t, dx, wl, na):
    FX, FY = F.freq_grid(t.shape, dx)
    return F.ifft2(F.fft2(t) * (np.hypot(FX, FY) <= na / wl))


# ----------------------------------------------------------------------------- comparison with ground truth
def compare(res, est, truth, prefix="", mask=None, error_panel=True, title=None, fit=False, metric_unit="", **panel_kw):
    """Standard image-quality metrics against the ground truth, added to `res` without overwriting:
    RMSE, PSNR (peak = range of the truth) and SSIM; plus a signed error map if the result has none.
    `fit` first matches the estimate's scale and offset to the truth (least squares), for quantities
    known only up to an affine factor. `metric_unit` is appended to the RMSE name, e.g. " (rad)"."""
    est = np.real(np.asarray(est, float) if not np.iscomplexobj(est) else est)
    truth = np.asarray(truth, float)
    if fit:
        est = F.fit_scale(est, truth)
    e = est - truth
    m = np.isfinite(e) if mask is None else (np.asarray(mask, bool) & np.isfinite(e))
    pre = f"{prefix} " if prefix else ""
    rmse = float(np.sqrt(np.mean(e[m] ** 2))) if m.any() else float("nan")
    rng = float(np.nanmax(truth[m]) - np.nanmin(truth[m])) if m.any() else 1.0
    have = lambda tag: any(tag in k and k.startswith(pre) for k in res.metrics)    # keep an existing, more specific one
    if not have("RMSE"):
        res.metrics[f"{pre}RMSE{metric_unit}"] = rmse
    if not have("PSNR"):
        res.metrics[f"{pre}PSNR (dB)"] = float(10 * np.log10(rng ** 2 / max(rmse ** 2, 1e-30)))
    if not have("SSIM"):
        if est.ndim == 2 and est.shape == truth.shape:
            a, b = np.where(m, est, 0.0), np.where(m, truth, 0.0)
            res.metrics[f"{pre}SSIM"] = F.ssim(a, b)
        elif est.ndim == 3:                       # 3-D: mean over the slices that contain the object
            amp_ = np.nanmax(np.abs(truth))
            keep = [k for k in range(len(truth)) if np.ptp(truth[k]) > 0.05 * amp_] or list(range(len(truth)))
            res.metrics[f"{pre}SSIM (mean over object slices)"] = float(np.mean(
                [F.ssim(np.where(np.isfinite(est[k]), est[k], 0), truth[k]) for k in keep]))
    if error_panel and not any("error" in k.lower() for k in res.panels):
        em = np.where(m, e, np.nan)
        if em.ndim == 3:
            for pn in res.panels.values():          # same geometry as the volume being compared
                if pn.kind == "volume" and np.shape(pn.data) == em.shape:
                    for f_ in ("extent", "unit", "zextent", "zlabel", "xlabel", "ylabel"):
                        if getattr(pn, f_) and f_ not in panel_kw:
                            panel_kw[f_] = getattr(pn, f_)
                    break
            res.panels[f"{pre}error".strip()] = Panel(em, "volume", title or f"{pre}error (estimate − truth)".strip(),
                                                      cmap="signed", **panel_kw)
        else:
            res.panels[f"{pre}error".strip()] = err(em, title or f"{pre}error (estimate − truth)".strip(), **panel_kw)
    return res


def _align_phase_panels(res, tr, m):
    """The global phase of a retrieved field is arbitrary. So that the displayed estimate and the truth
    share a colour scale, remove the constant offset from every estimated phase panel of the truth's shape
    (wrapped panels stay wrapped; unwrapped ones are shifted by the mean difference)."""
    for k, pn in res.panels.items():
        low = (k + " " + pn.title).lower()
        if pn.kind not in ("phase", "signed") or any(w in low for w in ("true", "truth", "error", "pupil", "dpc", "transfer", "spectrum", "λ")):
            continue
        a = np.asarray(pn.data)
        if a.shape != tr.shape or np.iscomplexobj(a) or not np.isfinite(a[m]).any():
            continue
        z = np.exp(1j * (a - tr))[m & np.isfinite(a)]
        if np.abs(np.mean(z)) < 0.3:          # not an estimate of the same phase
            continue
        if pn.kind == "phase" and np.nanmax(a) - np.nanmin(a) <= 2 * np.pi + 1e-6:
            pn.data = F.wrap(a - np.angle(np.mean(z)))
        else:
            pn.data = a - float(np.nanmean((a - tr)[m]))


def compare_phase(res, u, truth, mask=None, error_panel=True, title="phase error (rad)", **panel_kw):
    """Phase comparison: the wrapped difference minus its mean over `mask` (the global phase is
    arbitrary), scored with `compare` in radians."""
    tr = np.angle(truth) if np.iscomplexobj(truth) else np.asarray(truth, float)
    pu = np.angle(u) if np.iscomplexobj(u) else np.asarray(u, float)
    e = F.wrap(pu - tr)
    m = np.ones(e.shape, bool) if mask is None else np.asarray(mask, bool)
    c0 = np.angle(np.mean(np.exp(1j * e[m])))
    e = F.wrap(e - c0)
    _align_phase_panels(res, tr, m)
    return compare(res, tr + e, tr, "phase", mask=m, error_panel=error_panel, title=title,
                   metric_unit=" (rad)", **panel_kw)


def freq_extent(n, dx):
    """Extent of a centred (fftshift-ed) spectrum in cycles per unit of dx."""
    f = 1.0 / (2 * dx)
    return (-f, f - 2 * f / n, -f, f - 2 * f / n)
