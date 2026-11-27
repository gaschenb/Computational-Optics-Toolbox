"""Photon counting & quantum imaging: photon-limited (Poisson) imaging, first-photon imaging,
single-photon lidar, biphoton (SPDC) quantum ghost imaging."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from ..core import deconv as D
from ..core import depth as DP
from ..core import fields as F
from ..core import photon as PH
from ._common import (ITERS, N, SEED, Curve, compare, Experiment, Input, Panel, Param, Result, choice, conv, depth, err, flag,
                      inten, method, profile_curve, register, table_values)

AREA = "Photon counting & quantum"


def _obj(name, n, rng, floor=0.0):
    return floor + (1 - floor) * F.normalize(np.clip(F.test_object(name, n, rng), 0, None))


# ----------------------------------------------------------------------------- Poisson imaging
def _sim_poisson(p, rng):
    n = p["n"]
    x = _obj(p["object"], n, rng, 0.05)
    H = D.psf_otf(D.gaussian_psf(n, p["blur"]), (n, n)) if p["blur"] > 0 else None
    mean = p["flux"] * (np.real(np.fft.ifft2(np.fft.fft2(x) * H)) if H is not None else x)
    y = rng.poisson(np.clip(mean, 0, None) + p["dark"]).astype(float)
    return {"counts": y, "blur": p["blur"], "dark": p["dark"], "truth": x * p["flux"]}


def _solve_poisson(d, p, mon):
    y = d["counts"]
    n = y.shape
    H = D.psf_otf(D.gaussian_psf(n[0], d["blur"]), n) if d.get("blur", 0) > 0 and n[0] == n[1] else None
    m = p["method"]
    if m.startswith("Poisson TV"):
        x, hist = PH.poisson_tv(y, H, p["lam"], p["iters"], d.get("dark", 0.0), mon)
    else:
        x = PH.photon_limited(y, m, p["lam"] / 8 if m.startswith("Anscombe") else p["lam"], p["sigma"], p["iters"], H, mon)
        hist = None
    res = Result()
    res.panels["counts"] = inten(y, f"photon counts (mean {y.mean():.2f} / px)")
    res.panels["estimate"] = inten(x, m)
    if "truth" in d:
        t = d["truth"]
        res.panels["truth"] = inten(t, "true photon rate")
        res.metrics["PSNR raw (dB)"] = F.psnr(F.fit_scale(y, t), t)
        res.metrics["PSNR (dB)"] = F.psnr(F.fit_scale(x, t), t)
        res.metrics["SSIM"] = F.ssim(F.fit_scale(x, t), t)
        compare(res, x, t, fit=True)
        res.curves.append(profile_curve("centre row", {"truth": t, "counts": y, "estimate": x}))
    if hist is not None:
        res.curves.append(conv("relative update", {m: hist}))
    hy = np.bincount(y.astype(int).ravel())[:30]
    res.curves.append(Curve("count histogram", {"pixels": (np.arange(len(hy)), hy + 1e-1)}, "photons", "pixels", logy=True))
    return res


register(Experiment(
    key="photon_limited", title="Photon-limited imaging (Poisson)", area=AREA,
    blurb="Images with about one photon per pixel are dominated by shot noise. Compare Gaussian smoothing, "
          "variance-stabilised (Anscombe) TV denoising and a Poisson-likelihood TV model solved by a "
          "primal–dual (Chambolle–Pock) method, optionally with deblurring.",
    equations=[r"y\sim\mathrm{Poisson}(Ax+b)",
               r"\min_{x\geq 0}\ \sum_i\left[(Ax+b)_i-y_i\log(Ax+b)_i\right]+\lambda\,\mathrm{TV}(x)",
               r"\mathcal{A}(y)=2\sqrt{y+3/8}\ \Rightarrow\ \mathrm{Var}\approx 1"],
    scene=[N(128, 32, 512), choice("object", "object", F.objects("cells")), Param("flux", "photons / px (peak)", 2.0, 0.05, 1000.0, log=True),
           Param("blur", "blur σ (px, 0 = none)", 0.0, 0.0, 5.0), Param("dark", "dark counts / px", 0.0, 0.0, 2.0), SEED],
    solver=[method(PH.POISSON_METHODS, "Poisson TV (primal–dual)"), Param("lam", "regularisation λ", 2.0, 1e-3, 20.0, log=True),
            Param("sigma", "smoothing σ (px)", 1.5, 0.3, 6.0), ITERS(200, 10, 2000)],
    simulate=_sim_poisson, solve=_solve_poisson,
    inputs=[Input("counts", "photon-count image")], meta=("blur", "dark"),
    load=lambda a, m: {"counts": np.asarray(a["counts"], float), "blur": m["blur"], "dark": m["dark"]},
    example="Cells at 2 photons per pixel"))


# ----------------------------------------------------------------------------- first-photon imaging
def _scene_depth(p, rng, n):
    d = DP.depth_scene(n, p["scene"], rng)
    return 1e6 * (p["z_near"] + (p["z_far"] - p["z_near"]) * d)          # µm


def _sim_fpi(p, rng):
    n = p["n"]
    refl = _obj(p["object"], n, rng, 0.05)
    z = _scene_depth(p, rng, n)
    npulse, t, sig = PH.simulate_first_photon(refl, z, p["eta"], p["bg"], p["jitter"], p["gate"], rng=rng)
    return {"pulses": npulse, "arrival": t, "eta": p["eta"], "bg": p["bg"], "jitter": p["jitter"], "gate": p["gate"],
            "truth_refl": refl, "truth_depth": z, "signal_fraction": float(sig.mean())}


def _solve_fpi(d, p, mon):
    n, t = d["pulses"], d["arrival"]
    eta, bg = d["eta"], d["bg"]
    a_ml = PH.fpi_ml_reflectivity(n, eta, bg)
    z_raw = t * PH.C_UM_PER_PS / 2
    if p["method"].startswith("pixel"):
        a, z = a_ml, z_raw
    else:
        a = PH.fpi_reflectivity(n, eta, bg, p["lam_r"])
        mon.step(1, 2)
        tc, keep = PH.censor_and_regularise(t, a, eta, bg, d["gate"], d["jitter"], p["lam_d"])
        z = tc * PH.C_UM_PER_PS / 2
    mon.step(2, 2)
    res = Result()
    res.panels["pulses to first detection"] = inten(np.log10(n), "log₁₀ pulses until first photon")
    res.panels["reflectivity"] = inten(a, f"reflectivity ({p['method'].split(' (')[0]})")
    res.panels["depth"] = depth(z / 1e6, "depth (m)")
    res.panels["raw depth"] = depth(z_raw / 1e6, "raw first-photon depth (m)")
    res.metrics["mean pulses / pixel"] = float(np.mean(n))
    if "truth_refl" in d:
        tr, tz = d["truth_refl"], d["truth_depth"]
        res.panels["true reflectivity"] = inten(tr, "true reflectivity")
        res.panels["true depth"] = depth(tz / 1e6, "true depth (m)")
        res.metrics["signal-photon fraction"] = d["signal_fraction"]
        res.metrics["reflectivity PSNR (dB)"] = F.psnr(F.fit_scale(a, tr), tr)
        res.metrics["depth RMSE (cm)"] = float(np.sqrt(np.mean((z - tz) ** 2)) / 1e4)
        res.metrics["depth median error (cm)"] = float(np.median(np.abs(z - tz)) / 1e4)
        lv = (float(tz.min() / 1e6), float(tz.max() / 1e6))
        for k in ("depth", "raw depth", "true depth"):
            res.panels[k].levels = lv
        res.panels["depth error"] = err((z - tz) / 1e4, "depth error (cm)")
        compare(res, a, tr, prefix="reflectivity", fit=True, error_panel=False)
        compare(res, z / 1e6, tz / 1e6, prefix="depth", error_panel=False)
    return res


FPI_SCENE = [N(128, 32, 256), choice("object", "reflectivity", F.objects("cells")), choice("scene", "depth scene", ("ramp + bumps", "steps", "sphere", "waves")),
             Param("z_near", "nearest depth", 1.0, 0.1, 10.0, unit="m"), Param("z_far", "farthest depth", 1.6, 0.2, 10.0, unit="m")]

register(Experiment(
    key="first_photon", title="First-photon imaging", area=AREA,
    blurb="Reflectivity and depth from a single detected photon per pixel (Kirmani et al. 2014): the "
          "number of laser pulses before the first detection encodes reflectivity, its arrival time "
          "encodes depth. Spatial correlations (log-domain TV) and censoring of background arrivals "
          "turn the noisy per-pixel estimates into usable images.",
    equations=[r"P(n)=e^{-(\eta a+B)(n-1)}\left(1-e^{-(\eta a+B)}\right)",
               r"\hat{a}_{\mathrm{ML}}=\frac{1}{\eta}\left[-\log\left(1-\frac{1}{n}\right)-B\right]",
               r"P(\mathrm{signal})=\frac{\eta a}{\eta a+B},\qquad z=\frac{c\,t}{2}"],
    scene=FPI_SCENE + [Param("eta", "detection efficiency η", 0.05, 0.001, 1.0, log=True),
                       Param("bg", "background / pulse B", 0.002, 0.0, 0.1), Param("jitter", "timing jitter", 60.0, 5.0, 500.0, unit="ps"),
                       Param("gate", "gate width", 20000.0, 2000.0, 100000.0, unit="ps"), SEED],
    solver=[method(PH.FIRST_PHOTON_METHODS, PH.FIRST_PHOTON_METHODS[1]), Param("lam_r", "reflectivity TV λ", 1.5, 0.01, 10.0, log=True),
            Param("lam_d", "depth TV λ", 0.3, 0.01, 10.0, log=True)],
    simulate=_sim_fpi, solve=_solve_fpi,
    inputs=[Input("pulses", "pulses until first detection (per pixel)"), Input("arrival", "first-photon arrival time (ps)")],
    meta=("eta", "bg", "jitter", "gate"),
    load=lambda a, m: {"pulses": np.asarray(a["pulses"], float), "arrival": np.asarray(a["arrival"], float),
                       "eta": m["eta"], "bg": m["bg"], "jitter": m["jitter"], "gate": m["gate"]},
    example="One photon per pixel, 25 % background detections"))


# ----------------------------------------------------------------------------- SPAD lidar
def _sim_lidar(p, rng):
    n = p["n"]
    refl = _obj(p["object"], n, rng, 0.1)
    z = _scene_depth(p, rng, n)
    gate_m = p["bins"] * p["bin_ps"] * PH.C_UM_PER_PS / 2 / 1e6
    if p["z_far"] > gate_m:
        raise ValueError(f"farthest depth {p['z_far']:.2f} m is beyond the {gate_m:.2f} m timing gate")
    H, tb = PH.simulate_lidar(refl, z, p["signal"], p["background"], p["bins"], p["bin_ps"], p["pulse"],
                              int(p["cycles"]) or None, rng)
    return {"histograms": H, "bin_ps": p["bin_ps"], "pulse": p["pulse"], "cycles": int(p["cycles"]),
            "truth_refl": refl, "truth_depth": z}


def _solve_lidar(d, p, mon):
    H = d["histograms"]
    bins = H.shape[-1]
    tb = (np.arange(bins) + 0.5) * d["bin_ps"]
    Hc = PH.coates(H, d["cycles"]) if (p["coates"] and d.get("cycles")) else H
    z, inten_ = PH.lidar_depth(Hc, tb, d["pulse"], p["method"], p["lam"])
    mon.step(1, 1)
    res = Result()
    res.panels["photon counts"] = inten(H.sum(-1), "total detections / pixel")
    res.panels["intensity"] = inten(inten_, "signal intensity estimate")
    res.panels["depth"] = depth(z / 1e6, f"depth (m), {p['method']}")
    c = H.shape[0] // 2
    ser = {"recorded": (tb, H[c, c])}
    if Hc is not H:
        ser["Coates-corrected"] = (tb, Hc[c, c])
    res.curves.append(Curve("histogram, centre pixel", ser, "time (ps)", "counts", logy=False))
    res.metrics["mean detections / pixel"] = float(H.sum(-1).mean())
    if "truth_depth" in d:
        tz = d["truth_depth"]
        res.panels["true depth"] = depth(tz / 1e6, "true depth (m)")
        res.panels["depth error"] = err((z - tz) / 1e4, "depth error (cm)")
        res.panels["true depth"].levels = res.panels["depth"].levels = (float(tz.min() / 1e6), float(tz.max() / 1e6))
        e = np.abs(z - tz)
        res.metrics["depth RMSE (cm)"] = float(np.sqrt(np.mean(e ** 2)) / 1e4)
        res.metrics["median error (cm)"] = float(np.median(e) / 1e4)
        res.metrics["outliers (> 3 pulse widths) %"] = float(100 * np.mean(e > 3 * d["pulse"] * PH.C_UM_PER_PS / 2))
        compare(res, z / 1e6, tz / 1e6, prefix="depth", error_panel=False)
    return res


register(Experiment(
    key="spad_lidar", title="Single-photon lidar (SPAD / TCSPC)", area=AREA,
    blurb="Time-correlated single-photon counting histograms with a few signal photons per pixel in strong "
          "ambient background. Depth by histogram peak, Poisson log-matched filtering, or censoring + TV. "
          "At high flux the SPAD records only the first photon per laser cycle (pile-up); the Coates "
          "correction undoes the resulting bias towards early bins.",
    equations=[r"h_t\sim\mathrm{Poisson}\left(S\,a\,g(t-2z/c)+b\right)",
               r"\hat{\tau}=\arg\max_\tau\ \sum_t h_t\log\left(S\,g(t-\tau)+b\right)",
               r"\hat{\lambda}_t=-N\log\left(1-\frac{h_t}{N-\sum_{s<t}h_s}\right)\quad\mathrm{(Coates)}"],
    scene=[N(64, 16, 256), choice("object", "reflectivity", F.objects("texture")),
           choice("scene", "depth scene", ("ramp + bumps", "steps", "sphere", "waves")),
           Param("z_near", "nearest depth", 0.5, 0.05, 10.0, unit="m"), Param("z_far", "farthest depth", 1.3, 0.1, 10.0, unit="m"),
           Param("signal", "signal photons / px", 10.0, 0.1, 1000.0, log=True), Param("background", "background photons / px", 10.0, 0.0, 5000.0),
           Param("bins", "time bins", 256, 32, 2048, kind="int"), Param("bin_ps", "bin width", 40.0, 2.0, 500.0, unit="ps"),
           Param("pulse", "pulse width σ", 80.0, 5.0, 1000.0, unit="ps"), Param("cycles", "laser cycles (0 = low-flux model)", 0, 0, 5000, kind="int"),
           SEED],
    solver=[method(PH.LIDAR_METHODS, PH.LIDAR_METHODS[2]), flag("coates", "Coates pile-up correction", True),
            Param("lam", "TV weight", 0.5, 0.01, 10.0, log=True)],
    simulate=_sim_lidar, solve=_solve_lidar,
    inputs=[Input("histograms", "histogram cube (y, x, bins)", "stack")], meta=("bin_ps", "pulse", "cycles"),
    load=lambda a, m: {"histograms": (lambda h: h if h.ndim == 3 else h[None])(np.asarray(a["histograms"], float)),
                       "bin_ps": m["bin_ps"], "pulse": m["pulse"], "cycles": int(m["cycles"])},
    example="10 signal and 10 background photons per pixel"))


# ----------------------------------------------------------------------------- biphoton ghost imaging
def _sim_biphoton(p, rng):
    n = p["n"]
    obj = _obj(p["object"], n, rng, 0.0)
    out = PH.simulate_biphoton(obj, int(p["frames"]), p["mu"], p["sigma_corr"], p["eta_s"], p["eta_i"],
                               p["dark_bucket"], p["dark_cam"], None, rng)
    out["truth"] = obj
    out["sigma_corr"] = p["sigma_corr"]
    return out


def _solve_biphoton(d, p, mon):
    m = p["method"]
    if m == "coincidence image":
        x = d["coincidences"]
    elif m.startswith("accidental"):
        x = np.asarray(d["coincidences"], float) - (d["accidentals"] if "accidentals" in d else 0)
    else:
        x = d["raw"] if "raw" in d else d["coincidences"]
    if p["smooth"]:
        x = ndi.gaussian_filter(x.astype(float), p["smooth"])
    mon.step(1, 1)
    res = Result()
    res.panels["image"] = inten(x, m)
    res.panels["coincidences"] = inten(d["coincidences"], "heralded coincidences")
    if "accidentals" in d:
        res.panels["accidentals"] = inten(d["accidentals"], "accidentals (shifted-frame heralds)")
    if "raw" in d:
        res.panels["camera (unheralded)"] = inten(d["raw"], "idler camera, all frames")
    if "heralds" in d:
        res.metrics["pairs"] = d["pairs"]
        res.metrics["heralds"] = d["heralds"]
        res.metrics["coincidences"] = float(np.sum(d["coincidences"]))
        res.metrics["accidentals"] = float(np.sum(d["accidentals"]))
        res.metrics["CAR"] = float(np.sum(d["coincidences"]) / max(np.sum(d["accidentals"]), 1))
    if "truth" in d:
        t = ndi.gaussian_filter(d["truth"], d["sigma_corr"])
        res.panels["object"] = inten(d["truth"], "object (in the signal arm)")
        res.metrics["correlation with object"] = float(np.corrcoef(np.ravel(x), t.ravel())[0, 1])
        res.panels["object (at the imaging resolution)"] = inten(t, "object blurred by the pair correlation width")
        compare(res, x, t, fit=True)
        res.curves.append(profile_curve("centre row", {"object (blurred by σ_corr)": t, "image": F.fit_scale(x, t)}))
    return res


register(Experiment(
    key="biphoton", title="Quantum ghost imaging (biphoton)", area=AREA,
    blurb="Spontaneous parametric down-conversion makes position-correlated signal/idler photon pairs. "
          "The signal photon meets the object and is caught by a bucket detector with no spatial "
          "resolution; the idler, which never touched the object, hits a photon-counting camera. Only "
          "heralded idler frames form the image; the unheralded camera sees a uniform glow. Accidentals "
          "are measured by pairing heralds with the next frame.",
    equations=[r"|\psi\rangle\propto\iint\Phi(\mathbf{x}_s,\mathbf{x}_i)\,a_s^{\dagger}(\mathbf{x}_s)a_i^{\dagger}(\mathbf{x}_i)|0\rangle,\quad |\Phi|^2\propto e^{-|\mathbf{x}_s-\mathbf{x}_i|^2/2\sigma^2}",
               r"G^{(2)}(\mathbf{x}_i)=\int |T(\mathbf{x}_s)|^2\,|\Phi(\mathbf{x}_s,\mathbf{x}_i)|^2\,d\mathbf{x}_s",
               r"\mathrm{CAR}=\frac{N_{\mathrm{coinc}}}{N_{\mathrm{acc}}}"],
    scene=[N(32, 16, 96), choice("object", "object", F.objects("text")),
           Param("frames", "camera frames", 300000, 1000, 5000000, kind="int", log=True),
           Param("mu", "mean pairs / frame", 0.3, 0.01, 5.0, log=True), Param("sigma_corr", "pair correlation width σ (px)", 0.7, 0.1, 5.0),
           Param("eta_s", "signal (bucket) efficiency", 0.5, 0.01, 1.0), Param("eta_i", "idler (camera) efficiency", 0.6, 0.01, 1.0),
           Param("dark_bucket", "bucket dark-count probability / frame", 1e-3, 0.0, 0.1),
           Param("dark_cam", "camera dark-count probability / px / frame", 1e-5, 0.0, 1e-3), SEED],
    solver=[method(PH.BIPHOTON_METHODS, PH.BIPHOTON_METHODS[1]), Param("smooth", "display smoothing σ (px)", 0.0, 0.0, 3.0)],
    simulate=_sim_biphoton, solve=_solve_biphoton,
    inputs=[Input("coincidences", "coincidence image"), Input("accidentals", "accidental image", optional=True),
            Input("raw", "unheralded camera image", optional=True)],
    load=lambda a, m: {k: np.asarray(v, float) for k, v in a.items()},
    example="Text in the signal arm, 300 k frames"))
