"""Digital holographic microscopy experiments."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from ..core import fields as F
from ..core import holography as H
from ..core import unwrap as U
from ..core.propagation import angular_spectrum
from ._common import (DX, ITERS, compare, compare_phase, freq_extent, N, NOISE, SEED, WL, Curve, Experiment, Input, Panel, Param, Result, amp, bandlimit,
                      choice, conv, err, extent, flag, inten, method, phase, register, rng_from, signed,
                      stack_input, table_values)

AREA = "Digital holography"


def _interior(shape, frac=0.08):
    m = np.zeros(shape, bool)
    a, b = int(frac * shape[0]), int(frac * shape[1])
    m[a:shape[0] - a, b:shape[1] - b] = True
    return m


def _sample(p, rng):
    n = p["n"]
    t = F.complex_object("blobs", p["object"], n, phase_max=p["phase_max"], amp_contrast=p["amp_contrast"], rng=rng)
    O = bandlimit(t, p["dx"], p["wl"], p["na"])
    return t, O


def _scene_common(extra=()):
    return [N(256, 64, 1024), DX(0.2, 0.02, 5), WL(0.532), Param("na", "objective NA", 0.3, 0.05, 1.4),
            choice("object", "specimen", F.objects("cells")), Param("phase_max", "max phase", 2.5, 0.1, 30.0, unit="rad"),
            Param("amp_contrast", "absorption contrast", 0.2, 0.0, 0.9)] + list(extra) + [NOISE(3e3), SEED]


# ============================================================================ off-axis
def _sim_offaxis(p, rng):
    t, O = _sample(p, rng)
    n, dx, wl = p["n"], p["dx"], p["wl"]
    if p["defocus"]:
        Oz = angular_spectrum(O, wl, dx, p["defocus"])
    else:
        Oz = O
    r = p["na"] / wl
    fc = p["carrier"] * r
    a = np.deg2rad(p["angle"])
    R = H.reference_wave((n, n), dx, wl, fc * np.cos(a), fc * np.sin(a), p["curvature"], amp=p["ref_ratio"])
    I = H.hologram(Oz, R, p["photons"], rng)
    blank = H.hologram(bandlimit(np.ones((n, n)), dx, wl, p["na"]), R, p["photons"], rng)
    return {"hologram": I, "blank": blank, "truth": O, "wl": wl, "dx": dx, "z": p["defocus"]}


def _load_offaxis(arr, m):
    return {"hologram": np.asarray(arr["hologram"], float), "blank": arr.get("blank"), "wl": m["wl"], "dx": m["dx"],
            "z": m["defocus"]}


def _reconstruct_offaxis(d, p):
    I = d["hologram"]
    u, info = H.offaxis_filter(I, radius=None, window=p["window"], return_info=True)
    if p["radius"] != 1.0:
        u, info = H.offaxis_filter(I, order=info["order"], radius=info["radius"] * p["radius"], window=p["window"],
                                   return_info=True)
    ref = None
    if p["compensation"].startswith("reference") and d.get("blank") is not None:
        ref = H.offaxis_filter(np.asarray(d["blank"], float), order=info["order"], radius=info["radius"], window=p["window"])
    comp = p["compensation"] if (ref is not None or not p["compensation"].startswith("reference")) else "polynomial fit"
    if p["refocus"] == "known distance" and d.get("z"):
        u = angular_spectrum(u, d["wl"], d["dx"], -d["z"])
        if ref is not None:
            ref = angular_spectrum(ref, d["wl"], d["dx"], -d["z"])
    elif p["refocus"] == "autofocus":
        zs = np.linspace(-abs(p["af_range"]), abs(p["af_range"]), 41)
        v, _ = H.focus_curve(u, d["wl"], d["dx"], zs, "Tamura coefficient", "phase")
        zb = H.best_focus(zs, v)
        u = angular_spectrum(u, d["wl"], d["dx"], zb)
        if ref is not None:
            ref = angular_spectrum(ref, d["wl"], d["dx"], zb)
        info["autofocus"] = (zs, v, zb)
    u = H.compensate(u, comp, reference=ref)
    return u, info


def _solve_offaxis(d, p, mon):
    u, info = _reconstruct_offaxis(d, p)
    n = u.shape[0]
    ex = extent(n, d["dx"])
    res = Result()
    res.panels["hologram"] = inten(d["hologram"], "hologram I = |O + R|²", extent=ex, unit="µm")
    W = info["window"]
    res.panels["spectrum + filter"] = Panel(info["spectrum"] * (0.35 + 0.65 * (W > 0)), "intensity",
                                            "log spectrum with the +1-order window", extent=freq_extent(n, d["dx"]),
                                            xlabel="f_x (1/µm)", ylabel="f_y (1/µm)")
    res.panels["amplitude"] = amp(u, "reconstructed amplitude", extent=ex, unit="µm")
    res.panels["wrapped phase"] = phase(u, "reconstructed phase (wrapped)", extent=ex, unit="µm")
    unw = U.unwrap(np.angle(u), p["unwrap"])
    unw -= np.median(unw)
    res.panels["unwrapped phase"] = signed(unw, "unwrapped phase (rad)", extent=ex, unit="µm")
    res.metrics["+1 order (row, col)"] = float(info["order"][0]),
    res.metrics = {"order row": float(info["order"][0]), "order col": float(info["order"][1]),
                   "filter radius (px)": float(info["radius"])}
    if "truth" in d:
        tr = d["truth"]
        m = _interior(u.shape)
        tu = U.unwrap(np.angle(tr), "least squares (DCT)")
        tu -= np.median(tu)
        res.panels["true phase"] = signed(tu, "true phase (rad)", extent=ex, unit="µm")
        e = F.wrap(np.angle(u) - np.angle(tr))
        e = F.wrap(e - np.angle(np.mean(np.exp(1j * e[m]))))
        res.panels["phase error"] = err(e * m, "phase error (rad)", extent=ex, unit="µm")
        res.metrics["phase RMSE (rad)"] = F.phase_rmse(np.angle(u), np.angle(tr), m)
        res.metrics["phase RMSE, tilt removed (rad)"] = F.phase_rmse(np.angle(u), np.angle(tr), m, tilt=True)
        compare_phase(res, u, tr, m, error_panel=False)
        res.curves.append(Curve("phase profile (centre row)", {"reconstruction": (np.arange(n), unw[n // 2]),
                                                               "truth": (np.arange(n), tu[n // 2])}, "x (px)", "phase (rad)"))
    if "autofocus" in info:
        zs, v, zb = info["autofocus"]
        res.curves.append(Curve("autofocus (Tamura, negated)", {"criterion": (zs, v)}, "refocus z (µm)", "sharpness", logy=False))
        res.metrics["autofocus z (µm)"] = zb
    return res


register(Experiment(
    key="offaxis", title="Off-axis DHM", area=AREA,
    blurb="Single-shot off-axis digital holographic microscopy: automatic +1-order detection with sub-pixel "
          "refinement, Fourier filtering, numerical refocusing (known distance or autofocus) and aberration "
          "compensation by polynomial fit (numerical parametric lens), PCA or a blank reference hologram.",
    equations=[r"I=|O|^2+|R|^2+O R^{*}+O^{*}R,\qquad R=e^{i2\pi\mathbf{f}_c\cdot\mathbf{r}}",
               r"\hat{O}=\mathcal{F}^{-1}\left\{W(\mathbf{f}-\mathbf{f}_c)\,\mathcal{F}\{I\}(\mathbf{f}+\mathbf{f}_c)\right\}",
               r"\phi_{\mathrm{corr}}=\arg\hat{O}-\sum_{p+q\leq 2}a_{pq}x^p y^q"],
    scene=_scene_common([Param("carrier", "carrier (× NA/λ)", 3.2, 1.0, 6.0), Param("angle", "carrier angle", 45.0, 0.0, 90.0, unit="°"),
                         Param("ref_ratio", "|R| / |O|", 1.0, 0.2, 5.0), Param("curvature", "reference curvature R", 0.0, -1e6, 1e6, unit="µm",
                                                                                  tip="0 = plane reference"),
                         Param("defocus", "specimen defocus z", 0.0, -100.0, 100.0, unit="µm")]),
    solver=[choice("window", "filter window", ("hann", "tukey", "circle")),
            Param("radius", "filter radius (× auto)", 1.0, 0.3, 1.6),
            choice("compensation", "aberration compensation", ("polynomial fit", "PCA", "reference hologram", "none")),
            choice("refocus", "refocus", ("known distance", "autofocus", "none")),
            Param("af_range", "autofocus range ±", 40.0, 1.0, 500.0, unit="µm"),
            choice("unwrap", "unwrapping", U.METHODS, "least squares (DCT)")],
    simulate=_sim_offaxis, solve=_solve_offaxis,
    inputs=[Input("hologram", "off-axis hologram"), Input("blank", "blank (object-free) hologram", optional=True)],
    meta=("wl", "dx", "defocus"), load=_load_offaxis,
    example="Cell phantom, 45° carrier (raise the specimen defocus to see numerical refocusing)"))


# ============================================================================ phase shifting
def _sim_ps(p, rng):
    t, O = _sample(p, rng)
    N_ = p["frames"]
    d = 2 * np.pi * np.arange(N_) / N_ + p["shift_error"] * rng.standard_normal(N_)
    d[0] = 0
    frames = np.array([H.hologram(O, p["ref_ratio"] * np.exp(1j * dk) * np.ones_like(O), p["photons"], rng) for dk in d])
    return {"frames": frames, "truth": O, "shifts": d, "wl": p["wl"], "dx": p["dx"]}


def _solve_ps(d, p, mon):
    fr = d["frames"]
    res = Result()
    if p["method"].startswith("N-step"):
        u = H.phase_shifting(fr)
        est = 2 * np.pi * np.arange(len(fr)) / len(fr)
    elif p["method"].startswith("AIA"):
        u, est = H.aia(fr, iters=p["iters"], monitor=mon)
    else:
        u = H.pca_phase(fr)
        est = None
    n = u.shape[0]
    ex = extent(n, d.get("dx", 1.0))
    res.panels["frame 1"] = inten(fr[0], "interferogram 1", extent=ex, unit="µm")
    res.panels["amplitude"] = amp(u, "|O R*|", extent=ex, unit="µm")
    res.panels["phase"] = phase(u, "recovered phase", extent=ex, unit="µm")
    if "truth" in d:
        tr = d["truth"]
        m = _interior(u.shape)
        rm = F.phase_rmse(np.angle(u), np.angle(tr), m)
        if p["method"].startswith("PCA"):          # sign ambiguity of PCA
            rm2 = F.phase_rmse(-np.angle(u), np.angle(tr), m)
            if rm2 < rm:
                u, rm = np.conj(u), rm2
        res.panels["true phase"] = phase(tr, "true phase", extent=ex, unit="µm")
        e = F.wrap(np.angle(u) - np.angle(tr))
        e = F.wrap(e - np.angle(np.mean(np.exp(1j * e[m]))))
        res.panels["phase error"] = err(e, "phase error (rad)", extent=ex, unit="µm")
        res.metrics["phase RMSE (rad)"] = rm
        compare_phase(res, u, tr, m, error_panel=False)
        if est is not None:
            k = np.arange(len(fr))
            res.curves.append(Curve("phase shifts", {"true δ_k": (k, np.unwrap(d["shifts"])),
                                                     "assumed / estimated": (k, np.unwrap(est))}, "frame", "δ (rad)"))
            res.metrics["max shift error (rad)"] = float(np.max(np.abs(F.wrap(np.asarray(est) - d["shifts"]))))
    return res


register(Experiment(
    key="phase_shifting", title="Phase-shifting DHM", area=AREA,
    blurb="In-line phase-shifting holography: least-squares N-step demodulation with the nominal shifts, the "
          "advanced iterative algorithm (AIA) that estimates unknown shifts, and PCA demodulation that needs no "
          "shift knowledge at all. Increase the shift error to see N-step fail.",
    equations=[r"I_k=a+b\cos(\phi-\delta_k),\qquad O R^{*}\propto\sum_k I_k\,e^{i\delta_k}",
               r"\mathrm{AIA}:\ \min_{a,b,\phi}\sum_k\|I_k-a-b\cos(\phi-\delta_k)\|^2\ \leftrightarrow\ \min_{\delta_k}(\cdot)",
               r"\mathrm{PCA}:\ I-\bar{I}=U\Sigma V^{T},\ \ \phi=\arg(\sigma_1 v_1+i\sigma_2 v_2)"],
    scene=_scene_common([Param("frames", "frames N", 4, 3, 12, kind="int"), Param("ref_ratio", "|R| / |O|", 1.0, 0.3, 4.0),
                         Param("shift_error", "shift error σ", 0.0, 0.0, 1.5, unit="rad")]),
    solver=[method(("N-step least squares", "AIA (unknown shifts)", "PCA demodulation")), ITERS(30, 1, 200)],
    simulate=_sim_ps, solve=_solve_ps, inputs=[Input("frames", "phase-shifted interferograms", "stack")],
    meta=("wl", "dx"), example="4 frames at π/2 steps (raise the shift error to compare N-step, AIA and PCA)"))


# ============================================================================ slightly off-axis
def _sim_soa(p, rng):
    t, O = _sample(p, rng)
    n, dx, wl = p["n"], p["dx"], p["wl"]
    r = p["na"] / wl
    fc = p["carrier"] * r
    R = H.reference_wave((n, n), dx, wl, fc, 0.0, amp=1.0)
    I1 = H.hologram(O, R, p["photons"], rng)
    I2 = H.hologram(O, R * 1j, p["photons"], rng)
    Ri = np.abs(R) ** 2 + 0 * I1
    Oi = F.poisson_noise(np.abs(O) ** 2, p["photons"], rng)
    return {"I1": I1, "I2": I2, "R_int": Ri, "O_int": Oi, "truth": O, "dx": dx, "wl": wl}


def _solve_soa(d, p, mon):
    res = Result()
    if p["method"].startswith("two-frame"):
        u = H.slightly_offaxis(d["I1"], d["I2"], d["R_int"], d["O_int"])
        # sign convention: C = D1 + iD2 = 2 O R* → +1 order is O R*
    else:
        u = H.offaxis_filter(d["I1"], window="circle")
    tr = d.get("truth")
    if tr is not None:
        m = _interior(u.shape)
        u = H.compensate(u, "polynomial fit", order=1)
        rm = F.phase_rmse(np.angle(u), np.angle(tr), m, tilt=True)
        rmc = F.phase_rmse(-np.angle(u), np.angle(tr), m, tilt=True)
        if rmc < rm:
            u, rm = np.conj(u), rmc
        res.metrics["phase RMSE (rad)"] = rm
    n = u.shape[0]
    ex = extent(n, d.get("dx", 1.0))
    res.panels["hologram"] = inten(d["I1"], "hologram (frame 1)", extent=ex, unit="µm")
    res.panels["spectrum"] = Panel(np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(d["I1"] - d["I1"].mean())))), "intensity",
                                   "log spectrum: orders overlap at a small carrier", extent=freq_extent(n, d.get("dx", 1.0)),
                                   xlabel="f_x (1/µm)", ylabel="f_y (1/µm)")
    res.panels["phase"] = phase(u, f"phase ({p['method']})", extent=ex, unit="µm")
    if tr is not None:
        res.panels["true phase"] = phase(tr, "true phase", extent=ex, unit="µm")
        compare_phase(res, u, tr, m, extent=ex, unit="µm")
    return res


register(Experiment(
    key="slightly_offaxis", title="Slightly off-axis DHM", area=AREA,
    blurb="With a carrier of only ~1–2× the object bandwidth the ±1 orders overlap the DC term. Subtracting the "
          "measured |R|² + |O|² and combining two frames shifted by π/2 cancels both the DC and the −1 order "
          "(Shaked et al. 2009), recovering the full bandwidth. Compare with plain Fourier filtering.",
    equations=[r"D_k=I_k-|R|^2-|O|^2=2\,\mathrm{Re}\{O R^{*}e^{-i\delta_k}\}",
               r"D_1+iD_2=2\,O R^{*}\quad(\delta_1=0,\ \delta_2=\pi/2)"],
    scene=_scene_common([Param("carrier", "carrier (× NA/λ)", 1.6, 1.0, 4.0)]),
    solver=[method(("two-frame slightly off-axis", "standard Fourier filtering"))],
    simulate=_sim_soa, solve=_solve_soa,
    inputs=[Input("I1", "hologram, δ = 0"), Input("I2", "hologram, δ = π/2"), Input("R_int", "reference-only intensity"),
            Input("O_int", "object-only intensity")], meta=("wl", "dx"),
    example="Carrier at 1.6× bandwidth: standard filtering fails, two-frame method works"))


# ============================================================================ Kramers–Kronig
def _sim_kk(p, rng):
    t, O = _sample(p, rng)
    n, dx, wl = p["n"], p["dx"], p["wl"]
    r = p["na"] / wl
    A = p["ref_ratio"] * np.abs(O).max()
    fx = -p["carrier"] * r
    R = H.reference_wave((n, n), dx, wl, fx, 0.0, amp=A)
    I = H.hologram(O, R, p["photons"], rng)
    return {"hologram": I, "ref_amp": A, "carrier_x": fx * dx, "truth": O, "dx": dx, "wl": wl}


def _load_kk(arr, m):
    I = np.asarray(arr["hologram"], float)
    return {"hologram": I, "ref_amp": np.sqrt(np.median(I)) * m["ref_ratio"] / (1 + m["ref_ratio"]),
            "carrier_x": -m["carrier"] * m["na"] / m["wl"] * m["dx"], "dx": m["dx"], "wl": m["wl"]}


def _solve_kk(d, p, mon):
    I = d["hologram"]
    res = Result()
    if p["method"] == "Kramers–Kronig":
        u, _ = H.kramers_kronig(I, d["ref_amp"], (0.0, d["carrier_x"]))
    else:
        u = H.offaxis_filter(I, window="circle")
    n = I.shape[0]
    ex = extent(n, d.get("dx", 1.0))
    res.panels["hologram"] = inten(I, "hologram", extent=ex, unit="µm")
    res.panels["spectrum"] = Panel(np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(I - I.mean())))), "intensity", "log spectrum",
                                   extent=freq_extent(n, d.get("dx", 1.0)), xlabel="f_x (1/µm)", ylabel="f_y (1/µm)")
    res.panels["amplitude"] = amp(u, "amplitude", extent=ex, unit="µm")
    res.panels["phase"] = phase(u, "phase", extent=ex, unit="µm")
    tr = d.get("truth")
    if tr is not None:
        m = _interior(u.shape)
        cand = [u, np.conj(u)]
        errs = [F.phase_rmse(np.angle(c), np.angle(tr), m, tilt=True) for c in cand]
        k = int(np.argmin(errs))
        res.metrics["phase RMSE (rad)"] = errs[k]
        res.metrics["complex error"] = F.complex_error(H.compensate(cand[k], "polynomial fit", order=1), tr, m)
        uk = H.compensate(cand[k], "polynomial fit", order=1)
        res.panels["phase"] = phase(uk, "phase", extent=ex, unit="µm")
        res.panels["true phase"] = phase(tr, "true phase", extent=ex, unit="µm")
        compare_phase(res, uk, tr, m, extent=ex, unit="µm")
    return res


register(Experiment(
    key="kk", title="Kramers–Kronig holography", area=AREA,
    blurb="Single-shot reconstruction without Fourier filtering (Baek et al. 2019): with the reference wave at "
          "the edge of the pupil, ln(1 + S/R) is analytic, so its phase is the Hilbert transform of its log-"
          "amplitude. Uses the full detector bandwidth; Fourier filtering at the same carrier cannot separate "
          "the orders.",
    equations=[r"I=|R|^2\,|1+s|^2,\qquad s=S/R,\ \ \mathrm{supp}\,\hat{s}\subset\{\mathbf{f}\cdot\hat{\mathbf{n}}>0\}",
               r"\ln(1+s)=\frac{1}{2}\ln\frac{I}{|R|^2}+i\,\mathcal{H}\left\{\frac{1}{2}\ln\frac{I}{|R|^2}\right\}",
               r"|R|>|S|\ \Rightarrow\ 1+s\ \mathrm{minimum\ phase}"],
    scene=_scene_common([Param("carrier", "reference offset (× NA/λ)", 1.0, 0.8, 3.0),
                         Param("ref_ratio", "|R| / max|S|", 1.5, 1.01, 5.0)]),
    solver=[method(("Kramers–Kronig", "off-axis Fourier filtering"))],
    simulate=_sim_kk, solve=_solve_kk, inputs=[Input("hologram", "hologram (reference at pupil edge)")],
    meta=("wl", "dx", "na", "carrier", "ref_ratio"), load=_load_kk,
    example="Reference exactly at the pupil edge: KK recovers the field, filtering cannot"))


# ============================================================================ in-line / multi-height
def _sim_inline(p, rng):
    n, dx, wl = p["n"], p["dx"], p["wl"]
    t = F.complex_object("blobs", p["object"], n, phase_max=p["phase_max"], amp_contrast=p["amp_contrast"], rng=rng)
    zs = p["z"] + p["dz"] * np.arange(p["heights"])
    Is = np.array([F.poisson_noise(np.abs(angular_spectrum(t, wl, dx, z)) ** 2, p["photons"], rng) for z in zs])
    return {"holograms": Is, "zs": zs, "truth": t, "wl": wl, "dx": dx}


def _load_inline(arr, m):
    st = stack_input(arr["holograms"])
    zs = m["z"] + m["dz"] * np.arange(len(st)) if "zs" not in arr else table_values(arr["zs"])
    return {"holograms": st, "zs": np.asarray(zs, float), "wl": m["wl"], "dx": m["dx"]}


def _solve_inline(d, p, mon):
    Is, zs, wl, dx = d["holograms"], d["zs"], d["wl"], d["dx"]
    res = Result()
    if p["method"] == "back-propagation":
        t = H.gabor_backprop(Is[0], wl, dx, zs[0])
        e = np.array([])
    elif p["method"].startswith("iterative"):
        t, e = H.gabor_iterative(Is[0], wl, dx, zs[0], p["iters"], p["constraint"], monitor=mon)
    else:
        t, e = H.multi_height(Is, zs, wl, dx, p["iters"], monitor=mon)
    n = t.shape[0]
    ex = extent(n, dx)
    res.panels["hologram"] = inten(Is[0], f"in-line hologram at z = {zs[0]:g} µm", extent=ex, unit="µm")
    res.panels["amplitude"] = amp(t, "object amplitude", extent=ex, unit="µm")
    res.panels["phase"] = phase(t, "object phase", extent=ex, unit="µm")
    if "truth" in d:
        tr = d["truth"]
        m = _interior(t.shape)
        res.panels["true phase"] = phase(tr, "true phase", extent=ex, unit="µm")
        res.metrics["complex error"] = F.complex_error(t, tr, m)
        res.metrics["phase RMSE (rad)"] = F.phase_rmse(np.angle(t), np.angle(tr), m)
        compare_phase(res, t, tr, m, extent=ex, unit="µm")
        compare(res, np.abs(t), np.abs(tr), "amplitude", mask=m, error_panel=False)
    if len(e):
        res.curves.append(conv("amplitude mismatch at the sensor", {p["method"]: e}))
    return res


register(Experiment(
    key="inline", title="In-line (Gabor) & multi-height", area=AREA,
    blurb="Lensless in-line holography: plain back-propagation (twin image), iterative twin-image removal with an "
          "absorption or pure-phase object constraint, and multi-height phase retrieval that cycles through "
          "holograms recorded at several sample-to-sensor distances.",
    equations=[r"I_k=|\mathcal{P}_{z_k}\{t\}|^2",
               r"u\leftarrow\sqrt{I_k}\,\exp\left(i\arg\mathcal{P}_{z_k-z_{k-1}}u\right)",
               r"t\leftarrow t\,/\max(1,|t|)\quad\mathrm{(absorption\ constraint)}"],
    scene=[N(256, 64, 1024), DX(1.12, 0.1, 10), WL(0.53), choice("object", "specimen", F.objects("usaf")),
           Param("phase_max", "max phase", 1.5, 0.1, 10.0, unit="rad"), Param("amp_contrast", "absorption contrast", 0.3, 0.0, 0.9),
           Param("z", "first distance z₁", 500.0, 10.0, 5000.0, unit="µm"), Param("dz", "height step Δz", 300.0, 5.0, 2000.0, unit="µm"),
           Param("heights", "heights", 4, 1, 10, kind="int"), NOISE(1e4), SEED],
    solver=[method(("back-propagation", "iterative twin-image removal", "multi-height phase retrieval"), "multi-height phase retrieval"),
            choice("constraint", "object constraint", ("absorption", "phase")), ITERS(30, 1, 500)],
    simulate=_sim_inline, solve=_solve_inline,
    inputs=[Input("holograms", "hologram(s), nearest first", "stack"), Input("zs", "distances (µm), one per hologram", "table", optional=True)],
    meta=("wl", "dx", "z", "dz"), load=_load_inline,
    example="Four heights, 0.5–1.4 mm: twin-image-free phase of a USAF phase target"))


# ============================================================================ autofocus
def _sim_af(p, rng):
    n, dx, wl = p["n"], p["dx"], p["wl"]
    t = F.complex_object("blobs", p["object"], n, phase_max=p["phase_max"],
                         amp_contrast=p["amp_contrast"], rng=rng)
    O = bandlimit(t, dx, wl, p["na"])
    u = angular_spectrum(O, wl, dx, p["defocus"])
    u = u + (np.sqrt(1.0 / p["photons"]) * (rng.standard_normal(u.shape) + 1j * rng.standard_normal(u.shape)) / np.sqrt(2))
    return {"field": u, "truth": O, "true_z": p["defocus"], "wl": wl, "dx": dx}


def _load_af(arr, m):
    if "field" in arr and np.iscomplexobj(arr["field"]):
        u = arr["field"]
    else:
        u = H.compensate(H.offaxis_filter(np.asarray(arr["field"], float)), "polynomial fit")
    return {"field": u, "wl": m["wl"], "dx": m["dx"]}


def _solve_af(d, p, mon):
    u, wl, dx = d["field"], d["wl"], d["dx"]
    zs = np.linspace(p["zmin"], p["zmax"], p["steps"])
    res = Result()
    ser = {}
    best = {}
    for k, mtr in enumerate(H.focus_metrics()):
        v, st = H.focus_curve(u, wl, dx, zs, mtr, p["object_type"])
        vn = (v - v.min()) / (v.max() - v.min() + 1e-20)
        ser[mtr] = (-zs, vn)
        best[mtr] = H.best_focus(zs, v)
        mon.step(k + 1, 4)
    zb = best[p["method"]]
    uf = angular_spectrum(u, wl, dx, zb)
    res.curves.append(Curve("focus criteria (normalised)", ser, "focus distance z (µm)", "criterion", logy=False))
    n = u.shape[0]
    ex = extent(n, dx)
    res.panels["defocused amplitude"] = amp(u, "recorded field amplitude", extent=ex, unit="µm")
    res.panels["refocused amplitude"] = amp(uf, f"amplitude at z = {-zb:.2f} µm", extent=ex, unit="µm")
    res.panels["refocused phase"] = phase(uf, "refocused phase", extent=ex, unit="µm")
    if "truth" in d:
        res.panels["true phase"] = phase(d["truth"], "in-focus phase (truth)", extent=ex, unit="µm")
        compare_phase(res, uf, d["truth"], _interior(uf.shape), extent=ex, unit="µm")
    res.metrics.update({f"best z, {k} (µm)": -v for k, v in best.items()})
    if "true_z" in d:
        res.metrics["true z (µm)"] = d["true_z"]
        res.metrics["focus error (µm)"] = abs(-zb - d["true_z"])
    return res


register(Experiment(
    key="autofocus", title="Numerical autofocus", area=AREA,
    blurb="Refocus a complex field over a range of distances and score each plane. For phase objects the in-focus "
          "amplitude is least structured (curves are negated so the maximum is always best focus).",
    equations=[r"T(z)=\sqrt{\sigma(|u_z|)/\langle|u_z|\rangle}\quad\mathrm{(Tamura)}",
               r"G(z)=\langle|\nabla|u_z||^2\rangle,\qquad L_1(z)=\sum|\mathcal{F}\{|u_z|\}|",
               r"\hat{z}=\arg\max_z\ \pm C(z)"],
    scene=[N(256, 64, 1024), DX(0.2, 0.02, 5), WL(0.532), Param("na", "NA", 0.4, 0.05, 1.4), choice("object", "specimen", F.objects("cells")),
           Param("phase_max", "max phase", 2.0, 0.1, 10.0, unit="rad"), Param("amp_contrast", "absorption contrast", 0.0, 0.0, 0.9),
           Param("defocus", "true defocus", 12.0, -60.0, 60.0, unit="µm"), NOISE(1e4), SEED],
    solver=[method(H.focus_metrics()), choice("object_type", "object type", ("phase", "amplitude")),
            Param("zmin", "search from", -40.0, -500.0, 0.0, unit="µm"), Param("zmax", "search to", 40.0, 0.0, 500.0, unit="µm"),
            Param("steps", "planes", 81, 11, 401, kind="int")],
    simulate=_sim_af, solve=_solve_af,
    inputs=[Input("field", "complex field (.npy) or off-axis hologram")], meta=("wl", "dx"), load=_load_af,
    example="Phase-only cells defocused by 12 µm"))


# ============================================================================ compressive holography
def _sim_ch(p, rng):
    n, dx, wl = p["n"], p["dx"], p["wl"]
    Z = p["planes"]
    zs = p["z0"] + p["dz"] * np.arange(Z)
    f = np.zeros((Z, n, n))
    letters = "OPTICS"
    for k in range(Z):
        if p["scene"] == "particles":
            for _ in range(p["particles"]):
                y, x = rng.integers(n // 8, 7 * n // 8, 2)
                f[k, y - 1:y + 2, x - 1:x + 2] = 1
        else:
            img = (F.text_image(letters[k % len(letters)], 128) if p["scene"] == "letters"
                   else F.normalize(F.test_object(F.OBJECTS[k % len(F.OBJECTS)], 128, rng)))
            f[k] = F.pad_to(F.resize(img, (n // 2, n // 2)), (n, n)) * 0.6
            f[k] = np.roll(f[k], ((k - Z // 2) * n // 10, (k - Z // 2) * n // 10), (0, 1))
    g = H.compressive_forward(f * 0.05, wl, dx, zs)
    g = g + p["noise"] * np.std(g) * rng.standard_normal(g.shape)
    return {"hologram": g, "zs": zs, "truth": f, "wl": wl, "dx": dx}


def _solve_ch(d, p, mon):
    g, zs, wl, dx = d["hologram"], d["zs"], d["wl"], d["dx"]
    res = Result()
    if p["method"] == "back-propagation":
        x = H.compressive_adjoint(g, wl, dx, zs)
        h = np.array([])
    else:
        x, h = H.compressive_holography(g, wl, dx, zs, "TV" if "TV" in p["method"] else "l1", p["lam"], p["iters"],
                                        monitor=mon)
    res.panels["hologram"] = signed(g, "hologram g (|R|² removed)")
    res.panels["reconstruction"] = Panel(np.maximum(x, 0) if p["method"] != "back-propagation" else np.abs(x),
                                         "volume", "reconstructed planes (slider = z)", extent=extent(g.shape[-1], dx),
                                         unit="µm", zextent=(float(zs[0]), float(zs[-1])), zlabel="z (µm)")
    if "truth" in d:
        tr = d["truth"]
        res.panels["truth"] = Panel(tr, "volume", "true planes", extent=extent(g.shape[-1], dx), unit="µm",
                                    zextent=(float(zs[0]), float(zs[-1])), zlabel="z (µm)")
        xx = np.abs(x)
        res.metrics["normalised correlation"] = float(np.corrcoef(xx.ravel(), tr.ravel())[0, 1])
        compare(res, xx, tr, fit=True)
        # cross-talk: energy of plane k's reconstruction outside its true support
        res.curves.append(Curve("per-plane correlation", {"correlation": (zs, [np.corrcoef(xx[k].ravel(), tr[k].ravel())[0, 1]
                                                                                 for k in range(len(zs))])},
                                "z (µm)", "corr", logy=False))
    if len(h):
        res.curves.append(conv("data residual", {p["method"]: h}))
    return res


register(Experiment(
    key="compressive_holo", title="Compressive holography", area=AREA,
    blurb="Recover a multi-plane scene from ONE in-line hologram (Brady et al. 2009). Back-propagation smears every "
          "plane into the others; sparsity (ℓ1) or TV priors solved with FISTA separate them.",
    equations=[r"g=2\,\mathrm{Re}\left\{\sum_{z}\mathcal{P}_{z}\{f_z\}\right\}+e",
               r"\hat{f}=\arg\min_{f\geq 0}\ \frac{1}{2}\|g-Hf\|_2^2+\tau\sum_z \mathrm{TV}(f_z)"],
    scene=[N(128, 64, 512), DX(2.0, 0.2, 10), WL(0.633), Param("planes", "planes", 4, 2, 8, kind="int"),
           Param("z0", "first plane", 2000.0, 100.0, 50000.0, unit="µm"), Param("dz", "plane spacing", 1500.0, 50.0, 20000.0, unit="µm"),
           choice("scene", "scene", ("letters", "sample objects (one per plane)", "particles")), Param("particles", "particles per plane", 12, 1, 100, kind="int"),
           Param("noise", "noise (rel. σ)", 0.01, 0.0, 0.5), SEED],
    solver=[method(("TV (FISTA)", "ℓ1 sparsity (FISTA)", "back-propagation")), Param("lam", "τ (relative)", 0.02, 1e-4, 0.5, log=True),
            ITERS(60, 5, 500)],
    simulate=_sim_ch, solve=_solve_ch, inputs=[], meta=(), live=False,
    example="Four letters at 2–6.5 mm from one hologram"))
