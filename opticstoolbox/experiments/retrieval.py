"""Phase unwrapping and phase retrieval experiments."""
from __future__ import annotations

import numpy as np
import scipy.fft as sfft

from ..core import fields as F
from ..core import intensity as IT
from ..core import phaseretrieval as PR
from ..core import ptychography as PT
from ..core import unwrap as U
from ..core.propagation import angular_spectrum
from ._common import (DX, ITERS, N, NOISE, SEED, WL, Curve, compare, compare_phase, Experiment, Input, Panel, Param, Result, amp, choice, conv,
                      err, extent, flag, inten, method, phase, register, signed, stack_input, table_values)

# ============================================================================ unwrapping
SURFACES = ("peaks + tilt", "cells (large phase)", "step discontinuity", "vortex pair", "steep ramp")


def _surface(kind, n, scale, rng):
    X, Y = F.grid(n, 1.0 / n)
    if kind == "peaks + tilt":
        x, y = 3 * X / 0.5, 3 * Y / 0.5
        z = 3 * (1 - x) ** 2 * np.exp(-x ** 2 - (y + 1) ** 2) - 10 * (x / 5 - x ** 3 - y ** 5) * np.exp(-x ** 2 - y ** 2) \
            - np.exp(-(x + 1) ** 2 - y ** 2) / 3
        return scale * z / 8 * 2 * np.pi + 6 * np.pi * X
    if kind.startswith("cells"):
        return scale * 2 * np.pi * F.cell_phantom(n, rng)
    if kind.startswith("step"):
        z = scale * 2 * np.pi * (0.4 * F.smooth_random(n, 20, rng) + 0.5 * X)
        z[(np.abs(X) < 0.2) & (np.abs(Y) < 0.2)] += 2.5 * np.pi
        return z
    if kind.startswith("vortex"):
        return scale * 2 * np.pi * 0.3 * (np.arctan2(Y - 0.1, X - 0.15) - np.arctan2(Y + 0.1, X + 0.15)) / np.pi + 4 * np.pi * Y
    return scale * 2 * np.pi * (8 * X + 3 * Y ** 2)


def _sim_unwrap(p, rng):
    n = p["n"]
    phi = _surface(p["surface"], n, p["scale"], rng)
    amp_ = np.ones((n, n))
    if p["dark_fraction"] > 0:
        amp_ = np.clip(F.smooth_random(n, 10, rng) + 1 - 2 * p["dark_fraction"], 0.02, 1)
    u = amp_ * np.exp(1j * phi)
    sig = p["noise"]
    u = u + sig * (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / np.sqrt(2)
    return {"wrapped": np.angle(u), "amplitude": np.abs(u), "truth": phi}


def _solve_unwrap(d, p, mon):
    psi = np.asarray(d["wrapped"], float)
    if np.nanmax(np.abs(psi)) > np.pi + 1e-3:
        psi = F.wrap(psi)
    q = None
    if p["quality"] == "amplitude" and d.get("amplitude") is not None:
        q = np.asarray(d["amplitude"], float)
    elif p["quality"] == "phase-derivative variance":
        q = U.quality_pdv(psi)
    res = Result()
    phi = U.unwrap(psi, p["method"], quality=q, monitor=mon)
    R = U.residues(psi)
    res.panels["wrapped phase"] = phase(psi, "wrapped phase ψ")
    res.panels["residues"] = Panel(R.astype(float), "signed", f"residues (+{int((R > 0).sum())} / −{int((R < 0).sum())})")
    res.panels["unwrapped"] = signed(phi - np.median(phi), f"unwrapped ({p['method']})")
    res.metrics["residues"] = float(np.abs(R).sum())
    res.metrics["congruence (fraction of pixels with W(φ) = ψ)"] = float(np.mean(np.abs(F.wrap(phi - psi)) < 1e-3))
    if "truth" in d:
        t = d["truth"]
        e = phi - t
        e = e - np.median(e)
        res.panels["error"] = err(e, "φ − φ_true (rad)")
        res.metrics["RMSE (rad)"] = float(np.sqrt(np.mean(e ** 2)))
        res.metrics["fraction off by ≥ π"] = float(np.mean(np.abs(e) > np.pi))
        compare(res, t + e, t, error_panel=False, metric_unit=" (rad)")
        n = t.shape[0]
        res.curves.append(Curve("centre row", {"truth": (np.arange(n), t[n // 2] - np.median(t)),
                                              "unwrapped": (np.arange(n), phi[n // 2] - np.median(phi)),
                                              "wrapped": (np.arange(n), psi[n // 2])}, "x (px)", "phase (rad)"))
    return res


register(Experiment(
    key="unwrap", title="Phase unwrapping", area="Phase unwrapping",
    blurb="Six 2-D unwrappers on surfaces with noise, dark (low-modulation) regions, true discontinuities and "
          "vortices. Path-following methods are congruent but can leak errors; least-squares methods are smooth "
          "but bias steep regions; PUMA minimises an ℓ^p energy exactly by graph cuts.",
    equations=[r"\phi=\psi+2\pi k,\quad k\in\mathbb{Z}^{N}",
               r"\min_{\phi}\sum w\left(\Delta\phi-\mathrm{W}(\Delta\psi)\right)^2\ \Rightarrow\ \nabla\cdot(w\nabla\phi)=\nabla\cdot(w\,\mathrm{W}(\nabla\psi))",
               r"\mathrm{PUMA}:\ \min_{k}\sum_{(i,j)}w_{ij}\,|\psi_j-\psi_i+2\pi(k_j-k_i)|^p"],
    scene=[N(256, 64, 1024), choice("surface", "surface", SURFACES), Param("scale", "phase scale", 1.0, 0.2, 5.0),
           Param("noise", "noise σ (field)", 0.15, 0.0, 1.0), Param("dark_fraction", "low-modulation area", 0.0, 0.0, 0.6), SEED],
    solver=[method(U.METHODS, "quality-guided"), choice("quality", "quality / weights", ("phase-derivative variance", "amplitude", "none"))],
    simulate=_sim_unwrap, solve=_solve_unwrap,
    inputs=[Input("wrapped", "wrapped phase (rad)"), Input("amplitude", "amplitude / quality map", optional=True)], meta=(),
    example="Peaks + tilt with noise σ = 0.15"))


# ============================================================================ CDI
def _sim_cdi(p, rng):
    m = p["size"]
    os_ = p["oversampling"]
    n = int(round(m * np.sqrt(os_) / 2) * 2)
    obj = F.resize(F.test_object(p["object"], 128, rng), (m, m))
    if p["complex"]:
        obj = obj * np.exp(1j * 1.5 * F.resize(F.test_object("blobs", 128, rng), (m, m)))
    x = F.pad_to(obj, (n, n))
    I = F.poisson_noise(np.abs(F.fft2(x)) ** 2, p["photons"], rng)
    return {"intensity": I, "truth": x, "support_size": m, "complex": bool(p["complex"])}


def _load_cdi(arr, m):
    I = np.asarray(arr["intensity"], float)
    I = sfft.ifftshift(I) if p_centered(I) else I
    d = {"intensity": I, "support_size": int(m["size"]), "complex": bool(m["complex"])}
    if "support" in arr:
        d["support"] = np.asarray(arr["support"]) > 0
    return d


def p_centered(I):
    n = I.shape[0]
    return I[n // 2, n // 2] > I[0, 0]


def _solve_cdi(d, p, mon):
    I = d["intensity"]
    mag = np.sqrt(np.clip(I, 0, None))
    n = I.shape[0]
    if d.get("support") is not None:
        S = d["support"]
    else:
        s = int(round(d["support_size"] * p["support_scale"]))
        S = F.pad_to(np.ones((s, s)), (n, n)) > 0
    shrink = p["support"] == "shrink-wrap"
    cplx = bool(d.get("complex", False))
    x, S2, e = PR.cdi(mag, S, p["method"], p["iters"], p["beta"], real=not cplx, positive=p["positivity"] and not cplx,
                      shrink=shrink, monitor=mon, rng=np.random.default_rng(p["restart_seed"]))
    res = Result()
    res.panels["diffraction (log)"] = Panel(np.log10(sfft.fftshift(I) + 1), "intensity", "measured |F x|², log₁₀",
                                            extent=(-n // 2, n - n // 2, -n // 2, n - n // 2),
                                            xlabel="q_x (pixels of the detector)", ylabel="q_y")
    res.panels["support"] = Panel(S2.astype(float), "intensity", "final support")
    if "truth" in d:
        xa, e2 = F.align_translation_and_phase(x if cplx else np.real(x), d["truth"])
        res.panels["reconstruction |x|"] = amp(xa, "reconstruction |x| (aligned)")
        if cplx:
            res.panels["reconstruction phase"] = phase(np.where(np.abs(xa) > 0.1 * np.abs(xa).max(), xa, 0), "phase")
        res.panels["truth"] = amp(d["truth"], "truth |x|")
        res.metrics["object error (aligned)"] = e2
        compare(res, np.abs(xa), np.abs(d["truth"]))
    else:
        res.panels["reconstruction |x|"] = amp(x, "reconstruction |x|")
    res.metrics["Fourier error"] = float(e[-1])
    res.curves.append(conv("Fourier-modulus error", {p["method"]: e}))
    return res


register(Experiment(
    key="cdi", title="Coherent diffraction imaging", area="Phase retrieval",
    blurb="Recover an object from the modulus of its Fourier transform (oversampled far-field diffraction) with "
          "support and positivity constraints: ER, HIO, HIO+ER, Difference Map, RAAR, ASR, with optional "
          "shrink-wrap support refinement.",
    equations=[r"P_M x=\mathcal{F}^{-1}\left\{\sqrt{I}\,\frac{\mathcal{F}x}{|\mathcal{F}x|}\right\},\qquad P_S x=x\cdot 1_S",
               r"\mathrm{HIO}:\ x_{k+1}=P_M x_k\ \ (\mathbf{r}\in S),\qquad x_{k+1}=x_k-\beta P_M x_k\ \ (\mathbf{r}\notin S)",
               r"\mathrm{RAAR}:\ x_{k+1}=\frac{\beta}{2}(R_S R_M+I)x_k+(1-\beta)P_M x_k"],
    scene=[choice("object", "object", F.objects("text")), Param("size", "object size", 64, 16, 256, kind="int"),
           Param("oversampling", "oversampling ratio σ", 4.0, 2.0, 16.0), flag("complex", "complex object", False), NOISE(1e6), SEED],
    solver=[method(PR.CDI_METHODS, "Difference Map"), ITERS(500, 10, 5000), Param("beta", "β", 0.9, 0.1, 1.0),
            choice("support", "support", ("fixed box", "shrink-wrap")), Param("support_scale", "support size (× object)", 1.0, 0.8, 1.6),
            flag("positivity", "positivity (real objects)", True), Param("restart_seed", "random start", 0, 0, 999, kind="int")],
    simulate=_sim_cdi, solve=_solve_cdi,
    inputs=[Input("intensity", "diffraction intensity (centred or FFT order)"), Input("support", "support mask", optional=True)],
    meta=("size", "complex"), load=_load_cdi, live=False,
    example="64-px text object, oversampling 4, Difference Map with a box support"))


# ============================================================================ generic phase retrieval
def _sim_gpr(p, rng):
    if p["model"] == "coded diffraction (image)":
        n = p["img"]
        x = F.resize(F.complex_object(p["object"], "cells", 128, phase_max=2.0, amp_contrast=0.5), (n, n)).ravel()
        A = PR.CDPOp((n, n), p["masks"], rng)
        shape = (n, n)
    else:
        nn = p["dim"]
        x = rng.standard_normal(nn) + 1j * rng.standard_normal(nn)
        A = PR.GaussianOp(nn, int(round(p["ratio"] * nn)), rng)
        shape = None
    y = np.abs(A(x)) ** 2
    if p["noise"]:
        y = np.maximum(y + p["noise"] * y.mean() * rng.standard_normal(y.shape), 0)
    return {"A": A, "y": y, "truth": x, "shape": shape}


def _solve_gpr(d, p, mon):
    A, y, x = d["A"], d["y"], d["truth"]
    f = PR.GENERIC_METHODS[p["method"]]
    if "SDP" in p["method"] and not isinstance(A, PR.GaussianOp):
        raise ValueError("PhaseCut / PhaseLift are dense SDP methods: use the Gaussian model with a small dimension")
    if "SDP" in p["method"] and A.n > 128:
        raise ValueError("SDP methods are limited to n ≤ 128 unknowns")
    z, e = f(A, y, iters=p["iters"], monitor=mon, truth=x)
    c = np.vdot(z, x)
    z = z * (c / abs(c) if abs(c) else 1)
    res = Result()
    res.metrics["relative error dist(z, x)"] = float(e[-1])
    res.metrics["m / n"] = A.m / A.n
    res.curves.append(conv("relative error vs iteration", {p["method"]: e}))
    if d["shape"] is not None:
        sh = d["shape"]
        res.panels["truth |x|"] = amp(x.reshape(sh), "truth |x|")
        res.panels["truth arg x"] = phase(x.reshape(sh), "truth arg x")
        res.panels["estimate |z|"] = amp(z.reshape(sh), "estimate |z|")
        res.panels["estimate arg z"] = phase(z.reshape(sh), "estimate arg z")
        compare(res, np.abs(z.reshape(sh)), np.abs(x.reshape(sh)), "amplitude", error_panel=False)
        compare_phase(res, z.reshape(sh), x.reshape(sh), np.abs(x.reshape(sh)) > 0.05 * np.abs(x).max())
        res.panels["one coded pattern"] = Panel(np.log1p(sfft.fftshift(y[:A.n].reshape(sh))), "intensity", "pattern 1 (log)")
    else:
        k = np.arange(len(x))
        res.curves.append(Curve("signal (real part)", {"truth": (k, x.real), "estimate": (k, z.real)}, "index", "Re x", logy=False))
    return res


register(Experiment(
    key="generic_pr", title="Phase retrieval from |Ax|²", area="Phase retrieval",
    blurb="Modern non-convex and convex solvers for y = |Ax|² with Gaussian or coded-diffraction A: Wirtinger flow, "
          "truncated and reshaped WF, alternating minimisation (GS) from a spectral start, and the SDP "
          "relaxations PhaseCut and PhaseLift for small problems.",
    equations=[r"y_i=|\langle a_i,x\rangle|^2,\qquad z_0=\lambda\,v_{\max}\left(\frac{1}{m}\sum_i y_i a_i a_i^{H}\right)",
               r"\mathrm{WF}:\ z\leftarrow z-\frac{\mu}{\|z_0\|^2}\,\frac{1}{m}\sum_i(|a_i^{H}z|^2-y_i)\,a_i a_i^{H}z",
               r"\mathrm{PhaseLift}:\ \min_{X\succeq 0}\ \frac{1}{2}\sum_i(a_i^{H}Xa_i-y_i)^2+\lambda\,\mathrm{tr}X"],
    scene=[choice("model", "measurement model", ("coded diffraction (image)", "Gaussian (vector)")),
           choice("object", "image amplitude (coded diffraction)", F.objects("blobs")),
           Param("img", "image size", 64, 16, 256, kind="int"), Param("masks", "coded masks L", 6, 2, 20, kind="int"),
           Param("dim", "vector length n", 64, 8, 512, kind="int"), Param("ratio", "m / n (Gaussian)", 6.0, 2.0, 16.0),
           Param("noise", "noise (rel.)", 0.0, 0.0, 0.5), SEED],
    solver=[method(PR.GENERIC_METHODS, "truncated Wirtinger flow"), ITERS(300, 10, 3000)],
    simulate=_sim_gpr, solve=_solve_gpr, inputs=[], meta=(), live=False,
    example="64×64 complex image from 6 coded diffraction patterns"))


# ============================================================================ multi-plane / GS
def _sim_mp(p, rng):
    n, dx, wl = p["n"], p["dx"], p["wl"]
    t = F.complex_object("blobs", p["object"], n, phase_max=p["phase_max"], amp_contrast=p["amp_contrast"], rng=rng)
    zs = p["z0"] + p["dz"] * np.arange(p["planes"])
    I = np.array([F.poisson_noise(np.abs(angular_spectrum(t, wl, dx, z)) ** 2, p["photons"], rng) for z in zs])
    Ff = F.poisson_noise(np.abs(F.fft2(t)) ** 2, p["photons"] * n * n, rng)
    return {"stack": I, "zs": zs, "truth": t, "wl": wl, "dx": dx, "far_field": Ff, "object_amp": np.abs(t)}


def _solve_mp(d, p, mon):
    res = Result()
    if p["method"].startswith("Gerchberg"):
        x, e = PR.gerchberg_saxton(d["object_amp"], np.sqrt(d["far_field"]), p["iters"], monitor=mon)
        obj = x
    else:
        amps = np.sqrt(np.clip(d["stack"], 0, None))
        u, e = PR.multiplane_gs(amps, d["zs"], d["wl"], d["dx"], p["iters"], monitor=mon)
        obj = angular_spectrum(u, d["wl"], d["dx"], -d["zs"][0])
    ex = extent(obj.shape[0], d["dx"])
    res.panels["plane 1"] = inten(d["stack"][0], f"intensity at z₁ = {d['zs'][0]:g} µm", extent=ex, unit="µm")
    res.panels["amplitude"] = amp(obj, "recovered amplitude", extent=ex, unit="µm")
    res.panels["phase"] = phase(obj, "recovered phase", extent=ex, unit="µm")
    if "truth" in d:
        m = np.zeros(obj.shape, bool)
        k = obj.shape[0] // 10
        m[k:-k, k:-k] = True
        res.panels["true phase"] = phase(d["truth"], "true phase", extent=ex, unit="µm")
        res.metrics["phase RMSE (rad)"] = F.phase_rmse(np.angle(obj), np.angle(d["truth"]), m)
        res.metrics["complex error"] = F.complex_error(obj, d["truth"], m)
        compare_phase(res, obj, d["truth"], m, extent=ex, unit="µm")
    res.curves.append(conv("amplitude mismatch", {p["method"]: e}))
    return res


register(Experiment(
    key="multiplane", title="Multi-plane & Gerchberg–Saxton", area="Phase retrieval",
    blurb="Classic Gerchberg–Saxton (known amplitudes in object and Fourier planes) and its multi-distance "
          "generalisation (single-beam multiple-intensity reconstruction) that cycles through a defocus stack.",
    equations=[r"u\leftarrow|u_{\mathrm{meas}}|\,e^{i\arg u}\ \ \mathrm{in\ each\ plane}",
               r"u_{k}=\mathcal{P}_{z_k-z_{k-1}}\{u_{k-1}\}"],
    scene=[N(256, 64, 1024), DX(2.0, 0.2, 10), WL(0.633), choice("object", "specimen", F.objects("usaf")),
           Param("phase_max", "max phase", 2.0, 0.1, 10.0, unit="rad"), Param("amp_contrast", "absorption contrast", 0.0, 0.0, 0.9),
           Param("z0", "first plane", 2000.0, 10.0, 50000.0, unit="µm"), Param("dz", "plane spacing", 2000.0, 10.0, 50000.0, unit="µm"),
           Param("planes", "planes", 5, 2, 12, kind="int"), NOISE(1e5), SEED],
    solver=[method(("multi-plane GS (defocus stack)", "Gerchberg–Saxton (object + Fourier amplitudes)")), ITERS(60, 1, 2000)],
    simulate=_sim_mp, solve=_solve_mp,
    inputs=[Input("stack", "defocus stack", "stack"), Input("zs", "plane distances (µm)", "table", optional=True)],
    meta=("wl", "dx", "z0", "dz"),
    load=lambda a, m: {"stack": stack_input(a["stack"]), "wl": m["wl"], "dx": m["dx"],
                       "zs": table_values(a["zs"]) if "zs" in a else m["z0"] + m["dz"] * np.arange(len(stack_input(a["stack"])))},
    example="Phase-only USAF from five defocused intensities"))


# ============================================================================ ptychography
def _sim_ptycho(p, rng):
    n_obj, n, wl, dx = p["n_obj"], p["probe_px"], p["wl"], p["dx"]
    obj = F.complex_object("blobs", p["object"], n_obj, phase_max=p["phase_max"], amp_contrast=p["amp_contrast"], rng=rng)
    P = PT.make_probe(n, wl, dx, p["probe_radius"], p["probe_defocus"])
    step = max(1, int(round(n * (1 - p["overlap"]))))
    pos = PT.raster_positions(n_obj, n, step, jitter=1.0, rng=rng, pattern=p["scan"])
    I = PT.simulate_ptycho(obj, P, pos, p["photons"] * n * n, rng)
    pos_meas = np.clip(pos + rng.uniform(-p["pos_error"], p["pos_error"], pos.shape), 0, n_obj - n)
    return {"patterns": I, "positions": pos_meas, "true_positions": pos, "truth": obj, "probe_truth": P,
            "n_obj": n_obj, "wl": wl, "dx": dx, "probe_radius": p["probe_radius"]}


def _load_ptycho(arr, m):
    I = stack_input(arr["patterns"])
    pos = np.asarray(arr["positions"], float).reshape(-1, 2)
    if m.get("pos_units", "pixels") != "pixels":
        pos = pos / m["dx"]
    pos = pos - pos.min(0)
    n = I.shape[-1]
    n_obj = int(np.ceil(pos.max() + n + 2))
    pr = arr.get("probe")
    return {"patterns": np.fft.ifftshift(I, axes=(-2, -1)) if I[0, n // 2, n // 2] > I[0, 0, 0] else I,
            "positions": pos, "n_obj": n_obj, "wl": m["wl"], "dx": m["dx"], "probe_radius": m["probe_radius"],
            "probe0": None if pr is None else np.asarray(pr)}


def _solve_ptycho(d, p, mon):
    I, pos = d["patterns"], d["positions"]
    n = I.shape[-1]
    P0 = d.get("probe0")
    if P0 is None:
        P0 = PT.make_probe(n, d["wl"], d["dx"], d["probe_radius"], 0.0)
    O, P, pos_f, e = PT.ptycho(I, pos, d["n_obj"], P0, p["method"], p["iters"], probe_update=p["probe_update"],
                               position_correction=p["position_correction"], monitor=mon)
    res = Result()
    res.panels["pattern 1 (log)"] = Panel(np.log10(sfft.fftshift(I[0]) + 1e-3), "intensity", "diffraction pattern 1, log₁₀",
                                          extent=(-n // 2, n - n // 2, -n // 2, n - n // 2),
                                          xlabel="q_x (detector px)", ylabel="q_y (detector px)")
    mask = PT.scanned_mask(pos_f, d["n_obj"], n, P, 0.3)
    if "truth" in d:
        Oa = F.align_complex(O, d["truth"], mask)
        # a linear phase ramp across the object (traded against a probe shift) is a known ambiguity of
        # ptychography with probe recovery: remove the best-fit plane before comparing
        dph = np.angle(Oa * np.conj(d["truth"]))
        yy, xx = np.indices(dph.shape)
        A = np.c_[xx[mask], yy[mask], np.ones(mask.sum())]
        cf = np.linalg.lstsq(A, dph[mask], rcond=None)[0]
        Oa = Oa * np.exp(-1j * (cf[0] * xx + cf[1] * yy + cf[2]))
        res.metrics["object error (scanned area)"] = F.complex_error(O, d["truth"], mask)
    else:
        Oa = O
    res.panels["object amplitude"] = amp(Oa * mask, "object amplitude")
    res.panels["object phase"] = phase(np.where(mask, Oa, 0), "object phase" + (" (phase ramp removed)" if "truth" in d else ""))
    if "truth" in d:
        res.panels["truth phase"] = phase(d["truth"], "true object phase")
        compare_phase(res, Oa, d["truth"], mask, title="object phase error (scanned area, rad)")
    res.panels["probe"] = amp(P, "recovered probe |P|")
    res.panels["probe phase"] = phase(P, "recovered probe phase")
    res.curves.append(conv("Fourier error", {p["method"]: e}))
    if "true_positions" in d:
        tp = d["true_positions"]
        err0 = np.mean(np.hypot(*(d["positions"] - tp).T))
        err1 = np.mean(np.hypot(*(pos_f - tp).T))
        res.metrics["mean position error before (px)"] = float(err0)
        if p["position_correction"]:
            res.metrics["mean position error after (px)"] = float(err1)
        res.curves.append(Curve("scan positions", {"true": (tp[:, 1], tp[:, 0]), "used/refined": (pos_f[:, 1], pos_f[:, 0])},
                                "x (px)", "y (px)", logy=False))
    return res


register(Experiment(
    key="ptychography", title="Ptychography", area="Phase retrieval",
    blurb="Scanning coherent diffraction imaging with overlapping probe positions: ePIE, rPIE and momentum-"
          "accelerated mPIE with simultaneous probe recovery and annealing-based correction of position errors.",
    equations=[r"I_j=|\mathcal{F}\{P(\mathbf{r})\,O(\mathbf{r}-\mathbf{r}_j)\}|^2",
               r"O_j\leftarrow O_j+\frac{P^{*}(\psi'-\psi)}{(1-\alpha)|P|^2+\alpha\max|P|^2}\quad\mathrm{(rPIE)}",
               r"\psi'=\mathcal{F}^{-1}\left\{\sqrt{I_j}\,\frac{\mathcal{F}\psi}{|\mathcal{F}\psi|}\right\}"],
    scene=[Param("n_obj", "object size", 192, 96, 512, kind="int"), Param("probe_px", "probe / detector size", 64, 32, 128, kind="int"),
           WL(0.5), DX(0.1, 0.005, 5), choice("object", "specimen", F.objects("cells")),
           Param("phase_max", "max phase", 2.0, 0.1, 6.0, unit="rad"), Param("amp_contrast", "absorption contrast", 0.4, 0.0, 0.9),
           Param("probe_radius", "pinhole radius", 1.6, 0.3, 4.0, unit="µm"), Param("probe_defocus", "probe defocus", 8.0, 0.0, 50.0, unit="µm"),
           Param("overlap", "linear overlap", 0.8, 0.3, 0.95), choice("scan", "scan", ("raster", "spiral")),
           Param("pos_error", "position error ±", 2.0, 0.0, 6.0, unit="px"), NOISE(1e3), SEED],
    solver=[method(PT.PTYCHO_METHODS, "rPIE"), ITERS(60, 1, 1000), flag("probe_update", "recover probe", True),
            flag("position_correction", "position correction", True)],
    simulate=_sim_ptycho, solve=_solve_ptycho,
    inputs=[Input("patterns", "diffraction patterns", "stack"), Input("positions", "scan positions (y, x) rows", "table"),
            Input("probe", "initial probe (complex .npy)", optional=True)],
    meta=("wl", "dx", "probe_radius"), load=_load_ptycho, live=False,
    example="Cell phantom, 80 % overlap, rPIE with probe recovery"))


# ============================================================================ Fourier ptychography
def _sim_fpm(p, rng):
    N_, n_lr, wl = p["n_hr"], p["n_hr"] // p["upsample"], p["wl"]
    dx_hr = p["pixel"] / p["mag"] / p["upsample"]
    obj = F.complex_object(p["amp_obj"], p["object"], N_, phase_max=p["phase_max"], amp_contrast=p["amp_contrast"], rng=rng)
    k, na_ill, _ = PT.led_wavevectors(p["leds"], p["led_pitch"], p["led_height"], wl,
                                      shift_mm=(p["led_shift"], -0.7 * p["led_shift"]), rot_deg=p["led_rot"])
    Pc = PT.fpm_pupil(n_lr, dx_hr * N_ / n_lr, wl, p["na"], {4: p["defocus_w"], 6: 0.6 * p["defocus_w"]})
    I = PT.simulate_fpm(obj, dx_hr, n_lr, wl, p["na"], k, Pc, p["photons"], rng)
    k_nom, _, _ = PT.led_wavevectors(p["leds"], p["led_pitch"], p["led_height"], wl)
    return {"images": I, "k": k_nom, "truth": obj, "pupil_truth": Pc, "N": N_, "dx_hr": dx_hr, "wl": wl, "na": p["na"],
            "na_ill": float(na_ill.max())}


def _load_fpm(arr, m):
    I = stack_input(arr["images"])
    n_lr = I.shape[-1]
    if "led_xy" in arr:
        xy = np.asarray(arr["led_xy"], float).reshape(-1, 2)
        R = np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2 + m["led_height"] ** 2)
        k = np.stack([xy[:, 1] / R / m["wl"], xy[:, 0] / R / m["wl"]], 1)
    else:
        k, _, _ = PT.led_wavevectors(m["leds"], m["led_pitch"], m["led_height"], m["wl"])
    N_ = n_lr * m["upsample"]
    return {"images": I, "k": k, "N": N_, "dx_hr": m["pixel"] / m["mag"] / m["upsample"], "wl": m["wl"], "na": m["na"],
            "na_ill": float(np.max(np.hypot(*k.T)) * m["wl"])}


def _solve_fpm(d, p, mon):
    o, P, cen, e = PT.fpm(d["images"], d["k"], d["N"], d["dx_hr"], d["wl"], d["na"], p["iters"],
                          pupil_update=p["pupil_update"], method=p["method"], led_correction=p["led_correction"], monitor=mon)
    res = Result()
    I = d["images"]
    c = int(np.argmin(np.hypot(*d["k"].T)))
    res.panels["bright-field image"] = inten(I[c], "centre-LED low-resolution image")
    res.panels["amplitude"] = amp(o, "recovered amplitude")
    res.panels["phase"] = phase(o, "recovered phase")
    dx_lr = d["dx_hr"] * d["N"] / P.shape[0]             # the pupil lives on the low-resolution camera grid
    fn = d["wl"] / (2 * dx_lr)                            # which spans ± λ/(2 dx) in NA units
    res.panels["pupil phase"] = Panel(np.where(np.abs(P) > 0, np.angle(P), np.nan), "phase", "recovered pupil phase",
                                      extent=(-fn, fn, -fn, fn), xlabel="NA_x", ylabel="NA_y")
    res.metrics["synthetic NA"] = d["na"] + d["na_ill"]
    res.metrics["objective NA"] = d["na"]
    if "truth" in d:
        FX, FY = F.freq_grid(o.shape, d["dx_hr"])
        lp = np.hypot(FX, FY) <= (d["na"] + d["na_ill"]) / d["wl"]
        ref = F.ifft2(F.fft2(d["truth"]) * lp)
        res.metrics["complex error vs band-limited truth"] = F.complex_error(o, ref)
        res.panels["truth phase"] = phase(ref, "band-limited truth phase")
        compare_phase(res, o, ref, _border_mask(o.shape))
    res.curves.append(conv("amplitude error", {p["method"]: e}))
    return res


register(Experiment(
    key="fpm", title="Fourier ptychographic microscopy", area="Phase retrieval",
    blurb="Synthesise a high-NA complex image from low-NA images under an LED array: sequential Gauss–Newton "
          "(EPRY) with pupil (aberration) recovery, the original alternating projections, and LED position "
          "self-calibration.",
    equations=[r"I_\ell=\left|\mathcal{F}^{-1}\left\{P(\mathbf{k})\,\hat{O}(\mathbf{k}-\mathbf{k}_\ell)\right\}\right|^2",
               r"\hat{O}\leftarrow\hat{O}+\frac{|P|\,P^{*}}{|P|_{\max}(|P|^2+\delta_1)}\,\Delta,\qquad P\leftarrow P+\frac{|\hat{O}|\,\hat{O}^{*}}{|\hat{O}|_{\max}(|\hat{O}|^2+\delta_2)}\,\Delta",
               r"\mathrm{NA}_{\mathrm{syn}}=\mathrm{NA}_{\mathrm{obj}}+\mathrm{NA}_{\mathrm{ill}}"],
    scene=[Param("n_hr", "high-res size", 256, 128, 512, kind="int"), Param("upsample", "upsampling", 4, 2, 8, kind="int"),
           WL(0.63), Param("na", "objective NA", 0.1, 0.02, 0.5), Param("mag", "magnification", 4.0, 1.0, 40.0),
           Param("pixel", "camera pixel", 6.4, 1.0, 20.0, unit="µm"), Param("leds", "LEDs per side", 11, 3, 21, kind="int"),
           Param("led_pitch", "LED pitch", 4.0, 1.0, 10.0, unit="mm"), Param("led_height", "LED distance", 80.0, 20.0, 200.0, unit="mm"),
           choice("object", "phase object", F.objects("cells")), choice("amp_obj", "amplitude object", F.objects("usaf")),
           Param("phase_max", "max phase", 1.5, 0.1, 6.0, unit="rad"), Param("amp_contrast", "absorption contrast", 0.5, 0.0, 0.9),
           Param("defocus_w", "pupil aberration", 0.05, 0.0, 0.3, unit="waves"), Param("led_shift", "LED array offset", 0.0, 0.0, 2.0, unit="mm"),
           Param("led_rot", "LED array rotation", 0.0, 0.0, 5.0, unit="°"), NOISE(1e4), SEED],
    solver=[method(("Gauss–Newton (EPRY)", "alternating projections")), ITERS(20, 1, 200), flag("pupil_update", "recover pupil", True),
            flag("led_correction", "LED position correction", False)],
    simulate=_sim_fpm, solve=_solve_fpm,
    inputs=[Input("images", "low-resolution images (one per LED)", "stack"),
            Input("led_xy", "LED positions (y, x) in mm", "table", optional=True)],
    meta=("wl", "na", "mag", "pixel", "upsample", "leds", "led_pitch", "led_height"), load=_load_fpm, live=False,
    example="11 × 11 LEDs, NA 0.1 objective → synthetic NA 0.43"))


# ============================================================================ TIE
def _sim_tie(p, rng):
    n, dx, wl = p["n"], p["dx"], p["wl"]
    ph = p["phase_max"] * F.normalize(F.test_object(p["object"], n, rng))
    a = 1 - p["amp_contrast"] * F.normalize(F.test_object("blobs", n, rng))
    t = a * np.exp(1j * ph)
    K = p["planes"]
    zs = (np.arange(K) - K // 2) * p["dz"]
    st = np.array([F.poisson_noise(np.abs(angular_spectrum(t, wl, dx, z)) ** 2, p["photons"], rng) for z in zs])
    return {"stack": st, "zs": zs, "truth": ph, "wl": wl, "dx": dx}


def _solve_tie(d, p, mon):
    dI, I0 = IT.axial_derivative(d["stack"], d["zs"], p["order"])
    phi = IT.tie(dI, np.maximum(I0, 1e-6), d["wl"], d["dx"], p["method"], p["reg"])
    res = Result()
    res.panels["in focus"] = inten(I0, "in-focus intensity")
    res.panels["∂I/∂z"] = signed(dI, "axial derivative ∂I/∂z")
    res.panels["phase"] = signed(phi - np.median(phi), f"TIE phase ({p['method']})")
    if "truth" in d:
        t = d["truth"]
        e = F.remove_plane(phi - t, order=1)
        res.panels["true phase"] = signed(t - np.median(t), "true phase")
        res.panels["error"] = err(e, "error (tilt removed)")
        res.metrics["phase RMSE, tilt removed (rad)"] = float(np.sqrt(np.mean(e ** 2)))
        compare(res, t + e, t, error_panel=False)
    return res


register(Experiment(
    key="tie", title="Transport of intensity (TIE)", area="Phase retrieval",
    blurb="Non-interferometric phase from a defocus stack: axial derivative by least-squares polynomial fit over "
          "2–9 planes, then the TIE solved by FFT or DCT Poisson solvers (uniform intensity) or Teague's two-step "
          "method (non-uniform intensity), with Tikhonov regularisation of the low frequencies.",
    equations=[r"-k\frac{\partial I}{\partial z}=\nabla_\perp\cdot\left(I\,\nabla_\perp\phi\right)",
               r"\phi=-k\,\nabla^{-2}\left[\frac{1}{\bar{I}}\frac{\partial I}{\partial z}\right],\qquad \nabla^{-2}\to-\frac{q^2}{q^4+\epsilon^2}",
               r"\mathrm{Teague}:\ \nabla^2\psi=-k\,\partial_z I,\ \ \phi=\nabla^{-2}\nabla\cdot\left(I^{-1}\nabla\psi\right)"],
    scene=[N(256, 64, 1024), DX(0.25, 0.02, 5), WL(0.53), choice("object", "phase object", F.objects("cells")),
           Param("phase_max", "max phase", 1.5, 0.1, 10.0, unit="rad"), Param("amp_contrast", "absorption contrast", 0.0, 0.0, 0.9),
           Param("dz", "plane spacing Δz", 2.0, 0.1, 50.0, unit="µm"), Param("planes", "planes", 3, 2, 9, kind="int"), NOISE(1e4), SEED],
    solver=[method(IT.TIE_METHODS, "DCT Poisson (uniform I)"), Param("order", "derivative fit order", 1, 1, 4, kind="int"),
            Param("reg", "Tikhonov ε", 0.01, 0.0, 10.0)],
    simulate=_sim_tie, solve=_solve_tie,
    inputs=[Input("stack", "defocus stack (symmetric about focus)", "stack"), Input("zs", "plane positions (µm)", "table", optional=True)],
    meta=("wl", "dx", "dz"),
    load=lambda a, m: {"stack": stack_input(a["stack"]), "wl": m["wl"], "dx": m["dx"],
                       "zs": table_values(a["zs"]) if "zs" in a else (np.arange(len(stack_input(a["stack"]))) - len(stack_input(a["stack"])) // 2) * m["dz"]},
    example="Cells, three planes at ±2 µm"))


# ============================================================================ DPC
def _sim_dpc(p, rng):
    n, dx, wl = p["n"], p["dx"], p["wl"]
    ph = p["phase_max"] * F.normalize(F.test_object(p["object"], n, rng))
    mu = p["amp_contrast"] * F.normalize(F.test_object("blobs", n, rng))
    t = np.exp(-mu + 1j * ph)
    ims, srcs, P = IT.dpc_images(t, wl, dx, p["na"], p["na"] * p["sigma"], step=p["source_step"], photons=p["photons"], rng=rng)
    return {"images": ims, "srcs": srcs, "truth": ph, "mu": mu, "wl": wl, "dx": dx, "na": p["na"]}


def _load_dpc(arr, m):
    ims = stack_input(arr["images"], 4)
    srcs = [IT.source_points(ims.shape[-2:], m["dx"], m["wl"], m["na"] * m["sigma"], k, m["source_step"])
            for k in ("top", "bottom", "left", "right")]
    return {"images": ims, "srcs": srcs, "wl": m["wl"], "dx": m["dx"], "na": m["na"]}


def _solve_dpc(d, p, mon):
    phi, mu = IT.dpc_reconstruct(d["images"], d["wl"], d["dx"], d["na"], d["srcs"], p["method"], p["reg"], p["tv"],
                                 p["iters"], monitor=mon)
    res = Result()
    sig = IT.dpc_signals(d["images"])
    res.panels["DPC top−bottom"] = signed(sig[0], "(I_t − I_b)/(I_t + I_b)")
    res.panels["DPC left−right"] = signed(sig[1], "(I_l − I_r)/(I_l + I_r)")
    res.panels["phase"] = signed(phi - np.median(phi), f"phase ({p['method']})")
    Hs = IT.dpc_transfer(d["images"].shape[-2:], d["wl"], d["dx"], d["na"], d["srcs"])
    nn = d["images"].shape[-1]
    fe = (-1 / (2 * d["dx"]) * d["wl"] / d["na"], (1 / (2 * d["dx"]) - 1 / (nn * d["dx"])) * d["wl"] / d["na"])
    res.panels["phase transfer function"] = signed(np.imag(sfft.fftshift(Hs[0][1])), "Im H_φ (top/bottom)",
                                                   extent=fe + fe, xlabel="f_x / (NA/λ)", ylabel="f_y / (NA/λ)")
    if mu is not None:
        res.panels["absorption"] = signed(mu, "absorption μ")
    if "truth" in d:
        e = F.remove_plane(phi - d["truth"], order=1)
        res.panels["true phase"] = signed(d["truth"], "true phase")
        res.metrics["phase RMSE, tilt removed (rad)"] = float(np.sqrt(np.mean(e ** 2)))
        compare(res, d["truth"] + e, d["truth"], title="phase error, tilt removed (rad)")
    return res


register(Experiment(
    key="dpc", title="Differential phase contrast", area="Phase retrieval",
    blurb="Quantitative phase from four half-circle illuminations (LED array or programmable condenser): Abbe "
          "simulation of partially coherent images, exact weak-object transfer functions and Tikhonov or TV "
          "(ADMM) deconvolution.",
    equations=[r"I_{\mathrm{DPC}}=\frac{I_1-I_2}{I_1+I_2},\qquad \hat{I}_{\mathrm{DPC}}=H_\phi\,\hat{\phi}+H_\mu\,\hat{\mu}",
               r"H_\phi(\mathbf{u})=i\sum_{\mathbf{u}'}S(\mathbf{u}')\left[P^{*}(\mathbf{u}')P(\mathbf{u}'+\mathbf{u})-P(\mathbf{u}')P^{*}(\mathbf{u}'-\mathbf{u})\right]/B",
               r"\hat{\phi}=\frac{\sum_k H_{\phi,k}^{*}\,\hat{I}_k}{\sum_k|H_{\phi,k}|^2+\alpha}"],
    scene=[N(256, 64, 512), DX(0.25, 0.05, 2), WL(0.53), Param("na", "objective NA", 0.25, 0.05, 1.0),
           Param("sigma", "illumination NA / objective NA", 1.0, 0.3, 1.0), choice("object", "phase object", F.objects("cells")),
           Param("phase_max", "max phase", 1.0, 0.05, 6.0, unit="rad"), Param("amp_contrast", "absorption", 0.0, 0.0, 0.5),
           Param("source_step", "source sampling", 3, 1, 8, kind="int"), NOISE(1e4), SEED],
    solver=[method(IT.DPC_METHODS, "TV (ADMM)"), Param("reg", "Tikhonov α", 1e-2, 1e-5, 1.0, log=True),
            Param("tv", "TV weight", 2e-3, 1e-5, 0.1, log=True), ITERS(60, 5, 500)],
    simulate=_sim_dpc, solve=_solve_dpc,
    inputs=[Input("images", "four images: top, bottom, left, right", "stack")],
    meta=("wl", "dx", "na", "sigma", "source_step"), load=_load_dpc,
    example="Cells at NA 0.25 with matched illumination"))


def _border_mask(shape, frac=0.08):
    m = np.zeros(shape, bool)
    a, b = int(frac * shape[0]), int(frac * shape[1])
    m[a:shape[0] - a, b:shape[1] - b] = True
    return m
