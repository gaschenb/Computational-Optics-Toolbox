"""3-D shape: structure from focus/defocus, structure from polarisation, light-field depth, fringe projection."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from ..core import depth as DP
from ..core import fields as F
from ..core import unwrap as U
from ..core import wavefront as WF
from ._common import (N, SEED, Curve, compare, Experiment, Input, Panel, Param, Result, choice, depth, err, flag, inten, method,
                      phase, register, signed, stack_input, table_values)

AREA = "3-D shape"
SCENES = ("ramp + bumps", "steps", "sphere", "waves")


# ----------------------------------------------------------------------------- focus / defocus
DFD_METHODS = tuple(f"depth from focus ({m})" for m in DP.FOCUS_MEASURES) + ("depth from defocus (2 images)",)


def _sim_dfd(p, rng):
    n = p["n"]
    d = DP.depth_scene(n, p["scene"], rng)
    tex = F.natural_texture(n, rng) if p["texture"] == "1/f texture" else F.normalize(np.clip(F.test_object(p["texture"], n, rng), 0, None))
    if p["texture"] not in ("1/f texture", "texture"):
        tex = 0.1 + 0.9 * tex          # keep some light everywhere: focus measures need texture to work on
    tex = 0.2 + 0.8 * F.normalize(tex)
    fz = np.linspace(0, 1, p["frames"])
    st = DP.focal_stack(tex, d, fz, p["kappa"], p["noise"], rng)
    return {"stack": st, "focuses": fz, "kappa": p["kappa"], "truth": d, "texture": tex}


def _solve_dfd(d, p, mon):
    st, fz = d["stack"], np.asarray(d["focuses"], float)
    m = p["method"]
    res = Result()
    if m.startswith("depth from focus"):
        kind = m[len("depth from focus ("):-1]
        z, aif, FM = DP.depth_from_focus(st, fz, kind, p["window"])
        res.panels["all-in-focus"] = inten(aif, "all-in-focus composite")
    else:
        z, E = DP.depth_from_defocus(st[0], st[-1], fz[0], fz[-1], d["kappa"], p["candidates"], p["window"])
        res.panels["matching error"] = inten(np.log10(E + 1e-12), "log₁₀ relative-blur residual")
    if p["median"]:
        z = ndi.median_filter(z, p["median"])
    mon.step(1, 1)
    res.panels["near focus"] = inten(st[0], f"frame 1 (focus {fz[0]:.2f})")
    res.panels["far focus"] = inten(st[-1], f"frame {len(st)} (focus {fz[-1]:.2f})")
    res.panels["depth"] = depth(z, f"depth ({m})")
    fv = dict(zextent=(float(fz[0]), float(fz[-1])), zlabel="focus setting")
    res.panels["focus stack"] = Panel(st, "volume", "focus stack (slider = focus setting)", **fv)
    # depth slicing: the pixels whose estimated (and true) depth lies at the slider's focus setting
    half = 0.5 * np.median(np.diff(fz)) if len(fz) > 1 else 0.05
    res.panels["estimated depth slices"] = Panel((np.abs(z[None] - fz[:, None, None]) <= half) * st,
                                                 "volume", "in focus at this setting (estimated depth)", **fv)
    if "truth" in d:
        t = d["truth"]
        lv = (float(np.min(t)), float(np.max(t)))
        res.panels["true depth"] = depth(t, "true depth", levels=lv)
        res.panels["depth"].levels = lv
        res.panels["true depth slices"] = Panel((np.abs(t[None] - fz[:, None, None]) <= half) * st,
                                                "volume", "in focus at this setting (true depth)", **fv)
        res.panels["error"] = err(z - t, "depth error")
        res.metrics["depth RMSE"] = float(np.sqrt(np.mean((z - t) ** 2)))
        res.metrics["median |error|"] = float(np.median(np.abs(z - t)))
        compare(res, z, t)
        r = t.shape[0] // 2
        x = np.arange(t.shape[1])
        res.curves.append(Curve("centre-row depth", {"truth": (x, t[r]), "estimate": (x, z[r])}, "x (px)", "depth", logy=False))
    return res


register(Experiment(
    key="defocus", title="Structure from (de)focus", area=AREA,
    blurb="Depth from a focal stack (per-pixel argmax of a focus measure with Gaussian peak interpolation, "
          "plus an all-in-focus composite) or from just two differently focused images by matching their "
          "relative blur. Blur is depth dependent via the thin-lens circle of confusion.",
    equations=[r"\sigma(\mathbf{x})=\kappa\,|d(\mathbf{x})-f|+\sigma_0",
               r"\hat{d}(\mathbf{x})=\arg\max_k\ \mathrm{SML}_k(\mathbf{x}),\quad \mathrm{ML}=|2I-I_{x-1}-I_{x+1}|+|2I-I_{y-1}-I_{y+1}|",
               r"\hat{d}=\arg\min_d\ \sum_{W}\left(G_{\sqrt{\sigma_2^2-\sigma_1^2}}\ast I_1-I_2\right)^2\quad\mathrm{(DfD)}"],
    scene=[N(192, 64, 512), choice("scene", "depth scene", SCENES), choice("texture", "texture", ("1/f texture",) + F.objects("cells")),
           Param("frames", "focal-stack frames", 12, 2, 40, kind="int"), Param("kappa", "blur per unit depth κ", 6.0, 1.0, 20.0),
           Param("noise", "noise σ", 0.005, 0.0, 0.1), SEED],
    solver=[method(DFD_METHODS), Param("window", "window (px)", 9, 3, 31, kind="int"),
            Param("candidates", "DfD depth candidates", 64, 8, 256, kind="int"), Param("median", "median filter (px, 0 = off)", 5, 0, 15, kind="int")],
    simulate=_sim_dfd, solve=_solve_dfd,
    inputs=[Input("stack", "focal stack", "stack"), Input("focuses", "focus positions (one per frame)", "table", optional=True)],
    meta=("kappa",),
    load=lambda a, m: (lambda st: {"stack": st, "kappa": m["kappa"],
                                   "focuses": table_values(a["focuses"]) if "focuses" in a else np.linspace(0, 1, len(st))})(stack_input(a["stack"])),
    example="1/f texture on a ramp with bumps, 12-frame focal stack"))


# ----------------------------------------------------------------------------- polarisation
def _sim_sfp(p, rng):
    n = p["n"]
    h, mask = DP.sfp_scene(n, p["shape"], rng)
    Nt = DP.surface_normals(h)
    light = (p["light_x"], p["light_y"], 1.0)
    ang = tuple(np.linspace(0, 180, p["polarisers"], endpoint=False))
    ims = DP.polarisation_images(Nt, mask, p["n_ref"], light, 0.8, p["model"], ang, p["noise"], rng)
    return {"images": ims, "angles": np.array(ang), "mask": mask, "n_ref": p["n_ref"], "model": p["model"],
            "light": light, "truth_normals": Nt, "truth": h}


def _load_sfp(a, m):
    ims = stack_input(a["images"])
    ang = table_values(a["angles"]) if "angles" in a else np.linspace(0, 180, len(ims), endpoint=False)
    if "mask" in a:
        mask = np.asarray(a["mask"]) > 0
    else:
        s = ims.mean(0)
        mask = ndi.binary_fill_holes(s > 0.1 * s.max())
    return {"images": ims, "angles": ang, "mask": mask, "n_ref": m["n_ref"], "model": m["model"],
            "light": (m["light_x"], m["light_y"], 1.0)}


def _solve_sfp(d, p, mon):
    mask = d["mask"]
    Nn, info = DP.normals_from_polarisation(d["images"], mask, d["n_ref"], d["model"], p["method"], d["light"],
                                            angles_deg=tuple(d["angles"]))
    mon.step(1, 2)
    z = DP.integrate_normals(Nn, mask, p["integrator"])
    mon.step(2, 2)
    res = Result()
    res.panels["unpolarised intensity"] = inten(info["S0"], "S₀ (intensity)")
    res.panels["DoLP"] = inten(np.where(mask, info["dolp"], 0), "degree of linear polarisation")
    res.panels["AoLP"] = Panel(np.where(mask, 2 * info["aolp"], 0), "phase", "2 × angle of linear polarisation")
    res.panels["normals"] = Panel(DP.normals_rgb(Nn) * mask[..., None], "rgb", "surface normals (RGB = xyz)")
    res.panels["height"] = depth(np.where(mask, z, np.nan), f"integrated height ({p['integrator'].split(' (')[0]})")
    if "truth_normals" in d:
        Nt, h = d["truth_normals"], d["truth"]
        ae = DP.angular_error_deg(Nn, Nt, mask)
        res.panels["true normals"] = Panel(DP.normals_rgb(Nt) * mask[..., None], "rgb", "true normals")
        res.panels["angular error"] = Panel(np.where(mask, ae, np.nan), "intensity", "normal error (°)")
        res.metrics["mean normal error (°)"] = float(ae[mask].mean())
        res.metrics["median normal error (°)"] = float(np.median(ae[mask]))
        hz = F.fit_scale(z[mask], h[mask])
        res.metrics["height NRMSE"] = float(np.linalg.norm(hz - h[mask]) / np.linalg.norm(h[mask] - h[mask].mean()))
        hz_full = np.where(mask, 0.0, np.nan)
        hz_full[mask] = hz
        res.panels["true height"] = depth(np.where(mask, h, np.nan), "true height")
        compare(res, np.where(mask, hz_full, 0), np.where(mask, h, 0), "height", mask=mask, error_panel=False)
        r = h.shape[0] // 2
        x = np.arange(h.shape[1])
        zz = np.full(h.shape, np.nan); zz[mask] = hz
        res.curves.append(Curve("centre-row height", {"truth": (x, np.where(mask[r], h[r], np.nan)), "estimate": (x, zz[r])},
                                "x (px)", "height (px)", logy=False))
    return res


register(Experiment(
    key="polarization", title="Structure from polarisation", area=AREA,
    blurb="Polariser-angle images give the linear Stokes parameters. The degree of polarisation fixes the "
          "zenith angle through the Fresnel equations (diffuse or specular reflection), the angle of "
          "polarisation fixes the azimuth up to a π ambiguity, resolved by a convexity prior or by shading. "
          "Normals are integrated into a height map.",
    equations=[r"I(\alpha)=\frac{S_0}{2}\left(1+\rho\cos(2\alpha-2\phi)\right)",
               r"\rho_d(\theta)=\frac{(n-1/n)^2\sin^2\theta}{2+2n^2-(n+1/n)^2\sin^2\theta+4\cos\theta\sqrt{n^2-\sin^2\theta}}",
               r"\mathbf{n}=(\sin\theta\cos\phi,\ \sin\theta\sin\phi,\ \cos\theta),\quad \phi\in\{\phi_0,\ \phi_0+\pi\}",
               r"\min_z\ \|\nabla z-(p,q)\|^2\ \Rightarrow\ \nabla^2 z=\partial_x p+\partial_y q"],
    scene=[N(192, 64, 512), choice("shape", "object", ("sphere + bumps", "vase", "sphere")),
           choice("model", "reflection", ("diffuse", "specular")), Param("n_ref", "refractive index", 1.5, 1.1, 2.5),
           Param("polarisers", "polariser angles", 4, 3, 12, kind="int"), Param("light_x", "light direction x", 0.3, -1.0, 1.0),
           Param("light_y", "light direction y", -0.3, -1.0, 1.0), Param("noise", "noise σ", 0.002, 0.0, 0.05), SEED],
    solver=[method(DP.SFP_DISAMBIG, label="azimuth disambiguation"), choice("integrator", "normal integration", DP.INTEGRATORS, DP.INTEGRATORS[1])],
    simulate=_sim_sfp, solve=_solve_sfp,
    inputs=[Input("images", "polariser images", "stack"), Input("angles", "polariser angles (°)", "table", optional=True),
            Input("mask", "object mask", optional=True)],
    meta=("model", "n_ref", "light_x", "light_y"), load=_load_sfp,
    example="Diffuse sphere with bumps, 4 polariser angles"))


# ----------------------------------------------------------------------------- light field
def _lf_scene(n, rng, kind):
    bg = 0.2 + 0.6 * F.natural_texture(n, rng)
    layers = [(bg, np.ones((n, n)))]
    depths = [0.95]
    Y, X = np.indices((n, n)) / n
    shapes = []
    if kind == "three planes":
        shapes = [(np.hypot(X - 0.35, Y - 0.4) < 0.17, 0.15), ((np.abs(X - 0.65) < 0.15) & (np.abs(Y - 0.6) < 0.2), 0.5)]
    else:
        for k in range(5):
            c = rng.uniform(0.2, 0.8, 2)
            shapes.append((np.hypot(X - c[0], Y - c[1]) < rng.uniform(0.06, 0.14), rng.uniform(0.05, 0.75)))
    for sh, dpt in shapes:
        tex = 0.2 + 0.8 * F.natural_texture(n, rng)
        layers.append((tex, ndi.gaussian_filter(sh.astype(float), 0.7)))
        depths.append(dpt)
    truth = np.full((n, n), depths[0])
    for (tex, a), dpt in sorted(zip(layers[1:], depths[1:]), key=lambda z: -z[1]):
        truth = np.where(a > 0.5, dpt, truth)
    return layers, np.array(depths), truth


def _sim_lf(p, rng):
    n = p["n"]
    layers, depths, truth = _lf_scene(n, rng, p["scene"])
    L = DP.light_field(layers, depths, p["views"], p["max_disp"])
    if p["noise"]:
        L = L + p["noise"] * rng.standard_normal(L.shape)
    truth_slope = p["max_disp"] * (truth - 0.5) * 2
    return {"lightfield": L, "max_disp": p["max_disp"], "truth": truth_slope}


def _load_lf(a, m):
    L = np.asarray(a["lightfield"], float)
    if L.ndim == 3:
        k = int(round(np.sqrt(L.shape[0])))
        if k * k != L.shape[0]:
            raise ValueError("a 3-D light-field stack needs a square number of views (u-major)")
        L = L.reshape(k, k, *L.shape[1:])
    if L.ndim == 5:
        L = L.mean(-1)
    return {"lightfield": L, "max_disp": m["max_disp"]}


def _solve_lf(d, p, mon):
    L = d["lightfield"]
    md = d["max_disp"]
    slopes = np.linspace(-md * 1.1, md * 1.1, p["slopes"])
    s, V, C = DP.lf_depth(L, slopes, p["method"], p["window"])
    if p["median"]:
        s = ndi.median_filter(s, p["median"])
    mon.step(1, 1)
    res = Result()
    c = L.shape[0] // 2
    res.panels["centre view"] = inten(L[c, c], "centre sub-aperture view")
    res.panels["refocused near"] = inten(DP.refocus(L, slopes[0] * 0.8), f"refocused (slope {slopes[0] * 0.8:.2f})")
    res.panels["refocused far"] = inten(DP.refocus(L, slopes[-1] * 0.8), f"refocused (slope {slopes[-1] * 0.8:.2f})")
    epi = L[:, c, L.shape[2] // 2, :]
    res.panels["EPI"] = Panel(epi, "intensity", "epipolar image: lines slope with disparity", xlabel="x (px)",
                              ylabel="view u", aspect=False)
    res.panels["refocus stack"] = Panel(np.array([DP.refocus(L, sl) for sl in np.linspace(slopes[0], slopes[-1], 15)]),
                                        "volume", "synthetic refocus stack (slider = focal plane)",
                                        zextent=(float(slopes[0]), float(slopes[-1])), zlabel="refocus slope (px / view)")
    res.panels["disparity"] = depth(s, f"disparity / view ({p['method'].split(' (')[0]})")
    if "truth" in d:
        t = d["truth"]
        lv = (float(np.min(t)), float(np.max(t)))
        res.panels["true disparity"] = depth(t, "true disparity", levels=lv)
        res.panels["disparity"].levels = lv
        res.metrics["disparity RMSE (px)"] = float(np.sqrt(np.mean((s - t) ** 2)))
        res.metrics["bad pixels (>0.25 px) %"] = float(100 * np.mean(np.abs(s - t) > 0.25))
        compare(res, s, t)
    return res


register(Experiment(
    key="lightfield", title="Light-field refocusing and depth", area=AREA,
    blurb="A 4-D light field L(u, v, x, y) of a layered, occluding scene. Shift-and-add refocuses it "
          "after capture; depth follows from the shear that makes all views agree (angular variance) or "
          "that makes the refocused image sharpest (refocus contrast).",
    equations=[r"L(u,v,\mathbf{x})=I\left(\mathbf{x}-s(\mathbf{x})\,(u,v)\right)",
               r"\bar{E}_s(\mathbf{x})=\frac{1}{N_uN_v}\sum_{u,v}L\left(u,v,\mathbf{x}+s\,(u,v)\right)",
               r"\hat{s}(\mathbf{x})=\arg\min_s\ \mathrm{Var}_{u,v}\,L\left(u,v,\mathbf{x}+s\,(u,v)\right)"],
    scene=[N(128, 64, 256), choice("scene", "scene", ("three planes", "random discs")), Param("views", "views per axis", 7, 3, 11, kind="int"),
           Param("max_disp", "max disparity (px / view)", 1.5, 0.3, 4.0), Param("noise", "noise σ", 0.005, 0.0, 0.1), SEED],
    solver=[method(DP.LF_METHODS, DP.LF_METHODS[1]), Param("slopes", "disparity candidates", 33, 5, 129, kind="int"),
            Param("window", "window (px)", 7, 3, 21, kind="int"), Param("median", "median filter (px, 0 = off)", 3, 0, 15, kind="int")],
    simulate=_sim_lf, solve=_solve_lf,
    inputs=[Input("lightfield", "light field (u, v, y, x) array or u-major view stack", "stack")], meta=("max_disp",), load=_load_lf,
    live=False, example="Three textured planes, 7×7 views"))


# ----------------------------------------------------------------------------- fringe projection
def _sim_fpp(p, rng):
    n = p["n"]
    h = p["height"] * (1 - DP.depth_scene(n, p["scene"], rng))
    shifts = 2 * np.pi * np.arange(p["steps"]) / p["steps"]
    periods = [p["period"]] if p["frequencies"] == 1 else [p["period"] * 4 ** k for k in range(p["frequencies"] - 1, -1, -1)]
    alb = 0.6 + 0.4 * F.normalize(ndi.gaussian_filter(rng.random((n, n)), 6)) if p["albedo"] else None
    sets = [WF.fpp_fringes(h, per, p["angle"], shifts, albedo=alb, noise=p["noise"], rng=rng, gamma=p["gamma"]) for per in periods]
    return {"fringes": np.concatenate(sets), "periods": np.array(periods), "steps": p["steps"], "angle": p["angle"], "truth": h}


def _load_fpp(a, m):
    st = stack_input(a["fringes"])
    periods = table_values(a["periods"]) if "periods" in a else np.array([m["period"]])
    d = {"fringes": st, "periods": periods, "steps": int(m["steps"]), "angle": m["angle"]}
    if "reference" in a:
        d["reference"] = stack_input(a["reference"])
    return d


def _phases(st, periods, steps):
    shifts = 2 * np.pi * np.arange(steps) / steps
    x = np.arange(st.shape[-1])[None, :]
    out = []
    for k, per in enumerate(periods):
        ph = WF.psi_phase(st[k * steps:(k + 1) * steps], shifts)
        out.append(F.wrap(ph - 2 * np.pi * x / per))
    return out


def _solve_fpp(d, p, mon):
    st, periods, steps = d["fringes"], np.asarray(d["periods"], float), d["steps"]
    m = p["method"]
    res = Result()
    if m.startswith("Fourier"):
        per = periods[-1]
        frame = st[(len(periods) - 1) * steps]
        w = WF.ftp_phase(frame, per)
        if "reference" in d:
            w = F.wrap(w - WF.ftp_phase(d["reference"][(len(periods) - 1) * steps], per))
        Phi = U.unwrap(w, p["unwrap"], monitor=mon)
    elif m.startswith("N-step"):
        w = _phases(st[-steps:], periods[-1:], steps)[0]
        if "reference" in d:
            w = F.wrap(w - _phases(d["reference"][-steps:], periods[-1:], steps)[0])
        Phi = U.unwrap(w, p["unwrap"], monitor=mon)
    else:
        if len(periods) < 2:
            raise ValueError("temporal unwrapping needs fringe sets at ≥ 2 periods (coarse → fine)")
        ph = _phases(st, periods, steps)
        if "reference" in d:
            ph = [F.wrap(a - b) for a, b in zip(ph, _phases(d["reference"], periods, steps))]
        w = ph[-1]
        Phi = WF.temporal_unwrap(ph, periods)
    h = WF.fpp_phase_to_height(Phi, periods[-1], d["angle"])
    mon.step(1, 1)
    res.panels["fringe image"] = inten(st[(len(periods) - 1) * steps], f"fringes (period {periods[-1]:.0f} px)")
    res.panels["wrapped phase"] = phase(w, "wrapped phase (carrier removed)")
    res.panels["height"] = depth(h, f"height ({m.split(' +')[0]})")
    if "truth" in d:
        t = d["truth"]
        off = np.median(h - t)
        hc = h - off
        res.panels["true height"] = depth(t, "true height")
        res.panels["error"] = err(hc - t, "height error (px)")
        res.metrics["height RMSE (px)"] = float(np.sqrt(np.mean((hc - t) ** 2)))
        res.metrics["unwrap failures %"] = float(100 * np.mean(np.abs(hc - t) > 0.5 * periods[-1] / np.tan(np.deg2rad(d["angle"]))))
        lv = (float(np.min(t)), float(np.max(t)))
        res.panels["true height"].levels = res.panels["height"].levels = lv
        compare(res, hc, t)
        r = t.shape[0] // 2
        x = np.arange(t.shape[1])
        res.curves.append(Curve("centre-row height", {"truth": (x, t[r]), "estimate": (x, hc[r])}, "x (px)", "height (px)", logy=False))
    return res


register(Experiment(
    key="fringe", title="Fringe projection profilometry", area=AREA,
    blurb="Sinusoidal fringes projected at an angle are phase-modulated by the surface height. Phase is "
          "demodulated by Fourier-transform profilometry (single shot) or N-step phase shifting, then "
          "unwrapped spatially or temporally across several fringe periods.",
    equations=[r"I_k=a+b\cos\left(2\pi x/p+\Delta\varphi(\mathbf{x})+\delta_k\right),\quad \Delta\varphi=\frac{2\pi h\tan\theta}{p}",
               r"\varphi=-\arctan\frac{\sum_k I_k\sin\delta_k}{\sum_k I_k\cos\delta_k}",
               r"\Phi_k=\varphi_k+2\pi\,\mathrm{round}\left(\frac{\Phi_{k-1}p_{k-1}/p_k-\varphi_k}{2\pi}\right)"],
    scene=[N(256, 64, 1024), choice("scene", "surface", SCENES), Param("height", "max height", 40.0, 1.0, 200.0, unit="px"),
           Param("period", "fringe period (finest)", 16.0, 4.0, 128.0, unit="px"), Param("angle", "projection angle", 30.0, 5.0, 60.0, unit="°"),
           Param("steps", "phase steps", 4, 3, 12, kind="int"), Param("frequencies", "fringe frequencies", 3, 1, 4, kind="int"),
           Param("gamma", "projector gamma", 1.0, 0.5, 2.5), flag("albedo", "textured albedo", True),
           Param("noise", "noise σ", 0.01, 0.0, 0.2), SEED],
    solver=[method(WF.FPP_METHODS, WF.FPP_METHODS[2]), choice("unwrap", "spatial unwrapping", U.METHODS, "quality-guided")],
    simulate=_sim_fpp, solve=_solve_fpp,
    inputs=[Input("fringes", "fringe images (period-major, step-minor)", "stack"),
            Input("periods", "fringe periods (px), coarse → fine", "table", optional=True),
            Input("reference", "reference-plane fringe images", "stack", optional=True)],
    meta=("period", "angle", "steps"), load=_load_fpp,
    example="Ramp with bumps, 3 periods × 4 steps, temporal unwrapping"))
