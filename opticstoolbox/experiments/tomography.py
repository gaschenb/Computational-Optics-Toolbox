"""Tomography experiments: 2-D and 3-D projection tomography, optical diffraction tomography."""
from __future__ import annotations

import numpy as np

from ..core import fields as F
from ..core import tomography as T
from ._common import (ITERS, N, SEED, WL, Curve, compare, extent, Experiment, Input, Panel, Param, Result, choice, conv, err, flag, inten,
                      method, register, signed, table_values)

AREA = "Tomography"
CT_METHODS = tuple(f"FBP ({f})" for f in T.FBP_FILTERS) + ("SIRT", "OS-SART")


def _angles(p):
    return np.linspace(0, p["range"], p["angles"], endpoint=False)


def _sim_ct(p, rng):
    n = p["n"]
    img = F.test_object(p["object"], n, rng) if p["object"] != "shepp-logan" else F.shepp_logan(n)
    ang = _angles(p)
    s = T.radon(img, ang)
    if p["photons"]:
        I0 = p["photons"]
        mu = s / max(s.max(), 1e-12) * 3.0                  # Beer–Lambert with max attenuation e⁻³
        counts = rng.poisson(I0 * np.exp(-mu))
        s = -np.log(np.maximum(counts, 1) / I0) * max(s.max(), 1e-12) / 3.0
    return {"sinogram": s, "angles": ang, "truth": img}


def _ct_solve_core(sino, ang, p, mon):
    m = p["method"]
    if m.startswith("FBP"):
        return T.fbp(sino, ang, m[5:-1], p["cutoff"]), None
    subsets = 1 if m == "SIRT" else p["subsets"]
    return T.sirt(sino, ang, p["iters"], nonneg=p["nonneg"], subsets=subsets, monitor=mon)


def _solve_ct(d, p, mon):
    sino, ang = d["sinogram"], d["angles"]
    rec, hist = _ct_solve_core(sino, ang, p, mon)
    res = Result()
    res.panels["sinogram"] = Panel(sino.T, "intensity", "sinogram", extent=(ang[0], ang[-1], 0, sino.shape[1]),
                                   xlabel="projection angle θ (°)", ylabel="detector position s (px)", aspect=False)
    res.panels["reconstruction"] = inten(rec, f"reconstruction ({p['method']})")
    if "truth" in d:
        t = d["truth"]
        res.panels["truth"] = inten(t, "truth", levels=(float(t.min()), float(t.max())))
        res.panels["reconstruction"].levels = res.panels["truth"].levels
        res.panels["error"] = err(rec - t, "error (reconstruction − truth)")
        res.metrics["PSNR (dB)"] = F.psnr(rec, t)
        res.metrics["SSIM"] = F.ssim(rec, t)
        compare(res, rec, t)
    if hist is not None:
        res.curves.append(conv("projection residual", {p["method"]: hist}))
    return res


SCENE_CT = [N(128, 32, 512), choice("object", "phantom", F.objects("shepp-logan")),
            Param("angles", "projections", 90, 4, 720, kind="int"), Param("range", "angular range", 180.0, 30.0, 360.0, unit="°"),
            Param("photons", "incident photons (0 = noiseless)", 1e5, 0.0, 1e8), SEED]
SOLVER_CT = [method(CT_METHODS, "FBP (Shepp-Logan)"), Param("cutoff", "filter cutoff", 1.0, 0.2, 1.0),
             ITERS(30, 1, 500), Param("subsets", "OS-SART subsets", 10, 2, 60, kind="int"), flag("nonneg", "non-negativity", True)]

register(Experiment(
    key="ct", title="Projection tomography (2-D)", area=AREA,
    blurb="Parallel-beam CT with an exactly matched projector / back-projector pair: filtered back-projection with "
          "five windows, SIRT and ordered-subsets SART. Reduce the angular range to see limited-angle artefacts.",
    equations=[r"p(\theta,s)=\int f(s\,\boldsymbol{\theta}+t\,\boldsymbol{\theta}^{\perp})\,dt",
               r"f=\frac{\pi}{2K}\sum_k (p_{\theta_k}\ast h)(\mathbf{x}\cdot\boldsymbol{\theta}_k),\quad \hat{h}=|\omega|\,W(\omega)",
               r"x\leftarrow x+\lambda\,C A^{T}R\,(b-Ax)\quad\mathrm{(SIRT)}"],
    scene=SCENE_CT, solver=SOLVER_CT, simulate=_sim_ct, solve=_solve_ct,
    inputs=[Input("sinogram", "sinogram (angles × detector)"), Input("angles", "angles (degrees)", "table", optional=True)],
    meta=("range",),
    load=lambda a, m: {"sinogram": np.asarray(a["sinogram"], float),
                       "angles": table_values(a["angles"]) if "angles" in a else np.linspace(0, m["range"], np.asarray(a["sinogram"]).shape[0], endpoint=False)},
    example="Shepp–Logan, 90 views, low-dose noise"))


# ----------------------------------------------------------------------------- 3-D projection CT
def _sim_ct3d(p, rng):
    n = p["n"]
    vol = T.phantom_3d(p["phantom"], n, rng, p["object"], dn_cell=0.3, dn_nuc=0.6, beads=5, dn_bead=1.0)
    ang = _angles(p)
    # rotation axis = y (axis 1 of (z, y, x)): each y-slice is an independent (z, x) slice
    slices = np.moveaxis(vol, 1, 0)                     # (y, z, x)
    s = T.radon(slices, ang)                            # (y, A, x)
    if p["noise"]:
        s = s + p["noise"] * s.std() * rng.standard_normal(s.shape)
    return {"projections": np.moveaxis(s, 0, 1), "angles": ang, "truth": vol}  # (A, y, x)


def _solve_ct3d(d, p, mon):
    proj, ang = d["projections"], d["angles"]
    sino = np.moveaxis(proj, 1, 0)                      # (y, A, x)
    rec, hist = _ct_solve_core(sino, ang, p, mon)
    vol = np.moveaxis(rec, 0, 1)                        # (z, y, x)
    res = Result()
    res.panels["projection 0°"] = inten(proj[0], f"projection at {ang[0]:.0f}°", xlabel="x (px)", ylabel="y (px, rotation axis)")
    res.panels["reconstruction"] = Panel(vol, "volume", "reconstruction, x–y (slider = z)", xlabel="x (px)", ylabel="y (px)",
                                         zlabel="z (px)")
    res.panels["xz slice"] = inten(vol[:, vol.shape[1] // 2], "reconstruction, x–z slice", xlabel="x (px)", ylabel="z (px)")
    if "truth" in d:
        t = d["truth"]
        lv = (float(t.min()), float(t.max()))
        res.panels["truth"] = Panel(t, "volume", "truth, x–y (slider = z)", xlabel="x (px)", ylabel="y (px)", levels=lv,
                                    zlabel="z (px)")
        res.panels["reconstruction"].levels = lv
        res.panels["truth xz"] = inten(t[:, t.shape[1] // 2], "truth, x–z slice", xlabel="x (px)", ylabel="z (px)", levels=lv)
        res.panels["xz slice"].levels = lv
        res.metrics["PSNR (dB)"] = F.psnr(vol, t)
        res.metrics["correlation"] = float(np.corrcoef(vol.ravel(), t.ravel())[0, 1])
        compare(res, vol, t, xlabel="x (px)", ylabel="y (px)")
    if hist is not None:
        res.curves.append(conv("projection residual", {p["method"]: hist}))
    return res


register(Experiment(
    key="ct3d", title="Projection tomography (3-D)", area=AREA,
    blurb="Parallel-beam 3-D tomography about the y axis, reconstructed slice-by-slice (all slices at once) with "
          "FBP, SIRT or OS-SART.",
    equations=[r"p_\theta(x,y)=\int f(x\cos\theta+t\sin\theta,\ y,\ -x\sin\theta+t\cos\theta)\,dt"],
    scene=[N(64, 32, 192), choice("phantom", "phantom", T.PHANTOMS_3D), choice("object", "object (for the sphere phantom)", F.objects("usaf")),
           Param("angles", "projections", 90, 4, 720, kind="int"),
           Param("range", "angular range", 180.0, 30.0, 360.0, unit="°"), Param("noise", "noise (rel. σ)", 0.01, 0.0, 0.5), SEED],
    solver=SOLVER_CT, simulate=_sim_ct3d, solve=_solve_ct3d,
    inputs=[Input("projections", "projections (angle, y, x) stack", "stack"), Input("angles", "angles (degrees)", "table", optional=True)],
    meta=("range",),
    load=lambda a, m: {"projections": np.asarray(a["projections"], float),
                       "angles": table_values(a["angles"]) if "angles" in a else np.linspace(0, m["range"], len(a["projections"]), endpoint=False)},
    example="Cell volume, 90 views about the y axis"))


# ----------------------------------------------------------------------------- ODT
def _sim_odt(p, rng):
    n, dx, wl, nm = p["n"], p["dx"], p["wl"], p["n_medium"]
    dn = p["dn"] * T.phantom_3d(p["phantom"], n, rng, p["object"], dn_cell=0.4, dn_nuc=0.75, dn_bead=1.0)
    if p["geometry"] == "illumination scanning":
        fins = T.illumination_angles(p["views"], p["na_ill"], nm, wl, dx, n)
        us, uins = T.simulate_odt(dn, wl, dx, nm, fins, p["na_obj"], noise=p["noise"], rng=rng)
        angs = None
    else:
        fins = None
        angs = np.linspace(0, 360 if p["full_rotation"] else 180, p["views"], endpoint=False)
        us, uins = T.simulate_odt(dn, wl, dx, nm, None, p["na_obj"], "sample rotation", angs, noise=p["noise"], rng=rng)
    return {"fields": us, "incident": uins, "fins": fins, "angles": angs, "truth": dn, "wl": wl, "dx": dx,
            "n_medium": nm, "na_obj": p["na_obj"], "geometry": p["geometry"]}


def _load_odt(arr, m):
    us = np.asarray(arr["fields"])
    if not np.iscomplexobj(us):
        raise ValueError("ODT needs complex fields (.npy / .mat with complex values)")
    uins = np.asarray(arr["incident"]) if "incident" in arr else None
    fins = np.asarray(arr["illumination"], float).reshape(-1, 2) if "illumination" in arr else None
    if uins is None:
        if fins is None:
            raise ValueError("give either background fields or illumination frequencies")
        n = us.shape[-1]
        y = (np.arange(n) - n // 2) * m["dx"]
        Yg, Xg = np.meshgrid(y, y, indexing="ij")
        uins = np.array([np.exp(2j * np.pi * (f[0] * Yg + f[1] * Xg)) for f in fins])
    if fins is None:          # estimate each illumination from the background-field phase ramp
        fins = []
        n = us.shape[-1]
        for u in uins:
            S = np.abs(np.fft.fft2(u))
            iy, ix = np.unravel_index(np.argmax(S), S.shape)
            fy = np.fft.fftfreq(n, m["dx"])[iy]; fx = np.fft.fftfreq(n, m["dx"])[ix]
            fins.append((fy, fx))
        fins = np.array(fins)
    return {"fields": us, "incident": uins, "fins": fins, "wl": m["wl"], "dx": m["dx"], "n_medium": m["n_medium"],
            "na_obj": m["na_obj"], "geometry": "illumination scanning"}


def _solve_odt(d, p, mon):
    rec, cnt = T.odt_reconstruct(d["fields"], d["incident"], d["wl"], d["dx"], d["n_medium"], d.get("fins"), d.get("angles"),
                                 p["method"].split()[0], d["na_obj"], d["geometry"], p["gp_iters"], p["nonneg"], monitor=mon)
    res = Result()
    nz = rec.shape[0]
    n = rec.shape[-1]
    ex = extent(n, d["dx"])
    exz = (ex[0], ex[1], -nz * d["dx"] / 2, nz * d["dx"] / 2)
    res.panels["field phase (view 1)"] = Panel(np.angle(d["fields"][0] / d["incident"][0]), "phase", "phase of u/u_in, view 1",
                                               extent=ex, unit="µm")
    zv = dict(zextent=(-(nz - 1) * d["dx"] / 2, (nz - 1) * d["dx"] / 2), zlabel="z (µm)")
    res.panels["reconstruction Δn"] = Panel(rec, "volume", f"Δn ({p['method']}), x–y (slider = z)", extent=ex, unit="µm", **zv)
    res.panels["x–z slice"] = signed(rec[:, rec.shape[1] // 2], "Δn, x–z slice (missing cone)", extent=exz,
                                     xlabel="x (µm)", ylabel="z (µm)")
    kk = cnt[:, cnt.shape[1] // 2] > 0
    kmax = 1.0 / (2 * d["dx"])
    res.panels["k-space coverage"] = Panel(kk.astype(float), "intensity", "k-space samples (k_x–k_z slice)",
                                           extent=(-kmax, kmax, -kmax, kmax), xlabel="k_x (1/µm)", ylabel="k_z (1/µm)")
    if "truth" in d:
        t = d["truth"]
        lv = (float(min(t.min(), 0)), float(t.max()))
        res.panels["truth Δn"] = Panel(t, "volume", "true Δn, x–y (slider = z)", extent=ex, unit="µm", levels=lv, **zv)
        res.panels["reconstruction Δn"].levels = lv
        res.panels["truth x–z"] = signed(t[:, t.shape[1] // 2], "true Δn, x–z slice", extent=exz, xlabel="x (µm)", ylabel="z (µm)")
        compare(res, rec, t, metric_unit=" Δn", extent=ex, unit="µm")
        res.metrics["correlation"] = float(np.corrcoef(rec.ravel(), t.ravel())[0, 1])
        res.metrics["peak Δn (rec / true)"] = float(rec.max() / t.max())
        z = np.arange(nz)
        c = rec.shape[1] // 2
        res.curves.append(Curve("axial profile through the centre", {"truth": (z, t[:, c, c]), "reconstruction": (z, rec[:, c, c])},
                                "z (voxels)", "Δn", logy=False))
    return res


register(Experiment(
    key="odt", title="Optical diffraction tomography (3-D RI)", area=AREA,
    blurb="Refractive-index tomography of a cell from complex fields (e.g. from off-axis DHM) under many "
          "illumination angles or sample rotations. Fields are simulated by multislice beam propagation; the "
          "inversion maps each field onto its Ewald cap (Fourier diffraction theorem) with the Born or Rytov "
          "approximation, and Gerchberg–Papoulis non-negativity iterations fill the missing cone.",
    equations=[r"f(\mathbf{r})=k_0^2\left(n^2(\mathbf{r})-n_m^2\right)",
               r"\tilde{F}(\mathbf{K}-\mathbf{k}_{\mathrm{in}})=-2i\gamma\,\tilde{U}_s(\alpha,\beta;0),\quad \gamma=\sqrt{k_m^2-\alpha^2-\beta^2}",
               r"\mathrm{Born}:\ u_s=u-u_{\mathrm{in}},\qquad \mathrm{Rytov}:\ u_s=u_{\mathrm{in}}\ln(u/u_{\mathrm{in}})"],
    scene=[Param("n", "voxels per side", 64, 32, 128, kind="int"), Param("dx", "voxel size", 0.2, 0.05, 1.0, unit="µm"), WL(0.532),
           Param("n_medium", "medium index", 1.33, 1.0, 1.6), Param("dn", "max Δn", 0.04, 0.001, 0.2),
           choice("phantom", "phantom", T.PHANTOMS_3D), choice("object", "object (for the sphere phantom)", F.objects("cells")),
           choice("geometry", "geometry", ("illumination scanning", "sample rotation")), Param("views", "views", 60, 4, 400, kind="int"),
           Param("na_ill", "illumination NA", 1.0, 0.1, 1.3), Param("na_obj", "objective NA", 1.2, 0.2, 1.33),
           flag("full_rotation", "360° rotation", False), Param("noise", "field noise σ", 0.0, 0.0, 0.2), SEED],
    solver=[method(("Rytov", "Born")), Param("gp_iters", "positivity iterations", 20, 0, 200, kind="int"),
            flag("nonneg", "Δn ≥ 0", True)],
    simulate=_sim_odt, solve=_solve_odt,
    inputs=[Input("fields", "complex sample fields (views, y, x)", "stack"), Input("incident", "complex background fields", "stack", optional=True),
            Input("illumination", "illumination frequencies (f_y, f_x) in 1/µm", "table", optional=True)],
    meta=("wl", "dx", "n_medium", "na_obj"), load=_load_odt, live=False,
    example="64³ cell, 60 illuminations within NA 1.0, Rytov + positivity"))
