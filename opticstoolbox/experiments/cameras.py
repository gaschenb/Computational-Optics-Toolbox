"""Computational cameras: ghost imaging, single-pixel imaging, lensless (DiffuserCam), snapshot
hyperspectral (CASSI), coded exposure (flutter shutter), non-line-of-sight imaging."""
from __future__ import annotations

import numpy as np

from ..core import cassi as CA
from ..core import deconv as D
from ..core import fields as F
from ..core import ghost as G
from ..core import nlos as NL
from ._common import (ITERS, N, SEED, Curve, compare, Experiment, Input, Panel, Param, Result, choice, conv, depth, err, inten,
                      method, profile_curve, register, stack_input, table_values)

AREA = "Computational cameras"


def _obj(name, n, rng):
    return F.normalize(np.clip(F.test_object(name, n, rng), 0, None))


def _score(res, x, t):
    """RMSE / PSNR / SSIM and an error map, after a least-squares scale + offset fit (these modalities
    recover the object only up to an affine factor)."""
    compare(res, x, t, fit=True)


# ----------------------------------------------------------------------------- ghost imaging
def _sim_ghost(p, rng):
    n = p["n"]
    obj = _obj(p["object"], n, rng)
    M = max(1, int(round(p["ratio"] * n * n)))
    P = G.speckle_patterns(M, n, p["grain"], rng) if p["patterns"] == "pseudo-thermal speckle" else G.binary_patterns(M, n, rng)
    if p["fluct"]:      # source power drifts; the reference arm records the patterns with it
        P = P * np.clip(1 + p["fluct"] * rng.standard_normal(M), 0.05, None)[:, None, None]
    B = G.bucket(P, obj, p["noise"], rng)
    return {"patterns": P, "bucket": B, "truth": obj}


def _solve_ghost(d, p, mon):
    P, B = d["patterns"], np.asarray(d["bucket"], float)
    x = G.ghost_reconstruct(P, B, p["method"], p["lam"], p["iters"], mon)
    res = Result()
    res.panels["pattern"] = inten(P[0], "illumination pattern 1")
    res.panels["reconstruction"] = inten(x, f"{p['method']} ({len(B)} patterns)")
    res.curves.append(Curve("bucket signal", {"bucket": (np.arange(min(len(B), 500)), B[:500])}, "pattern", "counts", logy=False))
    if "truth" in d:
        t = d["truth"]
        res.panels["truth"] = inten(t, "object")
        _score(res, x, t)
    return res


register(Experiment(
    key="ghost", title="Computational ghost imaging", area=AREA,
    blurb="A single-pixel bucket detector records the total light transmitted by an object under many "
          "random illumination patterns. Correlating the bucket signal with the known patterns forms "
          "the image (classical, differential, normalised GI); pseudo-inverse and TV-regularised "
          "compressive recovery need far fewer patterns.",
    equations=[r"B_m=\sum_{\mathbf{x}}P_m(\mathbf{x})\,T(\mathbf{x})",
               r"G(\mathbf{x})=\langle B\,P(\mathbf{x})\rangle-\langle B\rangle\langle P(\mathbf{x})\rangle",
               r"G_{\mathrm{DGI}}=\langle B P\rangle-\frac{\langle B\rangle}{\langle S\rangle}\langle S P\rangle,\quad S_m=\sum_{\mathbf{x}}P_m",
               r"\min_{T}\ \frac{1}{2}\|\mathbf{P}T-B\|^2+\lambda\,\mathrm{TV}(T)"],
    scene=[N(48, 16, 128), choice("object", "object", F.objects("text")), choice("patterns", "patterns", ("pseudo-thermal speckle", "random binary (DMD)")),
           Param("ratio", "patterns / pixels", 1.0, 0.05, 20.0, log=True), Param("grain", "speckle grain (px)", 1.5, 1.0, 5.0),
           Param("fluct", "source power fluctuation (rel.)", 0.2, 0.0, 1.0),
           Param("noise", "bucket noise (rel.)", 0.01, 0.0, 0.5), SEED],
    solver=[method(G.GI_METHODS, "differential GI"), Param("lam", "TV weight", 0.02, 1e-4, 1.0, log=True), ITERS(150, 10, 2000)],
    simulate=_sim_ghost, solve=_solve_ghost,
    inputs=[Input("patterns", "illumination patterns", "stack"), Input("bucket", "bucket values (one per pattern)", "table")],
    load=lambda a, m: {"patterns": stack_input(a["patterns"]), "bucket": table_values(a["bucket"])},
    example="Text, speckle illumination, as many patterns as pixels"))


# ----------------------------------------------------------------------------- single-pixel
def _pow2(n):
    return int(2 ** np.round(np.log2(n)))


def _sim_spi(p, rng):
    n = _pow2(p["n"])
    obj = _obj(p["object"], n, rng)
    y, idx, H1 = G.hadamard_spi(obj, p["ratio"], p["ordering"], p["noise"], rng)
    Fs, mask = G.fourier_spi(obj, p["ratio"], p["noise"], rng)
    return {"y": y, "idx": idx, "H1": H1, "Fs": Fs, "mask": mask, "truth": obj, "ratio": p["ratio"]}


def _solve_spi(d, p, mon):
    m = p["method"]
    if m.startswith("Hadamard"):
        x = G.hadamard_recon(d["y"], d["idx"], d["H1"], m, p["lam"], p["iters"], mon)
        n = d["H1"].shape[0]
        C = np.zeros((n, n)); C[d["idx"][:, 0], d["idx"][:, 1]] = 1
        sampled = C
    else:
        x = G.fourier_recon(d["Fs"], d["mask"], m, p["lam"], p["iters"], mon)
        sampled = np.fft.fftshift(d["mask"]).astype(float)
    res = Result()
    res.panels["reconstruction"] = inten(x, f"{m} ({100 * d['ratio']:.0f} % sampling)")
    res.panels["sampled coefficients"] = Panel(sampled, "mask", "measured coefficients (white = sampled)",
                                               xlabel="kx index" if not m.startswith("Hadamard") else "Hadamard index u",
                                               ylabel="ky index" if not m.startswith("Hadamard") else "Hadamard index v")
    t = d["truth"]
    res.panels["truth"] = inten(t, "object")
    _score(res, x, t)
    res.curves.append(profile_curve("centre row", {"truth": t, "reconstruction": F.fit_scale(x, t)}))
    return res


register(Experiment(
    key="single_pixel", title="Single-pixel imaging (Hadamard / Fourier)", area=AREA,
    blurb="Deterministic single-pixel imaging with a DMD: differential Hadamard patterns (natural, "
          "sequency or cake-cutting order) or four-step Fourier patterns, sub-sampled to a fraction of "
          "the Nyquist count and reconstructed linearly or with TV.",
    equations=[r"y_k=\langle H_{i_k}\otimes H_{j_k},\,x\rangle=\langle P^{+}_k,x\rangle-\langle P^{-}_k,x\rangle",
               r"D(\mathbf{f})=(I_0-I_\pi)+i\,(I_{\pi/2}-I_{3\pi/2})\propto\tilde{x}(\mathbf{f})",
               r"\min_x\ \frac{1}{2}\|\mathcal{S}\mathcal{H}x-y\|^2+\lambda\,\mathrm{TV}(x)"],
    scene=[Param("n", "grid size (power of 2)", 64, 16, 256, kind="int"), choice("object", "object", F.objects("text")),
           Param("ratio", "sampling ratio", 0.15, 0.01, 1.0), choice("ordering", "Hadamard ordering", ("cake-cutting", "sequency", "natural")),
           Param("noise", "measurement noise (rel.)", 0.01, 0.0, 0.5), SEED],
    solver=[method(G.SPI_METHODS, "Hadamard + TV"), Param("lam", "TV weight", 0.01, 1e-4, 1.0, log=True), ITERS(120, 10, 1000)],
    simulate=_sim_spi, solve=_solve_spi, example="Text at 15 % sampling, cake-cutting Hadamard order"))


# ----------------------------------------------------------------------------- DiffuserCam
def _sim_lensless(p, rng):
    n = p["n"]
    psf = D.caustic_psf(n, rng, strength=p["strength"])
    obj = _obj(p["object"], n, rng)
    x = F.pad_to(obj, (2 * n, 2 * n))
    op = D.CropConv(psf)
    b = np.clip(op.A(x), 0, None)
    b = F.poisson_noise(b, p["photons"], rng)
    return {"raw": b, "psf": psf, "truth": x}


def _load_lensless(a, m):
    b = np.asarray(a["raw"], float)
    psf = np.asarray(a["psf"], float)
    if psf.shape != b.shape:
        raise ValueError("raw image and PSF must have the same size")
    if b.shape[0] != b.shape[1]:
        k = min(b.shape)
        b, psf = F.crop_to(b, (k, k)), F.crop_to(psf, (k, k))
    psf = np.clip(psf - np.median(psf), 0, None)
    return {"raw": b, "psf": psf / psf.sum()}


def _solve_lensless(d, p, mon):
    b, psf = d["raw"], d["psf"]
    n = b.shape[0]
    m = p["method"]
    hist = None
    if m.startswith("ADMM"):
        x, hist = D.diffusercam_admm(b, psf, p["tau"], iters=p["iters"], monitor=mon)
    elif m.startswith("FISTA"):
        x, hist = D.diffusercam_fista(b, psf, p["iters"], mon)
    else:
        x = F.pad_to(D.wiener(b, D.psf_otf(psf, b.shape), 1e-3), (2 * n, 2 * n))
    res = Result()
    res.panels["raw"] = inten(b, "sensor image (caustics)")
    res.panels["PSF"] = inten(psf, "PSF (diffuser caustic)")
    xc = F.crop_to(x, (n, n))
    res.panels["reconstruction"] = inten(xc, f"{m}")
    res.panels["full estimate"] = inten(x, "estimate on the extended (2n) grid")
    if "truth" in d:
        t = F.crop_to(d["truth"], (n, n))
        res.panels["truth"] = inten(t, "scene")
        _score(res, xc, t)
    if hist is not None and len(hist):
        res.curves.append(conv("data residual", {m: hist}))
    return res


register(Experiment(
    key="diffusercam", title="Lensless imaging (DiffuserCam)", area=AREA,
    blurb="A thin diffuser replaces the lens: every scene point casts a shifted caustic pattern onto the "
          "sensor. Reconstruction inverts a cropped convolution: ADMM with TV and non-negativity "
          "(splitting off the crop exactly), FISTA, or naive Wiener deconvolution for comparison.",
    equations=[r"b=\mathcal{C}(h\ast v)+n",
               r"\min_{v\geq 0}\ \frac{1}{2}\|\mathcal{C}(h\ast v)-b\|^2+\tau\|\nabla v\|_1",
               r"\xi=h\ast v,\ u=\nabla v,\ w=v\quad\mathrm{(ADMM\ splits)}"],
    scene=[N(128, 64, 512), choice("object", "scene", F.objects("cells")),
           Param("strength", "diffuser strength", 40.0, 5.0, 120.0), Param("photons", "photons / pixel", 1e4, 10, 1e7, log=True), SEED],
    solver=[method(D.LENSLESS_METHODS), ITERS(300, 10, 2000), Param("tau", "TV weight τ", 1e-4, 1e-8, 1e-2, log=True)],
    simulate=_sim_lensless, solve=_solve_lensless,
    inputs=[Input("raw", "raw sensor image"), Input("psf", "PSF (point source) image")], load=_load_lensless, live=False,
    example="Cells behind a smooth random diffuser"))


# ----------------------------------------------------------------------------- CASSI
def _sim_cassi(p, rng):
    n, L = p["n"], p["bands"]
    cube, wl, lab, spectra = CA.spectral_scene(n, L, rng, None if p["scene"] == "spectral regions" else p["scene"],
                                             (p["wl_lo"], p["wl_hi"]))
    mask = CA.coded_mask(n, p["density"], rng)
    y = CA.forward(cube, mask, p["shift"])
    if p["noise"]:
        y = y + p["noise"] * y.std() * rng.standard_normal(y.shape)
    return {"y": y, "mask": mask, "bands": L, "shift": p["shift"], "wl": wl, "truth": cube, "labels": lab}


def _load_cassi(a, m):
    y = np.asarray(a["measurement"], float)
    mask = np.asarray(a["mask"], float)
    L, s = int(m["bands"]), int(m["shift"])
    if y.shape[1] != mask.shape[1] + s * (L - 1):
        raise ValueError(f"measurement width {y.shape[1]} ≠ mask width + shift·(bands−1) = {mask.shape[1] + s * (L - 1)}")
    return {"y": y, "mask": mask, "bands": L, "shift": s, "wl": CA.wavelengths(L, m["wl_lo"], m["wl_hi"])}


def _solve_cassi(d, p, mon):
    y, mask, L, s, wl = d["y"], d["mask"], d["bands"], d["shift"], d["wl"]
    x, hist = CA.reconstruct(y, mask, L, p["method"], p["lam"], p["iters"], s, mon)
    res = Result()
    res.panels["measurement"] = inten(y, "coded, dispersed snapshot")
    res.panels["mask"] = inten(mask, "coded aperture")
    res.panels["reconstruction (RGB)"] = Panel(CA.to_rgb(x, wl), "rgb", f"reconstruction, rendered ({p['method']})")
    res.panels["bands"] = Panel(np.moveaxis(x, -1, 0), "volume", "spectral bands (slider = λ)",
                                zextent=(float(wl[0]), float(wl[-1])), zlabel="λ (nm)")
    if "truth" in d:
        t = d["truth"]
        res.panels["truth (RGB)"] = Panel(CA.to_rgb(t, wl), "rgb", "true cube, rendered")
        res.metrics["PSNR (dB)"] = F.psnr(x, t, peak=t.max())
        res.metrics["RMSE"] = float(np.sqrt(np.mean((x - t) ** 2)))
        res.metrics["SSIM (mean over bands)"] = float(np.mean([F.ssim(x[..., k], t[..., k]) for k in range(t.shape[-1])]))
        res.panels["true bands"] = Panel(np.moveaxis(t, -1, 0), "volume", "true spectral bands (slider = λ)",
                                         zextent=(float(wl[0]), float(wl[-1])), zlabel="λ (nm)")
        res.panels["error (band-mean)"] = err(x.mean(-1) - t.mean(-1), "band-mean error (estimate − truth)")
        cos = np.sum(x * t, -1) / (np.linalg.norm(x, axis=-1) * np.linalg.norm(t, axis=-1) + 1e-12)
        res.metrics["spectral angle (°)"] = float(np.degrees(np.arccos(np.clip(cos, -1, 1))).mean())
        lab = d["labels"]
        ser = {}
        for k in range(1, int(lab.max()) + 1):
            sel = lab == k
            if sel.any():
                ser[f"region {k} true"] = (wl, t[sel].mean(0))
                ser[f"region {k} est."] = (wl, x[sel].mean(0))
        res.curves.append(Curve("region spectra", ser, "wavelength (nm)", "reflectance", logy=False))
    if len(hist):
        res.curves.append(conv("measurement residual", {p["method"]: hist}))
    return res


register(Experiment(
    key="cassi", title="Snapshot hyperspectral imaging (CASSI)", area=AREA,
    blurb="Single-disperser coded-aperture snapshot spectral imaging: a random binary mask codes the "
          "scene, a prism shears each band by one pixel, and the sensor integrates over wavelength. The "
          "cube is recovered from one 2-D snapshot by GAP-TV or ADMM-TV.",
    equations=[r"y(x,y)=\sum_{\lambda}M(x-s\lambda,\,y)\,f(x-s\lambda,\,y,\,\lambda)",
               r"x_{k+1}=\mathcal{D}_{\mathrm{TV}}\left(x_k+\Phi^{T}(\Phi\Phi^{T})^{-1}(y-\Phi x_k)\right)\quad\mathrm{(GAP)}",
               r"\Phi\Phi^{T}=\mathrm{diag}\left(\sum_\lambda M_\lambda^2\right)"],
    scene=[N(64, 32, 256), choice("scene", "scene layout (materials)", ("spectral regions",) + F.OBJECTS,
                                  tip="Any sample object; its grey levels are quantised into five materials"),
           Param("bands", "spectral bands", 24, 4, 64, kind="int"), Param("shift", "dispersion (px / band)", 1, 1, 3, kind="int"),
           Param("density", "mask open fraction", 0.5, 0.1, 0.9), Param("noise", "noise (rel.)", 0.01, 0.0, 0.2),
           Param("wl_lo", "first band", 450.0, 350.0, 900.0, unit="nm"), Param("wl_hi", "last band", 650.0, 400.0, 1000.0, unit="nm"), SEED],
    solver=[method(CA.CASSI_METHODS, "GAP-TV"), Param("lam", "TV weight", 0.02, 1e-4, 0.5, log=True), ITERS(80, 5, 500)],
    simulate=_sim_cassi, solve=_solve_cassi,
    inputs=[Input("measurement", "snapshot (n × (n + s(L−1)))"), Input("mask", "coded aperture (n × n)")],
    meta=("bands", "shift", "wl_lo", "wl_hi"), load=_load_cassi,
    example="Coloured regions on texture, 24 bands"))


# ----------------------------------------------------------------------------- coded exposure
CODE_KINDS = {"optimised code (flutter shutter)": "optimised", "random code": "random", "box (conventional shutter)": "box"}


def _sim_coded(p, rng):
    n = p["n"]
    img = _obj(p["object"], n, rng)
    out = {"truth": img, "length": p["length"]}
    for name, kind in CODE_KINDS.items():
        c = D.code(kind, p["chips"], np.random.default_rng(int(p["seed"])))
        b, K = D.motion_blur_code(img, c, p["length"])
        # equal exposure per chip: the box collects more light → lower relative noise
        light = c.sum() / p["chips"]
        sig = p["noise"] / np.sqrt(max(light, 1e-3))
        out[kind] = {"blurred": b + sig * rng.standard_normal(b.shape), "K": K, "code": c}
    return out


def _solve_coded(d, p, mon):
    kind = CODE_KINDS[p["method"]]
    s = d[kind]
    x = D.deblur_1d(s["blurred"], s["K"], p["reg"])
    mon.step(1, 1)
    res = Result()
    res.panels["blurred"] = inten(s["blurred"], f"motion-blurred ({p['method']})")
    res.panels["deblurred"] = inten(x, "deblurred")
    t = d["truth"]
    res.panels["truth"] = inten(t, "sharp scene")
    _score(res, x, t)
    for name, k in CODE_KINDS.items():
        if k != kind:
            xx = D.deblur_1d(d[k]["blurred"], d[k]["K"], p["reg"])
            res.metrics[f"PSNR with {k} (dB)"] = F.psnr(F.fit_scale(xx, t), t)
    nfreq = len(s["K"]) // 2
    f = np.arange(nfreq) / len(s["K"])
    res.curves.append(Curve("blur MTF |K(f)| (zeros = lost frequencies)",
                            {k: (f, np.abs(d[k]["K"][:nfreq]) + 1e-6) for k in CODE_KINDS.values()},
                            "spatial frequency (cycles / px)", "|K|", logy=True))
    c = s["code"]
    res.curves.append(Curve("shutter code", {kind: (np.arange(len(c)), c)}, "chip", "open", logy=False))
    return res


register(Experiment(
    key="coded_exposure", title="Coded exposure (flutter shutter)", area=AREA,
    blurb="Fluttering the shutter open and closed during the exposure with a broadband binary code turns "
          "motion blur into a well-conditioned convolution, whose MTF has no zeros; a conventional box "
          "shutter destroys frequencies outright.",
    equations=[r"b(x)=\int c(t)\,I(x-vt)\,dt=(k_c\ast I)(x)",
               r"c^{\star}=\arg\max_{c\in\{0,1\}^L}\ \min_f\,|\tilde{c}(f)|",
               r"\hat{I}=\mathcal{F}^{-1}\left[\frac{\tilde{K}^{*}\tilde{B}}{|\tilde{K}|^2+\epsilon}\right]"],
    scene=[N(256, 64, 1024), choice("object", "scene", F.objects("usaf")),
           Param("length", "blur length", 40.0, 4.0, 200.0, unit="px"), Param("chips", "code length (chips)", 52, 8, 128, kind="int"),
           Param("noise", "noise σ", 0.003, 0.0, 0.1), SEED],
    solver=[method(tuple(CODE_KINDS), label="shutter"), Param("reg", "regularisation ε", 1e-3, 1e-7, 1.0, log=True)],
    simulate=_sim_coded, solve=_solve_coded, example="USAF target, 40 px motion, 52-chip code"))


# ----------------------------------------------------------------------------- NLOS
def _sim_nlos(p, rng):
    n, bins, bps = p["n"], p["bins"], p["bin_ps"]
    zmax = bins * bps * 1e-12 * NL.C / 2
    vol, zs = NL.hidden_scene(n, 64, p["wall"], (0.35 * zmax, 0.8 * zmax), p["scene"], rng)
    tau = NL.simulate_transient(vol, zs, p["wall"], bins, bps, p["jitter"], p["photons"] or None, p["background"], rng)
    return {"transient": tau, "wall": p["wall"], "bin_ps": bps, "truth": vol, "zs": zs}


def _solve_nlos(d, p, mon):
    tau = d["transient"]
    rec = NL.nlos_reconstruct(tau, p["method"], d["wall"], d["bin_ps"], p["snr"])
    mon.step(1, 1)
    rec = rec / max(rec.max(), 1e-20)
    front, dep, top = NL.projections(rec)
    z = NL._zgrid(tau.shape[0], d["bin_ps"])
    res = Result()
    n = tau.shape[1]
    w2 = d["wall"] / 2
    tmax = tau.shape[0] * d["bin_ps"]
    res.panels["transient"] = Panel(tau[:, n // 2, :], "intensity", "transient τ(t, x) at the centre row",
                                    extent=(-w2, w2, 0, tmax), xlabel="wall x (m)", ylabel="time (ps)", aspect=False)
    res.panels["front view"] = Panel(front, "intensity", f"front view ({p['method'].split(' (')[0]})",
                                     extent=(-w2, w2, -w2, w2), unit="m")
    res.panels["depth"] = depth(np.where(np.isnan(dep), np.nan, z[np.nan_to_num(dep).astype(int)]), "depth (m)",
                                extent=(-w2, w2, -w2, w2), unit="m")
    res.panels["top view"] = Panel(top, "intensity", "top view (depth × x)", extent=(-w2, w2, z[0], z[-1]),
                                   xlabel="x (m)", ylabel="depth z (m)", aspect=False)
    zv = dict(zextent=(float(z[0]), float(z[-1])), zlabel="depth z (m)")
    res.panels["volume"] = Panel(rec, "volume", "reconstructed volume (slider = depth)", extent=(-w2, w2, -w2, w2), unit="m", **zv)
    if "truth" in d:
        tf, td, tt = NL.projections(d["truth"])
        res.panels["true front"] = Panel(tf, "intensity", "hidden scene (front)", extent=(-w2, w2, -w2, w2), unit="m")
        # the true volume resampled onto the reconstruction's depth bins, so the two sliders line up
        tv = np.zeros_like(rec)
        for k, zk in enumerate(d["zs"]):
            i = int(np.argmin(np.abs(z - zk)))
            tv[i] = np.maximum(tv[i], F.resize(d["truth"][k], rec.shape[1:]) if d["truth"].shape[1:] != rec.shape[1:] else d["truth"][k])
        res.panels["true volume"] = Panel(tv / max(tv.max(), 1e-20), "volume", "hidden scene (slider = depth)",
                                          extent=(-w2, w2, -w2, w2), unit="m", **zv)
        fr = F.resize(front, tf.shape) if front.shape != tf.shape else front
        res.metrics["front-view correlation"] = float(np.corrcoef(fr.ravel(), tf.ravel())[0, 1])
        compare(res, fr, tf, prefix="front-view", fit=True, error_panel=False)
        zt = d["zs"]
        tz = np.where(np.isnan(td), np.nan, zt[np.nan_to_num(td).astype(int)])
        ez = np.where(np.isnan(dep), np.nan, z[np.nan_to_num(dep).astype(int)])
        ok = ~np.isnan(tz) & ~np.isnan(ez)
        if ok.any():
            res.metrics["depth error (cm)"] = float(100 * np.median(np.abs(ez[ok] - tz[ok])))
    t = np.arange(tau.shape[0]) * d["bin_ps"]
    res.curves.append(Curve("total photon counts vs time", {"Σ τ": (t, tau.sum((1, 2)))}, "time (ps)", "counts", logy=True))
    return res


register(Experiment(
    key="nlos", title="Non-line-of-sight imaging", area=AREA,
    blurb="Confocal NLOS: a pulsed laser and SPAD scan a visible relay wall; third-bounce photons from a "
          "hidden scene arrive with time-of-flight 2r/c. The hidden volume is reconstructed with the "
          "light-cone transform (a 3-D deconvolution after resampling t → t²), f–k migration (wave-based) "
          "or filtered back-projection.",
    equations=[r"\tau(x',y',t)=\iiint\frac{\rho(x,y,z)}{r^4}\,\delta\left(2r-ct\right)dx\,dy\,dz,\quad r=\sqrt{(x'-x)^2+(y'-y)^2+z^2}",
               r"v^{3/2}\,\tau\left(\frac{2\sqrt{v}}{c}\right)=h\ast\mathcal{R}\{\rho\},\quad h=\delta(x^2+y^2-v)\quad\mathrm{(LCT)}",
               r"\tilde{\phi}(k_x,k_y,k_z)=\frac{c\,k_z}{\sqrt{k_x^2+k_y^2+k_z^2}}\,\tilde{\tau}\left(k_x,k_y,\frac{c}{2}\sqrt{k_x^2+k_y^2+k_z^2}\right)\quad\mathrm{(f–k)}"],
    scene=[Param("n", "wall scan points per side", 32, 16, 64, kind="int"), Param("wall", "wall scan size", 1.0, 0.3, 3.0, unit="m"),
           choice("scene", "hidden scene", ("letters", "patches")), Param("bins", "time bins", 256, 64, 1024, kind="int"),
           Param("bin_ps", "bin width", 32.0, 4.0, 128.0, unit="ps"), Param("jitter", "SPAD jitter", 50.0, 0.0, 300.0, unit="ps"),
           Param("photons", "signal photons (0 = noiseless)", 1e5, 0.0, 1e8), Param("background", "background photons", 1.0, 0.0, 1e3), SEED],
    solver=[method(NL.NLOS_METHODS), Param("snr", "LCT Wiener SNR⁻¹", 0.1, 1e-4, 10.0, log=True)],
    simulate=_sim_nlos, solve=_solve_nlos,
    inputs=[Input("transient", "transient cube (t, y, x)", "stack")], meta=("wall", "bin_ps"),
    load=lambda a, m: {"transient": np.asarray(a["transient"], float), "wall": m["wall"], "bin_ps": m["bin_ps"]}, live=False,
    example="Letters N-L-S at three depths behind a 1 m relay wall"))
