"""Microscopy & sensing: deconvolution, structured illumination, Shack–Hartmann wavefront sensing,
imaging through scattering media, two-photon excitation microscopy."""
from __future__ import annotations

import numpy as np

from ..core import deconv as D
from ..core import fields as F
from ..core import photon as PH
from ..core import scattering as SC
from ..core import sim as S
from ..core import wavefront as WF
from ..core.waveoptics import ZERNIKE_NAMES, incoherent_psf, pupil, radial_profile
from ._common import (DX, ITERS, N, NOISE, SEED, WL, Curve, compare, freq_extent, extent, Experiment, Input, Panel, Param, Result, choice, conv, err,
                      flag, inten, logspec, method, profile_curve, register, signed, stack_input)

AREA = "Microscopy & sensing"


# ----------------------------------------------------------------------------- deconvolution
PSF_KINDS = ("Gaussian", "defocus (diffraction)", "motion", "disk")


def _make_psf(kind, n, size):
    if kind == "Gaussian":
        return D.gaussian_psf(n, size)
    if kind == "motion":
        return D.motion_psf(n, 4 * size, 30.0)
    if kind == "disk":
        return D.disk_psf(n, 1.5 * size)
    na = min(0.21 * 0.5 / (size * 0.1), 0.95)        # in-focus Airy σ ≈ size px (λ = 0.5 µm, dx = 0.1 µm)
    h = incoherent_psf(pupil((n, n), 0.5, 0.1, na, {4: 0.5}))
    return h / h.sum()


def _sim_deconv(p, rng):
    n = p["n"]
    x = np.clip(F.test_object(p["object"], n, rng), 0, None)
    psf = _make_psf(p["psf"], n, p["psf_size"])
    H = D.psf_otf(psf, (n, n))
    b = np.clip(D.convolve(x, H), 0, None) + p["background"]
    b = F.poisson_noise(b, p["photons"], rng)
    return {"blurred": b, "psf": psf, "truth": x + p["background"]}


def _load_deconv(a, m):
    b = np.asarray(a["blurred"], float)
    n = b.shape
    psf = np.asarray(a["psf"], float) if "psf" in a else None
    if psf is not None:
        psf = F.pad_to(psf / psf.sum(), n) if psf.shape != n else psf / psf.sum()
    else:
        psf = D.gaussian_psf(n[0], m["psf_size"]) if n[0] == n[1] else None
        if psf is None:
            raise ValueError("give a PSF image for non-square data")
    return {"blurred": b, "psf": psf}


def _solve_deconv(d, p, mon):
    b, psf = d["blurred"], d["psf"]
    H = D.psf_otf(psf, b.shape)
    m = p["method"]
    hist = None
    est_psf = None
    if m == "Wiener":
        x = D.wiener(b, H, p["nsr"])
    elif m == "Richardson–Lucy":
        x, hist = D.richardson_lucy(b, H, p["iters"], monitor=mon)
    elif m == "Richardson–Lucy + TV":
        x, hist = D.richardson_lucy(b, H, p["iters"], tv=p["tv"], monitor=mon)
    elif m == "TV (ADMM)":
        x, hist = D.tv_admm(b / b.max(), H, p["lam"], iters=p["iters"], monitor=mon)
        x = x * b.max()
    else:
        x, est_psf, hist = D.blind_rl(b, p["blind_size"], max(p["iters"] // 3, 5), monitor=mon)
    res = Result()
    res.panels["measured"] = inten(b, "measured (blurred + noise)")
    res.panels["deconvolved"] = inten(x, f"{m}")
    c = b.shape[0] // 2
    w = 24
    res.panels["PSF"] = inten(np.fft.fftshift(np.real(np.fft.ifft2(H)))[c - w:c + w, c - w:c + w] if psf.shape == b.shape
                              else psf, "PSF (centre, 48 × 48 px)", extent=(-w, w, -w, w), xlabel="x (px)", ylabel="y (px)")
    if est_psf is not None:
        res.panels["estimated PSF"] = inten(est_psf[c - w:c + w, c - w:c + w], "estimated PSF (blind)")
    if "truth" in d:
        t = d["truth"]
        res.panels["truth"] = inten(t, "truth")
        res.metrics["PSNR blurred (dB)"] = F.psnr(F.fit_scale(b, t), t)
        res.metrics["PSNR (dB)"] = F.psnr(F.fit_scale(x, t), t)
        res.metrics["SSIM"] = F.ssim(F.fit_scale(x, t), t)
        res.panels["truth"].levels = (float(t.min()), float(t.max()))
        compare(res, x, t, fit=True)
        res.curves.append(profile_curve("centre row", {"truth": t, "blurred": b, "deconvolved": F.fit_scale(x, t)}))
    if hist is not None:
        res.curves.append(conv("data residual", {m: hist}))
    return res


register(Experiment(
    key="deconvolution", title="Deconvolution (blind and non-blind)", area=AREA,
    blurb="Restore a blurred, photon-noisy image with a known PSF (Wiener, Richardson–Lucy, RL with TV, "
          "TV-ADMM) or estimate the PSF jointly (blind RL).",
    equations=[r"b=\mathrm{Poisson}(h\ast x)",
               r"x_{k+1}=x_k\cdot\left[h^{T}\ast\frac{b}{h\ast x_k}\right]\quad\mathrm{(Richardson–Lucy)}",
               r"\hat{X}=\frac{\hat{H}^{*}\hat{B}}{|\hat{H}|^2+\mathrm{NSR}}\quad\mathrm{(Wiener)}",
               r"\min_x\ \frac{1}{2}\|h\ast x-b\|^2+\lambda\|\nabla x\|_1"],
    scene=[N(256, 64, 1024), choice("object", "object", F.objects("cells")),
           choice("psf", "PSF", PSF_KINDS), Param("psf_size", "PSF size (σ px)", 2.5, 0.5, 10.0),
           NOISE(300, 5, 1e6), Param("background", "background", 0.0, 0.0, 0.5), SEED],
    solver=[method(D.DECONV_METHODS, "Richardson–Lucy"), ITERS(40, 1, 1000), Param("nsr", "Wiener NSR", 1e-2, 1e-5, 1.0, log=True),
            Param("tv", "RL-TV weight", 2e-3, 0.0, 0.05), Param("lam", "ADMM TV weight λ", 3e-3, 1e-5, 0.1, log=True),
            Param("blind_size", "blind PSF support (px)", 21, 5, 63, kind="int")],
    simulate=_sim_deconv, solve=_solve_deconv,
    inputs=[Input("blurred", "blurred image"), Input("psf", "PSF image (centred)", optional=True)],
    meta=("psf_size",), load=_load_deconv,
    example="Cells blurred by a σ = 2.5 px Gaussian at 300 photons/px"))


# ----------------------------------------------------------------------------- SIM
def _sim_sim(p, rng):
    n, dx, wl, na = p["n"], p["dx"], p["wl"], p["na"]
    obj = np.clip(F.test_object(p["object"], n, rng), 0, None)
    ang = tuple(7 + 180 / p["angles"] * i for i in range(p["angles"]))
    raw, psf, H = S.simulate_sim(obj, wl, dx, na, p["k_frac"], ang, p["n_phases"], p["m"], p["photons"], rng)
    fc = 2 * na / wl
    ks = np.array([[p["k_frac"] * fc * np.cos(np.deg2rad(a)), p["k_frac"] * fc * np.sin(np.deg2rad(a))] for a in ang])
    phs = np.tile(2 * np.pi * np.arange(p["n_phases"]) / p["n_phases"], (len(ang), 1))
    return {"raw": raw, "wl": wl, "dx": dx, "na": na, "truth": obj, "ks": ks, "phases": phs, "m": p["m"]}


def _load_sim(a, m):
    r = stack_input(a["raw"])
    P = int(m["n_phases"])
    if r.shape[0] % P:
        raise ValueError(f"{r.shape[0]} frames is not a multiple of {P} phases")
    return {"raw": r.reshape(r.shape[0] // P, P, *r.shape[1:]), "wl": m["wl"], "dx": m["dx"], "na": m["na"]}


def _solve_sim(d, p, mon):
    raw, wl, dx, na = d["raw"], d["wl"], d["dx"], d["na"]
    m = p["method"]
    _, H = S.sim_otf(raw.shape[-2:], wl, dx, na)
    info = {}
    if m.startswith("Wiener SIM"):
        if "true" in m and "ks" in d:
            img, info = S.reconstruct_sim(raw, dx, wl, na, d["ks"], d["phases"], None, p["w"], p["apod"], H)
        else:
            img, info = S.reconstruct_sim(raw, dx, wl, na, None, None, None, p["w"], p["apod"], H)
    elif m == "widefield":
        img = S.widefield(raw)
    else:
        img = S.wiener_widefield(raw, H, p["w"], dx, wl, na)
    mon.step(1, 1)
    res = Result()
    wf = S.widefield(raw)
    n = img.shape[-1]
    ex = extent(n, dx)
    res.panels["raw frame"] = inten(raw[0, 0], "raw frame (angle 1, phase 1)", extent=ex, unit="µm")
    res.panels["widefield"] = inten(wf, "widefield (sum of frames)", extent=ex, unit="µm")
    res.panels["reconstruction"] = inten(img, m, extent=ex, unit="µm")
    res.panels["spectrum"] = Panel(info["spectrum"] if "spectrum" in info else np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(img)))),
                                   "intensity", "reconstruction spectrum (log)", extent=freq_extent(n, dx),
                                   xlabel="f_x (1/µm)", ylabel="f_y (1/µm)")
    if "truth" in d:
        t = d["truth"]
        res.panels["truth"] = inten(t, "truth", extent=ex, unit="µm")
        res.metrics["PSNR widefield (dB)"] = F.psnr(F.fit_scale(wf, t), t)
        res.metrics["PSNR (dB)"] = F.psnr(F.fit_scale(img, t), t)
        compare(res, img, t, fit=True, extent=ex, unit="µm")
    if "k" in info:
        for i, (k, mo) in enumerate(zip(info["k"], info["modulation"])):
            res.metrics[f"pattern {i + 1}: |k| (1/µm), m"] = f"{np.hypot(*k):.3f}, {mo:.2f}"
    fr = np.fft.fftshift(np.fft.fftfreq(raw.shape[-1], dx))
    rp = {"widefield": radial_profile(np.abs(np.fft.fftshift(np.fft.fft2(wf)))),
          "reconstruction": radial_profile(np.abs(np.fft.fftshift(np.fft.fft2(img))))}
    fx = fr[raw.shape[-1] // 2:][: len(rp["widefield"])]
    res.curves.append(Curve("radially averaged spectrum", {k: (fx[: len(v)], v[: len(fx)]) for k, v in rp.items()},
                            "spatial frequency (1/µm)", "|spectrum|", logy=True))
    return res


register(Experiment(
    key="sim", title="Structured illumination microscopy", area=AREA,
    blurb="2-D linear SIM: sinusoidal illumination at several angles and phases shifts high-frequency "
          "object content into the OTF passband. Pattern vectors, phases and modulation depth are "
          "estimated from the raw data; generalised Wiener recombination doubles the resolution.",
    equations=[r"D_{a,p}(\mathbf{f})=\tilde{H}(\mathbf{f})\left[\tilde{S}(\mathbf{f})+\frac{m}{2}e^{i\phi_p}\tilde{S}(\mathbf{f}-\mathbf{k}_a)+\frac{m}{2}e^{-i\phi_p}\tilde{S}(\mathbf{f}+\mathbf{k}_a)\right]",
               r"\hat{S}=\frac{\sum_{a,b}\tilde{H}^{*}_{b}\,\tilde{D}_{a,b}}{\sum_{a,b}|\tilde{H}_{b}|^2+w^2}\,A(\mathbf{f})"],
    scene=[N(256, 128, 512), DX(0.04, 0.01, 0.2), WL(0.52), Param("na", "NA", 1.2, 0.2, 1.49),
           choice("object", "object", F.objects("siemens")),
           Param("angles", "pattern angles", 3, 1, 6, kind="int"), Param("n_phases", "phases per angle", 3, 3, 7, kind="int"),
           Param("k_frac", "pattern frequency (× cutoff)", 0.9, 0.3, 1.0), Param("m", "modulation depth", 0.8, 0.1, 1.0),
           NOISE(1000, 10, 1e6), SEED],
    solver=[method(S.SIM_METHODS), Param("w", "Wiener parameter", 0.05, 1e-3, 1.0, log=True),
            Param("apod", "apodisation exponent", 1.0, 0.0, 3.0)],
    simulate=_sim_sim, solve=_solve_sim,
    inputs=[Input("raw", "raw frames (angle-major, phase-minor)", "stack")], meta=("wl", "dx", "na", "n_phases"), load=_load_sim,
    example="Siemens star, 3 angles × 3 phases, NA 1.2"))


# ----------------------------------------------------------------------------- Shack–Hartmann
SH_KEYS = [(4, "z4"), (5, "z5"), (6, "z6"), (7, "z7"), (8, "z8"), (11, "z11")]


def _sim_sh(p, rng):
    coeffs = {j: p[k] for j, k in SH_KEYS}
    img, ref, valid, pf, W, ap = WF.sh_simulate(coeffs, p["n_lens"], p["s"], p["wl"], p["pitch"], p["focal"],
                                                p["photons"], p["read_noise"], rng)
    return {"spots": img, "reference": ref, "valid": valid, "pf": pf, "W": W, "ap": ap, "coeffs": coeffs,
            "n_lens": p["n_lens"], "s": p["s"], "wl": p["wl"], "pitch": p["pitch"], "focal": p["focal"]}


def _load_sh(a, m):
    img = np.asarray(a["spots"], float)
    nl, s = int(m["n_lens"]), int(m["s"])
    need = nl * s
    if img.shape[0] < need or img.shape[1] < need:
        raise ValueError(f"image must be at least {need}×{need} for {nl} lenslets of {s} px")
    img = img[:need, :need]
    ref = np.asarray(a["reference"], float)[:need, :need] if "reference" in a else None
    blocks = (ref if ref is not None else img).reshape(nl, s, nl, s).max(axis=(1, 3))
    valid = blocks > 0.2 * blocks.max()
    return {"spots": img / img.max(), "reference": None if ref is None else ref / ref.max(), "valid": valid,
            "pf": m["cam_pixel"], "n_lens": nl, "s": s, "wl": m["wl"], "pitch": m["pitch"], "focal": m["focal"]}


def _solve_sh(d, p, mon):
    nl, s, valid = d["n_lens"], d["s"], d["valid"]
    ref_img = d["reference"]
    cm = p["centroid"]
    c = WF.centroids(d["spots"], nl, s, valid, cm, p["thresh"], ref_img)
    if ref_img is not None and cm != "cross-correlation":
        c0 = WF.centroids(ref_img, nl, s, valid, cm, p["thresh"])
    else:
        c0 = np.zeros_like(c)
    sl = WF.slopes(c, c0, d["pf"], d["focal"], d["pitch"], s, d["wl"])
    mon.step(1, 2)
    res = Result()
    res.panels["spots"] = inten(d["spots"], "spot image")
    gy, gx = sl[..., 0], sl[..., 1]
    res.panels["slopes"] = Panel(np.where(valid, np.hypot(gx, gy), np.nan), "intensity", "|slope| (waves / pitch)",
                                 xlabel="lenslet x", ylabel="lenslet y")
    if p["method"] == "zonal (Southwell)":
        Wl = WF.zonal_southwell(sl, valid)
        coeffs = None
    else:
        coeffs = WF.modal_zernike(sl, valid, nl, s, p["J"])
        X, Y, R, T = WF.sh_pupil_grid(nl, s)
        Wf = WF.wavefront_waves(coeffs, R, T)
        Wl = Wf.reshape(nl, s, nl, s).mean(axis=(1, 3))
        Wl = np.where(valid, Wl - np.nanmean(np.where(valid, Wl, np.nan)), np.nan)
    mon.step(2, 2)
    res.panels["wavefront"] = Panel(Wl, "signed", f"wavefront (waves), {p['method']}", xlabel="lenslet x", ylabel="lenslet y")
    if "W" in d:
        Wt = d["W"].reshape(nl, s, nl, s).mean(axis=(1, 3))
        Wt = np.where(valid, Wt - np.nanmean(np.where(valid, Wt, np.nan)), np.nan)
        res.panels["true wavefront"] = Panel(Wt, "signed", "true wavefront (waves)", xlabel="lenslet x", ylabel="lenslet y")
        lv = float(np.nanmax(np.abs(Wt)))
        res.panels["true wavefront"].levels = res.panels["wavefront"].levels = (-lv, lv)
        e = Wl - Wt
        res.panels["error"] = Panel(e, "signed", "error (waves)", xlabel="lenslet x", ylabel="lenslet y")
        compare(res, Wl, Wt, mask=valid, error_panel=False, metric_unit=" (waves)")
        res.metrics["true RMS (waves)"] = float(np.sqrt(np.nanmean(Wt ** 2)))
        res.metrics["error RMS (waves)"] = float(np.sqrt(np.nanmean(e ** 2)))
    if coeffs is not None:
        js = np.array(sorted(coeffs))
        ser = {"estimated": (js, np.array([coeffs[j] for j in js]))}
        if "coeffs" in d:
            ser["true"] = (js, np.array([d["coeffs"].get(j, 0.0) for j in js]))
        res.curves.append(Curve("Zernike coefficients (Noll)", ser, "Noll index j", "coefficient (waves RMS)", logy=False))
        names = {j: n for j, n in ZERNIKE_NAMES.items()} if isinstance(ZERNIKE_NAMES, dict) else {}
        for j in (4, 5, 6, 7, 8, 11):
            if j in coeffs:
                res.metrics[f"Z{j} {names.get(j, '')}".strip()] = coeffs[j]
    return res


register(Experiment(
    key="shack_hartmann", title="Shack–Hartmann wavefront sensing", area=AREA,
    blurb="Spot images of a lenslet array are computed by wave optics; centroids give local slopes, "
          "integrated zonally (Southwell least squares) or fitted modally to Zernike derivatives.",
    equations=[r"\Delta\mathbf{x}_{ij}=f\,\langle\nabla W\rangle_{ij}",
               r"W_{i,j+1}-W_{i,j}=\frac{p}{2}\left(s^x_{i,j}+s^x_{i,j+1}\right)\quad\mathrm{(Southwell)}",
               r"\mathbf{s}=G\,\mathbf{c},\quad G_{kj}=\langle\nabla Z_j\rangle_k"],
    scene=[Param("n_lens", "lenslets across", 16, 6, 40, kind="int"), Param("s", "pixels per lenslet", 16, 8, 40, kind="int"),
           WL(0.633), Param("pitch", "lenslet pitch", 150.0, 50.0, 500.0, unit="µm"),
           Param("focal", "lenslet focal length", 5000.0, 1000.0, 30000.0, unit="µm"),
           Param("cam_pixel", "camera pixel (user data)", 5.0, 1.0, 20.0, unit="µm"),
           Param("z4", "defocus Z4", 0.4, -2.0, 2.0, unit="waves"), Param("z5", "astigmatism Z5", 0.2, -2.0, 2.0, unit="waves"),
           Param("z6", "astigmatism Z6", -0.1, -2.0, 2.0, unit="waves"), Param("z7", "coma Z7", 0.15, -2.0, 2.0, unit="waves"),
           Param("z8", "coma Z8", 0.0, -2.0, 2.0, unit="waves"), Param("z11", "spherical Z11", 0.1, -2.0, 2.0, unit="waves"),
           NOISE(2000, 10, 1e6), Param("read_noise", "read noise (e⁻)", 2.0, 0.0, 50.0), SEED],
    solver=[method(("modal (Zernike)", "zonal (Southwell)")), choice("centroid", "centroiding", WF.CENTROID_METHODS, "windowed CoG (iterative)"),
            Param("thresh", "threshold", 0.2, 0.0, 0.8), Param("J", "Zernike modes", 15, 3, 45, kind="int")],
    simulate=_sim_sh, solve=_solve_sh,
    inputs=[Input("spots", "spot image"), Input("reference", "reference (flat-wavefront) spot image", optional=True)],
    meta=("n_lens", "s", "wl", "pitch", "focal", "cam_pixel"), load=_load_sh,
    example="16×16 lenslets, defocus + astigmatism + coma + spherical"))


# ----------------------------------------------------------------------------- speckle correlation
def _speckle_geometry(p):
    return SC.speckle_geometry(n=p["n"], pixel=p["pixel"], wl=p["wl"], D=1e3 * p["iris"], v=1e3 * p["v"], u=1e3 * p["u"],
                               angle_deg=p["angle"], obj_mm=p["obj_mm"], fill=p["fill"])


def _sim_speckle(p, rng):
    g = _speckle_geometry(p)
    obj = SC.hidden_object(120, 120, p["object"], rng)
    cam, truth, diag = SC.simulate_scattering(obj, g, p["photons"], rng, p["read_noise"], p["bits"], p["background"],
                                              p["frames"])
    size = max(4, int(round(g["obj_px"])))
    return {"camera": cam, "truth": truth, "size": size, "geometry": g, "diag": diag}


def _load_speckle(a, m):
    """User data: the object's image size on the sensor (px) follows from the geometry, M = v/u."""
    size = m["obj_mm"] * 1e3 * (m["v"] / m["u"]) / m["pixel"]
    return {"camera": np.asarray(a["camera"], float), "size": max(4, int(round(size)))}


def _solve_speckle(d, p, mon):
    x, A, e = SC.speckle_reconstruct(d["camera"], d["size"], p["method"], p["iters"],
                                     np.random.default_rng(p["seed"]), mon)
    res = Result()
    n = d["camera"].shape[-1]
    c, w = n // 2, min(n // 2, int(1.5 * d["size"]))
    cam = d["camera"] if d["camera"].ndim == 2 else d["camera"][0]
    nf = 1 if d["camera"].ndim == 2 else d["camera"].shape[0]
    res.panels["camera"] = inten(cam, "camera image (speckle)" + (f", 1 of {nf} exposures" if nf > 1 else ""))
    res.panels["autocorrelation"] = inten(A[c - w:c + w, c - w:c + w], "autocorrelation (centre)", extent=(-w, w, -w, w),
                                          xlabel="lag Δx (px)", ylabel="lag Δy (px)")
    if "truth" in d:
        t = d["truth"]
        xa, _ = F.align_translation_and_phase(x, t)
        xa = np.real(xa)
        # refine the registration (± 3 px) on the scale-fitted error: the correlation peak of a blurred,
        # noisy estimate can sit a pixel or two off
        best = None
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                cnd = np.roll(xa, (dy, dx), (0, 1))
                ev = float(np.mean((F.fit_scale(cnd, t) - t) ** 2))
                if best is None or ev < best[0]:
                    best = (ev, cnd)
        xa = F.fit_scale(best[1], t)            # shown on the truth's scale
        res.panels["reconstruction"] = inten(xa[c - w:c + w, c - w:c + w], f"recovered object ({p['method']}), registered")
        res.panels["truth"] = inten(t[c - w:c + w, c - w:c + w], "hidden object as imaged by the pixels (inverted, M < 0)")
        res.panels["reconstruction"].levels = res.panels["truth"].levels = (0.0, float(t.max()))
        res.metrics["correlation"] = float(np.corrcoef(xa.ravel(), t.ravel())[0, 1])
        compare(res, xa[c - w:c + w, c - w:c + w], t[c - w:c + w, c - w:c + w], fit=True)
    else:
        res.panels["reconstruction"] = inten(np.fft.fftshift(x)[c - w:c + w, c - w:c + w], f"recovered object ({p['method']})")
    if "geometry" in d:
        g, dg = d["geometry"], d["diag"]
        ok = lambda b: "ok" if b else "VIOLATED"
        res.metrics.update({
            "speckle grain λv/D (px)": float(g["grain"]),
            "measured grain (px)": dg["measured grain (px)"],
            "speckle contrast (camera)": dg["speckle contrast (camera)"],
            "object image size (px)": float(g["obj_px"]),
            "Fresnel number (D/2)²/λv": float(g["fresnel_number"]),
            "sub-pixel samples q": int(g["q"]),
            "simulation grid": f"{g['L']}² (2× padded), dx = {g['dx']:.3g} µm",
            "intensity Nyquist dx ≤ λv/2D": ok(g["intensity_nyquist"]),
            "phase screen resolved dx ≤ ℓ/3": ok(g["screen_resolved"]),
            "ASM band limit ≥ steepest sensor ray": ok(g["band_limit_ok"]) + f" ({np.rad2deg(g['theta_bl']):.2f}° ≥ {np.rad2deg(g['theta_need']):.2f}°)",
            "camera Nyquist (grain ≥ 2 px)": ok(g["sensor_nyquist"]),
        })
    res.curves.append(conv("Fourier error", {p["method"]: e}))
    return res


register(Experiment(
    key="speckle", title="Imaging through scattering (speckle correlation)", area=AREA,
    blurb="A hidden incoherent object behind a thin diffuser produces a featureless speckle image. The "
          "point-source PSF is computed with the band-limited, zero-padded angular spectrum on a grid finer "
          "than the camera pixel, and the camera integrates intensity over each pixel's active area before "
          "shot and read noise. Within the memory effect the image autocorrelation equals the object's, so "
          "the object follows from a single Fourier phase-retrieval run (shrink-wrap, iterate averaging). "
          "Several exposures through independent diffuser positions average down the noise of the "
          "autocorrelation estimate, which is what makes that single run reproducible.",
    equations=[r"U_v=\mathcal{F}^{-1}\{\mathcal{F}\{A\,e^{i\phi}\,e^{ik\sqrt{r^2+u^2}}\}\,e^{i2\pi v\sqrt{\lambda^{-2}-f^2}}\}",
               r"I_{mn}=\int_{\mathrm{pixel}_{mn}}(O_M\ast|U_v|^2)\,dA,\quad M=v/u",
               r"I\star I\approx (O\star O)\ast(S\star S)\approx O\star O+C",
               r"|\tilde{O}|=\sqrt{\mathcal{F}\{I\star I\}}\ \Rightarrow\ O\ \mathrm{by\ HIO/ER}"],
    scene=[Param("n", "sensor pixels", 256, 64, 768, kind="int", tip="Camera pixels per side"),
           choice("object", "hidden object", F.objects("letter")),
           WL(), Param("pixel", "camera pixel pitch", 6.5, 1.0, 20.0, unit="µm"),
           Param("fill", "pixel fill factor", 1.0, 0.1, 1.0, tip="Active fraction of the pixel area"),
           Param("iris", "iris diameter D", 1.0, 0.2, 5.0, unit="mm", tip="Illuminated diffuser aperture; sets the grain λv/D"),
           Param("v", "diffuser → sensor", 30.0, 2.0, 200.0, unit="mm"),
           Param("u", "object → diffuser", 300.0, 20.0, 2000.0, unit="mm", tip="Magnification of the object's image is v/u"),
           Param("obj_mm", "object size", 2.5, 0.2, 20.0, unit="mm"),
           Param("angle", "diffuser scattering angle (rms)", 10.0, 1.0, 30.0, unit="°",
                 tip="Sets the phase-screen correlation length ℓ = λσφ√2 / 2πθ"),
           NOISE(1e4, 10, 1e7), Param("read_noise", "read noise", 2.0, 0.0, 50.0, unit="e⁻"),
           Param("bits", "ADC bits", 12, 8, 16, kind="int"),
           Param("background", "background (× mean)", 0.0, 0.0, 5.0),
           Param("frames", "exposures (independent diffuser positions)", 4, 1, 16, kind="int",
                 tip="Autocorrelations of exposures through independent diffuser realisations are averaged: "
                     "the estimate's noise drops as 1/√frames"), SEED],
    solver=[method(SC.SPECKLE_METHODS), ITERS(1500, 20, 5000), Param("seed", "random start", 0, 0, 9999, kind="int")],
    simulate=_sim_speckle, solve=_solve_speckle,
    inputs=[Input("camera", "camera image, or a stack of exposures through different diffuser positions")],
    meta=("pixel", "v", "u", "obj_mm"), load=_load_speckle, live=False,
    example="A letter 300 mm behind a ground-glass diffuser, camera 30 mm behind it"))


# ----------------------------------------------------------------------------- two-photon
def _sim_2p(p, rng):
    n, nz, dx, dz = p["n"], p["nz"], p["dx"], p["dz"]
    if p["phantom"] == "neurons":
        vol = PH.neuron_volume(n, nz, rng)
    elif p["phantom"] == "beads":
        vol = np.zeros((nz, n, n))
        for _ in range(25):
            z, y, x = rng.integers(2, nz - 2), rng.integers(4, n - 4), rng.integers(4, n - 4)
            vol[z, y, x] = rng.uniform(0.5, 1.0)
        from scipy import ndimage as ndi
        vol = ndi.gaussian_filter(vol, 0.7)
        vol /= vol.max()
    else:
        from ..core import tomography as T
        v = T.phantom_3d(p["phantom"], n, rng, p["object"])
        vol = np.array([v[int(k)] for k in np.linspace(0, n - 1, nz)])
    h = PH.effective_psf(p["mode"], n, nz, dx, dz, p["wl_1p"], p["wl_2p"], 0.52, p["na"], 1.33, p["pinhole"])
    expo = 2 if p["mode"].startswith("2P") else 1
    wf = p["wl_2p"] / p["wl_1p"] if expo == 2 else 1.0
    img = PH.image_3d(vol, h, p["photons"], rng, p["ls"] if p["scattering"] else None, wf ** 1.6, dz, expo)
    curves = {m: PH.sectioning_curve(PH.effective_psf(m, n, nz, dx, dz, p["wl_1p"], p["wl_2p"], 0.52, p["na"], 1.33, p["pinhole"]))
              for m in PH.TWO_PHOTON_MODES}
    return {"stack": img, "psf": h, "truth": vol, "sectioning": curves, "dz": dz, "dx": dx, "mode": p["mode"]}


def _solve_2p(d, p, mon):
    img, h = d["stack"], d["psf"]
    m = p["method"]
    if m == "raw":
        x = img
    else:
        x = PH.rl3d(img, h, p["iters"], monitor=mon)
    res = Result()
    nz = img.shape[0]
    dx_, dz_ = d.get("dx", 1.0), d.get("dz", 1.0)
    ny, nx = img.shape[1:]
    ex = (-nx * dx_ / 2, nx * dx_ / 2, -ny * dx_ / 2, ny * dx_ / 2)
    exz = (-nx * dx_ / 2, nx * dx_ / 2, -nz * dz_ / 2, nz * dz_ / 2)
    xz = dict(extent=exz, xlabel="x (µm)", ylabel="z (µm)")
    zv = dict(zextent=(-(nz - 1) * dz_ / 2, (nz - 1) * dz_ / 2), zlabel="z (µm)")
    res.panels["image stack"] = Panel(img, "volume", "recorded stack, x–y (slider = z)", extent=ex, unit="µm", **zv)
    res.panels["x–z section"] = inten(img[:, img.shape[1] // 2], "recorded x–z section", **xz)
    res.panels["processed"] = Panel(x, "volume", f"{m}, x–y (slider = z)", extent=ex, unit="µm", **zv)
    res.panels["processed x–z"] = inten(x[:, x.shape[1] // 2], f"{m}, x–z", **xz)
    c = h.shape[1] // 2
    w = min(16, c)
    res.panels["PSF x–z"] = inten(h[:, c, c - w:c + w] ** 0.5, "effective PSF x–z (√)",
                                  extent=(-w * dx_, w * dx_, -nz * dz_ / 2, nz * dz_ / 2), xlabel="x (µm)", ylabel="z (µm)")
    if "truth" in d:
        t = d["truth"]
        res.panels["truth"] = Panel(t, "volume", "truth, x–y (slider = z)", extent=ex, unit="µm", **zv)
        res.panels["truth x–z"] = inten(t[:, t.shape[1] // 2], "truth x–z", **xz)
        compare(res, x, t, fit=True, extent=ex, unit="µm")
        res.metrics["correlation (raw)"] = float(np.corrcoef(img.ravel(), t.ravel())[0, 1])
        res.metrics["correlation"] = float(np.corrcoef(x.ravel(), t.ravel())[0, 1])
    if "sectioning" in d:
        z = (np.arange(nz) - nz // 2) * d["dz"]
        res.curves.append(Curve("optical sectioning: ∑ over a plane vs defocus", {k: (z, v) for k, v in d["sectioning"].items()},
                                "z (µm)", "normalised integrated signal", logy=False))
        res.curves.append(Curve("axial PSF on axis", {"effective PSF": (z, h[:, c, c] / h[:, c, c].max())}, "z (µm)", "", logy=False))
    return res


register(Experiment(
    key="two_photon", title="Two-photon excitation microscopy", area=AREA,
    blurb="Compare one-photon wide-field-style scanning, confocal and two-photon excitation of a neuron-like "
          "volume. Two-photon fluorescence scales with I², so excitation is confined to the focus: the "
          "integrated signal per plane falls off with defocus (optical sectioning without a pinhole) and "
          "the longer wavelength scatters less with depth. 3-D Richardson–Lucy restores the stack.",
    equations=[r"h_{1P}=|h_{\lambda}|^2,\qquad h_{\mathrm{conf}}=|h_{\lambda_{ex}}|^2\,\left(|h_{\lambda_{em}}|^2\ast P\right),\qquad h_{2P}=|h_{\lambda_{2P}}|^4",
               r"\int\!\!\int |h|^2\,dx\,dy=\mathrm{const}(z)\quad\mathrm{vs}\quad\int\!\!\int |h|^4\,dx\,dy\propto z^{-2}",
               r"S(z)\propto\left(e^{-z/\ell_s}\right)^{n}"],
    scene=[Param("n", "lateral pixels", 64, 32, 128, kind="int"), Param("nz", "z planes", 32, 16, 64, kind="int"),
           choice("phantom", "specimen", ("neurons", "beads", "3-D Shepp–Logan", "sample object in a sphere")),
           choice("object", "object (for the sphere specimen)", F.objects("usaf")),
           DX(0.15, 0.05, 0.5), Param("dz", "z step", 0.4, 0.1, 2.0, unit="µm"), choice("mode", "excitation", PH.TWO_PHOTON_MODES, "2P excitation"),
           Param("na", "NA", 1.0, 0.3, 1.3), Param("wl_1p", "1P wavelength", 0.488, 0.35, 0.7, unit="µm"),
           Param("wl_2p", "2P wavelength", 0.92, 0.7, 1.3, unit="µm"), Param("pinhole", "pinhole (AU)", 1.0, 0.2, 5.0),
           flag("scattering", "tissue scattering", False), Param("ls", "scattering length", 15.0, 2.0, 200.0, unit="µm"),
           NOISE(200, 5, 1e6), SEED],
    solver=[method(("3-D Richardson–Lucy", "raw")), ITERS(15, 1, 200)],
    simulate=_sim_2p, solve=_solve_2p,
    inputs=[Input("stack", "z-stack", "stack"), Input("psf", "3-D PSF stack", "stack", optional=True)],
    meta=("dx", "dz", "na", "wl_2p", "mode"),
    load=lambda a, m: {"stack": stack_input(a["stack"]), "dz": m["dz"], "dx": m["dx"],
                       "psf": (lambda h: h / h.sum())(stack_input(a["psf"])) if "psf" in a else
                       PH.effective_psf(m["mode"], stack_input(a["stack"]).shape[-1], len(stack_input(a["stack"])),
                                        m["dx"], m["dz"], na=m["na"], wl_2p=m["wl_2p"])},
    example="Neuron-like volume, 2P vs confocal vs 1P"))
