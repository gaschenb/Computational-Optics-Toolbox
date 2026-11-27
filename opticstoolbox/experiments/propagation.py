"""Wave propagation experiments: diffraction integrals, aberrated imaging, Talbot self-imaging."""
from __future__ import annotations

import numpy as np

from ..core import fields as F
from ..core import propagation as PR
from ..core import waveoptics as WO
from ._common import (DX, N, WL, Curve, compare, freq_extent, Experiment, Input, Panel, Param, Result, amp, choice, extent, inten,
                      method, phase, register)

AREA = "Wave propagation"

APERTURES = ("circular", "square", "double slit") + F.OBJECTS
_OLD = {"USAF target": "usaf", "Siemens star": "siemens"}       # earlier names, still accepted


def _aperture(kind, n, dx, size_um):
    X, Y = F.grid(n, dx)
    r = size_um / 2
    if kind == "circular":
        return (np.hypot(X, Y) < r).astype(complex)
    if kind == "square":
        return ((np.abs(X) < r) & (np.abs(Y) < r)).astype(complex)
    if kind == "double slit":
        w = size_um / 8
        return (((np.abs(X - size_um / 4) < w / 2) | (np.abs(X + size_um / 4) < w / 2)) & (np.abs(Y) < r)).astype(complex)
    obj = _OLD.get(kind, kind)
    s = max(4, min(int(round(size_um / dx)), n))
    m2c = F.normalize(np.clip(F.resize(F.test_object(obj, 256), (s, s)), 0, None))
    m2 = F.pad_to(m2c, (n, n))
    # binary chrome-on-glass targets are thresholded; the others are grey-level amplitude transmittances
    return (m2 > 0.5).astype(complex) if obj in ("usaf", "siemens", "letter") else m2.astype(complex)


def _sim_prop(p, rng):
    n, dx = p["n"], p["dx"]
    u0 = _aperture(p["aperture"], n, dx, p["size"])
    return {"field": u0, "wl": p["wl"], "dx": dx, "z": p["z"]}


def _load_prop(arrays, m):
    a = np.asarray(arrays["amplitude"], float)
    ph = np.asarray(arrays.get("phase", np.zeros_like(a)), float)
    return {"field": np.sqrt(np.clip(a, 0, None)) * np.exp(1j * ph) if m.get("as_intensity") else a * np.exp(1j * ph),
            "wl": m["wl"], "dx": m["dx"], "z": m["z"]}


def _solve_prop(d, p, mon):
    u0, wl, dx, z = d["field"], d["wl"], d["dx"], d["z"]
    n = u0.shape[0]
    meth = p["method"]
    if meth in ("Fresnel single FFT", "Fraunhofer"):
        u, dx2 = PR.propagate(u0, wl, dx, z, meth)
    else:
        u, dx2 = PR.propagate(u0, wl, dx, z, meth), dx
    res = Result()
    res.panels["input |u₀|"] = amp(u0, "input amplitude |u₀|", extent=extent(n, dx), unit="µm")
    res.panels["|u(z)|²"] = inten(np.abs(u) ** 2, f"intensity at z = {z:g} µm ({meth})", extent=extent(n, dx2), unit="µm")
    res.panels["arg u(z)"] = phase(u, "phase at z", extent=extent(n, dx2), unit="µm")
    rep = PR.sampling_report(n, dx, wl, z)
    res.metrics.update({k: v for k, v in rep.items() if isinstance(v, float)})
    res.metrics["output pitch (µm)"] = dx2
    ser = {}
    xs = F.coords(n, dx2)
    ser[meth] = (xs, np.abs(u[n // 2]) ** 2)
    if p["compare"] and meth not in ("Fresnel single FFT", "Fraunhofer") and n <= 512:
        ref = PR.rayleigh_sommerfeld(u0, wl, dx, z)
        ser["Rayleigh–Sommerfeld (reference)"] = (xs, np.abs(ref[n // 2]) ** 2)
        res.metrics["relative error vs RS"] = float(np.linalg.norm(u - ref) / (np.linalg.norm(ref) + 1e-20))
        res.panels["|u − u_RS|"] = Panel(np.abs(u - ref), "intensity", "deviation from Rayleigh–Sommerfeld",
                                         extent=extent(n, dx2), unit="µm")
    res.curves.append(Curve("central line profile |u|²", ser, "x (µm)", "intensity"))
    res.notes = f"Recommended for this geometry: {rep['recommended']} (z_c = N dx²/λ = {rep['critical distance z_c']:.1f} µm)."
    return res


register(Experiment(
    key="propagation", title="Diffraction propagation", area=AREA,
    blurb="Propagate an aperture field with the exact angular spectrum (with and without Matsushima's band "
          "limit), the Fresnel transfer function and impulse response, the one-FFT Fresnel and Fraunhofer "
          "transforms, and compare against a Rayleigh–Sommerfeld reference.",
    equations=[r"U(f_x,f_y;z)=U(f_x,f_y;0)\,\exp\left(i2\pi z\sqrt{\lambda^{-2}-f_x^2-f_y^2}\right)",
               r"|f_x|<\left[\lambda\sqrt{(2\Delta f\,z)^2+1}\right]^{-1}",
               r"u(x,y;z)=\frac{e^{ikz}}{i\lambda z}\iint u(\xi,\eta)\,e^{\frac{i\pi}{\lambda z}[(x-\xi)^2+(y-\eta)^2]}\,d\xi\, d\eta",
               r"u(\mathbf{r})=\frac{z}{i\lambda}\iint u_0\,\frac{e^{ikr}}{r^2}\left(1+\frac{i}{kr}\right)d^2\rho"],
    scene=[N(256, 64, 1024), DX(1.0, 0.05, 20), WL(0.633), Param("z", "distance z", 2000.0, 1.0, 1e6, log=True, unit="µm"),
           choice("aperture", "aperture", APERTURES), Param("size", "aperture size", 60.0, 1.0, 5000.0, log=True, unit="µm")],
    solver=[method(list(PR.METHODS) + ["Fresnel single FFT", "Fraunhofer"]),
            Param("compare", "compare with RS reference", True, kind="bool")],
    simulate=_sim_prop, solve=_solve_prop,
    inputs=[Input("amplitude", "input amplitude (or intensity)"), Input("phase", "input phase (rad)", optional=True)],
    meta=("wl", "dx", "z"), load=_load_prop,
    example="Circular aperture, λ = 633 nm: ASM vs Fresnel vs RS at 2 mm"))


# ----------------------------------------------------------------------------- aberrated imaging
ZERN_KEYS = [(4, "defocus"), (5, "astig_45"), (6, "astig_0"), (7, "coma_y"), (8, "coma_x"), (9, "trefoil_y"), (11, "spherical")]


def _sim_psf(p, rng):
    n = p["n"]
    obj = F.test_object(p["object"], n, rng)
    return {"object": obj, "wl": p["wl"], "dx": p["dx"], "na": p["na"],
            "coeffs": {j: p[k] for j, k in ZERN_KEYS}}


def _load_psf(arrays, m):
    return {"object": np.asarray(arrays["object"], float), "wl": m["wl"], "dx": m["dx"], "na": m["na"],
            "coeffs": {j: m[k] for j, k in ZERN_KEYS}}


def _solve_psf(d, p, mon):
    obj, wl, dx, na, co = d["object"], d["wl"], d["dx"], d["na"], d["coeffs"]
    n = obj.shape[0]
    P = WO.pupil(obj.shape, wl, dx, na, co)
    P0 = WO.pupil(obj.shape, wl, dx, na)
    res = Result()
    import scipy.fft as sfft
    # show the pupil itself (± 1.2 NA), not the whole frequency grid, where it would be a dot
    Ps = sfft.fftshift(P)
    fx = (np.arange(n) - n // 2) / (n * dx) * wl
    sel = np.abs(fx) <= 1.2 * na
    i0, i1 = int(np.argmax(sel)), int(n - np.argmax(sel[::-1]))
    Pc = Ps[i0:i1, i0:i1]
    h = 0.5 * (fx[1] - fx[0])
    res.panels["pupil phase"] = Panel(np.where(np.abs(Pc) > 0, np.angle(Pc), np.nan), "phase", "pupil phase W(ρ, θ)",
                                      extent=(fx[i0] - h, fx[i1 - 1] + h, fx[i0] - h, fx[i1 - 1] + h),
                                      xlabel="NAₓ = λfₓ", ylabel="NA_y = λf_y")
    psf = WO.incoherent_psf(P)
    c = n // 2
    w = max(8, int(4 * wl / na / dx * 3))
    crop = (slice(max(c - w, 0), c + w), slice(max(c - w, 0), c + w))
    pc = psf[crop]
    res.panels["PSF (log)"] = Panel(np.log10(pc / psf.max() + 1e-6), "intensity", "PSF, log₁₀",
                                    extent=(-pc.shape[1] / 2 * dx, pc.shape[1] / 2 * dx, -pc.shape[0] / 2 * dx, pc.shape[0] / 2 * dx),
                                    unit="µm")
    if p["method"] == "incoherent imaging":
        img = WO.image_incoherent(obj, psf)
        ideal = WO.image_incoherent(obj, WO.incoherent_psf(P0))
    else:
        img = np.abs(WO.image_coherent(np.exp(1j * np.pi / 2 * obj) if p["object_type"] == "phase" else obj, P)) ** 2
        ideal = np.abs(WO.image_coherent(np.exp(1j * np.pi / 2 * obj) if p["object_type"] == "phase" else obj, P0)) ** 2
    res.panels["object"] = inten(obj, "object", extent=extent(n, dx), unit="µm")
    res.panels["diffraction-limited image"] = inten(ideal, "diffraction-limited image", extent=extent(n, dx), unit="µm")
    res.panels["aberrated image"] = inten(img, "aberrated image", extent=extent(n, dx), unit="µm")
    mtf = np.abs(sfft.fftshift(WO.otf(psf)))
    mtf0 = np.abs(sfft.fftshift(WO.otf(WO.incoherent_psf(P0))))
    f = F.coords(n, 1.0 / (n * dx))
    res.curves.append(Curve("MTF (horizontal cut)", {"aberrated": (f[c:], mtf[c, c:]), "diffraction limited": (f[c:], mtf0[c, c:]),
                                                     "vertical cut (aberrated)": (f[c:], mtf[c:, c])},
                            "spatial frequency (cycles/µm)", "MTF"))
    res.metrics["Strehl ratio"] = WO.strehl(P)
    compare(res, img, ideal, prefix="vs diffraction-limited", title="aberrated − diffraction-limited",
            extent=extent(n, dx), unit="µm")
    res.metrics["Rayleigh resolution 0.61λ/NA (µm)"] = 0.61 * wl / na
    res.metrics["RMS wavefront (waves)"] = float(np.sqrt(sum(v ** 2 for v in co.values())))
    return res


register(Experiment(
    key="psf", title="Aberrated imaging (Zernike)", area=AREA,
    blurb="Build a pupil from Zernike aberrations (Noll, waves RMS), compute the PSF, OTF/MTF and Strehl ratio, "
          "and image a test object coherently or incoherently.",
    equations=[r"P(\rho,\theta)=\mathrm{circ}(\rho)\,\exp\left(i2\pi\sum_j c_j Z_j(\rho,\theta)\right)",
               r"h_{\mathrm{inc}}=|\mathcal{F}^{-1}\{P\}|^2,\qquad \mathrm{OTF}=\mathcal{F}\{h\}/\mathcal{F}\{h\}(0)",
               r"S\approx \exp(-(2\pi\sigma_W)^2)"],
    scene=[N(256, 64, 1024), DX(0.05, 0.01, 2), WL(0.55), Param("na", "NA", 0.8, 0.05, 1.45),
           choice("object", "object", F.objects("siemens"))] +
          [Param(k, k.replace("_", " ") + " (Z" + str(j) + ")", 0.15 if k == "coma_y" else 0.0, -1.0, 1.0, unit="waves")
           for j, k in ZERN_KEYS],
    solver=[method(("incoherent imaging", "coherent imaging")), choice("object_type", "coherent object", ("amplitude", "phase"))],
    simulate=_sim_psf, solve=_solve_psf,
    inputs=[Input("object", "object (intensity or amplitude)")],
    meta=("wl", "dx", "na") + tuple(k for _, k in ZERN_KEYS), load=_load_psf,
    example="Siemens star through 0.15 waves RMS of coma"))


# ----------------------------------------------------------------------------- Talbot carpet
def _sim_talbot(p, rng):
    return {"period": p["period"], "wl": p["wl"], "duty": p["duty"], "n": p["n"], "dx": p["period"] / 32}


def _solve_talbot(d, p, mon):
    a, wl = d["period"], d["wl"]
    dx = d["dx"]
    n = d["n"]
    zT = 2 * a ** 2 / wl
    x = F.coords(n, dx)
    g = (((x / a) % 1.0) < d["duty"]).astype(complex)
    if p["grating"] == "sinusoidal":
        g = (0.5 + 0.5 * np.cos(2 * np.pi * x / a)).astype(complex)
    zs = np.linspace(0, p["span"] * zT, 300)
    u = g[None, :]
    FX = np.fft.fftfreq(n, dx)
    if p["method"] == "angular spectrum":
        H = np.exp(2j * np.pi * zs[:, None] * np.sqrt(np.clip(1 / wl ** 2 - FX ** 2, 0, None))) * (np.abs(FX) < 1 / wl)
    else:
        H = np.exp(-1j * np.pi * wl * zs[:, None] * FX ** 2)
    carpet = np.abs(np.fft.ifft(np.fft.fft(g)[None] * H, axis=1)) ** 2
    res = Result()
    m = n // 4
    res.panels["Talbot carpet"] = Panel(carpet[:, :m].T, "intensity", "intensity |u(x, z)|² (x vertical, z horizontal)",
                                        extent=(0, zs[-1], x[0], x[m - 1]), xlabel="z (µm)", ylabel="x (µm)", aspect=False)
    i_T = int(np.argmin(np.abs(zs - zT)))
    i_h = int(np.argmin(np.abs(zs - zT / 2)))
    res.curves.append(Curve("profiles", {"z = 0": (x[:m], carpet[0, :m]), "z = z_T/2 (shifted)": (x[:m], carpet[i_h, :m]),
                                         "z = z_T": (x[:m], carpet[i_T, :m])}, "x (µm)", "intensity"))
    res.metrics["Talbot length z_T = 2a²/λ (µm)"] = zT
    res.metrics["revival contrast at z_T"] = float(np.corrcoef(carpet[0], carpet[i_T])[0, 1])
    return res


register(Experiment(
    key="talbot", title="Talbot self-imaging", area=AREA,
    blurb="A periodic grating re-images itself at multiples of the Talbot length z_T = 2a²/λ, with a half-period "
          "shift at z_T/2 and fractional Talbot images in between (the 'Talbot carpet').",
    equations=[r"z_T=\frac{2a^2}{\lambda}", r"u(x,z)=\sum_m c_m\,e^{i2\pi m x/a}\,e^{i2\pi z\sqrt{\lambda^{-2}-m^2/a^2}}"],
    scene=[Param("period", "grating period a", 8.0, 1.0, 100.0, unit="µm"), WL(0.633),
           Param("duty", "duty cycle", 0.3, 0.05, 0.95), Param("n", "samples", 1024, 256, 4096, kind="int")],
    solver=[method(("angular spectrum", "Fresnel (paraxial)")), choice("grating", "grating", ("binary", "sinusoidal")),
            Param("span", "z range (× z_T)", 1.0, 0.25, 4.0)],
    simulate=_sim_talbot, solve=_solve_talbot, inputs=[], meta=(),
    example="8 µm binary grating at 633 nm: carpet over one Talbot length"))
