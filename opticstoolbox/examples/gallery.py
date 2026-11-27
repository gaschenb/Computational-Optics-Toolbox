"""Render every experiment (default scene, default method) to docs/figures/<key>.png.

    python -m opticstoolbox.examples.gallery [keys ...] [--out docs/figures]
"""
from __future__ import annotations

import argparse
import os
import time

from ..core import registry as R
from ..render import save_png

# Panels to show first for experiments where the default order is not the most telling.
PREFERRED = {
    "offaxis": ["hologram", "spectrum + filter", "amplitude", "unwrapped phase", "true phase", "phase error"],
    "sim": ["widefield", "reconstruction", "truth", "spectrum"],
    "ct": ["sinogram", "reconstruction", "truth", "error"],
    "odt": ["field phase (view 1)", "x–z slice", "truth x–z", "k-space coverage"],
    "deconvolution": ["measured", "deconvolved", "truth", "PSF"],
    "speckle": ["camera", "autocorrelation", "reconstruction", "truth"],
    "polarization": ["DoLP", "AoLP", "normals", "true normals", "height"],
    "fringe": ["fringe image", "wrapped phase", "height", "true height"],
    "ghost": ["pattern", "reconstruction", "truth"],
    "diffusercam": ["raw", "PSF", "reconstruction", "truth"],
    "cassi": ["measurement", "reconstruction (RGB)", "truth (RGB)"],
    "nlos": ["transient", "front view", "depth", "true front"],
    "first_photon": ["pulses to first detection", "reflectivity", "depth", "raw depth", "true depth"],
    "spad_lidar": ["photon counts", "depth", "true depth", "depth error"],
    "biphoton": ["camera (unheralded)", "coincidences", "image", "object"],
}


def render_one(key, out):
    e = R.get(key)
    t0 = time.perf_counter()
    data = e.make_data()
    res = e.run(data)
    method = e.resolve("solver").get("method", "")
    path = os.path.join(out, f"{key}.png")
    names = PREFERRED.get(key)
    if names:
        names = [k for k in names if k in res.panels]
        names = names + [k for k in res.panels if k not in names and res.panels[k].kind != "volume"
                         and ("error" in k.lower() or "true" in k.lower() or "truth" in k.lower())]
        names = names + [k for k in res.panels if k not in names]
    save_png(res, path, f"{e.title}" + (f" — {method}" if method else ""), panels=names, max_panels=7)
    return path, time.perf_counter() - t0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="*")
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--out", default=os.path.normpath(os.path.join(here, "..", "..", "docs", "figures")))
    a = ap.parse_args(argv)
    R.load_all()
    os.makedirs(a.out, exist_ok=True)
    for k in a.keys or list(R.EXPERIMENTS):
        try:
            p, dt = render_one(k, a.out)
            print(f"{k:18s} {dt:6.1f} s  {p}")
        except Exception as ex:  # keep going; report at the end
            print(f"{k:18s} FAILED: {type(ex).__name__}: {ex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
