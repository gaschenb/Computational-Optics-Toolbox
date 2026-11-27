"""Importing this package registers every experiment (order = sidebar order)."""
from . import propagation, holography, retrieval, tomography, microscopy, shape3d, cameras, photon  # noqa: F401


def _mark_surfaces(solve):
    """Holography results: every reconstructed / Ground Truth map (amplitude, phase, height, error) gets
    a 3-D surface view in the GUI. Raw holograms and spectra do not."""
    import functools
    import numpy as np

    @functools.wraps(solve)
    def run(data, params, monitor):
        r = solve(data, params, monitor)
        for name, p in r.panels.items():
            if (p.kind in ("amplitude", "phase", "signed", "depth") and np.ndim(p.data) == 2
                    and not any(w in name for w in ("hologram", "spectrum"))):
                p.surface = True
        return r
    return run


def _dhm_surfaces():
    from ..core.registry import AREAS, EXPERIMENTS
    for key in AREAS.get("Digital holography", []):
        e = EXPERIMENTS[key]
        if not getattr(e.solve, "_surfaces", False):
            e.solve = _mark_surfaces(e.solve)
            e.solve._surfaces = True


_dhm_surfaces()
