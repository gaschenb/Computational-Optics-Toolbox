"""Static (matplotlib) rendering of a Result: used by the CLI (--png) and the examples gallery."""
from __future__ import annotations

import numpy as np

CMAPS = {"intensity": "gray", "amplitude": "gray", "phase": "twilight", "signed": "RdBu_r", "depth": "viridis",
         "mask": "gray", "volume": "magma"}


def _panel_image(p):
    a = np.asarray(p.data)
    if p.kind == "volume":
        a = a[a.shape[0] // 2]
    if np.iscomplexobj(a):
        a = np.abs(a)
    return a


def _limits(p, a):
    if p.levels is not None:
        return p.levels
    kind = p.cmap or p.kind
    if kind == "phase":
        return (-np.pi, np.pi) if np.nanmax(np.abs(a)) <= np.pi + 1e-6 else (np.nanmin(a), np.nanmax(a))
    if kind == "signed":
        m = np.nanpercentile(np.abs(a), 99.5) if np.isfinite(a).any() else 1.0
        return (-m, m) if m > 0 else (-1, 1)
    if kind in ("intensity", "amplitude", "volume", "depth"):
        lo, hi = np.nanpercentile(a, [0.2, 99.8]) if np.isfinite(a).any() else (0, 1)
        return (lo, hi if hi > lo else lo + 1)
    return (np.nanmin(a), np.nanmax(a))


def pick_panels(names, k):
    """Up to k panel names in their original order, always keeping the first (the measurement), the
    truth and the error map when there are more panels than room."""
    if len(names) <= k:
        return list(names)
    low = {n: n.lower() for n in names}
    must = [names[0]]
    must += [n for n in names if ("true" in low[n] or "truth" in low[n]) and n not in must][:1]
    must += [n for n in names if "error" in low[n] and n not in must][:1]
    rest = [n for n in names if n not in must]
    keep = set(must + rest[:max(k - len(must), 0)])
    return [n for n in names if n in keep]


def render(result, title="", panels=None, max_panels=6, curves=True, figsize_per=(3.0, 3.0), dpi=110):
    """Return a matplotlib Figure with up to `max_panels` image panels and (optionally) the first curve."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    names = list(result.panels) if panels is None else [k for k in panels if k in result.panels]
    if panels is None:
        names = pick_panels(names, max_panels)
    names = names[:max_panels]
    show_curve = curves and result.curves
    k = len(names) + (1 if show_curve else 0)
    cols = min(k, 4)
    rows = int(np.ceil(k / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(figsize_per[0] * cols, figsize_per[1] * rows), dpi=dpi, squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i, name in enumerate(names):
        p = result.panels[name]
        ax = axes.ravel()[i]
        a = _panel_image(p)
        ext = None if p.extent is None else (p.extent[0], p.extent[1], p.extent[3], p.extent[2])   # row 0 at the top
        if p.kind == "rgb":
            ax.imshow(np.clip(np.nan_to_num(a), 0, 1), extent=ext)
        else:
            lo, hi = _limits(p, a)
            im = ax.imshow(a, cmap=CMAPS.get(p.cmap or p.kind, "gray"), vmin=lo, vmax=hi, extent=ext,
                           aspect="equal" if p.lock_aspect() else "auto")
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.ax.tick_params(labelsize=6)
        ax.set_title(p.title or name, fontsize=8)
        xl, yl = p.labels()
        if xl or yl:
            ax.axis("on")
            ax.set_xlabel(xl, fontsize=7)
            ax.set_ylabel(yl, fontsize=7)
            ax.tick_params(labelsize=6)
    if show_curve:
        c = result.curves[0]
        ax = axes.ravel()[len(names)]
        ax.axis("on")
        for s, (x, y) in c.series.items():
            ax.plot(np.asarray(x), np.asarray(y), label=s, lw=1.2)
        if c.logy:
            ax.set_yscale("log")
        if c.logx:
            ax.set_xscale("log")
        ax.set_title(c.title, fontsize=8)
        ax.set_xlabel(c.xlabel, fontsize=7)
        ax.set_ylabel(c.ylabel, fontsize=7)
        ax.tick_params(labelsize=6)
        if len(c.series) > 1:
            ax.legend(fontsize=6)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def save_png(result, path, title="", **kw):
    import matplotlib.pyplot as plt
    fig = render(result, title, **kw)
    fig.savefig(path)
    plt.close(fig)
    return path
