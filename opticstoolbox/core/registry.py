"""Experiment registry shared by the GUI, the CLI, the gallery and the tests.

An *experiment* is one imaging problem: a forward model that can simulate data, a list of
reconstruction methods, and a description of the user data it accepts. Everything the GUI
shows is generated from this description, so adding an experiment needs no GUI code.

    data   = exp.simulate(scene_params, rng)          # dict of arrays + physical metadata
    data   = exp.load(files, meta_params)             # same dict from the user's own files
    result = exp.solve(data, solver_params, monitor)  # Result: panels, curves, metrics
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


class Stopped(Exception):
    """Raised inside a solver when the monitor reports a cancellation."""


@dataclass
class Param:
    key: str
    label: str
    default: object
    lo: float | None = None
    hi: float | None = None
    kind: str = "float"          # float | int | choice | bool
    choices: tuple = ()
    log: bool = False
    tip: str = ""
    unit: str = ""

    def __post_init__(self):
        if self.choices and self.kind == "float":
            self.kind = "choice"
        elif isinstance(self.default, bool) and self.kind == "float":
            self.kind = "bool"
        elif isinstance(self.default, int) and not isinstance(self.default, bool) and self.kind == "float" \
                and self.lo is not None and float(self.lo).is_integer():
            self.kind = "int"

    def clip(self, v):
        if self.kind == "int":
            v = int(round(float(v)))
        elif self.kind == "float":
            v = float(v)
        elif self.kind == "bool":
            v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")
        elif self.kind == "choice":
            v = str(v)
            if v not in self.choices:
                raise ValueError(f"{self.key}: {v!r} not in {self.choices}")
            return v
        if self.lo is not None and self.kind in ("int", "float"):
            v = max(v, type(v)(self.lo))
        if self.hi is not None and self.kind in ("int", "float"):
            v = min(v, type(v)(self.hi))
        return v


@dataclass
class Input:
    """One user-supplied array. kind: image | stack | table | complex."""
    key: str
    label: str
    kind: str = "image"
    optional: bool = False
    tip: str = ""


import re as _re

_NO_CAP = {"arg", "exp", "sin", "cos", "abs"}
_DISPLAY = {"usaf": "USAF-1951 target", "siemens": "Siemens star", "shepp-logan": "Shepp–Logan phantom",
            "text": "Text", "letter": "Letter", "cells": "Cells", "blobs": "Blobs", "beads": "Beads",
            "texture": "Natural texture"}
_SHORT = {"in", "on", "at", "by", "no", "of", "to", "up", "an"}


def nice(s):
    """Display text: the ground truth is called "Ground Truth" (never "truth" / "true …"), and text starts
    with a capital letter unless it starts with a symbol, variable or unit (x, Δn, q_x, |u|², µm …)."""
    if not isinstance(s, str) or not s:
        return s
    if s in _DISPLAY:
        return _DISPLAY[s]
    t = _re.sub(r"(?i)\bground truth\b", "\x00", s)
    t = _re.sub(r"(?i)\btruth\b", "\x00", t)
    t = _re.sub(r"(?i)\btrue\b", "\x00", t)
    t = t.replace("\x00", "Ground Truth")
    m = _re.match(r"([A-Za-z]+)(?![A-Za-z_(])", t)
    w = m.group(1) if m else ""
    if w.islower() and w not in _NO_CAP and (len(w) >= 3 or w in _SHORT):
        t = t[0].upper() + t[1:]
    return t


@dataclass
class Panel:
    """An image to display. kind decides the colormap: intensity | amplitude | phase | depth | signed |
    rgb | mask | volume (3-D array, shown as orthogonal slices)."""
    data: np.ndarray
    kind: str = "intensity"
    title: str = ""
    extent: tuple | None = None        # (x0, x1, y0, y1) in physical units, for axis labels
    unit: str = ""
    levels: tuple | None = None
    xlabel: str = ""                   # axis labels (shown with tick marks); empty = plain image
    ylabel: str = ""
    aspect: bool | None = None         # lock 1:1 pixels; None = lock unless the extent's aspect is far from the array's
    cmap: str = ""                     # colour scheme as a kind name (e.g. "signed" for a signed volume); "" = from kind
    zextent: tuple | None = None       # volumes: (z0, z1) of the first and last slice, in physical units
    zlabel: str = ""                   # volumes: name of the slider axis, e.g. "z (µm)" or "λ (nm)"
    surface: bool = False              # 2-D map that can be shown as a rotatable 3-D surface (phase, height)

    def __post_init__(self):
        self.title, self.xlabel, self.ylabel, self.zlabel = map(nice, (self.title, self.xlabel, self.ylabel, self.zlabel))

    def labels(self):
        """Axis labels: explicit ones, else x / y with the unit when a physical extent is given."""
        if self.xlabel or self.ylabel:
            return self.xlabel, self.ylabel
        if self.extent is not None:
            u = f" ({self.unit})" if self.unit else ""
            return f"x{u}", f"y{u}"
        return "", ""

    def lock_aspect(self):
        if self.aspect is not None:
            return bool(self.aspect)
        if self.extent is None:
            return True
        a = np.asarray(self.data)
        ny, nx = (a.shape[-2:] if self.kind == "volume" else a.shape[:2])
        x0, x1, y0, y1 = self.extent
        r = (abs(y1 - y0) / max(abs(x1 - x0), 1e-30)) / (ny / max(nx, 1))
        return 1 / 3 < r < 3


@dataclass
class Curve:
    """A line plot with one or more named series: {name: (x, y)}."""
    title: str
    series: dict
    xlabel: str = ""
    ylabel: str = ""
    logy: bool = False
    logx: bool = False

    def __post_init__(self):
        self.title, self.xlabel, self.ylabel = map(nice, (self.title, self.xlabel, self.ylabel))
        self.series = {nice(k): v for k, v in self.series.items()}


@dataclass
class Result:
    panels: dict = field(default_factory=dict)     # name -> Panel
    curves: list = field(default_factory=list)     # list[Curve]
    metrics: dict = field(default_factory=dict)    # name -> float
    notes: str = ""
    runtime: float = 0.0


class Monitor:
    """Progress / cancellation channel passed to solvers.

    `step(i, n, preview=None)` reports progress (and optionally an intermediate Result for live
    display); it raises Stopped when the caller has cancelled."""

    def __init__(self, on_progress: Callable | None = None, stopped: Callable | None = None,
                 preview_every: float = 0.08):
        self._cb, self._stopped = on_progress, stopped
        self._last = 0.0
        self.preview_every = preview_every

    def check(self):
        if self._stopped is not None and self._stopped():
            raise Stopped

    def want_preview(self):
        return self._cb is not None and time.perf_counter() - self._last > self.preview_every

    def step(self, i, n, preview=None):
        self.check()
        if self._cb is not None:
            now = time.perf_counter()
            if preview is not None and callable(preview):
                if now - self._last > self.preview_every:
                    self._last = now
                    self._cb((i, n, preview()))
                    return
                preview = None
            self._cb((i, n, preview))


NULL_MONITOR = Monitor()


@dataclass
class Experiment:
    key: str
    title: str
    area: str
    blurb: str
    equations: list
    scene: list                       # list[Param] for simulation
    solver: list                      # list[Param] for reconstruction (first is usually 'method')
    simulate: Callable                # (params, rng) -> data dict
    solve: Callable                   # (data, params, monitor) -> Result
    inputs: list = field(default_factory=list)      # list[Input] for user data
    meta: tuple = ()                  # scene keys that are needed with user data (physical params)
    load: Callable | None = None      # (arrays: dict, meta: dict) -> data dict; default: merge
    live: bool = True                 # fast enough to re-run on every edit
    method_equations: dict = field(default_factory=dict)  # method -> extra equations
    example: str = ""                 # one-line description of the shipped example

    # ---------------------------------------------------------------- parameter helpers
    def defaults(self, which="scene"):
        return {p.key: p.default for p in getattr(self, which)}

    def resolve(self, which, values=None):
        out = self.defaults(which)
        specs = {p.key: p for p in getattr(self, which)}
        for k, v in (values or {}).items():
            if k in specs:
                out[k] = specs[k].clip(v)
        return out

    def methods(self):
        for p in self.solver:
            if p.key == "method":
                return list(p.choices)
        return []

    def meta_params(self):
        return [p for p in self.scene if p.key in self.meta]

    # ---------------------------------------------------------------- running
    def make_data(self, scene=None, seed=None):
        p = self.resolve("scene", scene)
        rng = np.random.default_rng(int(p.get("seed", 0)) if seed is None else int(seed))
        d = self.simulate(p, rng)
        d.setdefault("params", p)
        return d

    def make_user_data(self, arrays, meta=None):
        m = self.resolve("scene", meta)
        missing = [i.key for i in self.inputs if not i.optional and i.key not in arrays]
        if missing:
            raise ValueError(f"missing input(s): {', '.join(missing)}")
        if self.load is not None:
            d = self.load(dict(arrays), m)
        else:
            d = dict(arrays)
        d.setdefault("params", m)
        d["user"] = True
        return d

    def run(self, data, solver=None, monitor=None):
        p = self.resolve("solver", solver)
        t0 = time.perf_counter()
        r = self.solve(data, p, monitor or NULL_MONITOR)
        r.runtime = time.perf_counter() - t0
        return r


EXPERIMENTS: dict = {}
AREAS: dict = {}            # area -> list of experiment keys, in registration order


def register(exp: Experiment):
    if exp.key in EXPERIMENTS:
        raise KeyError(f"duplicate experiment {exp.key}")
    EXPERIMENTS[exp.key] = exp
    AREAS.setdefault(exp.area, []).append(exp.key)
    return exp


def get(key) -> Experiment:
    load_all()
    try:
        return EXPERIMENTS[key]
    except KeyError:
        raise KeyError(f"unknown experiment {key!r}; try one of: {', '.join(EXPERIMENTS)}") from None


_LOADED = False


def load_all():
    global _LOADED
    if not _LOADED:
        _LOADED = True
        from .. import experiments  # noqa: F401  (registers everything)
    return EXPERIMENTS
