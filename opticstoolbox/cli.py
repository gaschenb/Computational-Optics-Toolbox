"""Command line interface.

    opticstoolbox list [--area AREA]
    opticstoolbox info EXPERIMENT
    opticstoolbox run EXPERIMENT [--method M] [--set k=v ...] [--solver k=v ...] [--seed S]
                                 [--input key=path ...] [--meta meta.json] [--out DIR] [--png]
    opticstoolbox gui
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .core import registry as R


def _value(s):
    low = s.lower()
    if low in ("true", "false", "yes", "no", "on", "off"):
        return low in ("true", "yes", "on")
    for t in (int, float):
        try:
            return t(s)
        except ValueError:
            pass
    return s


def _pairs(items):
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"expected key=value, got {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = _value(v.strip())
    return out


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def cmd_list(args):
    R.load_all()
    for area, keys in R.AREAS.items():
        if args.area and args.area.lower() not in area.lower():
            continue
        print(area)
        for k in keys:
            e = R.EXPERIMENTS[k]
            tag = "" if e.inputs else "  (simulation only)"
            print(f"  {k:18s} {e.title}{tag}")
    return 0


def _param_lines(params):
    for p in params:
        rng = f"[{_fmt(p.lo)} … {_fmt(p.hi)}]" if p.lo is not None and p.kind in ("int", "float") else ""
        ch = " | ".join(p.choices) if p.choices else ""
        unit = f" {p.unit}" if p.unit else ""
        print(f"    {p.key:16s} = {_fmt(p.default)}{unit:6s} {rng or ch}   {p.label}")


def cmd_info(args):
    R.load_all()
    e = R.get(args.experiment)
    print(f"{e.title}  [{e.key}]  —  {e.area}\n")
    print(e.blurb + "\n")
    print("Equations (mathtext):")
    for q in e.equations:
        print("   ", q)
    print("\nScene / physical parameters (--set):")
    _param_lines(e.scene)
    print("\nSolver parameters (--solver, --method):")
    _param_lines(e.solver)
    if e.inputs:
        print("\nUser data (--input key=path):")
        for i in e.inputs:
            print(f"    {i.key:16s} {i.kind:7s} {'(optional) ' if i.optional else ''}{i.label}")
        if e.meta:
            print("  physical parameters read with user data (--set or --meta):", ", ".join(e.meta))
    else:
        print("\nThis experiment runs on simulated data only.")
    return 0


def cmd_run(args):
    from .core import io
    R.load_all()
    e = R.get(args.experiment)
    scene = _pairs(args.set)
    solver = _pairs(args.solver)
    if args.method:
        solver["method"] = args.method
    if args.input:
        if not e.inputs:
            raise SystemExit(f"{e.key} has no user-data inputs")
        kinds = {i.key: i.kind for i in e.inputs}
        arrays = {}
        for k, path in _pairs(args.input).items():
            if k not in kinds:
                raise SystemExit(f"unknown input {k!r}; expected one of {', '.join(kinds)}")
            path = str(path)
            paths = path.split(";") if ";" in path else path
            arrays[k] = io.read_input(paths, kinds[k])
        meta = dict(scene)
        if args.meta:
            meta.update(io.read_meta(args.meta))
        data = e.make_user_data(arrays, meta)
        source = "user data"
    else:
        data = e.make_data(scene, seed=args.seed)
        source = "simulated data"

    def progress(msg):
        i, n, _ = msg
        if not args.quiet and n:
            sys.stderr.write(f"\r  {e.key}: {i}/{n}")
            sys.stderr.flush()

    mon = R.Monitor(on_progress=progress, preview_every=1e9)
    res = e.run(data, solver, mon)
    if not args.quiet:
        sys.stderr.write("\r" + " " * 40 + "\r")
    method = e.resolve("solver", solver).get("method", "")
    print(f"{e.title} — {method or 'default'} on {source}  ({res.runtime:.2f} s)")
    for k, v in res.metrics.items():
        print(f"  {k:40s} {_fmt(v)}")
    notes = res.notes if isinstance(res.notes, (list, tuple)) else [res.notes] if res.notes else []
    for note in notes:
        print("  note:", note)
    if args.out:
        files = io.save_result(res, args.out, prefix=e.key)
        if args.png:
            from .render import save_png
            files.append(save_png(res, os.path.join(args.out, f"{e.key}.png"), f"{e.title} — {method}"))
        print(f"wrote {len(files)} files to {args.out}")
    if args.json:
        print(json.dumps({k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in res.metrics.items()}))
    return 0


def cmd_gui(args):
    from .gui.app import main
    return main()


def build_parser():
    ap = argparse.ArgumentParser(prog="opticstoolbox", description="Computational optics toolbox")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list", help="list experiments by area")
    p.add_argument("--area")
    p.set_defaults(func=cmd_list)
    p = sub.add_parser("info", help="describe an experiment, its parameters and inputs")
    p.add_argument("experiment")
    p.set_defaults(func=cmd_info)
    p = sub.add_parser("run", help="run an experiment on simulated or user data")
    p.add_argument("experiment")
    p.add_argument("--method")
    p.add_argument("--set", nargs="*", metavar="KEY=VALUE", help="scene / physical parameters")
    p.add_argument("--solver", nargs="*", metavar="KEY=VALUE", help="solver parameters")
    p.add_argument("--input", nargs="*", metavar="KEY=PATH", help="user data files (a folder for stacks; ';' joins files)")
    p.add_argument("--meta", help="JSON file of physical parameters (keys may end in _nm/_um/_mm/_m)")
    p.add_argument("--seed", type=int, default=None, help="random seed (same as --set seed=S)")
    p.add_argument("--out", help="folder for .npy panels + summary.json")
    p.add_argument("--png", action="store_true", help="also save a PNG overview (needs --out)")
    p.add_argument("--json", action="store_true", help="print metrics as JSON")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("gui", help="start the graphical interface")
    p.set_defaults(func=cmd_gui)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyError as ex:
        print(f"error: unknown experiment {ex}; see 'opticstoolbox list'", file=sys.stderr)
        return 2
    except ValueError as ex:
        print(f"error: {ex}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
