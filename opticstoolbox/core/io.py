"""Reading the user's data: images, stacks, tables and physical metadata.

Supported: TIFF (8/16/32-bit, multi-page stacks; tifffile if installed, else Pillow), PNG / JPG / BMP,
NumPy .npy / .npz, MATLAB .mat (v5 via SciPy, v7.3 via h5py), HDF5 .h5 / .hdf5 / .cxi (h5py),
CSV / TXT tables, and a JSON side-car with physical parameters (wavelength, pixel pitch, …).
A folder or a list of files is read as a stack in natural sort order."""
from __future__ import annotations

import json
import os
import re

import numpy as np

IMAGE_EXT = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".gif")
ARRAY_EXT = (".npy", ".npz", ".mat", ".h5", ".hdf5", ".cxi")
TABLE_EXT = (".csv", ".txt", ".dat")
ALL_EXT = IMAGE_EXT + ARRAY_EXT + TABLE_EXT
FILE_FILTER = ("Data (*.tif *.tiff *.png *.jpg *.jpeg *.bmp *.npy *.npz *.mat *.h5 *.hdf5 *.cxi *.csv *.txt);;"
               "All files (*)")


def _natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _to_gray(a):
    a = np.asarray(a)
    if a.ndim == 3 and a.shape[-1] in (3, 4) and a.shape[0] > 4:
        a = a[..., :3].astype(float) @ np.array([0.2126, 0.7152, 0.0722])
    return a


def read_tiff(path):
    try:
        import tifffile
        return np.asarray(tifffile.imread(path))
    except ImportError:
        from PIL import Image, ImageSequence
        im = Image.open(path)
        frames = [np.asarray(f.copy()) for f in ImageSequence.Iterator(im)]
        return frames[0] if len(frames) == 1 else np.stack(frames)


def read_image(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        a = read_tiff(path)
    else:
        from PIL import Image
        a = np.asarray(Image.open(path))
    if a.ndim == 3 and a.shape[-1] in (3, 4):
        a = _to_gray(a)
    return np.asarray(a, dtype=float)


def _first_array(d, key=None):
    if key is not None:
        return np.asarray(d[key])
    best = None
    for k, v in d.items():
        if k.startswith("__"):
            continue
        v = np.asarray(v)
        if v.dtype.kind in "biufc" and (best is None or v.size > best.size):
            best = v
    if best is None:
        raise ValueError("no numeric array found")
    return best


def _h5_arrays(path):
    import h5py
    out = {}
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.dtype.kind in "biufc" and obj.size > 0:
                out[name] = obj[()]
        f.visititems(visit)
    return out


def read_arrays(path):
    """All numeric arrays in a container file, as {name: array}."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return {"data": np.load(path, allow_pickle=False)}
    if ext == ".npz":
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    if ext == ".mat":
        try:
            from scipy.io import loadmat
            d = loadmat(path)
            return {k: np.asarray(v) for k, v in d.items() if not k.startswith("__")
                    and isinstance(v, np.ndarray) and v.dtype.kind in "biufc"}
        except NotImplementedError:      # v7.3 = HDF5
            return {k: np.asarray(v).T for k, v in _h5_arrays(path).items()}
    if ext in (".h5", ".hdf5", ".cxi"):
        return _h5_arrays(path)
    if ext in IMAGE_EXT:
        return {"data": read_image(path)}
    if ext in TABLE_EXT:
        return {"data": read_table(path)}
    raise ValueError(f"unsupported file type {ext!r}")


CXI_HINTS = {"data": ("entry_1/data_1/data", "entry_1/instrument_1/detector_1/data"),
             "positions": ("entry_1/sample_1/geometry_1/translation", "entry_1/data_1/translation")}


def read_array(path, key=None):
    """One array from any supported file (the largest numeric one unless key is given)."""
    d = read_arrays(path)
    if key is None and os.path.splitext(path)[1].lower() == ".cxi":
        for h in CXI_HINTS["data"]:
            if h in d:
                return np.asarray(d[h], float)
    a = _first_array(d, key)
    if a.ndim == 3 and a.shape[-1] in (3, 4) and a.shape[0] > 4:
        a = _to_gray(a)
    return a if np.iscomplexobj(a) else np.asarray(a, float)


def read_table(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        first = fh.readline()
    delim = "," if "," in first else None
    skip = 0 if re.match(r"^\s*[-+0-9.eE]", first) else 1
    return np.atleast_2d(np.loadtxt(path, delimiter=delim, skiprows=skip))


def read_stack(paths):
    """A folder, a glob list, or a single multi-frame file → (k, ny, nx) array."""
    if isinstance(paths, (str, os.PathLike)):
        p = os.fspath(paths)
        if os.path.isdir(p):
            files = sorted([os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith(IMAGE_EXT + (".npy",))],
                           key=_natural_key)
            if not files:
                raise ValueError(f"no images in {p}")
            return np.stack([read_array(f) for f in files])
        a = read_array(p)
        if a.ndim > 3:                     # e.g. (angles, phases, y, x) → frames in C order
            a = a.reshape(-1, *a.shape[-2:])
        return a if a.ndim == 3 else a[None]
    files = sorted([os.fspath(f) for f in paths], key=_natural_key)
    if len(files) == 1:
        return read_stack(files[0])
    return np.stack([read_array(f) for f in files])


def read_input(path_or_paths, kind="image"):
    """Read one registry Input (image | stack | complex | table)."""
    if kind == "stack":
        return read_stack(path_or_paths)
    if isinstance(path_or_paths, (list, tuple)):
        path_or_paths = path_or_paths[0]
    if kind == "table":
        ext = os.path.splitext(path_or_paths)[1].lower()
        return read_table(path_or_paths) if ext in TABLE_EXT else read_array(path_or_paths)
    a = read_array(path_or_paths)
    if kind == "image" and a.ndim == 3:
        a = a.mean(0) if a.shape[0] < 8 else a[0]
    return a


def read_meta(path):
    """JSON side-car → dict. Keys may carry units: "wavelength_nm", "pixel_um", "z_mm" are converted
    to micrometres; plain keys are taken as-is."""
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    out = {}
    conv = {"_nm": 1e-3, "_um": 1.0, "_mm": 1e3, "_m": 1e6}
    for k, v in d.items():
        for suf, f in conv.items():
            if k.endswith(suf) and isinstance(v, (int, float)):
                out[k[: -len(suf)]] = v * f
                break
        else:
            out[k] = v
    return out


def save_result(result, folder, prefix="result"):
    """Write every panel as .npy (+ .png preview) and metrics/curves as JSON."""
    os.makedirs(folder, exist_ok=True)
    written = []
    for name, p in result.panels.items():
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
        f = os.path.join(folder, f"{prefix}_{safe}.npy")
        np.save(f, p.data)
        written.append(f)
    meta = {"metrics": {k: (float(v) if isinstance(v, (int, float, np.integer, np.floating)) else str(v))
                         for k, v in result.metrics.items()}, "runtime_s": result.runtime,
            "curves": {c.title: {s: [np.asarray(xy[0]).tolist(), np.asarray(xy[1]).tolist()]
                                 for s, xy in c.series.items()} for c in result.curves}}
    f = os.path.join(folder, f"{prefix}_summary.json")
    with open(f, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    written.append(f)
    return written
