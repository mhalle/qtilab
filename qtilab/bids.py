"""Finding a diffusion series' .bval, .bvec and sidecar the way tractline does, and its b-tensor shape.

``_stem``, ``_sib``, ``_entities``, ``_root``, ``_inherited`` and ``_table`` are copied from tractline v0.11.0
(tractline/bids.py; same author, Apache-2.0) so that both programs find a series' files the same way: inside a
BIDS dataset (a folder above with dataset_description.json) by BIDS's inheritance principle, outside one (dcm2niix's
flat output) the file beside the series with its own name.

The b-tensor shape has no BIDS field. It is read from the ``acq-`` entity of the file name (``acq-linear``,
``acq-planar``, ``acq-spherical``, or LTE / PTE / STE; e.g. OpenNeuro ds005360) or from a ``BTensorShape`` sidecar
key (LTE, PTE, STE, or b_delta 1, -0.5, 0), and can always be given on the command line instead.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SHAPES = ("LTE", "PTE", "STE")
_ACQ = {"linear": "LTE", "lte": "LTE", "lin": "LTE", "planar": "PTE", "pte": "PTE", "plane": "PTE",
        "spherical": "STE", "ste": "STE", "sph": "STE", "isotropic": "STE"}


def _stem(p: Path) -> str:
    return p.name[:-7] if p.name.endswith(".nii.gz") else p.name[:-4] if p.name.endswith(".nii") else p.stem


def _sib(p: Path, ext: str) -> Path:
    return p.with_name(_stem(p) + ext)


def _entities(stem: str) -> tuple[dict, str]:
    parts = stem.split("_")
    return dict(x.split("-", 1) for x in parts[:-1] if "-" in x), parts[-1]


def _root(p: Path) -> Path | None:
    """The BIDS dataset p is in (the nearest folder above it with dataset_description.json), or None."""
    for d in p.parents:
        if (d / "dataset_description.json").exists():
            return d
    return None


def _inherited(p: Path, ext: str) -> list[Path]:
    """The files of this extension that apply to p, least specific first: inside a BIDS dataset, by BIDS's
    inheritance principle; outside one (dcm2niix's flat output), only the file beside it of its own name."""
    root = _root(p)
    if root is None:
        f = _sib(p, ext)
        return [f] if f.exists() else []
    ents, suffix = _entities(_stem(p))
    dirs = [root] + [d for d in reversed(p.parents) if root in d.parents]
    out = []
    for d in dirs:
        found = []
        for f in d.glob(f"*{ext}"):
            fe, fs = _entities(f.name[: -len(ext)])
            if fs == suffix and all(ents.get(k) == v for k, v in fe.items()):
                found.append((len(fe), f))
        out += [f for _, f in sorted(found)]
    return out


def _table(p: Path, ext: str):
    files = _inherited(p, ext)
    return np.loadtxt(files[-1], ndmin=1 if ext == ".bval" else 2) if files else None


def sidecar(p: Path) -> dict:
    """The series' sidecar merged with those it inherits ({} if none). An unreadable one is an error naming it."""
    merged = {}
    for f in _inherited(p, ".json"):
        try:
            d = json.loads(f.read_bytes())
        except (OSError, ValueError) as e:
            raise ValueError(f"{f}: the sidecar cannot be read as JSON ({e})") from e
        if isinstance(d, dict):
            merged.update(d)
    return merged


def shape_of(value) -> str | None:
    """LTE / PTE / STE from a shape name, an acq label or a b_delta value; None if it is none of them."""
    if value is None:
        return None
    s = str(value).strip()
    if s.upper() in SHAPES:
        return s.upper()
    if s.lower() in _ACQ:
        return _ACQ[s.lower()]
    try:
        v = float(s)
    except ValueError:
        return None
    near = {1.0: "LTE", -0.5: "PTE", 0.0: "STE"}
    k = min(near, key=lambda c: abs(c - v))
    return near[k] if abs(k - v) <= 0.1 else None


def btens_from_bids(p: Path) -> tuple[str | None, str]:
    """(shape, where it came from) for a whole series, from its sidecar's BTensorShape or its acq- entity."""
    side = sidecar(p)
    if "BTensorShape" in side and shape_of(side["BTensorShape"]):
        return shape_of(side["BTensorShape"]), "sidecar BTensorShape"
    acq = _entities(_stem(p))[0].get("acq")
    if acq is not None:
        if acq.upper() in SHAPES or acq.lower() in _ACQ:              # names only: acq-1 is not b_delta 1
            return shape_of(acq), f"acq-{acq}"
    return None, "none"
