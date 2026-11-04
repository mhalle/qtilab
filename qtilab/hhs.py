"""Loader for the Hormone Health Study (OpenNeuro ds005360).

Each session has 8 diffusion runs, each one or two shells:

* linear (LTE): b0/100/500, 1000, 1500, 2000
* spherical (STE): b0, 100/500, 1000, 1500

All runs are phase-encoded j-, so there is no reversed pair. Runs 2-4 and 7-8 have no b = 0
volume, so between-run differences in position or intensity scaling cannot be read off b0s.

``load_session`` concatenates the runs into one series with one b-tensor per volume. With
``align=True`` it rigidly registers each run's mean image to a session reference (mutual
information, so runs at different b-values can be aligned) and resamples the run. The b-tensors
are rotated with the run (B -> R B R^T); for STE that is a no-op.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import nibabel as nib
import numpy as np

from .protocol import Protocol

HHS_ROOT = Path(os.environ.get("QTILAB_HHS_ROOT", "data/hhs"))
RUN_RE = re.compile(r"acq-(linear|spherical)_run-(\d+)_dwi\.nii(\.gz)?$")


def session_runs(sub, ses, root=HHS_ROOT):
    d = Path(root) / f"sub-{sub}" / f"ses-{ses}" / "dwi"
    runs = []
    for p in sorted(d.glob("*_dwi.nii*")):
        m = RUN_RE.search(p.name)
        if not m:
            continue
        stem = str(p).split(".nii")[0]
        runs.append({
            "path": p, "kind": m.group(1), "run": int(m.group(2)),
            "bval": np.atleast_1d(np.loadtxt(stem + ".bval")),
            "bvec": np.loadtxt(stem + ".bvec").reshape(3, -1),
            "json": json.loads(Path(stem + ".json").read_text()),
        })
    return sorted(runs, key=lambda r: r["run"])


def _rigid(static, moving, static_aff, moving_aff):
    """Rigid MI registration of moving -> static. Returns (4x4 map in world coords, transformed image)."""
    from dipy.align import affine_registration
    _, M = affine_registration(moving, static, moving_affine=moving_aff, static_affine=static_aff,
                               nbins=32, metric="MI", pipeline=["center_of_mass", "translation", "rigid"],
                               level_iters=[1000, 100, 10], sigmas=[3.0, 1.0, 0.0], factors=[4, 2, 1])
    return M


def load_session(sub, ses, root=HHS_ROOT, align=True):
    """-> dict(data (X,Y,Z,N) float32, affine, protocol, shapes, run_index (N,), motion (list per run))."""
    runs = session_runs(sub, ses, root)
    imgs = [nib.load(r["path"]) for r in runs]
    affine = imgs[0].affine
    vols = [np.asarray(i.dataobj, dtype=np.float32) for i in imgs]
    vols = [v[..., None] if v.ndim == 3 else v for v in vols]

    motion = [None] * len(runs)
    if align:
        from dipy.align.imaffine import AffineMap
        ref = vols[0].mean(-1)                 # run 1: b0/100/500 LTE, best SNR and contrast
        for k in range(1, len(runs)):
            mov = vols[k].mean(-1)
            M = _rigid(ref, mov, affine, imgs[k].affine)
            amap = AffineMap(M, domain_grid_shape=ref.shape, domain_grid2world=affine,
                             codomain_grid_shape=mov.shape, codomain_grid2world=imgs[k].affine)
            vols[k] = np.stack([amap.transform(vols[k][..., i]) for i in range(vols[k].shape[-1])], -1).astype(np.float32)
            R = M[:3, :3]
            motion[k] = {"translation_mm": M[:3, 3].tolist(),
                         "rotation_deg": float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))}
            runs[k]["R"] = R

    data = np.concatenate(vols, axis=-1)
    bvals = np.concatenate([r["bval"] for r in runs])
    bvecs = np.concatenate([r["bvec"] for r in runs], axis=1)
    shapes = sum([["LTE" if r["kind"] == "linear" else "STE"] * len(r["bval"]) for r in runs], [])
    run_index = np.concatenate([[r["run"]] * len(r["bval"]) for r in runs])
    # bvecs in FSL convention are in voxel axes; the registration rotation is in world axes.
    # For the small rotations here we rotate the (world-converted) LTE directions and convert back.
    if align:
        A = affine[:3, :3] / np.linalg.norm(affine[:3, :3], axis=0)
        flip = np.diag([-1.0 if np.linalg.det(A) > 0 else 1.0, 1, 1])  # FSL bvec x-flip for radiological storage
        off = 0
        for r in runs:
            n = len(r["bval"])
            if "R" in r and r["kind"] == "linear":
                w = A @ flip @ bvecs[:, off:off + n]
                w = r["R"].T @ w                                # moving -> reference frame
                bvecs[:, off:off + n] = flip @ np.linalg.inv(A) @ w
            off += n
    prot = Protocol.from_table(bvals, bvecs, shapes)
    return {"data": data, "affine": affine, "protocol": prot, "shapes": shapes, "bvals": bvals,
            "bvecs": bvecs, "run_index": run_index, "motion": motion}
