"""Loader for the Cardiff b-tensor data (Afzali et al. 2021; scripts/fetch_cardiff.sh).

Two participants (``data1``, ``data2``), each with three runs on the 3T Connectom (3 mm isotropic, TE 88 ms):

* LTE: 10 b0 + shells b = 1, 2, 3, 4.5 (31 directions each) and 6, 7.5, 9, 10.5, 12 (61 each), 439 volumes
* PTE: the same table, 439 volumes; the bvecs repeat LTE's and are taken as the plane normals
* STE: 11 b0 + b = 0.2, 1, 2, 3, 4.5, 6, 7.5, 9, 10.5, 12 (6-30 rotations each), 182 volumes

b-values are the nominal ones in the bval files (ms/um^2 after scaling). There are no sidecars or waveforms.
The reversed-PE STE b0 is not used (no susceptibility correction).

``load_subject`` concatenates the runs into one series with one b-tensor per volume. With ``align=True`` it
rigidly registers each run's mean b0 to the LTE run's (mutual information, hhs._rigid) and resamples the
run; directions are rotated with it (B -> R B R^T; a no-op for STE). Within-run motion and eddy currents are
not corrected.
"""
from __future__ import annotations

import os
from pathlib import Path

import nibabel as nib
import numpy as np

from .protocol import Protocol

CARDIFF_ROOT = Path(os.environ.get("QTILAB_CARDIFF_ROOT", "data/cardiff/d.2021.0135083057"))
SUBJECTS = ("data1", "data2")
RUNS = (("LTE", "DTT2LTESTTE088s"), ("PTE", "DTT2PTETE088s"), ("STE", "DTT2STETE088s"))


def subject_runs(sub="data1", root=CARDIFF_ROOT):
    """[{shape, path, bval (N,) s/mm^2, bvec (3, N)}] in LTE, PTE, STE order."""
    out = []
    for shape, key in RUNS:
        (path,) = sorted((Path(root) / sub).glob(f"*{key}*a001.nii.gz"))
        stem = str(path)[: -len(".nii.gz")]
        out.append({"shape": shape, "path": path, "bval": np.loadtxt(stem + ".bval"),
                    "bvec": np.loadtxt(stem + ".bvec")})
    return out


def subject_protocol(sub="data1", root=CARDIFF_ROOT):
    """The subject's protocol from the tables alone (no images, no motion correction)."""
    runs = subject_runs(sub, root)
    shapes = sum([[r["shape"]] * len(r["bval"]) for r in runs], [])
    return Protocol.from_table(np.concatenate([r["bval"] for r in runs]),
                               np.concatenate([r["bvec"] for r in runs], axis=1), shapes)


def load_subject(sub="data1", root=CARDIFF_ROOT, align=True):
    """-> dict(data (X,Y,Z,N) float32, affine, protocol, shapes, bvals, bvecs, run_index (N,), motion)."""
    from .hhs import _rigid

    runs = subject_runs(sub, root)
    imgs = [nib.load(r["path"]) for r in runs]
    affine = imgs[0].affine
    vols = [np.asarray(i.dataobj, dtype=np.float32) for i in imgs]
    motion = [None] * len(runs)
    if align:
        from dipy.align.imaffine import AffineMap
        ref = vols[0][..., runs[0]["bval"] < 50].mean(-1)
        for k in range(1, len(runs)):
            mov = vols[k][..., runs[k]["bval"] < 50].mean(-1)
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
    shapes = sum([[r["shape"]] * len(r["bval"]) for r in runs], [])
    run_index = np.concatenate([[k] * len(r["bval"]) for k, r in enumerate(runs)])
    if align:          # FSL-convention bvecs are in voxel axes; the registration rotation is in world axes
        A = affine[:3, :3] / np.linalg.norm(affine[:3, :3], axis=0)
        flip = np.diag([-1.0 if np.linalg.det(A) > 0 else 1.0, 1, 1])
        off = 0
        for r in runs:
            n = len(r["bval"])
            if "R" in r and r["shape"] != "STE":
                w = r["R"].T @ (A @ flip @ bvecs[:, off:off + n])
                bvecs[:, off:off + n] = flip @ np.linalg.inv(A) @ w
            off += n
    prot = Protocol.from_table(bvals, bvecs, shapes)
    return {"data": data, "affine": affine, "protocol": prot, "shapes": shapes, "bvals": bvals,
            "bvecs": bvecs, "run_index": run_index, "motion": motion}
