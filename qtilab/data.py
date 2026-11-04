"""Loaders for the open tensor-valued datasets.

DiB: Szczepankiewicz, Hoge & Westin 2019 (Data in Brief), the Boito/QTI+ subsamples.
Encoding-shape order per volume is copied from DIPY's read_DiB_* functions.
"""
from __future__ import annotations

import os
from pathlib import Path

import nibabel as nib
import numpy as np

from .protocol import Protocol

DIB_ROOT = Path(os.environ.get("QTILAB_DIB_ROOT", "data/dib/DATA/brain/NII_Boito_SubSamples"))

_SHAPES = {
    "DiB_217_lte_pte_ste": ["LTE"] * 13 + ["LTE"] * 10 + ["PTE"] * 10 + ["STE"] * 10
    + ["LTE"] * 10 + ["PTE"] * 10 + ["STE"] * 10 + ["LTE"] * 16 + ["PTE"] * 16 + ["STE"] * 10
    + ["LTE"] * 46 + ["PTE"] * 46 + ["STE"] * 10,
    "DiB_70_lte_pte_ste": ["LTE"] * 1 + ["LTE"] * 4 + ["PTE"] * 3 + ["STE"] * 3 + ["STE"] * 4
    + ["LTE"] * 8 + ["PTE"] * 5 + ["STE"] * 5 + ["LTE"] * 21 + ["PTE"] * 10 + ["STE"] * 6,
}


def load_dib(name="DiB_217_lte_pte_ste", root=DIB_ROOT):
    """-> (data float32 (X,Y,Z,N), mask bool (X,Y,Z), affine, Protocol, bvals, bvecs, shapes)."""
    d = Path(root) / name
    parts = sorted(d.glob(f"{name}*.nii.gz"))
    imgs = [nib.load(p) for p in parts]
    data = np.concatenate([np.asarray(i.dataobj, dtype=np.float32) for i in imgs], axis=3)
    affine = imgs[0].affine
    bvals = np.loadtxt(d / f"bval_{name}.bval")
    bvecs = np.loadtxt(d / f"bvec_{name}.bvec")
    shapes = _SHAPES[name]
    assert data.shape[3] == len(bvals) == len(shapes), (data.shape, len(bvals), len(shapes))
    mask = np.asarray(nib.load(Path(root) / "DiB_mask.nii.gz").dataobj) > 0
    return data, mask, affine, Protocol.from_table(bvals, bvecs, shapes), bvals, bvecs, shapes
