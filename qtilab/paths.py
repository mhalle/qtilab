"""Where intermediate artifacts live. Override the root with QTILAB_WORK (default: ./work).

Data roots are separate: QTILAB_DIB_ROOT (brain subsamples), QTILAB_DIB_DATA (the DiB repo's DATA folder,
for phantoms) and QTILAB_HHS_ROOT (Hormone Health Study).
"""
from __future__ import annotations

import os
from pathlib import Path

WORK = Path(os.environ.get("QTILAB_WORK", "work"))
DIB_DATA = Path(os.environ.get("QTILAB_DIB_DATA", "data/dib/DATA"))


def aligned_session(sub="01", ses="01") -> Path:
    """Run-aligned, concatenated session (written by scripts/prepare_session.py)."""
    return WORK / f"hhs_sub{sub}_ses{ses}_aligned.npz"


def classical_fit(sub="01", ses="01") -> Path:
    """Brain mask and constrained-QTI scalars for that session (scripts/prepare_session.py)."""
    return WORK / f"hhs_sub{sub}_ses{ses}_qti.npz"


def posterior_slice(tag="_v2", sub="01", ses="01") -> Path:
    """Network posterior on slice 20 (scripts/posterior_maps_hhs.py)."""
    return WORK / f"hhs_sub{sub}_ses{ses}_posterior{tag}.npz"
