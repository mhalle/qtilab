"""Rotation-invariant QTI scalars from a 28-parameter vector, batched.

Formulas follow Westin et al. 2016 and match dipy.reconst.qti.QtiFit property by property.
"""
from __future__ import annotations

import torch

from .tensors import mat6, projection_tensors


def _inner(A, B):
    return (A * B).sum(dim=(-1, -2))


MD_MIN = 1e-6   # um^2/ms


def collapsed(sc: dict[str, torch.Tensor], md_min: float = MD_MIN) -> torch.Tensor:
    """Boolean mask of voxels whose fit collapsed to <D> = 0 (MD <= ``md_min``).

    This happens where the signal shows no attenuation above the noise floor (low-signal voxels at the
    brain-mask edge; 5 of 95,806 in HHS sub-01 ses-01, b0 ~50 vs a brain median of 534). There every
    ratio scalar is meaningless: FA 0, C_MD 1, K_bulk and K_mu ~1e29, µFA up to 1.2. Exclude them.
    """
    return sc["md"] <= md_min


def scalars(params: torch.Tensor, eps: float = 1e-30) -> dict[str, torch.Tensor]:
    """params (..., 28) = [ln S0, <D> (6, Mandel), C (21)] -> dict of (...) tensors."""
    E_iso, E_bulk, E_shear = projection_tensors(params.dtype, params.device)
    d = params[..., 1:7]
    C = mat6(params[..., 7:])
    dd = d.unsqueeze(-1) * d.unsqueeze(-2)          # <D> (x) <D>
    D2 = C + dd                                     # <D (x) D>

    md = d[..., :3].sum(-1) / 3.0
    md2 = _inner(dd, E_bulk)                        # MD^2
    v_md = _inner(C, E_bulk)
    v_shear = _inner(C, E_shear)
    v_iso = _inner(C, E_iso)
    c_md = v_md / (_inner(D2, E_bulk) + eps)
    c_mu = 1.5 * _inner(D2, E_shear) / (_inner(D2, E_iso) + eps)
    c_m = 1.5 * _inner(dd, E_shear) / (_inner(dd, E_iso) + eps)
    k_bulk = 3.0 * v_md / (md2 + eps)
    k_shear = 1.2 * v_shear / (md2 + eps)
    k_mu = 1.2 * _inner(D2, E_shear) / (md2 + eps)
    return {
        "s0": torch.exp(params[..., 0]),
        "md": md,
        "v_md": v_md, "v_shear": v_shear, "v_iso": v_iso,
        "c_md": c_md,
        "c_mu": c_mu, "ufa": c_mu.clamp_min(0).sqrt(),
        "c_m": c_m, "fa": c_m.clamp_min(0).sqrt(),
        "c_c": c_m / (c_mu + eps),
        "k_bulk": k_bulk, "k_shear": k_shear, "mk": k_bulk + k_shear, "k_mu": k_mu,
    }
