"""Mandel-notation helpers for second- and fourth-order diffusion tensors.

Conventions match DIPY's QTI module exactly, so parameter vectors are interchangeable:

* a symmetric 3x3 tensor T maps to the 6-vector
  [T_xx, T_yy, T_zz, sqrt2*T_yz, sqrt2*T_xz, sqrt2*T_xy];
* a symmetric 6x6 matrix C maps to a 21-vector of its diagonal and sqrt2-scaled
  off-diagonal entries, in DIPY's order (``IDX21`` below).

Both maps are isometries (Frobenius norm preserved), which is what lets a Euclidean
projection on the parameter vector act as an eigenvalue clip on the matrices.
"""
from __future__ import annotations

import math

import torch

SQ2 = math.sqrt(2.0)

# 3x3 -> 6: (row, col) pairs and scale factors
IDX6 = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
SCALE6 = [1.0, 1.0, 1.0, SQ2, SQ2, SQ2]

# 6x6 -> 21, DIPY's order (read off dipy.reconst.qti.from_6x6_to_21x1)
IDX21 = [(0, 0), (1, 1), (2, 2),
         (1, 2), (0, 2), (0, 1),
         (0, 3), (0, 4), (0, 5),
         (1, 3), (1, 4), (1, 5),
         (2, 3), (2, 4), (2, 5),
         (3, 3), (4, 4), (5, 5),
         (3, 4), (4, 5), (3, 5)]
SCALE21 = [1.0 if i == j else SQ2 for i, j in IDX21]


def vec6(T: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) symmetric -> (..., 6) Mandel."""
    out = [T[..., i, j] * s for (i, j), s in zip(IDX6, SCALE6)]
    return torch.stack(out, dim=-1)


def mat3(v: torch.Tensor) -> torch.Tensor:
    """(..., 6) Mandel -> (..., 3, 3) symmetric."""
    T = v.new_zeros(v.shape[:-1] + (3, 3))
    for k, ((i, j), s) in enumerate(zip(IDX6, SCALE6)):
        T[..., i, j] = v[..., k] / s
        T[..., j, i] = v[..., k] / s
    return T


def vec21(C: torch.Tensor) -> torch.Tensor:
    """(..., 6, 6) symmetric -> (..., 21)."""
    out = [C[..., i, j] * s for (i, j), s in zip(IDX21, SCALE21)]
    return torch.stack(out, dim=-1)


def mat6(v: torch.Tensor) -> torch.Tensor:
    """(..., 21) -> (..., 6, 6) symmetric."""
    C = v.new_zeros(v.shape[:-1] + (6, 6))
    for k, ((i, j), s) in enumerate(zip(IDX21, SCALE21)):
        C[..., i, j] = v[..., k] / s
        C[..., j, i] = v[..., k] / s
    return C


def projection_tensors(dtype=torch.float64, device="cpu"):
    """E_iso, E_bulk, E_shear as 6x6 Mandel matrices."""
    e_iso6 = vec6(torch.eye(3, dtype=dtype, device=device) / 3.0)
    E_bulk = torch.outer(e_iso6, e_iso6)
    E_iso = torch.eye(6, dtype=dtype, device=device) / 3.0
    return E_iso, E_bulk, E_iso - E_bulk


def psd_project(M: torch.Tensor) -> torch.Tensor:
    """Frobenius projection of symmetric matrices onto the PSD cone (eigenvalue clip)."""
    w, V = torch.linalg.eigh(M)
    return (V * w.clamp_min(0).unsqueeze(-2)) @ V.transpose(-1, -2)
