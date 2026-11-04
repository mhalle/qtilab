"""An acquisition protocol: one b-tensor per volume, and the QTI design matrix it implies.

Units are ms/um^2 for b (so diffusivities come out in um^2/ms, values near 1). That keeps
the 28 parameters on comparable scales, which matters for the constrained solver.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .tensors import vec6, vec21


def btensor(b: float, n: np.ndarray, shape: str) -> np.ndarray:
    """One b-tensor from a b-value, a unit vector and an encoding shape.

    LTE: b n n^T.  PTE: b/2 (I - n n^T), n the plane normal (DIPY's convention).
    STE: b/3 I.
    """
    n = np.asarray(n, float)
    nn = np.outer(n, n) if np.linalg.norm(n) > 0 else np.zeros((3, 3))
    shape = shape.upper()
    if shape == "LTE":
        return b * nn
    if shape == "PTE":
        return b / 2.0 * (np.eye(3) - nn)
    if shape == "STE":
        return b / 3.0 * np.eye(3)
    raise ValueError(f"unknown encoding shape {shape!r}")


@dataclass
class Protocol:
    B: torch.Tensor          # (N, 3, 3) b-tensors, ms/um^2
    shapes: list[str] | None = None

    @classmethod
    def from_table(cls, bvals, bvecs, shapes, *, bscale=1e-3, dtype=torch.float64):
        """bvals in s/mm^2 (scaled to ms/um^2 by ``bscale``), bvecs (3, N) or (N, 3)."""
        bvals = np.asarray(bvals, float) * bscale
        bvecs = np.asarray(bvecs, float)
        if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
            bvecs = bvecs.T
        B = np.stack([btensor(b, v, s) for b, v, s in zip(bvals, bvecs, shapes)])
        return cls(torch.as_tensor(B, dtype=dtype), list(shapes))

    def to(self, device=None, dtype=None):
        return Protocol(self.B.to(device=device, dtype=dtype), self.shapes)

    @property
    def n(self) -> int:
        return self.B.shape[0]

    @property
    def b(self) -> torch.Tensor:
        return torch.diagonal(self.B, dim1=-2, dim2=-1).sum(-1)

    @property
    def b6(self) -> torch.Tensor:
        return vec6(self.B)

    def b_delta(self) -> torch.Tensor:
        """Normalized b-tensor anisotropy (1 LTE, 0 STE, -1/2 PTE) for axisymmetric B."""
        w = torch.linalg.eigvalsh(self.B)            # ascending
        b = w.sum(-1).clamp_min(1e-12)
        # axisymmetric: the unique eigenvalue is whichever is farthest from the other two
        lo, mid, hi = w[..., 0], w[..., 1], w[..., 2]
        prolate = (hi - mid) >= (mid - lo)
        b_par = torch.where(prolate, hi, lo)
        b_perp = torch.where(prolate, (lo + mid) / 2, (mid + hi) / 2)
        return torch.where(w.sum(-1) > 1e-9, (b_par - b_perp) / b, torch.zeros_like(b))

    def design_matrix(self) -> torch.Tensor:
        """(N, 28): [1, -b6, 0.5 * vec21(b6 b6^T)], identical to DIPY's."""
        b6 = self.b6
        bb = vec21(b6.unsqueeze(-1) * b6.unsqueeze(-2))
        ones = torch.ones(self.n, 1, dtype=b6.dtype, device=b6.device)
        return torch.cat([ones, -b6, 0.5 * bb], dim=-1)

    def rank(self) -> int:
        return int(torch.linalg.matrix_rank(self.design_matrix()))
