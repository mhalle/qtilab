"""Batched QTI fits: OLS, WLS (DIPY-equivalent) and positivity-constrained WLS (QTI+ "dc").

Everything is batched over voxels and runs on whatever device the tensors live on.

The constrained fit solves, per voxel,

    minimize 1/2 || W (X beta - y) ||^2   s.t.  <D>(beta) >= 0,  C(beta) >= 0

which is DIPY's SDPdc problem (same weights, same objective up to a monotone transform),
by ADMM. The x-update is one cached Cholesky solve per voxel; the z-update is an
eigenvalue clip of a 3x3 and a 6x6 matrix. Only voxels whose unconstrained WLS solution
violates a constraint go through ADMM.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .tensors import mat3, mat6, psd_project, vec6, vec21

MIN_SIGNAL = 1e-4  # DIPY's MIN_POSITIVE_SIGNAL


@dataclass
class FitInfo:
    method: str
    n_voxels: int
    n_violating: int = 0
    iterations: int = 0
    converged_fraction: float = 1.0
    extra: dict = field(default_factory=dict)


def _log_signal(S, S0_floor=MIN_SIGNAL):
    return torch.log(S.clamp_min(S0_floor))


def _solve_normal(H, g, full_rank: bool):
    if full_rank:
        L, info = torch.linalg.cholesky_ex(H)
        if bool((info == 0).all()):
            return torch.cholesky_solve(g.unsqueeze(-1), L).squeeze(-1)
    return (torch.linalg.pinv(H, hermitian=True) @ g.unsqueeze(-1)).squeeze(-1)


def ols(X: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """X (N, 28), S (V, N) -> params (V, 28). Minimum-norm solution if X is rank-deficient."""
    y = _log_signal(S)
    return y @ torch.linalg.pinv(X).T


def _weights(X, S, weights):
    """Square-root weights per measurement; DIPY's default is the OLS-predicted signal."""
    if weights is not None:
        return weights.sqrt()
    return torch.exp(ols(X, S) @ X.T)


def _normal_equations(X, y, w):
    w2 = w * w
    H = torch.einsum("vn,ni,nj->vij", w2, X, X)
    g = torch.einsum("vn,ni->vi", w2 * y, X)
    return H, g


def wls(X, S, weights=None, chunk=20000):
    """DIPY-equivalent WLS: sqrt-weights = OLS-predicted signal unless given."""
    full_rank = int(torch.linalg.matrix_rank(X)) == X.shape[1]
    out = []
    for i in range(0, S.shape[0], chunk):
        s = S[i:i + chunk]
        w = _weights(X, s, None if weights is None else weights[i:i + chunk])
        H, g = _normal_equations(X, _log_signal(s), w)
        out.append(_solve_normal(H, g, full_rank))
    return torch.cat(out)


def violates(params, tol=0.0):
    """Boolean (V,): mean tensor or covariance has a negative eigenvalue."""
    ed = torch.linalg.eigvalsh(mat3(params[..., 1:7]))[..., 0]
    ec = torch.linalg.eigvalsh(mat6(params[..., 7:]))[..., 0]
    return (ed < -tol) | (ec < -tol)


def _project(z):
    zd = vec6(psd_project(mat3(z[..., 1:7])))
    zc = vec21(psd_project(mat6(z[..., 7:])))
    return torch.cat([z[..., :1], zd, zc], dim=-1)


def _admm(H, g, x0, rho=1.0, max_iter=1000, eps_abs=1e-6, eps_rel=1e-4,
          adapt_every=10, adapt_mu=10.0):
    """Batched ADMM for min 1/2 x^T H x - g^T x over x in R x PSD3 x PSD6 (Mandel coords).

    * Variables are block-rescaled per voxel (S0 / tensor / covariance) so that H's diagonal
      is about 1 in each block; positive scaling of a block leaves its PSD cone unchanged.
    * rho adapts per voxel by residual balancing (Boyd et al. 2011, sec. 3.4.1).
    * Converged voxels leave the working set, so late iterations cost only the stragglers.
    """
    V, P = g.shape
    dev, dt = g.device, g.dtype
    dg = torch.diagonal(H, dim1=-2, dim2=-1).clamp_min(1e-30)
    s = torch.empty_like(g)
    for sl in (slice(0, 1), slice(1, 7), slice(7, 28)):
        s[:, sl] = dg[:, sl].mean(-1, keepdim=True).rsqrt()
    Hs = H * s.unsqueeze(-1) * s.unsqueeze(-2)
    gs = g * s
    eye = torch.eye(P, dtype=dt, device=dev)

    z_out = _project(x0 / s)
    z, u = z_out.clone(), torch.zeros_like(z_out)
    rh = torch.full((V,), float(rho), dtype=dt, device=dev)
    act = torch.arange(V, device=dev)                       # working set (indices into V)
    L = torch.linalg.cholesky(Hs + rh[:, None, None] * eye)
    it = 0
    for it in range(1, max_iter + 1):
        r_ = rh[act].unsqueeze(-1)
        x = torch.cholesky_solve((gs[act] + r_ * (z[act] - u[act])).unsqueeze(-1), L[act]).squeeze(-1)
        z_old = z[act]
        za = _project(x + u[act])
        ua = u[act] + x - za
        z[act], u[act] = za, ua
        rp = (x - za).norm(dim=-1)
        rd = r_.squeeze(-1) * (za - z_old).norm(dim=-1)
        eps_p = eps_abs * P ** 0.5 + eps_rel * torch.maximum(x.norm(dim=-1), za.norm(dim=-1))
        eps_d = eps_abs * P ** 0.5 + eps_rel * (r_ * ua).norm(dim=-1)
        conv = (rp <= eps_p) & (rd <= eps_d)
        if it % adapt_every == 0:
            up = (rp > adapt_mu * rd) & ~conv
            dn = (rd > adapt_mu * rp) & ~conv
            factor = torch.where(up, 2.0, torch.where(dn, 0.5, 1.0)).to(dt)
            chg = factor != 1.0
            if bool(chg.any()):
                idx = act[chg]
                rh[idx] = rh[idx] * factor[chg]
                u[idx] = u[idx] / factor[chg].unsqueeze(-1)    # scaled dual follows rho
                L[idx] = torch.linalg.cholesky(Hs[idx] + rh[idx][:, None, None] * eye)
        act = act[~conv]
        if act.numel() == 0:
            break
    return z * s, it, 1.0 - act.numel() / V


def constrained(X, S, weights=None, chunk=20000, ridge=1e-6, **admm_kw):
    """QTI+ (dc): WLS with <D> >= 0 and C >= 0. Returns (params, FitInfo).

    For a rank-deficient protocol (e.g. LTE + STE only, rank 23) the objective is flat along the
    unobserved directions of C, so the constrained solution is not unique and ADMM crawls. A tiny
    ridge (``ridge`` x the mean diagonal of H) picks the minimum-norm solution among the optima;
    the standard scalars do not depend on the unobserved directions.
    """
    full_rank = int(torch.linalg.matrix_rank(X)) == X.shape[1]
    out, n_bad, iters, conv = [], 0, 0, []
    for i in range(0, S.shape[0], chunk):
        s = S[i:i + chunk]
        w = _weights(X, s, None if weights is None else weights[i:i + chunk])
        H, g = _normal_equations(X, _log_signal(s), w)
        p = _solve_normal(H, g, full_rank)
        bad = violates(p)
        if bool(bad.any()):
            Hb = H[bad]
            if not full_rank and ridge > 0:
                lam = ridge * torch.diagonal(Hb, dim1=-2, dim2=-1).mean(-1)
                Hb = Hb + lam[:, None, None] * torch.eye(Hb.shape[-1], dtype=Hb.dtype, device=Hb.device)
            pb, it, cf = _admm(Hb, g[bad], p[bad], **admm_kw)
            p = p.clone()
            p[bad] = pb
            n_bad += int(bad.sum())
            iters = max(iters, it)
            conv.append((cf, int(bad.sum())))
        out.append(p)
    cf = (sum(c * n for c, n in conv) / max(1, sum(n for _, n in conv))) if conv else 1.0
    return torch.cat(out), FitInfo("constrained", S.shape[0], n_bad, iters, cf)


def predict(X, params):
    return torch.exp(params @ X.T)
