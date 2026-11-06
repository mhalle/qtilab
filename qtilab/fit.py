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

from .tensors import batched, mat3, mat6, psd_project, vec6, vec21

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


def _row_space(X: torch.Tensor, rank: int) -> torch.Tensor:
    """(28, rank) orthonormal basis of the row space of X, computed once in float64 on the CPU.

    ``rank`` must come from ``torch.linalg.matrix_rank`` at X's own precision: a float32 design cast to
    float64 carries float32 rounding in its null singular values, so a float64 cutoff would keep them.

    With positive weights H = X' W^2 X has the same null space as X for every voxel, so the minimum-norm
    WLS solution lies in this space and can be found from an r x r Cholesky solve instead of a pseudo-inverse.
    """
    X64 = X.detach().cpu().double()                         # move first: MPS has no float64
    _, _, Vh = torch.linalg.svd(X64, full_matrices=False)
    return Vh[:rank].T.to(dtype=X.dtype, device=X.device)


def _pinv_solve(H, g):
    # Pseudo-inverse in sub-batches (``tensors.batched``; a 200k batch asked for 66 GB of CUDA workspace).
    return batched(lambda h, b: (torch.linalg.pinv(h, hermitian=True) @ b.unsqueeze(-1)).squeeze(-1), H, g)


def _solve_normal(H, g, full_rank: bool, Q=None):
    """Solve H x = g per voxel (minimum-norm solution if H is singular).

    Full rank: Cholesky. Rank-deficient with a row-space basis ``Q`` of the design (``_row_space``): Cholesky
    of Q'HQ, which is well conditioned (median condition number ~1e3 on HHS) and gives the same
    minimum-norm solution as the pseudo-inverse (2e-11 in float64). That replaced a batched ``pinv``,
    which in float32 on MPS was off by up to 5e-2 in µFA, and is 9-21x faster. Voxels whose reduced matrix
    is not positive definite (e.g. zero weights shrinking the observed space) fall back to ``pinv``.
    """
    if full_rank:
        L, info = torch.linalg.cholesky_ex(H)
        if bool((info == 0).all()):
            return torch.cholesky_solve(g.unsqueeze(-1), L).squeeze(-1)
    if Q is None:
        return _pinv_solve(H, g)
    Hr = Q.T @ H @ Q
    L, info = torch.linalg.cholesky_ex(Hr)
    x = torch.cholesky_solve((g @ Q).unsqueeze(-1), L).squeeze(-1) @ Q.T
    bad = info != 0
    if bool(bad.any()):
        x[bad] = _pinv_solve(H[bad], g[bad])
    return x


def ols(X: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """X (N, 28), S (V, N) -> params (V, 28). Minimum-norm solution if X is rank-deficient."""
    y = _log_signal(S)
    return y @ torch.linalg.pinv(X).T


def _weights(X, S, weights):
    """Square-root weights per measurement; DIPY's default is the OLS-predicted signal."""
    if weights is not None:
        return weights.sqrt()
    return torch.exp(ols(X, S) @ X.T)


CPU_CHUNK = 20_000
GPU_CHUNK_MAX = 250_000
GPU_CHUNK_MIN = 8_192
GPU_MEMORY_FRACTION = 0.4


def _gpu_free_bytes(device: torch.device) -> int | None:
    """Memory a new allocation can use on ``device``, or None if unknown."""
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        cached = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
        return free + cached
    if device.type == "mps":
        return torch.mps.recommended_max_memory() - torch.mps.current_allocated_memory()   # cache is reusable
    return None


def _bytes_per_voxel(S) -> int:
    """Conservative peak per voxel of a fit chunk.

    WLS's normal equations materialize the weighted design (V, N, 28) a few times; ADMM keeps H, its scaled
    copy and the Cholesky factor (V, 28, 28). Measured at N = 186, float32: about 60 kB per voxel (WLS 12 GB,
    constrained 11 GB for 200k voxels on MPS); this estimate gives 88 kB.
    """
    return S.element_size() * (3 * S.shape[1] * 28 + 8 * 28 * 28)


def _chunk(S, chunk):
    """Voxels per chunk: ``chunk`` if given; else 20k on CPU, and on a GPU as many as fit in
    ``GPU_MEMORY_FRACTION`` of its free memory, between 8k and 250k.

    On a GPU each chunk ends in a long ADMM tail of tiny, sync-bound steps (a few hundred voxels left
    after a few hundred iterations), so fewer chunks is much faster: 200k voxels on MPS take 27 s as one
    chunk vs 102 s as ten. Sizing from free memory keeps one chunk on large GPUs (64 GB M1 Max, 24 GB cards)
    while a 16 GB Mac gets roughly 50k-voxel chunks instead of running out of memory.
    """
    if chunk is not None:
        return int(chunk)
    if S.device.type == "cpu":
        return CPU_CHUNK
    free = _gpu_free_bytes(S.device)
    if free is None:
        return GPU_CHUNK_MAX
    n = int(GPU_MEMORY_FRACTION * free) // _bytes_per_voxel(S)
    return max(GPU_CHUNK_MIN, min(GPU_CHUNK_MAX, n))


def _normal_equations(X, y, w):
    w2 = w * w
    H = torch.einsum("vn,ni,nj->vij", w2, X, X)
    g = torch.einsum("vn,ni->vi", w2 * y, X)
    return H, g


def wls(X, S, weights=None, chunk=None):
    """DIPY-equivalent WLS: sqrt-weights = OLS-predicted signal unless given. ``chunk``: see ``_chunk``."""
    chunk = _chunk(S, chunk)
    rank = int(torch.linalg.matrix_rank(X))
    full_rank = rank == X.shape[1]
    Q = None if full_rank else _row_space(X, rank)
    out = []
    for i in range(0, S.shape[0], chunk):
        s = S[i:i + chunk]
        w = _weights(X, s, None if weights is None else weights[i:i + chunk])
        H, g = _normal_equations(X, _log_signal(s), w)
        out.append(_solve_normal(H, g, full_rank, Q))
    return torch.cat(out)


def violates(params, tol=0.0):
    """Boolean (V,): mean tensor or covariance has a negative eigenvalue."""
    ed = batched(torch.linalg.eigvalsh, mat3(params[..., 1:7]))[..., 0]
    ec = batched(torch.linalg.eigvalsh, mat6(params[..., 7:]))[..., 0]
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


def constrained(X, S, weights=None, chunk=None, ridge=1e-6, **admm_kw):
    """QTI+ (dc): WLS with <D> >= 0 and C >= 0. Returns (params, FitInfo).

    For a rank-deficient protocol (e.g. LTE + STE only, rank 23) the objective is flat along the
    unobserved directions of C, so the constrained solution is not unique and ADMM crawls. A tiny
    ridge (``ridge`` x the mean diagonal of H) picks the minimum-norm solution among the optima;
    the standard scalars do not depend on the unobserved directions.

    ``chunk``: voxels per batch, default 20k on CPU and 250k on GPUs (see ``_chunk``).
    """
    chunk = _chunk(S, chunk)
    rank = int(torch.linalg.matrix_rank(X))
    full_rank = rank == X.shape[1]
    Q = None if full_rank else _row_space(X, rank)
    out, n_bad, iters, conv = [], 0, 0, []
    for i in range(0, S.shape[0], chunk):
        s = S[i:i + chunk]
        w = _weights(X, s, None if weights is None else weights[i:i + chunk])
        H, g = _normal_equations(X, _log_signal(s), w)
        p = _solve_normal(H, g, full_rank, Q)
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
