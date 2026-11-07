"""Batched log-barrier interior-point solver for the QTI+ (dc) problem, an alternative to ``fit._admm``.

Per voxel it solves the same problem as ADMM,

    minimize f(x) = 1/2 x'Hx - g'x   s.t.  D(x) = mat3(x[1:7]) >= 0,  C(x) = mat6(x[7:]) >= 0,

by following the central path of

    phi_t(x) = t f(x) - log det D(x) - log det C(x),

whose minimizer is within nu / t of the optimum in f (nu = 3 + 6 = 9, the barrier parameter). Each iteration
is one Newton step on phi_t for all active voxels at once: the Newton matrix t H + (barrier Hessians) is
28 x 28 and positive definite even when H is singular, so it is one batched Cholesky; the barrier terms need
only Cholesky factorizations of the 3 x 3 and 6 x 6 blocks. The step length is the exact minimizer of phi_t
along the Newton direction (from the eigenvalues of the step relative to D and C; ``line_search=False``
uses the damped step 1 / (1 + lambda), safe because phi_t is self-concordant, but takes twice the steps).
When a voxel is nearly centred (Newton decrement below ``center``) its t grows by ``mu``. It stops when
centred with nu / t below ``eps_rel`` x its objective scale (the WLS residual sum of squares / 2).

Convergence does not depend on conditioning the way ADMM's does: about 20 Newton steps (at most ~50) reach
a relative gap of 1e-8 on HHS, Cardiff and DiB, where ADMM leaves 15% of HHS voxels unconverged after 1000
iterations. The price is a 28 x 28 factorization per voxel per step instead of a cached triangular solve.
Validation and timings: NOTES 2026-11-07 (IPM entry), scripts/validate_dipy_hhs.py, scripts/bench_ipm.py.

Mandel coordinates are an isometry, so with A = D^-1 the barrier gradient is -vec(A) and its Hessian is the
symmetric Kronecker product A (x)_s A: in the Mandel basis E_k = scaled (e_i e_j' + e_j e_i'),
[A (x)_s A]_kl = (A_ip A_jq + A_iq A_jp) s_k s_l / 2 for k = (i, j), l = (p, q), s = 1 on the diagonal and
sqrt 2 off it.

Optimization notes (measured on the M1 Max, 60k systems; not done yet):
* MPS: PyTorch's batched 28 x 28 Cholesky + solve takes 39 ms; a one-thread-per-voxel Metal kernel
  (torch.mps.compile_shader) takes 7.6 ms. A kernel for the whole Newton step (assemble K from H, t and the
  barrier Hessians, factor, solve, line search) would avoid most memory traffic. torch.compile of an
  unrolled Cholesky fails on MPS (the generated Metal kernel exceeds the buffer-argument limit).
* MPS: the 3 x 3 / 6 x 6 Cholesky inverses take 33 ms through torch.linalg and 1.6 ms (6 x 6) written as
  column operations; sym_kron with index_select is 2x faster than fancy indexing (4x slower on CPU).
* CPU (float64): Cholesky + solve 68 ms, sym_kron 55 ms, small inverses 28 ms per step at 60k voxels.
* Not tried: a primal-dual method with Nesterov-Todd scaling and Mehrotra predictor-corrector (CLARABEL's
  algorithm; typically 10-20 iterations), a fixed batch without per-step compaction (fewer GPU syncs,
  CUDA graphs), CUDA at all.
"""
from __future__ import annotations

import math

import torch

from .tensors import IDX6, IDX21, SCALE6, SCALE21, batched, mat3, mat6, vec6, vec21

NU = 9.0
_BLOCKS = (slice(0, 1), slice(1, 7), slice(7, 28))


def _kron_index(idx, scale, device, dtype):
    i = torch.tensor([a for a, _ in idx], device=device)
    j = torch.tensor([b for _, b in idx], device=device)
    s = torch.tensor(scale, dtype=dtype, device=device)
    return i, j, torch.outer(s, s) / 2


def sym_kron(A, index):
    """(V, n, n) symmetric -> (V, m, m) Mandel matrix of X -> A X A (m = n (n + 1) / 2)."""
    i, j, ss = index
    Aip = A[:, i[:, None], i[None, :]]
    Ajq = A[:, j[:, None], j[None, :]]
    Aiq = A[:, i[:, None], j[None, :]]
    Ajp = A[:, j[:, None], i[None, :]]
    return (Aip * Ajq + Aiq * Ajp) * ss


def _inv_chol(M):
    """Inverse of SPD (V, n, n) and its Cholesky factor; ok = factorization succeeded."""
    L, info = torch.linalg.cholesky_ex(M)
    eye = torch.eye(M.shape[-1], dtype=M.dtype, device=M.device).expand_as(M)
    return torch.cholesky_solve(eye, L), L, info == 0


def _relative_eigs(L, dM):
    """Eigenvalues of L^-1 dM L^-T: M + a dM is PD iff 1 + a e > 0 for all of them, and
    log det(M + a dM) = log det M + sum log(1 + a e)."""
    Y = torch.linalg.solve_triangular(L, dM, upper=False)
    Z = torch.linalg.solve_triangular(L, Y.transpose(-1, -2), upper=False)
    return torch.linalg.eigvalsh(0.5 * (Z + Z.transpose(-1, -2)))


def _line_search(t, a1, a2, e, iters=8, cap=1.0, margin=0.99):
    """Exact minimizer over a in [0, min(cap, margin a_max)] of the barrier along the Newton direction:
    phi(a) = t (a a1 + a^2 a2 / 2) - sum log(1 + a e), convex. Safeguarded 1-D Newton from the cap."""
    neg = e < 0
    amax = torch.where(neg, -1 / e.where(neg, -torch.ones_like(e)), torch.full_like(e, float("inf"))).amin(-1)
    hi = torch.minimum(torch.full_like(t, cap), margin * amax)
    lo = torch.zeros_like(t)

    def d1(a):
        return t * (a1 + a * a2) - (e / (1 + a[:, None] * e)).sum(-1)

    def d2(a):
        return t * a2 + (e * e / (1 + a[:, None] * e) ** 2).sum(-1)

    a = hi.clone()
    g = d1(a)
    ok_hi = g <= 0                       # still descending at the cap: take it
    for _ in range(iters):
        lo = torch.where(g < 0, a, lo)
        hi = torch.where(g >= 0, a, hi)
        an = a - g / d2(a).clamp_min(1e-30)
        an = torch.where((an > lo) & (an < hi), an, 0.5 * (lo + hi))
        a = an
        g = d1(a)
    a = torch.where(g < 0, a, lo.where(lo > 0, a))   # never return a point past the minimum's bracket
    return torch.where(ok_hi, hi.where(hi > 0, a), a)


def _interior_start(x, frac=0.1):
    """Strictly feasible point near x: eigenvalue-clip each block, then shift it by ``frac`` x its mean eigenvalue.

    Uses eigh once, on the starting point only."""
    out = x.clone()
    for sl, m, v in ((slice(1, 7), mat3, vec6), (slice(7, 28), mat6, vec21)):
        M = m(x[:, sl])
        w, Q = torch.linalg.eigh(M)
        n = M.shape[-1]
        wc = w.clamp_min(0)
        scale = wc.mean(-1, keepdim=True)
        fallback = x[:, sl].norm(dim=-1, keepdim=True) / math.sqrt(n)
        eps = frac * torch.where(scale > 1e-6 * fallback, scale, fallback).clamp_min(1e-12)
        out[:, sl] = v((Q * (wc + eps).unsqueeze(-2)) @ Q.transpose(-1, -2))
    return out


def barrier(H, g, x0, f_scale, grad0=None, eps_rel=None, mu=30.0, center=0.5, max_iter=200, frac=0.1,
            line_search=True, polish=None):
    """Batched barrier method for min 1/2 x'Hx - g'x over R x PSD3 x PSD6 (Mandel coordinates).

    H (V, 28, 28) positive semidefinite, g (V, 28), x0 (V, 28) the reference point (normally the unconstrained
    minimizer; made strictly feasible for the start), f_scale (V,) > 0 the objective's natural size.
    ``grad0``: H x0 - g, if known exactly (for the WLS solution with a ridge lam it is lam x0). The solver works
    with the step d = x - x0 and the gradient H d + grad0, which avoids the cancellation in H x - g (both
    terms are on the scale of the signal energy, the difference on the scale of the residuals): in float32
    that cancellation left the Newton decrement too noisy to certify centring.

    Stops when centred (Newton decrement < ``center``) with nu / t <= eps_rel x f_scale (``eps_rel`` 1e-8). In float32
    most voxels stop before that, at float32's own limit: a factorization fails and the last strictly feasible
    iterate is kept. That is counted as converged if nu / t was within 100x of the target (``failed_near_target``;
    other failures in ``failed_fraction``); µFA is then within float32's floor (about 2e-4 p99 on HHS, the same
    as ADMM in float32). Also in float32 only, rounding can stall the Newton decrement above ``center``: a
    voxel not centred within ``polish`` steps (12, or 40 without the line search) moves on to the next t, or
    stops at the final one (``uncertified_fraction``). ``line_search``: exact line search along the Newton
    direction (small eigvalsh calls); it halves the Newton steps and is a net gain on CPU and MPS.
    Returns (x, iterations, converged fraction, extra) with x strictly feasible.
    """
    V, P = g.shape
    dev, dt = g.device, g.dtype
    if eps_rel is None:
        eps_rel = 1e-8
    if polish is None:
        polish = 12 if line_search else 40
    stall_rule = dt != torch.float64                 # the decrement only stalls from rounding
    # the same block scaling as ADMM: unit mean diagonal of H in each block (helps float32)
    dg = torch.diagonal(H, dim1=-2, dim2=-1).clamp_min(1e-30)
    s = torch.empty_like(g)
    for sl in _BLOCKS:
        s[:, sl] = dg[:, sl].mean(-1, keepdim=True).rsqrt()
    Hs = H * s.unsqueeze(-1) * s.unsqueeze(-2)
    x0s = x0 / s
    r0 = (torch.einsum("vij,vj->vi", H, x0) - g if grad0 is None else grad0) * s      # gradient at x0, scaled
    k3, k6 = _kron_index(IDX6, SCALE6, dev, dt), _kron_index(IDX21, SCALE21, dev, dt)

    d = _interior_start(x0s, frac) - x0s
    # t0 balances the objective against the barrier: nu / t0 = suboptimality bound of the start
    f0 = 0.5 * torch.einsum("vi,vij,vj->v", d, Hs, d) + (r0 * d).sum(-1)
    t = NU / f0.clamp_min(1e-3 * f_scale).clamp_min(1e-30)
    tol = eps_rel * f_scale
    act = torch.arange(V, device=dev)
    done = torch.zeros(V, dtype=torch.bool, device=dev)
    failed = torch.zeros(V, dtype=torch.bool, device=dev)
    uncert = torch.zeros(V, dtype=torch.bool, device=dev)
    late_fail = torch.zeros(V, dtype=torch.bool, device=dev)
    why_dc = torch.zeros(V, dtype=torch.bool, device=dev)       # failure: D or C factorization (left the cone)
    why_k = torch.zeros(V, dtype=torch.bool, device=dev)        # failure: Newton matrix
    newton_steps = torch.zeros(V, dtype=torch.int32, device=dev)
    since = torch.zeros(V, dtype=torch.int32, device=dev)       # Newton steps since t last changed
    d_ok = d.clone()                                             # last iterate whose D and C factorized
    lam_last = torch.zeros(V, dtype=dt, device=dev)
    it = 0
    for it in range(1, max_iter + 1):
        da, ta, Ha = d[act], t[act], Hs[act]
        xa = x0s[act] + da
        AD, LD, okd = _inv_chol(mat3(xa[:, 1:7]))
        AC, LC, okc = _inv_chol(mat6(xa[:, 7:]))
        # float32: a step can round a tiny eigenvalue to <= 0; such a voxel stops at its previous iterate
        d[act] = torch.where((okd & okc)[:, None], da, d_ok[act])
        d_ok[act] = d[act]
        gf = torch.einsum("vij,vj->vi", Ha, da) + r0[act]                  # gradient of f
        grad = ta[:, None] * gf
        grad[:, 1:7] -= vec6(AD)
        grad[:, 7:] -= vec21(AC)
        K = ta[:, None, None] * Ha
        K[:, 1:7, 1:7] += sym_kron(AD, k3)
        K[:, 7:, 7:] += sym_kron(AC, k6)
        # Jacobi equilibration: barrier terms grow like 1 / slack^2 near the boundary, so K's diagonal spans many
        # decades at large t; scaling it to unit diagonal keeps the float32 Cholesky from breaking down
        q = torch.diagonal(K, dim1=-2, dim2=-1).clamp_min(1e-30).rsqrt()
        L, info = torch.linalg.cholesky_ex(K * q.unsqueeze(-1) * q.unsqueeze(-2))
        dx = -q * torch.cholesky_solve((q * grad).unsqueeze(-1), L).squeeze(-1)
        lam = (-(grad * dx).sum(-1)).clamp_min(0).sqrt()
        ok = okd & okc & (info == 0) & torch.isfinite(lam)
        why_dc[act[~(okd & okc)]] = True
        why_k[act[(okd & okc) & ~ok]] = True
        dx = torch.where(ok[:, None], dx, torch.zeros_like(dx))       # failed voxels stop; keep the math finite
        lam = torch.where(ok, lam, torch.zeros_like(lam))
        damped = 1 / (1 + lam)
        if line_search:
            okb = ok[:, None, None]
            LDs = torch.where(okb, LD, torch.eye(3, dtype=dt, device=dev))
            LCs = torch.where(okb, LC, torch.eye(6, dtype=dt, device=dev))
            e = torch.cat([batched(_relative_eigs, LDs, mat3(dx[:, 1:7])), batched(_relative_eigs, LCs, mat6(dx[:, 7:]))], -1)
            alpha = _line_search(ta, (gf * dx).sum(-1), torch.einsum("vi,vij,vj->v", dx, Ha, dx), e)
            alpha = torch.maximum(alpha, damped)                          # never shorter than the damped step
            alpha = torch.where(torch.isfinite(alpha), alpha, damped)
        else:
            alpha = torch.where(lam < center, torch.ones_like(lam), damped)
        d[act] = da + torch.where(ok[:, None], alpha[:, None] * dx, torch.zeros_like(dx))
        newton_steps[act] += 1
        lam_last[act] = lam
        final = NU / ta <= tol[act]
        since[act] += 1
        centred = ok & (lam < center)
        stuck = (since[act] > polish) & stall_rule     # float32: the decrement can stall above ``center``
        conv = final & (centred | stuck)
        up = ~final & (centred | stuck)
        t[act] = torch.where(up, ta * mu, ta)
        since[act] = torch.where(up, torch.zeros_like(since[act]), since[act])
        fin = conv | ~ok
        done[act[conv]] = True
        uncert[act[(conv | up) & ~centred]] = True
        failed[act[~ok]] = True
        near = NU / ta <= 100 * tol[act]                  # float32 precision limit within 100x of the target
        late_fail[act[~ok & near]] = True
        done[act[~ok & near]] = True
        act = act[~fin]
        if act.numel() == 0:
            break
    extra = {"failed_fraction": float(failed.float().mean()), "failed_near_target": float(late_fail.float().mean()),
             "failed_dc": float(why_dc.float().mean()), "failed_newton": float(why_k.float().mean()), "uncertified_fraction": float(uncert.float().mean()),
             "median_steps": float(newton_steps.float().median()), "max_steps": int(newton_steps.max()),
             "gap_rel": NU / t / f_scale, "lam": lam_last, "steps": newton_steps, "done": done, "failed": failed}
    return (x0s + d) * s, it, float(done.float().mean()), extra
