"""Simulate tensor-valued diffusion signals from explicit diffusion tensor distributions (DTDs).

Each voxel is a cloud of M axisymmetric micro-tensors drawn from a hierarchical prior:
1-3 compartments (plus optional free water), each with its own size, shape, mean orientation,
orientation dispersion and size spread. The signal is the exact Laplace transform of that
discrete DTD,

    S / S0 = mean_m exp(-<B, D_m>),

not its cumulant truncation, so QTI's truncation bias is present in the simulated data just
as in real tissue. Ground-truth QTI parameters are the exact first and second moments of the
same discrete DTD.

Restriction and exchange are not modeled yet (time-independent Gaussian compartments).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .tensors import vec6, vec21


@dataclass
class Prior:
    max_comp: int = 3
    d_iso: tuple[float, float] = (0.2, 3.0)        # um^2/ms, log-uniform
    d_delta: tuple[float, float] = (-0.5, 1.0)     # uniform
    size_spread: tuple[float, float] = (0.0, 0.5)  # sd of log D_iso within a compartment
    dispersion: tuple[float, float] = (0.02, 3.0)  # Gaussian perturbation sd, log-uniform
    p_free_water: float = 0.5
    free_water_frac: tuple[float, float] = (0.0, 0.5)
    d_free: float = 3.0
    snr: tuple[float, float] = (15.0, 80.0)        # at b = 0, uniform


def _logu(lo, hi, shape, g, dtype, device):
    u = torch.rand(shape, generator=g, dtype=dtype, device=device)
    return torch.exp(torch.log(torch.tensor(lo, dtype=dtype)) + u * (torch.log(torch.tensor(hi / lo, dtype=dtype))))


def _unif(lo, hi, shape, g, dtype, device):
    return lo + (hi - lo) * torch.rand(shape, generator=g, dtype=dtype, device=device)


def sample_dtd(V, M=48, prior=Prior(), g=None, dtype=torch.float64, device="cpu"):
    """-> micro-tensors in Mandel form (V, M, 6)."""
    K = prior.max_comp
    kw = dict(g=g, dtype=dtype, device=device)
    # component parameters (V, K)
    n_comp = torch.randint(1, K + 1, (V,), generator=g, device=device)
    active = torch.arange(K, device=device).unsqueeze(0) < n_comp.unsqueeze(1)
    w = torch.distributions.Dirichlet(torch.ones(K, dtype=dtype, device=device)).sample((V,))
    w = torch.where(active, w, torch.zeros_like(w))
    w = w / w.sum(-1, keepdim=True)
    d_iso = _logu(*prior.d_iso, (V, K), **kw)
    d_del = _unif(*prior.d_delta, (V, K), **kw)
    spread = _unif(*prior.size_spread, (V, K), **kw)
    disp = _logu(*prior.dispersion, (V, K), **kw)
    mu = torch.randn((V, K, 3), generator=g, dtype=dtype, device=device)
    mu = mu / mu.norm(dim=-1, keepdim=True)

    # free water as an extra compartment
    has_fw = torch.rand((V,), generator=g, device=device) < prior.p_free_water
    f_fw = _unif(*prior.free_water_frac, (V,), **kw) * has_fw
    w = torch.cat([w * (1 - f_fw).unsqueeze(-1), f_fw.unsqueeze(-1)], dim=-1)          # (V, K+1)
    d_iso = torch.cat([d_iso, torch.full((V, 1), prior.d_free, dtype=dtype, device=device)], -1)
    d_del = torch.cat([d_del, torch.zeros((V, 1), dtype=dtype, device=device)], -1)
    spread = torch.cat([spread, torch.zeros((V, 1), dtype=dtype, device=device)], -1)
    disp = torch.cat([disp, torch.ones((V, 1), dtype=dtype, device=device)], -1)
    mu = torch.cat([mu, torch.tensor([0, 0, 1.0], dtype=dtype, device=device).expand(V, 1, 3)], 1)

    # assign each micro-tensor to a compartment
    comp = torch.multinomial(w, M, replacement=True, generator=g)                    # (V, M)
    take = lambda t: torch.gather(t, 1, comp)
    di = take(d_iso) * torch.exp(take(spread) * torch.randn((V, M), generator=g, dtype=dtype, device=device))
    dd = take(d_del)
    axis = torch.gather(mu, 1, comp.unsqueeze(-1).expand(V, M, 3))
    axis = axis + take(disp).unsqueeze(-1) * torch.randn((V, M, 3), generator=g, dtype=dtype, device=device)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    d_par = di * (1 + 2 * dd)
    d_perp = di * (1 - dd)
    nn = axis.unsqueeze(-1) * axis.unsqueeze(-2)
    eye = torch.eye(3, dtype=dtype, device=device)
    D = d_perp[..., None, None] * eye + (d_par - d_perp)[..., None, None] * nn
    return vec6(D)


def dtd_params(d6, s0=None):
    """Exact QTI parameters (V, 28) of a discrete, equally weighted DTD (V, M, 6)."""
    mean = d6.mean(1)
    x = d6 - mean.unsqueeze(1)
    C = torch.einsum("vmi,vmj->vij", x, x) / d6.shape[1]
    ls0 = torch.zeros(d6.shape[0], 1, dtype=d6.dtype, device=d6.device) if s0 is None else torch.log(s0).unsqueeze(-1)
    return torch.cat([ls0, mean, vec21(C)], dim=-1)


def signal(d6, B6, chunk=8192):
    """Noise-free S/S0 (V, N) for micro-tensors (V, M, 6) and b-tensors (N, 6), Mandel."""
    out = []
    for i in range(0, d6.shape[0], chunk):
        e = torch.exp(-torch.einsum("vmk,nk->vmn", d6[i:i + chunk], B6))
        out.append(e.mean(1))
    return torch.cat(out)


def rician(S, sigma, g=None):
    """Magnitude of S + complex Gaussian noise with per-voxel sd ``sigma`` (V,)."""
    s = sigma.unsqueeze(-1)
    n1 = torch.randn(S.shape, generator=g, dtype=S.dtype, device=S.device) * s
    n2 = torch.randn(S.shape, generator=g, dtype=S.dtype, device=S.device) * s
    return torch.sqrt((S + n1) ** 2 + n2 ** 2)


def simulate(protocol, V, M=48, prior=Prior(), g=None, noise=True, dtype=torch.float64, device="cpu"):
    """-> dict(signal (V, N), clean (V, N), params (V, 28) ground truth, snr (V,), d6 (V, M, 6))."""
    d6 = sample_dtd(V, M, prior, g, dtype, device)
    B6 = protocol.b6.to(dtype=dtype, device=device)
    clean = signal(d6, B6)
    snr = _unif(*prior.snr, (V,), g=g, dtype=dtype, device=device)
    noisy = rician(clean, 1.0 / snr, g) if noise else clean
    return {"signal": noisy, "clean": clean, "params": dtd_params(d6), "snr": snr, "d6": d6}
