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


def _dirichlet(alpha: int, shape, g, dtype, device):
    """Symmetric Dirichlet(alpha) weights over the last axis of ``shape``, for integer alpha, drawn from ``g``.

    torch.distributions.Dirichlet ignores generators, so simulations seeded through ``g`` were not
    reproducible. Gamma(alpha) for integer alpha is a sum of alpha Exp(1) draws; normalized, they are
    Dirichlet(alpha) exactly.
    """
    u = torch.rand((alpha, *shape), generator=g, dtype=dtype, device=device)
    gam = -torch.log1p(-u).sum(0)                       # rand is in [0, 1), so every term is finite
    return gam / gam.sum(-1, keepdim=True).clamp_min(torch.finfo(dtype).tiny)


def sample_dtd(V, M=48, prior=Prior(), g=None, dtype=torch.float64, device="cpu", return_free=False):
    """-> micro-tensors in Mandel form (V, M, 6)."""
    K = prior.max_comp
    kw = dict(g=g, dtype=dtype, device=device)
    # component parameters (V, K)
    n_comp = torch.randint(1, K + 1, (V,), generator=g, device=device)
    active = torch.arange(K, device=device).unsqueeze(0) < n_comp.unsqueeze(1)
    w = _dirichlet(1, (V, K), g, dtype, device)
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

    if return_free:
        d6, comp = micro_tensors(w, d_iso, d_del, spread, disp, mu, M, g, return_comp=True)
        return d6, comp == K                       # the free-water compartment is the last one
    return micro_tensors(w, d_iso, d_del, spread, disp, mu, M, g)


def micro_tensors(w, d_iso, d_del, spread, disp, mu, M, g=None, return_comp=False):
    """Draw M micro-tensors per voxel from per-compartment parameters, each (V, K) (mu: (V, K, 3)).

    w: compartment weights (rows sum to 1); d_iso: compartment mean diffusivity; d_del: shape
    (D_Delta in [-0.5, 1]); spread: sd of log D_iso within the compartment; disp: sd of the
    Gaussian perturbation of the axis before renormalizing (about 3 is isotropic); mu: mean axis.
    """
    V = w.shape[0]
    dtype, device = d_iso.dtype, d_iso.device
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
    return (vec6(D), comp) if return_comp else vec6(D)


BRAIN_CLASSES = ("WM", "GM", "WM+CSF", "GM+CSF")


def _unit(shape, g, dtype, device):
    v = torch.randn(shape, generator=g, dtype=dtype, device=device)
    return v / v.norm(dim=-1, keepdim=True)


BRAIN_PRIORS = {
    # v1: first brain-like prior (2026-11-04). Real WM turned out more anisotropic than this.
    1: dict(f_ax=(0.4, 0.7), ax_d=(0.65, 0.85), ex_d=(0.8, 1.1), ex_dd=(0.2, 0.5), wm_disp=(0.05, 0.4),
            gm_fn=(0.2, 0.45), gm_nd=(0.6, 0.9), gm_fs=(0.1, 0.3), gm_sd=(0.3, 0.7), gm_ed=(0.9, 1.3), gm_edd=(0.0, 0.2),
            fw_wm=(0.0, 0.10), fw_gm=(0.0, 0.15)),
    # v2: wider: higher axonal fraction, more anisotropic extra-axonal space, wider diffusivities.
    2: dict(f_ax=(0.4, 0.85), ax_d=(0.55, 0.95), ex_d=(0.6, 1.2), ex_dd=(0.2, 0.8), wm_disp=(0.03, 0.4),
            gm_fn=(0.2, 0.55), gm_nd=(0.5, 1.0), gm_fs=(0.05, 0.3), gm_sd=(0.2, 0.8), gm_ed=(0.7, 1.4), gm_edd=(0.0, 0.3),
            fw_wm=(0.0, 0.10), fw_gm=(0.0, 0.15)),
    # v3: deliberately different, for prior-sensitivity tests: much wider diffusivities, dispersion and
    # compartment fractions, and up to 25% free water in WM and GM.
    3: dict(f_ax=(0.3, 0.9), ax_d=(0.4, 1.1), ex_d=(0.5, 1.4), ex_dd=(0.0, 0.9), wm_disp=(0.02, 0.8),
            gm_fn=(0.1, 0.6), gm_nd=(0.4, 1.2), gm_fs=(0.0, 0.4), gm_sd=(0.1, 1.0), gm_ed=(0.6, 1.6), gm_edd=(0.0, 0.4),
            fw_wm=(0.0, 0.25), fw_gm=(0.0, 0.25)),
    # v4: v2 matched to real classical maps by FastSurfer region (scripts/prior_match.py, NOTES 2026-11-06):
    # real 2 x 2 x 4 mm WM is far less coherent (classical C_c 0.28 vs v2's 0.86), so mostly 2-3 bundles with
    # more equal weights and moderate-to-wide dispersion (a wider range gave the right median but a bimodal
    # C_c); less free water and wider extra-axonal diffusivity (real WM C_MD 0.026 vs v2's 0.066); a lower
    # GM neurite fraction (cortex µFA 0.53 vs v2's 0.60).
    4: dict(f_ax=(0.4, 0.85), ax_d=(0.55, 0.95), ex_d=(0.7, 1.3), ex_dd=(0.1, 0.7), wm_disp=(0.25, 1.0),
            wm_nb=(0.1, 0.4, 0.5), wm_alpha=3,
            gm_fn=(0.15, 0.45), gm_nd=(0.5, 1.0), gm_fs=(0.05, 0.3), gm_sd=(0.2, 0.8), gm_ed=(0.7, 1.4),
            gm_edd=(0.0, 0.3), fw_wm=(0.0, 0.05), fw_gm=(0.0, 0.15)),
}


def sample_brain(V, M=64, g=None, dtype=torch.float64, device="cpu", class_probs=(0.35, 0.35, 0.15, 0.15),
                 return_free=False, version=2, return_comp=False):
    """Brain-like tensor distributions. -> (d6 (V, M, 6), class index (V,)[, free-water mask (V, M)]).

    Compartments are Gaussian (no restriction or exchange); values in um^2/ms are typical of the
    biophysical-modeling literature, not fitted to any dataset.

    * WM: 1-3 fiber bundles (60/30/10%), each an intra-axonal stick (D_iso 0.65-0.85, D_Delta
      0.9-1.0) plus an extra-axonal zeppelin on the same axis (D_iso 0.8-1.1, D_Delta 0.2-0.5);
      axonal fraction 0.4-0.7, bundle dispersion 0.05-0.4; 0-10% free water.
    * GM: highly dispersed neurites (stick, fraction 0.2-0.45, dispersion 0.8-3), an isotropic
      soma-like compartment (D_iso 0.3-0.7, fraction 0.1-0.3) and a mildly anisotropic
      extracellular space for the rest; 0-15% free water.
    * WM+CSF, GM+CSF: the same tissue plus 10-90% free water (D 3.0): partial volume at
      ventricles and sulci.

    Ranges above are ``version=1``; ``version=2`` (default) widens them, see ``BRAIN_PRIORS``. ``version`` may
    also be a dict of the same keys (for prior experiments); optional keys ``wm_nb`` (probabilities of
    1, 2, 3 bundles, default (0.6, 0.3, 0.1)) and ``wm_alpha`` (integer Dirichlet concentration of the
    bundle weights, default 2).
    """
    P = version if isinstance(version, dict) else BRAIN_PRIORS[version]
    kw = dict(dtype=dtype, device=device)
    u = lambda lo, hi, shape=(V,): lo + (hi - lo) * torch.rand(shape, generator=g, **kw)
    lu = lambda lo, hi, shape=(V,): torch.exp(torch.log(torch.tensor(lo, **kw)) + torch.rand(shape, generator=g, **kw) * torch.log(torch.tensor(hi / lo, **kw)))
    cls = torch.multinomial(torch.tensor(class_probs, **kw), V, replacement=True, generator=g)
    wm = (cls == 0) | (cls == 2)
    K = 7                                   # 3 bundles x (axon, extra) | free water
    w = torch.zeros(V, K, **kw); d_iso = torch.zeros(V, K, **kw); d_del = torch.zeros(V, K, **kw)
    spread = u(0.0, 0.15, (V, K)); disp = torch.ones(V, K, **kw); mu = _unit((V, K, 3), g, dtype, device)

    # --- white matter
    nb = torch.multinomial(torch.tensor(P.get("wm_nb", (0.6, 0.3, 0.1)), **kw), V, replacement=True, generator=g) + 1
    bw = _dirichlet(P.get("wm_alpha", 2), (V, 3), g, dtype, device)
    bw = torch.where(torch.arange(3, device=device) < nb[:, None], bw, torch.zeros_like(bw))
    bw = bw / bw.sum(-1, keepdim=True)
    f_ax = u(*P['f_ax'], (V, 3))
    for k in range(3):
        a, e = 2 * k, 2 * k + 1
        w[:, a] = bw[:, k] * f_ax[:, k]; w[:, e] = bw[:, k] * (1 - f_ax[:, k])
        d_iso[:, a] = u(*P['ax_d']); d_del[:, a] = u(0.9, 1.0)
        d_iso[:, e] = u(*P['ex_d']); d_del[:, e] = u(*P['ex_dd'])
        dk = lu(*P['wm_disp']); disp[:, a] = dk; disp[:, e] = dk
        mu[:, e] = mu[:, a]
    fw_wm = u(*P['fw_wm'])

    # --- gray matter (slots 0 neurite, 1 soma, 2 extracellular)
    gw = torch.zeros(V, K, **kw); gd = torch.zeros(V, K, **kw); gdd = torch.zeros(V, K, **kw); gdisp = torch.ones(V, K, **kw)
    f_n, f_s = u(*P['gm_fn']), u(*P['gm_fs'])
    f_s = torch.minimum(f_s, 0.95 - f_n)                 # keep the extracellular fraction positive
    gw[:, 0], gw[:, 1], gw[:, 2] = f_n, f_s, 1 - f_n - f_s
    gd[:, 0], gdd[:, 0], gdisp[:, 0] = u(*P['gm_nd']), u(0.8, 1.0), lu(0.8, 3.0)
    gd[:, 1], gdd[:, 1] = u(*P['gm_sd']), 0.0
    gd[:, 2], gdd[:, 2], gdisp[:, 2] = u(*P['gm_ed']), u(*P['gm_edd']), lu(0.5, 3.0)
    fw_gm = u(*P['fw_gm'])

    sel = wm[:, None]
    w = torch.where(sel, w, gw); d_iso = torch.where(sel, d_iso, gd)
    d_del = torch.where(sel, d_del, gdd); disp = torch.where(sel, disp, gdisp)

    # --- free water (slot 6), with CSF partial volume for the boundary classes
    fw = torch.where(wm, fw_wm, fw_gm)
    fw = torch.where(cls >= 2, u(0.1, 0.9), fw)
    w[:, :6] = w[:, :6] * (1 - fw)[:, None]
    w[:, 6] = fw
    d_iso[:, 6], d_del[:, 6], spread[:, 6] = 3.0, 0.0, 0.0
    d6, comp = micro_tensors(w, d_iso, d_del, spread, disp, mu, M, g, return_comp=True)
    if return_comp:
        return d6, cls, comp
    return (d6, cls, comp == 6) if return_free else (d6, cls)


# Time dependence: per-compartment strength tau (U[0, max]); the compartment's diffusivities scale
# as 1 + tau (sqrt(omega) - 1) at relative encoding frequency omega (omega = 1 is the reference).
# sqrt(omega) is the low-frequency form for one-dimensional short-range disorder (e.g. beading
# along axons); it stands in here for any restriction-driven frequency dependence.
TAU_MAX = {"WM": (0.15, 0.35, 0.15, 0.35, 0.15, 0.35, 0.0),      # axon, extra (x3 bundles), free water
           "GM": (0.15, 0.40, 0.35, 0.0, 0.0, 0.0, 0.0)}          # neurite, soma, extracellular, free water


def sample_tau(cls, comp, g=None, wm_scale=1.0, gm_scale=1.0):
    """Per-micro-tensor time-dependence strength (V, M) from tissue class and compartment index.

    wm_scale / gm_scale multiply the TAU_MAX ranges of white- and gray-matter compartments."""
    V = cls.shape[0]
    dtype = torch.float64
    wm = ((cls == 0) | (cls == 2)).unsqueeze(-1)
    tmax = torch.where(wm, wm_scale * torch.tensor(TAU_MAX["WM"], dtype=dtype),
                       gm_scale * torch.tensor(TAU_MAX["GM"], dtype=dtype))
    tau_c = tmax * torch.rand((V, 7), generator=g, dtype=dtype)
    return torch.gather(tau_c, 1, comp)


def signal_td(d6, B6, omega, tau, chunk=8192):
    """Noise-free S/S0 with time-dependent compartments.

    omega (N,) relative encoding frequency per measurement; tau (V, M) per micro-tensor strength.
    """
    f = torch.sqrt(torch.as_tensor(omega, dtype=d6.dtype)) - 1.0                     # (N,)
    out = []
    for i in range(0, d6.shape[0], chunk):
        base = torch.einsum("vmk,nk->vmn", d6[i:i + chunk], B6)
        scale = 1.0 + tau[i:i + chunk].unsqueeze(-1) * f
        out.append(torch.exp(-base * scale).mean(1))
    return torch.cat(out)


def dtd_params(d6, s0=None, weights=None):
    """Exact QTI parameters (V, 28) of a discrete DTD (V, M, 6); equal weights unless ``weights`` (V, M)."""
    if weights is None:
        weights = torch.ones(d6.shape[:2], dtype=d6.dtype, device=d6.device)
    w = weights / weights.sum(1, keepdim=True).clamp_min(1e-30)
    mean = (w.unsqueeze(-1) * d6).sum(1)
    x = d6 - mean.unsqueeze(1)
    C = torch.einsum("vm,vmi,vmj->vij", w, x, x)
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
