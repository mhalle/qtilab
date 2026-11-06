"""Simulation-based posterior estimation for tensor-valued diffusion data.

A set encoder reads a voxel's measurements as an unordered set of (b-tensor, signal) pairs and
produces an embedding; a conditional normalizing flow (zuko NSF) turns the embedding into a
posterior over tissue quantities. Training data are simulated on the fly, so every batch is new.

Targets separate free water from tissue:

    f_fw     free-water fraction
    md_t, fa_t, ufa_t, c_md_t    moments of the *tissue* tensor distribution (free water removed)
    md, ufa, c_md                moments of the *full* distribution (what QTI's cumulant targets)

Bounded quantities are modeled in logit space and diffusivities in log space.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from . import scalars, simulate

TARGETS = ["f_fw", "md_t", "fa_t", "ufa_t", "c_md_t", "md", "ufa", "c_md"]
_LOG = {"md_t", "md"}
EPS = 1e-4


def to_unconstrained(y: torch.Tensor) -> torch.Tensor:
    out = []
    for i, k in enumerate(TARGETS):
        v = y[..., i]
        if k in _LOG:
            out.append(torch.log(v.clamp_min(EPS)))
        else:
            v = v.clamp(EPS, 1 - EPS)
            out.append(torch.log(v) - torch.log1p(-v))
    return torch.stack(out, -1)


def from_unconstrained(z: torch.Tensor) -> torch.Tensor:
    out = []
    for i, k in enumerate(TARGETS):
        out.append(torch.exp(z[..., i]) if k in _LOG else torch.sigmoid(z[..., i]))
    return torch.stack(out, -1)


def targets(d6: torch.Tensor, free: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    """(V, M, 6) micro-tensors and (V, M) free-water mask [and (V, M) weights] -> (V, len(TARGETS))."""
    w = torch.ones(free.shape, dtype=d6.dtype, device=d6.device) if weights is None else weights
    full = scalars.scalars(simulate.dtd_params(d6, weights=w))
    tissue = scalars.scalars(simulate.dtd_params(d6, weights=w * (~free).to(d6.dtype)))
    f_fw = (w * free.to(d6.dtype)).sum(1) / w.sum(1)
    cols = {"f_fw": f_fw, "md_t": tissue["md"], "fa_t": tissue["fa"], "ufa_t": tissue["ufa"],
            "c_md_t": tissue["c_md"], "md": full["md"], "ufa": full["ufa"], "c_md": full["c_md"]}
    return torch.stack([cols[k] for k in TARGETS], -1)


def features(S: torch.Tensor, B6: torch.Tensor, b: torch.Tensor, b_delta: torch.Tensor) -> torch.Tensor:
    """Per-measurement inputs (V, N, 10): b6/2, b, b_delta, normalized signal and its log.

    The signal is normalized by the mean of the b ~ 0 measurements in each voxel.
    """
    ref = S[:, b < 0.05].mean(1, keepdim=True).clamp_min(1e-6)
    s = (S / ref).clamp(0, 2)
    V, N = S.shape
    geo = torch.cat([B6 / 2.0, b[:, None], b_delta[:, None]], -1).to(S.dtype)       # (N, 8)
    return torch.cat([geo.expand(V, N, 8), s[..., None], torch.log(s.clamp_min(1e-3))[..., None]], -1)


class SetEncoder(nn.Module):
    """phi on each measurement, mean+max pooling, then rho. Order-free and length-free."""

    def __init__(self, d_in=10, width=128, d_emb=64):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(d_in, width), nn.GELU(), nn.Linear(width, width), nn.GELU(),
                                 nn.Linear(width, width))
        self.rho = nn.Sequential(nn.Linear(2 * width, 256), nn.GELU(), nn.Linear(256, d_emb), nn.LayerNorm(d_emb))

    def forward(self, x, mask=None):
        h = self.phi(x)
        if mask is None:
            pooled = torch.cat([h.mean(1), h.amax(1)], -1)
        else:
            m = mask[..., None].to(h.dtype)
            pooled = torch.cat([(h * m).sum(1) / m.sum(1).clamp_min(1), h.masked_fill(m == 0, -1e9).amax(1)], -1)
        return self.rho(pooled)


class PosteriorNet(nn.Module):
    def __init__(self, d_emb=64, transforms=4, hidden=(128, 128)):
        super().__init__()
        import zuko
        self.encoder = SetEncoder(d_emb=d_emb)
        self.flow = zuko.flows.NSF(features=len(TARGETS), context=d_emb, transforms=transforms,
                                   hidden_features=hidden)

    def embed(self, x, mask=None):
        return self.encoder(x, mask)

    def log_prob(self, x, y, mask=None):
        return self.flow(self.embed(x, mask)).log_prob(to_unconstrained(y))

    @torch.no_grad()
    def sample(self, x, n=100, mask=None, chunk=64, compile=False, generator=None):
        """-> (n, V, len(TARGETS)) posterior samples in natural units.

        Base noise z ~ N(0, I) (the flow's base is a standard normal independent of the context, drawn from
        ``generator`` if given) is pushed through the inverse flow, one chunk of voxels at a time.

        ``compile=True`` runs the inverse through ``torch.compile`` (fused generated kernels). Sampling inverts
        each autoregressive transform one dimension at a time (32 passes of MLP + spline rebuild + spline
        inverse); in eager mode that is ~2,800 small kernels per chunk and, on CUDA, mostly PyTorch's slow
        cumsum over 8-bin rows. Compiled: 13.5x faster on an A10, 6.3x on MPS, within 2e-3 of eager
        (NOTES 2026-11-06). The first call per (n, chunk) compiles (20-50 s); the last chunk is padded to
        ``chunk`` so one graph serves every chunk. Use chunk >= 512 when compiling.
        """
        inv = self._compiled_inverse() if compile else self._inverse
        outs = []
        for i in range(0, x.shape[0], chunk):
            xi = x[i:i + chunk]
            mi = None if mask is None else mask[i:i + chunk]
            m = xi.shape[0]
            if compile and m < chunk:                     # pad, so the compiled graph keeps one shape
                xi = torch.cat([xi, xi[-1:].expand(chunk - m, *xi.shape[1:])])
                if mi is not None:
                    mi = torch.cat([mi, mi[-1:].expand(chunk - m, *mi.shape[1:])])
            z = torch.randn(n, xi.shape[0], len(TARGETS), generator=generator, device=x.device, dtype=x.dtype)
            outs.append(from_unconstrained(inv(self.embed(xi, mi), z)[:, :m]))
        return torch.cat(outs, 1)

    def _inverse(self, ctx, z):
        return self.flow(ctx).transform.inv(z)

    def _compiled_inverse(self):
        if self.__dict__.get("_inv_compiled") is None:
            self.__dict__["_inv_compiled"] = torch.compile(self._inverse, dynamic=False)
        return self.__dict__["_inv_compiled"]


@dataclass
class SimBatch:
    x: torch.Tensor
    y: torch.Tensor
    cls: torch.Tensor
    S: torch.Tensor


def simulate_batch(protocol, V, g=None, snr=(15.0, 60.0), M=64, dtype=torch.float32, prior_version=2,
                   log_snr=True, device=None, sim_dtype=None, exact_fw=False, td_ratio=None, alpha=None,
                   pulsation=0.0, gain_sd=0.0):
    """Brain-prior voxels through ``protocol`` with Rician noise -> SimBatch (features, targets, class, signal).

    SNR (at b = 0) is drawn log-uniformly over ``snr`` unless ``log_snr`` is False.

    ``device``: None (default) simulates on CPU in float64, the path every saved result used.
    Given a device, the whole simulation runs there (``g`` must be a generator on that device) in
    ``sim_dtype``, default float32: MPS has no float64 and consumer GPUs run it slowly. On a GPU this
    removes the CPU simulation that otherwise dominates a training step. ``dtype`` is the output dtype
    of features and targets.

    ``exact_fw``: free water as an exactly weighted compartment (``simulate.sample_brain``), so true f_fw is
    the voxel's free-water fraction rather than a random count out of M (brain priors only).

    Forward-model options (all off by default; brain priors only):
    * ``td_ratio``: time dependence in WM only (``simulate.sample_tau`` with GM strengths 0): spherical
      encodings (b_delta ~ 0) at relative frequency ``td_ratio``, the others at 1.
    * ``alpha`` (lo, hi): signal-proportional noise, S -> S (1 + a eps) per measurement, with a per-voxel
      a ~ U(lo, hi) (fitted tissue values about 0.005-0.02, NOTES 2026-11-05).
    * ``pulsation``: extra fractional noise for fluid, a += U(0, pulsation) * f_fw * max(0, 1 - b), b in
      ms/um^2 (CSF: about 0.1 at b = 0.1, falling to about 0.03 by b = 1).
    * ``gain_sd``: per-measurement gain S -> S (1 + gain_sd * eps), independent per voxel (per-volume
      calibration offsets such as the STE period-3 pattern, about 0.5-1%).
    Multiplicative terms act on the clean signal before Rician thermal noise at the drawn SNR.
    """
    if device is None:
        kw = {}
    else:
        kw = dict(device=device, dtype=torch.float32 if sim_dtype is None else sim_dtype)
    wts, comp = None, None
    need_comp = td_ratio is not None
    if prior_version == 0:                          # the generic, non-anatomical prior (simulate.Prior)
        assert not (need_comp or alpha or pulsation or gain_sd), "forward-model options need a brain prior"
        d6, free = simulate.sample_dtd(V, M=M, g=g, return_free=True, **kw)
        cls = torch.full((V,), -1, dtype=torch.long, device=d6.device)
    elif exact_fw:
        out = simulate.sample_brain(V, M=M, g=g, return_free=True, version=prior_version, exact_fw=True,
                                    return_comp=need_comp, **kw)
        d6, cls, free, wts = out[:4]
        comp = out[4] if need_comp else None
    elif need_comp:
        d6, cls, comp = simulate.sample_brain(V, M=M, g=g, version=prior_version, return_comp=True, **kw)
        free = comp == 6
    else:
        d6, cls, free = simulate.sample_brain(V, M=M, g=g, return_free=True, version=prior_version, **kw)
    dev = d6.device
    b6 = protocol.b6.to(dtype=d6.dtype, device=dev)
    b = protocol.b.to(dtype=d6.dtype, device=dev)
    bd = protocol.b_delta().to(dtype=d6.dtype, device=dev)
    y = targets(d6, free, wts)
    if need_comp:
        tau = simulate.sample_tau(cls, comp, g, gm_scale=0.0, dtype=d6.dtype)
        omega = 1.0 + (td_ratio - 1.0) * ((bd.abs() < 0.1) & (b > 0.05)).to(d6.dtype)
        clean = simulate.signal_td(d6, b6, omega, tau, weights=wts)
    else:
        clean = simulate.signal(d6, b6, weights=wts)
    N = clean.shape[1]
    if gain_sd:
        clean = clean * (1 + gain_sd * torch.randn(V, N, generator=g, dtype=d6.dtype, device=dev))
    if alpha is not None or pulsation:
        a = torch.zeros(V, 1, dtype=d6.dtype, device=dev)
        if alpha is not None:
            a = a + alpha[0] + (alpha[1] - alpha[0]) * torch.rand(V, 1, generator=g, dtype=d6.dtype, device=dev)
        if pulsation:
            amp = pulsation * torch.rand(V, 1, generator=g, dtype=d6.dtype, device=dev)
            a = a + amp * y[:, :1].to(d6.dtype) * (1 - b).clamp_min(0)[None]
        clean = clean * (1 + a * torch.randn(V, N, generator=g, dtype=d6.dtype, device=dev))
    u = torch.rand(V, dtype=d6.dtype, generator=g, device=dev)
    lo, hi = snr
    s = torch.exp(torch.log(torch.tensor(lo)) + u * torch.log(torch.tensor(hi / lo))) if log_snr else lo + (hi - lo) * u
    S = simulate.rician(clean, 1.0 / s, g)
    x = features(S, b6, b, bd)
    return SimBatch(x.to(dtype), y.to(dtype), cls, S)
