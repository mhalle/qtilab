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


def targets(d6: torch.Tensor, free: torch.Tensor) -> torch.Tensor:
    """(V, M, 6) micro-tensors and (V, M) free-water mask -> (V, len(TARGETS))."""
    full = scalars.scalars(simulate.dtd_params(d6))
    tissue = scalars.scalars(simulate.dtd_params(d6, weights=(~free).to(d6.dtype)))
    f_fw = free.to(d6.dtype).mean(1)
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
    def sample(self, x, n=100, mask=None, chunk=64):
        """-> (n, V, len(TARGETS)) posterior samples in natural units."""
        outs = []
        for i in range(0, x.shape[0], chunk):
            ctx = self.embed(x[i:i + chunk], None if mask is None else mask[i:i + chunk])
            outs.append(from_unconstrained(self.flow(ctx).sample((n,))))
        return torch.cat(outs, 1)


@dataclass
class SimBatch:
    x: torch.Tensor
    y: torch.Tensor
    cls: torch.Tensor
    S: torch.Tensor


def simulate_batch(protocol, V, g=None, snr=(15.0, 60.0), M=64, dtype=torch.float32, prior_version=2,
                   log_snr=True):
    """Brain-prior voxels through ``protocol`` with Rician noise -> SimBatch (features, targets, class, signal).

    SNR (at b = 0) is drawn log-uniformly over ``snr`` unless ``log_snr`` is False.
    """
    if prior_version == 0:                          # the generic, non-anatomical prior (simulate.Prior)
        d6, free = simulate.sample_dtd(V, M=M, g=g, return_free=True)
        cls = torch.full((V,), -1, dtype=torch.long)
    else:
        d6, cls, free = simulate.sample_brain(V, M=M, g=g, return_free=True, version=prior_version)
    clean = simulate.signal(d6, protocol.b6.to(d6.dtype))
    u = torch.rand(V, dtype=d6.dtype, generator=g)
    lo, hi = snr
    s = torch.exp(torch.log(torch.tensor(lo)) + u * torch.log(torch.tensor(hi / lo))) if log_snr else lo + (hi - lo) * u
    S = simulate.rician(clean, 1.0 / s, g)
    x = features(S, protocol.b6.to(d6.dtype), protocol.b.to(d6.dtype), protocol.b_delta().to(d6.dtype))
    return SimBatch(x.to(dtype), targets(d6, free).to(dtype), cls, S)
