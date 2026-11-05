"""Noise model with a thermal and a signal-proportional (physiological) term.

    var(S) = s2 + alpha(b)^2 * S^2

alpha may depend on b: pulsatile flow and motion perturb lightly diffusion-weighted images most.

s2 is the thermal variance (per voxel; spatially varying with coil sensitivity), alpha the
fractional signal fluctuation (pulsation, residual motion, intensity drift), which a b = 1500 repeat
estimate misses because the signal there is small.

Estimation uses groups of repeated volumes with identical expected signal: spherical-encoding (STE)
volumes at the same b, whatever their nominal direction. The sample variance of a group with n
volumes has n - 1 degrees of freedom; alpha is chosen by profile likelihood with s2 solved per voxel.
"""
from __future__ import annotations

import numpy as np


def ste_groups(bvals, shapes, exclude=None, tol=1.0):
    """Index arrays of STE repeat groups (bvals in s/mm^2), optionally excluding some volumes."""
    bvals = np.asarray(bvals, float)
    shapes = np.asarray([str(s).upper() for s in shapes])
    ex = np.zeros(len(bvals), bool) if exclude is None else np.asarray(exclude, bool)
    levels = sorted({round(float(b)) for b, s in zip(bvals, shapes) if s == "STE"})
    groups = [np.flatnonzero((shapes == "STE") & (np.abs(bvals - lv) < tol) & ~ex) for lv in levels]
    return [g for g in groups if len(g) >= 2], [lv for lv, g in zip(levels, groups) if len(g) >= 2]


def group_moments(S, groups):
    """S (V, N) -> means (V, G), variances (V, G, ddof=1), dof (G,)."""
    m = np.stack([S[:, g].mean(1) for g in groups], 1)
    v = np.stack([S[:, g].var(1, ddof=1) for g in groups], 1)
    return m, v, np.array([len(g) - 1 for g in groups], float)


def _s2_mle(v, a2m2, dof, iters=30):
    """Per-voxel maximum-likelihood thermal variance given the signal-proportional part a2m2 (V, G).

    Stationarity: sum_g dof_g (v_g - s2 - a_g) / (s2 + a_g)^2 = 0, solved by reweighted fixed point.
    """
    s2 = np.clip(((v - a2m2) * dof).sum(1) / dof.sum(), 1e-10, None)
    for _ in range(iters):
        w = dof / (s2[:, None] + a2m2) ** 2
        s2 = np.clip((w * (v - a2m2)).sum(1) / w.sum(1), 1e-10, None)
    return s2


def profile(m, v, dof, alpha):
    """Thermal variance s2 (V,) and log-likelihood (V,) at a given alpha."""
    a2m2 = alpha ** 2 * m ** 2 + np.zeros_like(m)
    s2 = _s2_mle(v, a2m2, dof)
    sig2 = s2[:, None] + a2m2
    ll = (-0.5 * dof * (np.log(sig2) + v / sig2)).sum(1)
    return s2, ll


def fit_alpha(m, v, dof, alphas=np.linspace(0.0, 0.15, 61)):
    """Pooled alpha for a set of voxels by profile likelihood -> (alpha, s2 per voxel, ll curve)."""
    lls = np.array([profile(m, v, dof, a)[1].sum() for a in alphas])
    a = float(alphas[int(np.argmax(lls))])
    return a, profile(m, v, dof, a)[0], lls


def profile_groups(m, v, dof, alphas):
    """Like ``profile`` but with one alpha per group (G,)."""
    a2m2 = (np.asarray(alphas)[None, :] ** 2) * m ** 2
    s2 = _s2_mle(v, a2m2, dof)
    sig2 = s2[:, None] + a2m2
    return s2, (-0.5 * dof * (np.log(sig2) + v / sig2)).sum(1)


def fit_alpha_groups(m, v, dof, grid=np.linspace(0.0, 0.2, 81), sweeps=4):
    """Pooled alpha per repeat group (i.e. per b) by coordinate ascent of the profile likelihood,
    started from the best single (b-independent) alpha."""
    al = np.full(m.shape[1], fit_alpha(m, v, dof, alphas=grid)[0])
    for _ in range(sweeps):
        for j in range(len(al)):
            best = None
            for a in grid:
                trial = al.copy(); trial[j] = a
                ll = profile_groups(m, v, dof, trial)[1].sum()
                if best is None or ll > best[0]:
                    best = (ll, a)
            al[j] = best[1]
    s2, ll = profile_groups(m, v, dof, al)
    return al, s2, float(ll.sum())


def alpha_for_bvals(bvals, levels, alphas):
    """Per-volume alpha from per-level estimates: linear in b between levels, held flat outside them."""
    return np.interp(np.asarray(bvals, float), np.asarray(levels, float), np.asarray(alphas, float))


def region_of(fa, md):
    """Coarse noise regions from classical scalars: 0 WM, 1 GM, 2 CSF."""
    return np.where(md > 2.0, 2, np.where(fa > 0.4, 0, 1))


def sigma(S, s2, alpha):
    """Noise sd for signals S (V, N) given per-voxel s2 (V,) and alpha (scalar or (V,))."""
    alpha = np.asarray(alpha, float)
    a = alpha[:, None] if (alpha.ndim == 1 and alpha.shape[0] == S.shape[0] and S.shape[0] != S.shape[1]) else alpha
    return np.sqrt(s2[:, None] + (a * S) ** 2)
