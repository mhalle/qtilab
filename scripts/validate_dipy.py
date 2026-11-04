"""Validate qtilab's batched fits against DIPY on the DiB open data.

WLS is compared on every brain voxel; the constrained fit (QTI+ dc) on a random subset,
because DIPY solves one SDP per voxel. DIPY's constrained fit is run with SCS (its default)
and with CLARABEL (an interior-point solver, tighter) as the reference.
"""
import json
import sys
import time

import numpy as np
import torch
from dipy.core.gradients import gradient_table
import dipy.reconst.qti as dq

sys.path.insert(0, ".")
from qtilab import fit, scalars  # noqa: E402
from qtilab.data import load_dib  # noqa: E402

KEYS = ["md", "fa", "ufa", "c_md", "c_c", "mk", "k_bulk", "k_mu"]
UNIT = {"md": 1e-3}  # DIPY md in mm^2/s, ours in um^2/ms


def dipy_scalars(f):
    return {k: getattr(f, k) for k in KEYS}


def ours(params):
    s = scalars.scalars(params)
    return {k: (s[k] * UNIT.get(k, 1.0)).numpy() for k in KEYS}


def compare(a, b, label):
    rows = {}
    for k in KEYS:
        x, y = np.asarray(a[k]).ravel(), np.asarray(b[k]).ravel()
        ok = np.isfinite(x) & np.isfinite(y)
        d = np.abs(x[ok] - y[ok])
        scale = np.nanpercentile(np.abs(y[ok]), 50) or 1.0
        rows[k] = {"median_abs": float(np.median(d)), "p99_abs": float(np.percentile(d, 99)),
                   "max_abs": float(d.max()), "median_rel_to_typical": float(np.median(d) / scale)}
    print(f"\n{label}")
    print(f"{'scalar':8s} {'median|d|':>11s} {'p99|d|':>11s} {'max|d|':>11s}")
    for k, r in rows.items():
        print(f"{k:8s} {r['median_abs']:11.2e} {r['p99_abs']:11.2e} {r['max_abs']:11.2e}")
    return rows


def main(name="DiB_217_lte_pte_ste", n_sub=1500, seed=0):
    data, mask, aff, prot, bvals, bvecs, shapes = load_dib(name)
    gtab = gradient_table(bvals, bvecs=bvecs, btens=np.array(shapes))
    S = data[mask].astype(np.float64)
    S = np.maximum(S, 1e-4)
    V = S.shape[0]
    X = prot.design_matrix()
    St = torch.as_tensor(S)
    out = {"dataset": name, "voxels": V, "volumes": prot.n}

    # --- WLS, all voxels
    t = time.perf_counter(); fd = dq.QtiModel(gtab, fit_method="WLS").fit(S); t_dipy = time.perf_counter() - t
    t = time.perf_counter(); p = fit.wls(X, St); t_ours = time.perf_counter() - t
    out["wls"] = {"dipy_s": t_dipy, "ours_s": t_ours, "diff": compare(ours(p), dipy_scalars(fd), f"WLS, {V} voxels")}
    print(f"time: DIPY {t_dipy:.1f} s, qtilab {t_ours:.2f} s")

    # --- constrained, all voxels (ours)
    t = time.perf_counter(); pc, info = fit.constrained(X, St); t_c = time.perf_counter() - t
    print(f"\nconstrained (qtilab), all {V} voxels: {t_c:.2f} s; {info.n_violating} violated WLS "
          f"({100*info.n_violating/V:.1f}%), ADMM iterations {info.iterations}, converged {info.converged_fraction:.4f}")
    out["constrained_all"] = {"s": t_c, "n_violating": info.n_violating, "iterations": info.iterations,
                              "converged_fraction": info.converged_fraction}

    # --- constrained vs DIPY on a subset (oversample violators so the comparison means something)
    rng = np.random.default_rng(seed)
    bad = fit.violates(p).numpy()
    idx_bad = rng.choice(np.flatnonzero(bad), size=min(n_sub * 2 // 3, bad.sum()), replace=False)
    idx_ok = rng.choice(np.flatnonzero(~bad), size=n_sub - len(idx_bad), replace=False)
    idx = np.r_[idx_bad, idx_ok]
    Ssub = S[idx]
    res = {}
    for solver in ["CLARABEL", "SCS"]:
        t = time.perf_counter()
        f = dq.QtiModel(gtab, fit_method="SDPdc", cvxpy_solver=solver).fit(Ssub)
        res[solver] = (dipy_scalars(f), time.perf_counter() - t)
        print(f"DIPY SDPdc/{solver}: {res[solver][1]:.1f} s for {len(idx)} voxels "
              f"({1e3*res[solver][1]/len(idx):.1f} ms/voxel)")
    ref = res["CLARABEL"][0]
    mine = ours(pc[idx])
    out["constrained_subset"] = {
        "n": int(len(idx)), "n_violating_in_subset": int(len(idx_bad)),
        "dipy_clarabel_s": res["CLARABEL"][1], "dipy_scs_s": res["SCS"][1],
        "ours_vs_clarabel": compare(mine, ref, f"qtilab constrained vs DIPY SDPdc/CLARABEL ({len(idx)} voxels, {len(idx_bad)} violators)"),
        "scs_vs_clarabel": compare(res["SCS"][0], ref, "DIPY SDPdc/SCS vs DIPY SDPdc/CLARABEL (solver-to-solver baseline)"),
    }
    json.dump(out, open(f"results/validate_{name}.json", "w"), indent=1)


if __name__ == "__main__":
    main(*(sys.argv[1:2] or []))
