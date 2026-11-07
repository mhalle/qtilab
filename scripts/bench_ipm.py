"""Constrained QTI fit on whole real sessions: ADMM (``fit._admm``) vs the interior-point solver (``qtilab.ipm``).

For each dataset, every brain voxel is fitted on each device (CPU float64, MPS/CUDA float32) with
``fit.constrained`` (default chunking) and both solvers at default settings, plus ADMM capped at 300
iterations, IPM without its line search, and IPM at eps_rel 1e-7 (the default is 1e-8). Accuracy (violating, non-collapsed voxels) is against a tight reference, IPM at eps_rel 1e-11 on CPU float64 (which matches tight
ADMM to <= 3e-7 in µFA and DIPY/CLARABEL as closely as tight ADMM does: scripts/validate_dipy_hhs.py).
WLS time (shared by both solvers) is reported separately. Feasibility is relative: no eigenvalue below
-1e-6 x the block's largest.

Datasets: ``hhs`` (sub-01 ses-01, LTE + STE to b = 2, rank 23) and ``cardiff`` (data1, motion-corrected,
LTE + PTE + STE to b = 2, full rank).

    uv run python scripts/bench_ipm.py [--devices cpu mps] [--data hhs cardiff]
-> results/bench_ipm_<host>.json
"""
import argparse
import json
import platform
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from qtilab import fit, paths, scalars  # noqa: E402
from qtilab.protocol import Protocol  # noqa: E402
from qtilab.tensors import mat3, mat6  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--devices", nargs="+", default=["cpu", "mps"])
ap.add_argument("--data", nargs="+", default=["hhs", "cardiff"])
ap.add_argument("--host", default=platform.node().split(".")[0].lower())
args = ap.parse_args()
torch.set_num_threads(8)
KEYS = ("md", "fa", "ufa", "c_md", "c_mu")


def load(name):
    if name == "hhs":
        a, c = np.load(paths.aligned_session("01", "01")), np.load(paths.classical_fit("01", "01"))
        S = a["data"][c["mask"]].astype(np.float64)
        prot = Protocol.from_table(a["bvals"], a["bvecs"], [str(s) for s in a["shapes"]])
    else:
        z = np.load(paths.WORK / "cardiff_data1ec_aligned.npz")
        keep = z["bvals"] <= 2060
        S = z["data"][z["mask"]][:, keep].astype(np.float64)
        prot = Protocol.from_table(z["bvals"][keep], z["bvecs"][:, keep] if z["bvecs"].shape[0] == 3 else z["bvecs"][keep],
                                   [str(s) for s in z["shapes"][keep]])
    return prot.design_matrix(), np.maximum(S, 1e-4 * S.max())


def sync(dev):
    if dev == "mps":
        torch.mps.synchronize()
    elif dev == "cuda":
        torch.cuda.synchronize()


def timed(dev, fn):
    sync(dev)
    t = time.perf_counter()
    out = fn()
    sync(dev)
    return out, time.perf_counter() - t


RUNS = {"admm": {}, "admm_300": {"max_iter": 300}, "ipm": {"solver": "ipm"},
        "ipm_nols": {"solver": "ipm", "line_search": False}, "ipm_1e-7": {"solver": "ipm", "eps_rel": 1e-7}}


def feasible(p, rel=1e-6):
    """No eigenvalue of <D> or C below -rel x that block's largest |eigenvalue| (float32 rounding)."""
    out = torch.ones(len(p), dtype=torch.bool)
    for m in (mat3(p[:, 1:7]), mat6(p[:, 7:])):
        e = torch.linalg.eigvalsh(m)
        out &= e[:, 0] >= -rel * e.abs().amax(-1)
    return out


res = {"host": args.host, "torch": torch.__version__, "data": {}}
for name in args.data:
    X, S = load(name)
    Xc, Sc = X.double(), torch.as_tensor(S)
    ref, info = fit.constrained(Xc, Sc, solver="ipm", eps_rel=1e-11)
    rsc = scalars.scalars(ref)
    bad = fit.violates(fit.wls(Xc, Sc)).numpy()
    ok = bad & ~scalars.collapsed(rsc).numpy()          # collapsed fits (<D> = 0): ratio scalars meaningless
    out = {"n_vox": len(S), "n_vol": X.shape[0], "rank": int(torch.linalg.matrix_rank(X)), "violating": float(bad.mean()), "collapsed": int(scalars.collapsed(rsc).sum()),
           "ref_newton_steps": info.extra, "runs": {}}
    print(f"\n== {name}: {len(S)} voxels x {X.shape[0]} volumes, rank {out['rank']}, WLS violates in {bad.mean():.1%}", flush=True)
    for dev in args.devices:
        dt = torch.float64 if dev == "cpu" else torch.float32
        Xd, Sd = X.to(dev, dt), torch.as_tensor(S, dtype=dt, device=dev)
        fit.constrained(Xd, Sd[:2000], solver="ipm"); fit.constrained(Xd, Sd[:2000])      # warm-up
        _, t_wls = timed(dev, lambda: fit.violates(fit.wls(Xd, Sd)))
        for rname, kw in RUNS.items():
            (p, info), t = timed(dev, lambda: fit.constrained(Xd, Sd, **kw))
            sc = scalars.scalars(p.cpu().double())
            row = {"seconds": round(t, 2), "wls_seconds": round(t_wls, 2), "iterations": info.iterations,
                   "converged_fraction": round(info.converged_fraction, 4), **({"ipm": info.extra} if info.extra else {}),
                   "infeasible_fraction": float((~feasible(p.cpu().double())).float().mean())}
            for k in KEYS:
                d = (sc[k] - rsc[k]).abs().numpy()[ok & ~scalars.collapsed(sc).numpy()]
                d = d[np.isfinite(d)]
                row[k] = {"median": float(np.median(d)), "p99": float(np.percentile(d, 99)), "max": float(d.max())}
            out["runs"][f"{dev}_{rname}"] = row
            print(f"{dev:4s} {rname:9s} {t:6.1f} s (WLS + check {t_wls:4.1f} s)  iters {info.iterations:4d}  conv {info.converged_fraction:.3f}  "
                  + "  ".join(f"{k} {row[k]['p99']:.0e}/{row[k]['max']:.0e}" for k in ("md", "ufa", "c_md"))
                  + (f"  steps {info.extra['median_steps']:.0f}/{info.extra['max_steps']} failed {info.extra['failed_fraction']:.1e}"
                     f" (near target {info.extra['failed_near_target']:.1e}) uncert {info.extra['uncertified_fraction']:.1e}"
                     if info.extra else ""), flush=True)
    res["data"][name] = out
json.dump(res, open(f"results/bench_ipm_{args.host}.json", "w"), indent=1)
print(f"-> results/bench_ipm_{args.host}.json")
