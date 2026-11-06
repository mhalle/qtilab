"""Constrained QTI fit vs DIPY SDPdc on the Hormone Health protocol (LTE + STE, rank 23 of 28).

The DIPY validation so far (validate_dipy.py, bench_constrained.py) used the full-rank DiB protocol.
On HHS the design is rank-deficient: fit.constrained adds a small ridge and ADMM converges slowly,
and every real-data result rests on it. This compares, on the same voxels of one aligned session:

* qtilab constrained, default settings, and with tight tolerances (separates solver tolerance from a
  formulation difference);
* DIPY SDPdc with CLARABEL (reference) and with SCS (DIPY's default; a solver-vs-solver baseline).

Scalars are compared directly. Because the scalars cannot tell which solution is better, each solution is
also scored on the shared objective 1/2 x'Hx - g'x (DIPY-equivalent weights, from fit._normal_equations):
lower is better among feasible points.

    uv run python scripts/validate_dipy_hhs.py [sub] [ses] [n_violating] [n_ok]
Needs work/hhs_sub{sub}_ses{ses}_{aligned,qti}.npz (scripts/prepare_session.py).
-> results/validate_dipy_hhs_sub{sub}_ses{ses}.json; DIPY fits cached in work/.
"""
import json
import sys
import time
import warnings

import numpy as np
import torch
from dipy.core.gradients import gradient_table
import dipy.reconst.qti as dq

sys.path.insert(0, ".")
from qtilab import fit, paths, scalars  # noqa: E402
from qtilab.protocol import Protocol  # noqa: E402
from qtilab.tensors import mat3, mat6  # noqa: E402

sub = sys.argv[1] if len(sys.argv) > 1 else "01"
ses = sys.argv[2] if len(sys.argv) > 2 else "01"
n_bad = int(sys.argv[3]) if len(sys.argv) > 3 else 2500
n_ok = int(sys.argv[4]) if len(sys.argv) > 4 else 500
KEYS = ["md", "fa", "ufa", "c_md", "c_mu", "c_c", "mk", "k_bulk", "k_mu"]

a = np.load(paths.aligned_session(sub, ses))
c = np.load(paths.classical_fit(sub, ses))
shapes = [str(s) for s in a["shapes"]]
prot = Protocol.from_table(a["bvals"], a["bvecs"], shapes)
X = prot.design_matrix()
print(f"protocol: {prot.n} volumes, rank {prot.rank()}", flush=True)

S_all = np.maximum(a["data"][c["mask"]].astype(np.float64), 1e-4)
p_wls = fit.wls(X, torch.as_tensor(S_all))
bad = fit.violates(p_wls).numpy()
rng = np.random.default_rng(0)
idx = np.concatenate([rng.choice(np.flatnonzero(bad), n_bad, replace=False),
                      rng.choice(np.flatnonzero(~bad), n_ok, replace=False)])
viol = np.r_[np.ones(n_bad, bool), np.zeros(n_ok, bool)]
S = S_all[idx]
St = torch.as_tensor(S)
print(f"{bad.mean():.0%} of {len(bad)} brain voxels violate under WLS; testing {n_bad} violating + {n_ok} not", flush=True)

# ---- fits
res = {}
t = time.perf_counter(); res["qtilab"], info = fit.constrained(X, St); t_q = time.perf_counter() - t
t = time.perf_counter()
res["qtilab_tight"], info_t = fit.constrained(X, St, eps_abs=1e-9, eps_rel=1e-7, max_iter=20000)
t_qt = time.perf_counter() - t
print(f"qtilab default: {t_q:.1f} s, converged {info.converged_fraction:.4f}, iters {info.iterations}", flush=True)
print(f"qtilab tight:   {t_qt:.1f} s, converged {info_t.converged_fraction:.4f}, iters {info_t.iterations}", flush=True)

gtab = gradient_table(a["bvals"], bvecs=a["bvecs"].T if a["bvecs"].shape[0] == 3 else a["bvecs"],
                      btens=np.array(shapes))
timing = {"qtilab": t_q, "qtilab_tight": t_qt}
for solver in ("CLARABEL", "SCS"):
    cache = paths.WORK / f"dipy_sdpdc_{solver}_sub{sub}_ses{ses}_{n_bad}_{n_ok}.npz"
    if cache.exists() and np.array_equal(np.load(cache)["idx"], idx):
        z = np.load(cache); p, dt = z["params"], float(z["seconds"])
    else:
        t = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = dq.QtiModel(gtab, fit_method="SDPdc", cvxpy_solver=solver).fit(S).params
        dt = time.perf_counter() - t
        np.savez(cache, idx=idx, params=p, seconds=dt)
    # DIPY works in the units of the bvals it is given (s/mm^2), so its <D> is in mm^2/s and C in (mm^2/s)^2;
    # qtilab's parameters are in um^2/ms. Convert before comparing MD or scoring the shared objective.
    p = np.array(p, dtype=np.float64)
    p[:, 1:7] *= 1e3
    p[:, 7:] *= 1e6
    res[f"dipy_{solver}"] = torch.as_tensor(p)
    timing[f"dipy_{solver}"] = dt
    print(f"DIPY SDPdc/{solver}: {dt:.0f} s ({1000 * dt / len(idx):.0f} ms/voxel)", flush=True)

# ---- objective and feasibility
w = fit._weights(X, St, None)
H, g = fit._normal_equations(X, fit._log_signal(St), w)


def objective(p):
    p = p.to(H.dtype)
    return 0.5 * torch.einsum("vi,vij,vj->v", p, H, p) - (g * p).sum(-1)


def min_eig(p):
    return torch.minimum(torch.linalg.eigvalsh(mat3(p[:, 1:7]))[:, 0], torch.linalg.eigvalsh(mat6(p[:, 7:]))[:, 0])


J = {k: objective(v) for k, v in res.items()}
sc = {k: scalars.scalars(v.to(torch.float64)) for k, v in res.items()}
scale = J["dipy_CLARABEL"].abs().clamp_min(1e-12)


def compare(a_, b_, mask):
    out = {}
    for k in KEYS:
        d = (sc[a_][k] - sc[b_][k]).abs().numpy()[mask]
        out[k] = {"median": float(np.median(d)), "p99": float(np.percentile(d, 99)), "max": float(d.max())}
    gap = ((J[a_] - J[b_]) / scale).numpy()[mask]          # > 0: a_ worse than b_
    out["objective_rel_gap"] = {"median": float(np.median(gap)), "p99": float(np.percentile(gap, 99)),
                                "frac_worse_1e-6": float((gap > 1e-6).mean()),
                                "frac_better_1e-6": float((gap < -1e-6).mean())}
    return out


report = {"sub": sub, "ses": ses, "protocol_volumes": prot.n, "rank": prot.rank(), "n_violating": n_bad,
          "n_nonviolating": n_ok, "seconds": timing,
          "qtilab_converged_fraction": {"default": info.converged_fraction, "tight": info_t.converged_fraction},
          "min_eigenvalue": {k: {"min": float(min_eig(v).min()), "frac_below_-1e-9": float((min_eig(v) < -1e-9).float().mean())}
                             for k, v in res.items()},
          "violating": {}, "nonviolating": {}}
pairs = [("qtilab", "dipy_CLARABEL"), ("qtilab_tight", "dipy_CLARABEL"), ("dipy_SCS", "dipy_CLARABEL"),
         ("qtilab", "qtilab_tight")]
for name, m in (("violating", viol), ("nonviolating", ~viol)):
    for a_, b_ in pairs:
        report[name][f"{a_} vs {b_}"] = compare(a_, b_, m)

out = f"results/validate_dipy_hhs_sub{sub}_ses{ses}.json"
json.dump(report, open(out, "w"), indent=1)

print("\nmin eigenvalue of <D>, C (feasibility):", {k: f"{v['min']:.1e}" for k, v in report["min_eigenvalue"].items()})
for name in ("violating", "nonviolating"):
    print(f"\n== {name} voxels: |scalar difference| median / p99 / max; objective gap (relative, + = first is worse)")
    for pair, r in report[name].items():
        print(f"  {pair}")
        print("    " + "  ".join(f"{k} {r[k]['median']:.0e}/{r[k]['p99']:.0e}/{r[k]['max']:.0e}" for k in ("md", "fa", "ufa", "c_md", "k_bulk")))
        o = r["objective_rel_gap"]
        print(f"    objective gap median {o['median']:+.1e}, p99 {o['p99']:+.1e}; "
              f"worse in {o['frac_worse_1e-6']:.1%}, better in {o['frac_better_1e-6']:.1%} of voxels (|gap| > 1e-6)")
print(f"-> {out}")
