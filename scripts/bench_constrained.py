"""Speed/accuracy of qtilab's constrained fit against a cached DIPY SDPdc/CLARABEL reference."""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from dipy.core.gradients import gradient_table
import dipy.reconst.qti as dq

sys.path.insert(0, ".")
from qtilab import fit, scalars  # noqa: E402
from qtilab.data import load_dib  # noqa: E402

KEYS = ["md", "fa", "ufa", "c_md", "c_c", "mk", "k_bulk", "k_mu"]
name = sys.argv[1] if len(sys.argv) > 1 else "DiB_217_lte_pte_ste"
data, mask, aff, prot, bvals, bvecs, shapes = load_dib(name)
S = np.maximum(data[mask].astype(np.float64), 1e-4)
X = prot.design_matrix()
St = torch.as_tensor(S)

cache = Path(f"results/clarabel_ref_{name}.npz")
p_wls = fit.wls(X, St)
bad = fit.violates(p_wls).numpy()
if not cache.exists():
    rng = np.random.default_rng(0)
    idx = rng.choice(np.flatnonzero(bad), size=1000, replace=False)
    gtab = gradient_table(bvals, bvecs=bvecs, btens=np.array(shapes))
    f = dq.QtiModel(gtab, fit_method="SDPdc", cvxpy_solver="CLARABEL").fit(S[idx])
    np.savez(cache, idx=idx, **{k: getattr(f, k) for k in KEYS})
ref = np.load(cache)
idx = ref["idx"]

for kw in [dict(), dict(eps_abs=1e-7, eps_rel=1e-5)]:
    t = time.perf_counter()
    pc, info = fit.constrained(X, St, **kw)
    dt = time.perf_counter() - t
    sc = scalars.scalars(pc[idx])
    errs = {k: np.median(np.abs(sc[k].numpy() * (1e-3 if k == "md" else 1) - ref[k])) for k in KEYS}
    p99 = {k: np.percentile(np.abs(sc[k].numpy() * (1e-3 if k == "md" else 1) - ref[k]), 99) for k in KEYS}
    print(f"{kw or 'defaults'}: {dt:.1f} s for {S.shape[0]} voxels ({info.n_violating} constrained), "
          f"iters {info.iterations}, converged {info.converged_fraction:.4f}")
    print("   median |d| vs CLARABEL:", {k: f"{v:.1e}" for k, v in errs.items()})
    print("   p99    |d| vs CLARABEL:", {k: f"{v:.1e}" for k, v in p99.items()})
