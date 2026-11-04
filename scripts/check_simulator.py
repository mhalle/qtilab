"""Simulator sanity checks with the DiB_217 protocol.

1. Exact Gaussian DTD, tiny b: the fit must recover ground truth (cumulant expansion exact as b -> 0).
2. Real b-values, no noise: what's left is QTI's truncation bias, a property of the method.
3. Real b-values with Rician noise: WLS vs constrained, error against ground truth.
4. Throughput of simulation.
"""
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from qtilab import fit, scalars, simulate  # noqa: E402
from qtilab.data import load_dib  # noqa: E402
from qtilab.protocol import Protocol  # noqa: E402

KEYS = ["md", "fa", "ufa", "c_md", "c_c", "mk", "k_bulk", "k_mu"]
g = torch.Generator().manual_seed(1)
*_, prot, bvals, bvecs, shapes = load_dib("DiB_217_lte_pte_ste")
X = prot.design_matrix()


def report(label, est, truth):
    se, st = scalars.scalars(est), scalars.scalars(truth)
    print(f"\n{label}")
    print(f"{'scalar':8s} {'bias':>10s} {'MAE':>10s} {'truth IQR':>18s}")
    for k in KEYS:
        d = (se[k] - st[k]).numpy()
        ok = np.isfinite(d)
        q1, q3 = np.percentile(st[k].numpy()[ok], [25, 75])
        print(f"{k:8s} {np.median(d[ok]):10.4f} {np.median(np.abs(d[ok])):10.4f} {q1:8.3f}-{q3:<8.3f}")


V = 20000
sim = simulate.simulate(prot, V, g=g, noise=True)

# 1. tiny b: scale protocol by 1e-3 so the cumulant expansion is (nearly) exact
tiny = Protocol(prot.B * 1e-3)
clean_tiny = simulate.signal(sim["d6"], tiny.b6)
p = fit.wls(tiny.design_matrix(), clean_tiny)
s_est, s_tru = scalars.scalars(p), scalars.scalars(sim["params"])
print("1. tiny-b recovery, median / p99 |error|:",
      {k: f"{np.nanmedian(np.abs((s_est[k] - s_tru[k]).numpy())):.1e} / {np.nanpercentile(np.abs((s_est[k] - s_tru[k]).numpy()), 99):.1e}" for k in ["md", "fa", "ufa", "c_md", "k_mu"]})

# 2. real b, noise-free
report("2. real b-values, no noise (truncation bias only), WLS", fit.wls(X, sim["clean"]), sim["params"])

# 2b. truncation bias by how heterogeneous the voxel is (true MD-variance kurtosis)
pw0 = fit.wls(X, sim["clean"])
se0, st0 = scalars.scalars(pw0), scalars.scalars(sim["params"])
kb = st0["k_bulk"].numpy()
print("\n2b. noise-free bias vs true k_bulk tercile:")
for lo, hi in zip(np.percentile(kb, [0, 33, 67]), np.percentile(kb, [33, 67, 100])):
    m = (kb >= lo) & (kb <= hi)
    print(f"   k_bulk {lo:5.2f}-{hi:5.2f}: " + "  ".join(f"{k} {np.median((se0[k]-st0[k]).numpy()[m]):+.3f}" for k in ["md", "ufa", "c_md", "mk", "k_mu"]))

# 3. real b, Rician noise (SNR 15-80 at b=0)
pw = fit.wls(X, sim["signal"])
pc, info = fit.constrained(X, sim["signal"])
print(f"\n3. noisy: {info.n_violating}/{V} voxels violated the constraints under WLS")
report("3a. noisy, WLS", pw, sim["params"])
report("3b. noisy, constrained (QTI+ dc)", pc, sim["params"])

# 4. throughput
for n in [100_000]:
    t = time.perf_counter(); simulate.simulate(prot, n, M=48, g=g); dt = time.perf_counter() - t
    print(f"\n4. simulate {n} voxels x {prot.n} volumes (48 micro-tensors each): {dt:.1f} s on CPU, float64")
    t = time.perf_counter(); simulate.simulate(prot, n, M=48, g=g, dtype=torch.float32); dt = time.perf_counter() - t
    print(f"   same in float32: {dt:.1f} s")
