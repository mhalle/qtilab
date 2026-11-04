"""Recheck three early observations under the brain-like prior, for both real protocols.

1. Truncation bias of the QTI fit (noise-free), by tissue class.
2. Effect of the positivity constraints on µFA and other scalars under noise, by tissue class.
3. How close the true covariance is to singular (eigenvalues of C), by tissue class.
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from qtilab import fit, scalars, simulate  # noqa: E402
from qtilab.data import load_dib  # noqa: E402
from qtilab.hhs import session_protocol  # noqa: E402
from qtilab.tensors import mat6  # noqa: E402

KEYS = ["md", "fa", "ufa", "c_md", "c_c", "k_bulk", "k_mu"]
V = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
SNR = {"DiB_217": (25.0, 40.0), "HHS": (20.0, 35.0)}   # at b = 0, rough guesses for 3T

protocols = {"DiB_217": load_dib("DiB_217_lte_pte_ste")[3], "HHS": session_protocol("01", "01")}
g = torch.Generator().manual_seed(7)
d6, cls = simulate.sample_brain(V, g=g)
truth = simulate.dtd_params(d6)
st = scalars.scalars(truth)
names = simulate.BRAIN_CLASSES
out = {"V": V, "classes": {}}

print("true scalars by class (median):")
for c, n in enumerate(names):
    m = (cls == c).numpy()
    print(f"  {n:7s} " + "  ".join(f"{k} {np.nanmedian(st[k].numpy()[m]):.2f}" for k in KEYS))

ev = torch.linalg.eigvalsh(mat6(truth[:, 7:]))
rel = (ev / ev[:, -1:].clamp_min(1e-30)).numpy()
nsmall = (rel < 0.01).sum(1)
print("\n3. true covariance: # of 6 eigenvalues below 1% of the largest, median [IQR]")
for c, n in enumerate(names):
    m = (cls == c).numpy()
    print(f"  {n:7s} {np.median(nsmall[m]):.0f} [{np.percentile(nsmall[m], 25):.0f}-{np.percentile(nsmall[m], 75):.0f}]")
    out["classes"].setdefault(n, {})["cov_small_eigs_median"] = float(np.median(nsmall[m]))

for pname, prot in protocols.items():
    X = prot.design_matrix()
    clean = simulate.signal(d6, prot.b6)
    snr = torch.empty(V, dtype=torch.float64).uniform_(*SNR[pname], generator=g)
    noisy = simulate.rician(clean, 1.0 / snr, g)
    s_clean = scalars.scalars(fit.wls(X, clean))
    pw = fit.wls(X, noisy)
    pc, info = fit.constrained(X, noisy)
    s_w, s_c = scalars.scalars(pw), scalars.scalars(pc)
    print(f"\n=== {pname} (rank {prot.rank()}), SNR {SNR[pname]}: {100 * info.n_violating / V:.0f}% of voxels "
          f"violate under WLS; ADMM converged {info.converged_fraction:.3f}")
    print("1. truncation bias (noise-free WLS - truth), median by class:")
    for c, n in enumerate(names):
        m = (cls == c).numpy()
        print(f"  {n:7s} " + "  ".join(f"{k} {np.nanmedian((s_clean[k] - st[k]).numpy()[m]):+.3f}" for k in KEYS))
    print("2. noisy: median bias / MAE vs truth, WLS -> constrained:")
    for c, n in enumerate(names):
        m = (cls == c).numpy()
        cells = []
        for k in ("ufa", "c_md", "k_mu", "k_bulk"):
            dw, dc = (s_w[k] - st[k]).numpy()[m], (s_c[k] - st[k]).numpy()[m]
            cells.append(f"{k} {np.nanmedian(dw):+.3f}/{np.nanmedian(np.abs(dw)):.3f} -> {np.nanmedian(dc):+.3f}/{np.nanmedian(np.abs(dc)):.3f}")
            out["classes"][n][f"{pname}_{k}"] = {
                "trunc_bias": float(np.nanmedian((s_clean[k] - st[k]).numpy()[m])),
                "wls_bias": float(np.nanmedian(dw)), "wls_mae": float(np.nanmedian(np.abs(dw))),
                "con_bias": float(np.nanmedian(dc)), "con_mae": float(np.nanmedian(np.abs(dc)))}
        print(f"  {n:7s} " + " | ".join(cells))
json.dump(out, open("results/brain_prior_checks.json", "w"), indent=1)
