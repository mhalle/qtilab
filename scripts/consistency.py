"""Consistency checks.

A. Same voxels, two protocols: DiB_217 vs DiB_70 (both subsampled from one acquisition, so their
   noise is partly shared; this tests protocol dependence, not scan-rescan reproducibility).
B. One subject, three Hormone Health sessions (sub-01): classical QTI regional medians and the v2
   network's regional medians on a mid-brain slice of each session. Regions are defined per session
   (FA and free-water thresholds), so slice placement differences matter less than voxelwise
   comparison would.
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from qtilab import fit, posterior, scalars  # noqa: E402
from qtilab.data import load_dib  # noqa: E402
from qtilab import hhs  # noqa: E402

torch.set_num_threads(2)
out = {}

# ---- A. DiB 217 vs 70, same voxels (every 3rd brain voxel to save CPU)
d217, mask, _, p217, *_ = load_dib("DiB_217_lte_pte_ste")
d70, _, _, p70, *_ = load_dib("DiB_70_lte_pte_ste")
idx = np.flatnonzero(mask.ravel())[::3]
S217 = torch.as_tensor(np.maximum(d217.reshape(-1, d217.shape[-1])[idx], 1e-4).astype(np.float64))
S70 = torch.as_tensor(np.maximum(d70.reshape(-1, d70.shape[-1])[idx], 1e-4).astype(np.float64))
a = scalars.scalars(fit.constrained(p217.design_matrix(), S217)[0])
b = scalars.scalars(fit.constrained(p70.design_matrix(), S70)[0])
print(f"A. DiB_217 vs DiB_70, {len(idx)} voxels, constrained fit: median of (70 - 217), median |diff|, correlation")
out["dib_70_vs_217"] = {}
for k in ("md", "fa", "ufa", "c_md", "c_c", "k_bulk", "k_mu"):
    x, y = a[k].numpy(), b[k].numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    r = np.corrcoef(x[ok], y[ok])[0, 1]
    out["dib_70_vs_217"][k] = {"median_diff": float(np.median(y[ok] - x[ok])), "median_abs": float(np.median(np.abs(y[ok] - x[ok]))), "r": float(r)}
    print(f"   {k:6s} {np.median(y[ok] - x[ok]):+.3f}   {np.median(np.abs(y[ok] - x[ok])):.3f}   r = {r:.2f}")

# ---- B. three sessions of sub-01
net = posterior.PosteriorNet()
net.load_state_dict(torch.load("results/posterior_smoke_hhs_v2.pt"))
net.eval()
from dipy.segment.mask import median_otsu  # noqa: E402
rows = []
for ses in ("01", "02", "03"):
    s = hhs.load_session("01", ses, align=True)
    b0 = s["data"][..., s["bvals"] < 50].mean(-1)
    _, m = median_otsu(b0, median_radius=2, numpass=1)
    X = s["protocol"].design_matrix()
    S = torch.as_tensor(np.maximum(s["data"][m], 1e-4).astype(np.float64))
    sc = scalars.scalars(fit.wls(X, S))
    fa, md = sc["fa"].numpy(), sc["md"].numpy()
    # network on the slice with the most brain voxels near the middle
    zc = int(np.argmax(m.sum((0, 1))[10:28])) + 10
    slab = np.zeros_like(m); slab[:, :, zc] = True
    ms = m & slab
    sel = slab[m]
    Ss = torch.as_tensor(np.maximum(s["data"][ms], 1e-4).astype(np.float64))
    x = posterior.features(Ss, s["protocol"].b6, s["protocol"].b, s["protocol"].b_delta()).float()
    pm = net.sample(x, n=50).mean(0).numpy()
    T = posterior.TARGETS
    fw = pm[:, T.index("f_fw")]
    fa_s = fa[sel]
    row = {"session": ses, "slice": zc}
    wm_all, gm_all = (fa > 0.4) & (md < 1.2), (fa < 0.2) & (md < 1.2)
    row["classical_wm_ufa"] = float(np.median(sc["ufa"].numpy()[wm_all]))
    row["classical_gm_ufa"] = float(np.median(sc["ufa"].numpy()[gm_all]))
    row["classical_wm_md"] = float(np.median(md[wm_all]))
    wm, gm = (fa_s > 0.4) & (fw < 0.2), (fa_s < 0.2) & (fw < 0.2)
    for reg, mm in (("wm", wm), ("gm", gm)):
        for k in ("ufa_t", "md_t", "f_fw", "c_md_t"):
            row[f"net_{reg}_{k}"] = float(np.median(pm[mm, T.index(k)]))
    rows.append(row)
    print(f"B. ses-{ses} (slice {zc}): " + ", ".join(f"{k} {v:.3f}" for k, v in row.items() if k not in ("session", "slice")), flush=True)
out["hhs_sub01_sessions"] = rows
keys = [k for k in rows[0] if k not in ("session", "slice")]
print("\nacross the three sessions: mean, max - min")
for k in keys:
    v = np.array([r[k] for r in rows])
    print(f"   {k:20s} {v.mean():.3f}   {v.max() - v.min():.3f}")
json.dump(out, open("results/consistency.json", "w"), indent=1)
