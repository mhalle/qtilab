"""Check intensity scaling between Hormone Health runs, across all sessions.

Two model-light comparisons per session, using a brain mask from run 1's b = 0:

* run 5 (STE, b = 0) vs run 1 (LTE, b = 0): identical contrast, so the ratio should be 1.
* run 6 (STE, b = 100 and 500) vs run 1 (LTE, b = 100 and 500), direction-averaged: to first order
  in b both equal exp(-b MD); the second-order difference (from microscopic anisotropy) is a
  small, predictable excess of LTE over STE that grows like b^2.

Ratios are medians over brain voxels of per-voxel ratios. No registration is applied, so small
head motion adds scatter, not bias.
"""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from dipy.segment.mask import median_otsu  # noqa: E402

from qtilab import hhs  # noqa: E402
import nibabel as nib  # noqa: E402

rows = []
for sub_dir in sorted(hhs.HHS_ROOT.glob("sub-*")):
    sub = sub_dir.name[4:]
    for ses_dir in sorted(sub_dir.glob("ses-*")):
        ses = ses_dir.name[4:]
        runs = {r["run"]: r for r in hhs.session_runs(sub, ses)}
        r1, r5, r6 = runs[1], runs[5], runs[6]
        d1 = np.asarray(nib.load(r1["path"]).dataobj, np.float32)
        d5 = np.asarray(nib.load(r5["path"]).dataobj, np.float32)
        d6 = np.asarray(nib.load(r6["path"]).dataobj, np.float32)
        b1, b6 = r1["bval"], r6["bval"]
        b0_1 = d1[..., b1 < 50].mean(-1)
        _, m = median_otsu(b0_1, median_radius=2, numpass=1)
        m &= b0_1 > np.percentile(b0_1[m], 10)
        row = {"sub": sub, "ses": ses, "b0_run5_over_run1": float(np.median(d5.mean(-1)[m] / b0_1[m]))}
        for b in (100, 500):
            lte = d1[..., np.abs(b1 - b) < 1].mean(-1)
            ste = d6[..., np.abs(b6 - b) < 1].mean(-1)
            row[f"lte_over_ste_b{b}"] = float(np.median(lte[m] / np.maximum(ste[m], 1e-3)))
        rows.append(row)
        print(row, flush=True)

json.dump(rows, open("results/hhs_run_scaling.json", "w"), indent=1)
a = {k: np.array([r[k] for r in rows]) for k in rows[0] if k not in ("sub", "ses")}
print("\nacross", len(rows), "sessions (median [5th-95th percentile]):")
for k, v in a.items():
    print(f"  {k:22s} {np.median(v):.3f} [{np.percentile(v, 5):.3f}-{np.percentile(v, 95):.3f}]")
