"""Prepare one Hormone Health session for the analysis scripts.

Writes (under QTILAB_WORK, default ./work):
  hhs_sub{sub}_ses{ses}_aligned.npz   data, affine, bvals, bvecs, shapes (runs rigidly aligned)
  hhs_sub{sub}_ses{ses}_qti.npz       brain mask and constrained (QTI+ dc) scalars; voxels whose fit
                                      collapsed to <D> = 0 (scalars.collapsed) are dropped from the mask

    python scripts/prepare_session.py [sub] [ses]          # default 01 01, about 6 min on 2 CPU cores
"""
import sys
import time

import numpy as np
import torch
from dipy.segment.mask import median_otsu

sys.path.insert(0, ".")
from qtilab import fit, hhs, paths, scalars  # noqa: E402

sub = sys.argv[1] if len(sys.argv) > 1 else "01"
ses = sys.argv[2] if len(sys.argv) > 2 else "01"
paths.WORK.mkdir(parents=True, exist_ok=True)
t = time.perf_counter()
s = hhs.load_session(sub, ses, align=True)
np.savez_compressed(paths.aligned_session(sub, ses), data=s["data"], affine=s["affine"], bvals=s["bvals"],
                    bvecs=s["bvecs"], shapes=np.array(s["shapes"]))
print(f"aligned in {time.perf_counter() - t:.0f} s; motion: " +
      ", ".join(f"run {k + 1} {m['rotation_deg']:.2f} deg" for k, m in enumerate(s["motion"]) if m), flush=True)
b0 = s["data"][..., s["bvals"] < 50].mean(-1)
_, mask = median_otsu(b0, median_radius=2, numpass=1)
S = torch.as_tensor(np.maximum(s["data"][mask].astype(np.float64), 1e-4))
t = time.perf_counter()
pc, info = fit.constrained(s["protocol"].design_matrix(), S)
sc = scalars.scalars(pc)
n_fit = int(mask.sum())
keep = ~scalars.collapsed(sc).numpy()
mask[mask] = keep                         # drop collapsed fits (MD = 0: every ratio scalar meaningless)
np.savez_compressed(paths.classical_fit(sub, ses), mask=mask,
                    **{k: sc[k].numpy()[keep] for k in ("md", "fa", "ufa", "c_md", "c_c", "k_bulk", "k_mu")})
print(f"fit {n_fit} voxels in {time.perf_counter() - t:.0f} s; {100 * info.n_violating / n_fit:.0f}% constrained; "
      f"{n_fit - int(keep.sum())} collapsed (MD = 0) dropped from the mask")
