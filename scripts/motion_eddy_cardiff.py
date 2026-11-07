"""Within-run motion and eddy-current correction of a run-aligned Cardiff subject with tractline.

Same scheme as scripts/motion_eddy_hhs.py: one tractline.motion_eddy call per encoding shape (LTE, PTE,
STE), each with every b = 0 volume as the common location reference, so all three end in one frame.
PTE volumes are predicted from their plane normals (the bvecs), STE volumes get one nominal direction.
Between-run alignment stays with qtilab.cardiff's mutual-information step (the input here).

Only volumes with b <= --bmax (default 6 ms/µm²) are corrected and kept: above it the PTE and STE signal is
within a few times the noise SD and registration would follow noise. Phase encoding is along the second
voxel axis (A-P; the dataset's reversed b0 is P-A), with no susceptibility field (the reversed b0 is unused).
Writes QTILAB_WORK/cardiff_{sub}ec_aligned.npz and results/motion_eddy_cardiff_{sub}.json.

    python scripts/motion_eddy_cardiff.py data1 [--device mps] [--bmax 6]
"""
import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from qtilab import paths  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("sub", nargs="?", default="data1")
ap.add_argument("--device", default="mps")
ap.add_argument("--steps", type=int, default=40)
ap.add_argument("--bmax", type=float, default=6.0)
args = ap.parse_args()
from tractline import motion_eddy as me  # noqa: E402

a = np.load(paths.WORK / f"cardiff_{args.sub}_aligned.npz")
keep_all = a["bvals"] <= args.bmax * 1000 + 60
data, affine = a["data"][..., keep_all].astype(np.float32), a["affine"]
bvals, bvecs = a["bvals"][keep_all].astype(float), a["bvecs"][:, keep_all].astype(float).copy()
shapes, run_index, mask = a["shapes"][keep_all], a["run_index"][keep_all], a["mask"]
isb0 = bvals < 50
PE_AXIS = 1
out, new_bvecs, qc = data.copy(), bvecs.copy(), {}
opts = dict(steps=args.steps, check_memory=False)
for name in ("LTE", "PTE", "STE"):
    sel = (shapes == name) & ~isb0
    idx = np.flatnonzero(sel | isb0)
    bv = bvecs[:, idx].copy()
    if name == "STE":
        bv[:, ~isb0[idx]] = np.array([[0.0], [0.0], [1.0]])
    t = time.perf_counter()
    r = me.estimate(data[..., idx], bvals[idx], bv, affine, PE_AXIS, mask=mask, options=opts, device=args.device)
    corr, valid = me.apply_volumes(data[..., idx], r, device=args.device)
    corr = np.asarray(corr)
    dt = time.perf_counter() - t
    keep = sel[idx] | (isb0[idx] & (name == "LTE"))       # b0s from the LTE call
    out[..., idx[keep]] = corr[..., keep]
    if name != "STE":
        new_bvecs[:, idx[keep]] = np.asarray(r.bvec)[:, keep]
    qc[name] = {"seconds": dt, "series_index": idx.tolist(),
                "excluded": {str(k): str(v) for k, v in r.excluded.items()}, "record": me.record(r)}
    print(f"{name}: {len(idx)} volumes corrected in {dt:.0f} s; excluded {qc[name]['excluded']}", flush=True)

dst = paths.WORK / f"cardiff_{args.sub}ec_aligned.npz"
np.savez(dst, data=out, affine=affine, bvals=bvals, bvecs=new_bvecs, shapes=shapes, run_index=run_index, mask=mask)
json.dump(qc, open(f"results/motion_eddy_cardiff_{args.sub}.json", "w"), indent=1, default=str)
print("wrote", dst)
