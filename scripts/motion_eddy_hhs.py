"""Within-run motion and eddy-current correction of a run-aligned HHS session with tractline (CPU or GPU).

tractline.motion_eddy follows FSL eddy: each volume is registered to a leave-one-out prediction from
the other volumes of its shell (rigid motion + an eddy polynomial along phase encoding, b-vectors
rotated with the head). It groups shells by b-value alone, so for tensor-valued data:

  * LTE and STE volumes share b-values here (100, 500, 1000, 1500): they are corrected in two
    separate calls, each with every b = 0 volume (b = 0 has no encoding shape) as the common
    location reference, so both calls end in the same frame;
  * STE volumes have no direction: they get one common nominal direction, so the shell predictor
    reduces to (nearly) the mean of the other repeats;
  * each HHS run is (almost) a whole shell, and a leave-one-out predictor cannot see a shell's
    common position: between-run alignment stays with qtilab.hhs's mutual-information step (the input
    here), and tractline corrects within runs.

No reversed phase-encoding series exists for HHS, so no susceptibility field (eddy without topup).
The b = 0 volumes are taken from the LTE call. Writes paths.aligned_session(sub, ses + "ec").

    python scripts/motion_eddy_hhs.py 01 01 [--device cpu] [--steps 40]
"""
import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from qtilab import paths  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("sub", nargs="?", default="01"); ap.add_argument("ses", nargs="?", default="01")
ap.add_argument("--device", default="cpu")
ap.add_argument("--steps", type=int, default=40)
ap.add_argument("--model", default="quadratic")
args = ap.parse_args()
from tractline import motion_eddy as me  # noqa: E402

a = np.load(paths.aligned_session(args.sub, args.ses))
c = np.load(paths.classical_fit(args.sub, args.ses))
data, affine = a["data"].astype(np.float32), a["affine"]
bvals, bvecs = a["bvals"].astype(float), a["bvecs"].astype(float).copy()
shapes = np.array([str(s) for s in a["shapes"]])
isb0 = bvals < 50
lte = (shapes == "LTE") & ~isb0
ste = (shapes == "STE") & ~isb0
mask = c["mask"]
PE_AXIS = 1                                       # PhaseEncodingDirection j- in every HHS sidecar
out = data.copy()
new_bvecs = bvecs.copy()
qc = {}
opts = dict(steps=args.steps, model=args.model, check_memory=False)
for name, sel in (("LTE", lte), ("STE", ste)):
    idx = np.flatnonzero(sel | isb0)              # this call's volumes, b0s included, in series order
    bv = bvecs[:, idx].copy()
    if name == "STE":
        bv[:, ~isb0[idx]] = np.array([[0.0], [0.0], [1.0]])     # one nominal direction for direction-less volumes
    t = time.perf_counter()
    r = me.estimate(data[..., idx], bvals[idx], bv, affine, PE_AXIS, mask=mask, options=opts, device=args.device,
                    progress=lambda rec: print(f"  {name} step {rec.get('step', '?')}", flush=True) if isinstance(rec, dict) and rec.get('step', 0) % 10 == 0 else None)
    corr, valid = me.apply_volumes(data[..., idx], r, device=args.device)
    dt = time.perf_counter() - t
    corr = np.asarray(corr)
    keep = sel[idx] | (isb0[idx] & (name == "LTE"))                # STE call contributes only its DW volumes
    out[..., idx[keep]] = corr[..., keep]
    if name == "LTE":
        new_bvecs[:, idx[keep]] = np.asarray(r.bvec)[:, keep]
    rec = me.record(r)
    qc[name] = {"seconds": dt, "series_index": idx.tolist(), "excluded": {str(k): str(v) for k, v in r.excluded.items()},
                "record": rec}
    print(f"{name}: {len(idx)} volumes corrected in {dt:.0f} s; excluded {qc[name]['excluded']}", flush=True)

np.savez_compressed(paths.aligned_session(args.sub, args.ses + "ec"), data=out, affine=affine, bvals=bvals,
                    bvecs=new_bvecs, shapes=shapes)
json.dump(qc, open(f"results/motion_eddy_sub{args.sub}_ses{args.ses}.json", "w"), indent=1, default=str)
print("wrote", paths.aligned_session(args.sub, args.ses + "ec"))
