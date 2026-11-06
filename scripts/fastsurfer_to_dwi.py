"""FastSurfer (T1) region probabilities as partial-volume fractions on the diffusion grid.

The segmentation comes from haversack (github.com/mhalle/haversack), which runs FastSurfer 2.5.4 on
MPS and writes a ranked store (a rankfield of the output distribution) instead of labels:

    haversack segment data/hhs/sub-01/ses-01/anat/sub-01_ses-01_T1w.nii --task fastsurfer:asegdkt \\
        --device mps -o work/hhs_sub01_ses01_asegdkt.duckn.zip         # ~100 s; 7.4 MB store

Per conformed (0.94 mm) voxel: a softmax over the classes the store names (the winner at 0, the others
at minus their stored gap; classes past the clip are dropped). FastSurfer adds its three views' raw
logits (alpha 0.4 each), so this is a log-linear pool: a probability, but not a calibrated one.
Classes are summed into region groups, each voxel center is mapped into the diffusion grid through
scripts/t1_to_dwi.py's rigid transform, and the groups are box-averaged per diffusion voxel.

Writes QTILAB_WORK/hhs_sub{sub}_ses{ses}_fastsurfer_dwi.npz: frac (groups, X, Y, Z), names, count
(conformed voxels per diffusion voxel), covered (count >= 80% of the median).

Runs in haversack's environment, which has zarr and rankfield (qtilab's does not); qtilab is not
imported, so the file names below repeat qtilab/paths.py:

    ../haversack/.venv/bin/python scripts/fastsurfer_to_dwi.py [sub] [ses]     # ~5 s
"""
import os
import sys
from pathlib import Path

import numpy as np
import zarr
from rankfield import levels
from zarr.storage import ZipStore

sub = sys.argv[1] if len(sys.argv) > 1 else "01"
ses = sys.argv[2] if len(sys.argv) > 2 else "01"
WORK = Path(os.environ.get("QTILAB_WORK", "work"))
store_path = WORK / f"hhs_sub{sub}_ses{ses}_asegdkt.duckn.zip"
reg_path = WORK / f"hhs_sub{sub}_ses{ses}_t1_to_dwi.npz"
out_path = WORK / f"hhs_sub{sub}_ses{ses}_fastsurfer_dwi.npz"

GROUPS = {                                   # FreeSurfer aseg / aparc ids
    "background": [0],
    "cortex": None,                          # every id >= 1000 (DKT parcels)
    "cerebral_wm": [2, 41, 77],
    "thalamus": [10, 49], "caudate": [11, 50], "putamen": [12, 51], "pallidum": [13, 52],
    "hippo_amyg": [17, 53, 18, 54], "accumbens_vdc": [26, 58, 28, 60],
    "cerebellum_ctx": [8, 47], "cerebellum_wm": [7, 46], "brainstem": [16],
    "csf_ventricles": [4, 5, 43, 44, 14, 15, 24, 31, 63],
}

g = zarr.open_group(ZipStore(store_path, mode="r"), mode="r")["parts/0"]
meta = g.attrs["duckn"]["extensions"]["ranked"]
chan_id = np.array(meta["labels"])                         # channel -> aseg/aparc id
names = list(GROUPS)
id2grp = {}
for gi, n in enumerate(names):
    for i in (chan_id[chan_id >= 1000] if GROUPS[n] is None else GROUPS[n]):
        id2grp[int(i)] = gi
missing = set(chan_id.tolist()) ^ set(id2grp)
assert not missing, f"unmapped or extra ids: {missing}"
chan2grp = np.array([id2grp[int(i)] for i in chan_id], np.int16)

ranks = np.asarray(g["ranks"])                             # (N, Z, Y, X): class + 1, 0 = none
support = np.asarray(g["support"])                         # (N - 1, Z, Y, X)
lut = np.asarray(levels(meta), np.float32)                 # support byte -> logits behind the winner
N, shape = ranks.shape[0], ranks.shape[1:]
w = np.empty((N,) + shape, np.float32)
w[0] = 1.0
for j in range(1, N):
    w[j] = np.where(ranks[j] > 0, np.exp(-lut[support[j - 1]]), 0.0)
w /= w.sum(0, keepdims=True)
grp = np.where(ranks > 0, chan2grp[np.maximum(ranks.astype(np.int32) - 1, 0)], -1)
del support

# conformed index -> store LPS world -> RAS (T1 world) -> DWI world -> DWI voxel, as one affine
ad = g["ranks"].attrs["duckn"]
assert ad["space"] == "left-posterior-superior"
D = np.array([ax["space_direction"] for ax in ad["axes"][1:]])        # one row per spatial array axis
lps2ras = np.diag([-1.0, -1.0, 1.0])
idx2ras = np.eye(4)
idx2ras[:3, :3], idx2ras[:3, 3] = lps2ras @ D.T, lps2ras @ np.array(ad["space_origin"])
r = np.load(reg_path)
dshape = tuple(int(n) for n in r["dwi_shape"])
M = np.linalg.inv(r["dwi_aff"]) @ np.linalg.inv(r["reg"]) @ idx2ras
vox = np.floor(M[:3, :3] @ np.indices(shape, dtype=np.float32).reshape(3, -1) + M[:3, 3:4] + 0.5)
vox = vox.astype(np.int64)
inside = np.all((vox >= 0) & (vox < np.array(dshape)[:, None]), 0)
flat = np.ravel_multi_index(tuple(vox[:, inside]), dshape)
del vox
nd = int(np.prod(dshape))
count = np.bincount(flat, minlength=nd).astype(np.float32)

frac = np.zeros((len(names), nd), np.float32)
wf, gf = w.reshape(N, -1)[:, inside], grp.reshape(N, -1)[:, inside]
for gi in range(len(names)):
    frac[gi] = np.bincount(flat, weights=np.where(gf == gi, wf, 0.0).sum(0), minlength=nd)
frac = (frac / np.maximum(count, 1)).reshape((len(names),) + dshape)
count = count.reshape(dshape)
covered = count >= 0.8 * np.median(count[count > 0])
np.savez_compressed(out_path, frac=frac, names=np.array(names), count=count, covered=covered)

vox_mm3 = abs(np.linalg.det(r["dwi_aff"][:3, :3]))
print(f"{store_path.name}: {len(chan_id)} classes, conformed grid {shape}; "
      f"median {np.median(count[count > 0]):.0f} conformed voxels per diffusion voxel")
for gi, n in enumerate(names[1:], 1):
    print(f"  {n:15s} {frac[gi].sum() * vox_mm3 / 1e3:7.1f} cm3")
print("wrote", out_path)
