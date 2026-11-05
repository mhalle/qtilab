"""Phantom check: water, oil and hexagonal liquid crystal from the Szczepankiewicz-Hoge-Westin 2019 data.

Expected behavior (room temperature, no restriction at these length scales for water):
  water           isotropic, MD about 2.0-2.3 um^2/ms, FA ~ 0, uFA ~ 0, C_MD ~ 0
  oil             isotropic and slow; uFA ~ 0; may show size variance (several molecular species)
  liquid crystal  anisotropic domains in random orientations: uFA near 1, FA well below it, C_c low

b-tensors come from each dataset's md-dmri xps file (bt in s/m^2, order xx yy zz sqrt2*xy sqrt2*xz
sqrt2*yz), converted to qtilab's Mandel order and ms/um^2.
"""
import json
import os
import sys

import nibabel as nib
import numpy as np
import scipy.io as sio
import torch

sys.path.insert(0, ".")
from qtilab import fit, scalars  # noqa: E402
from qtilab.protocol import Protocol  # noqa: E402
from qtilab.tensors import mat3  # noqa: E402

ROOT = os.environ.get("QTILAB_DIB_DATA", "data/dib/DATA")
out = {}
for name, folder in (("water", "water"), ("oil", "oil"), ("liquid crystal", "hex")):
    d = os.path.join(ROOT, folder, "MD-dMRI")
    nii = [f for f in os.listdir(d) if f.endswith(".nii.gz")][0]
    xps = sio.loadmat(os.path.join(d, nii.replace(".nii.gz", "_xps.mat")), squeeze_me=True,
                      struct_as_record=False)["xps"]
    bt = np.asarray(xps.bt, float)[:, [0, 1, 2, 5, 4, 3]] * 1e-9               # -> Mandel, ms/um^2
    B = mat3(torch.as_tensor(bt))
    prot = Protocol(B)
    data = np.asarray(nib.load(os.path.join(d, nii)).dataobj, np.float32)
    b = prot.b.numpy()
    b0 = data[..., b < 0.05].mean(-1)
    m = b0 > 0.5 * np.percentile(b0[b0 > 0], 99)                                   # the bright phantom
    # erode by one voxel in-plane to avoid edge partial volume
    core = m.copy()
    for ax in (0, 1):
        core &= np.roll(m, 1, ax) & np.roll(m, -1, ax)
    S = torch.as_tensor(np.maximum(data[core].astype(np.float64), 1e-4))
    pc, info = fit.constrained(prot.design_matrix(), S)
    sc = scalars.scalars(pc)
    shapes = {round(float(x), 2) for x in np.asarray(xps.b_delta)}
    row = {"voxels": int(core.sum()), "volumes": prot.n, "b_max": float(b.max()), "b_delta_values": sorted(shapes),
           "rank": prot.rank()}
    print(f"\n{name}: {core.sum()} voxels, {prot.n} volumes, b_max {b.max():.1f}, b_delta {sorted(shapes)}, rank {row['rank']}")
    for k in ("md", "fa", "ufa", "c_c", "c_md", "k_bulk", "k_mu"):
        v = sc[k].numpy()
        q = np.nanpercentile(v, [25, 50, 75])
        row[k] = q.tolist()
        print(f"  {k:6s} median {q[1]:.3f}  (IQR {q[0]:.3f}-{q[2]:.3f})")
    out[name] = row
json.dump(out, open("results/phantoms_dib.json", "w"), indent=1)
