"""Prepare one Cardiff b-tensor subject (qtilab.cardiff) and check the data.

Writes QTILAB_WORK/cardiff_{sub}_aligned.npz (data, affine, bvals, bvecs, shapes, run_index, mask) and
results/cardiff_{sub}_checks.json:

* b-value check: in CSF (b0-bright voxels with LTE b = 1 attenuation near free water) an isotropic Gaussian
  compartment attenuates the same at a given b whatever the b-tensor shape, so -ln(S/S0)/b at b = 1 should
  agree across LTE, PTE and STE if the nominal b-values are right;
* noise: sigma from the 31 b0s (pooled within-run SD), b0 SNR in tissue, and each shell's powder-averaged
  signal against the Rician floor (sigma sqrt(pi/2));
* classical constrained QTI (CPU, float64) at b_max = 2, 3, 4.5, 6: regional medians in WM-like
  (FA > 0.4) and GM-like (FA < 0.2, MD < 1.2) tissue, the apparent values' drift with the b-range.

    python scripts/prepare_cardiff.py [data1|data2]
"""
import json
import sys
import time

import numpy as np
import torch
from dipy.segment.mask import median_otsu

sys.path.insert(0, ".")
from qtilab import cardiff, fit, paths, scalars  # noqa: E402

sub = sys.argv[1] if len(sys.argv) > 1 else "data1"
paths.WORK.mkdir(parents=True, exist_ok=True)
t = time.perf_counter()
s = cardiff.load_subject(sub, align=True)
d, b, shapes, run = s["data"], s["bvals"] / 1000, np.array(s["shapes"]), s["run_index"]
b0 = d[..., b < 0.05].mean(-1)
_, mask = median_otsu(b0, median_radius=2, numpass=1)
out = paths.WORK / f"cardiff_{sub}_aligned.npz"
np.savez(out, data=d, affine=s["affine"], bvals=s["bvals"], bvecs=s["bvecs"], shapes=shapes, run_index=run, mask=mask)
print(f"{sub}: aligned in {time.perf_counter() - t:.0f} s ({', '.join(f'{m['rotation_deg']:.2f} deg' for m in s['motion'] if m)}); "
      f"{mask.sum()} mask voxels -> {out}", flush=True)
res = {"sub": sub, "motion": s["motion"], "n_mask": int(mask.sum())}

S = d[mask].astype(np.float64)
# per-run S0 (runs can differ in scaling)
s0 = np.stack([S[:, (run == k) & (b < 0.05)].mean(1) for k in range(3)], 1)            # (V, 3)
res["run_s0_ratio_median"] = (np.median(s0 / s0[:, :1], 0)).round(4).tolist()
Sn = S / s0[:, run]
# noise from b0 repeats, pooled within run
dev = np.concatenate([S[:, (run == k) & (b < 0.05)] - s0[:, k:k + 1] for k in range(3)], 1)
nb0 = sum(((run == k) & (b < 0.05)).sum() for k in range(3))
sigma = np.sqrt((dev ** 2).sum(1) / (nb0 - 3))

lte1 = (shapes == "LTE") & (np.abs(b - 1) < 0.06)
adc_l = -np.log(np.clip(Sn[:, lte1].mean(1), 1e-4, None))
csf = (adc_l > 2.4) & (s0[:, 0] > np.percentile(s0[:, 0], 75))
tissue = (adc_l > 0.5) & (adc_l < 1.0)
res["b_check_csf"] = {"n": int(csf.sum())}
print(f"b check: -ln(S/S0)/b at b = 1, CSF ({csf.sum()} voxels) vs tissue ({tissue.sum()})")
for sh in ("LTE", "PTE", "STE"):
    sel = (shapes == sh) & (np.abs(b - 1) < 0.06)
    adc = -np.log(np.clip(Sn[:, sel].mean(1), 1e-4, None)) / 1.0
    res["b_check_csf"][sh] = round(float(np.median(adc[csf])), 3)
    res.setdefault("b_check_tissue", {})[sh] = round(float(np.median(adc[tissue])), 3)
    print(f"  {sh}: CSF {np.median(adc[csf]):.3f}  tissue {np.median(adc[tissue]):.3f}")
ste02 = (shapes == "STE") & (np.abs(b - 0.2) < 0.03)
res["b_check_csf"]["STE_b0.2"] = round(float(np.median(-np.log(np.clip(Sn[:, ste02].mean(1), 1e-4, None))[csf] / 0.2)), 3)
print(f"  STE b = 0.2: CSF {res['b_check_csf']['STE_b0.2']:.3f}")

snr = s0[:, 0] / sigma
res["b0_snr_tissue_median"] = round(float(np.median(snr[tissue])), 1)
floor = np.median((sigma * np.sqrt(np.pi / 2) / s0[:, 0])[tissue])
res["rician_floor_rel"] = round(float(floor), 4)
print(f"b0 SNR in tissue: median {np.median(snr[tissue]):.0f}; Rician floor {floor:.4f} of S0")
res["shell_signal_tissue"] = {}
for sh in ("LTE", "PTE", "STE"):
    row = {}
    for bb in sorted(set(np.round(b[(shapes == sh) & (b > 0.05)], 1))):
        sel = (shapes == sh) & (np.abs(b - bb) < 0.06)
        row[f"{bb:g}"] = round(float(np.median(Sn[tissue][:, sel].mean(1))), 4)
    res["shell_signal_tissue"][sh] = row
    print(f"  {sh} " + " ".join(f"{k}:{v:.3f}" for k, v in row.items()) + f"   (floor {floor:.3f})")

prot = s["protocol"]
res["qti_vs_bmax"] = {}
print("classical constrained QTI vs b_max (medians): FA, µFA, MD, C_MD, C_c")
for bmax in (2.0, 3.0, 4.5, 6.0):
    keep = b <= bmax + 0.06
    X = prot.subset(torch.as_tensor(keep)).design_matrix()
    t = time.perf_counter()
    pc, info = fit.constrained(X, torch.as_tensor(np.maximum(S[:, keep], 1e-4)))
    sc = {k: v.numpy() for k, v in scalars.scalars(pc).items()}
    ok = ~scalars.collapsed(scalars.scalars(pc)).numpy()
    wm = ok & (sc["fa"] > 0.4)
    gm = ok & (sc["fa"] < 0.2) & (sc["md"] < 1.2) & (sc["md"] > 0.5)
    row = {"n_vol": int(keep.sum()), "rank": int(torch.linalg.matrix_rank(X)),
           **{f"{r}_{k}": round(float(np.median(sc[k][m])), 3) for r, m in (("wm", wm), ("gm", gm))
              for k in ("fa", "ufa", "md", "c_md", "c_c")}}
    res["qti_vs_bmax"][f"{bmax:g}"] = row
    print(f"  b <= {bmax:g} ({keep.sum()} vols, rank {row['rank']}, {time.perf_counter() - t:.0f} s): "
          f"WM {row['wm_fa']:.3f} {row['wm_ufa']:.3f} {row['wm_md']:.3f} {row['wm_c_md']:.3f} {row['wm_c_c']:.3f} | "
          f"GM {row['gm_fa']:.3f} {row['gm_ufa']:.3f} {row['gm_md']:.3f} {row['gm_c_md']:.3f} {row['gm_c_c']:.3f}", flush=True)
json.dump(res, open(f"results/cardiff_{sub}_checks.json", "w"), indent=1)
print(f"-> results/cardiff_{sub}_checks.json")
