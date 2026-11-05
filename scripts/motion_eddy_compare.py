"""Did motion/eddy correction remove structured error? Held-out prediction residuals before vs after, in
absolute signal units (fraction of b0, x 1000), so the comparison does not depend on a noise model.

Resampling smooths thermal noise (fitted thermal SNR rises after correction), so noise-normalized errors
(heldout_hhs.py, heldout_diagnostics.py) are not comparable across the two series. Here both series are
predicted by the same k-NN over the same simulated bank (brain prior v2), on the same voxels (mask and
regions from the uncorrected classical fit), and errors are reported as signal fractions:

  rms       median per-voxel RMS residual of the held-out run
  bias      median residual
  vol_sd    sd across volumes of the region-median residual (volume-level offsets; run 4 only, LTE)
  noise     a noise floor for reference: median within-voxel sd across the STE b = 1500 repeats (run 8)

    python scripts/motion_eddy_compare.py [--sub 01 --ses 01 --series ec]
"""
import argparse
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from qtilab import paths, simulate  # noqa: E402
from qtilab.protocol import Protocol  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sub", default="01"); ap.add_argument("--ses", default="01")
ap.add_argument("--series", default="ec")
ap.add_argument("--bank", type=int, default=100_000); ap.add_argument("--k", type=int, default=20)
ap.add_argument("--slices", type=int, nargs="*", default=None, help="axial slices (default: fullest +/- 3)")
args = ap.parse_args()
torch.set_num_threads(2)

c = np.load(paths.classical_fit(args.sub, args.ses))
full = c["mask"]
z0 = int(np.argmax(full.sum((0, 1))[10:28])) + 10
zs = args.slices if args.slices else [z0 - 3, z0, z0 + 3]
slab = np.zeros_like(full); slab[:, :, zs] = True
mask = full & slab
fa, md = c["fa"][slab[full]], c["md"][slab[full]]
regions = {"WM-like": (fa > 0.4) & (md < 1.2), "GM-like": (fa < 0.2) & (md < 1.2), "CSF-ish": md > 2.0}
run = np.concatenate([[i] * n for i, n in enumerate([13, 10, 16, 30, 2, 12, 9, 15], 1)])

series = {}
for name, suf in (("raw", ""), ("corrected", args.series)):
    a = np.load(paths.aligned_session(args.sub, args.ses + suf))
    bvals = a["bvals"]
    S = np.maximum(a["data"][mask].astype(np.float64), 1e-4)
    series[name] = (S / S[:, bvals < 50].mean(1, keepdims=True), a)

a0 = series["raw"][1]
shapes = [str(s) for s in a0["shapes"]]
g = torch.Generator().manual_seed(21)
prot_raw = Protocol.from_table(a0["bvals"], a0["bvecs"], shapes)
bank_d6 = []
for i in range(0, args.bank, 20000):
    d6, _ = simulate.sample_brain(min(20000, args.bank - i), M=48, g=g, version=2)
    bank_d6.append(d6)
bank_d6 = torch.cat(bank_d6)


def predict(S, prot, train, test):
    bank = simulate.signal(bank_d6, prot.b6).float()          # per series: the corrected one has rotated b-vectors
    pred = np.zeros((S.shape[0], test.sum()))
    for i in range(0, S.shape[0], 256):
        d = torch.cdist(torch.as_tensor(S[i:i + 256, train], dtype=torch.float32), bank[:, train])
        idx = d.topk(args.k, largest=False).indices
        pred[i:i + 256] = bank[:, test][idx].double().mean(1).numpy()
    return pred


out = {"sub": args.sub, "ses": args.ses, "slices": zs, "units": "signal fraction of b0 x 1000"}
for name, (S, a) in series.items():
    prot = Protocol.from_table(a["bvals"], a["bvecs"], shapes)
    ste = run == 8
    noise_floor = np.median(S[:, ste].std(1, ddof=1))
    row = {"noise_ste1500_sd": 1000 * float(noise_floor)}
    for held in (4, 8):
        test = run == held
        r = S[:, test] - predict(S, prot, ~test, test)
        e = np.sqrt(np.mean(r ** 2, 1))
        for reg, m in regions.items():
            row[f"run{held}_{reg}"] = {"rms": 1000 * float(np.median(e[m])), "bias": 1000 * float(np.median(r[m]))}
            if held == 4:
                row[f"run{held}_{reg}"]["vol_sd"] = 1000 * float(np.median(r[m], 0).std())
    out[name] = row
    print(f"{name:9s} noise floor (STE b1500 repeat sd) {row['noise_ste1500_sd']:.1f}")
    for held in (4, 8):
        print(f"   run {held}: " + "   ".join(f"{reg} rms {row[f'run{held}_{reg}']['rms']:.1f} bias {row[f'run{held}_{reg}']['bias']:+.1f}"
                                      + (f" vol_sd {row[f'run{held}_{reg}']['vol_sd']:.1f}" if held == 4 else "")
                                      for reg in regions))
json.dump(out, open(f"results/motion_eddy_compare_sub{args.sub}_ses{args.ses}.json", "w"), indent=1)
