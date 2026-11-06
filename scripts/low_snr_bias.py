"""Is a network's anisotropy biased at low SNR, and how much real tissue is there?

* Real data: b0 SNR per voxel (mean b0 / SD of the 15 repeated STE b = 1500 volumes, which also carries
  physiological noise and the per-volume offsets) by FastSurfer region (voxels >= 80% one region).
* Simulation: bias, RMSE and 90% coverage of the posterior means of FA_t and µFA_t in WM and GM, by SNR band,
  on prior v4 with v5's forward model (exact free water, r = 1.5, alpha, pulsation, gain jitter).

    python scripts/low_snr_bias.py [--tags _v4longfw _v5 _v5_snr20] [--n 12000]
-> results/low_snr_bias_s{sub}{ses}.json
"""
import argparse
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from qtilab import paths, posterior, simulate  # noqa: E402
from qtilab.hhs import session_protocol  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tags", nargs="+", default=["_v4longfw", "_v5", "_v5_snr20"])
ap.add_argument("--n", type=int, default=12000)
ap.add_argument("--sub", default="01")
ap.add_argument("--ses", default="01")
args = ap.parse_args()
torch.set_num_threads(8)
T = posterior.TARGETS
REGIONS = ["cortex", "caudate", "putamen", "thalamus", "pallidum", "cerebral_wm", "cerebellum_wm", "brainstem"]
res = {"tags": args.tags, "sub": args.sub, "ses": args.ses, "real_snr": {}, "simulation": {}}

a = np.load(paths.aligned_session(args.sub, args.ses))
q = np.load(paths.classical_fit(args.sub, args.ses))
mask = q["mask"]
S = a["data"][mask].astype(float)
b, sh = a["bvals"], a["shapes"]
ref = (sh == "STE") & (np.abs(b - 1500) < 1)
snr = S[:, b < 50].mean(1) / S[:, ref].std(1, ddof=1)
f = np.load(paths.fastsurfer_fractions(args.sub, args.ses))
names = [str(n) for n in f["names"]]
cov_fs = f["covered"][mask]
print(f"sub-{args.sub} ses-{args.ses}: b0 SNR (mean b0 / SD of {ref.sum()} STE b1500 repeats)")
for r in REGIONS:
    s = cov_fs & (f["frac"][names.index(r)][mask] >= 0.8)
    x = snr[s]
    res["real_snr"][r] = {"n": int(s.sum()), "median": round(float(np.median(x)), 1),
                          "p10": round(float(np.percentile(x, 10)), 1), "pct_below_20": round(float((x < 20).mean() * 100), 1)}
    print(f"  {r:14s} n={s.sum():5d}  median {np.median(x):5.1f}  p10 {np.percentile(x, 10):5.1f}  "
          f"below 20: {(x < 20).mean():5.1%}")

prot = session_protocol(args.sub, args.ses)
nets = {}
for tag in args.tags:
    nets[tag] = posterior.PosteriorNet()
    nets[tag].load_state_dict(torch.load(f"results/posterior_smoke_hhs{tag}.pt", map_location="cpu"))
    nets[tag].eval()
print("\nsimulation (prior v4, v5 forward model): bias / RMSE / 90% coverage of the posterior mean")
for lo, hi in [(10, 20), (20, 40), (40, 150)]:
    bt = posterior.simulate_batch(prot, args.n, g=torch.Generator().manual_seed(4242), snr=(lo, hi), prior_version=4,
                                  exact_fw=True, td_ratio=1.5, alpha=(0, 0.03), pulsation=0.15, gain_sd=0.007)
    y, cls = bt.y.numpy(), bt.cls.numpy()
    band = f"{lo}-{hi}"
    res["simulation"][band] = {}
    for tag, net in nets.items():
        smp = net.sample(bt.x, n=100, chunk=512, generator=torch.Generator().manual_seed(0)).numpy()
        mean, pit = smp.mean(0), (smp < y[None]).mean(0)
        cov = (pit > 0.05) & (pit < 0.95)
        out, txt = {}, []
        for c, nm in enumerate(simulate.BRAIN_CLASSES[:2]):
            m = cls == c
            for k in ["fa_t", "ufa_t"]:
                i = T.index(k)
                e = mean[m, i] - y[m, i]
                out[f"{nm}_{k}"] = {"bias": round(float(e.mean()), 4), "rmse": round(float(np.sqrt((e ** 2).mean())), 4),
                                    "coverage90": round(float(cov[m, i].mean()), 3)}
                txt.append(f"{nm} {k} {e.mean():+.3f}/{np.sqrt((e ** 2).mean()):.3f}/{cov[m, i].mean():.2f}")
        res["simulation"][band][tag] = out
        print(f"  SNR {band:7s} {tag:11s} " + "  ".join(txt), flush=True)
out = f"results/low_snr_bias_s{args.sub}{args.ses}.json"
json.dump(res, open(out, "w"), indent=1)
print(f"-> {out}")
