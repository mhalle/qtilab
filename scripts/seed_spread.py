"""How much do networks trained identically except for the random seed disagree on real data?

For whole-brain posteriors from several training runs (posterior_maps_hhs.py --whole-brain), per target:
* voxelwise: the SD across runs of the posterior mean, against the median posterior SD (the network's own
  uncertainty) - if the between-run SD is a sizeable fraction of it, single-run maps understate uncertainty;
* regional: the range (max - min) across runs of the regional medians, in voxels >= 80% one FastSurfer
  region - the smallest regional difference worth interpreting from one run.

    python scripts/seed_spread.py [--tags _v4long _v4long_s1 _v4long_s2]
-> results/seed_spread_s{sub}{ses}.json
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from qtilab import paths, posterior  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tags", nargs="+", default=["_v4long", "_v4long_s1", "_v4long_s2"])
ap.add_argument("--sub", default="01")
ap.add_argument("--ses", default="01")
args = ap.parse_args()
T = posterior.TARGETS
REGIONS = ["cortex", "caudate", "putamen", "thalamus", "pallidum", "cerebral_wm", "cerebellum_wm", "brainstem"]

ps = [np.load(paths.posterior_volume(t, args.sub, args.ses)) for t in args.tags]
mask = ps[0]["mask"]
assert all((p["mask"] == mask).all() for p in ps)
means = np.stack([p["mean"].astype(float) for p in ps])           # (runs, V, T)
sds = np.stack([p["sd"].astype(float) for p in ps])
f = np.load(paths.fastsurfer_fractions(args.sub, args.ses))
names = [str(n) for n in f["names"]]
cov = f["covered"][mask]
pure = {r: cov & (f["frac"][names.index(r)][mask] >= 0.8) for r in REGIONS}
tissue = means[0][:, T.index("f_fw")] < 0.3

res = {"tags": args.tags, "sub": args.sub, "ses": args.ses, "voxelwise": {}, "regional_range": {}, "regional_medians": {}}
print(f"{len(args.tags)} runs: {', '.join(args.tags)}; sub-{args.sub} ses-{args.ses}, {mask.sum()} voxels")
print("\nvoxelwise, tissue voxels (f_fw < 0.3): median between-run SD of the posterior mean vs median posterior SD")
for i, k in enumerate(T):
    between = means[:, tissue, i].std(0, ddof=1)
    own = sds[:, tissue, i].mean(0)
    r = {"between_run_sd_median": float(np.median(between)), "posterior_sd_median": float(np.median(own)),
         "ratio_median": float(np.median(between / np.maximum(own, 1e-9)))}
    res["voxelwise"][k] = r
    print(f"  {k:7s} between-run {r['between_run_sd_median']:.4f}   posterior {r['posterior_sd_median']:.4f}   "
          f"median ratio {r['ratio_median']:.2f}")
print("\nregional medians: range across runs (max - min)")
print(f"  {'region':14s} " + " ".join(f"{k:>7s}" for k in T))
for r in REGIONS:
    med = np.array([[np.median(m[pure[r], i]) for i in range(len(T))] for m in means])   # (runs, T)
    res["regional_medians"][r] = med.round(4).tolist()
    res["regional_range"][r] = dict(zip(T, (med.max(0) - med.min(0)).round(4).tolist()))
    print(f"  {r:14s} " + " ".join(f"{x:7.3f}" for x in med.max(0) - med.min(0)))
out = f"results/seed_spread_s{args.sub}{args.ses}.json"
json.dump(res, open(out, "w"), indent=1)
print(f"-> {out}")
