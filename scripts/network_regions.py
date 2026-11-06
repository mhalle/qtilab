"""Whole-brain network posteriors (several priors) vs the classical fit, by FastSurfer region.

For each tag's whole-brain posterior (posterior_maps_hhs.py --whole-brain), medians in voxels >= 80% one
FastSurfer region of the tissue moments (FA_t, µFA_t, C_c,t = (FA_t / µFA_t)^2, MD_t, C_MD,t), f_fw and
the median posterior SD of FA_t, next to the classical constrained fit's FA, µFA, C_c, MD, C_MD.
Also the share of tissue (f_fw < 0.3) with network FA_t in 0.3-0.5 (the v2 prior's gap).

    python scripts/network_regions.py [--tags _v2 _v3 _v4] [--sub 01 --ses 01]
-> results/network_regions_s{sub}{ses}.json
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from qtilab import paths, posterior  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tags", nargs="+", default=["_v2", "_v3", "_v4"])
ap.add_argument("--sub", default="01")
ap.add_argument("--ses", default="01")
args = ap.parse_args()
T = posterior.TARGETS
REGIONS = ["cortex", "caudate", "putamen", "thalamus", "pallidum", "cerebral_wm", "cerebellum_wm", "brainstem"]

q = np.load(paths.classical_fit(args.sub, args.ses))
f = np.load(paths.fastsurfer_fractions(args.sub, args.ses))
mask = q["mask"]
names = [str(n) for n in f["names"]]
cov = f["covered"][mask]
pure = {r: cov & (f["frac"][names.index(r)][mask] >= 0.8) for r in REGIONS}
med = lambda v, s: round(float(np.median(v[s])), 3)  # noqa: E731

res = {"sub": args.sub, "ses": args.ses, "classical": {}, "networks": {}}
for r in REGIONS:
    s = pure[r]
    res["classical"][r] = {"n": int(s.sum()), "fa": med(q["fa"], s), "ufa": med(q["ufa"], s), "c_c": med(q["c_c"], s),
                           "md": med(q["md"], s), "c_md": med(q["c_md"], s)}
for tag in args.tags:
    p = np.load(paths.posterior_volume(tag, args.sub, args.ses))
    assert (p["mask"] == mask).all()
    m, sd = p["mean"], p["sd"]
    col = lambda a, k: a[:, T.index(k)].astype(float)  # noqa: E731
    fa_t, ufa_t, fw = col(m, "fa_t"), col(m, "ufa_t"), col(m, "f_fw")
    cc_t = (fa_t / np.maximum(ufa_t, 1e-6)) ** 2
    out = {}
    for r in REGIONS:
        s = pure[r]
        out[r] = {"fa_t": med(fa_t, s), "ufa_t": med(ufa_t, s), "c_c_t": med(cc_t, s), "md_t": med(col(m, "md_t"), s),
                  "c_md_t": med(col(m, "c_md_t"), s), "f_fw": med(fw, s), "sd_fa_t": med(col(sd, "fa_t"), s)}
    tis = fw < 0.3
    out["pct_tissue_fa_t_0.3_0.5"] = round(float(((fa_t[tis] >= .3) & (fa_t[tis] < .5)).mean() * 100), 1)
    res["networks"][tag] = out

print(f"sub-{args.sub} ses-{args.ses}: medians in voxels >= 80% one FastSurfer region")
for key, ck, lab in [("fa_t", "fa", "FA"), ("ufa_t", "ufa", "µFA"), ("c_c_t", "c_c", "C_c"), ("md_t", "md", "MD"),
                     ("c_md_t", "c_md", "C_MD"), ("f_fw", None, "f_fw"), ("sd_fa_t", None, "posterior SD of FA_t")]:
    print(f"\n{lab}: " + ("classical | " if ck else "") + " | ".join(args.tags))
    for r in REGIONS:
        cl = f"{res['classical'][r][ck]:6.3f} | " if ck else ""
        print(f"  {r:14s} {cl}" + " | ".join(f"{res['networks'][t][r][key]:6.3f}" for t in args.tags))
print("\n% of tissue (f_fw < 0.3) with network FA_t in 0.3-0.5: "
      + ", ".join(f"{t} {res['networks'][t]['pct_tissue_fa_t_0.3_0.5']:.1f}%" for t in args.tags))
out = f"results/network_regions_s{args.sub}{args.ses}.json"
json.dump(res, open(out, "w"), indent=1)
print(f"-> {out}")
