"""H7: test-retest reliability of regional QTI values over the Hormone Health cohort, classical vs network.

For every session with a classical fit (prepare_session.py), FastSurfer fractions (fastsurfer_to_dwi.py)
and a whole-brain posterior of the network TAG (posterior_maps_hhs.py --whole-brain; scripts/cohort_run.sh
makes all three), the median of each quantity in voxels >= 80% one FastSurfer region (and covered).
Then, per region and quantity, over subjects with all three sessions:

* ICC(1,1) (one-way random effects; the sessions are menstrual-cycle phases, not a fixed factor), with a
  95% bootstrap interval over subjects;
* the within-subject CV (sqrt(MS_within) / grand mean) and the between/within SD ratio;
* for each classical/network pair, the bootstrap interval of ICC(network) - ICC(classical) (same resamples).

Pairs: classical FA, µFA, C_c, MD, C_MD against the network's tissue moments (FA_t, µFA_t, C_c,t, MD_t,
C_MD,t) and against its full-distribution moments where it has them (µFA, MD, C_MD: the same quantity as
the classical fit, without the b-range truncation). f_fw is reported on its own.

    python scripts/cohort_icc.py [--tag _v5] [--boot 2000]
-> results/cohort_icc{tag}.json (regional medians for every session included)
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from qtilab import paths, posterior  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="_v5")
ap.add_argument("--boot", type=int, default=2000)
args = ap.parse_args()
T = posterior.TARGETS
REGIONS = ["cortex", "caudate", "putamen", "thalamus", "pallidum", "hippo_amyg", "cerebral_wm", "cerebellum_ctx",
           "cerebellum_wm", "brainstem"]
PAIRS = [("fa", "fa_t"), ("ufa", "ufa_t"), ("c_c", "c_c_t"), ("md", "md_t"), ("c_md", "c_md_t"),
         ("ufa", "ufa"), ("md", "md"), ("c_md", "c_md")]

# ---- regional medians per session ----
med = {}                                  # (sub, ses) -> region -> {"cls_x": v, "net_x": v}
for q in sorted(paths.WORK.glob("hhs_sub*_ses*_qti.npz")):
    sub, ses = re.match(r"hhs_sub(\d+)_ses(\d+)_qti", q.stem).groups()
    pf, ff = paths.posterior_volume(args.tag, sub, ses), paths.fastsurfer_fractions(sub, ses)
    if not (pf.exists() and ff.exists()):
        continue
    c, p, f = np.load(q), np.load(pf), np.load(ff)
    mask = c["mask"]
    assert (p["mask"] == mask).all(), (sub, ses)
    names = [str(n) for n in f["names"]]
    cov = f["covered"][mask]
    net = {k: p["mean"][:, i].astype(float) for i, k in enumerate(T)}
    net["c_c_t"] = (net["fa_t"] / np.maximum(net["ufa_t"], 1e-6)) ** 2
    vals = {f"cls_{k}": c[k].astype(float) for k in ("fa", "ufa", "c_c", "md", "c_md")}
    vals |= {f"net_{k}": v for k, v in net.items()}
    med[sub, ses] = {}
    for r in REGIONS:
        s = cov & (f["frac"][names.index(r)][mask] >= 0.8)
        med[sub, ses][r] = {"n": int(s.sum()), **{k: float(np.median(v[s])) for k, v in vals.items()}}

subs = defaultdict(list)
for sub, ses in med:
    subs[sub].append(ses)
full = sorted(s for s, ss in subs.items() if len(ss) == 3)
print(f"network {args.tag}: {len(med)} sessions, {len(subs)} subjects, {len(full)} with all three sessions")
if len(full) < 3:
    sys.exit("need at least 3 subjects with three sessions")


def icc_stats(Y):
    """Y (subjects, k). ICC(1,1), within-subject CV, between/within SD ratio."""
    n, k = Y.shape
    m = Y.mean(1)
    msb = k * ((m - Y.mean()) ** 2).sum() / (n - 1)
    msw = ((Y - m[:, None]) ** 2).sum() / (n * (k - 1))
    icc = (msb - msw) / (msb + (k - 1) * msw)
    return icc, np.sqrt(msw) / abs(Y.mean()), np.sqrt(max(msb - msw, 0) / k) / np.sqrt(msw)


def icc_only(Y):          # Y (B, subjects, k), vectorized over bootstrap resamples
    n, k = Y.shape[1:]
    m = Y.mean(2)
    msb = k * ((m - m.mean(1, keepdims=True)) ** 2).sum(1) / (n - 1)
    msw = ((Y - m[..., None]) ** 2).sum((1, 2)) / (n * (k - 1))
    return (msb - msw) / (msb + (k - 1) * msw)


rng = np.random.default_rng(0)
idx = rng.integers(0, len(full), (args.boot, len(full)))
keys = sorted({k for d in med.values() for r in d.values() for k in r if k != "n"})
res = {"tag": args.tag, "subjects": full, "n_sessions": len(med), "boot": args.boot, "icc": {}, "pairs": {},
       "regional_medians": {f"{s}_{e}": d for (s, e), d in sorted(med.items())}}
for r in REGIONS:
    res["icc"][r], res["pairs"][r] = {}, {}
    Y = {k: np.array([[med[s, e][r][k] for e in sorted(subs[s])] for s in full]) for k in keys}
    boot = {k: icc_only(Y[k][idx]) for k in keys}
    for k in keys:
        icc, cv, ratio = icc_stats(Y[k])
        lo, hi = np.percentile(boot[k], [2.5, 97.5])
        res["icc"][r][k] = {"icc": round(float(icc), 3), "ci95": [round(float(lo), 3), round(float(hi), 3)],
                            "cv_within": round(float(cv), 4), "between_within": round(float(ratio), 2),
                            "cohort_mean": round(float(Y[k].mean()), 4)}
    for ck, nk in PAIRS:
        d = boot[f"net_{nk}"] - boot[f"cls_{ck}"]
        res["pairs"][r][f"{ck}|{nk}"] = {
            "diff": round(res["icc"][r][f"net_{nk}"]["icc"] - res["icc"][r][f"cls_{ck}"]["icc"], 3),
            "ci95": [round(float(x), 3) for x in np.percentile(d, [2.5, 97.5])]}

print("\nICC(1,1) [95% bootstrap CI over subjects]: classical | network (tissue) | network (full distribution)")
for ck, tk in PAIRS[:5]:
    print(f"\n{ck}:")
    for r in REGIONS:
        row = [f"cls {res['icc'][r][f'cls_{ck}']['icc']:5.2f} [{res['icc'][r][f'cls_{ck}']['ci95'][0]:5.2f},{res['icc'][r][f'cls_{ck}']['ci95'][1]:5.2f}]",
               f"net_t {res['icc'][r][f'net_{tk}']['icc']:5.2f} [{res['icc'][r][f'net_{tk}']['ci95'][0]:5.2f},{res['icc'][r][f'net_{tk}']['ci95'][1]:5.2f}]"]
        if f"{ck}|{ck}" in res["pairs"][r]:
            row.append(f"net_full {res['icc'][r][f'net_{ck}']['icc']:5.2f}")
        p = res["pairs"][r][f"{ck}|{tk}"]
        row.append(f"diff_t {p['diff']:+.2f} [{p['ci95'][0]:+.2f},{p['ci95'][1]:+.2f}]")
        print(f"  {r:14s} " + "  ".join(row))
print("\nf_fw: " + ", ".join(f"{r} {res['icc'][r]['net_f_fw']['icc']:.2f}" for r in REGIONS))
out = Path(f"results/cohort_icc{args.tag}.json")
json.dump(res, open(out, "w"), indent=1)
print(f"-> {out}")
