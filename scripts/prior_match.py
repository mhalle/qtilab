"""Does a brain prior produce real-looking classical QTI maps? Region-matched histograms, no training.

Simulates pure WM and pure GM voxels from each prior through the session's own protocol, with Rician
noise at SNRs drawn from the real voxels of the matching FastSurfer region (per-voxel SNR from the
STE b = 1.5 repeats, as in surprise_check.py), fits them with the same constrained QTI fit as the real
data, and compares the classical scalars with the real ones in voxels >= 80% one FastSurfer region:

    sim WM  <->  cerebral white matter        sim GM  <->  cortex

A second, label-free check uses no anatomy: simulated WM + GM (half each) against all real tissue
(FastSurfer fluid + background < 20%, any region), as joint 2-D histograms in the QTI planes (µFA, C_c)
and (MD, C_MD); it reports the fraction of real tissue in bins where the simulation has under 10% of the
real density. Deep gray matter (thalamus, pallidum), which has no simulated class, is covered or not
by what the prior's continuous parameters produce.

Comparing classical fits on both sides (rather than the prior's true values with real classical
values) puts the fit's own biases on both sides. The comparison is summary statistics only, never
voxels; held-out prediction (heldout_hhs.py) stays the arbiter between priors.

    python scripts/prior_match.py [sub] [ses] [--priors 2 3 4] [--n 20000]
        # needs prepare_session.py and fastsurfer_to_dwi.py; ~30 s per prior on CPU
-> results/prior_match_s{sub}{ses}.json and .png
"""
import argparse
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import wasserstein_distance  # noqa: E402

sys.path.insert(0, ".")
from qtilab import fit, paths, scalars, simulate  # noqa: E402
from qtilab.protocol import Protocol  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("sub", nargs="?", default="01")
ap.add_argument("ses", nargs="?", default="01")
ap.add_argument("--priors", nargs="+", default=["1", "2", "3"])
ap.add_argument("--n", type=int, default=20000, help="simulated voxels per class")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=None, help="output stem (default results/prior_match_s{sub}{ses})")
args = ap.parse_args()
sub, ses = args.sub, args.ses

KEYS = ["fa", "ufa", "md", "c_md", "c_c"]
LABEL = {"fa": "FA", "ufa": "µFA", "md": "MD (µm²/ms)", "c_md": "C_MD", "c_c": "C_c"}
RANGE = {"fa": (0, 1), "ufa": (0, 1.1), "md": (0, 1.6), "c_md": (0, 0.4), "c_c": (0, 1.1)}
PAIRS = [("WM", 0, "cerebral_wm"), ("GM", 1, "cortex")]
REFERENCE = ["thalamus", "pallidum", "putamen", "cerebellum_wm", "brainstem"]   # real only: no sim class
QS = [5, 25, 50, 75, 95]

# ---- real data
a = np.load(paths.aligned_session(sub, ses))
q = np.load(paths.classical_fit(sub, ses))
f = np.load(paths.fastsurfer_fractions(sub, ses))
mask = q["mask"]
names = [str(n) for n in f["names"]]
cov = f["covered"][mask]
pure = {n: cov & (f["frac"][names.index(n)][mask] >= 0.8) for n in names}
shapes = [str(s) for s in a["shapes"]]
prot = Protocol.from_table(a["bvals"], a["bvecs"], shapes)
b = prot.b.numpy()
S = np.maximum(a["data"][mask].astype(np.float64), 1e-4)
rep = (np.array(shapes) == "STE") & (np.abs(b - 1.5) < 0.01)
snr_real = S[:, b < 0.05].mean(1) / np.maximum(S[:, rep].std(1, ddof=1), 1e-6)
real = {k: q[k] for k in KEYS}


def summary(v):
    v = v[np.isfinite(v)]
    return {"q": np.percentile(v, QS).round(4).tolist(), "mean": round(float(v.mean()), 4)}


def gap(fa):
    return round(float(((fa >= 0.3) & (fa < 0.5)).mean() * 100), 2)


res = {"sub": sub, "ses": ses, "n_per_class": args.n, "quantiles": QS, "real": {}, "priors": {}}
for _, _, region in PAIRS:
    sel = pure[region]
    res["real"][region] = {"n": int(sel.sum()), "snr_median": round(float(np.median(snr_real[sel])), 1),
                           "pct_fa_0.3_0.5": gap(real["fa"][sel]), **{k: summary(real[k][sel]) for k in KEYS}}
for region in REFERENCE:
    sel = pure[region]
    res["real"][region] = {"n": int(sel.sum()), "pct_fa_0.3_0.5": gap(real["fa"][sel]),
                           **{k: summary(real[k][sel]) for k in KEYS}}

tissue = cov & (f["frac"][names.index("background")][mask] + f["frac"][names.index("csf_ventricles")][mask] < 0.2)
res["real_tissue_voxels_label_free"] = int(tissue.sum())
PLANES = {"ufa_c_c": ("ufa", "c_c", (0, 1.05), (0, 1.05)), "md_c_md": ("md", "c_md", (0.4, 1.6), (0, 0.3))}


def undercovered(sim, kx, ky, rx, ry, bins=20):
    """% of real tissue in (kx, ky) bins where the simulated density is < 10% of the real density."""
    ex, ey = np.linspace(*rx, bins + 1), np.linspace(*ry, bins + 1)
    hr = np.histogram2d(np.clip(real[kx][tissue], *rx), np.clip(real[ky][tissue], *ry), [ex, ey])[0]
    hs = np.histogram2d(np.clip(sim[kx], *rx), np.clip(sim[ky], *ry), [ex, ey])[0]
    hr, hs = hr / hr.sum(), hs / hs.sum()
    return round(float(100 * hr[hs < 0.1 * hr].sum()), 1)


# ---- simulated, per prior
X = prot.design_matrix()
sims = {}
for pv in args.priors:
    version = int(pv)
    g = torch.Generator().manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = {}
    for cname, ci, region in PAIRS:
        probs = tuple(1.0 if i == ci else 0.0 for i in range(4))
        d6, cls = simulate.sample_brain(args.n, M=64, g=g, class_probs=probs, version=version)
        snr = torch.as_tensor(rng.choice(snr_real[pure[region]], args.n))
        Ssim = simulate.rician(simulate.signal(d6, prot.b6.to(d6.dtype)), 1.0 / snr, g).clamp_min(1e-4)
        p, _ = fit.constrained(X, Ssim)
        sc = scalars.scalars(p)
        sc = {k: sc[k].numpy() for k in KEYS}
        keep = ~scalars.collapsed(scalars.scalars(p)).numpy()
        sc = {k: v[keep] for k, v in sc.items()}
        rsel = pure[region]
        out[cname] = {"pct_fa_0.3_0.5": gap(sc["fa"]), "collapsed": int((~keep).sum()),
                      **{k: {**summary(sc[k]), "w1_to_real": round(float(wasserstein_distance(sc[k], real[k][rsel])), 4)}
                         for k in KEYS}}
        sims[(pv, cname)] = sc
    # label-free: half WM, half GM against all real tissue, in the QTI planes
    n2 = args.n
    d6, _ = simulate.sample_brain(n2, M=64, g=g, class_probs=(0.5, 0.5, 0, 0), version=version)
    snr = torch.as_tensor(rng.choice(snr_real[tissue], n2))
    p, _ = fit.constrained(X, simulate.rician(simulate.signal(d6, prot.b6.to(d6.dtype)), 1.0 / snr, g).clamp_min(1e-4))
    sc = scalars.scalars(p)
    keep = ~scalars.collapsed(sc).numpy()
    out["label_free_pct_real_tissue_under_covered"] = {
        name: undercovered({k: sc[k].numpy()[keep] for k in (kx, ky)}, kx, ky, rx, ry)
        for name, (kx, ky, rx, ry) in PLANES.items()}
    res["priors"][pv] = out
    print(f"prior v{pv}: " + "; ".join(
        f"{c} FA 0.3-0.5 {out[c]['pct_fa_0.3_0.5']:.1f}% (real {res['real'][r]['pct_fa_0.3_0.5']:.1f}%), "
        + " ".join(f"W1 {k} {out[c][k]['w1_to_real']:.3f}" for k in KEYS) for c, _, r in PAIRS)
          + "; label-free, % of real tissue under-covered: "
          + ", ".join(f"{k} {v:.0f}%" for k, v in out["label_free_pct_real_tissue_under_covered"].items()), flush=True)

print("\nmedians [5th, 95th]:  " + "  ".join(f"{LABEL[k]}" for k in KEYS))
for cname, _, region in PAIRS:
    r = res["real"][region]
    print(f"  real {region:12s} " + "  ".join(f"{r[k]['q'][2]:.3f} [{r[k]['q'][0]:.2f},{r[k]['q'][4]:.2f}]" for k in KEYS))
    for pv in args.priors:
        s = res["priors"][pv][cname]
        print(f"  v{pv:<3s} {cname:12s} " + "  ".join(f"{s[k]['q'][2]:.3f} [{s[k]['q'][0]:.2f},{s[k]['q'][4]:.2f}]" for k in KEYS))
for region in REFERENCE:
    r = res["real"][region]
    print(f"  real {region:12s} " + "  ".join(f"{r[k]['q'][2]:.3f} [{r[k]['q'][0]:.2f},{r[k]['q'][4]:.2f}]" for k in KEYS))

stem = args.out or f"results/prior_match_s{sub}{ses}"
with open(f"{stem}.json", "w") as fh:
    json.dump(res, fh, indent=1)

colors = ["#7E57C2", "#EF6C00", "#2E7D32", "#1565C0", "#AD1457"]
fig, axs = plt.subplots(len(PAIRS), len(KEYS), figsize=(4 * len(KEYS), 3.4 * len(PAIRS)))
for ri, (cname, _, region) in enumerate(PAIRS):
    for ki, k in enumerate(KEYS):
        ax = axs[ri, ki]
        bins = np.linspace(*RANGE[k], 56)
        ax.hist(np.clip(real[k][pure[region]], *RANGE[k]), bins, density=True, color="#bbb", label=f"real {region}")
        for pv, c in zip(args.priors, colors):
            ax.hist(np.clip(sims[(pv, cname)][k], *RANGE[k]), bins, density=True, histtype="step", lw=1.5, color=c,
                    label=f"prior v{pv}")
        ax.set_title(f"{cname}: classical {LABEL[k]}", fontsize=10)
        ax.set_yticks([])
        if ki == 0:
            ax.legend(fontsize=8)
fig.suptitle(f"Classical constrained QTI on simulated vs real voxels (sub-{sub} ses-{ses}; real = voxels ≥ 80% one "
             "FastSurfer region; simulated SNR drawn from that region)", fontsize=11)
plt.tight_layout(); plt.savefig(f"{stem}.png", dpi=120)
print(f"\n-> {stem}.json, {stem}.png")
