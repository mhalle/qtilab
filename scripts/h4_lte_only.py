"""H4: how much of QTI's information is recoverable from linear encoding (LTE) alone?

Two networks, identical except for the measurements they see (same prior v2, SNR range, steps):
  full      all 107 volumes of the HHS protocol (LTE + STE)          results/posterior_smoke_hhs_v2.pt
  LTE-only  the 69 LTE volumes plus every b = 0 volume (71 volumes)  results/posterior_smoke_hhs_v2lte.pt

1. Simulation, same voxels for both: error of the posterior mean, width of the 90% interval and its
   coverage, by tissue class and target. Test data from prior v2 (the training prior) and from priors
   v1 and v3 (misspecified): with less information, a network leans more on its prior, so the
   LTE-only model should degrade more when the prior is wrong.
2. Real data (HHS sub-01 ses-01, slice 20): agreement between the two networks and the ratio of their
   posterior SDs, per voxel.
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, ".")
from qtilab import paths, posterior, simulate  # noqa: E402
from qtilab.hhs import session_protocol  # noqa: E402
from qtilab.protocol import Protocol  # noqa: E402

import argparse  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--full-model", default="results/posterior_smoke_hhs_v2.pt")
ap.add_argument("--lte-model", default="results/posterior_smoke_hhs_v2lte.pt")
ap.add_argument("--test-priors", default="2,1,3,0", help="comma list; 0 = generic")
ap.add_argument("--no-real", action="store_true")
ap.add_argument("--tag", default="")
args = ap.parse_args()
torch.set_num_threads(2)
T = posterior.TARGETS
SHOW = ["ufa_t", "md_t", "fa_t", "c_md_t", "f_fw", "ufa", "c_md"]
prot = session_protocol("01", "01")
keep = prot.keep_shapes(["LTE"])
prot_lte = prot.subset(keep)
kn = keep.numpy()
nets = {}
for name, f in (("full", args.full_model), ("lte", args.lte_model)):
    n = posterior.PosteriorNet()
    n.load_state_dict(torch.load(f))
    n.eval()
    nets[name] = n


def feats(S, p):
    return posterior.features(S, p.b6.to(S.dtype), p.b.to(S.dtype), p.b_delta().to(S.dtype)).float()


def run_both(S):
    """S (V, 107) float64 -> {name: samples (n, V, T) numpy}."""
    return {"full": nets["full"].sample(feats(S, prot), n=100).numpy(),
            "lte": nets["lte"].sample(feats(S[:, keep], prot_lte), n=100).numpy()}


out = {"sim": {}, "real": {}}
names = simulate.BRAIN_CLASSES
for pv in [int(x) for x in args.test_priors.split(",")]:
    g = torch.Generator().manual_seed(500 + pv)
    test = posterior.simulate_batch(prot, 2000, g=g, snr=(30.0, 140.0), prior_version=pv, dtype=torch.float64)
    y = test.y.numpy()
    cls = test.cls.numpy()
    smp = run_both(test.S)
    res = {}
    label = {2: "brain v2 (training prior)", 1: "brain v1", 3: "brain v3 (wide)", 0: "generic (non-anatomical)"}[pv]
    print(f"\n=== test data from prior {label}: median |error| / median 90% width / 90% coverage, full -> LTE-only")
    groups = [("all", cls == -1)] if pv == 0 else [(cname, cls == c) for c, cname in enumerate(names)]
    for cname, m in groups:
        row = {}
        cells = []
        for k in SHOW:
            i = T.index(k)
            r = {}
            for nm, s in smp.items():
                pm = s.mean(0)[m, i]
                lo, hi = np.quantile(s[:, m, i], [0.05, 0.95], axis=0)
                r[nm] = {"mae": float(np.median(np.abs(pm - y[m, i]))), "w90": float(np.median(hi - lo)),
                         "cov90": float(np.mean((y[m, i] >= lo) & (y[m, i] <= hi)))}
            row[k] = r
            if k in ("ufa_t", "md_t", "c_md_t", "f_fw"):
                cells.append(f"{k} {r['full']['mae']:.3f}/{r['full']['w90']:.2f}/{r['full']['cov90']:.2f} -> "
                             f"{r['lte']['mae']:.3f}/{r['lte']['w90']:.2f}/{r['lte']['cov90']:.2f}")
        res[cname] = row
        print(f"  {cname:7s} " + " | ".join(cells), flush=True)
        del m
    out["sim"][f"v{pv}"] = res
    del smp, test
    import gc; gc.collect()

if args.no_real:
    json.dump(out, open(f"results/h4_lte_only{args.tag}.json", "w"), indent=1)
    sys.exit(0)
# ---- real slice
a = np.load(paths.aligned_session())
c = np.load(paths.classical_fit())
full_mask = c["mask"]
z = 20
slab = np.zeros_like(full_mask); slab[:, :, z] = True
mask = full_mask & slab
sel = slab[full_mask]
fa = c["fa"][sel]
p_real = Protocol.from_table(a["bvals"], a["bvecs"], [str(s) for s in a["shapes"]])
assert torch.equal(p_real.keep_shapes(["LTE"]), keep), "session protocol differs from the training protocol"
prot = p_real                                   # use the motion-rotated b-tensors for the real data
prot_lte = prot.subset(keep)
S = torch.as_tensor(np.maximum(a["data"][mask].astype(np.float64), 1e-4))
smp = run_both(S)
pm = {k: v.mean(0) for k, v in smp.items()}
sd = {k: v.std(0) for k, v in smp.items()}
fw = pm["full"][:, T.index("f_fw")]
regions = {"WM-like": (fa > 0.4) & (fw < 0.2), "GM-like": (fa < 0.2) & (fw < 0.2), "CSF-ish": fw > 0.5}
print("\n=== real slice: median full / LTE-only estimate; median SD ratio (LTE-only / full); voxel correlation")
for rname, m in regions.items():
    row = {}
    for k in ("ufa_t", "md_t", "c_md_t", "f_fw"):
        i = T.index(k)
        ratio = sd["lte"][m, i] / np.maximum(sd["full"][m, i], 1e-6)
        r = np.corrcoef(pm["full"][m, i], pm["lte"][m, i])[0, 1]
        row[k] = {"full": float(np.median(pm["full"][m, i])), "lte": float(np.median(pm["lte"][m, i])),
                  "sd_ratio": float(np.median(ratio)), "r": float(r)}
    out["real"][rname] = row
    print(f"  {rname:8s} n={m.sum():5d}  " + " | ".join(
        f"{k} {v['full']:.3f}/{v['lte']:.3f} sd x{v['sd_ratio']:.1f} r={v['r']:.2f}" for k, v in row.items()))
json.dump(out, open(f"results/h4_lte_only{args.tag}.json", "w"), indent=1)


def vol(v):
    o = np.full(mask.shape, np.nan); o[mask] = v; return o


iu, ic = T.index("ufa_t"), T.index("c_md_t")
panels = [("tissue µFA, full protocol", pm["full"][:, iu], "gray", (0, 1)),
          ("tissue µFA, linear encoding only", pm["lte"][:, iu], "gray", (0, 1)),
          ("difference (LTE-only − full)", pm["lte"][:, iu] - pm["full"][:, iu], "RdBu_r", (-0.2, 0.2)),
          ("posterior SD ratio, µFA (LTE-only / full)", sd["lte"][:, iu] / np.maximum(sd["full"][:, iu], 1e-6), "magma", (1, 3)),
          ("tissue C_MD, full", pm["full"][:, ic], "gray", (0, 0.2)),
          ("tissue C_MD, linear encoding only", pm["lte"][:, ic], "gray", (0, 0.2)),
          ("difference (LTE-only − full)", pm["lte"][:, ic] - pm["full"][:, ic], "RdBu_r", (-0.1, 0.1)),
          ("posterior SD ratio, C_MD", sd["lte"][:, ic] / np.maximum(sd["full"][:, ic], 1e-6), "magma", (1, 3))]
fig, axes = plt.subplots(2, 4, figsize=(15, 8), facecolor="black")
for ax, (t, v, cm, (lo, hi)) in zip(axes.ravel(), panels):
    ax.imshow(np.rot90(vol(v)[:, :, z]), cmap=cm, vmin=lo, vmax=hi); ax.set_title(t, color="white", fontsize=9); ax.axis("off")
fig.suptitle("H4: what linear encoding alone recovers (HHS sub-01 ses-01, slice 20)", color="white")
plt.tight_layout(); plt.savefig(f"results/h4_lte_only{args.tag}.png", dpi=110, facecolor="black")
