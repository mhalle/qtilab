"""Cardiff b-tensor data: classical QTI and posterior networks from b <= 2 against a b <= 6 reference.

Networks (scripts/posterior_smoke.py --protocol cardiff --bmax B, prior v4, v5's forward model at Cardiff's
SNR; the b <= 6 ones with two-channel magnitude noise, --channels 2, the floor measured in the data) are applied to the motion/eddy-corrected data restricted to their own b-range, signals
normalized by each run's mean b0. Posterior means from n = 50 samples (CPU, chunk 512, seeded).

Reference: the b <= 6 network (prior v4, two-channel noise). Alternative references show its dependence on
the prior (stick axons, intra-axonal D_Delta 0.99-1, ``--prior-set ax_dd=0.99,1.0``, which predicts the real
high-b WM signal better; scripts/cardiff_extrapolation.py) and on the noise model (single-channel Rician,
``_b6v4``). Compared against the reference, voxelwise (median |difference|) and as medians in WM-like / GM-like
tissue (b <= 2 classical fit: FA > 0.4; FA < 0.2 and 0.5 < MD < 1.2):
* classical constrained QTI at b <= 2 (full-distribution moments; also at b <= 6, the apparent values' drift);
* the b <= 2 network (full-distribution and tissue moments).

    python scripts/cardiff_networks.py [--sub data1] [--b2 _b2v4] [--ref _b6v4c2] [--alts _b6v4stickc2 _b6v4]
-> results/cardiff_networks_{sub}.json; posterior means in QTILAB_WORK/cardiff_{sub}_networks.npz
"""
import argparse
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from qtilab import fit, paths, posterior, scalars  # noqa: E402
from qtilab.protocol import Protocol  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sub", default="data1")
ap.add_argument("--b2", default="_b2v4")
ap.add_argument("--ref", default="_b6v4c2")
ap.add_argument("--alts", nargs="*", default=["_b6v4stickc2", "_b6v4"])
ap.add_argument("--n", type=int, default=50)
args = ap.parse_args()
torch.set_num_threads(8)
T = posterior.TARGETS

z = np.load(paths.WORK / f"cardiff_{args.sub}ec_aligned.npz")
mask, b, shapes, run, bvecs = z["mask"], z["bvals"] / 1000, z["shapes"], z["run_index"], z["bvecs"]
S = z["data"][mask].astype(np.float64)
s0 = np.stack([S[:, (run == k) & (b < 0.05)].mean(1) for k in range(3)], 1)
good = (s0 > 0).all(1)
Sn = np.maximum(S[good] / s0[good][:, run], 1e-4)
prot_all = Protocol.from_table(b * 1000, bvecs, list(shapes))


def subset(bmax):
    keep = b <= bmax + 0.06
    return prot_all.subset(torch.as_tensor(keep)), torch.as_tensor(Sn[:, keep])


est = {}
for bmax in (2.0, 6.0):
    p, s = subset(bmax)
    sc = scalars.scalars(fit.constrained(p.design_matrix(), s)[0])
    est[f"cls_b{bmax:g}"] = {k: sc[k].numpy() for k in ("md", "ufa", "c_md", "fa", "c_c")}
fa2, md2 = est["cls_b2"]["fa"], est["cls_b2"]["md"]
groups = {"WM-like": fa2 > 0.4, "GM-like": (fa2 < 0.2) & (md2 > 0.5) & (md2 < 1.2)}

for tag, bmax in [(args.b2, 2.0), (args.ref, 6.0)] + [(a, 6.0) for a in args.alts]:
    t = time.perf_counter()
    p, s = subset(bmax)
    net = posterior.PosteriorNet()
    net.load_state_dict(torch.load(f"results/posterior_smoke_cardiff{tag}.pt", map_location="cpu"))
    net.eval()
    x = posterior.features(s, p.b6, p.b, p.b_delta()).float()
    with torch.no_grad():
        smp = net.sample(x, n=args.n, chunk=512, generator=torch.Generator().manual_seed(0))
    m = smp.mean(0).numpy()
    est[f"net{tag}"] = {k: m[:, i] for i, k in enumerate(T)}
    est[f"net{tag}"]["sd_ufa"] = smp.std(0).numpy()[:, T.index("ufa")]
    print(f"network {tag} (b <= {bmax:g}, {p.n} volumes): {len(x)} voxels in {time.perf_counter() - t:.0f} s", flush=True)

ref, b2 = est[f"net{args.ref}"], est[f"net{args.b2}"]
alts = {f"alt ref {a}": est[f"net{a}"] for a in args.alts}
res = {"sub": args.sub, "n_vox": int(good.sum()), "reference": args.ref, "alt_references": args.alts, "b2_network": args.b2,
       "groups": {}}
QS = ("ufa", "md", "c_md")
print("\nmedian |estimate - reference| (and median signed difference), reference = b <= 6 network (v4)")
for gname, gm in groups.items():
    out = {"n": int(gm.sum()), "medians": {}, "abs_diff": {}, "signed_diff": {}}
    rows = {"classical b<=2": est["cls_b2"], "classical b<=6": est["cls_b6"], f"net b<=2 {args.b2}": b2, **alts}
    for k in QS:
        out["medians"][k] = {"reference": round(float(np.median(ref[k][gm])), 3),
                             **{n: round(float(np.median(r[k][gm])), 3) for n, r in rows.items()}}
        out["abs_diff"][k] = {n: round(float(np.median(np.abs(r[k][gm] - ref[k][gm]))), 3) for n, r in rows.items()}
        out["signed_diff"][k] = {n: round(float(np.median(r[k][gm] - ref[k][gm])), 3) for n, r in rows.items()}
    for k in ("ufa_t", "md_t", "c_md_t", "f_fw"):
        nets = {f"net b<=2 {args.b2}": b2, **alts}
        out["medians"][k] = {"reference": round(float(np.median(ref[k][gm])), 3),
                             **{n: round(float(np.median(r[k][gm])), 3) for n, r in nets.items()}}
        out["abs_diff"][k] = {n: round(float(np.median(np.abs(r[k][gm] - ref[k][gm]))), 3) for n, r in nets.items()}
    out["ref_posterior_sd_ufa"] = round(float(np.median(ref["sd_ufa"][gm])), 3)
    out["b2_posterior_sd_ufa"] = round(float(np.median(b2["sd_ufa"][gm])), 3)
    res["groups"][gname] = out
    print(f"\n{gname} (n = {gm.sum()}); posterior SD of µFA: reference {out['ref_posterior_sd_ufa']}, b<=2 net {out['b2_posterior_sd_ufa']}")
    for k in out["medians"]:
        md = out["medians"][k]
        print(f"  {k:7s} medians: " + "  ".join(f"{n} {v:.3f}" for n, v in md.items()))
        if k in out["abs_diff"]:
            print(f"          |diff|:  " + "  ".join(f"{n} {v:.3f}" + (f" ({out['signed_diff'][k][n]:+.3f})" if k in out["signed_diff"] else "")
                                         for n, v in out["abs_diff"][k].items()))
np.savez_compressed(paths.WORK / f"cardiff_{args.sub}_networks.npz", mask=mask, good=good,
                    **{f"{e}__{k}": v.astype(np.float32) for e, d in est.items() for k, v in d.items()})
json.dump(res, open(f"results/cardiff_networks_{args.sub}.json", "w"), indent=1)
print(f"-> results/cardiff_networks_{args.sub}.json")
