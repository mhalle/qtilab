"""Held-out prediction on real Hormone Health data.

Leave one run out, fit on the rest, predict the left-out images:

  QTI        classical WLS cumulant fit on the remaining volumes, extrapolated
  simulator  the k nearest simulated tissues (matched on the remaining volumes); prediction = mean of
             their signals for the held-out volumes, spread = their sd

Errors are RMS over held-out volumes of (observed - predicted) / noise sd, so a perfect predictor
scores about 1. Noise sd per voxel and volume:
  --noise repeat  one sd per voxel from repeated STE volumes (b = 1500, or b = 1000 when run 8 is
                  held out); the original analysis
  --noise model   thermal + b-dependent signal-proportional model (qtilab.noise), with alpha(b) per
                  region and thermal variance per voxel, both fitted on STE repeat groups that exclude
                  the held-out volumes

Splits: run 4 (LTE, b = 2000, extrapolation) and run 8 (STE, b = 1500).
Regions: from the posterior f_fw (original) or from the classical fit (--regions classical:
WM FA > 0.4 & MD < 1.2, GM FA < 0.2 & MD < 1.2, CSF MD > 2.0), which needs no network output.
"""
import argparse
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from qtilab import fit, noise, paths, simulate  # noqa: E402
from qtilab.protocol import Protocol  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("bank", type=int, nargs="?", default=200_000)
ap.add_argument("--prior", type=int, default=2, help="0 = generic, 1-3 = brain prior versions")
ap.add_argument("--td", type=float, default=None, help="add time dependence with this LTE/STE spectral ratio")
ap.add_argument("--tau-wm", type=float, default=1.0, help="scale of WM time-dependence strengths")
ap.add_argument("--tau-gm", type=float, default=1.0, help="scale of GM time-dependence strengths")
ap.add_argument("--sub", default="01"); ap.add_argument("--ses", default="01")
ap.add_argument("--slice", type=int, default=20, help="axial slice; -1 picks the fullest slice in 10-27")
ap.add_argument("--noise", choices=["repeat", "model"], default="repeat")
ap.add_argument("--regions", choices=["posterior", "classical"], default="posterior")
ap.add_argument("--seed", type=int, default=21)
ap.add_argument("--tag", default="")
args = ap.parse_args()
torch.set_num_threads(2)
K = 20
a = np.load(paths.aligned_session(args.sub, args.ses))
c = np.load(paths.classical_fit(args.sub, args.ses))
full = c["mask"]
z = args.slice if args.slice >= 0 else int(np.argmax(full.sum((0, 1))[10:28])) + 10
slab = np.zeros_like(full); slab[:, :, z] = True
mask = full & slab
sel = slab[full]
fa, md = c["fa"][sel], c["md"][sel]
shapes = np.array([str(s) for s in a["shapes"]])
bvals = a["bvals"]
prot = Protocol.from_table(bvals, a["bvecs"], list(shapes))
run = np.concatenate([[i] * n for i, n in enumerate([13, 10, 16, 30, 2, 12, 9, 15], 1)])
assert len(run) == prot.n
S = np.maximum(a["data"][mask].astype(np.float64), 1e-4)
S = S / S[:, bvals < 50].mean(1, keepdims=True)
if args.regions == "posterior":
    post = np.load(paths.posterior_slice("_v2", args.sub, args.ses))
    fw = post["mean"][:, list(post["targets"]).index("f_fw")]
    regions = {"WM-like": (fa > 0.4) & (fw < 0.2), "GM-like": (fa < 0.2) & (fw < 0.2), "CSF-ish": fw > 0.5}
else:
    regions = {"WM-like": (fa > 0.4) & (md < 1.2), "GM-like": (fa < 0.2) & (md < 1.2), "CSF-ish": md > 2.0}
nreg = noise.region_of(fa, md)

g = torch.Generator().manual_seed(args.seed)
is_ste = torch.as_tensor((shapes == "STE").astype(np.float64))
chunks = []
for i in range(0, args.bank, 20000):
    k = min(20000, args.bank - i)
    if args.prior == 0:
        d6 = simulate.sample_dtd(k, M=48, g=g)
        chunks.append(simulate.signal(d6, prot.b6).float())
    elif args.td is None:
        d6, _ = simulate.sample_brain(k, M=48, g=g, version=args.prior)
        chunks.append(simulate.signal(d6, prot.b6).float())
    else:
        d6, cls, comp = simulate.sample_brain(k, M=48, g=g, version=args.prior, return_comp=True)
        tau = simulate.sample_tau(cls, comp, g, wm_scale=args.tau_wm, gm_scale=args.tau_gm)
        omega = 1.0 + (args.td - 1.0) * is_ste
        chunks.append(simulate.signal_td(d6, prot.b6, omega, tau).float())
bank = torch.cat(chunks)                                            # (BANK, N), S0 = 1
out = {"bank": args.bank, "k": K, "prior": args.prior, "td": args.td, "tau_wm": args.tau_wm, "tau_gm": args.tau_gm,
       "sub": args.sub, "ses": args.ses, "slice": z, "noise": args.noise, "regions": args.regions}

for held in (4, 8):
    test = run == held
    train = ~test
    if args.noise == "repeat":
        ref = (shapes == "STE") & (np.abs(bvals - (1000 if held == 8 else 1500)) < 1)
        sig = np.repeat(S[:, ref].std(1, ddof=1)[:, None], test.sum(), 1)
    else:
        groups, levels = noise.ste_groups(bvals, shapes, exclude=test | (bvals < 50))
        m, v, dof = noise.group_moments(S, groups)
        s2 = np.zeros(len(S)); al_vox = np.zeros((len(S), test.sum()))
        for r in range(3):
            rr = nreg == r
            if rr.sum() < 20:
                continue
            al, s2r, _ = noise.fit_alpha_groups(m[rr], v[rr], dof)
            s2[rr] = s2r
            al_vox[rr] = noise.alpha_for_bvals(bvals[test], levels, al)[None, :]
        sig = np.sqrt(s2[:, None] + (al_vox * S[:, test]) ** 2)
    X = prot.design_matrix()
    p = fit.wls(X[train], torch.as_tensor(S[:, train]))
    pred_qti = fit.predict(X[test], p).numpy()
    Bt = bank[:, train]
    pred_sim = np.zeros((S.shape[0], test.sum()))
    spread = np.zeros_like(pred_sim)
    for i in range(0, S.shape[0], 256):
        d = torch.cdist(torch.as_tensor(S[i:i + 256, train], dtype=torch.float32), Bt)
        idx = d.topk(K, largest=False).indices
        nb = bank[:, test][idx].double()
        pred_sim[i:i + 256] = nb.mean(1).numpy()
        spread[i:i + 256] = nb.std(1).numpy()
    zq = (S[:, test] - pred_qti) / sig
    zs = (S[:, test] - pred_sim) / sig
    e_q, e_s = np.sqrt(np.mean(zq ** 2, 1)), np.sqrt(np.mean(zs ** 2, 1))
    cover = (np.abs((S[:, test] - pred_sim) / np.sqrt(spread ** 2 + sig ** 2)) < 1.96).mean(1)
    print(f"\nheld out run {held} ({shapes[test][0]}, b = {int(bvals[test][0])}, {test.sum()} volumes), "
          f"sub-{args.sub} ses-{args.ses} slice {z}, noise={args.noise}: median RMS error / noise sd (1 = perfect)")
    res = {}
    for name, m in list(regions.items()) + [("all", np.ones(len(fa), bool))]:
        res[name] = {"n": int(m.sum()), "qti": float(np.median(e_q[m])), "sim": float(np.median(e_s[m])),
                     "sim_95pct_coverage": float(np.median(cover[m])),
                     "sim_bias": float(np.median(zs[m])), "qti_bias": float(np.median(zq[m]))}
        r = res[name]
        print(f"  {name:8s} n={m.sum():5d}  QTI {r['qti']:.2f} (bias {r['qti_bias']:+.2f})   "
              f"simulator {r['sim']:.2f} (bias {r['sim_bias']:+.2f}, 95% interval covers {100*r['sim_95pct_coverage']:.0f}%)")
    out[f"run{held}"] = res
json.dump(out, open(f"results/heldout_hhs{args.tag}.json", "w"), indent=1)
