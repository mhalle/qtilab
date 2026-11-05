"""Held-out prediction on a real Hormone Health session (sub-01 ses-01, slice 20).

Leave one run out, fit on the rest, predict the left-out images:

  QTI        classical WLS cumulant fit on the remaining volumes, extrapolated
  simulator  the k nearest simulated tissues (brain prior v2, matched on the remaining volumes);
             the prediction is the mean of their signals for the held-out volumes, the spread their sd

Errors are in units of each voxel's noise sd (from the 15 repeated STE b = 1500 volumes). A perfect
predictor scores about 1, the noise in the held-out data itself. Two splits:
  run 4   linear, b = 2000 (30 directions): the highest b, an extrapolation
  run 8   spherical, b = 1500 (15 volumes): the highest spherical shell

For run 8 the noise sd comes from run 7's 9 repeats (STE b = 1000) instead, since run 8 is held out.
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from qtilab import fit, simulate  # noqa: E402
from qtilab.protocol import Protocol  # noqa: E402

import argparse  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("bank", type=int, nargs="?", default=200_000)
ap.add_argument("--prior", type=int, default=2, help="0 = generic, 1-3 = brain prior versions")
ap.add_argument("--td", type=float, default=None, help="add time dependence with this LTE/STE spectral ratio")
ap.add_argument("--tag", default="")
args = ap.parse_args()
torch.set_num_threads(2)
K = 20
BANK = args.bank
a = np.load("/home/claude/data/hhs_sub01_ses01_aligned.npz")
c = np.load("/home/claude/qtilab/results/hhs_sub01_ses01.npz")
post = np.load("results/hhs_sub01_ses01_posterior_v2.npz")
full = c["mask"]
z = 20
slab = np.zeros_like(full); slab[:, :, z] = True
mask = full & slab
sel = slab[full]
fa = c["fa"][sel]
fw = post["mean"][:, list(post["targets"]).index("f_fw")]
shapes = np.array([str(s) for s in a["shapes"]])
bvals = a["bvals"]
prot = Protocol.from_table(bvals, a["bvecs"], list(shapes))
run = np.concatenate([[i] * n for i, n in enumerate([13, 10, 16, 30, 2, 12, 9, 15], 1)])
assert len(run) == prot.n
S = np.maximum(a["data"][mask].astype(np.float64), 1e-4)
S = S / S[:, bvals < 50].mean(1, keepdims=True)
regions = {"WM-like": (fa > 0.4) & (fw < 0.2), "GM-like": (fa < 0.2) & (fw < 0.2), "CSF-ish": fw > 0.5}

g = torch.Generator().manual_seed(21)
is_ste = torch.as_tensor((shapes == "STE").astype(np.float64))
chunks = []
for i in range(0, BANK, 20000):
    k = min(20000, BANK - i)
    if args.prior == 0:
        d6 = simulate.sample_dtd(k, M=48, g=g)
        chunks.append(simulate.signal(d6, prot.b6).float())
    elif args.td is None:
        d6, _ = simulate.sample_brain(k, M=48, g=g, version=args.prior)
        chunks.append(simulate.signal(d6, prot.b6).float())
    else:
        d6, cls, comp = simulate.sample_brain(k, M=48, g=g, version=args.prior, return_comp=True)
        tau = simulate.sample_tau(cls, comp, g)
        omega = 1.0 + (args.td - 1.0) * is_ste
        chunks.append(simulate.signal_td(d6, prot.b6, omega, tau).float())
bank = torch.cat(chunks)                                            # (BANK, N), S0 = 1
out = {"bank": BANK, "k": K, "prior": args.prior, "td": args.td}

for held, noise_ref in ((4, (shapes == "STE") & (np.abs(bvals - 1500) < 1)),
                        (8, (shapes == "STE") & (np.abs(bvals - 1000) < 1))):
    test = run == held
    train = ~test
    sig = S[:, noise_ref].std(1, ddof=1)                            # noise sd, normalized units
    # classical QTI on the training volumes
    X = prot.design_matrix()
    p = fit.wls(X[train], torch.as_tensor(S[:, train]))
    pred_qti = fit.predict(X[test], p).numpy()
    # simulator nearest neighbors on the training volumes
    Bt = bank[:, train]
    pred_sim = np.zeros((S.shape[0], test.sum()))
    spread = np.zeros_like(pred_sim)
    for i in range(0, S.shape[0], 256):
        d = torch.cdist(torch.as_tensor(S[i:i + 256, train], dtype=torch.float32), Bt)
        idx = d.topk(K, largest=False).indices                       # (n, K)
        nb = bank[:, test][idx].double()                             # (n, K, Ntest)
        pred_sim[i:i + 256] = nb.mean(1).numpy()
        spread[i:i + 256] = nb.std(1).numpy()
    e_q = np.sqrt(np.mean((S[:, test] - pred_qti) ** 2, 1)) / sig
    e_s = np.sqrt(np.mean((S[:, test] - pred_sim) ** 2, 1)) / sig
    zs = (S[:, test] - pred_sim) / np.sqrt(spread ** 2 + sig[:, None] ** 2)
    cover = (np.abs(zs) < 1.96).mean(1)
    print(f"\nheld out run {held} ({shapes[test][0]}, b = {int(bvals[test][0])}, {test.sum()} volumes): "
          "RMS error / noise sd (1 = perfect), median")
    res = {}
    for name, m in list(regions.items()) + [("all", np.ones(len(fa), bool))]:
        res[name] = {"qti": float(np.median(e_q[m])), "sim": float(np.median(e_s[m])),
                     "sim_95pct_coverage": float(np.median(cover[m])),
                     "sim_bias": float(np.median((S[:, test] - pred_sim)[m] / sig[m, None])),
                     "qti_bias": float(np.median((S[:, test] - pred_qti)[m] / sig[m, None]))}
        r = res[name]
        print(f"  {name:8s} n={m.sum():5d}  QTI {r['qti']:.2f} (bias {r['qti_bias']:+.2f})   "
              f"simulator {r['sim']:.2f} (bias {r['sim_bias']:+.2f}, 95% interval covers {100*r['sim_95pct_coverage']:.0f}%)")
    out[f"run{held}"] = res
json.dump(out, open(f"results/heldout_hhs{args.tag}.json", "w"), indent=1)
