"""Calibration of a posterior network on its training prior: 90% coverage and PIT by target, class, and
whether the simulated voxel's true free-water fraction is exactly 0.

Simulated targets are moments of each voxel's M = 64 sampled micro-tensors, and the free-water fraction is
the share of those samples that landed in the free-water compartment. With a few percent of free water
(WM: 0-5% in v4) none of the 64 lands there in a sizeable share of voxels: true f_fw is then exactly 0
(clamped to the edge of the logit transform) and the full-distribution moments lose their free-water term,
a realization the signal barely constrains. This splits coverage by that case.

    python scripts/calibration.py [MODEL] [--prior 4] [--n 8000] [--exact-fw]
-> results/calibration_<model stem>[_exactfw_test].json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
from qtilab import posterior, simulate  # noqa: E402
from qtilab.hhs import session_protocol  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("model", nargs="?", default="results/posterior_smoke_hhs_v4long.pt")
ap.add_argument("--prior", type=int, default=4)
ap.add_argument("--n", type=int, default=8000)
ap.add_argument("--exact-fw", action="store_true", help="test voxels with exactly weighted free water (true f_fw continuous)")
ap.add_argument("--snr", type=float, nargs=2, default=(20.0, 150.0), help="test SNR range (the training range)")
args = ap.parse_args()
torch.set_num_threads(6)

prot = session_protocol("01", "01")
net = posterior.PosteriorNet()
net.load_state_dict(torch.load(args.model, map_location="cpu"))
net.eval()
b = posterior.simulate_batch(prot, args.n, g=torch.Generator().manual_seed(777), snr=tuple(args.snr),
                             prior_version=args.prior, exact_fw=args.exact_fw)
smp = net.sample(b.x, n=200, chunk=512, generator=torch.Generator().manual_seed(0)).numpy()   # (n, V, T)
y = b.y.numpy()
T = posterior.TARGETS
pit = (smp < y[None]).mean(0)
cov = (pit > 0.05) & (pit < 0.95)
cls = b.cls.numpy()
fw0 = y[:, T.index("f_fw")] < 1e-9

res = {"model": args.model, "exact_fw_test": args.exact_fw, "prior": args.prior, "n": args.n, "snr": list(args.snr),
       "pct_true_f_fw_zero": round(float(fw0.mean() * 100), 1),
       "pct_true_f_fw_zero_by_class": {n: round(float(fw0[cls == c].mean() * 100), 1)
                                       for c, n in enumerate(simulate.BRAIN_CLASSES)},
       "coverage90": {}, "pit_hist_f_fw_positive": {}}
rows = [("all", np.ones_like(fw0)), ("f_fw = 0", fw0), ("f_fw > 0", ~fw0)]
rows += [(f"{n}, f_fw > 0", (cls == c) & ~fw0) for c, n in enumerate(simulate.BRAIN_CLASSES)]
print(f"{args.model}: 90% coverage (prior v{args.prior}, SNR {args.snr[0]:.0f}-{args.snr[1]:.0f}); "
      f"true f_fw = 0 in {fw0.mean():.1%} of voxels")
print(f"  {'':18s} {'n':>5s} " + " ".join(f"{k:>6s}" for k in T))
for lab, m in rows:
    res["coverage90"][lab] = {"n": int(m.sum()), **{k: round(float(cov[m, i].mean()), 3) for i, k in enumerate(T)}}
    print(f"  {lab:18s} {m.sum():5d} " + " ".join(f"{cov[m, i].mean():6.2f}" for i in range(len(T))))
print("PIT histograms, f_fw > 0 (10 bins, 1 = uniform; U-shape = overconfident, hump = underconfident):")
for i, k in enumerate(T):
    h = np.histogram(pit[~fw0, i], bins=10, range=(0, 1))[0] / (~fw0).sum() * 10
    res["pit_hist_f_fw_positive"][k] = h.round(3).tolist()
    print(f"  {k:7s} " + " ".join(f"{x:.2f}" for x in h))
out = f"results/calibration_{Path(args.model).stem}{'_exactfw_test' if args.exact_fw else ''}.json"
json.dump(res, open(out, "w"), indent=1)
print(f"-> {out}")
