"""CPU smoke test of the posterior network on the Hormone Health protocol.

Trains briefly on brain-prior simulations, then compares on held-out simulations:
* network posterior mean vs truth (tissue and full targets), by tissue class;
* classical QTI (WLS and constrained) vs the full-moment truth it targets, and vs tissue truth;
* coverage of the network's 50% and 90% central intervals.
"""
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from qtilab import fit, posterior, scalars, simulate  # noqa: E402
from qtilab.hhs import session_protocol  # noqa: E402

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
torch.set_num_threads(2)
prot = session_protocol("01", "01")
g = torch.Generator().manual_seed(11)
net = posterior.PosteriorNet()
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.05)

t0 = time.perf_counter()
for step in range(1, steps + 1):
    b = posterior.simulate_batch(prot, 512, g=g)
    loss = -net.log_prob(b.x, b.y).mean()
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
    opt.step(); sched.step()
    if step % 250 == 0:
        print(f"step {step}: NLL {loss.item():.3f}  ({time.perf_counter() - t0:.0f} s)", flush=True)
torch.save(net.state_dict(), "results/posterior_smoke_hhs.pt")

# ---- held-out evaluation
net.eval()
gt = torch.Generator().manual_seed(12345)
test = posterior.simulate_batch(prot, 8000, g=gt, snr=(20.0, 35.0))
smp = net.sample(test.x, n=200)                         # (n, V, T)
pm = smp.mean(0).numpy()
y = test.y.numpy()
lo50, hi50 = np.quantile(smp.numpy(), [0.25, 0.75], axis=0)
lo90, hi90 = np.quantile(smp.numpy(), [0.05, 0.95], axis=0)
X = prot.design_matrix()
S64 = test.S.to(torch.float64)
cw = scalars.scalars(fit.wls(X, S64))
cc = scalars.scalars(fit.constrained(X, S64)[0])
T = posterior.TARGETS
cls = test.cls.numpy()
names = simulate.BRAIN_CLASSES
res = {"steps": steps, "train_seconds": time.perf_counter() - t0, "by_class": {}}


def stat(est, tru, m):
    d = est[m] - tru[m]
    return float(np.median(d)), float(np.median(np.abs(d)))


print("\nmedian bias / median |error| by class   (network posterior mean | classical constrained)")
for c, n in enumerate(names):
    m = cls == c
    row = {}
    for k in ["f_fw", "md_t", "ufa_t", "c_md_t", "md", "ufa", "c_md"]:
        i = T.index(k)
        row[f"net_{k}"] = stat(pm[:, i], y[:, i], m)
    for k_full, k_c in [("md", "md"), ("ufa", "ufa"), ("c_md", "c_md")]:
        i = T.index(k_full)
        row[f"qti_{k_c}_vs_full"] = stat(cc[k_c].numpy(), y[:, i], m)
        row[f"qti_{k_c}_vs_tissue"] = stat(cc[k_c].numpy(), y[:, T.index(k_full + '_t')], m)
    res["by_class"][n] = row
    print(f"\n{n}:")
    for k in ["ufa", "md", "c_md"]:
        nb, ne = row[f"net_{k}_t"]
        fb, fe = row[f"net_{k}"]
        qb, qe = row[f"qti_{k}_vs_full"]
        qtb, qte = row[f"qti_{k}_vs_tissue"]
        print(f"  {k:5s} tissue: net {nb:+.3f}/{ne:.3f}  QTI {qtb:+.3f}/{qte:.3f}   |  full: net {fb:+.3f}/{fe:.3f}  QTI {qb:+.3f}/{qe:.3f}")
    fb, fe = row["net_f_fw"]
    print(f"  f_fw  net {fb:+.3f}/{fe:.3f}")

cov50 = ((y >= lo50) & (y <= hi50)).mean(0)
cov90 = ((y >= lo90) & (y <= hi90)).mean(0)
print("\ninterval coverage (nominal 0.50 / 0.90):")
for i, k in enumerate(T):
    print(f"  {k:7s} {cov50[i]:.2f} / {cov90[i]:.2f}   median 90% width {np.median(hi90[:, i] - lo90[:, i]):.3f}")
res["coverage50"] = dict(zip(T, map(float, cov50)))
res["coverage90"] = dict(zip(T, map(float, cov90)))
json.dump(res, open("results/posterior_smoke_hhs.json", "w"), indent=1)
