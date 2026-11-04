"""Apply the smoke-test posterior network to one real Hormone Health session and render maps.

Inputs: the aligned session saved by the loader (npz with data, bvals, bvecs, shapes) and the
classical-fit results for the same session (mask plus QTI scalars).
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, ".")
from qtilab import posterior  # noqa: E402
from qtilab.protocol import Protocol  # noqa: E402

aligned = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/data/hhs_sub01_ses01_aligned.npz"
classical = sys.argv[2] if len(sys.argv) > 2 else "/home/claude/qtilab/results/hhs_sub01_ses01.npz"
model = sys.argv[3] if len(sys.argv) > 3 else "results/posterior_smoke_hhs.pt"
torch.set_num_threads(2)

a = np.load(aligned)
c = np.load(classical)
full_mask = c["mask"]
z = 20                                   # one slice keeps CPU time to a few minutes
slab = np.zeros_like(full_mask)
slab[:, :, z] = True
mask = full_mask & slab
sel = slab[full_mask]                    # which classical-fit voxels fall in the slice
c = {k: c[k][sel] for k in ("ufa", "c_md", "md", "fa")}
prot = Protocol.from_table(a["bvals"], a["bvecs"], [str(s) for s in a["shapes"]])
S = torch.as_tensor(np.maximum(a["data"][mask], 1e-4).astype(np.float64))
x = posterior.features(S, prot.b6, prot.b, prot.b_delta()).float()

net = posterior.PosteriorNet()
net.load_state_dict(torch.load(model))
net.eval()
smp = net.sample(x, n=100)                                  # (n, V, T)
pm, sd = smp.mean(0).numpy(), smp.std(0).numpy()
T = posterior.TARGETS
np.savez_compressed("results/hhs_sub01_ses01_posterior.npz", mask=mask, mean=pm, sd=sd, targets=np.array(T))


def vol(v):
    o = np.full(mask.shape, np.nan)
    o[mask] = v
    return o


panels = [
    ("free-water fraction (posterior mean)", pm[:, T.index("f_fw")], (0, 1)),
    ("tissue µFA (posterior mean)", pm[:, T.index("ufa_t")], (0, 1)),
    ("tissue µFA posterior SD", sd[:, T.index("ufa_t")], (0, 0.2)),
    ("tissue MD (µm²/ms)", pm[:, T.index("md_t")], (0, 1.5)),
    ("classical QTI µFA (constrained)", c["ufa"], (0, 1)),
    ("classical µFA − network full-distribution µFA", c["ufa"] - pm[:, T.index("ufa")], (-0.2, 0.2)),
    ("tissue C_MD (posterior mean)", pm[:, T.index("c_md_t")], (0, 0.2)),
    ("classical C_MD", c["c_md"], (0, 0.5)),
]
fig, axes = plt.subplots(2, 4, figsize=(15, 8), facecolor="black")
for ax, (title, v, (lo, hi)) in zip(axes.ravel(), panels):
    cmap = "RdBu_r" if lo < 0 else "gray"
    ax.imshow(np.rot90(vol(v)[:, :, z]), cmap=cmap, vmin=lo, vmax=hi)
    ax.set_title(title, color="white", fontsize=9)
    ax.axis("off")
fig.suptitle("Hormone Health sub-01 ses-01: smoke-test posterior network (trained on simulation only, CPU, brain-like prior)",
             color="white", fontsize=10)
plt.tight_layout()
plt.savefig("results/hhs_sub01_ses01_posterior_maps.png", dpi=110, facecolor="black")
print("median posterior SD in mask:", dict(zip(T, np.round(np.median(sd, 0), 3))))
print("median f_fw:", np.median(pm[:, 0]), " fraction of voxels with f_fw > 0.5:", np.mean(pm[:, 0] > 0.5))
