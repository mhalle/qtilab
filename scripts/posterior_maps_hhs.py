"""Apply the smoke-test posterior network to one real Hormone Health session and render maps.

Inputs: the aligned session saved by the loader (npz with data, bvals, bvecs, shapes) and the
classical-fit results for the same session (mask plus QTI scalars).

    python scripts/posterior_maps_hhs.py [aligned] [classical] [model] [tag]       # slice 20, CPU (as before)
    python scripts/posterior_maps_hhs.py "" "" results/posterior_smoke_hhs_v2.pt _v2 --whole-brain --device mps

Slice mode (default) samples slice 20 eagerly on CPU and writes paths.posterior_slice, which the held-out
and surprise scripts read. ``--whole-brain`` samples every brain voxel with compiled sampling
(PosteriorNet.sample(compile=True), chunk 512, seeded) on ``--device`` and writes paths.posterior_volume
(``*_full.npz``) and a ``*_full.png`` of slice 20; it does not touch the slice outputs.
Options: --whole-brain, --device DEV (default cpu), --seed N (default 0), --sub SS --ses SS (default 01 01;
used for default inputs and output names).
"""
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, ".")
from qtilab import posterior  # noqa: E402
from qtilab.protocol import Protocol  # noqa: E402

from qtilab import paths  # noqa: E402

args, opts, flags = [], {}, set()
argv = sys.argv[1:]
while argv:
    t = argv.pop(0)
    if t == "--whole-brain":
        flags.add(t)
    elif t in ("--device", "--seed", "--sub", "--ses"):
        opts[t] = argv.pop(0)
    else:
        args.append(t)
whole = "--whole-brain" in flags
device = opts.get("--device", "cpu")
seed = int(opts.get("--seed", 0))
sub, ses = opts.get("--sub", "01"), opts.get("--ses", "01")
aligned = args[0] if len(args) > 0 and args[0] else str(paths.aligned_session(sub, ses))
classical = args[1] if len(args) > 1 and args[1] else str(paths.classical_fit(sub, ses))
model = args[2] if len(args) > 2 and args[2] else "results/posterior_smoke_hhs_v2.pt"
tag = args[3] if len(args) > 3 else ""
torch.set_num_threads(2)

a = np.load(aligned)
c = np.load(classical)
full_mask = c["mask"]
z = 20                                   # slice mode: one slice kept CPU time to a few minutes on 2 cores
slab = np.zeros_like(full_mask)
slab[:, :, z] = True
mask = full_mask if whole else full_mask & slab
sel = np.ones(full_mask.sum(), bool) if whole else slab[full_mask]   # which classical-fit voxels are used
c = {k: c[k][sel] for k in ("ufa", "c_md", "md", "fa")}
prot = Protocol.from_table(a["bvals"], a["bvecs"], [str(s) for s in a["shapes"]])
t0 = time.perf_counter()
S = torch.as_tensor(np.maximum(a["data"][mask], 1e-4).astype(np.float64))
x = posterior.features(S, prot.b6, prot.b, prot.b_delta()).float()
t_feat = time.perf_counter() - t0

net = posterior.PosteriorNet()
net.load_state_dict(torch.load(model, map_location="cpu"))
net.eval()
T = posterior.TARGETS
paths.WORK.mkdir(parents=True, exist_ok=True)
t0 = time.perf_counter()
if whole:
    net = net.to(device)
    smp = net.sample(x.to(device), n=100, chunk=512, compile=True,
                     generator=torch.Generator(device=device).manual_seed(seed))
    pm, sd = smp.mean(0).cpu().numpy(), smp.std(0).cpu().numpy()
    del smp
else:
    smp = net.sample(x, n=100)                              # (n, V, T)
    pm, sd = smp.mean(0).numpy(), smp.std(0).numpy()
t_sample = time.perf_counter() - t0
print(f"{int(mask.sum())} voxels on {device if whole else 'cpu'}: features {t_feat:.1f} s, sampling "
      f"{t_sample:.1f} s{' (compiled; includes compilation on first use)' if whole else ''}", flush=True)
if whole:
    np.savez_compressed(paths.posterior_volume(tag or "_v1", sub, ses), mask=mask, mean=pm, sd=sd,
                        targets=np.array(T), seed=seed, device=device)
else:
    np.savez_compressed(paths.posterior_slice(tag or "_v1"), mask=mask, mean=pm, sd=sd, targets=np.array(T))


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
    im = ax.imshow(np.rot90(vol(v)[:, :, z]), cmap=cmap, vmin=lo, vmax=hi)
    ax.set_title(title, color="white", fontsize=9)
    ax.axis("off")
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.046, pad=0.02, ticks=[lo, (lo + hi) / 2, hi])
    cb.ax.tick_params(colors="white", labelsize=8)
    cb.outline.set_edgecolor("#888")
fig.suptitle(f"Hormone Health sub-{sub} ses-{ses}: posterior network {tag or 'v1'} (trained on simulation only, "
             f"{'whole brain, slice ' + str(z) + ' shown' if whole else 'CPU'})", color="white", fontsize=10)
plt.tight_layout()
plt.savefig(f"results/hhs_sub{sub}_ses{ses}_posterior_maps{tag}{'_full' if whole else ''}.png", dpi=110, facecolor="black")
print("median posterior SD in mask:", dict(zip(T, np.round(np.median(sd, 0), 3))))
print("median f_fw:", np.median(pm[:, 0]), " fraction of voxels with f_fw > 0.5:", np.mean(pm[:, 0] > 0.5))
fw, fa_c = pm[:, T.index("f_fw")], c["fa"]
for name, m in [("WM-like (FA>0.4, f_fw<0.2)", (fa_c > 0.4) & (fw < 0.2)), ("GM-like (FA<0.2, f_fw<0.2)", (fa_c < 0.2) & (fw < 0.2)),
                ("CSF-ish (f_fw>0.5)", fw > 0.5)]:
    print(f"{name:28s} n={m.sum():5d}  classical µFA {np.median(c['ufa'][m]):.2f}  net full µFA {np.median(pm[m, T.index('ufa')]):.2f}"
          f"  net tissue µFA {np.median(pm[m, T.index('ufa_t')]):.2f}  classical - net full {np.median(c['ufa'][m] - pm[m, T.index('ufa')]):+.2f}"
          f"  tissue MD {np.median(pm[m, T.index('md_t')]):.2f}")
