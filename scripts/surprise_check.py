"""Calibrated surprise: can any simulated tissue explain a voxel's signal within its noise?

For each prior version:
1. Build a bank of noise-free normalized signals from the prior, through the session's actual
   (motion-corrected) protocol.
2. For each voxel, r = RMS distance to the nearest bank signal, in units of the voxel's noise sd.
   Noise sd comes from the spread of the 15 repeated spherical-encoding b = 1500 volumes, which have
   identical expected signal; S0 from the mean of the b = 0 volumes. Real and simulated voxels are
   processed identically.
3. Calibrate with simulated noisy voxels from the same prior, with SNRs drawn from the real
   per-voxel SNR estimates, *within SNR bins* (the reference distribution of r depends on SNR,
   because a fixed absolute mismatch, or the bank's finite resolution, grows relative to noise as
   SNR rises): a real voxel is "surprising" if its r exceeds the 95th / 99th percentile of r over
   simulated voxels in its SNR bin.

Real noise sd from repeats also absorbs residual motion and physiological noise, which makes this
check conservative (it can only under-flag).
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, ".")
from qtilab import simulate  # noqa: E402
from qtilab.protocol import Protocol  # noqa: E402

torch.set_num_threads(2)
BANK = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
from qtilab import paths  # noqa: E402

a = np.load(paths.aligned_session())
c = np.load(paths.classical_fit())
full = c["mask"]
z = 20
slab = np.zeros_like(full); slab[:, :, z] = True
mask = full & slab
sel = slab[full]
fa = c["fa"][sel]
post = np.load(paths.posterior_slice("_v2"))                   # network posterior, for the f_fw regions
fw = post["mean"][:, list(post["targets"]).index("f_fw")]

shapes = [str(s) for s in a["shapes"]]
prot = Protocol.from_table(a["bvals"], a["bvecs"], shapes)
b = prot.b.numpy()
is_b0 = b < 0.05
rep = (np.array(shapes) == "STE") & (np.abs(b - 1.5) < 0.01)


def normalize(S):
    """S (V, N) -> (normalized signal, sigma relative to S0) using b0 mean and STE-repeat spread."""
    s0 = S[:, is_b0].mean(1, keepdims=True)
    sig = S[:, rep].std(1, ddof=1) / s0[:, 0]
    return S / s0, sig


def nearest_rms(Sn, bank, chunk=256):
    out = []
    for i in range(0, Sn.shape[0], chunk):
        d = torch.cdist(torch.as_tensor(Sn[i:i + chunk], dtype=torch.float32), bank)   # L2 over N
        out.append(d.min(1).values.numpy() / np.sqrt(bank.shape[1]))
    return np.concatenate(out)


S_real = np.maximum(a["data"][mask].astype(np.float64), 1e-4)
Sn_real, sig_real = normalize(S_real)
snr_real = 1.0 / np.maximum(sig_real, 1e-6)
regions = {"WM-like (FA>0.4, f_fw<0.2)": (fa > 0.4) & (fw < 0.2),
           "GM-like (FA<0.2, f_fw<0.2)": (fa < 0.2) & (fw < 0.2),
           "CSF-ish (f_fw>0.5)": fw > 0.5}
out = {"bank": BANK, "slice": z, "versions": {}}
maps = {}
B6 = prot.b6
for ver in (1, 2):
    g = torch.Generator().manual_seed(100 + ver)
    banks = []
    for i in range(0, BANK, 20000):
        d6, _ = simulate.sample_brain(min(20000, BANK - i), M=48, g=g, version=ver)
        banks.append(simulate.signal(d6, B6).float())
    bank = torch.cat(banks)
    # calibration voxels: same prior, SNR drawn from the real voxels' estimates, processed like real data
    nc = 10000
    d6, _ = simulate.sample_brain(nc, M=48, g=g, version=ver)
    clean = simulate.signal(d6, B6)
    snr = torch.as_tensor(np.random.default_rng(ver).choice(np.clip(snr_real, 5, 300), nc))
    S_sim = simulate.rician(clean, 1.0 / snr, g).numpy()
    Sn_sim, sig_sim = normalize(S_sim)
    r_sim = nearest_rms(Sn_sim, bank) / sig_sim
    r_real = nearest_rms(Sn_real, bank) / sig_real
    # SNR-conditional reference: bins by *estimated* SNR, the same quantity known for real voxels
    snr_sim_est = 1.0 / np.maximum(sig_sim, 1e-6)
    edges = np.percentile(snr_real, [0, 20, 40, 60, 80, 100]); edges[0], edges[-1] = 0, np.inf
    bin_real = np.clip(np.searchsorted(edges, snr_real) - 1, 0, 4)
    bin_sim = np.clip(np.searchsorted(edges, snr_sim_est) - 1, 0, 4)
    q95b = np.array([np.percentile(r_sim[bin_sim == k], 95) for k in range(5)])
    q99b = np.array([np.percentile(r_sim[bin_sim == k], 99) for k in range(5)])
    pct = np.array([np.mean(r_sim[bin_sim == bin_real[i]] <= r_real[i]) for i in range(len(r_real))])
    print(f"  per-SNR-bin sim 95th percentile of r: {np.round(q95b, 2)} (bins of estimated SNR, low to high)")
    q95, q99 = q95b[bin_real], q99b[bin_real]
    r_sim_med = np.median(r_sim)
    res = {"r_sim_median": float(r_sim_med), "q95_by_snr_bin": q95b.tolist(), "q99_by_snr_bin": q99b.tolist(),
           "snr_bin_edges": edges.tolist(), "regions": {}}
    print(f"\nprior v{ver}: simulated r median {r_sim_med:.2f}")
    for name, m in regions.items():
        rr, a95, a99 = r_real[m], q95[m], q99[m]
        res["regions"][name] = {"n": int(m.sum()), "r_median": float(np.median(rr)),
                                "frac_gt_q95": float((rr > a95).mean()), "frac_gt_q99": float((rr > a99).mean())}
        print(f"  {name:28s} n={m.sum():5d}  real r median {np.median(rr):.2f}  "
              f"> sim 95th: {100*(rr > a95).mean():.0f}%   > sim 99th: {100*(rr > a99).mean():.0f}%")
    allr = {"r_median": float(np.median(r_real)), "frac_gt_q95": float((r_real > q95).mean()),
            "frac_gt_q99": float((r_real > q99).mean())}
    res["all"] = allr
    print(f"  {'all voxels in slice':28s} n={len(r_real):5d}  real r median {allr['r_median']:.2f}  "
          f"> sim 95th: {100*allr['frac_gt_q95']:.0f}%   > sim 99th: {100*allr['frac_gt_q99']:.0f}%")
    out["versions"][ver] = res
    maps[ver] = pct                                                # SNR-conditional surprise percentile
    del bank
json.dump(out, open("results/surprise_check.json", "w"), indent=1)


def vol(v):
    o = np.full(mask.shape, np.nan); o[mask] = v; return o


fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), facecolor="black")
for ax, (t, v, cm, lim) in zip(axes, [("surprise percentile, prior v1", maps[1], "magma", (0.5, 1)),
                                      ("surprise percentile, prior v2 (widened)", maps[2], "magma", (0.5, 1)),
                                      ("estimated SNR at b=0", snr_real, "viridis", (0, 150))]):
    ax.imshow(np.rot90(vol(v)[:, :, z]), cmap=cm, vmin=lim[0], vmax=lim[1]); ax.set_title(t, color="white"); ax.axis("off")
fig.suptitle("Hormone Health sub-01 ses-01, slice 20: is the signal explainable by any simulated tissue?", color="white")
plt.tight_layout(); plt.savefig("results/hhs_sub01_ses01_surprise.png", dpi=110, facecolor="black")
