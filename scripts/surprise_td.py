"""Does time-dependent diffusion explain the deep-white-matter surprise? With a calibration control.

Same calibrated surprise statistic as surprise_check.py (nearest simulated signal, in noise units,
SNR-conditional reference). Models compared on sub-01 ses-01, slice 20:

  v2       brain prior v2, time-independent Gaussian compartments
  td       + time dependence: compartment diffusivities scale by 1 + tau (sqrt(omega) - 1);
           linear encoding at omega = 1, spherical at omega = r (the unknown LTE/STE spectral ratio)
  bscale   + calibration error: spherical-encoding b-values off by a factor (1 + eps), affecting
           every compartment including free water

Stage 1 lets r and eps vary per simulated voxel. Because the real waveforms and calibration are the
same in every voxel, a real effect should make the best-matching r (or eps) agree across voxels.
Stage 2 fixes r (or eps) at the white-matter consensus value and recomputes surprise, so each
alternative adds exactly one global parameter.
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
post = np.load(paths.posterior_slice("_v2"))
fw = post["mean"][:, list(post["targets"]).index("f_fw")]
shapes = np.array([str(s) for s in a["shapes"]])
prot = Protocol.from_table(a["bvals"], a["bvecs"], list(shapes))
B6 = prot.b6
b = prot.b.numpy()
is_b0 = b < 0.05
rep = (shapes == "STE") & (np.abs(b - 1.5) < 0.01)
is_ste = torch.as_tensor((shapes == "STE").astype(np.float64))


def normalize(S):
    s0 = S[:, is_b0].mean(1, keepdims=True)
    return S / s0, S[:, rep].std(1, ddof=1) / s0[:, 0]


def nearest(Sn, bank, chunk=256):
    dist, idx = [], []
    for i in range(0, Sn.shape[0], chunk):
        d = torch.cdist(torch.as_tensor(Sn[i:i + chunk], dtype=torch.float32), bank)
        m = d.min(1)
        dist.append(m.values.numpy() / np.sqrt(bank.shape[1]))
        idx.append(m.indices.numpy())
    return np.concatenate(dist), np.concatenate(idx)


def simulate_signals(n, model, g, r=None, eps=None):
    """-> (signals (n, N) float64, per-voxel r, per-voxel eps)."""
    out, rs, es = [], [], []
    for i in range(0, n, 20000):
        k = min(20000, n - i)
        d6, cls, comp = simulate.sample_brain(k, M=48, g=g, version=2, return_comp=True)
        base = torch.einsum("vmk,nk->vmn", d6, B6)
        rv = torch.ones(k, dtype=torch.float64); ev = torch.zeros(k, dtype=torch.float64)
        if model == "td":
            tau = simulate.sample_tau(cls, comp, g)
            rv = (torch.full((k,), float(r), dtype=torch.float64) if r is not None else
                  torch.exp(np.log(0.25) + torch.rand(k, generator=g, dtype=torch.float64) * np.log(16.0)))
            omega = 1.0 + (rv[:, None] - 1.0) * is_ste[None, :]                         # (k, N)
            base = base * (1.0 + tau.unsqueeze(-1) * (torch.sqrt(omega).unsqueeze(1) - 1.0))
        elif model == "bscale":
            ev = (torch.full((k,), float(eps), dtype=torch.float64) if eps is not None else
                  -0.15 + 0.30 * torch.rand(k, generator=g, dtype=torch.float64))
            base = base * (1.0 + ev[:, None, None] * is_ste[None, None, :])
        out.append(torch.exp(-base).mean(1)); rs.append(rv); es.append(ev)
    return torch.cat(out), torch.cat(rs).numpy(), torch.cat(es).numpy()


S_real = np.maximum(a["data"][mask].astype(np.float64), 1e-4)
Sn_real, sig_real = normalize(S_real)
snr_real = 1.0 / np.maximum(sig_real, 1e-6)
edges = np.percentile(snr_real, [0, 20, 40, 60, 80, 100]); edges[0], edges[-1] = 0, np.inf
bin_real = np.clip(np.searchsorted(edges, snr_real) - 1, 0, 4)
regions = {"WM-like": (fa > 0.4) & (fw < 0.2), "GM-like": (fa < 0.2) & (fw < 0.2), "CSF-ish": fw > 0.5}


def surprise(model, seed, r=None, eps=None, nc=10000):
    g = torch.Generator().manual_seed(seed)
    bank, br, be = simulate_signals(BANK, model, g, r, eps)
    bank = bank.float()
    clean, _, _ = simulate_signals(nc, model, g, r, eps)
    snr = torch.as_tensor(np.random.default_rng(seed).choice(np.clip(snr_real, 5, 300), nc))
    Sn_sim, sig_sim = normalize(simulate.rician(clean, 1.0 / snr, g).numpy())
    r_sim = nearest(Sn_sim, bank)[0] / sig_sim
    d_real, i_real = nearest(Sn_real, bank)
    r_real = d_real / sig_real
    bin_sim = np.clip(np.searchsorted(edges, 1.0 / np.maximum(sig_sim, 1e-6)) - 1, 0, 4)
    q95 = np.array([np.percentile(r_sim[bin_sim == k], 95) for k in range(5)])[bin_real]
    q99 = np.array([np.percentile(r_sim[bin_sim == k], 99) for k in range(5)])[bin_real]
    pct = np.array([np.mean(r_sim[bin_sim == bin_real[i]] <= r_real[i]) for i in range(len(r_real))])
    res = {}
    for name, m in regions.items():
        res[name] = {"gt95": float((r_real[m] > q95[m]).mean()), "gt99": float((r_real[m] > q99[m]).mean()),
                     "r_median": float(np.median(r_real[m]))}
    res["all"] = {"gt95": float((r_real > q95).mean()), "gt99": float((r_real > q99).mean())}
    return res, pct, br[i_real], be[i_real]


def show(label, res):
    print(f"{label:30s} " + "   ".join(f"{k} {100*v['gt95']:.0f}%/{100*v['gt99']:.0f}%" for k, v in res.items()), flush=True)


out = {"bank": BANK}
print("percent of real voxels above the simulated 95th / 99th percentile (expected 5% / 1%)")
res_v2, pct_v2, _, _ = surprise("v2", 1)
show("v2 (time-independent)", res_v2); out["v2"] = res_v2

# stage 1: nuisance free per voxel; where do real voxels' nearest matches sit?
res_td, _, r_nn, _ = surprise("td", 2)
show("td, r free per voxel", res_td); out["td_free"] = res_td
res_bs, _, _, e_nn = surprise("bscale", 3)
show("bscale, eps free per voxel", res_bs); out["bscale_free"] = res_bs
wm = regions["WM-like"]; gm = regions["GM-like"]; csf = regions["CSF-ish"]
for name, m in (("WM", wm), ("GM", gm), ("CSF", csf)):
    lr = np.log(r_nn[m])
    print(f"  {name}: nearest-match spectral ratio r median {np.exp(np.median(lr)):.2f} "
          f"(IQR {np.exp(np.percentile(lr, 25)):.2f}-{np.exp(np.percentile(lr, 75)):.2f}); "
          f"b-scale eps median {np.median(e_nn[m]):+.3f} (IQR {np.percentile(e_nn[m], 25):+.3f} to {np.percentile(e_nn[m], 75):+.3f})")
    out[f"r_nn_{name}"] = [float(np.exp(np.median(lr))), float(np.exp(np.percentile(lr, 25))), float(np.exp(np.percentile(lr, 75)))]
    out[f"eps_nn_{name}"] = [float(np.median(e_nn[m])), float(np.percentile(e_nn[m], 25)), float(np.percentile(e_nn[m], 75))]
# prior reference: what the same statistic gives when r is uninformative (log-uniform 0.25-4 -> median 1, IQR 0.5-2)
r_star = float(np.exp(np.median(np.log(r_nn[wm]))))
e_star = float(np.median(e_nn[csf]))       # calibration error is best constrained by free water

# stage 2: one global parameter each
res_td2, pct_td2, _, _ = surprise("td", 4, r=r_star)
show(f"td, r fixed at {r_star:.2f}", res_td2); out["td_fixed"] = {"r": r_star, **res_td2}
res_bs2, pct_bs2, _, _ = surprise("bscale", 5, eps=e_star)
show(f"bscale, eps fixed at {e_star:+.3f}", res_bs2); out["bscale_fixed"] = {"eps": e_star, **res_bs2}
json.dump(out, open("results/surprise_td.json", "w"), indent=1)


def vol(v):
    o = np.full(mask.shape, np.nan); o[mask] = v; return o


fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), facecolor="black")
for ax, (t, v) in zip(axes, [("v2: time-independent", pct_v2), (f"time-dependent, r = {r_star:.2f}", pct_td2),
                             (f"STE b-scale error, eps = {e_star:+.3f}", pct_bs2)]):
    ax.imshow(np.rot90(vol(v)[:, :, z]), cmap="magma", vmin=0.5, vmax=1); ax.set_title(t, color="white"); ax.axis("off")
fig.suptitle("Surprise percentile (SNR-conditional), sub-01 ses-01 slice 20", color="white")
plt.tight_layout(); plt.savefig("results/hhs_sub01_ses01_surprise_td.png", dpi=110, facecolor="black")
