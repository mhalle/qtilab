"""Why the network's tissue FA_t sits below classical FA on real data: the brain prior's FA_t gap.

1. Simulated voxels (brain prior, through the session's protocol, Rician noise at SNR 30-140): bias of
   the network's posterior-mean FA_t and of the classical constrained FA against the truth.
2. The prior's tissue FA_t histogram against the real classical FA histogram (tissue voxels, f_fw < 0.3).
3. Network FA_t as a function of classical FA, simulated vs real.
4. With --with-v3: regional FA_t / uFA_t / f_fw for the v2 and v3 networks in FastSurfer regions (needs
   the _v3_full posterior and fastsurfer_to_dwi.py's fractions).

Writes results/prior_fa_gap_s{sub}{ses}.json and .png.

    python scripts/prior_fa_gap.py [sub] [ses] [--device mps] [--with-v3]     # ~1 min on MPS
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, ".")
from qtilab import fit, paths, posterior, scalars, simulate  # noqa: E402
from qtilab.protocol import Protocol  # noqa: E402

argv = sys.argv[1:]
device = argv.pop(argv.index("--device") + 1) if "--device" in argv else "cpu"
argv = [a for a in argv if a != "--device"]
with_v3 = "--with-v3" in argv
argv = [a for a in argv if a != "--with-v3"]
sub = argv[0] if len(argv) > 0 else "01"
ses = argv[1] if len(argv) > 1 else "01"
V, SNR = 20000, (30.0, 140.0)
T = posterior.TARGETS
BINS = [0, .1, .2, .3, .4, .5, .6, .8, 1.01]
res = {"sub": sub, "ses": ses, "simulated_voxels": V, "snr": SNR}

a = np.load(paths.aligned_session(sub, ses))
prot = Protocol.from_table(a["bvals"], a["bvecs"], [str(s) for s in a["shapes"]])
g = torch.Generator().manual_seed(1)
d6, cls, free = simulate.sample_brain(V, M=64, g=g, return_free=True, version=2)
u = torch.rand(V, dtype=d6.dtype, generator=g)
snr = torch.exp(np.log(SNR[0]) + u * np.log(SNR[1] / SNR[0]))
S = simulate.rician(simulate.signal(d6, prot.b6.to(d6.dtype)), 1.0 / snr, g)
y = posterior.targets(d6, free).numpy()
true_fa_t, true_ufa_t = y[:, T.index("fa_t")], y[:, T.index("ufa_t")]
true_fa = scalars.scalars(simulate.dtd_params(d6))["fa"].numpy()
cl = {k: v.numpy() for k, v in scalars.scalars(fit.constrained(prot.design_matrix(), S)[0]).items()}
net = posterior.PosteriorNet()
net.load_state_dict(torch.load("results/posterior_smoke_hhs_v2.pt", map_location="cpu"))
net = net.eval().to(device)
x = posterior.features(S, prot.b6, prot.b, prot.b_delta()).float().to(device)
pm = net.sample(x, n=100, chunk=512, compile=device != "cpu",
                generator=torch.Generator(device=device).manual_seed(0)).mean(0).cpu().numpy()
net_fa_t, net_ufa_t = pm[:, T.index("fa_t")], pm[:, T.index("ufa_t")]
cls = cls.numpy()

print("1) Simulated voxels, medians (truth known)")
print(f"   {'class':7s} {'true FA_t':>9s} {'net-true':>9s} {'cl FA-true FA_t':>15s} {'cl FA-true FA':>13s} "
      f"{'true uFA_t':>10s} {'net-true':>9s} {'cl uFA-true':>11s}")
res["simulated"] = {}
for c, name in enumerate(simulate.BRAIN_CLASSES):
    s = cls == c
    md = lambda v: float(np.median(v[s]))  # noqa: E731
    r = {"n": int(s.sum()), "true_fa_t": md(true_fa_t), "net_minus_true_fa_t": md(net_fa_t - true_fa_t),
         "cl_fa_minus_true_fa_t": md(cl["fa"] - true_fa_t), "cl_fa_minus_true_fa": md(cl["fa"] - true_fa),
         "true_ufa_t": md(true_ufa_t), "net_minus_true_ufa_t": md(net_ufa_t - true_ufa_t),
         "cl_ufa_minus_true_ufa_t": md(cl["ufa"] - true_ufa_t)}
    res["simulated"][name] = r
    print(f"   {name:7s} {r['true_fa_t']:9.3f} {r['net_minus_true_fa_t']:+9.3f} {r['cl_fa_minus_true_fa_t']:+15.3f} "
          f"{r['cl_fa_minus_true_fa']:+13.3f} {r['true_ufa_t']:10.3f} {r['net_minus_true_ufa_t']:+9.3f} "
          f"{r['cl_ufa_minus_true_ufa_t']:+11.3f}")

p = np.load(paths.posterior_volume("_v2", sub, ses))
q = np.load(paths.classical_fit(sub, ses))
fw, fa_t = p["mean"][:, T.index("f_fw")], p["mean"][:, T.index("fa_t")]
tis_sim, tis_real = cls < 2, fw < 0.3
h_prior = np.histogram(true_fa_t[tis_sim], BINS)[0] / tis_sim.sum() * 100
h_real = np.histogram(q["fa"][tis_real], BINS)[0] / tis_real.sum() * 100
h_other = {}
for v in (1, 3):                                   # the other brain priors, same seed and size; truth only
    d6v, clsv, freev = simulate.sample_brain(V, M=64, g=torch.Generator().manual_seed(1), return_free=True, version=v)
    fav = posterior.targets(d6v, freev).numpy()[:, T.index("fa_t")]
    tv = clsv.numpy() < 2
    h_other[v] = np.histogram(fav[tv], BINS)[0] / tv.sum() * 100
res["fa_histogram_pct"] = {"bins": BINS, "prior_v2_true_fa_t": h_prior.round(2).tolist(),
                           "prior_v1_true_fa_t": h_other[1].round(2).tolist(),
                           "prior_v3_true_fa_t": h_other[3].round(2).tolist(),
                           "real_classical_fa": h_real.round(2).tolist()}
gap = lambda h: float(h[3] + h[4])  # noqa: E731   FA 0.3-0.5
res["pct_tissue_fa_0.3_0.5"] = {"prior_v1": gap(h_other[1]), "prior_v2": gap(h_prior), "prior_v3": gap(h_other[3]),
                                "real_classical": gap(h_real)}
print("\n2) Tissue FA, % per bin:  bin    prior v1   prior v2   prior v3   real classical FA")
for i in range(len(BINS) - 1):
    print(f"   [{BINS[i]:.1f},{min(BINS[i + 1], 1):.1f})  {h_other[1][i]:9.1f} {h_prior[i]:10.1f} {h_other[3][i]:10.1f} "
          f"{h_real[i]:18.1f}")
print("   FA 0.3-0.5:  " + "  ".join(f"{k} {v:.1f}%" for k, v in res["pct_tissue_fa_0.3_0.5"].items()))

print("\n3) Median network FA_t by classical FA bin: simulated vs real")
res["net_fa_t_by_classical_fa"] = []
for lo, hi in [(0, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .5), (.5, .6), (.6, .7), (.7, .9)]:
    sa = tis_sim & (cl["fa"] >= lo) & (cl["fa"] < hi)
    sb = tis_real & (q["fa"] >= lo) & (q["fa"] < hi)
    sim = float(np.median(net_fa_t[sa])) if sa.sum() > 20 else None
    real = float(np.median(fa_t[sb]))
    res["net_fa_t_by_classical_fa"].append({"bin": [lo, hi], "simulated": sim, "real": real, "n_real": int(sb.sum())})
    print(f"   [{lo:.1f},{hi:.1f})  simulated {('%.3f' % sim) if sim is not None else '  -  '}  real {real:.3f}  (n {sb.sum()})")

if with_v3:
    f = np.load(paths.fastsurfer_fractions(sub, ses))
    names = [str(n) for n in f["names"]]
    p3 = np.load(paths.posterior_volume("_v3", sub, ses))
    mask = p["mask"]
    assert (p3["mask"] == mask).all()
    res["regions_v2_v3"] = {}
    print("\n4) FastSurfer regions (>= 80%), medians: classical | v2 | v3")
    print(f"   {'region':15s} {'FA':>6s} {'FA_t':>6s} {'FA_t':>6s} | {'uFA':>6s} {'uFA_t':>6s} {'uFA_t':>6s} | "
          f"{'f_fw':>6s} {'f_fw':>6s}")
    for r in ["cortex", "caudate", "putamen", "thalamus", "cerebral_wm", "cerebellum_wm", "pallidum", "brainstem"]:
        s = f["frac"][names.index(r)][mask] >= 0.8
        md = lambda v: float(np.median(v[s]))  # noqa: E731
        c2 = lambda k: md(p["mean"][:, T.index(k)])  # noqa: E731
        c3 = lambda k: md(p3["mean"][:, T.index(k)])  # noqa: E731
        row = {"n": int(s.sum()), "cl_fa": md(q["fa"]), "v2_fa_t": c2("fa_t"), "v3_fa_t": c3("fa_t"),
               "cl_ufa": md(q["ufa"]), "v2_ufa_t": c2("ufa_t"), "v3_ufa_t": c3("ufa_t"),
               "v2_f_fw": c2("f_fw"), "v3_f_fw": c3("f_fw")}
        res["regions_v2_v3"][r] = row
        print(f"   {r:15s} {row['cl_fa']:6.3f} {row['v2_fa_t']:6.3f} {row['v3_fa_t']:6.3f} | {row['cl_ufa']:6.3f} "
              f"{row['v2_ufa_t']:6.3f} {row['v3_ufa_t']:6.3f} | {row['v2_f_fw']:6.3f} {row['v3_f_fw']:6.3f}")

tag = f"s{sub}{ses}"
with open(f"results/prior_fa_gap_{tag}.json", "w") as fh:
    json.dump(res, fh, indent=1)
fig, axs = plt.subplots(1, 3, figsize=(17, 5))
centers = np.array([(BINS[i] + min(BINS[i + 1], 1)) / 2 for i in range(len(BINS) - 1)])
widths = np.diff(np.minimum(BINS, 1))
axs[0].bar(centers - widths * 0.2, h_prior / widths, width=widths * 0.4, color="#7E57C2", label="prior v2: true tissue FA_t")
axs[0].bar(centers + widths * 0.2, h_real / widths, width=widths * 0.4, color="#555", label="real: classical FA")
axs[0].set_xlabel("FA"); axs[0].set_ylabel("% of tissue voxels per unit FA"); axs[0].legend(fontsize=9)
axs[0].set_title("The prior has almost no tissue at FA 0.3-0.5")
for ax, (xx, yy, ttl, c) in zip(axs[1:], [(cl["fa"][tis_sim], net_fa_t[tis_sim], "Simulated (prior v2)", np.where(cls[tis_sim] == 0, "#C62828", "#2E7D32")),
                                          (q["fa"][tis_real], fa_t[tis_real], f"Real sub-{sub} ses-{ses}", "#555")]):
    ax.scatter(xx, yy, s=1, alpha=0.1, c=c, rasterized=True)
    ax.plot([0, 1], [0, 1], ":", c="#999"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("classical constrained FA"); ax.set_ylabel("network posterior-mean FA_t"); ax.set_title(ttl)
plt.tight_layout(); plt.savefig(f"results/prior_fa_gap_{tag}.png", dpi=140, facecolor="white")
print(f"\nwrote results/prior_fa_gap_{tag}.json and .png")
