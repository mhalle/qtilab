"""Does what we know from b <= 2 predict the real high-b signal? Cardiff b-tensor data, motion/eddy corrected.

Targets: per voxel, the powder-averaged signal (mean over directions / rotations, normalized by its run's
mean b0) of every LTE, PTE and STE shell at b = 3, 4.5 and 6 ms/µm². Predictors, from the b <= 2 volumes only:

* classical: constrained QTI (dc) fitted to the normalized b <= 2 signal, its cumulant signal predicted at
  each held-out volume and powder-averaged;
* simulator banks: noise-free simulated voxels (brain prior N, exact free water, WM time dependence r = 1.5)
  matched to the voxel by k-NN on rotation-invariant b <= 2 features: the powder averages of every b <= 2
  shell and the l = 2 SH power (relative to the mean) of each LTE and PTE shell. The prediction is the mean
  of the neighbours' held-out powder averages. Rotation invariance avoids the orientation-matching floor of
  heldout_hhs.py's k-NN.

Reported by tissue type (from the b <= 2 classical fit: WM-like FA > 0.4; GM-like FA < 0.2 and
0.5 < MD < 1.2; CSF-like MD > 2): median signed and absolute relative error, (pred - obs) / obs, and the same
in units of the observed powder average's thermal noise SD (sigma / sqrt(directions)).

Noise floor. Observed signals are magnitudes. Every prediction is made per volume and passed through the mean
magnitude at the voxel's noise before powder averaging (``--noise``):
* ``ncchi`` (default): non-central chi with N effective channels. The noise is measured from the STE plateau
  (b = 10.5-12, 57 volumes of the run-aligned series, where the true signal is negligible: CSF and tissue
  plateau at the same level). N is fitted to the plateau mean/SD ratio in CSF-like voxels, and each voxel's
  sigma comes from its plateau SD given N.
* ``rician``: single-channel Rician with sigma = the SD over STE b = 4.5 rotations (the first version; it sets
  the floor too low);
* ``none``: noise-free predictions.
Gradient-nonlinearity check: the WM-like signed errors are also split into the inner and outer thirds of
distance from the brain mask's centroid (b deviates most from nominal away from isocentre).
Each shell's median observed signal / floor (the voxel's mean magnitude at zero signal) is reported, so
floor-dominated shells are visible.

    python scripts/cardiff_extrapolation.py [--sub data1] [--priors 4 2] [--bank 200000] [--k 20]
-> results/cardiff_extrapolation_{sub}.json
"""
import argparse
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from qtilab import fit, paths, posterior, scalars, simulate  # noqa: E402
from qtilab.protocol import Protocol  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sub", default="data1")
ap.add_argument("--priors", nargs="+", default=["4", "2"],
                help="brain prior versions; suffix 'stick' sets intra-axonal D_Delta to 0.99-1, 'sticks' also GM neurites")
ap.add_argument("--bank", type=int, default=200000)
ap.add_argument("--k", type=int, default=20)
ap.add_argument("--noise", default="ncchi", choices=["ncchi", "rician", "none"])
ap.add_argument("--series", default="ec", help="ec (motion/eddy corrected, default) or '' (run-aligned only)")
args = ap.parse_args()
torch.set_num_threads(8)

z = np.load(paths.WORK / f"cardiff_{args.sub}{args.series}_aligned.npz")
mask, b, shapes, run = z["mask"], z["bvals"] / 1000, z["shapes"], z["run_index"]
keep = b <= 6.06
S = z["data"][mask][:, keep].astype(np.float64)
b, shapes, run, bvecs = b[keep], shapes[keep], run[keep], z["bvecs"][:, keep]
prot = Protocol.from_table(b * 1000, bvecs, list(shapes))
s0 = np.stack([S[:, (run == k) & (b < 0.05)].mean(1) for k in range(3)], 1)
good = (s0 > 0).all(1)
S, s0 = S[good], s0[good]
Sn = S / s0[:, run]
V = len(Sn)
lo = b <= 2.06
SHELLS = [(sh, bb) for sh in ("LTE", "PTE", "STE") for bb in sorted(set(np.round(b[(shapes == sh) & (b > 0.05)], 1)))]
sel = {(sh, bb): (shapes == sh) & (np.abs(b - bb) < 0.06) for sh, bb in SHELLS}
low_shells = [s for s in SHELLS if s[1] <= 2.0]
high_shells = [s for s in SHELLS if s[1] > 2.0]


def sh2_basis(v):          # real l = 2 SH (equal norms) at unit vectors v (N, 3)
    x, y, zz = v[:, 0], v[:, 1], v[:, 2]
    return np.stack([np.ones_like(x), x * y, x * zz, y * zz, (x ** 2 - y ** 2) / 2, (3 * zz ** 2 - 1) / (2 * np.sqrt(3))], 1)


def features(sig, dirs):
    """sig (V, N) normalized; dirs (N, 3). -> (V, F) rotation-invariant b <= 2 features."""
    f = []
    for sh, bb in low_shells:
        m = sel[sh, bb]
        pa = sig[:, m].mean(1)
        f.append(pa)
        if sh in ("LTE", "PTE"):
            Y = sh2_basis(dirs[m])
            c = np.linalg.lstsq(Y, sig[:, m].T, rcond=None)[0].T          # (V, 6)
            f.append(np.sqrt((c[:, 1:] ** 2).sum(1)) / np.maximum(np.abs(c[:, 0]), 1e-6))
    return np.stack(f, 1)


def powder(sig):
    return np.stack([sig[:, sel[s]].mean(1) for s in high_shells], 1)


def ncchi_table(N):
    """g(lam) = E sqrt(X), X ~ noncentral chi^2(2N, lam^2): mean magnitude / sigma at signal A = lam sigma."""
    from scipy.stats import ncx2
    lam = np.concatenate([np.linspace(0, 10, 401), np.linspace(10.1, 60, 300)])
    g = np.array([ncx2.expect(np.sqrt, args=(2 * N, max(l * l, 1e-12))) for l in lam])
    return lam, g


def ncchi_mean(A, sig, table):
    lam, g = table
    x = A / sig
    out = np.interp(x, lam, g)
    big = x > lam[-1]
    out[big] = np.sqrt(x[big] ** 2 + 2 * N_EFF - 1)
    return sig * out


def chi_moments(N):       # mean and SD / sigma of the magnitude at zero signal
    from scipy.special import gammaln
    c = np.sqrt(2) * np.exp(gammaln(N + 0.5) - gammaln(N))
    return c, np.sqrt(2 * N - c * c)


def rician_mean(A, sig):
    """E|A + noise| for complex Gaussian noise of SD sig per channel (A >= 0)."""
    from scipy.special import i0e, i1e
    x = -(A ** 2) / (2 * sig ** 2)
    L = (1 - x) * i0e(-x / 2) + (-x) * i1e(-x / 2)          # exp(x/2)-scaled Laguerre L_1/2(x)
    return sig * np.sqrt(np.pi / 2) * L


hi_vol = np.flatnonzero(np.any([sel[s] for s in high_shells], 0))


dirs = bvecs.T / np.maximum(np.linalg.norm(bvecs.T, axis=1, keepdims=True), 1e-9)
obs = powder(Sn)
fe_real = features(Sn, dirs)

# ---- classical fit on b <= 2
t = time.perf_counter()
X_lo = prot.subset(torch.as_tensor(lo)).design_matrix()
pc, _ = fit.constrained(X_lo, torch.as_tensor(np.maximum(Sn[:, lo], 1e-4)))
sc = {k: v.numpy() for k, v in scalars.scalars(pc).items()}
pred_full = fit.predict(prot.design_matrix(), pc).numpy()
print(f"{args.sub}{args.series}: {V} voxels; classical b <= 2 fit + prediction in {time.perf_counter() - t:.0f} s", flush=True)
groups = {"WM-like": sc["fa"] > 0.4, "GM-like": (sc["fa"] < 0.2) & (sc["md"] > 0.5) & (sc["md"] < 1.2),
          "CSF-like": sc["md"] > 2.0}

# ---- noise: the STE plateau (b >= 10.5) of the run-aligned series (same mask and voxel order)
za = np.load(paths.WORK / f"cardiff_{args.sub}_aligned.npz")
assert (za["mask"] == mask).all()
ba, sha, runa = za["bvals"] / 1000, za["shapes"], za["run_index"]
plat = (sha == "STE") & (ba > 10.4)
Pa = za["data"][mask][good][:, plat].astype(np.float64) / s0[:, 2:3]
p_mean, p_sd = Pa.mean(1), Pa.std(1, ddof=1)
csf_like = sc["md"] > 2.5
ratio = float(np.median(p_mean[csf_like] / p_sd[csf_like]))
Ns = np.linspace(1, 16, 1501)
N_EFF = float(Ns[np.argmin(np.abs(np.array([chi_moments(n)[0] / chi_moments(n)[1] for n in Ns]) - ratio))])
c_N, sd_N = chi_moments(N_EFF)
if args.noise == "ncchi":
    sigma_v = np.maximum(p_sd / sd_N, 1e-5)
    table = ncchi_table(N_EFF)
else:
    sigma_v = np.maximum(Sn[:, sel["STE", 4.5]].std(1, ddof=1), 1e-5)
floor_v = sigma_v * (c_N if args.noise == "ncchi" else np.sqrt(np.pi / 2))
print(f"noise: STE plateau mean/SD in CSF-like voxels {ratio:.2f} -> N_eff {N_EFF:.2f}; median sigma {np.median(sigma_v):.4f}, "
      f"floor {np.median(floor_v):.4f} of S0 (plateau mean {np.median(p_mean):.4f}); noise model {args.noise}", flush=True)
noise_pa = sigma_v[:, None] / np.sqrt(np.array([sel[s].sum() for s in high_shells]))[None]


def magnitude(A, sig):
    if args.noise == "ncchi":
        return ncchi_mean(np.clip(A, 0, 10), np.broadcast_to(sig, A.shape), table)
    return rician_mean(np.clip(A, 0, 10), sig)


def to_obs(full):          # per-volume noise-free prediction (V, N) -> predicted observed powder averages
    if args.noise == "none":
        return powder(full)
    out = full.copy()
    out[:, hi_vol] = magnitude(full[:, hi_vol], sigma_v[:, None])
    return powder(out)


preds = {"classical": to_obs(pred_full)}
for pv in args.priors:
    t = time.perf_counter()
    base = int(pv.rstrip("stick"))
    P = dict(simulate.BRAIN_PRIORS[base])
    if pv.endswith("stick") or pv.endswith("sticks"):
        P["ax_dd"] = (0.99, 1.0)
    if pv.endswith("sticks"):
        P["gm_ndd"] = (0.99, 1.0)
    g = torch.Generator().manual_seed(2026 + base)
    bank_f, bank_p = [], []
    for c in range(0, args.bank, 20000):
        sb = posterior.simulate_batch(prot, min(20000, args.bank - c), g=g, snr=(1e7, 1e7), prior_version=P,
                                      dtype=torch.float64, exact_fw=True, td_ratio=1.5)
        sig = sb.S.numpy()
        bank_f.append(features(sig, dirs)); bank_p.append(sig[:, hi_vol].astype(np.float32))
    bank_f, bank_p = np.concatenate(bank_f), np.concatenate(bank_p)
    mu, sd = bank_f.mean(0), bank_f.std(0)
    Bf = torch.as_tensor((bank_f - mu) / sd, dtype=torch.float32)
    Rf = torch.as_tensor((fe_real - mu) / sd, dtype=torch.float32)
    pr = np.empty_like(obs)
    for c in range(0, V, 1000):
        d = torch.cdist(Rf[c:c + 1000], Bf)
        idx = d.topk(args.k, largest=False).indices.numpy()
        nb = bank_p[idx].astype(np.float64)                         # (chunk, k, n_hi)
        if args.noise != "none":
            nb = magnitude(nb, sigma_v[c:c + 1000, None, None])
        full = np.zeros((len(idx), Sn.shape[1]))
        full[:, hi_vol] = nb.mean(1)
        pr[c:c + 1000] = powder(full)
    preds[f"prior v{pv}"] = pr
    print(f"bank v{pv}: {args.bank} voxels simulated and matched in {time.perf_counter() - t:.0f} s", flush=True)

ijk = np.argwhere(mask)[good]
xyz = ijk @ z["affine"][:3, :3].T
rad = np.linalg.norm(xyz - xyz.mean(0), axis=1)
res = {"sub": args.sub, "series": args.series, "noise": args.noise, "n_eff": N_EFF, "plateau_ratio_csf": ratio,
       "sigma_median": float(np.median(sigma_v)), "floor_median": float(np.median(floor_v)), "k": args.k, "bank": args.bank, "n_vox": V,
       "shells": [f"{s}{bb:g}" for s, bb in high_shells], "results": {}}
print("\nmedian relative error % (signed / absolute) and median |error| in noise SD, by shell")
for gname, gm in groups.items():
    fr = np.median(obs[gm] / floor_v[gm, None], 0)
    res["results"][gname] = {"n": int(gm.sum()), "obs_over_floor": dict(zip(res["shells"], fr.round(1).tolist()))}
    print(f"\n{gname} (n = {gm.sum()}):  " + "  ".join(f"{s}{bb:g}" for s, bb in high_shells))
    print(f"  {'obs/floor':10s} " + "  ".join(f"{x:16.1f}" for x in fr))
    for name, p in preds.items():
        rel = (p[gm] - obs[gm]) / np.maximum(obs[gm], 1e-6)
        nz = np.abs(p[gm] - obs[gm]) / np.maximum(noise_pa[gm], 1e-9)
        r = {f"{s}{bb:g}": {"bias_pct": round(float(np.median(rel[:, i]) * 100), 1),
                            "abs_pct": round(float(np.median(np.abs(rel[:, i])) * 100), 1),
                            "abs_noise_sd": round(float(np.median(nz[:, i])), 2)} for i, (s, bb) in enumerate(high_shells)}
        if gname == "WM-like":
            q1, q2 = np.percentile(rad[gm], [33.3, 66.7])
            rr = rad[gm]
            for lab, part in (("inner", rr < q1), ("outer", rr > q2)):
                for i, k in enumerate(r):
                    r[k][f"bias_pct_{lab}"] = round(float(np.median(rel[part, i]) * 100), 1)
        res["results"][gname][name] = r
        print(f"  {name:10s} " + "  ".join(f"{v['bias_pct']:+5.1f}/{v['abs_pct']:4.1f}/{v['abs_noise_sd']:4.1f}" for v in r.values()))
        if gname == "WM-like":
            print(f"  {'  inner/outer':10s} " + "  ".join(f"{v['bias_pct_inner']:+7.1f}/{v['bias_pct_outer']:+6.1f}  " for v in r.values()))
out = f"results/cardiff_extrapolation_{args.sub}{'' if args.noise == 'ncchi' else '_' + args.noise}.json"
json.dump(res, open(out, "w"), indent=1)
print(f"-> {out}")
