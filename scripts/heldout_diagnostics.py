"""Why does white-matter LTE b = 2000 prediction miss? Three diagnostics, real vs simulated.

The held-out predictor averages the K nearest simulated tissues. In anisotropic tissue at high b the
signal depends sharply on fiber direction, and neighbors with slightly different orientations blur
the angular profile. That is an algorithmic error, present even when the tissue model is exactly
right. So every statistic here is computed twice, identically:

  real       HHS session (sub-01 ses-01 by default), fullest axial slice
  simulated  voxels drawn from the same prior (brain v2) through the same protocol, with noise from
             the fitted noise model (thermal SNR and alpha(b) per region sampled from the real slice);
             the tissue model is right by construction, so any error is the predictor's (or noise)

Diagnostics, for held-out run 4 (LTE, b = 2000) and run 8 (STE, b = 1500):
  A. median RMS error / noise sd by region (the headline number), real vs simulated
  B. per-volume structure (run 4): is the error concentrated in a few volumes for all WM voxels?
     reported as the sd across volumes of the WM-median residual (volume-level offsets), real vs sim
  C. residual vs angle between gradient and fiber (run 4, WM): mean z per |cos| bin, real vs sim
  D. resolution: repeat A with a larger bank and fewer neighbors (--bank2, --k2)
"""
import argparse
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from qtilab import fit, noise, paths, scalars, simulate  # noqa: E402
from qtilab.protocol import Protocol  # noqa: E402
from qtilab.tensors import mat3  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sub", default="01"); ap.add_argument("--ses", default="01")
ap.add_argument("--bank", type=int, default=100_000); ap.add_argument("--k", type=int, default=20)
ap.add_argument("--bank2", type=int, default=400_000); ap.add_argument("--k2", type=int, default=5)
ap.add_argument("--nsim", type=int, default=4000)
ap.add_argument("--tag", default="")
ap.add_argument("--series", default="", help="suffix of the series to read, e.g. ec (motion/eddy corrected); mask and regions stay from the uncorrected fit")
args = ap.parse_args()
torch.set_num_threads(2)
rng = np.random.default_rng(0)

a = np.load(paths.aligned_session(args.sub, args.ses + args.series))
c = np.load(paths.classical_fit(args.sub, args.ses))
full = c["mask"]
z = int(np.argmax(full.sum((0, 1))[10:28])) + 10
slab = np.zeros_like(full); slab[:, :, z] = True
mask = full & slab
shapes = np.array([str(s) for s in a["shapes"]])
bvals = a["bvals"]
prot = Protocol.from_table(bvals, a["bvecs"], list(shapes))
X = prot.design_matrix()
run = np.concatenate([[i] * n for i, n in enumerate([13, 10, 16, 30, 2, 12, 9, 15], 1)])
S_real = np.maximum(a["data"][mask].astype(np.float64), 1e-4)
S_real = S_real / S_real[:, bvals < 50].mean(1, keepdims=True)

# ---- noise model on the real slice (all STE repeat groups, per region), used to simulate noise
groups, levels = noise.ste_groups(bvals, shapes, exclude=bvals < 50)
m, v, dof = noise.group_moments(S_real, groups)
fa_r, md_r = c["fa"][slab[full]], c["md"][slab[full]]
reg_r = noise.region_of(fa_r, md_r)
alpha_reg, s2_r = {}, np.zeros(len(S_real))
for r in range(3):
    rr = reg_r == r
    al, s2, _ = noise.fit_alpha_groups(m[rr], v[rr], dof)
    alpha_reg[r] = al
    s2_r[rr] = s2
print(f"slice {z}; alpha(b) by region (WM, GM, CSF): " + "; ".join(f"{np.round(alpha_reg[r], 3).tolist()}" for r in range(3)))

# ---- simulated voxels: prior v2, noise = thermal (sampled from real s2) + alpha(b) of the class's region
g = torch.Generator().manual_seed(77)
d6, cls = simulate.sample_brain(args.nsim, M=48, g=g, version=2)
clean = simulate.signal(d6, prot.b6).numpy()
cls = cls.numpy()
sim_reg = np.where(cls == 0, 0, np.where(cls == 1, 1, -1))           # mixtures assigned below
pw = fit.wls(X, torch.as_tensor(clean))
sc = scalars.scalars(pw)
sim_reg = np.where(sim_reg >= 0, sim_reg, noise.region_of(sc["fa"].numpy(), sc["md"].numpy()))
s2_sim = rng.choice(s2_r, args.nsim)
al_sim = np.stack([noise.alpha_for_bvals(bvals, levels, alpha_reg[r]) for r in sim_reg])     # (V, N)
phys = clean * (1 + al_sim * rng.standard_normal(clean.shape))
n1, n2 = rng.standard_normal(clean.shape), rng.standard_normal(clean.shape)
sd = np.sqrt(s2_sim)[:, None]
S_sim = np.sqrt((phys + sd * n1) ** 2 + (sd * n2) ** 2)
S_sim = S_sim / S_sim[:, bvals < 50].mean(1, keepdims=True)


def regions_of(S):
    p = fit.wls(X, torch.as_tensor(S))
    s = scalars.scalars(p)
    fa, md = s["fa"].numpy(), s["md"].numpy()
    return p, {"WM-like": (fa > 0.4) & (md < 1.2), "GM-like": (fa < 0.2) & (md < 1.2), "CSF-ish": md > 2.0}


def make_bank(n, seed):
    gg = torch.Generator().manual_seed(seed)
    out = []
    for i in range(0, n, 20000):
        dd, _ = simulate.sample_brain(min(20000, n - i), M=48, g=gg, version=2)
        out.append(simulate.signal(dd, prot.b6).float())
    return torch.cat(out)


def predict(S, train, test, bank, k):
    pred = np.zeros((S.shape[0], test.sum()))
    Bt = bank[:, train]
    for i in range(0, S.shape[0], 256):
        d = torch.cdist(torch.as_tensor(S[i:i + 256, train], dtype=torch.float32), Bt)
        idx = d.topk(k, largest=False).indices
        pred[i:i + 256] = bank[:, test][idx].double().mean(1).numpy()
    return pred


def noise_sd(S, test, reg):
    """Thermal + alpha(b) noise sd, fitted without the held-out volumes (as heldout_hhs.py --noise model)."""
    gr, lv = noise.ste_groups(bvals, shapes, exclude=test | (bvals < 50))
    mm, vv, dd = noise.group_moments(S, gr)
    s2 = np.zeros(len(S)); al = np.zeros((len(S), test.sum()))
    for r in range(3):
        rr = reg == r
        if rr.sum() < 20:
            continue
        a_, s2_, _ = noise.fit_alpha_groups(mm[rr], vv[rr], dd)
        s2[rr] = s2_
        al[rr] = noise.alpha_for_bvals(bvals[test], lv, a_)[None, :]
    return np.sqrt(s2[:, None] + (al * S[:, test]) ** 2)


data = {"real": S_real, "sim": S_sim}
fits = {k: regions_of(v) for k, v in data.items()}
nreg = {"real": reg_r, "sim": sim_reg}
out = {"slice": z, "A": {}, "B": {}, "C": {}, "D": {}}
banks = {"main": make_bank(args.bank, 21)}

for held in (4, 8):
    test = run == held
    train = ~test
    print(f"\n=== held out run {held} ({shapes[test][0]}, b = {int(bvals[test][0])}): median RMS error / noise sd")
    for which in ("real", "sim"):
        S = data[which]
        sig = noise_sd(S, test, nreg[which])
        zres = (S[:, test] - predict(S, train, test, banks["main"], args.k)) / sig
        e = np.sqrt(np.mean(zres ** 2, 1))
        regs = fits[which][1]
        row = {r: {"n": int(mm_.sum()), "rms": float(np.median(e[mm_])), "bias": float(np.median(zres[mm_]))}
               for r, mm_ in regs.items()}
        out["A"][f"run{held}_{which}"] = row
        print(f"  {which:4s}: " + "   ".join(f"{r} {v['rms']:.2f} (bias {v['bias']:+.2f}, n={v['n']})" for r, v in row.items()))
        if held == 4:
            wm = regs["WM-like"]
            # B: volume-level structure in WM
            vol_med = np.median(zres[wm], 0)                     # (30,)
            out["B"][which] = {"sd_of_volume_medians": float(vol_med.std()), "max_abs_volume_median": float(np.abs(vol_med).max()),
                               "volume_medians": vol_med.round(3).tolist()}
            # C: residual vs |cos(angle)| between gradient and principal fiber direction (WM)
            D = mat3(fits[which][0][:, 1:7]).numpy()
            w, V = np.linalg.eigh(D)
            fiber = V[..., -1]                                    # (V, 3)
            Bt = prot.B.numpy()[test]
            gdir = np.linalg.eigh(Bt)[1][..., -1]                 # (Ntest, 3)
            cosang = np.abs(fiber @ gdir.T)                        # (V, Ntest)
            bins = [0, 0.3, 0.6, 0.85, 1.0]
            cb = []
            for lo, hi in zip(bins[:-1], bins[1:]):
                sel = (cosang >= lo) & (cosang < hi) & wm[:, None]
                cb.append(float(np.mean(zres[sel])) if sel.any() else float("nan"))
            out["C"][which] = {"bins": bins, "mean_z": cb}
    print(f"  B (run 4, WM): sd of per-volume median residual  real {out['B']['real']['sd_of_volume_medians']:.2f} "
          f"vs sim {out['B']['sim']['sd_of_volume_medians']:.2f}; max |volume median| real "
          f"{out['B']['real']['max_abs_volume_median']:.2f} vs sim {out['B']['sim']['max_abs_volume_median']:.2f}") if held == 4 else None
    if held == 4:
        print("  C (run 4, WM): mean residual by |cos(gradient, fiber)| bin " + str(out["C"]["real"]["bins"]))
        for which in ("real", "sim"):
            print(f"     {which:4s}: " + "  ".join(f"{x:+.2f}" for x in out["C"][which]["mean_z"]))

# D: resolution check, larger bank + fewer neighbors, run 4 only
banks["big"] = make_bank(args.bank2, 22)
test = run == 4
print(f"\n=== D: run 4 with bank {args.bank2} and k = {args.k2} (was {args.bank}, k = {args.k})")
for which in ("real", "sim"):
    S = data[which]
    sig = noise_sd(S, test, nreg[which])
    zres = (S[:, test] - predict(S, ~test, test, banks["big"], args.k2)) / sig
    e = np.sqrt(np.mean(zres ** 2, 1))
    row = {r: float(np.median(e[mm_])) for r, mm_ in fits[which][1].items()}
    out["D"][which] = row
    print(f"  {which:4s}: " + "   ".join(f"{r} {v:.2f}" for r, v in row.items()))
json.dump(out, open(f"results/heldout_diagnostics{args.tag}.json", "w"), indent=1)
