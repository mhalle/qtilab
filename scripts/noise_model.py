"""Estimate a thermal + b-dependent signal-proportional noise model on one HHS session (whole brain).

    var(S) = s2 (per voxel) + alpha(b, region)^2 * S^2

Repeat groups are spherical-encoding volumes at the same b (b = 0 excluded: the voxel normalization
uses those images). Regions from the classical fit: WM (FA > 0.4, MD < 1.2), GM (FA < 0.2, MD < 1.2),
CSF (MD > 2.0). Goodness check: median observed / modeled variance per group, about 1 if adequate.
"""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from qtilab import noise, paths  # noqa: E402

sub = sys.argv[1] if len(sys.argv) > 1 else "01"
ses = sys.argv[2] if len(sys.argv) > 2 else "01"
a = np.load(paths.aligned_session(sub, ses))
c = np.load(paths.classical_fit(sub, ses))
mask = c["mask"]
bvals, shapes = a["bvals"], [str(s) for s in a["shapes"]]
S = np.maximum(a["data"][mask].astype(np.float64), 1e-4)
S = S / S[:, bvals < 50].mean(1, keepdims=True)
groups, levels = noise.ste_groups(bvals, shapes, exclude=bvals < 50)
m, v, dof = noise.group_moments(S, groups)
fa, md = c["fa"], c["md"]
regions = {"WM": (fa > 0.4) & (md < 1.2), "GM": (fa < 0.2) & (md < 1.2), "CSF": md > 2.0}
out = {"sub": sub, "ses": ses, "levels": levels, "dof": dof.tolist(), "regions": {}}
print(f"STE repeat groups at b = {levels} (dof {dof.astype(int).tolist()})")
for name, r in regions.items():
    al, s2, ll = noise.fit_alpha_groups(m[r], v[r], dof)
    s2_th, ll_th = noise.profile_groups(m[r], v[r], dof, np.zeros(len(levels)))
    a1, s2_1, _ = noise.fit_alpha(m[r], v[r], dof)
    _, ll_1 = noise.profile(m[r], v[r], dof, a1)
    model = s2[:, None] + (al[None, :] * m[r]) ** 2
    out["regions"][name] = {
        "n": int(r.sum()), "alpha_by_b": al.tolist(), "alpha_single": a1,
        "snr_thermal_median": float(np.median(1 / np.sqrt(s2))),
        "loglik_gain_vs_thermal": {"single_alpha": float(ll_1.sum() - ll_th.sum()), "alpha_by_b": float(ll - ll_th.sum())},
        "obs_over_model": {"thermal": np.median(v[r] / s2_th[:, None], 0).tolist(),
                           "alpha_by_b": np.median(v[r] / model, 0).tolist()},
        "median_signal": np.median(m[r], 0).tolist()}
    o = out["regions"][name]
    print(f"\n{name}: n={r.sum()}  thermal SNR(b=0) median {o['snr_thermal_median']:.0f}; log-lik gain over thermal: "
          f"single alpha {o['loglik_gain_vs_thermal']['single_alpha']:.0f}, alpha by b {o['loglik_gain_vs_thermal']['alpha_by_b']:.0f}")
    print("   b:                  " + "  ".join(f"{lv:6d}" for lv in levels))
    print("   alpha(b):           " + "  ".join(f"{x:6.3f}" for x in al))
    print("   median signal:      " + "  ".join(f"{x:6.3f}" for x in o["median_signal"]))
    print("   obs/model thermal:  " + "  ".join(f"{x:6.2f}" for x in o["obs_over_model"]["thermal"]))
    print("   obs/model alpha(b): " + "  ".join(f"{x:6.2f}" for x in o["obs_over_model"]["alpha_by_b"]))
json.dump(out, open(f"results/noise_model_sub{sub}_ses{ses}.json", "w"), indent=1)
