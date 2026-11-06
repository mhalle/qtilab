"""FastSurfer (T1) region fractions vs the QTI posterior, on the diffusion grid.

The QTI classes integrate the posterior's marginal Gaussians (mean, SD) over fixed regions, and are
named for what defines them: free_water f_fw > 0.7; mixed 0.3-0.7; and for f_fw < 0.3, low_FA FA_t < 0.15,
mid_FA 0.15-0.4, high_FA > 0.4. They are anisotropy classes, not tissue types: FA_t is the tissue mean
tensor's FA, FA_t^2 = uFA_t^2 * C_c, so they mix microscopic anisotropy with orientation coherence (the
pallidum and thalamus land in mid/high_FA). The FastSurfer regions never see the diffusion data or these
thresholds, so they are an independent check, and "pure" voxels (>= 80% one region) give the regional
FA-uFA scatter and the per-region C_c = (FA/uFA)^2 (from posterior means for the network).

Writes results/fastsurfer_vs_qti_s{sub}{ses}.json (correlations, contingency, regional medians) and
results/fastsurfer_{regional_scatter,vs_qti_maps}_s{sub}{ses}.png.

    python scripts/fastsurfer_vs_qti.py [sub] [ses]
        # needs: posterior_maps_hhs.py --whole-brain, t1_to_dwi.py, fastsurfer_to_dwi.py
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import special  # noqa: E402

sys.path.insert(0, ".")
from qtilab import paths  # noqa: E402

sub = sys.argv[1] if len(sys.argv) > 1 else "01"
ses = sys.argv[2] if len(sys.argv) > 2 else "01"
PURE = 0.8

f = np.load(paths.fastsurfer_fractions(sub, ses))
names = [str(n) for n in f["names"]]
p = np.load(paths.posterior_volume("_v2", sub, ses))
q = np.load(paths.classical_fit(sub, ses))
mask = p["mask"]
assert mask.shape == f["frac"].shape[1:] and (q["mask"] == mask).all()
fsv = {n: f["frac"][i][mask] for i, n in enumerate(names)}
cov = f["covered"][mask]
t = [str(k) for k in p["targets"]]
m, s = p["mean"].astype(float), p["sd"].astype(float)
col = lambda a, k: a[:, t.index(k)]  # noqa: E731
fw, fa_t, ufa_t = col(m, "f_fw"), col(m, "fa_t"), col(m, "ufa_t")
sfw, sfa = col(s, "f_fw"), col(s, "fa_t")


def interval(lo, hi, mu, sd):
    cdf = lambda x: 0.5 * (1 + special.erf((x - mu) / (sd * np.sqrt(2) + 1e-12)))  # noqa: E731
    return cdf(hi) - cdf(lo)


QN = ["free_water", "mixed", "low_FA", "mid_FA", "high_FA"]
tis = interval(0, .3, fw, sfw)
P = np.stack([interval(.7, 1, fw, sfw), interval(.3, .7, fw, sfw), tis * interval(0, .15, fa_t, sfa),
              tis * interval(.15, .4, fa_t, sfa), tis * interval(.4, 1, fa_t, sfa)], 1)
P /= P.sum(1, keepdims=True)
qwin = P.argmax(1)

deep = ["thalamus", "caudate", "putamen", "pallidum", "hippo_amyg", "accumbens_vdc"]
fs_gm = fsv["cortex"] + sum(fsv[n] for n in deep) + fsv["cerebellum_ctx"]
fs_wm = fsv["cerebral_wm"] + fsv["cerebellum_wm"] + fsv["brainstem"]
fs_fluid = fsv["csf_ventricles"] + fsv["background"]
qt_low, qt_aniso = P[:, 2], P[:, 3] + P[:, 4]
res = {"sub": sub, "ses": ses, "voxels": int(mask.sum()), "covered": int(cov.sum()), "pure_threshold": PURE}

r = lambda a, b: float(np.corrcoef(a[cov], b[cov])[0, 1])  # noqa: E731
res["pearson_r"] = {"fs_gm_vs_P_low_FA": r(fs_gm, qt_low), "fs_wm_vs_P_mid_high_FA": r(fs_wm, qt_aniso),
                    "fs_fluid_vs_f_fw": r(fs_fluid, fw),
                    "fs_fluid_vs_P_free_water_mixed": r(fs_fluid, P[:, 0] + P[:, 1])}
print(f"sub-{sub} ses-{ses}: {mask.sum()} posterior voxels, {cov.mean():.3f} covered by the T1 field")
for k, v in res["pearson_r"].items():
    print(f"  r {k:30s} {v:.3f}")

rows = ["cortex", "cerebral_wm"] + deep + ["cerebellum_ctx", "cerebellum_wm", "brainstem",
                                            "csf_ventricles", "background"]
pure = {n: cov & (fsv[n] >= PURE) for n in rows}
res["contingency_row_pct"], res["regional_medians"] = {}, {}
print(f"\nFastSurfer region holding >= {PURE:.0%} of the voxel x QTI winner (row %), and medians")
print(f"  {'region':15s} {'n':>6s} " + " ".join(f"{c[:7]:>7s}" for c in QN)
      + "".join(f"{c:>7s}" for c in ("f_fw", "FA_t", "uFA_t", "Cc_t", "uFA_cl", "FA_cl", "Cc_cl")))
for n in rows:
    sel = pure[n]
    if sel.sum() < 10:
        continue
    pct = np.bincount(qwin[sel], minlength=5) / sel.sum() * 100
    med = {"n": int(sel.sum()), "f_fw": np.median(fw[sel]), "fa_t": np.median(fa_t[sel]),
           "ufa_t": np.median(ufa_t[sel]), "c_c_t": np.median((fa_t[sel] / ufa_t[sel]) ** 2),
           "ufa_classical": np.median(q["ufa"][sel]), "fa_classical": np.median(q["fa"][sel]),
           "c_c_classical": np.median(q["c_c"][sel])}
    res["contingency_row_pct"][n] = dict(zip(QN, pct.round(1).tolist()))
    res["regional_medians"][n] = {k: (v if k == "n" else round(float(v), 4)) for k, v in med.items()}
    print(f"  {n:15s} {sel.sum():6d} " + " ".join(f"{x:7.1f}" for x in pct)
          + "".join(f"{med[k]:7.3f}" for k in ("f_fw", "fa_t", "ufa_t", "c_c_t", "ufa_classical",
                                                "fa_classical", "c_c_classical")))
res["mask_voxels_mostly_fs_background"] = int(pure["background"].sum())
tag = f"s{sub}{ses}"
with open(f"results/fastsurfer_vs_qti_{tag}.json", "w") as fh:
    json.dump(res, fh, indent=1)

# regional FA-uFA scatter: classical fit and network, same voxels
regs = [("cortex", "#2E7D32"), ("cerebral_wm", "#C62828"), ("thalamus", "#6A1B9A"), ("caudate", "#00838F"),
        ("putamen", "#F9A825"), ("pallidum", "#EF6C00"), ("cerebellum_ctx", "#8D6E63"),
        ("cerebellum_wm", "#AD1457"), ("brainstem", "#1565C0")]
big = {"cortex", "cerebral_wm"}
fig, axs = plt.subplots(1, 2, figsize=(13, 5.8), sharex=True, sharey=True)
for ax, (x, y, title) in zip(axs, [(q["fa"], q["ufa"], "Classical constrained QTI (full distribution)"),
                                   (fa_t, ufa_t, "Network posterior mean (tissue moments)")]):
    for n, c in regs:
        sel = pure[n]
        if sel.sum() < 10:
            continue
        ax.scatter(x[sel], y[sel], s=1 if n in big else 3, alpha=0.08 if n in big else 0.5, c=c, rasterized=True)
        ax.scatter(np.median(x[sel]), np.median(y[sel]), s=90, c=c, edgecolors="k", linewidths=1.2, zorder=5)
    ax.plot([0, 1], [0, 1], color="#999", lw=0.8, ls=":")
    ax.set_title(title, fontsize=11); ax.set_xlabel("FA"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
axs[0].set_ylabel("µFA")
for n, c in regs:
    if pure[n].sum() >= 10:
        axs[1].scatter([], [], c=c, s=30, edgecolors="k", label=f"{n} (n={pure[n].sum()})")
axs[1].legend(loc="lower right", fontsize=8, framealpha=0.9)
fig.suptitle(f"sub-{sub} ses-{ses}: voxels ≥ {PURE:.0%} one FastSurfer region (from the T1); "
             "large dots = medians", fontsize=11)
plt.tight_layout(); plt.savefig(f"results/fastsurfer_regional_scatter_{tag}.png", dpi=160, facecolor="white")


def vol(v):
    out = np.full(mask.shape, np.nan, np.float32)
    out[mask] = v
    return out


pairs = [("GM", fs_gm, qt_low, "P(low_FA)"), ("WM", fs_wm, qt_aniso, "P(mid_FA + high_FA)"),
         ("fluid", fs_fluid, fw, "f_fw")]
zs = (np.array([0.42, 0.58]) * mask.shape[2]).astype(int)
fig, axs = plt.subplots(len(zs) * 2, 3, figsize=(10, 13.5))
for zi, z in enumerate(zs):
    for ci, (lab, a, b, qlab) in enumerate(pairs):
        for ri, (img, ttl) in enumerate([(vol(a), f"FastSurfer {lab} fraction"), (vol(b), f"QTI {qlab}")]):
            ax = axs[zi * 2 + ri, ci]
            ax.imshow(img[:, :, z].T, origin="lower", cmap="magma", vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(f"{ttl}, z={z}", fontsize=9); ax.axis("off")
            ax.text(0.02, 0.5, "R", transform=ax.transAxes, color="w", fontsize=7)
plt.tight_layout(); plt.savefig(f"results/fastsurfer_vs_qti_maps_{tag}.png", dpi=130, facecolor="white")
print(f"\nwrote results/fastsurfer_vs_qti_{tag}.json and the two figures")
