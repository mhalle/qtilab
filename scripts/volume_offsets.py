"""What are the volume-level offsets in white matter at b = 2000? Three explanations, three signatures.

Each diffusion-weighted volume v is predicted from a WLS QTI fit to all the other volumes (leave one
volume out; no k-NN, so no angular blur). The log-domain residual r_vi = ln S_obs - ln S_pred, per
voxel i, is then explained per run by:

  gain      r_vi = a_v                      intensity scaling of the whole volume (all tissue alike)
  b-error   r_vi = eps_v * d_vi             b-value / gradient-amplitude error of that volume:
                                            d_vi = d ln S_pred / d ln b (analytic from the fit), so
                                            the residual scales with how b-sensitive the voxel is,
                                            in GM as well as WM
  angle     r_vi = eta * (cos^2_vi - 1/3)   a model misfit that depends on the angle between the
                                            gradient and the voxel's fiber (WM only, one slope per
                                            run): with real, non-uniform fiber orientations this
                                            makes volume-level WM offsets without any acquisition
                                            error (the simulated null had uniform orientations)

Reported per run (LTE b1000/1500/2000, STE b1000/1500): the sd across volumes of the WM and GM mean
residual (per mille), how much of the per-volume structure each model removes, and, across sessions,
the correlation of the per-volume estimates (a scanner gradient error recurs in every session on the
same gradient table; motion or dropouts do not).

    python scripts/volume_offsets.py [--sessions 01-01 01-02 ...]
"""
import argparse
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from qtilab import fit, paths  # noqa: E402
from qtilab.protocol import Protocol  # noqa: E402
from qtilab.tensors import mat3  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sessions", nargs="*", default=["01-01", "01-02", "01-03", "02-01", "06-01"])
ap.add_argument("--max-vox", type=int, default=60000)
ap.add_argument("--tag", default="")
args = ap.parse_args()
torch.set_num_threads(2)
RUNS = {1: "LTE b100/500", 2: "LTE b1000", 3: "LTE b1500", 4: "LTE b2000", 6: "STE b100/500", 7: "STE b1000", 8: "STE b1500"}
run_of = np.concatenate([[i] * n for i, n in enumerate([13, 10, 16, 30, 2, 12, 9, 15], 1)])
rng = np.random.default_rng(0)


def session(sub, ses):
    a = np.load(paths.aligned_session(sub, ses))
    c = np.load(paths.classical_fit(sub, ses))
    m = c["mask"]
    fa, md = c["fa"], c["md"]
    tissue = (md < 1.2) & ((fa > 0.4) | (fa < 0.2))
    sel = np.flatnonzero(tissue)
    if len(sel) > args.max_vox:
        sel = np.sort(rng.choice(sel, args.max_vox, replace=False))
    wm = fa[sel] > 0.4
    flat = np.flatnonzero(m)                      # mask order -> volume index
    vox = flat[sel]
    data = a["data"].reshape(-1, a["data"].shape[-1])[vox].astype(np.float64)
    bvals = a["bvals"]
    S = np.maximum(data / data[:, bvals < 50].mean(1, keepdims=True), 1e-3)
    shapes = [str(s) for s in a["shapes"]]
    prot = Protocol.from_table(bvals, a["bvecs"], shapes)
    return S, prot, wm, bvals, shapes, a["bvecs"]


def loo(S, prot, v):
    """Predict volume v from the others: log prediction and its derivative in ln b."""
    keep = np.ones(prot.B.shape[0], bool); keep[v] = False
    X = prot.design_matrix()
    p = fit.wls(X[torch.as_tensor(keep)], torch.as_tensor(S[:, keep]))
    x = X[v]
    h = 1e-4
    xh = Protocol(prot.B[v:v + 1] * (1 + h), [prot.shapes[v]]).design_matrix()[0]
    lp = (p @ x).numpy()
    d = ((p @ xh).numpy() - lp) / h
    return lp, d, p


out = {"runs": RUNS, "sessions": {}}
per_volume = {}                                    # (session, run) -> dict of vectors
for sess in args.sessions:
    sub, ses = sess.split("-")
    S, prot, wm, bvals, shapes, bvecs = session(sub, ses)
    gm = ~wm
    p_all = fit.wls(prot.design_matrix(), torch.as_tensor(S))
    fiber = np.linalg.eigh(mat3(p_all[:, 1:7]).numpy())[1][..., -1]
    res = {}
    for run, label in RUNS.items():
        vols = np.flatnonzero((run_of == run) & (bvals >= 50))
        R, Dd, C2 = [], [], []
        for v in vols:
            lp, d, _ = loo(S, prot, v)
            R.append(np.log(S[:, v]) - lp); Dd.append(d)
            gdir = np.linalg.eigh(prot.B[v].numpy())[1][:, -1]
            C2.append((fiber @ gdir) ** 2)
        R, Dd, C2 = np.array(R), np.array(Dd), np.array(C2)          # (n_vol, n_vox)
        w = np.exp(2 * np.clip(np.log(S[:, vols].T), -5, 0))           # log-domain weights ~ S^2
        tis = wm | gm
        Dd = Dd - (Dd * w).sum(1, keepdims=True) / w.sum(1, keepdims=True)   # centered: gain = mean offset, eps = differential
        # angle term: one slope per run, WM only (LTE); STE has no direction
        eta = 0.0
        if label.startswith("LTE"):
            xa = (C2 - 1 / 3) * wm[None, :]
            rc = R - (R * w).sum(1, keepdims=True) / w.sum(1, keepdims=True)    # remove per-volume means
            eta = float((w * xa * rc).sum() / (w * xa * xa).sum())
        Ra = R - eta * (C2 - 1 / 3) * wm[None, :] if label.startswith("LTE") else R
        # per-volume gain + b-error over WM and GM voxels
        a_v, e_v = np.zeros(len(vols)), np.zeros(len(vols))
        Rab = np.empty_like(R)
        for j in range(len(vols)):
            A = np.stack([np.ones(tis.sum()), Dd[j, tis]], 1)
            ww = w[j, tis]
            coef = np.linalg.solve(A.T @ (A * ww[:, None]), A.T @ (ww * Ra[j, tis]))
            a_v[j], e_v[j] = coef
            Rab[j] = Ra[j] - coef[0] - coef[1] * Dd[j]

        def vol_means(Rx, region):
            return 1000 * (Rx[:, region] * w[:, region]).sum(1) / w[:, region].sum(1)

        rows = {}
        for name, Rx in (("raw", R), ("minus angle", Ra), ("minus angle, gain, b-error", Rab)):
            rows[name] = {"WM_sd": float(vol_means(Rx, wm).std()), "GM_sd": float(vol_means(Rx, gm).std())}
        # gain-only and b-only alternatives (after angle), for model comparison
        for name, cols in (("minus angle, gain only", [0]), ("minus angle, b-error only", [1])):
            Rx = np.empty_like(R)
            for j in range(len(vols)):
                A = np.stack([np.ones(tis.sum()), Dd[j, tis]], 1)[:, cols]
                ww = w[j, tis]
                coef = np.linalg.solve(A.T @ (A * ww[:, None]), A.T @ (ww * Ra[j, tis]))
                Rx[j] = Ra[j] - (np.stack([np.ones(len(wm)), Dd[j]], 1)[:, cols] @ coef)
            rows[name] = {"WM_sd": float(vol_means(Rx, wm).std()), "GM_sd": float(vol_means(Rx, gm).std())}
        cwm = (C2[:, wm]).mean(1)
        om = vol_means(R, wm)
        og = vol_means(R, gm)
        half = rng.random(len(wm)) < 0.5                               # noise floor of a volume mean: split-half
        floor = {}
        for rname, reg in (("WM", wm), ("GM", gm)):
            dA, dB = vol_means(R, reg & half), vol_means(R, reg & ~half)
            floor[rname] = float(np.std(dA - dB) / 2)
        res[label] = {"eta_permille": 1000 * eta, "eps_sd": float(e_v.std()), "eps_mean": float(e_v.mean()),
                      "gain_sd_permille": 1000 * float(a_v.std()), "rows": rows,
                      "corr_WM_GM_offsets": float(np.corrcoef(om, og)[0, 1]),
                      "slope_WM_on_GM": float(np.polyfit(og, om, 1)[0]), "noise_floor_volume_mean": floor,
                      "corr_WMoffset_vs_meancos2": float(np.corrcoef(om, cwm)[0, 1]) if label.startswith("LTE") else None}
        per_volume[(sess, label)] = {"eps": e_v, "gain": a_v, "wm_offset": om, "gm_offset": og, "bvecs": bvecs[:, vols], "bvals": bvals[vols], "volumes": vols}
        r4 = res[label]["rows"]
        print(f"{sess} {label}: WM vol-sd {r4['raw']['WM_sd']:.1f} -> angle {r4['minus angle']['WM_sd']:.1f} -> "
              f"+gain&b {r4['minus angle, gain, b-error']['WM_sd']:.1f} (gain only {r4['minus angle, gain only']['WM_sd']:.1f}, "
              f"b only {r4['minus angle, b-error only']['WM_sd']:.1f}); GM {r4['raw']['GM_sd']:.1f} -> {r4['minus angle, gain, b-error']['GM_sd']:.1f}; "
              f"eta {1000 * eta:+.0f}, eps sd {e_v.std():.4f} mean {e_v.mean():+.4f}, gain sd {1000 * a_v.std():.1f}"
              + f"; floor WM {floor['WM']:.1f} GM {floor['GM']:.1f}; corr(WM, GM offsets) {res[label]['corr_WM_GM_offsets']:+.2f} slope {res[label]['slope_WM_on_GM']:.2f}"
              + (f", corr(WM offset, mean cos2) {res[label]['corr_WMoffset_vs_meancos2']:+.2f}" if label.startswith("LTE") else ""), flush=True)
    out["sessions"][sess] = res

# cross-session consistency of per-volume estimates (same gradient table?)
out["cross_session"] = {}
for label in RUNS.values():
    keys = [s for s in args.sessions if (s, label) in per_volume]
    same = min(float(np.abs((per_volume[(keys[0], label)]["bvecs"] * per_volume[(k, label)]["bvecs"]).sum(0)).min())
               for k in keys)
    cs = {}
    for q in ("eps", "gain", "wm_offset"):
        M = np.array([per_volume[(k, label)][q] for k in keys])
        Cm = np.corrcoef(M)
        iu = np.triu_indices(len(keys), 1)
        same_sub = [Cm[i, j] for i, j in zip(*iu) if keys[i][:2] == keys[j][:2]]
        diff_sub = [Cm[i, j] for i, j in zip(*iu) if keys[i][:2] != keys[j][:2]]
        cs[q] = {"same_subject_mean_r": float(np.mean(same_sub)) if same_sub else None,
                 "different_subject_mean_r": float(np.mean(diff_sub)) if diff_sub else None}
    out["cross_session"][label] = {"min_abs_bvec_dot": same, **cs}
    print(f"cross-session {label} (min |bvec dot| {same:.3f}): " + "; ".join(
        f"{q} r same-subj {v['same_subject_mean_r'] if v['same_subject_mean_r'] is None else round(v['same_subject_mean_r'], 2)}, "
        f"diff-subj {round(v['different_subject_mean_r'], 2)}" for q, v in cs.items()), flush=True)
out["per_volume"] = {f"{k[0]}|{k[1]}": {q: np.asarray(v).tolist() for q, v in d.items() if q != "bvecs"} for k, d in per_volume.items()}
json.dump(out, open(f"results/volume_offsets{args.tag}.json", "w"), indent=1)
