"""iRICE + CWLLS (Yu et al., arXiv 2608.27212) against qtilab's QTI fits, using the authors' own package.

The package is imported from a local clone (default work/ext/iRICE-CWLLS, github.com/CollapsarEcho/iRICE-CWLLS,
commit 8cf9c90; --rice-src). Its CWLLS uses MOSEK when licensed; without it, CLARABEL (as here), which the README
calls a fallback that can leave residual violations (they are counted).

Part 1, simulation against known truth (``--sim``):
* protocols: ``rice20``, the paper's 20-encoding LTE+STE protocol (b0; LTE b = 1 x 10 and b = 2 x 6; STE b = 2 x 3,
  from the package's test data), and ``hhs``, the HHS session's 107 volumes (LTE+STE, b to 2);
* tissue: the generic prior (``simulate.Prior()``) and brain prior v4 (WM and GM classes);
* signals: ``exact``, the second-order cumulant of the true moments (ln S = X_QTI theta, no truncation), so only
  the representation differs; ``clean``, the noise-free DTD signal; Rician SNR 50 and 20 at b = 0;
* estimators: iRICE WLLS and CWLLS (D_max 3.1); qtilab QTI WLS and constrained (QTI+ dc);
* truth: the QTI scalars of the true distribution. MD and C_MD have the same definition in both. iRICE's uFA is
  Lasič's µFA (no isotropic variance in the denominator), not QTI's (Westin's); it is scored as ``ufa_l`` against
  that definition of the truth, and qtilab reports both (``ufa`` = Westin, ``ufa_l`` = Lasič, from the same fit).

Part 2, real data (``--hhs``): one prepared HHS session (sub-01 ses-01) fitted by iRICE CWLLS and by qtilab's
constrained fit (CPU float64), with wall times and voxelwise agreement in WM-like (qtilab FA > 0.4) and GM-like
voxels.

    uv run python scripts/rice_compare.py --sim --hhs [--n 3000] [--workers 8]
-> results/rice_compare.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
from qtilab import fit, paths, scalars, simulate  # noqa: E402
from qtilab.protocol import Protocol  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--rice-src", default=str(paths.WORK / "ext/iRICE-CWLLS/src"))
ap.add_argument("--sim", action="store_true")
ap.add_argument("--hhs", action="store_true")
ap.add_argument("--n", type=int, default=3000)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--sub", default="01")
ap.add_argument("--ses", default="01")
args = ap.parse_args()
sys.path.insert(0, args.rice_src)
import irice_cwlls as ir  # noqa: E402

QS = ("md", "ufa", "ufa_l", "c_md")
RMAP = {"md": "MD", "ufa": "uFA", "c_md": "C_MD"}


def lasic_ufa(sc):
    """µFA without the isotropic variance in the denominator (Lasič 2014; iRICE's uFA), from QTI scalars.

    QTI (Westin 2016): µFA² = 1.5 a / (a + 3 (MD² + V_MD)), a = <|dev D|²> over micro-tensors. This one:
    1.5 a / (a + 3 MD²)."""
    c, md2 = sc["c_mu"].clamp(0, 1.4999), sc["md"] ** 2
    a = 3 * c * (md2 + sc["v_md"]) / (1.5 - c)
    return (1.5 * a / (a + 3 * md2).clamp_min(1e-30)).clamp_min(0).sqrt()


def rice_protocol():
    z = np.load(Path(args.rice_src).parent / "tests/data/golden_irice13.npz")
    b, n, bd = z["b_values"], z["directions"], z["b_shapes"]
    shapes = ["LTE" if (s > 0.5 and bb > 0.01) else ("STE" if bb > 0.01 else "LTE") for s, bb in zip(bd, b)]
    return {"b": b, "n": n, "bd": bd, "prot": Protocol.from_table(b * 1000, n, shapes)}


def hhs_protocol():
    z = np.load(paths.aligned_session(args.sub, args.ses))
    b, n, sh = z["bvals"] / 1000, z["bvecs"].T, list(z["shapes"])
    bd = np.array([1.0 if s == "LTE" else 0.0 for s in sh])
    return {"b": b, "n": n, "bd": bd, "prot": Protocol.from_table(z["bvals"], z["bvecs"], sh)}


def run_rice(S, p, constrained=True):
    sig = np.asarray(S, float).reshape(-1, 1, 1, S.shape[-1])
    mask = np.ones(sig.shape[:3], bool)
    kw = dict(signal=sig, b_values=p["b"], directions=p["n"], b_shapes=p["bd"], mask=mask)
    t = time.perf_counter()
    r = ir.fit_cwlls(**kw, config=ir.FitConfig(D_max=3.1, n_workers=args.workers)) if constrained else ir.fit_wlls(**kw)
    dt = time.perf_counter() - t
    out = {k: r.maps[RMAP[k]].ravel() for k in RMAP}
    out["ufa_l"] = out["ufa"]            # iRICE's uFA is the Lasič definition
    out["ufa"] = np.full_like(out["ufa"], np.nan)   # it does not estimate QTI's (Westin) µFA
    diag = {}
    if constrained:
        st = r.diagnostics["status"].ravel()
        diag = {"screened": float(r.diagnostics["screened"].mean()), "residual_violation": float(r.diagnostics["any_violation"].mean()),
                "failed": float((st == -1).mean())}
    return out, dt, diag


def run_qti(S, p, constrained=True):
    X = p["prot"].design_matrix()
    St = torch.as_tensor(np.asarray(S, float))
    t = time.perf_counter()
    prm = fit.constrained(X, St)[0] if constrained else fit.wls(X, St)
    dt = time.perf_counter() - t
    sc = scalars.scalars(prm)
    return {k: sc[k].numpy() for k in RMAP} | {"ufa_l": lasic_ufa(sc).numpy()}, dt, {}


EST = {"irice_wlls": (run_rice, False), "irice_cwlls": (run_rice, True), "qti_wls": (run_qti, False), "qti_constrained": (run_qti, True)}


def summarize(est, tru, sel):
    row = {}
    for k in QS:
        e, t = est[k][sel], tru[k][sel]
        ok = np.isfinite(e)
        if not ok.any():
            continue
        d = e[ok] - t[ok]
        row[k] = {"bias": round(float(np.median(d)), 4), "mae": round(float(np.median(np.abs(d))), 4),
                  "p90": round(float(np.percentile(np.abs(d), 90)), 4), "nonfinite": round(float(1 - ok.mean()), 4)}
    u = est["ufa_l"][sel]
    row["ufa_gt1"] = round(float(np.mean(u[np.isfinite(u)] > 1)), 4)
    row["c_md_lt0"] = round(float(np.mean(est["c_md"][sel][np.isfinite(est["c_md"][sel])] < 0)), 4)
    return row


def simulation(res):
    g = torch.Generator().manual_seed(0)
    tissues = {}
    d6 = simulate.sample_dtd(args.n, 48, simulate.Prior(), g)
    tissues["generic"] = (d6, None)
    d6b, cls = simulate.sample_brain(args.n, 64, g, version=4)
    tissues["brain_v4"] = (d6b, cls.numpy())
    protos = {"rice20": rice_protocol(), "hhs": hhs_protocol()}
    res["sim"] = {}
    for pname, p in protos.items():
        X = p["prot"].design_matrix()
        res["sim"][pname] = {"n_volumes": p["prot"].n, "qti_rank": p["prot"].rank()}
        print(f"\n== protocol {pname}: {p['prot'].n} volumes, QTI design rank {p['prot'].rank()} of 28", flush=True)
        for tname, (d6, cls) in tissues.items():
            truth_p = simulate.dtd_params(d6)
            tsc = scalars.scalars(truth_p)
            tru = {k: tsc[k].numpy() for k in RMAP} | {"ufa_l": lasic_ufa(tsc).numpy()}
            exact = torch.exp(truth_p @ X.T)
            clean = simulate.signal(d6, p["prot"].b6)
            groups = {"all": np.ones(args.n, bool)} if cls is None else {"WM": cls == 0, "GM": cls == 1}
            for sname in ("exact", "clean", "snr50", "snr20"):
                if sname in ("exact", "clean"):
                    S = (exact if sname == "exact" else clean).numpy()
                else:
                    snr = float(sname[3:])
                    S = simulate.rician(clean, torch.full((args.n,), 1.0 / snr), g).numpy()
                for ename, (fn, con) in EST.items():
                    est, dt, diag = fn(S, p, con)
                    for gname, sel in groups.items():
                        row = summarize(est, tru, sel) | {"seconds": round(dt, 2), **diag}
                        res["sim"][pname].setdefault(tname, {}).setdefault(sname, {}).setdefault(gname, {})[ename] = row
                        print(f"{tname:8s} {sname:5s} {gname:3s} {ename:15s} " + "  ".join(
                            f"{k} {row[k]['bias']:+.3f}/{row[k]['mae']:.3f}" for k in QS if k in row)
                            + f"  µFA>1 {row['ufa_gt1']:.3f}  {dt:5.1f} s" + (f"  screened {diag['screened']:.2f}" if diag else ""), flush=True)


def real_hhs(res):
    z = np.load(paths.aligned_session(args.sub, args.ses))
    q = np.load(paths.classical_fit(args.sub, args.ses))
    mask = q["mask"]
    p = hhs_protocol()
    S = z["data"][mask].astype(np.float64)
    S = np.maximum(S, 1e-4 * S.max())
    print(f"\n== HHS sub-{args.sub} ses-{args.ses}: {mask.sum()} voxels, {S.shape[1]} volumes", flush=True)
    out = {"n_vox": int(mask.sum())}
    ests = {}
    for ename in ("irice_wlls", "irice_cwlls", "qti_constrained"):
        fn, con = EST[ename]
        ests[ename], dt, diag = fn(S, p, con)
        out[ename] = {"seconds": round(dt, 1), **diag}
        print(f"{ename}: {dt:.1f} s {diag}", flush=True)
    ref = ests["qti_constrained"]
    fa = q["fa"]
    groups = {"WM-like": fa > 0.4, "GM-like": (fa < 0.2) & (q["md"] > 0.5) & (q["md"] < 1.2)}
    for gname, sel in groups.items():
        out[gname] = {"n": int(sel.sum())}
        for ename in ("irice_wlls", "irice_cwlls", "qti_constrained"):
            e = ests[ename]
            out[gname][ename] = {k: {"median": round(float(np.nanmedian(e[k][sel])), 4),
                                     "abs_diff_vs_qti": round(float(np.nanmedian(np.abs(e[k][sel] - ref[k][sel]))), 4),
                                     "signed_diff_vs_qti": round(float(np.nanmedian(e[k][sel] - ref[k][sel])), 4)}
                                 for k in QS if np.isfinite(e[k][sel]).any()}
            u = e["ufa_l"][sel]
            out[gname][ename]["ufa_gt1"] = round(float(np.mean(u[np.isfinite(u)] > 1)), 4)
            print(f"{gname} {ename:15s} " + "  ".join(
                f"{k} {out[gname][ename][k]['median']:.3f} ({out[gname][ename][k]['signed_diff_vs_qti']:+.3f})" for k in QS
                if k in out[gname][ename])
                + f"  µFA>1 {out[gname][ename]['ufa_gt1']:.4f}", flush=True)
    res["hhs"] = out


if __name__ == "__main__":
    torch.set_num_threads(8)
    res = {"rice_commit": "8cf9c90", "solver": "CLARABEL (no MOSEK licence)", "n_sim": args.n}
    if args.sim:
        simulation(res)
    if args.hhs:
        real_hhs(res)
    json.dump(res, open("results/rice_compare.json", "w"), indent=1)
    print("-> results/rice_compare.json")
