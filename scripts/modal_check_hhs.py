"""Real-data GPU check on Modal: fit consistency and performance on one HHS session, CUDA vs CPU float64.

Input: work/modal_check_sub01_ses01.npz, packed locally with the masked session signals, the protocol and
CPU float64 references (whole-session constrained parameters, a tight fit of 3,000 validation voxels, and
network embeddings / log-probabilities on 10,000 voxels). On the GPU, in float32 and float64:

* WLS and constrained fits of the whole session, with ADMM and the interior-point solver (``solver="ipm"``):
  time, chunk, iterations, peak memory, and scalar differences vs the CPU float64 ADMM fit (all voxels) and
  vs the tight fit (validation voxels; IPM matches tight ADMM to ~1e-6 in µFA on CPU);
* the v2 posterior network: features, embeddings and log-probabilities vs CPU (deterministic), and
  sampling throughput at several chunk sizes.

    modal run scripts/modal_check_hhs.py          # -> results/modal_check_hhs_sub01_ses01_<gpu>.json

Like modal_smoke.py: GPU fallback list (a single type can sit unscheduled with no local output; check
`modal app list` / `modal app logs <app-id>`), JSON return (the local CLI has no torch).
"""
import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
GPU = ["L4", "A10G", "L40S"]
IGNORE = ["**/__pycache__", "**/*.pyc"]
REPO = "/root/qtilab_repo"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.15.0", "numpy", "nibabel", "zuko>=1.2", "scipy")
    .add_local_dir(ROOT / "qtilab", f"{REPO}/qtilab", ignore=IGNORE)
    .add_local_file(ROOT / "results/posterior_smoke_hhs_v2.pt", f"{REPO}/posterior_smoke_hhs_v2.pt")
    .add_local_file(ROOT / "work/modal_check_sub01_ses01.npz", f"{REPO}/modal_check_sub01_ses01.npz")
)
app = modal.App("qtilab-check-hhs", image=image)
KEYS = ("md", "fa", "ufa", "c_md", "c_c")


def _check():
    import sys
    import time

    import numpy as np
    import torch

    sys.path.insert(0, REPO)
    from qtilab import fit, posterior, scalars
    from qtilab.protocol import Protocol

    d = np.load(f"{REPO}/modal_check_sub01_ses01.npz")
    prot = Protocol.from_table(d["bvals"], d["bvecs"], [str(s) for s in d["shapes"]])
    S64 = torch.as_tensor(np.maximum(d["S"].astype(np.float64), 1e-4))
    idx = torch.as_tensor(d["idx"])
    ref_all = scalars.scalars(torch.as_tensor(d["p_cpu"]))
    ref_tight = scalars.scalars(torch.as_tensor(d["p_tight"]))
    dev = "cuda"
    sync = torch.cuda.synchronize
    out = {"gpu": torch.cuda.get_device_name(0), "torch": str(torch.__version__), "voxels": int(S64.shape[0]),
           "cpu_f64_seconds": {"constrained": float(d["t_cpu_constrained"]), "wls": float(d["t_cpu_wls"])}, "fits": {}}

    def stats(sc, ref, sel=None):
        r = {}
        for k in KEYS:
            x = (sc[k] - ref[k]).abs()
            x = x if sel is None else x[sel]
            r[k] = {"median": float(x.median()), "p99": float(x.quantile(0.99)),
                    "p9999": float(torch.quantile(x, 0.9999)), "max": float(x.max())}
        return r

    for dt in (torch.float32, torch.float64):
        name = str(dt).split(".")[-1]
        X = prot.design_matrix().to(dt).to(dev)
        S = S64.to(dt).to(dev)
        fit.wls(X, S[:2000]); sync()
        t = time.perf_counter(); fit.wls(X, S); sync(); t_wls = time.perf_counter() - t
        for solver in ("admm", "ipm"):
            fit.constrained(X, S[:2000], solver=solver); sync()
            torch.cuda.reset_peak_memory_stats()
            t = time.perf_counter(); p, info = fit.constrained(X, S, solver=solver); sync(); t_con = time.perf_counter() - t
            peak = torch.cuda.max_memory_allocated() / 2**30
            sc = scalars.scalars(p.cpu().double())
            p_v, _ = fit.constrained(X, S[idx.to(dev)], solver=solver)
            sc_v = scalars.scalars(p_v.cpu().double())
            out["fits"][name if solver == "admm" else f"{name}_ipm"] = {
                "solver": solver, "seconds": {"wls": t_wls, "constrained": t_con}, "chunk": fit._chunk(S, None),
                "iterations": info.iterations, "converged_fraction": info.converged_fraction,
                "n_violating": info.n_violating, "peak_memory_gb": peak, "ipm": info.extra,
                "vs_cpu_f64_all_voxels": stats(sc, ref_all),       # the reference is CPU ADMM at default settings
                "vs_f64_tight_validation_voxels": stats(sc_v, ref_tight),
            }

    # network: deterministic parts vs CPU, then sampling throughput
    net = posterior.PosteriorNet()
    net.load_state_dict(torch.load(f"{REPO}/posterior_smoke_hhs_v2.pt", map_location="cpu"))
    net = net.to(dev).eval()
    nidx = torch.as_tensor(d["nidx"])
    S32 = S64.float().to(dev)
    b6, b, bd = (t.float().to(dev) for t in (prot.b6, prot.b, prot.b_delta()))
    x_all = posterior.features(S32, b6, b, bd)
    x_ref = posterior.features(S64[nidx], prot.b6, prot.b, prot.b_delta()).float()
    with torch.no_grad():
        emb = net.embed(x_all[nidx.to(dev)]).cpu()
        lp = net.log_prob(x_all[nidx[:2000].to(dev)], torch.as_tensor(d["y"]).to(dev)).cpu()
    net_out = {
        "features_f32_gpu_vs_f64_cpu_max_abs": float((x_all[nidx.to(dev)].cpu() - x_ref).abs().max()),
        "embedding_max_abs_diff": float((emb - torch.as_tensor(d["emb"])).abs().max()),
        "embedding_scale": float(torch.as_tensor(d["emb"]).abs().max()),
        "log_prob_max_abs_diff": float((lp - torch.as_tensor(d["lp"])).abs().max()),
        "log_prob_median": float(torch.as_tensor(d["lp"]).median()),
        "sampling": [],
    }
    sub = x_all[:20000]
    for chunk in (64, 512, 2048):
        torch.cuda.reset_peak_memory_stats()
        net.sample(sub[:chunk], n=50, chunk=chunk); sync()
        t = time.perf_counter(); net.sample(sub, n=50, chunk=chunk); sync(); dt_ = time.perf_counter() - t
        net_out["sampling"].append({"chunk": chunk, "seconds_20k_voxels_n50": dt_,
                                    "seconds_whole_session_est": dt_ * S64.shape[0] / 20000,
                                    "peak_memory_gb": torch.cuda.max_memory_allocated() / 2**30})
    out["network"] = net_out
    return out


@app.function(gpu=GPU, cpu=4.0, memory=32768, timeout=3600)
def check():
    try:
        return json.dumps(_check())
    except Exception:              # torch exceptions cannot be unpickled by the torch-less local CLI
        import traceback
        return json.dumps({"error": traceback.format_exc()})


@app.local_entrypoint()
def main():
    r = json.loads(check.remote())
    if "error" in r:
        raise SystemExit("remote error:\n" + r["error"])
    print(f"{r['gpu']} (torch {r['torch']}), {r['voxels']} voxels; CPU f64 (M1 Max): "
          f"constrained {r['cpu_f64_seconds']['constrained']:.0f} s, WLS {r['cpu_f64_seconds']['wls']:.2f} s")
    for name, f in r["fits"].items():
        print(f"\n== {name} ({f['solver']}): WLS {f['seconds']['wls']:.2f} s, constrained {f['seconds']['constrained']:.1f} s "
              f"(chunk {f['chunk']}, {f['iterations']} it, converged {f['converged_fraction']:.3f}, peak {f['peak_memory_gb']:.1f} GB)")
        for lab in ("vs_cpu_f64_all_voxels", "vs_f64_tight_validation_voxels"):
            print(f"   {lab}: " + "  ".join(f"{k} {v['p99']:.0e}/{v['max']:.0e}" for k, v in f[lab].items()) + "   (p99/max)")
    n = r["network"]
    print(f"\n== network: features vs CPU f64 max {n['features_f32_gpu_vs_f64_cpu_max_abs']:.1e}; embedding max diff "
          f"{n['embedding_max_abs_diff']:.1e} (scale {n['embedding_scale']:.1f}); log-prob max diff {n['log_prob_max_abs_diff']:.1e} "
          f"(median {n['log_prob_median']:.1f})")
    for s in n["sampling"]:
        print(f"   sample n=50, chunk {s['chunk']:5d}: whole session ~{s['seconds_whole_session_est']:.0f} s, peak {s['peak_memory_gb']:.1f} GB")
    gpu = r["gpu"].replace("NVIDIA ", "").replace(" ", "_")
    path = ROOT / "results" / f"modal_check_hhs_sub01_ses01_{gpu}.json"
    path.write_text(json.dumps(r, indent=1))
    print(f"-> {path}")
