"""Where posterior sampling spends its time on a CUDA GPU, and what torch.compile buys.

Sampling the v2 network's flow (zuko NSF: 4 masked autoregressive transforms x 8 targets, rational-quadratic
splines) inverts each transform one dimension at a time: 32 sequential passes of MLP + spline rebuild +
spline inverse, about 10k small PyTorch ops per chunk. Locally (MPS, CPU) time per sample is flat from
chunk 256 up, so it is throughput-bound, not launch-bound. This measures on CUDA, on identical inputs
(fixed base noise, so outputs are comparable value for value):

* eager at chunk 512 and 2048, with a CUDA kernel profile (kernel count, GPU time vs wall time, top kernels);
* torch.compile (Inductor -> Triton kernels; default and max-autotune): compile time, speed, max |difference|.

    modal run scripts/modal_profile_flow.py                  # profile -> results/modal_profile_flow_<gpu>.json
    modal run scripts/modal_profile_flow.py --what bench     # scripts/bench_flow.py on CUDA -> results/bench_flow_modal_<gpu>.json
Uses work/modal_check_sub01_ses01.npz (packed for modal_check_hhs.py) for real voxels.
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
    .apt_install("build-essential")                       # Triton (torch.compile on CUDA) needs a C compiler
    .pip_install("torch==2.15.0", "numpy", "nibabel", "zuko>=1.2", "scipy")
    .add_local_dir(ROOT / "qtilab", f"{REPO}/qtilab", ignore=IGNORE)
    .add_local_file(ROOT / "results/posterior_smoke_hhs_v2.pt", f"{REPO}/results/posterior_smoke_hhs_v2.pt")
    .add_local_dir(ROOT / "scripts", f"{REPO}/scripts", ignore=IGNORE)
    .add_local_file(ROOT / "work/modal_check_sub01_ses01.npz", f"{REPO}/modal_check_sub01_ses01.npz")
)
app = modal.App("qtilab-profile-flow", image=image)
N_SAMPLES, VOXELS = 50, 20480


def _profile():
    import sys
    import time

    import numpy as np
    import torch
    from torch.profiler import ProfilerActivity, profile

    sys.path.insert(0, REPO)
    from qtilab import posterior
    from qtilab.protocol import Protocol

    dev = "cuda"
    sync = torch.cuda.synchronize
    d = np.load(f"{REPO}/modal_check_sub01_ses01.npz")
    prot = Protocol.from_table(d["bvals"], d["bvecs"], [str(s) for s in d["shapes"]])
    S = torch.as_tensor(np.maximum(d["S"][:VOXELS].astype(np.float32), 1e-4)).to(dev)
    b6, b, bd = (t.float().to(dev) for t in (prot.b6, prot.b, prot.b_delta()))
    net = posterior.PosteriorNet()
    net.load_state_dict(torch.load(f"{REPO}/results/posterior_smoke_hhs_v2.pt", map_location="cpu"))
    net = net.to(dev).eval()
    torch.set_grad_enabled(False)
    x = posterior.features(S, b6, b, bd)
    ctx_all = net.embed(x)                                      # (V, 64)
    g = torch.Generator(device=dev).manual_seed(0)
    z_all = torch.randn(N_SAMPLES, VOXELS, 8, generator=g, device=dev)
    out = {"gpu": torch.cuda.get_device_name(0), "torch": str(torch.__version__), "voxels": VOXELS, "n": N_SAMPLES,
           "eager": {}, "compiled": {}}

    def inv(ctx, z):
        return net.flow(ctx).transform.inv(z)

    def run(fn, chunk, reps=1):
        ys = []
        sync(); t = time.perf_counter()
        for _ in range(reps):
            ys = [fn(ctx_all[i:i + chunk], z_all[:, i:i + chunk]) for i in range(0, VOXELS, chunk)]
        sync()
        return torch.cat(ys, 1), (time.perf_counter() - t) / reps

    ref = {}
    for chunk in (512, 2048):
        run(inv, chunk)
        y, t = run(inv, chunk, reps=2)
        ref[chunk] = y
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
            inv(ctx_all[:chunk], z_all[:, :chunk]); sync()
        ev = prof.key_averages()
        kern = [e for e in ev if e.device_time_total > 0 and not e.key.startswith(("aten::", "cuda", "Memcpy", "Memset"))]
        n_k = sum(e.count for e in kern)
        gpu_us = sum(e.self_device_time_total for e in kern)
        t1 = time.perf_counter(); inv(ctx_all[:chunk], z_all[:, :chunk]); sync(); wall_one = time.perf_counter() - t1
        top = sorted(kern, key=lambda e: -e.self_device_time_total)[:8]
        out["eager"][str(chunk)] = {
            "seconds_20k_voxels": t, "us_per_sample": 1e6 * t / (VOXELS * N_SAMPLES),
            "whole_session_est_s": t * 95801 / VOXELS, "kernels_per_chunk": n_k,
            "gpu_kernel_ms_per_chunk": gpu_us / 1e3, "wall_ms_per_chunk": 1e3 * wall_one,
            "gpu_busy_fraction": (gpu_us / 1e6) / wall_one,
            "top_kernels": [{"name": e.key[:90], "count": e.count, "ms": e.self_device_time_total / 1e3} for e in top],
        }

    for mode in ("default", "max-autotune-no-cudagraphs"):
        try:
            torch._dynamo.reset()
            cf = torch.compile(inv, mode=mode, dynamic=False)
            t0 = time.perf_counter(); cf(ctx_all[:2048], z_all[:, :2048]); sync(); t_compile = time.perf_counter() - t0
            res = {"compile_seconds": t_compile}
            for chunk in (2048,):
                run(cf, chunk)
                y, t = run(cf, chunk, reps=2)
                diff = (y - ref[chunk]).abs()
                res[str(chunk)] = {"seconds_20k_voxels": t, "us_per_sample": 1e6 * t / (VOXELS * N_SAMPLES),
                                   "whole_session_est_s": t * 95801 / VOXELS,
                                   "speedup_vs_eager": out["eager"][str(chunk)]["seconds_20k_voxels"] / t,
                                   "max_abs_diff_vs_eager": float(diff.max()),
                                   "frac_diff_gt_1e-4": float((diff > 1e-4).float().mean())}
            out["compiled"][mode] = res
        except Exception as e:                                  # noqa: BLE001
            out["compiled"][mode] = {"error": f"{type(e).__name__}: {str(e)[:500]}"}
    return out


@app.function(gpu=GPU, cpu=4.0, memory=32768, timeout=3600)
def profile_flow():
    try:
        return json.dumps(_profile())
    except Exception:              # torch exceptions cannot be unpickled by the torch-less local CLI
        import traceback
        return json.dumps({"error": traceback.format_exc()})


@app.function(gpu=GPU, cpu=4.0, memory=32768, timeout=3600)
def bench():
    try:
        import sys
        sys.path[:0] = [REPO, f"{REPO}/scripts"]
        import bench_flow
        r = bench_flow.run("cuda")
        return json.dumps({"result": r, "summary": bench_flow.summary(r)})
    except Exception:
        import traceback
        return json.dumps({"error": traceback.format_exc()})


@app.local_entrypoint()
def main(what: str = "profile"):
    if what == "bench":
        r = json.loads(bench.remote())
        if "error" in r:
            raise SystemExit("remote error:\n" + r["error"])
        print(r["summary"])
        gpu = r["result"]["gpu"].replace("NVIDIA ", "").replace(" ", "_")
        path = ROOT / "results" / f"bench_flow_modal_{gpu}.json"
        path.write_text(json.dumps([r["result"]], indent=1))
        print(f"-> {path}")
        return
    r = json.loads(profile_flow.remote())
    if "error" in r:
        raise SystemExit("remote error:\n" + r["error"])
    print(f"{r['gpu']} (torch {r['torch']}): {r['voxels']} voxels x {r['n']} samples")
    for chunk, e in r["eager"].items():
        print(f"\n== eager, chunk {chunk}: {e['us_per_sample']:.2f} us/sample (whole session ~{e['whole_session_est_s']:.0f} s); "
              f"{e['kernels_per_chunk']} kernels/chunk; GPU busy {e['gpu_busy_fraction']:.0%} "
              f"({e['gpu_kernel_ms_per_chunk']:.0f} of {e['wall_ms_per_chunk']:.0f} ms)")
        for k in e["top_kernels"]:
            print(f"     {k['ms']:7.1f} ms  {k['count']:5d}x  {k['name']}")
    for mode, c in r["compiled"].items():
        if "error" in c:
            print(f"\n== compiled ({mode}): FAILED {c['error']}")
            continue
        s = c["2048"]
        print(f"\n== compiled ({mode}): compile {c['compile_seconds']:.0f} s; {s['us_per_sample']:.2f} us/sample "
              f"(whole session ~{s['whole_session_est_s']:.0f} s), {s['speedup_vs_eager']:.1f}x eager; "
              f"max |diff| {s['max_abs_diff_vs_eager']:.1e}, >1e-4 in {s['frac_diff_gt_1e-4']:.2%}")
    gpu = r["gpu"].replace("NVIDIA ", "").replace(" ", "_")
    path = ROOT / "results" / f"modal_profile_flow_{gpu}.json"
    path.write_text(json.dumps(r, indent=1))
    print(f"-> {path}")
