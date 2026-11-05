"""Device benchmark: the same float32 workload on CPU, MPS or CUDA, so runs are comparable.

Needs no downloaded data: it uses a synthetic HHS-like protocol (b = 0 plus five shells of LTE and STE).

Three parts:
* ``simfit``: simulate V voxels on the device (generic prior, Rician noise), then WLS and constrained fits
  (default ``chunk``: 20k on CPU, 250k on GPUs).
  Accuracy is the median absolute parameter error against the simulated ground truth.
* ``net``: PosteriorNet training steps on one pre-simulated batch already on the device (network compute only).
* ``train_step``: a training step with ``posterior.simulate_batch`` on CPU (float64, the default path),
  then the batch moves to the device.
* ``train_step_device``: the same step with the batch simulated on the device (float32).

    uv run python scripts/bench_devices.py cpu mps          # -> results/bench_devices_<host>.json
Modal runs the same ``run()`` on CUDA (scripts/modal_smoke.py).
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qtilab import fit, posterior, simulate  # noqa: E402
from qtilab.protocol import Protocol  # noqa: E402


def hhs_like_protocol(seed=0) -> Protocol:
    """b = 0 x 6, then b in {100, 500, 1000, 1500, 2000} s/mm^2 with 24 LTE + 12 STE each (186 volumes)."""
    rng = np.random.default_rng(seed)
    bvals, bvecs, shapes = [0.0] * 6, [[0, 0, 1]] * 6, ["LTE"] * 6
    for b in (100, 500, 1000, 1500, 2000):
        for shape, n in (("LTE", 24), ("STE", 12)):
            v = rng.normal(size=(n, 3))
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            bvals += [b] * n
            bvecs += v.tolist()
            shapes += [shape] * n
    return Protocol.from_table(bvals, np.array(bvecs).T, shapes)


def _sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def _timed(fn, device):
    _sync(device)
    t = time.perf_counter()
    out = fn()
    _sync(device)
    return out, time.perf_counter() - t


def bench_simfit(prot, device, V, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    X = prot.design_matrix().to(torch.float32).to(device)
    sim, t_sim = _timed(lambda: simulate.simulate(prot, V, g=g, dtype=torch.float32, device=device), device)
    p_wls, t_wls = _timed(lambda: fit.wls(X, sim["signal"]), device)
    (p_con, info), t_con = _timed(lambda: fit.constrained(X, sim["signal"]), device)
    truth = sim["params"]
    return {
        "V": V, "t_simulate": t_sim, "t_wls": t_wls, "t_constrained": t_con,
        "n_violating": info.n_violating, "admm_iterations": info.iterations,
        "converged_fraction": info.converged_fraction,
        "median_abs_err_wls": (p_wls - truth).abs().median().item(),
        "median_abs_err_constrained": (p_con - truth).abs().median().item(),
        "any_nan": bool(p_con.isnan().any()),
    }


def _net_and_opt(device, seed=0):
    torch.manual_seed(seed)
    net = posterior.PosteriorNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    return net, opt


def _step(net, opt, x, y):
    loss = -net.log_prob(x, y).mean()
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
    opt.step()
    return loss


def bench_net(prot, device, batch, steps, warmup=3):
    g = torch.Generator().manual_seed(1)
    b = posterior.simulate_batch(prot, batch, g=g)
    x, y = b.x.to(device), b.y.to(device)
    net, opt = _net_and_opt(device)
    for _ in range(warmup):
        _step(net, opt, x, y)

    def loop():
        for _ in range(steps):
            loss = _step(net, opt, x, y)
        return loss.item()

    nll, t = _timed(loop, device)
    return {"batch": batch, "steps": steps, "s_per_step": t / steps, "final_nll": nll}


def bench_train_step(prot, device, batch, steps, warmup=2, sim_on_device=False):
    g = torch.Generator(device=device if sim_on_device else "cpu").manual_seed(2)
    sim_kw = {"device": device} if sim_on_device else {}
    net, opt = _net_and_opt(device)
    t_sim = 0.0

    def one():
        nonlocal t_sim
        _sync(device)                 # so queued GPU work from the last step is not billed to simulation
        t = time.perf_counter()
        b = posterior.simulate_batch(prot, batch, g=g, **sim_kw)
        _sync(device)
        t_sim += time.perf_counter() - t
        return _step(net, opt, b.x.to(device), b.y.to(device))

    for _ in range(warmup):
        one()
    t_sim = 0.0
    _, t = _timed(lambda: [one() for _ in range(steps)][-1].item(), device)
    return {"batch": batch, "steps": steps, "s_per_step": t / steps, "s_per_step_simulation": t_sim / steps}


def run(device, simfit_sizes=(20_000, 200_000), net_batches=(512, 4096), steps=20) -> dict:
    prot = hhs_like_protocol()
    out = {
        "device": device, "torch": str(torch.__version__), "python": platform.python_version(),
        "host": platform.node(), "machine": platform.machine(), "cpu_threads": torch.get_num_threads(),
        "protocol_volumes": prot.n,
    }
    if device.startswith("cuda"):
        out["gpu"] = torch.cuda.get_device_name(0)
    bench_simfit(prot, device, 2_000)                       # warm-up (kernel compilation, allocator)
    out["simfit"] = [bench_simfit(prot, device, V) for V in simfit_sizes]
    out["net"] = [bench_net(prot, device, B, steps) for B in net_batches]
    out["train_step"] = [bench_train_step(prot, device, B, max(5, steps // 2)) for B in net_batches]
    if device != "cpu":
        out["train_step_device"] = [bench_train_step(prot, device, B, max(5, steps // 2), sim_on_device=True)
                                    for B in net_batches]
    return out


def summary(r: dict) -> str:
    lines = [f"== {r['device']} {r.get('gpu', r['machine'])} (torch {r['torch']}, {r['cpu_threads']} threads)"]
    for s in r["simfit"]:
        lines.append(f"  simfit V={s['V']:>7}: simulate {s['t_simulate']:6.2f} s  wls {s['t_wls']:6.2f} s  "
                     f"constrained {s['t_constrained']:6.2f} s ({s['converged_fraction']:.0%} conv, "
                     f"{s['admm_iterations']} it)  err {s['median_abs_err_constrained']:.4f}  nan {s['any_nan']}")
    for s in r["net"]:
        lines.append(f"  net      B={s['batch']:>5}: {1000 * s['s_per_step']:7.1f} ms/step  (NLL {s['final_nll']:.2f})")
    for key, label in (("train_step", "train (sim CPU)"), ("train_step_device", "train (sim dev)")):
        for s in r.get(key, []):
            lines.append(f"  {label} B={s['batch']:>5}: {1000 * s['s_per_step']:7.1f} ms/step, of which simulation "
                         f"{1000 * s['s_per_step_simulation']:.1f} ms")
    return "\n".join(lines)


if __name__ == "__main__":
    devices = sys.argv[1:] or ["cpu"]
    results = []
    for d in devices:
        r = run(d)
        print(summary(r), flush=True)
        results.append(r)
    path = Path("results") / f"bench_devices_{platform.node().split('.')[0]}.json"
    path.write_text(json.dumps(results, indent=1))
    print(f"-> {path}")
