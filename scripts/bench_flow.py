"""Posterior network: eager vs torch.compile for sampling and for a training step, on CPU, MPS or CUDA.

Needs no data (synthetic HHS-like protocol from bench_devices.py, the committed v2 model). Correctness is
checked value for value: sampling with the same generator and chunking, training from identical weights on
the same batch (loss and every gradient).

    uv run python scripts/bench_flow.py mps            # -> results/bench_flow_<host>_<device>.json
    uv run python scripts/bench_flow.py cpu
Run one device per process: compiling for MPS and then CPU in the same process segfaulted (torch 2.15).
Modal runs the same ``run()`` on CUDA (scripts/modal_profile_flow.py --what bench).
"""
from __future__ import annotations

import copy
import json
import platform
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
from bench_devices import hhs_like_protocol  # noqa: E402
from qtilab import posterior  # noqa: E402

MODEL = HERE.parent / "results" / "posterior_smoke_hhs_v2.pt"


def _sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def _net(device, model=MODEL):
    net = posterior.PosteriorNet()
    net.load_state_dict(torch.load(model, map_location="cpu"))
    return net.to(device).eval()


def bench_sampling(device, V, n=50, chunks=(512, 2048)):
    prot = hhs_like_protocol()
    sim_kw = {} if device == "cpu" else {"device": device}
    x = posterior.simulate_batch(prot, V, g=torch.Generator(device=device).manual_seed(0), **sim_kw).x.to(device)
    net = _net(device)
    out = []
    for chunk in chunks:
        r = {"chunk": chunk, "voxels": V, "n": n}
        for mode in ("eager", "compiled"):
            kw = dict(n=n, chunk=chunk, compile=(mode == "compiled"))
            t = time.perf_counter(); net.sample(x[:chunk], **kw); _sync(device)
            r[f"{mode}_first_call_s"] = time.perf_counter() - t           # includes compilation when compiled
            t = time.perf_counter()
            y = net.sample(x, generator=torch.Generator(device=device).manual_seed(1), **kw); _sync(device)
            dt = time.perf_counter() - t
            r[f"{mode}_s"] = dt
            r[f"{mode}_us_per_sample"] = 1e6 * dt / (V * n)
            r[f"{mode}_whole_session_est_s"] = dt * 95801 / V
            r[f"_{mode}"] = y
        d = (r.pop("_compiled") - r.pop("_eager")).abs()
        r["speedup"] = r["eager_s"] / r["compiled_s"]
        r["max_abs_diff"] = float(d.max())
        r["frac_diff_gt_1e-4"] = float((d > 1e-4).float().mean())
        out.append(r)
    return out


def bench_training(device, batch, steps=20):
    prot = hhs_like_protocol()
    sim_kw = {} if device == "cpu" else {"device": device}
    g = torch.Generator(device=device).manual_seed(2)
    batches = [posterior.simulate_batch(prot, batch, g=g, **sim_kw) for _ in range(steps + 3)]
    batches = [(b.x.to(device), b.y.to(device)) for b in batches]
    base = _net(device).train()
    res = {"batch": batch, "steps": steps}
    grads = {}
    for mode in ("eager", "compiled"):
        net = copy.deepcopy(base)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
        loss_fn = lambda xb, yb: -net.log_prob(xb, yb).mean()            # noqa: E731
        if mode == "compiled":
            loss_fn = torch.compile(loss_fn, dynamic=False)
        xb, yb = batches[0]
        t = time.perf_counter()
        loss = loss_fn(xb, yb); opt.zero_grad(); loss.backward(); _sync(device)
        res[f"{mode}_first_step_s"] = time.perf_counter() - t           # includes compilation when compiled
        res[f"{mode}_loss0"] = float(loss)
        grads[mode] = [p.grad.detach().clone() for p in net.parameters()]
        opt.step()
        for xb, yb in batches[1:3]:                                      # warm-up
            loss = loss_fn(xb, yb); opt.zero_grad(); loss.backward(); opt.step()
        _sync(device); t = time.perf_counter()
        for xb, yb in batches[3:]:
            loss = loss_fn(xb, yb); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
        _sync(device)
        res[f"{mode}_ms_per_step"] = 1e3 * (time.perf_counter() - t) / steps
    res["speedup"] = res["eager_ms_per_step"] / res["compiled_ms_per_step"]
    res["loss0_abs_diff"] = abs(res["eager_loss0"] - res["compiled_loss0"])
    res["grad_max_rel_diff"] = max(float((ge - gc).abs().max() / ge.abs().max().clamp_min(1e-30))
                                   for ge, gc in zip(grads["eager"], grads["compiled"]))
    return res


def run(device) -> dict:
    big = device != "cpu"
    out = {"device": device, "torch": str(torch.__version__), "host": platform.node(), "machine": platform.machine()}
    if device.startswith("cuda"):
        out["gpu"] = torch.cuda.get_device_name(0)
    out["sampling"] = bench_sampling(device, V=8192 if big else 2048, chunks=(512, 2048) if big else (512,))
    out["training"] = [bench_training(device, b) for b in ((512, 4096) if big else (512,))]
    return out


def summary(r: dict) -> str:
    lines = [f"== {r['device']} {r.get('gpu', r['machine'])} (torch {r['torch']})"]
    for s in r["sampling"]:
        lines.append(f"  sample n={s['n']} chunk {s['chunk']:5d}: eager {s['eager_us_per_sample']:6.2f} us/sample, compiled "
                     f"{s['compiled_us_per_sample']:6.2f} ({s['speedup']:.1f}x; whole session {s['eager_whole_session_est_s']:.0f} -> "
                     f"{s['compiled_whole_session_est_s']:.0f} s); compile {s['compiled_first_call_s']:.0f} s; "
                     f"max |diff| {s['max_abs_diff']:.1e}, >1e-4 in {s['frac_diff_gt_1e-4']:.3%}")
    for t in r["training"]:
        lines.append(f"  train batch {t['batch']:5d}: eager {t['eager_ms_per_step']:7.1f} ms/step, compiled {t['compiled_ms_per_step']:7.1f} "
                     f"({t['speedup']:.1f}x); compile {t['compiled_first_step_s']:.0f} s; loss diff {t['loss0_abs_diff']:.1e}, "
                     f"grad max rel diff {t['grad_max_rel_diff']:.1e}")
    return "\n".join(lines)


if __name__ == "__main__":
    devices = sys.argv[1:] or ["cpu"]
    if len(devices) > 1:
        raise SystemExit("one device per process (compiling for two devices in one process segfaulted)")
    r = run(devices[0])
    print(summary(r), flush=True)
    path = HERE.parent / "results" / f"bench_flow_{platform.node().split('.')[0]}_{devices[0]}.json"
    path.write_text(json.dumps([r], indent=1))
    print(f"-> {path}")
