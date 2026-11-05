"""Modal smoke test: run the test suite and the device benchmark (scripts/bench_devices.py) on a CUDA GPU.

Needs no data. Run from the repo root with the Modal CLI (installed as a uv tool, not in the project venv):

    modal run scripts/modal_smoke.py                  # -> results/bench_devices_modal_<gpu>.json

GPU is a fallback list: Modal takes the first type with capacity. A single type can sit in
"waiting to be scheduled" with no local output (it did for L4 on 2026-11-05), so check
`modal app list` / `modal app logs <app-id>` a few minutes after launch.

The image pins the same torch as the local venv. Code is mounted from the working tree at run time,
so local edits are picked up without rebuilding the image.
"""
import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
GPU = ["L4", "A10G", "L40S"]
IGNORE = ["**/__pycache__", "**/*.pyc"]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.15.0", "numpy", "nibabel", "zuko>=1.2",
                 "pytest", "dipy>=1.9", "cvxpy>=1.4", "scipy")
    .add_local_file(ROOT / "pyproject.toml", "/root/qtilab_repo/pyproject.toml")
    .add_local_dir(ROOT / "qtilab", "/root/qtilab_repo/qtilab", ignore=IGNORE)
    .add_local_dir(ROOT / "tests", "/root/qtilab_repo/tests", ignore=IGNORE)
    .add_local_dir(ROOT / "scripts", "/root/qtilab_repo/scripts", ignore=IGNORE)
)
app = modal.App("qtilab-smoke", image=image)


@app.function(gpu=GPU, cpu=4.0, memory=16384, timeout=1800)
def smoke():
    import subprocess
    import sys

    repo = "/root/qtilab_repo"
    t = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=repo, capture_output=True, text=True)
    sys.path[:0] = [repo, f"{repo}/scripts"]
    import bench_devices

    try:
        r = bench_devices.run("cuda")
    except Exception:              # torch exceptions cannot be unpickled by the torch-less local CLI
        import traceback
        return json.dumps({"error": traceback.format_exc()}), ""
    r["pytest"] = {"returncode": t.returncode, "tail": t.stdout.strip().splitlines()[-3:]}
    # plain JSON: the local Modal CLI has no torch, so torch objects cannot be unpickled there
    return json.dumps(r), bench_devices.summary(r)


@app.local_entrypoint()
def main():
    js, text = smoke.remote()
    r = json.loads(js)
    if "error" in r:
        raise SystemExit("remote error:\n" + r["error"])
    print("pytest:", r["pytest"]["returncode"], *r["pytest"]["tail"], sep="\n  ")
    print(text)
    gpu = r["gpu"].replace("NVIDIA ", "").replace(" ", "_")
    path = ROOT / "results" / f"bench_devices_modal_{gpu}.json"
    path.write_text(json.dumps([r], indent=1))
    print(f"-> {path}")
