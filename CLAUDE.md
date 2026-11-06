# qtilab: notes for Claude Code sessions

Research code for q-space trajectory imaging (QTI, tensor-valued diffusion MRI, Westin et al. 2016)
with a GPU-friendly PyTorch core. The scientific direction and findings so far are in
`docs/NOTES.md`; read it before proposing new work.

## Layout

- `qtilab/` is the library:
  - `tensors.py`: Mandel conventions, identical to DIPY.
  - `protocol.py`: one b-tensor per volume, and the design matrix.
  - `fit.py`: OLS, WLS, and constrained QTI+ via batched ADMM.
  - `scalars.py`: the QTI scalars.
  - `simulate.py`: explicit tensor distributions.
    - Brain priors v1–v3 (`BRAIN_PRIORS`) and a generic prior (`sample_dtd`).
    - Time dependence (`sample_tau`, `signal_td`).
  - `posterior.py`: set encoder plus zuko normalizing flow.
    - Targets are tissue moments plus free-water fraction, and the full moments.
  - Loaders: `data.py` (DiB open data) and `hhs.py` (OpenNeuro ds005360, with run-to-run alignment).
- `scripts/` holds validation, experiments, maps and data fetchers. Outputs go to `results/`.
- `tests/` needs no downloaded data. `uv run pytest` takes about 6 s warm (about 35 s on a cold first run).

## Local setup (macOS, M1 Max)

- The repo lives in Dropbox (classic client, not File Provider). `.venv/`, `data/` and `work/` carry the `com.dropbox.ignored` attribute so they stay local and are not synced.
  - Recreate the attribute after deleting any of them: `xattr -w com.dropbox.ignored 1 <dir>`.
  - Make the directory and set the attribute **before** filling it. `uv venv --allow-existing .venv` keeps the attribute on an existing directory.
- The system `python3` is 3.9 and the project needs ≥ 3.11. Always run through uv: `uv sync`, `uv run pytest`, `uv run python scripts/...`. Every `python` command in this file means `uv run python`.
- tractline (needed by the motion/eddy scripts) is installed from its git tag, not from `../tractline`:
  `uv pip install --no-deps "tractline @ git+https://github.com/mhalle/tractline@v0.11.0"`.
  - The `../tractline` checkout is on its `ismrm2027` branch, which predates v0.11.0 and has no `motion_eddy`.
  - This install is not part of `uv sync`, so redo it after `uv sync` or after recreating `.venv`.
- **libomp clash (macOS):** dipy's wheel and torch each bundle `libomp`, and importing `dipy.align` (used by `hhs.load_session`) with torch aborts with `OMP: Error #15`. The tests don't import `dipy.align`, so they don't catch it.
  - Fix: `sh scripts/macos_fix_libomp.sh` symlinks dipy's copy to torch's (idempotent; rerun after recreating `.venv` or after `uv sync` upgrades dipy or torch).
  - Do not use `KMP_DUPLICATE_LIB_OK=TRUE` instead: OpenMP warns it can silently give incorrect results.

## Data

- **DiB:** run `scripts/fetch_dib.sh data/dib`, then set `QTILAB_DIB_ROOT` (the script prints it).
  - The phantoms used by `scripts/phantoms_dib.py` live in the same repo under `DATA/{water,oil,hex}/MD-dMRI`; add them to the sparse checkout.
- **Hormone Health Study** (22 subjects × 3 sessions, LTE+STE, CC0, 8.8 GB): run `python scripts/fetch_hhs.py data/hhs`, then set `QTILAB_HHS_ROOT=data/hhs` if it lives elsewhere.
  - Several scripts read an aligned session saved by `hhs.load_session` (`hhs_sub01_ses01_aligned.npz`) and a classical fit of it (`results/hhs_sub01_ses01.npz`). Regenerate both on a new machine; paths are at the top of each script.
- **Cardiff b-tensor data** (Afzali 2021, CC BY, DOI 10.17035/d.2021.0135083057, under 1 GB): not fetched yet. The repository sits behind a browser check, so download it by hand.

## Reproducing the results (in order)

Data roots: `QTILAB_DIB_ROOT`, `QTILAB_DIB_DATA` and `QTILAB_HHS_ROOT`. Intermediate artifacts go to `QTILAB_WORK`, default `./work` (see `qtilab/paths.py`). Run from the repo root.

```
scripts/fetch_dib.sh data/dib --phantoms        # prints the two DiB exports
python scripts/fetch_hhs.py data/hhs            # 8.8 GB
export QTILAB_HHS_ROOT=data/hhs

# DiB validation (no prerequisites)
python scripts/validate_dipy.py; python scripts/bench_constrained.py; python scripts/maps.py
python scripts/phantoms_dib.py; python scripts/check_simulator.py; python scripts/brain_prior_checks.py

# HHS: prepare one session, then the network and the real-data tests
python scripts/hhs_run_scaling.py                               # all 66 sessions, b0 only
python scripts/prepare_session.py 01 01                         # -> work/hhs_sub01_ses01_{aligned,qti}.npz
python scripts/posterior_smoke.py 3000 --prior 2 --snr 20 150 --test-snr 30 140 --tag _v2   # train + evaluate
python scripts/posterior_maps_hhs.py "" "" results/posterior_smoke_hhs_v2.pt _v2            # -> work/..._posterior_v2.npz
python scripts/surprise_check.py; python scripts/surprise_td.py  # need the _v2 posterior file
python scripts/heldout_hhs.py --prior 2 --tag _v2                # also --prior 0/1/3, --td 2.5
python scripts/consistency.py AB; python scripts/consistency.py C
```

- Trained models are committed in `results/*.pt`. `posterior_smoke.py 0 ...` evaluates a saved model without training.
- `posterior_maps_hhs.py` takes positional arguments: aligned, classical, model, tag. Empty strings use the defaults.

## Conventions

- b in ms/µm², diffusivities in µm²/ms.
- Parameter vector (28) = [ln S0, ⟨D⟩ Mandel 6, ℂ Mandel 21 in DIPY's order].
- Everything is batched over voxels and device-agnostic. Keep it that way: no per-voxel Python loops and no NumPy inside hot paths.
- Validate any change to fitting against DIPY (`scripts/validate_dipy.py`, `scripts/bench_constrained.py`).
- Report network outputs as **tissue moments plus f_fw**. Full-distribution µFA is dominated by the poorly identified free-water fraction (about −0.02 µFA per 1% free water).

## Compute

Development now runs locally on an Apple M1 Max (10 cores, 64 GB, MPS). The results through 2026-11-05 were produced in a 2-core, 7 GB CPU sandbox (Cowork), so the timings in README and NOTES are from that sandbox.
- In the sandbox, two heavy jobs at once ran out of memory. That limit no longer applies.
- zuko flow sampling is memory-hungry (about 90 kB per voxel-sample), so `PosteriorNet.sample` chunks at 64 voxels by default. That was sized for the sandbox; chunk 1024 with n = 100 peaks at 9 GB, fine here.

GPU runs use `device="mps"` (PyTorch 2.15 MPS confirmed working), or Modal for CUDA.
- MPS has **no float64**. DIPY validation and the reference benchmarks stay float64 on CPU; use float32 on MPS.
- Device benchmark, 2026-11-05 (NOTES; `scripts/bench_devices.py cpu mps`): MPS is 3–4× faster than CPU for simulate and WLS and 1.4× for the network step, and with the default `chunk=20000` slower for `fit.constrained` (139 vs 100 s at 200k voxels). That is per-chunk overhead, not compute:
  - Fixed: `chunk=None` (the default) is 20k on CPU; on a GPU it is sized from free memory (40% of it at a conservative 88 kB per voxel, between 8k and 250k; `fit._chunk`). This Mac gets one 250k chunk (200k voxels: 26–32 s on MPS vs 100 s on CPU). A 24 GB A10 gets about 100k chunks: 10.1 s (7.4 s as one chunk, 37 s at the old fixed 20k). A 16 GB M2 (simulated) gets 52k chunks: 56 s, 6.1 GB peak. Each chunk pays a ~1000-iteration ADMM tail of small, sync-bound steps.
  - On CPU the cost is the batched `eigh` in `fit._project` (57 of ~65 ms per iteration at 17k voxels).

### Modal

- The `modal` CLI is a uv tool (`~/.local/bin/modal`, profile `mhalle`), not in the project venv. Run from the repo root: `modal run scripts/modal_smoke.py` (tests plus `bench_devices.run("cuda")`, a few minutes, under $1).
- **Modal can stall silently.** An L4 job sat in "waiting to be scheduled on a GPU_L4 worker" with no local output. After launching, check `modal app list` (Tasks 0 = no container) and `perl -e 'alarm 25; exec @ARGV' modal app logs <app-id>` (macOS has no `timeout`). Use a GPU fallback list (`gpu=["L4", "A10G", "L40S"]`). Stop stuck apps with `modal app stop -y <app-id>`.
- Return plain JSON from remote functions: the local CLI has no torch, so torch objects (even `torch.__version__`) fail to unpickle.
- Train on GPU with `scripts/posterior_smoke.py ... --device cuda` (or `mps`): batches are then simulated on the device in float32 (`posterior.simulate_batch(device=...)`). A10: 66 ms per step at batch 4096, vs 257 ms with CPU simulation. Without `--device` the original float64 CPU path runs unchanged.
- **CUDA batched eigensolvers need huge workspace** (`pinv` at 200k asked for 66 GB, `eigvalsh` on 3×3 for 51 GB). Route any new batched `eigh`/`eigvalsh`/`pinv`/`svd` through `tensors.batched` (16k per call). Peak memory per stage is in NOTES (2026-11-05 evening).

## Next tasks (handoff, updated 2026-11-05), in priority order

1. **Motion/eddy: done on CPU for three sessions (sub-01, -02, -06; ses-01), and it does not explain the WM residuals** (NOTES 2026-11-05).
   - `scripts/motion_eddy_hhs.py SUB SES` runs tractline's FSL-style eddy (CPU, about 15 min) as LTE and STE passes, and writes `work/hhs_subXX_sesYYec_aligned.npz`.
   - `scripts/motion_eddy_compare.py` compares residuals in signal units. Noise-normalized scores are not comparable, because resampling raises fitted SNR.
   - The WM STE deficit (H5) is unchanged after correction, and r ≈ 4 time dependence removes it equally.
   - The b = 2000 volume offsets drop only slightly. The along-fiber excess persists.
   - Within-run STE shifts along PE (up to ~1 mm) vary by subject; b-value and acquisition time are confounded, because STE runs are acquired in b order.
   - Open:
     - run on the cohort (GPU, `--device mps`);
     - a waveform-specific STE eddy model (tractline's field is linear in the gradient vector);
     - ~~check per-direction b errors~~ done: the offsets are a per-volume level shared by WM and GM. In STE it is a period-3 pattern (three waveform rotations, ~1–1.5% peak-to-peak) reproducible across subjects (`scripts/volume_offsets.py`, `volume_offsets_followup.py`, `results/volume_offsets_pattern.png`). To do: a cohort per-volume gain template as a preprocessing step.
   - Held-out scripts take `--series ec` to read the corrected series.
   - Needs tractline (github.com/mhalle/tractline, v0.11.0): install as in *Local setup* (scipy ≥ 1.15). v0.11.0 also has `motion_eddy_metal.py`, which may be the MPS path for the cohort run.
2. **Rotation-aware predictor.** Plain k-NN on the simulation bank has an algorithmic error floor of about 2 noise sd in coherent WM at b = 2000. Simulated voxels score the same as real ones, so it is not tissue.
   - Option: coarse match on rotation-invariant features, then re-simulate candidate tissues rotated to the voxel's fiber direction.
   - Or predict from the network posterior.
   - Also calibrate surprise within FA bins, because the floor differs between WM and GM.
3. **Cohort run for H7 (reliability).**
   - Align, fit (constrained) and run network inference for all 66 sessions on GPU.
   - Report whole-brain regional ICCs, classical vs network.
   - The 5-subject pilot: the network keeps between-subject differences while cutting session-to-session variation about 5× (`results/consistency_*.json`).
4. **Longer network training on GPU**, across several protocols with random measurement subsets, for protocol invariance. This is the hook for the embedding work.
   - Include time dependence (r ≈ 4, WM only) in the training prior.
   - Include the fitted noise model: `qtilab/noise.py`, with α(b) about 1–2% in tissue and about 10% at b = 100 in CSF.
5. **H4 follow-ups.**
   - Brain prior: LTE-only recovers tissue µFA nearly as well as the full protocol.
   - Generic prior: spherical encoding cuts GM µFA error by about 25–30%, with no gain in WM.
   - Next: apply LTE-only models to a legacy multi-shell dataset (HCP-style), and quantify prior dependence there.
6. **Real waveforms.** DiB ships its waveforms. For HHS, ask the authors (Erlangen waveforms, Martin et al. 2020). Replace the single spectral ratio with real encoding spectra.
7. **Cardiff b = 3–10 extrapolation test** once the data is downloaded by hand.
