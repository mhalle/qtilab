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
- **Cardiff b-tensor data** (Afzali 2021, CC BY 4.0, DOI 10.17035/d.2021.0135083057): run `scripts/fetch_cardiff.sh data/cardiff` (251.5 MB zip, md5-checked; Figshare's API skips the web page's browser check).
  - Two participants (`data1`, `data2`) on the 3T Connectom, 3 mm isotropic, TE 88 ms, 70×70×42. LTE and PTE have 439 volumes each, STE 182, plus a reversed-PE STE b0. There are no JSON sidecars or waveforms.
  - The bval files list nominal shells 0–12 ms/µm² for all three encodings, more than the dataset description gives. The powder-averaged decay is consistent with the nominal b at low b (LTE > PTE > STE at every shell). STE reaches the noise floor (~0.01 of S0) by b ≈ 6.
  - PTE bvecs repeat LTE's directions; presumably they are the plane normals (unverified).

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
python scripts/posterior_maps_hhs.py "" "" results/posterior_smoke_hhs_v2.pt _v2 --whole-brain --device mps   # -> work/..._posterior_v2_full.npz, ~30 s
python scripts/surprise_check.py; python scripts/surprise_td.py  # need the _v2 posterior file
python scripts/heldout_hhs.py --prior 2 --tag _v2                # also --prior 0/1/3, --td 2.5
python scripts/consistency.py AB; python scripts/consistency.py C

# FastSurfer (T1) vs the QTI tissue classes; needs the _v2_full posterior and ../haversack (with its .venv)
../haversack/.venv/bin/haversack segment data/hhs/sub-01/ses-01/anat/sub-01_ses-01_T1w.nii \
    --task fastsurfer:asegdkt --device mps -o work/hhs_sub01_ses01_asegdkt.duckn.zip   # ~100 s
python scripts/t1_to_dwi.py                                     # rigid T1 -> b0
../haversack/.venv/bin/python scripts/fastsurfer_to_dwi.py      # needs zarr + rankfield (haversack's env)
python scripts/fastsurfer_vs_qti.py                             # -> results/fastsurfer_*_s0101.*
python scripts/posterior_maps_hhs.py "" "" results/posterior_smoke_hhs_v3.pt _v3 --whole-brain --device mps   # -> work/..._posterior_v3_full.npz
python scripts/prior_fa_gap.py --device mps --with-v3          # needs the _v3_full posterior
python scripts/prior_match.py --priors 1 2 3 4                 # priors vs real classical maps, by region and label-free; ~2 min
python scripts/posterior_smoke.py 12000 --batch 2048 --prior 4 --snr 20 150 --test-snr 30 140 --tag _v4long --device mps   # 22 min
python scripts/posterior_maps_hhs.py "" "" results/posterior_smoke_hhs_v4long.pt _v4long --whole-brain --device mps
python scripts/network_regions.py --tags _v2 _v4 _v4long         # needs those _full posteriors
python scripts/calibration.py results/posterior_smoke_hhs_v4long.pt; python scripts/calibration.py results/posterior_smoke_hhs_v4longfw.pt --exact-fw
python scripts/seed_spread.py --tags _v4long _v4long_s1 _v4long_s2   # long runs with --seed 1 / 2, each mapped --whole-brain
# v5: v4 + exact free water + WM time dependence r = 1.5 + noise model + gain jitter + SNR from 10 (23 min on MPS)
python scripts/posterior_smoke.py 12000 --batch 2048 --prior 4 --snr 10 150 --test-snr 30 140 --exact-fw --td-ratio 1.5 --alpha 0 0.03 --pulsation 0.15 --gain-sd 0.007 --seed 0 --device mps --tag _v5
python scripts/posterior_maps_hhs.py "" "" results/posterior_smoke_hhs_v5.pt _v5 --whole-brain --device mps
# v5 ablations (each about 23 min training + map): drop td / pulsation / the SNR floor of 10, then compare
#   tags _v5_notd (no --td-ratio), _v5_nopuls (no --pulsation), _v5_snr20 (--snr 20 150); map each --whole-brain
python scripts/network_regions.py --tags _v2 _v4long _v4longfw _v5 _v5_notd _v5_nopuls _v5_snr20
python scripts/low_snr_bias.py --tags _v4longfw _v5 _v5_snr20   # real SNR by region + simulated low-SNR bias, ~5 min
python scripts/prepare_session.py 01 02   # also 01 03, 02 01, 06 01; then the held-out grid in scripts/prior_heldout_summary.py's docstring
python scripts/prior_heldout_summary.py
# cohort (H7): all 66 sessions, then regional ICCs (classical vs network)
scripts/cohort_run.sh                                            # 1 h 45 min; resumable; logs in work/logs/cohort
python scripts/cohort_icc.py --tag _v5                           # -> results/cohort_icc_v5.json
# v2-era results rerun with v4 (NOTES 2026-11-06, last entry)
python scripts/posterior_smoke.py 0 --prior 4 --test-snr 20 35 --model results/posterior_smoke_hhs_v4long.pt --tag _v4long_test20_35
python scripts/consistency.py B results/posterior_smoke_hhs_v4long.pt _v4long; python scripts/consistency.py C results/posterior_smoke_hhs_v4long.pt _v4long
python scripts/posterior_smoke.py 12000 --batch 2048 --prior 4 --snr 20 150 --test-snr 30 140 --tag _v4longlte --device mps --shapes LTE   # 17 min
python scripts/posterior_smoke.py 3000 --prior 4 --snr 20 150 --test-snr 30 140 --tag _v4lte --shapes LTE
python scripts/h4_lte_only.py --full-model results/posterior_smoke_hhs_v4long.pt --lte-model results/posterior_smoke_hhs_v4longlte.pt --test-priors 4,2,3,0 --tag _v4long
python scripts/h4_lte_only.py --full-model results/posterior_smoke_hhs_v4.pt --lte-model results/posterior_smoke_hhs_v4lte.pt --test-priors 4 --tag _v4
```

- Trained models are committed in `results/*.pt`. `posterior_smoke.py 0 ...` evaluates a saved model without training.
- `posterior_maps_hhs.py` takes positional arguments: aligned, classical, model, tag. Empty strings use the defaults.
  - `--whole-brain` samples every brain voxel (compiled, chunk 512, seeded with `--seed`) on `--device`. It writes `paths.posterior_volume` and leaves the slice-20 file the held-out and surprise scripts read untouched. Use `--sub`/`--ses` for other sessions.

## Conventions

- b in ms/µm², diffusivities in µm²/ms.
- Parameter vector (28) = [ln S0, ⟨D⟩ Mandel 6, ℂ Mandel 21 in DIPY's order].
- Everything is batched over voxels and device-agnostic. Keep it that way: no per-voxel Python loops and no NumPy inside hot paths.
- Validate any change to fitting against DIPY: `scripts/validate_dipy.py`, `scripts/bench_constrained.py` (DiB, full rank) and `scripts/validate_dipy_hhs.py` (HHS, rank 23; needs a prepared session).
  - On HHS the fit matches DIPY SDPdc/CLARABEL as closely as DIPY's SCS does, and is strictly feasible (NOTES 2026-11-05 night).
  - On HHS `FitInfo.converged_fraction` is about 0.85 even though the objective is at the optimum: the residual stopping rule is too strict on rank-deficient designs.
  - DIPY returns ⟨D⟩ in mm²/s and ℂ in (mm²/s)² when given bvals in s/mm². Multiply by 1e3 and 1e6 before comparing with qtilab's µm²/ms.
- Voxels whose constrained fit collapses to ⟨D⟩ = 0 (`scalars.collapsed`) are dropped from the mask in `prepare_session.py`. Near-collapsed voxels still give extreme K values (K_bulk up to ~6e3), so aggregate K by medians, not means.
- The "dc" constraints allow µFA > 1 (140 voxels in sub-01 ses-01) and C_c > 1 (3,069). This is a known limitation of the relaxation, not a bug; see NOTES 2026-11-06.
- Report network outputs as **tissue moments plus f_fw**. Full-distribution µFA is dominated by the poorly identified free-water fraction (about −0.02 µFA per 1% free water).

## Compute

Development now runs locally on an Apple M1 Max (10 cores, 64 GB, MPS). The results through 2026-11-05 were produced in a 2-core, 7 GB CPU sandbox (Cowork), so the timings in README and NOTES are from that sandbox.
- In the sandbox, two heavy jobs at once ran out of memory. That limit no longer applies.
- zuko flow sampling is memory-hungry (about 90 kB per voxel-sample), so `PosteriorNet.sample` chunks at 64 voxels by default. That was sized for the sandbox; chunk 1024 with n = 100 peaks at 9 GB, fine here.

GPU runs use `device="mps"` (PyTorch 2.15 MPS confirmed working), or Modal for CUDA.
- Run **one MPS training at a time**: two together tripped macOS's GPU watchdog ("Impacting Interactivity") and killed one at step 9,000.
- MPS has **no float64**. DIPY validation and the reference benchmarks stay float64 on CPU; use float32 on MPS.
- Rank-deficient WLS solves in X's row space (Cholesky), not with `pinv`. MPS's batched `pinv` was off by up to 1e-2 in µFA in float32. With the fix, MPS float32 matches CPU float64 at default settings (µFA within 1e-3 of the tight reference; NOTES 2026-11-05 late night). Don't reintroduce batched `pinv` on hot paths.
- An objective-based ADMM stopping rule was tried and rejected (NOTES); don't retry it without a new idea.
- Device benchmark, 2026-11-05 (NOTES; `scripts/bench_devices.py cpu mps`): MPS is 3–4× faster than CPU for simulate and WLS and 1.4× for the network step, and with the default `chunk=20000` slower for `fit.constrained` (139 vs 100 s at 200k voxels). That is per-chunk overhead, not compute:
  - Fixed: `chunk=None` (the default) is 20k on CPU; on a GPU it is sized from free memory (40% of it at a conservative 88 kB per voxel, between 8k and 250k; `fit._chunk`). This Mac gets one 250k chunk (200k voxels: 26–32 s on MPS vs 100 s on CPU). A 24 GB A10 gets about 100k chunks: 10.1 s (7.4 s as one chunk, 37 s at the old fixed 20k). A 16 GB M2 (simulated) gets 52k chunks: 56 s, 6.1 GB peak. Each chunk pays a ~1000-iteration ADMM tail of small, sync-bound steps.
  - On CPU the cost is the batched `eigh` in `fit._project` (57 of ~65 ms per iteration at 17k voxels).

### Modal

- The `modal` CLI is a uv tool (`~/.local/bin/modal`, profile `mhalle`), not in the project venv. Run from the repo root: `modal run scripts/modal_smoke.py` (tests plus `bench_devices.run("cuda")`, a few minutes, under $1).
- **Modal can stall silently.** An L4 job sat in "waiting to be scheduled on a GPU_L4 worker" with no local output. After launching, check `modal app list` (Tasks 0 = no container) and `perl -e 'alarm 25; exec @ARGV' modal app logs <app-id>` (macOS has no `timeout`). Use a GPU fallback list (`gpu=["L4", "A10G", "L40S"]`). Stop stuck apps with `modal app stop -y <app-id>`.
- Real-data check: `scripts/modal_check_hhs.py` (needs `work/modal_check_sub01_ses01.npz`, packed locally; NOTES 2026-11-06). On an A10, float64 matches CPU float64 to ≤ 1e-11 and float32 is within 1e-3 in µFA. Whole session: constrained fit 7 s (float32) / 34 s (float64), vs 106 s on the M1 CPU.
- Posterior sampling in eager mode is about 50 s per session on an A10 and 41 s on MPS (n = 50). On CUDA, 76% of that is PyTorch's `cumsum` over 8-bin rows, a pathological scan kernel; on MPS it is many small unfused ops.
  - Use `PosteriorNet.sample(x, n, chunk=512, compile=True)` for whole-brain inference: 11.7× faster on an L40S (≈1 s per session), 5.1× on MPS (≈8 s), 1.9× on CPU, within 3e-4 of eager. Pass `generator=` for reproducible samples. In eager mode use chunk 512, not the default 64.
  - `posterior_smoke.py --compile` (compiled training loss) helps only on CUDA (1.6–4.3×). On MPS it is 2.5× slower at batch 4096, and on CPU 0.9×.
  - Compile for one device per process: compiling for MPS and then CPU in one process segfaulted.
- Inspect a running container with `modal container exec <ta-id> -- nvidia-smi` (the `--` is required; the image has no `ps`).
- Return plain JSON from remote functions: the local CLI has no torch, so torch objects (even `torch.__version__`) fail to unpickle.
- Train on GPU with `scripts/posterior_smoke.py ... --device cuda` (or `mps`): batches are then simulated on the device in float32 (`posterior.simulate_batch(device=...)`). A10: 66 ms per step at batch 4096, vs 257 ms with CPU simulation. Without `--device` the original float64 CPU path runs unchanged.
- **CUDA batched eigensolvers need huge workspace** (`pinv` at 200k asked for 66 GB, `eigvalsh` on 3×3 for 51 GB). Route any new batched `eigh`/`eigvalsh`/`pinv`/`svd` through `tensors.batched` (16k per call). Peak memory per stage is in NOTES (2026-11-05 evening).

## Next tasks (handoff, updated 2026-11-05), in priority order

1. **Motion/eddy: done on CPU for three sessions (sub-01, -02, -06; ses-01), and it does not explain the WM residuals** (NOTES 2026-11-05).
   - `scripts/motion_eddy_hhs.py SUB SES` runs tractline's FSL-style eddy (CPU, about 15 min) as LTE and STE passes, and writes `work/hhs_subXX_sesYYec_aligned.npz`.
   - `scripts/motion_eddy_compare.py` compares residuals in signal units. Noise-normalized scores are not comparable, because resampling raises fitted SNR.
   - The WM STE deficit (H5) is unchanged after correction, and r ≈ 4 time dependence removes it equally.
     With prior v4, about 80% of the deficit goes away without time dependence, in all five sessions tested; r ≈ 1.5 removes the rest and r = 4 overshoots (NOTES 2026-11-06, `scripts/prior_heldout_summary.py`). H5 is a small effect; its earlier size was mostly prior misspecification.
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
3. **Cohort run for H7 (reliability): done for regional medians** (NOTES 2026-11-06, cohort entry).
   - All 66 sessions are prepared, segmented and mapped with v5 (`scripts/cohort_run.sh`, 1 h 45 min; resumable). `scripts/cohort_icc.py` gives ICC(1,1) by FastSurfer region.
   - The network does not raise regional ICCs. It shrinks within- and between-subject spread together. WM FA_t is better (+0.23); GM MD_t is worse (−0.4).
   - Open:
     - voxelwise/small-ROI reliability;
     - a simulated cohort with known subject differences;
     - prior dependence (`TAG=_v4long scripts/cohort_run.sh`).
   - The 5-subject pilot: under v2 the network cut session-to-session variation about 5× while keeping between-subject differences, but most of that was the prior. With v4long, WM tissue µFA between/within is 1.5 (classical 0.9, v2 3.9; `results/consistency_*_v4long.json`, NOTES 2026-11-06).
4. **Longer network training on GPU**, across several protocols with random measurement subsets, for protocol invariance. This is the hook for the embedding work.
   - Fix the prior's FA_t gap first: v2 has 1.3% of tissue at FA 0.3–0.5, real data 22.5%, so FA_t (and C_c) is biased low in dispersed WM (NOTES 2026-11-06, `scripts/prior_fa_gap.py`).
     - Prior v4 (`BRAIN_PRIORS[4]`) matches real classical maps in WM and cortex and covers real tissue in the (µFA, C_c) and (MD, C_MD) planes (`scripts/prior_match.py`, NOTES 2026-11-06). Trained (`_v4`, `_v4long`): narrows the FA_t gap; longer training moves real-data estimates by a few hundredths, and 90% coverage stays at 0.77–0.89 (NOTES 2026-11-06).
   - Include time dependence (WM only) in the training prior: r ≈ 1.5 with v4 (five sessions; r ≈ 4 was fitted with v2).
   - **v5 (`results/posterior_smoke_hhs_v5.pt`) is the current best network** (NOTES 2026-11-06, v5 entry); its time dependence lowers WM µFA_t by about 0.016. Its lower pallidum/brainstem anisotropy comes from the SNR floor of 10: about 35% of those voxels have SNR below 20, where networks trained from SNR 20 overestimate FA_t by 0.02–0.05 (NOTES, v5 ablations; `scripts/low_snr_bias.py`). Open: the WM FA_t gap (~0.04).
   - Train with `--exact-fw` (free water as an exact weight; the old random count made the network look overconfident) and long (12,000 × 2048 or more: short runs are not converged on real data; seeds then agree within ~0.01 regionally). NOTES 2026-11-06, calibration entry.
   - Include the fitted noise model: `qtilab/noise.py`, with α(b) about 1–2% in tissue and about 10% at b = 100 in CSF.
5. **H4 follow-ups.**
   - Brain prior: under v2, LTE-only recovered WM tissue µFA nearly as well as the full protocol (+5% error). Under v4 it is about 75% worse in WM and the posterior SD about 2× larger on real WM, at either training budget: v2's coherent WM hid it (NOTES 2026-11-06).
   - Generic prior: spherical encoding cuts GM µFA error by about 25–30%, with no gain in WM.
   - Next: apply LTE-only models to a legacy multi-shell dataset (HCP-style), and quantify prior dependence there.
6. **Real waveforms.** DiB ships its waveforms. For HHS, ask the authors (Erlangen waveforms, Martin et al. 2020). Replace the single spectral ratio with real encoding spectra.
7. **Cardiff b = 3–12 extrapolation test** (data fetched 2026-11-06; needs a loader, protocol and motion/eddy correction first).
