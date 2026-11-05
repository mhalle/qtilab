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
- `tests/` needs no downloaded data. `uv run pytest` takes about 10 s.

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

Development so far has been on a 2-core, 7 GB CPU sandbox.
- Two heavy jobs at once ran it out of memory.
- zuko flow sampling is slow and memory-hungry on CPU, so `PosteriorNet.sample` chunks at 256 voxels.

GPU runs belong on Modal or the Apple-Silicon GPU (`device="mps"`).

## Next tasks (handoff, updated 2026-11-05), in priority order

1. **Motion/eddy: done on CPU for one session, and it does not explain the WM residuals** (NOTES 2026-11-05).
   - `scripts/motion_eddy_hhs.py SUB SES` runs tractline's FSL-style eddy (CPU, about 15 min) as LTE and STE passes, and writes `work/hhs_subXX_sesYYec_aligned.npz`.
   - `scripts/motion_eddy_compare.py` compares residuals in signal units. Noise-normalized scores are not comparable, because resampling raises fitted SNR.
   - The WM STE deficit (H5) is unchanged after correction, and r ≈ 4 time dependence removes it equally.
   - The b = 2000 volume offsets drop only slightly. The along-fiber excess persists.
   - Open:
     - run on more sessions (GPU, `--device mps`);
     - a waveform-specific STE eddy model (tractline's field is linear in the gradient vector);
     - check per-direction b-value or gradient-amplitude errors as the source of the volume offsets.
   - Held-out scripts take `--series ec` to read the corrected series.
   - Needs tractline (github.com/mhalle/tractline, v0.11.0): `pip install --no-deps -e path/to/tractline` (scipy ≥ 1.15).
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
