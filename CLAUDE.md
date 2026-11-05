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

## Next tasks (handoff, 2026-11-04), in priority order

1. **Motion and eddy-current correction for b-tensor data.** This gates H5.
   - Residuals show a run-specific, direction-sensitive excess in white matter and misfit in white-matter spherical encoding (`docs/NOTES.md`, "time dependence" and "validation" sections).
   - Extend tractline's GPU eddy model to b-tensors:
     - Rotate B → R B Rᵀ per volume.
     - Predict the reference signal per encoding shape, since STE has no direction.
     - Model eddy currents specific to each waveform.
   - Then rerun `scripts/surprise_td.py` and `scripts/heldout_hhs.py`, both the default run and `--td`.
2. **Cohort run for H7 (reliability).**
   - Align, fit (constrained) and run network inference for all 66 sessions on GPU.
   - Report whole-brain regional ICCs, classical vs network.
   - A 5-subject single-slice pilot showed the network keeps between-subject differences while cutting session-to-session variation about 5× (`results/consistency_*.json`).
3. **Longer network training on GPU.**
   - The current models had 3k steps × 512 voxels on CPU.
   - Train across several protocols with random measurement subsets, so the encoder becomes protocol-invariant. This is the hook for the embedding work: test DiB_70 vs DiB_217 invariance.
4. **Time dependence, properly.** Held-out STE prediction in WM improves from 1.78 to 1.24 noise sd with one global LTE/STE spectral ratio (r = 2.5), with LTE prediction unchanged. But GM gets worse, so the per-compartment τ ranges are probably wrong for GM.
   - Fit r and the τ ranges per tissue class by held-out accuracy.
   - Repeat on other subjects and sessions.
   - Get the real waveforms: DiB ships its waveforms; for HHS, contact the authors (Erlangen waveforms, Martin et al. 2020).
5. **Noise model.** Add a signal-proportional (physiological) term. Low-b and CSF residuals exceed the thermal noise estimated from the b = 1500 repeats.
6. **Cardiff b = 3–10 extrapolation test** with `scripts/heldout_hhs.py`-style code, once the data is downloaded.
