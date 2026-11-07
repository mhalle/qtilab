# qtilab

Batched, device-agnostic (CUDA / MPS / CPU) QTI fitting and tensor-valued diffusion
simulation in PyTorch. This is the foundation for simulation-based posterior inference
over diffusion tensor distributions.

## Modules

- `qtilab/tensors.py` — Mandel helpers. Conventions are identical to `dipy.reconst.qti`,
  so parameter vectors are interchangeable.
- `qtilab/protocol.py` — `Protocol`: one b-tensor per volume (LTE/PTE/STE or arbitrary),
  the 28-column design matrix, and b_Δ. Units are ms/µm².
- `qtilab/fit.py` — batched OLS, WLS (DIPY-equivalent weights) and the constrained fit
  (QTI+ "dc": ⟨D⟩ ⪰ 0, ℂ ⪰ 0).
  - Only voxels that violate a constraint under WLS are re-solved. Converged voxels drop out of the working set.
  - The default solver (`qtilab/ipm.py`, since 2026-11-07) is a batched log-barrier interior-point method: one 28 × 28 Newton system per voxel per step, about 20 steps.
  - The original solver, ADMM with per-voxel block scaling and adaptive ρ, is `solver="admm"`. The two agree to ≤ 1e-3 in µFA, and the interior-point solver is 5–9× faster and converges every voxel (docs/NOTES.md, 2026-11-07).
- `qtilab/scalars.py` — MD, FA, µFA, C_MD, C_μ, C_M, C_c, K_bulk, K_shear, MK, K_μ,
  V_MD, V_shear and V_iso. These match DIPY's `QtiFit` property by property.
- `qtilab/simulate.py` — hierarchical prior over discrete tensor distributions:
  - 1–3 compartments plus optional free water.
  - Each compartment has its own size, shape, orientation, dispersion and size spread.
  - The signal is the exact Laplace transform of the distribution, not the cumulant, so
    truncation bias is present.
  - Ground truth is the exact moments of the distribution.
  - Rician noise.
- `qtilab/data.py` — loader for the Szczepankiewicz–Hoge–Westin 2019 open data (DiB
  subsamples).
- `qtilab/hhs.py` — loader for the Hormone Health Study (OpenNeuro ds005360). It concatenates
  the 8 runs of each session and rigidly aligns them, rotating the b-tensors to match.

## Setup

```
uv sync                                   # or: pip install -e ".[validate]"
uv run pytest
scripts/fetch_dib.sh data/dib             # ~340 MB; prints QTILAB_DIB_ROOT to export
python scripts/fetch_hhs.py data/hhs      # 8.8 GB
```

## Install

The command-line tool, from a release tag (stable):

```
uv tool install git+https://github.com/mhalle/qtilab@v0.1.0      # then: qtilab fit ..., qtilab info ...
uvx --from git+https://github.com/mhalle/qtilab@v0.1.0 qtilab --help   # or run without installing
```

**Branches:** `main` is stable; every release is a tag on it (`v0.1.0`, ...). `dev` is where work happens and may be unstable. CI (GitHub Actions: macOS, and Linux with CPU-only torch) runs the tests on both branches and on pull requests into `main`.

## Command line

`qtilab fit` fits constrained QTI (QTI+ "dc") to NIfTI series and writes scalar maps. `qtilab info` prints the protocol (shells, shapes, design rank) without fitting, and `qtilab docs` prints every command's help and this README as one Markdown text. The command line follows [tractline](https://github.com/mhalle/tractline)'s, so the two can be merged later.

```
qtilab fit sub-01/dwi/sub-01_acq-linear_dwi.nii.gz sub-01/dwi/sub-01_acq-spherical_dwi.nii.gz --nifti out/sub-01
qtilab info data/hhs/sub-01/ses-01/dwi/sub-01_ses-01_acq-*_dwi.nii
```

- **Series** are positional NIfTI paths. Each one's `.bval` and `.bvec` are found beside it or by BIDS inheritance, as in tractline; `--bval`/`--bvec` (one per series) override them.
  - Several series are concatenated and must already be on one grid: motion/eddy-corrected and aligned. qtilab fits; it does not register.
- **The b-tensor shape** comes from the BIDS name (`acq-linear`, `acq-planar`, `acq-spherical`, or LTE/PTE/STE) or a `BTensorShape` sidecar key. Otherwise give `--btens`: `LTE`, `PTE` or `STE` for a whole series, or a file with one entry per volume (a shape, or b_Δ 1, −0.5, 0). Give one for all series, or one per series in order.
- **Mask:** `--mask median-otsu` (the default: DIPY's median_otsu with its defaults, the same code as tractline's) or a mask NIfTI.
- **Output:** `--nifti out/sub-01` writes `out/sub-01_<map>.nii.gz` (float32; MD in µm²/ms) and `out/sub-01_mask.nii.gz` (the fitted voxels; collapsed fits with ⟨D⟩ = 0 left out).
  - It also writes a provenance sidecar `out/sub-01.qtilab.json`, as tractline writes `<trx>.tractline.json`. Paths in it are relative to the BIDS dataset unless `--record-absolute-paths`.
  - The default maps are `md fa ufa c_md c_mu c_c k_bulk k_shear mk k_mu`. Choose others with `--maps`, or `all`. `--params` also writes the 28 parameters.
- **Messages and errors:** progress goes to standard error. `--json FILE` (or `-` for standard output) writes the run's summary, and `-q` prints nothing but errors. Input errors end in one line; `QTILAB_DEBUG=1` shows the traceback.
- **Defaults:**
  - the interior-point solver (`--solver admm` for the original);
  - `--device auto`, which picks the Apple GPU, then CUDA, then the CPU;
  - float64 on the CPU and float32 on a GPU.
  - `--method wls` gives the unconstrained, DIPY-equivalent WLS.
- **Timing:** one HHS session (95,801 voxels, 107 volumes) takes 11 s on the M1 Max CPU and 4 s on MPS. That matches the library fit (the saved maps to ≤ 1.2e-3 in µFA, median 1e-7).
  - Note: `prepare_session.py` uses a lighter median_otsu (radius 2, 1 pass), so its masks are larger than `--mask median-otsu`'s (95.8k vs 85.9k voxels on sub-01 ses-01).

## Validation (2 CPU cores, float64, DiB_217: 72,532 brain voxels × 217 volumes)

| | result |
|---|---|
| Design matrix vs DIPY | max difference 2e-15 |
| WLS vs DIPY WLS, all voxels | median scalar difference ~1e-9; 4.2 s vs DIPY's 24.6 s |
| Constrained vs DIPY SDPdc (CLARABEL), 1000 violating voxels | median difference ≤ 8e-6, 99th percentile ≤ 2e-4 |
| Same comparison, DIPY's own SCS vs CLARABEL | same size: median ~1e-6, 99th percentile ~1e-4 |
| Constrained, whole brain | 65 s, 100% converged |

**Constrained-fit timing in detail**
- 59,968 of 72,532 voxels (83%) had a non-PSD covariance under WLS and needed ADMM.
- That is about 1.1 ms per constrained voxel on 2 CPU cores.
- DIPY takes 22 ms per voxel with CLARABEL and 66 ms per voxel with SCS.

**Simulator**
- Fits on tiny b-values recover the ground truth (median error ≤ 1e-4).
- 100,000 voxels × 217 volumes simulate in 5.4 s on CPU in float32.

Scripts: `scripts/validate_dipy.py`, `scripts/bench_constrained.py`,
`scripts/check_simulator.py`, `scripts/maps.py`. Outputs go in `results/`.
