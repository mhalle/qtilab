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
