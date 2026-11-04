# qtilab: notes for Claude Code sessions

Research code for q-space trajectory imaging (QTI, tensor-valued diffusion MRI, Westin et al. 2016)
with a GPU-friendly PyTorch core. The scientific direction and findings so far are in
`docs/NOTES.md`; read it before proposing new work.

## Layout

- `qtilab/` is the library: tensors (Mandel conventions identical to DIPY), protocol, fit (OLS/WLS/constrained QTI+ via
  batched ADMM), scalars, simulate, and loaders (`data.py` for the DiB open data, `hhs.py` for OpenNeuro
  ds005360).
- `scripts/` holds validation, benchmarks, maps, and data fetchers. Outputs go to `results/`.
- `tests/` needs no downloaded data. `uv run pytest` takes about 10 s.

## Data

- DiB: `scripts/fetch_dib.sh data/dib`, then set `QTILAB_DIB_ROOT` (the script prints it).
- Hormone Health Study (22 subjects × 3 sessions, LTE+STE, CC0, 8.8 GB): `python scripts/fetch_hhs.py data/hhs`, then
  set `QTILAB_HHS_ROOT=data/hhs` if it lives elsewhere.

## Conventions

- b in ms/µm², diffusivities in µm²/ms. Parameter vector (28) = [ln S0, ⟨D⟩ Mandel 6, ℂ Mandel 21 in DIPY's order].
- Everything is batched over voxels and device-agnostic. Keep it that way: no per-voxel Python loops and no NumPy
  inside hot paths.
- Validate any change to fitting against DIPY (`scripts/validate_dipy.py`, `scripts/bench_constrained.py`).

## Compute

Development has been on CPU. GPU runs (cohort fits, posterior-network training) are meant to go through Modal or
the Apple-Silicon GPU (`device="mps"`).
