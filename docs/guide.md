# qtilab: the command-line guide

This guide explains how to use the `qtilab` command. It covers what the program does, what it needs as input, what
it writes, and the choices you can make when you run it. It assumes you know what tensor-valued diffusion encoding
and q-space trajectory imaging (QTI) are. The code is at https://github.com/mhalle/qtilab.

qtilab is research software. It is not approved for clinical use.

## 1. What the program does

`qtilab` fits the QTI model to diffusion MRI data acquired with more than one b-tensor shape, such as linear (LTE)
and spherical (STE) encoding. It writes the QTI scalar maps, such as MD, FA, µFA and the kurtosis terms, as NIfTI
files.

The fit is the constrained one from QTI+ (Herberthson et al. 2021), the variant called "dc". It requires the mean
diffusion tensor and the covariance tensor to be positive semidefinite. An unconstrained fit can give a negative
variance in noisy voxels, and the scalars computed from it can then fall outside their valid range. The constraints
prevent that.

The program fits every voxel at once, on the CPU or a GPU. On an Apple M1 Max, one session of the Hormone Health
Study (about 86,000 brain voxels and 107 volumes) takes about 4 seconds on the GPU and about 12 seconds on the CPU.

The program does not correct for head motion, eddy currents or susceptibility distortion. Do that first with another
tool. Section 8 says more about this.

The program has three commands:

| Command | What it does |
|---|---|
| `qtilab info` | Reads the series and prints the protocol: the shells, the encoding shapes and the rank of the QTI design. It does not fit anything. |
| `qtilab fit` | Fits the model and writes the maps. |
| `qtilab docs` | Prints this guide, the README and the help for every command as one Markdown text. |

## 2. Installing and running it

The program needs Python 3.11 or later. The easiest way to run it is with [uv](https://docs.astral.sh/uv/). This
command runs a released version without a separate install step:

```
uvx --from git+https://github.com/mhalle/qtilab@v0.2.0 qtilab --help
```

The first run downloads PyTorch and the other dependencies, so it takes a while. Later runs reuse them.

To install the command so that `qtilab` is on your path:

```
uv tool install git+https://github.com/mhalle/qtilab@v0.2.0
```

The `@v0.2.0` part names a release. Use the same release for every scan in a study, so that all of them are processed
with the same code. `qtilab --version` prints the version you are running. The `main` branch always holds the newest
release. The `dev` branch is where development happens and may be unstable.

From a checkout of the repository, `uv run qtilab ...` runs the program with the checkout's code.

## 3. A first run

Start with `qtilab info`. It shows how the program reads your data before you spend time on a fit. Here it reads the
eight runs of one session from the Hormone Health Study (OpenNeuro ds005360):

```
qtilab info data/hhs/sub-01/ses-01/dwi/*_dwi.nii
```

The first line of the output lists each series with its number of volumes, its encoding shape and where the shape
came from. The rest looks like this:

```
107 volumes (3 with b < 50 s/mm^2); b-values read as s/mm2
  LTE b =   100 s/mm^2:   6 volumes
  LTE b =   500 s/mm^2:   6 volumes
  LTE b =  1000 s/mm^2:  10 volumes
  LTE b =  1500 s/mm^2:  16 volumes
  LTE b =  2000 s/mm^2:  30 volumes
  STE b =   100 s/mm^2:   6 volumes
  STE b =   500 s/mm^2:   6 volumes
  STE b =  1000 s/mm^2:   9 volumes
  STE b =  1500 s/mm^2:  15 volumes
QTI design rank 23 of 28 (rank-deficient: ...)
```

Check three things here. The shells should be the ones you acquired. Each series should have the shape you expect.
The rank tells you how much of the model the protocol can measure; section 4.4 explains it.

Then fit. This example fits two corrected series and writes the maps with the prefix `out/sub-01`:

```
qtilab fit sub-01_acq-linear_desc-preproc_dwi.nii.gz sub-01_acq-spherical_desc-preproc_dwi.nii.gz --nifti out/sub-01
```

While it runs, the program prints its progress. For one Hormone Health Study session it looks like this:

```
running on mps (float32): 85651 voxels in the mask (median-otsu), constrained fit (ipm)
4 voxels collapsed to <D> = 0 (no attenuation above the floor): left out of the maps and the fit mask
fit in 3.9 s: 58377 voxels needed the constrained solve (68%), converged 0.9850
11 NIfTI files at out/sub-01_*.nii.gz; provenance out/sub-01.qtilab.json; 5.8 s in all
```

The maps are now in `out/`, one file per scalar: `out/sub-01_md.nii.gz`, `out/sub-01_ufa.nii.gz`, and so on.

## 4. Inputs

### 4.1 Series

Give the diffusion series as NIfTI paths after the command name. You can give one series or several. The program
joins several series along the volume axis and fits them together. This is how you combine, for example, separate
LTE and STE runs.

All the series must be on the same voxel grid, with the same affine. If they are not, the program stops and says so.
Being on the same grid is not enough, though. The series must also be aligned with each other, and the program
cannot check that.

### 4.2 b-values and b-vectors

Each series needs a `.bval` and a `.bvec` file in FSL's format. The program looks for them the way tractline does:

- Inside a BIDS dataset (a folder above the series contains `dataset_description.json`), it uses BIDS's inheritance
  principle. The most specific matching file wins.
- Outside a BIDS dataset, it uses the files next to the series that have the same name. This is the layout dcm2niix
  writes.

If the files are somewhere else, give them with `--bval` and `--bvec`. Give one of each for every series, in the same
order as the series.

The program decides the b-value units from the values. If any b-value is above 100, it reads them as s/mm²;
otherwise it reads them as ms/µm². Use `--bval-units` to set the units yourself.

### 4.3 The b-tensor shape

QTI needs to know the shape of each volume's b-tensor. Neither NIfTI nor BIDS has a standard field for it, so the
program looks in two places:

1. A `BTensorShape` key in the series' JSON sidecar. Its value can be `LTE`, `PTE` or `STE`, or a b_Δ value: 1 for
   linear, −0.5 for planar, 0 for spherical.
2. The `acq-` entity in the file name. `acq-linear`, `acq-planar` and `acq-spherical` work, and so do `acq-lte`,
   `acq-pte` and `acq-ste`. The Hormone Health Study names its files this way.

If neither is there, give `--btens`. It takes a shape for the whole series (`--btens STE`) or the path of a text file
with one entry per volume. Each entry is a shape name or a b_Δ value. Give one `--btens` to use for every series, or
one per series in order.

`qtilab info` reports where each series' shape came from. Check it before you fit.

### 4.4 What the protocol can measure

The QTI model has 28 parameters: the signal at b = 0, the six elements of the mean diffusion tensor, and the 21
elements of the covariance tensor. A protocol of LTE and STE only can measure at most 23 combinations of them, not
all 28. The data say nothing about the rest of the covariance. The scalar maps do not depend on that part, so they are
well defined. The 28 parameters written by
`--params` do depend on it, so treat them with care.

If the rank is below 23, the program warns you. Such a protocol cannot separate the covariance terms that QTI is
built on. A reduced model, such as RICE, may suit it better.

### 4.5 The mask

By default the program fits the voxels inside a brain mask that it makes from the mean b = 0 image. It uses DIPY's
`median_otsu` with its default settings. This is the same code that tractline uses. To use your own mask, give a
NIfTI file on the same grid: `--mask brain_mask.nii.gz`. Any voxel above zero is fitted.

The default mask is fairly tight. If you need the edge of the brain, make a mask with another tool.

## 5. Outputs

### 5.1 The maps

`--nifti out/sub-01` writes each map as `out/sub-01_<name>.nii.gz`. If the value ends in a slash, as in
`--nifti out/`, the files go into that folder with the prefix `qti`. The program creates the folder if it does not
exist.

The maps are float32 and use the series' affine. Voxels outside the mask are zero. By default the program writes the
first ten maps in this table. Choose others with `--maps`, as a comma-separated list, or write them all with
`--maps all`.

| Name | Scalar | Units |
|---|---|---|
| `md` | Mean diffusivity | µm²/ms |
| `fa` | Fractional anisotropy of the mean tensor | none |
| `ufa` | Microscopic fractional anisotropy, µFA | none |
| `c_md` | Normalized variance of the isotropic diffusivities, C_MD | none |
| `c_mu` | Microscopic anisotropy, C_µ (µFA squared) | none |
| `c_c` | Orientation coherence, C_c = C_M / C_µ | none |
| `k_bulk` | Bulk kurtosis, K_bulk | none |
| `k_shear` | Shear kurtosis, K_shear | none |
| `mk` | Total kurtosis, K_bulk + K_shear | none |
| `k_mu` | Microscopic kurtosis, K_µ | none |
| `c_m` | Macroscopic anisotropy, C_M (FA squared) | none |
| `v_md` | Variance of the isotropic diffusivities, V_MD | (µm²/ms)² |
| `v_shear` | Shear variance, V_shear | (µm²/ms)² |
| `v_iso` | Isotropic variance, V_iso | (µm²/ms)² |
| `s0` | The fitted signal at b = 0 | the series' signal units |

The formulas are those of Westin et al. (2016). They match DIPY's `QtiFit`, property by property.

`--params` also writes the 28 fitted parameters as one 4D file, `out/sub-01_params.nii.gz`. The order is ln S0, then
the mean tensor in Mandel notation (6 values), then the covariance in Mandel notation (21 values). This is DIPY's
order. The tensors are in the frame of the b-vectors. Section 4.4 explains why some of these parameters are not
determined by the data.

### 5.2 The fit mask

`out/sub-01_mask.nii.gz` marks the voxels that were fitted and kept. It is the brain mask minus any voxel whose fit
collapsed. A fit collapses when the signal shows no decay above the noise floor, which happens in a few dark voxels
at the edge of the mask. The constrained fit then returns a mean tensor of zero, and every ratio map is meaningless
there. The program leaves those voxels out and reports how many there were. In the example above there were 4 out of
85,651.

### 5.3 The provenance file

Next to the maps the program writes `out/sub-01.qtilab.json`. It records how the maps were made:

- the versions of qtilab, PyTorch and Python;
- each input series, with its number of volumes and where its shape came from;
- the protocol summary that `qtilab info` prints;
- the mask, the method, the solver, the device and the precision;
- the number of voxels fitted and the number that collapsed;
- the maps written and their units.

Paths are recorded relative to the BIDS dataset, or as bare file names outside one. This lets you share the file
without revealing your folder layout. Add `--record-absolute-paths` to record full paths instead.

tractline writes the same kind of file next to its output, `<name>.tractline.json`.

### 5.4 The run summary

`--json summary.json` writes a summary of the run as JSON. It holds the same settings and counts as the provenance
file, plus the timings, the number of voxels that needed the constrained solve, the fraction of them that converged, and the
solver's statistics. `--json -` writes it to standard output, so you can pipe it to another program. `qtilab info`
takes `--json` too and writes the protocol summary.

## 6. How the fit runs

### 6.1 Method

`--method constrained` is the default. The program first computes the weighted least squares (WLS) fit, with the
same weights as DIPY. Voxels where that fit already meets the constraints keep it. The rest are solved again with the
constraints. In brain data this is usually most voxels: 68% in the example above.

`--method wls` stops after the WLS step. It gives the same result as DIPY's WLS fit. Use it when you want to compare
with an unconstrained analysis.

### 6.2 Solver

`--solver ipm` is the default. It is an interior-point method that solves every voxel at the same time. In float64
it converges in every voxel. `--solver admm` selects the older ADMM solver, which was the default before 7 November
2026. The two agree to about 0.001 in µFA, and the interior-point method is 5 to 9 times faster. Both have been
checked against DIPY's constrained fit, which uses the CLARABEL and SCS solvers. Use ADMM only to reproduce older
results exactly.

### 6.3 Device and precision

`--device auto` is the default. It uses the Apple GPU (MPS) if there is one, then a CUDA GPU, then the CPU. You can
choose one with `--device mps`, `--device cuda` or `--device cpu`.

`--precision auto` uses float64 on the CPU and float32 on a GPU. The Apple GPU cannot compute in float64. In
float32 the fit converges in about 98% of voxels. The rest stop at the limit of float32's precision, and their maps
still agree with the float64 fit to about 0.001 in µFA.

On the CPU, `--workers N` sets the number of threads PyTorch uses.

## 7. Messages and errors

Progress messages go to standard error. Standard output stays free for `--json -`. Add `-q` to print nothing but
errors.

When the program cannot read its input, it stops with one line that names the file and the problem. For example,
it stops if a series has no `.bval` file, or if the number of b-values does not match the number of volumes. Set
`QTILAB_DEBUG=1` to see the full Python traceback instead.

## 8. Limits

- The program fits. It does not correct the data. Correct each series for head motion, eddy currents and
  susceptibility distortion first, and align the series with each other. tractline's `--motion-eddy` corrects
  within one series. It treats eddy currents as linear in the gradient direction, which has not been checked for
  spherical or planar encoding.
- The constraints keep the tensors valid, but they do not keep every scalar inside its textbook range. A few voxels
  can still have µFA or C_c slightly above 1. This is a known property of the "dc" constraints, not an error in the
  fit.
- In voxels near the collapse described in section 5.2, the kurtosis maps can take extreme values. Summarize regions
  of K_bulk, K_shear and K_µ with medians, not means.
- The program assumes the b-tensors have one of three shapes: linear, planar or spherical. It does not yet read
  arbitrary b-tensors or encoding waveforms.

## 9. Relation to tractline

qtilab's command line follows the conventions of [tractline](https://github.com/mhalle/tractline), so that the two
programs could later be combined. Both take the series as positional paths and find the b-tables the same way. Both
use the same `median-otsu` mask, the same `--device auto` order, the same `--json` and `-q` options, the same kind of
provenance file and the same one-line errors. Both have a `docs` command.

## References

- Westin C-F, Knutsson H, Pasternak O, et al. Q-space trajectory imaging for multidimensional diffusion MRI of the
  human brain. NeuroImage 2016;135:345–362.
- Herberthson M, Boito D, Dela Haije T, et al. Q-space trajectory imaging with positivity constraints (QTI+).
  NeuroImage 2021;238:118198.
- Garyfallidis E, Brett M, Amirbekian B, et al. Dipy, a library for the analysis of diffusion MRI data. Frontiers in
  Neuroinformatics 2014;8:8.
