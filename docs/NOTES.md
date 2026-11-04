# QTI work log

## 2026-11-04 — qtilab skeleton built and validated

**What exists.** `qtilab` is a PyTorch package (CUDA / MPS / CPU) with four parts:
- protocol and design matrix
- batched OLS, WLS and constrained fits (QTI+ "dc", solved by ADMM)
- QTI scalars
- a simulator over discrete tensor distributions with Rician noise

Tensor conventions are identical to DIPY's. It currently lives in the session sandbox (`/home/claude/qtilab`); it is not yet in a repository.

**Data.** Szczepankiewicz–Hoge–Westin 2019 open brain data (DiB subsamples), cloned from GitHub. DIPY's fetcher was blocked from the sandbox.

**Validation** (2 CPU cores, DiB_217: 72,532 voxels × 217 volumes)
- The design matrix equals DIPY's (difference ~2e-15).
- WLS matches DIPY to ~1e-9 (median scalar difference) and runs about 6× faster (4.2 s vs 24.6 s).
- The constrained fit agrees with DIPY's SDPdc/CLARABEL about as closely as DIPY's own two solvers (SCS and CLARABEL) agree with each other: median ≤ 8e-6, 99th percentile ≤ 2e-4.
  - Whole brain: 65 s. That is about 1.1 ms per constrained voxel, against 22 ms (CLARABEL) and 66 ms (SCS) per voxel in DIPY.
- Simulator: tiny-b fits recover the ground truth. 100k voxels × 217 volumes simulate in 5.4 s (float32, CPU).

**Observations worth following up**
1. **Constraints matter in most voxels, not just at the edges.** 83% of brain voxels have a non-PSD covariance under WLS; only 0.4% have a non-PSD mean tensor.
2. **Truncation bias grows with heterogeneity.** Simulated with no noise at b_max = 2 ms/µm², split by true K_bulk tercile:

   | true K_bulk | MD bias (µm²/ms) | MK bias |
   |---|---|---|
   | low | −0.004 | −0.06 |
   | high (free-water-like mixtures) | −0.22 | −1.5 |

   A posterior network trained against the *true* distribution moments could undo this. That is a concrete, testable advantage over the classical fit.
3. **Constraints may bias µFA upward.** In noisy simulation, the constraints reduced FA bias but flipped µFA bias from −0.01 to +0.03 and raised µFA error (MAE 0.085 → 0.10).
   - *Hypothesis:* forcing ℂ ⪰ 0 inflates shear variance in low-anisotropy tissue, which would matter for gray-matter µFA.
   - The effect depends on the prior; check with a brain-like prior.
4. **Stripe texture in the maps.** The K_bulk and C_MD maps from the real brain show a fine vertical stripe pattern. It could be noise or an acquisition/reconstruction artifact; inspect it before using those maps.

**Follow-up: is the covariance really on the boundary?**
- In simulation, the *true* covariance is often nearly singular: a median of 2 of its 6 eigenvalues fall below 1% of the largest (quartiles 0–4, broad prior).
- In the real brain, the constrained fits sit on the boundary: median 1 near-zero eigenvalue (quartiles 1–2).
- *Reading:* the 83% violation rate largely reflects real tissue covariances lying on or near the edge of the PSD cone. Noise then pushes them over. It is not mainly a defect of the fit.

**Implications for the planned directions**
- **Posterior network:** use a prior that puts weight on low-rank covariances, and report posteriors rather than boundary-pinned point estimates.
- **Surprise maps:** define surprise against the exact-distribution model, not QTI's truncated cumulant. Otherwise heterogeneous voxels (CSF partial volume) look surprising for the wrong reason, and real restriction or exchange gets mixed up with that bias.
- **µFA-based tractography stopping:** an upward µFA bias in gray matter from the constraints would shrink the white–gray contrast the rule depends on. Truncation bias also distorts µFA at CSF boundaries.
- **Test-retest:** a constraint bias that depends on SNR would look like tissue change between sessions with different SNR.
- **Stripe texture:** a natural first test case for surprise maps, which should be checked for artifact versus biology.

## 2026-11-04 (later) — Hormone Health Study downloaded and first session fit

**Download.** OpenNeuro ds005360 (CC0), 8.8 GB, now in the sandbox at `/home/claude/data/hhs`:
- 22 subjects × 3 sessions × 8 runs; all 528 runs present.
- Siemens Prisma fit, TE 80 ms, 2×2×4 mm.
- Linear encoding: b 0/100/500, 1000, 1500, 2000. Spherical encoding: b 0, 100/500, 1000, 1500.
- No planar encoding, so the protocol has rank 23 of 28. All runs are phase-encoded j−, so there is no reversed pair for susceptibility correction.

**Loader.** `qtilab/hhs.py` concatenates the 8 runs into one series. Each run's mean image is rigidly registered to run 1 (mutual information), and the linear-encoding directions are rotated to match.
- sub-01 ses-01 needed up to 1.3 mm translation and 0.6° rotation between runs. That is enough to matter at 2 mm voxels.

**Fit.** The constrained fit on sub-01 ses-01 took about 4.5 min on CPU.
- 61% of brain voxels needed the constraint, versus 83% in DiB_217.
- About 10–15% of voxels stopped at the ADMM iteration cap. This is because the rank-deficient protocol leaves the objective flat in some directions. Their scalars still agree with the converged runs to about 1e-4, so the stopping rule is stricter than needed for this protocol.
- Maps look as expected. µFA is high throughout white matter and in deep gray matter.
- **No stripe texture** in the K_bulk / C_MD maps, so the DiB stripes are probably specific to that dataset.

**Known gaps**
- Runs 2–4 and 7–8 contain no b = 0 volume, so differences in intensity scaling between runs cannot be checked or corrected. This could bias the fit.
- No eddy-current or susceptibility correction yet; the latter isn't possible without a reversed pair.

## 2026-11-04 (evening) — cheap checks: stripes, run scaling, brain-like prior

The code is now in **github.com/mhalle/qtilab**, and the repo copy of these notes is `docs/NOTES.md`.

**Stripe texture in DiB (resolved: correlated noise).**
- The raw images show no stripes: the mean b0, the mean b = 2 spherical image, and their ratio all look clean.
- The noise, measured as the difference between repeated spherical-encoding volumes, is spatially correlated, and more so along one in-plane axis: lag-1 autocorrelation 0.26 vs 0.13 (0.06 across slices). That is typical of reconstruction filtering or interpolation along the phase-encoding axis.
- K_bulk and C_MD amplify noise, so the correlated grain shows up as elongated "stripes." It is not biology and not a structured artifact.
- Two consequences:
  - Real noise is not independent between voxels, as the simulator assumes.
  - Any spatial-context model or surprise map has to model correlated noise.
- The DiB header reports 1 mm voxels, which is not the acquired resolution.

**Hormone Health intensity scaling between runs (reassuring).** Medians across all 66 sessions, with the 5th–95th percentile range:

| comparison | ratio | expected |
|---|---|---|
| b0 run 5 / b0 run 1 | 0.998 [0.982–1.002] | 1 |
| linear / spherical at b = 100 (runs 1 vs 6) | 1.002 [0.997–1.010] | about 1.001 |
| linear / spherical at b = 500 (runs 1 vs 6) | 1.030 [1.025–1.038] | small positive excess from microscopic anisotropy |

- The b = 500 excess implies an anisotropic variance of about 0.24 µm⁴/ms², a plausible size.
- The scanner does not rescale between these runs. Runs without b0 cannot be checked directly, but the evidence now argues against a scaling problem.

**Brain-like prior (`simulate.sample_brain`).** Four tissue classes, all compartments Gaussian:
- **WM:** 1–3 bundles, each an axon stick plus an extra-axonal zeppelin.
- **GM:** dispersed neurites, a soma-like isotropic compartment and extracellular space.
- **WM+CSF and GM+CSF:** the same tissue plus 10–90% free water.

Main results (12k voxels, both real protocols):
1. **Truncation bias is the dominant error, bigger than noise at realistic SNR, and systematic by tissue.** Noise-free fits at b_max = 2:
   - µFA is overestimated by +0.03 to +0.06 in tissue and by +0.06 to +0.11 at CSF boundaries.
   - FA is overestimated by about +0.11 at WM/CSF boundaries.
   - MD is underestimated by 0.04–0.06 in tissue and by 0.13–0.16 at CSF boundaries.
   - C_MD and K_bulk (MK_I) are underestimated by about half (K_bulk −0.3 to −0.58).
   - Free water is nearly invisible above b ≈ 1, so the fit misses its share of the variance. QTI values are therefore *apparent* quantities tied to the b-range.
   - Open question: should a posterior network target the full distribution moments, or tissue moments plus a separate free-water fraction?
2. **The constraints' push on µFA is real but secondary.** It adds about +0.01 in GM and +0.03 to +0.05 at CSF boundaries, in the same direction as the truncation bias. Both are worst at CSF boundaries, where tractography stopping decisions happen.
3. **Revision of the "tissue sits on the boundary" reading.**
   - Under the brain-like prior, the true covariance is nearly singular in WM (median 1 near-zero eigenvalue of 6) and at WM/CSF boundaries (median 3), but not in GM (0).
   - Even noise-free, WLS gives a non-PSD covariance in about 40% of WM voxels and 58–76% of WM+CSF voxels, but about 0% of GM voxels.
   - So violations come from three sources: truncation bias (WM, boundaries), noise (GM), and genuinely near-singular covariances (WM, boundaries).
   - For rank-23 protocols such as Hormone Health, the unconstrained covariance is not unique, so violation rates are not comparable across protocols.
   - The earlier claim that violations are "mostly tissue at the edge" was too strong.

**Implications**
- **H2 (bias in mixed tissue)** is now the strongest, most concrete hypothesis. It affects every published QTI value at typical b-ranges, and it is directly testable with the posterior network.
- **H6 (µFA-based stopping)** needs bias-corrected µFA, since both biases inflate µFA where stopping decisions are made.
- **The simulator** needs correlated noise to match real data.

**Next**
- Posterior network (set encoder + normalizing flow) on the brain-like prior: a small CPU smoke test here, real training on GPU.
- Decide the target definition: full moments vs tissue moments plus free-water fraction.
- Correlated-noise option in the simulator.
- Cohort fits and test-retest (H7): GPU, or CPU batches here.
