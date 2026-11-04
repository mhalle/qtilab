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

## 2026-11-04 (night) — posterior network smoke test

**Design decision.** The network's main targets are *tissue* moments (free water removed) plus a free-water fraction. It also reports the full-distribution moments that QTI's cumulant fit targets, so the two can be compared directly. Targets: f_fw, md_t, fa_t, ufa_t, c_md_t, md, ufa, c_md.

**Model** (`qtilab/posterior.py`):
- A set encoder reads (b-tensor, normalized signal) pairs and pools them (mean + max) into a 64-dimensional embedding.
- A zuko NSF flow, conditioned on the embedding, gives the posterior. Bounded targets use logit space; diffusivities use log space.
- Training: 3,000 steps × 512 voxels, simulated on the fly from the brain-like prior through sub-01 ses-01's actual protocol. SNR 15–60. About 11 minutes on 2 CPU cores.

**Held-out simulation**, 4,000 voxels from the same prior, SNR 20–35. Median bias / median |error|:

| class | µFA tissue: net | µFA tissue: QTI | µFA full: net | µFA full: QTI |
|---|---|---|---|---|
| WM | −0.005 / 0.025 | −0.029 / 0.037 | −0.036 / 0.059 | +0.070 / 0.072 |
| GM | +0.014 / 0.052 | −0.062 / 0.076 | −0.020 / 0.063 | +0.063 / 0.073 |
| WM+CSF | −0.038 / 0.047 | −0.356 / 0.356 | 0.000 / 0.026 | +0.176 / 0.176 |
| GM+CSF | +0.022 / 0.075 | −0.260 / 0.260 | +0.027 / 0.036 | +0.142 / 0.142 |

- **MD.** QTI's MD misses tissue MD by about 0.9 µm²/ms at CSF boundaries, because it reports the mixture. The network's tissue-MD error is 0.03–0.06.
- **C_MD.** The network's full C_MD is unbiased at CSF boundaries, where QTI underestimates it by 0.06–0.07. In WM and GM the network's error is about 0.05–0.06 versus QTI's 0.11–0.13.
- **Coverage.** The network is slightly overconfident: 90% intervals cover 81–90% of the truth, and 50% intervals cover 44–51%. That is acceptable for a short CPU run.
- **Caveat.** This test is in-distribution: same prior, Gaussian compartments, independent Rician noise. It shows the method works, not that it is right on real tissue.

**Real data: sub-01 ses-01, slice 20** (`results/hhs_sub01_ses01_posterior_maps.png`)
- The free-water map is anatomically sensible: ventricles and sulci. Median f_fw is 0.08; 15% of voxels have f_fw > 0.5.
- Tissue µFA is higher in WM than GM, and its uncertainty is largest where tissue is scarce (CSF), as it should be.
- Tissue MD is nearly flat. It is probably prior-dominated, since the prior on tissue diffusivity is narrow.
- Classical vs network µFA:

  | region | classical µFA | network full µFA | network tissue µFA |
  |---|---|---|---|
  | WM-like (FA > 0.4, f_fw < 0.2) | 0.87 | 0.69 | 0.84 |
  | GM-like (FA < 0.2, f_fw < 0.2) | 0.54 | 0.47 | 0.61 |
  | CSF-ish (f_fw > 0.5) | 0.32 | 0.13 | 0.63 |

- **The sim-to-real gap.**
  - The gap between classical and network full µFA matches simulation in GM (+0.07 observed vs +0.06 predicted) and at CSF boundaries (+0.20 vs +0.14 to +0.18).
  - In WM it is much larger than simulation predicts (+0.18 vs +0.07).
  - Most likely, the brain-like prior's WM is less anisotropic than real WM: true median µFA 0.74 in the prior, while the classical fit gives about 0.87. The network then shrinks toward the prior.
  - This is the first concrete sim-to-real mismatch. Fixes are to widen the WM prior and to add posterior-predictive checks, which would also be the first form of surprise map.

**Engineering notes**
- zuko NSF sampling on CPU is slow and memory-hungry (about 7 s and 2 GB per 256 voxels × 200 samples), so sampling is now chunked at 256 voxels.
- A larger chunk (4096) ran the 7 GB sandbox out of memory and restarted the VM. Training had already finished and the model was saved.
- Whole-brain sampling belongs on GPU.

## 2026-11-04 (late) — the quick fixes: widened prior, real SNR, surprise check

**Widened prior (`BRAIN_PRIORS[2]`, now the default; v1 kept for comparison).**
- Changes: axonal fraction up to 0.85; extra-axonal D_Δ up to 0.8; wider tissue diffusivities in WM and GM.
- Simulated WM full-distribution µFA: median 0.80 (was 0.74), 95th percentile 0.94.

**Real SNR is high.** Estimated from the 15 repeated spherical-encoding b = 1500 volumes (identical expected signal), b = 0 SNR in sub-01 ses-01 has a median of about 67 (10–90%: 28–139). The smoke test trained on 15–60, so the v2 network trains on 20–150, log-uniform.

**Correlated noise: not applied to the voxelwise network, on purpose.** The correlation is between neighboring voxels, and the network sees one voxel at a time. It matters for spatial models and for cluster-level inference on surprise maps.

**Calibrated surprise check (`scripts/surprise_check.py`).** For each voxel: the RMS distance from its normalized signal to the nearest of 100k noise-free simulated signals, in units of that voxel's noise sd (taken from the spherical-encoding repeats).
- It is calibrated against simulated noisy voxels processed identically, *within SNR bins*. The reference distribution rises steeply with SNR, so unconditional calibration made high-SNR (posterior, coil-near) regions look surprising.
- Real-data noise from repeats also absorbs residual motion, which makes the check conservative.

Percentage of real voxels above the simulated 95th / 99th percentile (5% / 1% expected if the prior explains the data):

| region | prior v1 | prior v2 |
|---|---|---|
| WM-like | 60% / 25% | 32% / 7% |
| GM-like | 4% / 1% | 1% / 0% |
| CSF-ish | 63% / 47% | 51% / 36% |
| whole slice | 29% / 15% | 17% / 8% |

- Gray matter is explained.
- The widened prior halves WM misfit, but about a third of WM voxels remain unexplained, concentrated in deep, coherent WM. This is consistent with diffusion-time (restriction) effects that Gaussian compartments cannot represent, which the literature predicts are largest in coherent WM. It is a first data-driven pointer for H5; one session is not evidence.
- CSF-edge misfit most likely reflects residual motion between runs and CSF pulsation (artifact, not biology).

**v2 network** (3,000 steps, about 12 min CPU)
- Held-out, in-distribution: similar to v1.
  - Tissue µFA error 0.024 (WM) to 0.083 (GM+CSF).
  - Tissue MD error 0.04–0.09; C_MD error 0.01–0.04.
  - QTI's tissue-MD error at CSF boundaries is about 0.9.
  - Coverage: 90% intervals contain 82–90% of true values.
- Real slice, v1 → v2:

  | region | classical µFA | network tissue µFA | network full µFA |
  |---|---|---|---|
  | WM-like | 0.87 | 0.84 → **0.87** | 0.69 → 0.68 |
  | GM-like | 0.54 | 0.60 → 0.62 | 0.47 → 0.47 |
  | CSF-ish | 0.31 | 0.62 → 0.73 | 0.14 → 0.15 |

- **Correction: the WM gap was not a prior-width problem.**
  - Network tissue µFA now matches classical µFA in WM (0.87). The gap is between classical µFA and the network's *full-distribution* µFA.
  - In simulated WM, full µFA ≈ tissue µFA − 1.93 × f_fw. Each 1% of free water lowers full µFA by about 0.02, and the classical fit at b ≤ 2 barely sees free water.
  - The network infers f_fw of about 0.09 in real WM (low-b decay matches simulated WM closely). That alone accounts for most of the 0.19 gap.
  - So full-distribution µFA is a fragile quantity, dominated by a small, poorly identifiable free-water fraction. This strongly supports reporting **tissue µFA plus f_fw** rather than full moments.
- **Open question:** is about 9% free water in WM real (partial volume, interstitial fluid) or overestimated? It sits near the top of the WM prior (0–10%). Test it with spatial pattern (proximity to ventricles) and test-retest across sessions.

**Engineering:** held-out evaluation peaks at about 5.6 GB, close to the sandbox's 7 GB. Training and evaluation in one process ran out of memory once, but the trained model had already been saved. Run evaluation separately here.

## 2026-11-04 (evening, cont.) — does time dependence explain the deep-WM surprise? Not as modeled

**Setup.** The HHS waveforms are not distributed with the data, so each encoding's frequency content is unknown.
- Time dependence (`simulate.sample_tau`, `simulate.signal_td`): each compartment's diffusivities scale by 1 + τ(√ω − 1). Linear encoding is at ω = 1 and spherical at ω = r, the unknown spectral ratio. τ ranges are per compartment; free water has τ = 0.
- Control: spherical-encoding b-values off by a factor (1 + ε). A calibration error like this would also act differently on LTE and STE.
- Stage 1 lets r and ε vary per voxel. A real global effect should make the best-matching value agree across voxels.
- Stage 2 fixes r (or ε) at the consensus value, so each alternative adds exactly one global parameter.
- Script: `scripts/surprise_td.py`.

**Results** (percentage of real voxels above the simulated 95th / 99th percentile; expected 5% / 1%):

| model | WM-like | GM-like | CSF-ish |
|---|---|---|---|
| v2, time-independent | 33% / 9% | 2% / 0% | 51% / 37% |
| time dependence, r free per voxel | 29% / 7% | 1% / 0% | 50% / 36% |
| STE b-scale error, ε free per voxel | 24% / 5% | 1% / 0% | 47% / 31% |
| time dependence, r fixed at 1.46 | 34% / 5% | 2% / 0% | 51% / 36% |
| b-scale, ε fixed at +0.076 | 30% / 7% | 2% / 1% | 51% / 35% |

- Best-matching r in WM: median 1.46, IQR 0.78–2.51. This is barely narrower than the prior (IQR 0.5–2), so there is no consensus.
- Best-matching ε is positive in WM (+0.054) and CSF (+0.076) but near zero in GM (+0.011). A real calibration error would be global, so this is inconsistent with one.
- **Conclusion:** neither a simple time-dependence model nor an STE calibration error explains the deep-WM surprise or the CSF misfit. H5 is not supported *as modeled*. The simple model (isotropic scaling per compartment, a single unknown spectral ratio) may be too crude, but the residuals point somewhere else first.

**Residual analysis** (classical WLS fit, residuals in noise units, by run):

| | run 1 | run 2 | run 3 | run 4 | run 5 | run 6 | run 7 | run 8 |
|---|---|---|---|---|---|---|---|---|
| WM | 1.80 | **2.01** | 1.46 | 0.90 | 2.20 | 1.65 | 1.24 | 1.01 |
| GM | 2.09 | 1.64 | 1.39 | 0.78 | **3.56** | 2.02 | 1.30 | 1.02 |

- **Low-b volumes misfit in every tissue (runs 1, 5, 6).** The noise sd comes from the b = 1500 spherical repeats, which see mostly thermal noise. Physiological and pulsation noise scales with signal and is largest at low b. So the noise model is incomplete, and that most likely explains much of the CSF and low-b misfit.
- **WM has a specific excess in run 2 (LTE, b = 1000)**: 4 of its worst 8 volumes are there. Run 2 also needed the largest rotation in run-to-run alignment (0.61°). An error that depends on gradient direction, such as within-run motion, uncorrected eddy currents or b-vector rotation, would hit anisotropic WM hardest and spare nearly isotropic GM.
- **Reading:** the WM surprise is, at least in part, an acquisition / preprocessing signature rather than tissue microstructure. The surprise statistic did its job. It flagged that the data deviate, and the residual structure says why. It cannot yet distinguish biology from artifact without better preprocessing.

**Implications**
- Before testing H5 again, the pipeline needs:
  - per-volume motion and eddy-current correction. tractline's GPU eddy model is the obvious tool, extended to b-tensors.
  - a noise model with a signal-proportional (physiological) term, fit from the b = 0 and low-b repeats.
- Then rerun the surprise test. If WM surprise survives, revisit time dependence with real waveform spectra, which would need a dataset that ships its waveforms (e.g. the BWH DiB data) or the HHS authors' waveform files.

**Next**
- Preprocessing: per-volume motion/eddy correction for b-tensor data (tractline), and a noise model with a physiological term; then rerun surprise (H5).
- Free water in WM: is the ~9% real? Spatial pattern and test-retest.
- Real training on GPU: longer runs, several protocols, protocol-subset training for embedding invariance.
- Cohort fits and test-retest (H7).
