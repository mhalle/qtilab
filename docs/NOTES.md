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

## 2026-11-04 (night) — validation against real data

Goal: move from "works in simulation" to "survives contact with real data", using public data only. Scripts: `heldout_hhs.py`, `phantoms_dib.py`, `consistency.py`. Prior variants: v3 and the generic prior.

**1. Held-out prediction** (HHS sub-01 ses-01, slice 20).
- Method: leave one run out and predict it from the rest. Two predictors: the classical WLS cumulant fit extrapolated, and the mean of the 20 nearest simulated tissues (brain prior v2, matched on the remaining volumes).
- Error is RMS / noise sd, where 1 means as good as the noise allows.

| held out | WM-like | GM-like | CSF-ish |
|---|---|---|---|
| run 4, LTE b = 2000 (extrapolation): QTI → simulator | 5.32 → 2.12 | 3.74 → 1.34 | 9.68 → 1.34 |
| run 8, STE b = 1500: QTI → simulator | 1.37 → **1.78** | 1.38 → 1.18 | 1.26 → 1.08 |

- The simulator predicts unseen real images 2–7× better than classical extrapolation, near the noise floor in GM and CSF.
- **WM spherical signal decays faster than the simulator predicts from the other data** (bias −1.5 noise sd). This is the signature time dependence would produce (spherical waveforms probe shorter effective diffusion times). It agrees with the positive best-match STE b-scale in WM (+0.05) but not GM. H5 therefore stays open, pending motion/eddy correction.
- The simulator's 95% intervals are too wide (coverage 93–100%). The nearest-neighbour spread is conservative.

**2. Phantoms (DiB water, oil, hexagonal liquid crystal)**, using b-tensors from the md-dmri xps files.
- **Water:** MD 1.98 µm²/ms (expected about 2.0 near room temperature), FA 0.05, C_MD 0.00. But µFA is **0.26** (WLS 0.21), against a true value of 0.
  - The simulator reproduces this almost exactly for pure water at matched SNR (about 71): constrained 0.254, WLS 0.170.
  - This is a real-data confirmation of both the noise model and the upward µFA bias from noise plus positivity in near-isotropic material.
- **Liquid crystal:** µFA 0.98 (expected about 1), FA 0.61, C_c 0.39, C_MD 0.02. All as expected for anisotropic domains in dispersed orientations.
- **Oil:** MD 0.012 µm²/ms, so b·MD ≤ 0.024 and the signal changes about 2%. Its fits are meaningless (µFA 1.1, K_bulk 34). This is a useful reminder that the outputs are uninterpretable outside the protocol's sensitivity range.

**3. Prior sensitivity.** Four networks with identical training and the HHS protocol, evaluated on the same real slice. Region medians:

| | v1 | v2 | v3 (very wide) | generic | classical |
|---|---|---|---|---|---|
| WM tissue µFA | 0.84 | 0.87 | 0.90 | 0.78 | 0.87 |
| GM tissue µFA | 0.60 | 0.62 | 0.60 | 0.35 | 0.55 |
| WM tissue MD | 0.80 | 0.79 | 0.79 | 0.71 | — |
| GM tissue MD | 0.85 | 0.85 | 0.82 | 0.78 | — |
| median f_fw | 0.08 | 0.095 | 0.13 | 0.21 | — |

- **Robust across brain-like priors:** tissue µFA (WM, GM) and tissue MD. The earlier worry that tissue MD was prior-driven was overstated.
- **Prior-driven:** f_fw, which tracks the allowed range, and therefore full-distribution µFA. Do not report these as findings yet.
- **The generic, non-anatomical prior disagrees strongly in GM.** Next: choose between priors by held-out predictive accuracy. The machinery now exists.

**4. Consistency**
- **DiB 70 vs 217 volumes, same voxels, constrained fit.** Correlations: MD 0.97, µFA 0.94, K_μ 0.96, C_MD 0.78, C_c 0.73. The shorter protocol reads µFA +0.03 and FA +0.06 higher, the same noise-driven upward bias as the water phantom. Anisotropy values are protocol-dependent, which matters for cross-study comparisons.
- **Within-subject (sub-01, 3 sessions) vs between-subject (5 other subjects, 1 session each).** Regional medians on a mid-brain slice; standard deviations:

| measure | within SD | between SD | between / within |
|---|---|---|---|
| classical WM µFA | 0.013 | 0.010 | **0.8** |
| classical GM µFA | 0.033 | 0.040 | 1.2 |
| classical WM MD | 0.008 | 0.019 | 2.3 |
| network WM tissue µFA | 0.0026 | 0.010 | **3.9** |
| network WM tissue MD | 0.0033 | 0.012 | 3.5 |
| network WM tissue C_MD | 0.0006 | 0.0019 | 3.3 |
| network GM tissue µFA | 0.0022 | 0.006 | 2.7 |
| network GM tissue MD | 0.0094 | 0.010 | 1.1 |
| network f_fw (WM / GM) | 0.002 / 0.003 | 0.004 / 0.004 | 2.2 / 1.7 |

- **The network's session-to-session stability is not just prior shrinkage.** It keeps between-subject differences at the same size as the classical fit (WM µFA SD about 0.010 for both) while cutting within-subject variation about 5×. The classical WM µFA cannot tell these subjects apart from repeat scans (ratio 0.8); the network can (3.9).
- Weak spots are GM tissue MD (1.1) and free water (1.7–2.2).
- Caveats:
  - The within-subject estimate comes from one subject and three sessions.
  - Those sessions span menstrual-cycle phases, so they include possible real change.
  - Each measure is a single-slice regional median, and slice placement differs between sessions.
  - Proper ICCs need the full cohort (H7).

**5. Cardiff high-b data: blocked.** Cardiff's repository is behind a Cloudflare browser check that the sandbox cannot pass. It needs to be downloaded by hand (under 1 GB) or fetched from Claude Code.

**Overall assessment:** the approach survives its first real-data tests.
- The simulator predicts unseen real images near the noise floor except in WM.
- It reproduces a phantom bias quantitatively.
- The network's tissue estimates are robust across plausible priors and more reliable than classical QTI.

Still open:
- WM spherical misfit: time dependence or preprocessing.
- Prior choice for free water.
- Biological validity, which needs histology or patients.

## 2026-11-04 (late night) — choosing priors by held-out prediction

Same leave-one-run-out test as above (HHS sub-01 ses-01, slice 20; 20 nearest of 200k simulated tissues). Error is RMS / noise sd (1 = noise-limited). Script: `scripts/heldout_hhs.py --prior N [--td r]`.

| model | run 4, LTE b2000: WM / GM / CSF | run 8, STE b1500: WM / GM / CSF | WM STE bias |
|---|---|---|---|
| classical QTI | 5.32 / 3.74 / 9.68 | 1.37 / 1.38 / 1.26 | |
| generic prior | 2.37 / 1.26 / 1.47 | **3.51** / 1.54 / 1.10 | −3.39 |
| brain v1 | 2.37 / 1.37 / 1.36 | 1.61 / 1.17 / 1.12 | −1.18 |
| brain v2 | 2.12 / 1.34 / 1.34 | 1.78 / 1.18 / 1.08 | −1.47 |
| brain v3 | 2.09 / 1.33 / 1.30 | 1.91 / 1.20 / 1.06 | −1.69 |
| v2 + time dependence, r = 1.46 | 2.09 / 1.38 / 1.34 | 1.54 / 1.21 / 1.09 | −1.10 |
| v2 + time dependence, r = 2.5 | 2.12 / 1.43 / 1.34 | **1.24** / 1.34 / 1.09 | −0.46 |

- **The generic, non-anatomical prior is rejected.** It cannot predict WM spherical signal (3.5 noise sd). Its very different GM answers (tissue µFA 0.35) are therefore not credible. Choosing priors by data works.
- **WM tension under time-independent priors.**
  - Widening the prior v1 → v3 improves WM extrapolation of linear-encoding images (2.37 → 2.09) but worsens WM spherical prediction (1.61 → 1.91).
  - No time-independent brain prior fits both encodings in WM.
- **One global spectral ratio resolves most of it in WM.** Time dependence with STE at a higher effective frequency (r = 2.5) brings WM spherical prediction to 1.24, close to GM levels, and leaves WM linear prediction unchanged. GM gets worse (STE 1.18 → 1.34, LTE 1.34 → 1.43), so the GM time-dependence ranges (soma τ up to 0.4, extracellular up to 0.35) are probably too large.
- **Reading.** This is the most H5-supportive evidence so far. It is a specific, WM-only, encoding-specific effect, in the direction time dependence predicts. Caveats:
  - one session, one slice
  - only two values of r tried
  - the τ ranges are guesses
  - the motion/eddy confound in WM (largest inter-run rotation, run-specific residuals) is not excluded
- Next: fit r and per-tissue τ by held-out accuracy across subjects and sessions, after motion/eddy correction.

## 2026-11-05 — noise model, H4, time dependence across sessions

### Noise model (`qtilab/noise.py`, `scripts/noise_model.py`)

Model: var(S) = s2 (thermal, per voxel) + α(b)² S².
- Fitted by profile likelihood on spherical-encoding repeat groups at b = 100, 500, 1000 and 1500.
- b = 0 is excluded, because the voxel normalization uses those images.
- s2 per voxel is solved by maximum likelihood.

Whole brain, sub-01 ses-01:

| region | α at b = 100 / 500 / 1000 / 1500 | thermal SNR (b = 0) | observed / modeled variance per b |
|---|---|---|---|
| WM | 0.010 / 0.018 / 0.005 / 0.007 | 58 | 0.97 / 0.90 / 0.93 / 0.96 |
| GM | 0.015 / 0.018 / 0.013 / 0.015 | 79 | 0.72 / 0.90 / 0.96 / 0.99 |
| CSF | 0.105 / 0.083 / 0.030 / 0.030 | 69 | 0.16 / 0.71 / 1.00 / 0.85 |

- Tissue: about 1–2% signal-proportional fluctuation.
- CSF: about 10% at b = 100, falling with b. This is pulsatile flow, not uniform physiological noise.
- CSF pulsation is very heterogeneous. A pooled α over-predicts the typical CSF voxel at b = 100 (ratio 0.16), so CSF tests that use this model are conservative.
- The earlier "SNR about 67" (from b = 1500 repeats) understated thermal SNR in GM and overstated noise at low b.
- `heldout_hhs.py --noise model` uses it, with α(b) per region and thermal variance per voxel, both fitted without the held-out volumes.
- **Not done yet:** surprise maps with the noise model.

### H4: linear encoding only (`scripts/h4_lte_only.py`)

Setup:
- Two networks with identical prior (brain v2), SNR range and steps.
- One sees all 107 HHS volumes; the other sees only the 69 LTE volumes plus b = 0 (71 volumes; design rank 22, the DKI information, vs 23).
- Tested on the same simulated voxels (2,000 per test prior) and on the real slice.

Results (tissue µFA median |error|, full → LTE-only):

| test data | WM | GM |
|---|---|---|
| brain v2 (training prior) | 0.022 → 0.023 | 0.073 → 0.078 |
| brain v1 | 0.026 → 0.029 | 0.049 → 0.057 |
| brain v3 (wide, misspecified) | 0.036 → 0.044 | 0.088 → 0.099 |
| generic (non-anatomical) | all voxels 0.175 → 0.196, both badly miscalibrated (90% coverage about 0.4) | |

- Interval widths and coverage are nearly identical between the two networks. Tissue MD, C_MD and f_fw behave similarly.
- **Real slice:** LTE-only and full agree.
  - Tissue µFA r = 0.93 (WM), 0.84 (GM); tissue MD r = 0.76 / 0.93.
  - Median WM µFA 0.852 vs 0.873.
  - LTE-only posterior SD is 1.1–1.2× larger.
- **Reading:**
  - Under a brain-like prior, linear encoding alone recovers tissue µFA nearly as well as the full protocol. That is promising for legacy multi-shell data (HCP, UK Biobank), and consistent with the spherical-mean literature.
  - But prior misspecification hurt *both* networks about equally. Both learned the prior's compartment-shape assumptions, which is what resolves the anisotropy-versus-size confound that spherical encoding exists to break.
  - So under this prior, the spherical measurements add mainly precision (10–20%) and a check on the prior, not robustness.
- **Broad (generic) training prior** (`--full-model results/posterior_smoke_hhs_v0.pt --lte-model results/posterior_smoke_hhs_v0lte.pt`): both networks must resolve the anisotropy/size confound from the data.

  Tissue µFA median \|error\|, full → LTE-only:

  | test data | WM | GM |
  |---|---|---|
  | generic | all voxels 0.102 → 0.117 (90% width 0.55 → 0.66; coverage about 0.9) | |
  | brain v2 | 0.035 → 0.030 | **0.197 → 0.261** |
  | brain v3 | 0.060 → 0.054 | 0.182 → 0.254 |

  - Real slice, both networks: GM tissue µFA 0.35 vs 0.35 (r = 0.63; LTE-only SD 1.3× larger). WM 0.79 vs 0.75.
  - **Reading:** spherical encoding's value is concentrated in gray matter and other dispersed or heterogeneous tissue (about 25–30% lower µFA error), where anisotropy and size variance are confounded. In coherent WM, LTE's directional pattern suffices.
  - **The prior's contribution dwarfs it.** GM µFA error is about 0.07 with the brain prior vs about 0.2 with the generic one, and real GM tissue µFA is 0.62 vs 0.35. The generic prior was already rejected by held-out prediction.
  - **H4 in one sentence:** WM µFA is recoverable from standard multi-shell data under any reasonable prior. GM µFA from such data is mostly a statement about the prior, and spherical encoding is what makes it less so.

### Time dependence across sessions (`heldout_hhs.py`, `td_summary.py`)

Setup:
- Five sessions: sub-01 ses-01/02/03, sub-02 ses-01, sub-06 ses-01.
- Fullest axial slice per session; classical-fit regions; noise model; 100k-tissue bank.
- Configurations:
  - prior v2 time-independent (base);
  - time dependence with LTE/STE spectral ratio r ∈ {1.5, 2.5, 4}, with GM strengths off (g0) or on (g1).

Held-out simulator error (median RMS / noise sd, mean over sessions; 1 = noise-limited):

| config | STE b1500: WM / GM / CSF | LTE b2000: WM / GM / CSF |
|---|---|---|
| base | 1.66 / 1.21 / 0.86 | 2.16 / 1.48 / 1.54 |
| r1.5 g0 | 1.50 / 1.21 / 0.86 | 2.17 / 1.46 / 1.54 |
| r2.5 g0 | 1.32 / 1.21 / 0.87 | 2.16 / 1.46 / 1.54 |
| r4 g0 | **1.27** / 1.21 / 0.88 | 2.18 / 1.46 / 1.53 |
| r2.5 g1 | 1.24 / 1.35 / 0.88 | 2.17 / 1.56 / 1.52 |
| r4 g1 | 1.21 / **1.51** / 0.90 | 2.19 / 1.63 / 1.52 |

Per session, WM STE error base → r4 g0:

| session | WM STE error | GM change | LTE b2000 WM error |
|---|---|---|---|
| sub-01 ses-01 | 1.72 → 1.29 | none | 1.94 → 1.95 |
| sub-01 ses-02 | 1.59 → 1.24 | none | 2.68 → 2.76 |
| sub-01 ses-03 | 1.85 → 1.37 | none | 1.95 → 1.93 |
| sub-02 ses-01 | 1.34 → 1.19 | none | 2.01 → 2.07 |
| sub-06 ses-01 | 1.81 → 1.26 | none | 2.22 → 2.20 |

- **The WM-specific improvement replicates in every session and both new subjects.**
- Time dependence in GM compartments always makes GM worse. Under this model, the effect is white-matter specific.
- Improvement continues up to r = 4. The extended settings below locate the optimum.
- **WM LTE b = 2000 extrapolation stays poor (about 2.2) under every config.** The diagnostics below show this is mostly the predictor.
**Extended settings, all five sessions** (r = 6, 10, and r = 4 with doubled WM strength; all with GM strengths off):

| config | WM STE b1500 error | WM LTE b2000 error |
|---|---|---|
| base | 1.66 | 2.16 |
| r = 1.5 | 1.50 | 2.17 |
| r = 2.5 | 1.32 | 2.16 |
| **r = 4** | **1.27** | 2.18 |
| r = 6 | 1.37 | 2.25 |
| r = 10 | 1.55 | 2.33 |
| r = 4, WM strength ×2 | 1.50 | 2.33 |

- **There is a well-defined optimum near r ≈ 4 with the default WM strengths.** Stronger effects overshoot, on both STE and LTE.
- A model with too much freedom would keep absorbing misfit. A physical effect of a particular size produces an optimum, and that is what we see.
- GM is unchanged in every WM-only config.

### Is the WM LTE b = 2000 misfit the predictor? (`scripts/heldout_diagnostics.py`)

Method: the same held-out procedure applied to *simulated* voxels from the same prior (v2) through the same protocol, with noise from the fitted noise model (thermal variance and α(b) sampled from the real slice). The tissue model is exactly right for these voxels, so their error is the predictor's resolution plus noise. Two sessions (sub-01 ses-01, sub-06 ses-01).

| | sub-01 ses-01: real / sim | sub-06 ses-01: real / sim |
|---|---|---|
| WM, LTE b2000: RMS error | 1.93 / 1.93 | 2.21 / 2.08 |
| WM, LTE b2000: median bias | +0.48 / +0.11 | +1.07 / +0.10 |
| WM, STE b1500: RMS error | 1.73 / 1.35 | 1.73 / 1.33 |
| WM, STE b1500: median bias | **−1.37 / −0.11** | **−1.43 / −0.13** |
| GM, LTE b2000: RMS error | 1.21 / 1.46 | 1.27 / 1.46 |
| sd of per-volume WM median residual, run 4 | 0.20 / 0.06 | 0.26 / 0.05 |
| max \|volume median\| | 0.95 / 0.23 | 1.56 / 0.21 |

Mean residual vs \|cos(gradient, fiber)\|, bins 0–0.3 / 0.3–0.6 / 0.6–0.85 / 0.85–1:

| | real | sim |
|---|---|---|
| sub-01 ses-01 | +0.65 / +0.32 / +0.60 / +1.29 | +0.50 / 0.00 / +0.03 / +0.35 |
| sub-06 ses-01 | +1.70 / +0.97 / +0.92 / +1.54 | +0.74 / +0.07 / +0.04 / +0.35 |

A larger bank (400k, k = 5) lowers simulated WM b2000 error to 1.75–1.86 but not real error (1.96, 2.19).

**Findings**
1. **The RMS size of the WM LTE b = 2000 misfit (about 2) is almost entirely algorithmic.** Simulated voxels with an exactly right tissue model score the same. The k-nearest-neighbour predictor blurs the sharp angular signal of coherent WM at high b. A bigger bank helps only slowly. So the earlier "WM high-b misfit" was mostly our predictor, as suspected.
2. **On top of that floor, real WM shows two smaller, real deviations:**
   - **Volume-level offsets:** some of the 30 b = 2000 volumes are off for all WM voxels together (max \|volume median\| about 1–1.6 vs about 0.2 in simulation). This is an acquisition signature: motion, eddy currents or dropouts within the run.
   - **Excess signal for gradients near the fiber axis:** about +1.3 to +1.5 vs +0.35 simulated. Signal along axons decays less at b = 2000 than predicted. This could be physics (axial non-Gaussianity, or time-dependent axial diffusivity from beading, which is the one-dimensional disorder our √ω model represents) or direction-dependent eddy currents.
3. **The STE effect is not algorithmic.** With the tissue model right, simulated WM STE bias is about −0.1. Real WM is about −1.4 in both sessions. That is the time-dependence signal, and it survives the full-size test.

**Implications**
- **Surprise maps:** nearest-neighbour misfit carries a higher algorithmic floor in coherent WM than in GM. The calibration is SNR-conditional but not anisotropy-conditional, so part of the earlier "a third of WM is surprising" is algorithmic. Fix: calibrate within FA bins, or use a rotation-aware predictor.
- **Predictor:** replace plain k-NN for anisotropic tissue. Options:
  - rotation-aware matching: coarse match on rotation-invariant features, then re-simulate candidates rotated to the voxel's fiber direction;
  - or prediction from the network's posterior.
- **H5 status:** the STE time-dependence effect is now supported by:
  - replication in five sessions and three subjects;
  - a well-defined optimum (r ≈ 4);
  - WM specificity;
  - a simulated null that rules out the predictor.

  The remaining confound is acquisition (volume-level offsets are present in WM), which motion/eddy correction for b-tensors resolves.

**Next**
- Rotation-aware predictor (or network-posterior prediction) for anisotropic tissue; anisotropy-conditional calibration of surprise.
- Preprocessing: per-volume motion/eddy correction for b-tensor data (tractline), and a noise model with a physiological term; then rerun surprise and the held-out STE test in WM (H5).
- Choose priors by held-out predictive accuracy (v1, v2, v3, generic, time-dependent).
- Cohort run for H7 (proper ICCs), best on GPU.
- Cardiff b = 3–10 extrapolation test once the data can be fetched.
- Free water in WM: is the ~9% real? Spatial pattern and test-retest.
- Real training on GPU: longer runs, several protocols, protocol-subset training for embedding invariance.
- Cohort fits and test-retest (H7).

## 2026-11-05: Motion and eddy-current correction (tractline, CPU) does not explain the WM residuals

**What was run.** `scripts/motion_eddy_hhs.py 01 01`: tractline `motion_eddy` (v0.11.0, plain-torch CPU backend, 40 steps, quadratic eddy model, no susceptibility field because HHS has no reversed-PE series). This is the first end-to-end CPU run of tractline's eddy on a real scan: about 15 min for 107 volumes (LTE 582 s, STE 329 s), 1.9 GB peak.
- Adaptations for b-tensors (tractline groups shells by b-value only):
  - LTE and STE are corrected in two calls, each with all b0s as the common reference.
  - STE volumes get one nominal direction, so the leave-one-out predictor is close to the mean of the other repeats.
  - Each HHS run is roughly a whole shell, so between-run alignment stays with the MI step; tractline corrects within runs.
- No volumes excluded, no outlier slices replaced (so no imputation in the output).

**Estimates.**
- Motion is small: rigid edge displacement 0.1–0.2 mm in LTE runs, framewise displacement mean 0.3 mm.
- LTE eddy displacement grows with b (edge RMS mean 0.45 mm, max 2.0 mm, largest at b = 2000).
- STE volumes shift along phase encoding by 0.5–0.6 mm SD (up to ~1 mm) within the b = 1000/1500 runs. With one nominal direction, tractline cannot separate eddy from motion, so this is booked as translation.

**Before vs after, sub-01 ses-01.** `scripts/motion_eddy_compare.py`: same voxels, three slices, same k-NN bank. Residuals are in signal units (×1000 of b0) because resampling smooths thermal noise; fitted thermal SNR rises 59→67 WM, 69→91 GM, 87→128 CSF, so noise-normalized scores are not comparable across the two series.

| ×1000 of b0 | raw | corrected |
|---|---|---|
| STE b1500 repeat sd (noise floor) | 13.9 | 10.9 |
| WM b2000 RMS | 34.2 | 33.6 |
| WM b2000 volume-offset sd | 3.7 | 3.1 |
| GM b2000 RMS | 17.1 | 16.3 |
| WM STE b1500 RMS / bias | 28.5 / −22.7 | 28.0 / −22.9 |
| GM STE b1500 RMS / bias | 15.8 / −2.1 | 13.7 / −2.3 |

- Held-out test with time dependence (`heldout_hhs.py --noise model --regions classical --slice -1`, bank 200k). WM STE error in noise units:

  | | no time dependence | r = 4, WM only |
  |---|---|---|
  | raw | 1.72 (bias −1.36) | 1.27 (bias −0.33) |
  | corrected | 1.89 (bias −1.47) | 1.30 (bias −0.34) |

- The excess signal along the fiber axis at b = 2000 also persists after correction (`heldout_diagnostics_s0101ec.json`, diagnostic C).

**Findings**
1. **H5 survives motion/eddy correction.** The WM STE deficit is unchanged in signal units (−22.7 vs −22.9), and r ≈ 4 time dependence removes it equally in both series. A spatial misregistration would produce edge-signed errors, not a uniform WM-specific deficit.
2. **The WM b = 2000 volume-level offsets are only slightly reduced** (sd 3.7 → 3.1). They are not mainly within-run motion or correctable eddy. Remaining candidates: per-volume signal scaling such as gradient-amplitude or b-value error per direction, gradient nonlinearity, and the predictor's angular blur (which the simulated null already shows at about 1/4 of the real offset).
3. **The along-fiber excess also survives,** which weakens direction-dependent eddy as its cause and favors physics (axial non-Gaussianity or time-dependent axial diffusivity).
4. **The correction mostly reduces noise-like variance** (STE repeat sd −22%, part real correction of the ~0.5 mm STE shifts, part interpolation smoothing). That improves GM and CSF modestly.

**Caveats**
- One session.
- Between-run (whole-shell) offsets are left to the MI alignment.
- A constant STE-waveform eddy field common to all STE volumes would not be seen by a leave-one-out predictor.
- tractline's eddy model is FSL-style (field linear in the gradient vector); a waveform-specific model for STE is still open.

### 2026-11-05 (later): replication in sub-06 ses-01 and sub-02 ses-01

Same pipeline (`/home/claude/chain9.sh` equivalent: `motion_eddy_hhs.py`, `motion_eddy_compare.py`, diagnostics, held-out with `--series`). About 15 min of correction per session; no volumes excluded; 2 outlier slices replaced in total (sub-06 STE).

Signal units (×1000 of b0), raw → corrected:

| | sub-01 | sub-06 | sub-02 |
|---|---|---|---|
| WM STE b1500 bias | −22.7 → −22.9 | −23.9 → −24.0 | −18.4 → −18.8 |
| WM LTE b2000 RMS | 34.2 → 33.6 | 36.1 → 35.6 | 34.8 → 34.7 |
| WM b2000 volume-offset sd | 3.7 → 3.1 | 3.9 → 3.7 | 3.9 → 3.9 |
| STE b1500 repeat sd | 13.9 → 10.9 | 14.4 → 11.7 | 11.4 → 10.7 |
| STE PE-shift sd by run (b 100 / 1000 / 1500), mm | ~0.26 / 0.50 / 0.62 | 0.27 / 0.47 / 0.65 | 0.04 / 0.06 / 0.11 |

WM STE held-out error, noise units (bias), no time dependence → r = 4 WM-only:

| | raw | corrected |
|---|---|---|
| sub-01 | 1.72 (−1.36) → 1.27 (−0.33) | 1.89 (−1.47) → 1.30 (−0.34) |
| sub-06 | 1.78 (−1.41) → 1.25 (−0.34) | 1.83 (−1.51) → 1.24 (−0.39) |
| sub-02 | 1.38 (−0.88) → 1.19 (+0.14) | 1.38 (−0.91) → 1.20 (+0.12) |

**Findings**
1. **H5 survives motion/eddy correction in three subjects.** The WM STE deficit is unchanged in signal units, and r ≈ 4 time dependence removes it equally in raw and corrected data. Sub-02's deficit is smaller, and r = 4 slightly overshoots, a hint that the strength varies between subjects.
2. **The b = 2000 volume offsets are not motion or correctable eddy** in any subject (sd barely changes).
3. **Within-run STE shifts along PE are subject-dependent.** They are large in sub-01 and sub-06 (up to ~1 mm) and small in sub-02. They grow from run 6 to run 8 in all three, but **b-value and acquisition time are confounded** (STE runs are acquired in b order), so "eddy from STE waveforms" and "drift over time" cannot be separated here.
4. **The noise drop splits roughly as** about 6% from interpolation smoothing (sub-02, little to correct) plus about 15% from real correction of STE shifts (sub-01, sub-06).

## 2026-11-05: What the volume-level offsets are: a reproducible per-volume level, period 3 in STE

`scripts/volume_offsets.py` predicts each diffusion-weighted volume from a leave-one-volume-out WLS QTI fit to all the others (no k-NN, so no angular blur). The log residual per voxel is then explained per run by:
- a per-volume **gain** (shared by all tissue);
- a per-volume **b-error** (residual ∝ d ln S / d ln b, centered so it is separable from the gain);
- a **fiber-angle** term (one slope per run, WM only; real fiber orientations are not uniform, so this can make WM-only volume offsets without any acquisition error).

Five sessions (sub-01 ses-01/02/03, sub-02 ses-01, sub-06 ses-01), 15–30k tissue voxels each. `scripts/volume_offsets_followup.py` tests drift vs systematic, b-scaling, cross-session reproducibility and the effect on α(b). Figure: `results/volume_offsets_pattern.png`.

**Findings**
1. **The fiber-angle term explains almost nothing at b = 2000** (it matters only in a couple of b1000/b1500 runs). The volume offsets are not anatomy interacting with predictor bias.
2. **WM and GM offsets move together** (STE r 0.8–0.98, slope ~1; LTE b2000 r up to 0.8). A per-volume level removes most of the structure, down to near the split-half noise floor of a volume mean. A separate b-error term adds nothing (in STE it cannot be separated from a gain, since WM and GM MD are similar).
3. **STE: a period-3 pattern** (low, high, middle), repeating through every STE run.
   - Peak-to-peak about 10–15‰ at b ≥ 500.
   - Lag-1 autocorrelation −0.4 to −0.56, far below the permutation null, so it is not drift.
   - Each session's pattern correlates r = 0.62–0.97 with the mean of the other sessions, across three subjects.
   - Most likely the STE waveform is played in three rotations, and each rotation has a slightly different effective trace b (about 1–1.5% at b = 1000) or signal level: gradient-axis calibration, gradient nonlinearity or concomitant fields.
4. **LTE: weaker and less reproducible.** The gain sd grows with b (1.4‰ at b100 to 4.0‰ at b2000), with leave-one-session-out r 0.2–0.7. This is consistent with a per-direction effective b error plus session-specific effects. These are the b = 2000 "volume-level offsets" of diagnostic B: not motion, not eddy, not anatomy.
5. **α(b) is not materially inflated** by these offsets. Removing the other sessions' STE gain template changes α only at the grid resolution: 0.5% level offsets are small next to thermal noise (sd ~1.7% of b0) and α ~1–2%.
6. **H5 is not affected.**
   - The offsets are shared by WM and GM, and they average out over the balanced rotations in each STE shell.
   - A common STE b-scale error would hit GM equally, but the deficit is WM-specific (−18 to −24 vs about −2 in GM).
   - Residual waveform anisotropy, averaged over rotations, would make WM STE signal higher, not lower (Jensen).

**Implications**
- These are acquisition fingerprints that recur across subjects: a cohort-level per-volume calibration (a gain template per protocol volume) is cheap and removes them.
- They are a clean example of a non-anatomical, reproducible "anomaly", relevant to the anomaly framework discussion.
- With real waveforms (handoff item 6) the period-3 pattern could be checked against each rotation's actual b-tensor.

## 2026-11-05: Device benchmark — M1 Max CPU vs MPS vs Modal A10G (no data needed)

`scripts/bench_devices.py` runs one float32 workload per device on a synthetic HHS-like protocol (186 volumes, LTE + STE, rank 23). `scripts/modal_smoke.py` runs the same code plus the test suite on Modal; L4 had no capacity (one job sat in "waiting to be scheduled" with no local output), so the GPU fallback list gave an A10G. Raw numbers: `results/bench_devices_m1max.json`, `results/bench_devices_modal_A10G.json`.

| 200k voxels / batch 4096 | CPU (M1 Max, 8 threads) | MPS | A10G |
|---|---|---|---|
| simulate | 2.5 s | 0.61 s | 0.15 s |
| WLS | 5.5 s | 1.5 s | 0.27 s |
| constrained (ADMM) | 100 s | 139 s | 37 s |
| network step, batch on device | 381 ms | 277 ms | 51 ms |
| training step as written (`simulate_batch` on CPU) | 490 ms | 301 ms | 279 ms, of which 250 ms CPU simulation |

Accuracy is the same everywhere (median |param error| 0.069 constrained), tests pass on CUDA (5/5).

**Findings**
1. **GPU training is simulation-bound.** The A10G does the network step 7.5× faster than the M1 CPU, but `posterior.simulate_batch` simulates on CPU (no `device` argument), and on Modal's 4 CPUs that is 90% of the step. Moving simulation onto the device (`simulate.sample_brain` already takes `device`) should give ~5× on training. Prerequisite for handoff task 4.
2. **MPS looked slower than CPU for the constrained fit** (139 vs 100 s), but that is the default `chunk=20000`. With one chunk (`chunk=200000`) MPS takes 26.7 s vs 102 s chunked (same session), 3.7× faster than CPU. Each chunk runs a ~1000-iteration ADMM tail of tiny, sync-bound GPU steps, so ten chunks pay it ten times.
3. **ADMM stragglers (correction below: they do not set the cost).** ~0.35% of violating voxels (55 of 17.6k) do not converge in 1000 iterations on this rank-23 protocol, in float64 as well as float32, so it is not precision. `FitInfo.converged_fraction` prints as "100%" (0.9965). Every chunk runs the full 1000 iterations for those few voxels. This is the rank-deficient case `fit.constrained`'s docstring warns about; the DiB protocol (full rank) converged 100%. Worth a look before the 66-session cohort fit: a lower `max_iter` with a fallback for the stragglers, or a larger ridge.

**Follow-up profiling (20k voxels, CPU float32)**
- Where the work goes: the working set is 17.6k at iteration 50, 7.3k at 100, 2.4k at 200, 710 at 400 and 59 at 1000. Iterations past 200 are only 18% of voxel-iterations; capping at 100 iterations still costs 5.9 of 9.9 s. The stragglers matter only through the per-chunk tail on GPU (finding 2).
- Per iteration on CPU, `fit._project` (batched `eigh` of 3×3 and 6×6 matrices) takes 57 ms of ~65 ms at 17k voxels (cholesky_solve 6 ms). On MPS the same projection is 7.7 ms. A closed-form 3×3 and a vectorized Jacobi 6×6 eigensolver are the candidate fix on CPU.
- Truncating ADMM: against the 1000-iteration result, `max_iter=400` changes µFA by at most 1.5e-3 (99.9th percentile 6e-4) and FA by at most 9e-4, but saves only ~10% on CPU. Not worth it there.

## 2026-11-05 (evening): GPU speed-ups implemented — device-aware fit chunks, simulation on device

**Changes** (CPU default path bit-identical to before: simulate_batch for priors 0–3, WLS and constrained fits compared tensor-for-tensor against the previous commit)
- `fit.wls` / `fit.constrained`: `chunk=None` now means 20k voxels on CPU (as before); on a GPU it is sized from free memory (`fit._chunk`: 40% of free memory at a conservative 88 kB per voxel, between 8k and 250k). A fixed 250k would not fit a 16 GB Mac (WLS peaks at 12 GB for 200k voxels). Simulating a 16 GB M2 by capping Metal's limit at 10.7 GB: 52k chunks, constrained fit of 200k voxels 56 s (vs 26 s as one chunk on this Mac, 100 s on CPU), 6.1 GB peak. On a 24 GB A10 the sizing gives about 100k chunks: 10.1 s for 200k voxels (7.4 s as one chunk). The estimate is deliberately conservative (88 kB vs ~60 kB measured); raise `GPU_MEMORY_FRACTION` if a known-large GPU should take bigger chunks.
- `tensors.batched`: every batched eigensolver call (`psd_project`'s `eigh`, `violates`' `eigvalsh`, `_solve_normal`'s `pinv`) runs in sub-batches of 16k matrices. PyTorch's CUDA eigensolvers need workspace far beyond the matrices: at 200k, `pinv` asked for 66 GB on an A10G and `eigvalsh` on 3×3 matrices for 51 GB (about 270 kB per matrix) on an L40S. MPS did not have this problem.
- `posterior.simulate_batch(..., device=, sim_dtype=)`: simulates on the device, float32 by default. `device=None` keeps the float64 CPU path every saved result used. Target distributions on MPS match CPU float64 to within sampling noise (≤ 0.02 sd for all priors).
- `scripts/posterior_smoke.py --device mps|cuda`: trains on the device, evaluates on CPU as before. Checked for 100 steps on MPS and CPU with the synthetic protocol (HHS data not downloaded here).

**Speed, before → after** (`results/bench_devices_m1max.json`, `bench_devices_modal_A10.json`; the "before" CUDA run was an A10G, the "after" an A10: same class)

| | M1 MPS | Modal A10(G) |
|---|---|---|
| constrained fit, 200k voxels | 139 → 32 s | 37 → 7.4 s as one chunk; 10.1 s with the memory-sized default (the saved JSON) |
| training step, batch 4096 | — (network-bound: ~290 ms) | 257 → 66 ms |
| training step, batch 512 | — | 54 → 29 ms |

On MPS the network step itself dominates (290 ms at batch 4096), so simulating on device gains little there; `simulate_batch` alone is 25 ms on MPS vs 105 ms on CPU.

**Peak memory** (float32, measured on the M1: MPS driver allocation, CPU peak RSS)

| stage | peak |
|---|---|
| simulate 200k voxels (generic prior, 186 volumes) | 4.1 GB |
| WLS, 200k voxels, one chunk (MPS) | 12.0 GB |
| constrained, 200k voxels, one chunk (MPS) | 11.0 GB |
| constrained, 200k voxels, 20k chunks (CPU) | 4.8 GB |
| training, batch 4096 / 16384 (MPS) | 4.2 / 12.4 GB |
| posterior sampling, 20k voxels × 100 samples, chunk 64 / 1024 (MPS) | 2.2 / 9.1 GB |

- Fits on a GPU cost roughly 55–60 kB per voxel per chunk; the peak is WLS's normal equations (`einsum` materializes the weighted design, V × N × 28). The 250k cap is about 14 GB: fine on 24 GB cards (L4, A10, A10G) and MPS, tight on a 16 GB card.
- Training costs about 0.75 MB per batch element (M = 64 micro-tensors × 186 volumes, plus flow activations).
- Posterior sampling costs about 90 kB per voxel-sample; the default chunk of 64 voxels was sized for the 7 GB sandbox and can be raised 10–20× here.

## 2026-11-05 (night): local reproduction on the M1 Max — matches the sandbox

HHS downloaded locally (2,273 files, 8.78 GB, `data/hhs`, Dropbox-ignored). Chain for sub-01 ses-01: `prepare_session.py 01 01` → `posterior_maps_hhs.py "" "" results/posterior_smoke_hhs_v2.pt _v2` → `heldout_hhs.py --prior 2 --tag _v2_m1` (`results/heldout_hhs_v2_m1.json`, compared with the sandbox's `heldout_hhs_v2.json`).

- **Alignment:** run-to-run rotations identical to the sandbox (run 2: 0.61°, the largest).
- **v2 posterior maps:** classical / network full / network tissue µFA = WM 0.87 / 0.68 / 0.87, GM 0.54 / 0.47 / 0.62, CSF-ish 0.31 / 0.15 / 0.73, identical to the 2026-11-04 (late) table to two decimals.
- **Held-out prediction (prior v2):** all 42 numbers within 0.042 of the sandbox (median RMS error / noise sd). Run 4 simulator WM / GM / CSF 2.12 / 1.34 / 1.32 (sandbox 2.12 / 1.34 / 1.34); run 8 1.81 / 1.18 / 1.07 (1.78 / 1.18 / 1.08). The differences are posterior-sampling noise (the flow is sampled unseeded) feeding the region assignment, plus small numeric differences in alignment; the conclusions are unchanged.
- **Timing:** prepare 2 min 45 s (sandbox ~6 min), posterior maps 32 s, held-out 8 s.
- **Blocker found and fixed:** on macOS, dipy's wheel and torch each bundle `libomp`; importing `dipy.align` with torch aborts (`OMP: Error #15`). `scripts/macos_fix_libomp.sh` makes dipy share torch's runtime. The tests do not import `dipy.align`, so they did not catch it.
