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

## 2026-11-05 (night): the constrained fit on the HHS protocol agrees with DIPY SDPdc

Until now the constrained fit had been checked against DIPY only on the full-rank DiB protocol; on HHS (107 volumes, LTE + STE, rank 23) it had been checked only against itself. `scripts/validate_dipy_hhs.py` fits 2,500 WLS-violating and 500 non-violating voxels of sub-01 ses-01 four ways and scores every solution on the shared objective ½x'Hx − g'x (DIPY-equivalent weights). Result: `results/validate_dipy_hhs_sub01_ses01.json`. (61% of the session's 95,806 brain voxels violate a constraint under WLS.)

**Violating voxels**, |scalar difference| median / 99th percentile / max:

| pair | FA | µFA | C_MD | K_bulk | relative objective gap, median / p99 |
|---|---|---|---|---|---|
| qtilab default vs CLARABEL | 1e-5 / 1e-4 / 5e-4 | 1e-5 / 3e-4 / 1e-3 | 6e-6 / 4e-4 / 1e-3 | 2e-5 / 1e-3 / 6e-3 | +6e-12 / +8e-10 |
| qtilab tight vs CLARABEL | 9e-6 / 7e-5 / 2e-4 | 1e-5 / 1e-4 / 4e-4 | 6e-6 / 9e-5 / 3e-4 | 2e-5 / 3e-4 / 5e-3 | +5e-12 / +7e-10 |
| DIPY SCS vs CLARABEL | 3e-6 / 7e-5 / 2e-4 | 4e-6 / 1e-4 / 2e-3 | 1e-6 / 8e-5 / 3e-3 | 5e-6 / 3e-4 / 8e-3 | 0 / +6e-9 |

MD agrees to 2e-5 µm²/ms median (p99 3e-4). Non-violating voxels agree to ≤ 1e-5 in every scalar (all four fits reduce to WLS there).

**Findings**
1. **Same optimum.** qtilab's objective is within 1e-9 (relative, p99) of CLARABEL's in every voxel, and the scalar differences are as small as DIPY's own SCS-vs-CLARABEL differences. The ridge used for the rank-deficient case does not move the solution measurably, and the scalars do not depend measurably on the unobserved directions (max µFA difference 1e-3, about equal to the solver-to-solver spread).
2. **Strictly feasible.** Minimum eigenvalue of ⟨D⟩ and ℂ: qtilab −4e-15 (round-off), CLARABEL −1e-5, SCS −1e-3 (µm²/ms units).
3. **"Unconverged" voxels are a stopping-rule artifact.** With default settings only 85% of violating voxels meet ADMM's residual tolerance within 1000 iterations (tight settings: 99.8% after 20,000). But the objective of the default fit is already at the optimum (gap vs the tight fit: 1.5e-10 at p99), and its scalars differ from the tight fit by at most 1e-3. On a rank-deficient design the residuals in the unobserved directions shrink slowly without affecting the objective, so the residual-based rule is much stricter than needed here. `FitInfo.converged_fraction` therefore understates fit quality on HHS. A stopping rule on the objective, or residuals measured in the observed subspace, would end most voxels far earlier (and make the fit faster).
4. **Speed on CPU (3,000 voxels, float64):** qtilab 5.5 s (2.2 ms per voxel, mostly constrained), CLARABEL 35 s (12 ms), SCS 93 s (31 ms).

Reservation 1 from the algorithm review (no independent check on the rank-deficient protocol) is resolved for this session; repeating it on another subject would make it general.

## 2026-11-05 (late night): objective-based ADMM stopping — not worth it; MPS float32 WLS solve is the real GPU accuracy problem

**Objective-stall stopping rule (tried, reverted).** Idea: on HHS, ~15% of violating voxels hit the 1000-iteration cap although their objective is already optimal (previous entry), so also stop a voxel once the objective at its feasible iterate stops improving. Tested on the 3,000 validation voxels (vs a float64 tight fit), on synthetic full-rank and rank-23 protocols, and on the whole session (95,806 voxels) on CPU float64 and MPS float32.
- A tolerance relative to |J| looked good on HHS (27% faster, accuracy unchanged), but |J| is dominated by the ln S0 term, so its strictness depends on the signal's units: it never fired on synthetic data (S0 = 1). Rejected.
- Scale-free version (improvement relative to the constraint cost J(z) − J(WLS)): at equal accuracy it saves nothing on CPU (114.5 vs 114.3 s whole session); faster settings save up to 29% but let K_bulk move by up to 0.1–0.2 in the worst voxels. On MPS it saves 20% (29.6 → 23.7 s); the last iteration stays at 1000.
- Computing J(z) − J(x0) directly as ½d'Hd + (Hx0 − g)'d matters in float32: as a difference of two J values it cancels (J is dominated by ln S0) and the stall test fires spuriously.
- Why it cannot help much: most ADMM work is the bulk of 50–200 iterations per voxel, not the tail. Faster convergence (better ρ initialization, acceleration), not stopping, is the lever.

**MPS float32 accuracy.** The GPU path as it stands (MPS, float32) differs from the float64 reference by up to 1e-2 in µFA (p99 2e-3) on the 3,000 HHS voxels — ten times the float64 solver spread. CPU float32 is fine (µFA within 1e-3). Stage by stage on identical inputs, MPS matches CPU in the normal equations, OLS weights, `eigh` and ADMM (2e-4); the culprit is MPS's batched `pinv` in the rank-deficient WLS solve (µFA vs CPU f32: p99 6e-3, max 5e-2).
- **Fixed (`fit._solve_normal`, `fit._row_space`):** with positive weights, null(H) = null(X) for every voxel, so the rank-deficient WLS solve now works in X's row space: one float64 SVD of X, then an r × r Cholesky solve per voxel (r = 23 on HHS), with `pinv` only as a fallback for voxels whose reduced matrix is not positive definite. Full-rank protocols are untouched (bit-identical).
  - Results: MPS float32 constrained fit vs the float64 tight reference, µFA p99 / max 2e-3 / 1e-2 → **2e-4 / 1e-3** (the same as CPU float64 at default settings); C_MD 3e-4 / 1e-3. CPU float64 session fit identical to the saved one to ≤ 1e-11 in every scalar; DIPY check unchanged; held-out result unchanged (differences come only from the unseeded posterior's region assignment).
  - Speed: WLS of the whole session 0.46 s (CPU f64) / 0.85 s (MPS); on an A10G, WLS of 200k voxels 0.30 → 0.08 s. The constrained fit is ADMM-bound and barely changes (MPS whole session 30 → 25 s).
  - Pitfall found while applying it: the row-space rank must come from `matrix_rank` at X's own precision. A float32 design cast to float64 has float32 rounding in its null singular values, so a float64 cutoff kept all 28 directions, the reduced Cholesky failed for 97% of voxels and everything silently fell back to `pinv`. `tests/test_core.py::test_rank_deficient_wls_float32_matches_float64` guards this.

## 2026-11-06: collapsed fits dropped from the mask; non-physical scalars from the "dc" relaxation

**Whole-session GPU accuracy (after the row-space fix):** MPS float32 vs CPU float64 over all 95,801 non-degenerate voxels of sub-01 ses-01: µFA median 2e-5, p99 2e-4, max 1e-3; FA max 4e-4; MD max 5e-4 µm²/ms; C_MD max 1e-3; C_c max 4e-3. The same size as ADMM's own tolerance (CPU float64 vs a fully converged fit, or vs DIPY CLARABEL). CUDA not yet compared on real data.

**Collapsed fits.** 5 voxels at the brain-mask edge (z = 37, 11, 6; b0 ≈ 50 vs a brain median of 534) fit ⟨D⟩ = 0: no attenuation above the noise floor. Every ratio scalar is then meaningless (FA 0, C_MD 1, K_bulk and K_µ ~1e29, µFA up to 1.2). `scalars.collapsed` (MD ≤ 1e-6 µm²/ms) flags them and `prepare_session.py` drops them from the saved mask (95,806 → 95,801 voxels; all other scalars identical; slice-20 analyses unaffected by construction). Downstream regional statistics are medians, so no saved result had been distorted. `consistency.py` uses plain WLS, which cannot collapse, and is unchanged.
- Near-collapsed voxels remain: the lowest MD is 0.004 µm²/ms and K_bulk still reaches 5,706 (K_µ 4,290) in a few of them (13 voxels with MD ≤ 0.05 beyond the 5 dropped). Not degenerate, so no principled cutoff; any mean-based statistic over K should be robust to them.

**Non-physical values allowed by the "dc" constraints** (left as they are; a modelling question): µFA > 1 in 140 voxels (max 1.224) and C_c > 1 in 3,069. ⟨D⟩ ⪰ 0 and ℂ ⪰ 0 are necessary but not sufficient for the data to come from a distribution of positive-definite micro-tensors, so they do not bound µFA by 1 or C_c by 1. This is reservation 3 from the algorithm review made concrete. Options: the stronger QTI+ constraint set (to be checked against Herberthson et al. 2021), or clipping only for display.

## 2026-11-06: real-data GPU check on Modal (A10) — consistent with CPU, and the cohort timings

`scripts/modal_check_hhs.py` runs sub-01 ses-01 (95,801 voxels) on a Modal GPU against CPU float64 references packed locally (`work/modal_check_sub01_ses01.npz`); result `results/modal_check_hhs_sub01_ses01_A10.json`.

**Consistency** (|difference| p99 / max)
- CUDA float64 vs CPU float64, all voxels: ≤ 1e-11 in every scalar. The same answer on both platforms.
- CUDA float32 vs CPU float64, all voxels: MD 2e-4 / 5e-4, FA 1e-4 / 4e-4, µFA 2e-4 / 1e-3, C_MD 2e-4 / 9e-4, C_c 2e-4 / 1e-2 (p99.99 1e-3: a handful of voxels). Against the float64 tight fit on the validation voxels: µFA 2e-4 / 7e-4, C_c 3e-4 / 2e-3. Same as MPS float32 and as ADMM's own tolerance.
- Network on GPU vs CPU: features 5e-7, embeddings 3e-6 (scale 3.4), log-probability 2e-3 (median 10.8).

**Performance, whole session**

| | constrained fit | WLS | network sampling, n = 50 (best chunk) |
|---|---|---|---|
| CPU float64 (M1 Max) | 106 s | 0.41 s | — |
| MPS float32 (M1 Max) | 25 s | 0.85 s | 41 s (chunk 512) |
| A10 float32 | 7.1 s (chunk 153k, 5.3 GB peak) | 0.04 s | 50 s (chunk 512–2048) |
| A10 float64 | 34 s (chunk 76k, 10.2 GB) | 0.17 s | — |

- Posterior sampling is no faster on the A10 than on MPS. (First explained here as launch-bound; that was wrong — see the next entry: on CUDA it is one pathological kernel, and torch.compile makes it 6–14× faster.) Chunk 512 is the sweet spot in eager mode on both (the default of 64 is 1.2–1.6× slower); chunk 2048 only costs memory (11 GB on the A10, 16 GB on MPS).
- Cohort implications (66 sessions): constrained fits ≈ 8 min on an A10 in float32 (37 min in float64), ≈ 27 min on MPS, ≈ 2 h on CPU; network inference ≈ 1 h on either GPU; run alignment (~1 min per session, CPU/dipy) ≈ 1 h serially. Float64 on an A10 is still 3× faster than the M1 CPU, so reference-quality fits need not be local.

## 2026-11-06: why posterior sampling was slow on GPUs — and torch.compile makes it 6–14× faster

**Structure.** The v2 flow (zuko NSF) has 4 masked autoregressive transforms over the 8 targets with 8-bin rational-quadratic splines; the conditioner is a 72→128→128→184 masked MLP. Sampling inverts each transform one dimension at a time: 32 sequential passes, each running the full MLP, rebuilding all splines (softmax, cumsum) and inverting all 8 dimensions although only one becomes final. Training's log-probability needs 4 passes.

**Where the time goes** (`scripts/modal_profile_flow.py`, A10, real voxels, n = 50; `results/modal_profile_flow_A10.json`):
- Not launch-bound: the GPU is busy ~100% of wall time (2,816 kernels per chunk). Locally, time per sample is flat from chunk 256 up (MPS 9 µs, CPU 16 µs); only tiny chunks are launch-bound (MPS 35 µs at chunk 16).
- Not memory bandwidth in general either (an earlier estimate): **76–78% of GPU time is one kernel**, PyTorch's CUDA `cumsum` over the 8-bin innermost dimension (`tensor_kernel_scan_innermost_dim`, 64 calls per chunk: 203 of 268 ms at chunk 512). The scan kernel is pathological for rows of 8. On MPS `cumsum` is cheap (0.27 ms vs 0.10 ms for an add over the same 7M elements); MPS's time is the many small unfused ops. Replacing cumsum with a matmul by a triangular matrix is slower on both, so no eager-mode fix there.

**torch.compile** (Inductor generates fused kernels — Triton on CUDA, Metal on MPS), compared value for value with eager on identical base noise:

| | eager | compiled | speed-up | max \|diff\| (> 1e-4) | compile time |
|---|---|---|---|---|---|
| A10, chunk 2048 | 10.6 µs/sample (≈51 s/session) | 0.78 µs (≈4 s) | 13.5× | 1.0e-3 (0.00%) | 52 s |
| A10, max-autotune | | 0.77 µs | 13.7× | 7.7e-4 | 140 s |
| MPS, chunk 512 | 8.3 µs (≈40 s) | 1.32 µs (≈6 s) | 6.3× | 1.6e-3 (0.00%) | 22 s |

The differences are float32 reassociation, far below the Monte Carlo error of a 50-sample posterior mean. Hand-written kernels are not needed for this; compiling is the fix. Remaining levers, if ever needed: a coupling or inverse-autoregressive flow (2 passes instead of 8 per transform; needs retraining), fewer samples, or a fused hand-written kernel. Compilation is per input shape, so a fixed chunk (padding the last one) avoids recompiles.

## 2026-11-06: compiled sampling and training implemented — results per device

**Changes**
- `PosteriorNet.sample(..., compile=False, generator=None)`: base noise is drawn with `torch.randn` (optionally from a generator, so sampling is now reproducible) and pushed through the flow's inverse; `compile=True` uses `torch.compile` of that inverse and pads the last chunk so one graph serves every chunk. The eager path is equivalent to the old `flow(ctx).sample` (statistically identical; the noise now comes from `torch.randn`).
- `scripts/posterior_smoke.py --compile`: compiles the training loss (forward + backward).
- `scripts/bench_flow.py` (any device, no data needed; value-for-value correctness: same noise and chunking for sampling, identical weights and batch for training) and `scripts/modal_profile_flow.py --what bench` (CUDA). `tests/test_core.py::test_compiled_sampling_matches_eager` (skipped without a C compiler; ~16 s).

**Results** (n = 50; whole-session estimate for 95,801 voxels; files `results/bench_flow_m1max_{mps,cpu}.json`, `results/bench_flow_modal_L40S.json`)

| device | sampling, chunk 512 | training step, batch 512 | training step, batch 4096 |
|---|---|---|---|
| L40S (CUDA) | 16 → 1 s per session (11.7×) | 23.1 → 5.4 ms (4.3×) | 32.0 → 20.3 ms (1.6×) |
| M1 Max MPS | 42 → 8 s (5.1×) | 38 → 32 ms (1.2×) | 276 → 621 ms (**0.4×**, slower) |
| M1 Max CPU | 110 → 59 s (1.9×) | 63 → 67 ms (0.9×) | — |

- Accuracy: compiled sampling within 2.6e-4 of eager everywhere (no value differs by more than 1e-4 at the 0.001% level); compiled training loss within 3e-5 and gradients within 4e-4 relative.
- Compilation: 20–60 s the first time per shape, then 1–6 s from PyTorch's on-disk cache.
- Recommendation: compiled sampling for any whole-brain inference on any device (chunk 512); compiled training only on CUDA; on MPS and CPU it is slower at realistic batch sizes.
- Pitfall: compiling for MPS and then CPU in one process segfaulted (exit 139, torch 2.15); run one device per process.
- The L40S is much faster in eager mode than the A10 (3.3 vs 10.4 µs per sample), presumably a better `cumsum` path; the compiled speed-up is relative to each GPU's own eager time.

## 2026-11-06: whole-brain posterior inference

`posterior_maps_hhs.py --whole-brain --device mps` samples every brain voxel with compiled sampling (chunk 512, n = 100, seeded) and writes `work/hhs_subXX_sesYY_posterior_v2_full.npz` (`paths.posterior_volume`) plus a `_full.png`. Slice mode and its outputs (read by the held-out and surprise scripts) are unchanged. The slice restriction dated from the 2-core sandbox ("one slice keeps CPU time to a few minutes"; the whole brain eagerly on 2 cores would be ~11 min).

sub-01 ses-01, 95,801 voxels, M1 Max MPS:
- **Time:** 47 s on first use (includes compilation), **29 s** warm (n = 100; ≈15 s at n = 50). Features 0.8 s.
- **Reproducible:** a seeded rerun is bit-identical.
- **Agrees with the slice file:** on the 4,012 slice-20 voxels, (whole-brain mean − slice mean) / Monte Carlo SE has mean ≈ 0 and SD 0.995–1.019 for every target, with ≤ 0.02% beyond 4σ: identical up to sampling noise.
- **Whole-brain regions** (classical µFA / network full µFA / network tissue µFA): WM-like (n = 14,583) 0.87 / 0.67 / 0.87; GM-like (29,111) 0.54 / 0.48 / 0.63; CSF-ish (19,570) 0.33 / 0.14 / 0.74. Slice 20 gave 0.87 / 0.68 / 0.87, 0.54 / 0.47 / 0.62, 0.31 / 0.15 / 0.73, so the slice was representative.
- Median f_fw 0.10 whole brain; 20% of voxels have f_fw > 0.5 (slice 20: 15%), the extra coming from CSF-rich slices.

## 2026-11-06: anisotropy classes from the posterior, checked against FastSurfer (T1)

**Classes from the posterior.** Integrating the posterior's marginal Gaussians (mean, SD) over fixed regions gives soft classes per voxel (the f_fw and FA_t probabilities are multiplied, which assumes the two are independent in the posterior; integrating the flow's joint samples would avoid that), named for what defines them: free_water f_fw > 0.7; mixed 0.3–0.7; for f_fw < 0.3, low_FA FA_t < 0.15, mid_FA 0.15–0.4, high_FA > 0.4. A first try split GM from WM on µFA_t and failed: tissue µFA_t exceeds 0.5 in 99.6% of tissue voxels.
- **They are anisotropy classes, not tissue types** (the pallidum and thalamus below land in mid/high_FA), and an earlier draft's GM / WM_disp / WM_coh names were wrong. Even "isotropic" would be wrong for low_FA: cortex has µFA_t 0.63.
- **They use little of what is specific to QTI.** FA_t is the FA of the tissue mean tensor, and FA_t² = µFA_t² · C_c (scalars.py: FA = √C_M, µFA = √C_µ, C_c = C_M / C_µ). Thresholding FA_t mixes microscopic anisotropy with orientation coherence; apart from the free-water separation, free-water DTI could draw the same map. Classes on (µFA_t, C_c) would use the tensor-valued encoding, but see the C_c caveat below.
- The class boundaries are thresholds, not gaps in the data; the FA_t–µFA_t cloud is continuous.
- Boundary uncertainty: the posterior SD of FA_t is 0.049 at FA_t ≈ 0.15 and 0.100 at 0.40 (largest at 0.3–0.5, the crossing/fanning range), so the mid/high_FA split is the less certain one.
- The lower edge of the cloud is µFA ≥ FA (C_c ≤ 1): it holds in 99.85% of voxels, with violations ≤ 0.05 from taking marginal posterior means.
- The narrow FA–µFA curve of the posterior means is mostly prior shrinkage: in low_FA voxels the SD of FA is 0.071 (classical) vs 0.019 (network), of µFA 0.160 vs 0.061. The rising-then-saturating shape is present in the classical fit too.

**rankfield compression of the posterior: not worth it.** K-means prototypes (K = 8–64) plus rankfield's ranked encoding reconstruct the posterior means within 0.1–0.3 posterior SD, but the store (2.5 MB at depth 2, 6.4 MB at depth 6) saves little against the raw means + SDs (6.6 MB) and is much larger than a label map + codebook (0.6 MB). The five class probabilities do not compress either (2.5 vs 2.4 MB): too few classes on too small a grid.

**FastSurfer as an independent check.** haversack (github.com/mhalle/haversack) runs FastSurfer 2.5.4 `asegdkt` on MPS in 100 s and writes a ranked store: 7.4 MB for 79 classes on the 257³ conformed grid (220 MB of logits), the case rankfield was built for. Pipeline: `scripts/t1_to_dwi.py` (rigid MI registration of the T1 to the mean b0: 0.34°, 1.9 mm; EPI distortion near the frontal sinuses and temporal poles is not corrected), `scripts/fastsurfer_to_dwi.py` (softmax over the classes the store names, grouped into 13 regions, box-averaged into each 2 × 2 × 4 mm voxel from ~16 conformed voxels; FastSurfer sums its views' raw logits, so the probabilities are a log-linear pool, not calibrated), `scripts/fastsurfer_vs_qti.py`. Volumes are plausible: cortex 471 cm³, cerebral WM 434, thalamus 17, cerebellar cortex 103.

sub-01 ses-01, 95,801 posterior voxels (`results/fastsurfer_vs_qti_s0101.json`):
- Voxelwise r: FastSurfer GM fraction vs P(low_FA) 0.62; WM fraction vs P(mid_FA + high_FA) 0.73; fluid fraction (ventricles + CSF + background) vs f_fw 0.75.
- Voxels ≥ 80% one FastSurfer region, with the QTI winner and medians (C_c = (FA/µFA)², from posterior means for the network):

  | region | n | QTI winner | f_fw | FA_t | µFA_t | C_c,t | classical FA | µFA | C_c |
  |---|---|---|---|---|---|---|---|---|---|
  | cortex | 12,975 | 85% low_FA | 0.088 | 0.10 | 0.63 | 0.025 | 0.13 | 0.53 | 0.068 |
  | caudate | 343 | 84% low_FA | 0.078 | 0.11 | 0.68 | 0.027 | 0.15 | 0.61 | 0.072 |
  | putamen | 444 | 83% low_FA | 0.068 | 0.12 | 0.69 | 0.028 | 0.19 | 0.68 | 0.085 |
  | cerebellar cortex | 4,850 | 66% low_FA, 28% mid_FA | 0.075 | 0.14 | 0.74 | 0.035 | 0.20 | 0.58 | 0.130 |
  | thalamus | 808 | 61% mid_FA, 25% low_FA | 0.081 | 0.19 | 0.77 | 0.060 | 0.32 | 0.76 | 0.197 |
  | cerebral WM | 18,263 | 51% mid_FA, 36% high_FA, 12% low_FA | 0.094 | 0.32 | 0.84 | 0.140 | 0.43 | 0.84 | 0.281 |
  | cerebellar WM | 1,096 | 70% mid_FA, 27% high_FA | 0.081 | 0.29 | 0.87 | 0.110 | 0.40 | 0.88 | 0.214 |
  | pallidum | 176 | 44% mid_FA, 56% high_FA | 0.078 | 0.43 | 0.90 | 0.226 | 0.45 | 0.90 | 0.254 |
  | brainstem | 1,057 | 78% mid/high_FA, 21% free_water/mixed | 0.097 | 0.49 | 0.86 | 0.321 | 0.46 | 0.82 | 0.402 |
  | ventricles | 717 | 95% free_water | 0.94 | — | — | — | — | — | — |

- **Regions separate on the FA–µFA plane independently of the thresholds** (`results/fastsurfer_regional_scatter_s0101.png`): at similar low FA, µFA rises from cortex to caudate/putamen, then thalamus, cerebral WM, and pallidum/brainstem, in both the classical fit and the network. Exception: the network puts cerebellar cortex above the striatum (0.74), the classical fit below (0.58); possibly partial volume with cerebellar WM in 4 mm slices. Not interpreted yet.
- The cortex and striatum differ from WM in both factors: lower µFA and much lower coherence (C_c,t 0.03 vs 0.14). Pallidum and thalamus are fiber-rich deep gray: µFA as high as WM, coherence between.
- **The network's FA_t is below classical FA in every region** (cerebral WM 0.32 vs 0.43, cortex 0.10 vs 0.13), although removing free water should raise FA, so its C_c is about half the classical one. Explained in the next entry: the brain prior has almost no tissue at FA 0.3–0.5. Do not build classes on the network's C_c until the prior is fixed.
- Cortical µFA_t (0.63) sits well above classical µFA (0.53) because about 9% free water is removed; it inherits the open question of whether that f_fw is real (2026-11-04).
- **The brain mask includes non-brain:** 17,978 mask voxels (19%) are ≥ 80% FastSurfer background. Most are fluid by QTI (51% free_water, 28% mixed), but 21% get tissue classes. The anterior free-water blobs at z ≈ 16 are the eyes. Whole-brain statistics that use `prepare_session.py`'s mask include them; a T1-based brain mask (FastSurfer or SynthStrip via haversack) would remove them.
- Caveats: one session; uncalibrated FastSurfer probabilities; few pure deep-gray voxels at this resolution; rigid registration only; FastSurfer has a single cerebral-WM label, so the corpus callosum, internal capsule and centrum semiovale still need a tract atlas or tractography.

## 2026-11-06: why the network's FA_t is below classical FA — the brain prior's FA_t gap

`scripts/prior_fa_gap.py --device mps --with-v3` (`results/prior_fa_gap_s0101.{json,png}`; 20,000 simulated v2-prior voxels through sub-01's protocol, Rician SNR 30–140).

- **Neither estimator is biased on the prior's own voxels.** Simulated WM: network FA_t − true +0.002, classical FA − true FA_t −0.028 (classical reads FA_t slightly low, as free water lowers full FA; vs true full FA it is +0.04). GM: +0.001 and +0.007. So the classical fit is not inflated, and the real-data gap is not a classical bias.
- **The v2 prior's tissue FA_t is bimodal with an empty middle.** % of tissue voxels per FA bin, prior true FA_t vs real classical FA (f_fw < 0.3):

  | FA | 0–0.1 | 0.1–0.2 | 0.2–0.3 | 0.3–0.4 | 0.4–0.5 | 0.5–0.6 | 0.6–0.8 | 0.8–1 |
  |---|---|---|---|---|---|---|---|---|
  | prior v1 | 22.3 | 25.7 | 2.2 | 0.5 | 2.4 | 6.5 | 27.1 | 13.3 |
  | prior v2 | 16.4 | 28.4 | 5.0 | 0.5 | 0.8 | 3.1 | 18.2 | 27.7 |
  | prior v3 | 15.8 | 26.7 | 7.8 | 3.1 | 4.0 | 6.4 | 16.6 | 19.6 |
  | real | 14.5 | 31.6 | 19.8 | 12.7 | 9.8 | 6.5 | 4.5 | 0.5 |

  Prior WM has median FA_t 0.82 (5th percentile 0.56); real cerebral WM has classical FA 0.43 (5th–95th 0.20–0.69). WM in the prior is 60% single bundles with axis dispersion 0.03–0.4: real 2 × 2 × 4 mm WM voxels are far more dispersed. (The v1 → v2 widening made WM *more* anisotropic, after real µFA came out higher than v1's; macroscopic FA went the wrong way.)
- Reproducibility: `simulate.sample_brain` (and `sample_dtd`) draw mixture weights with `torch.distributions.Dirichlet(...).sample()`, which ignores the generator, so reruns of these simulations differ in the third decimal (a rerun gave classical-FA bin 0.4–0.5 → network 0.52 instead of 0.51). The v1/v3 rows were added to the JSON from such a rerun; the other values are from the original run. Fixed the same day (`simulate._dirichlet`, integer-α Dirichlet from sums of exponentials drawn from the generator; test in `tests/test_core.py`): simulations are now identical for a given seed, but differ from every earlier saved draw, so committed simulation results reproduce only statistically.
- **In the gap the posterior mean is a compromise between modes.** Network FA_t against classical FA, simulated: 0.21 at classical 0.3–0.4, then 0.51 at 0.4–0.5 (the posterior switches modes). Real voxels at classical 0.4–0.5 get 0.32, between the two, and lower than simulated in every bin above 0.4. The FA_t posterior SD peak at 0.3–0.5 (previous entry) is the same effect. Posterior means there do not describe any tissue the prior contains.
- **Real-data confirmation with the v3 prior** (wider dispersion, 7.1% of tissue at FA 0.3–0.5 vs v2's 1.3% and v1's 3.0%, `pct_tissue_fa_0.3_0.5` in the JSON; `posterior_maps_hhs.py ... _v3.pt _v3 --whole-brain`): FA_t rises toward the classical value where v2 had the gap (cerebral WM 0.32 → 0.38, thalamus 0.19 → 0.23, cerebellar WM 0.29 → 0.32) and does not move where both priors have mass (cortex, caudate, putamen ≈ 0.10–0.11). v3 also raises WM µFA_t (0.84 → 0.88) and f_fw (0.09 → 0.13, its free-water range is 0–25%).
- **Consequences.**
  - The network's FA_t, and C_c = (FA_t/µFA_t)², are biased low in dispersed WM and fiber-rich deep gray (thalamus); not in GM. The mid/high_FA class boundary at 0.4 sits in the gap.
  - WM µFA_t may be affected too: in simulation classical µFA reads 0.04 below true tissue µFA in WM, so real tissue µFA_t near 0.88 would be consistent with classical 0.84; v3 gives 0.88, v2 0.84.
  - Earlier statements that "network tissue µFA matches classical µFA in WM" (2026-11-04, 2026-11-06 whole brain) need this caveat.
- **Fix:** a prior whose WM covers the real FA range: more 2–3-bundle voxels with comparable weights and wider dispersion (up to ~1), checked by matching the simulated classical-FA histogram to the real one before training. Then retrain (CLAUDE.md task 4) and rerun held-out prediction, which is the arbiter between priors.

## 2026-11-06: prior v4, matched to real classical maps by region (no training yet)

`scripts/prior_match.py --priors 1 2 3 4` (`results/prior_match_s0101.{json,png}`). Each prior's pure WM and pure GM voxels go through sub-01 ses-01's protocol with Rician noise at SNRs drawn from the matching FastSurfer region (per-voxel SNR from the STE b = 1.5 repeats) and the same constrained fit as the real data; classical scalars are compared with real voxels ≥ 80% cerebral WM or cortex. Classical on both sides, so the fit's own biases cancel; summary statistics only, never voxels.

- **WM was wrong in orientation coherence, not microstructure.** Real cerebral WM: classical C_c median 0.28 (5th–95th 0.08–0.61); v1–v3: 0.76–0.87. µFA (0.84 real vs 0.81–0.87) and MD were roughly right. Real WM C_MD is also low (0.026 vs v2's 0.068): little free water or compartment-diffusivity spread.
- **GM was close** in every prior; v2's cortex µFA was high (0.60 vs 0.53).
- **v4** (`simulate.BRAIN_PRIORS[4]`; `sample_brain` now takes optional `wm_nb`, `wm_alpha`, or a whole dict for experiments): mostly 2–3 bundles (10/40/50%) with more equal weights (Dirichlet 3), dispersion 0.25–1.0 (v2: 0.03–0.4), free water 0–5% in WM (v2: 0–10%), extra-axonal D_iso 0.7–1.3 and D_Δ 0.1–0.7, GM neurite fraction 0.15–0.45. A wider dispersion range (0.15–2.0) matched the medians but gave a bimodal C_c; the narrower one matches the shape.

  | WM, classical | FA | µFA | MD | C_MD | C_c | % FA 0.3–0.5 |
  |---|---|---|---|---|---|---|
  | real cerebral WM | 0.43 [0.20, 0.69] | 0.84 [0.63, 0.93] | 0.88 | 0.026 | 0.28 [0.08, 0.61] | 48.3 |
  | v2 | 0.79 [0.52, 0.92] | 0.87 | 0.87 | 0.068 | 0.87 [0.40, 1.01] | 3.8 |
  | v4 | 0.42 [0.21, 0.69] | 0.85 [0.73, 0.91] | 0.86 | 0.025 | 0.25 [0.06, 0.64] | 48.4 |

  Wasserstein distances to real WM, v2 → v4: FA 0.32 → 0.012, C_c 0.50 → 0.022, C_MD 0.027 → 0.007. GM: µFA 0.60 → 0.55 (real 0.53).
- **Label-free check (the WM/GM classes are anatomy, QTI measures something else).** Simulated WM + GM against all real tissue (FastSurfer fluid + background < 20%, any region, 59,279 voxels) as joint histograms in the QTI planes. % of real tissue where the prior has under 10% of the real density: (µFA, C_c) v1 33, v2 26, v3 4, v4 6; (MD, C_MD) v1 7, v2 6, v3 22, v4 5. v4 is the only prior that covers both planes; v3 covered (µFA, C_c) by being wide everywhere, at the cost of MD and C_MD.
- **Not covered:** the real low-µFA tail of cortex (µFA < 0.35, CSF partial volume in 4 mm slices) and the cortex C_c tail (WM partial volume); deep gray matter has no class of its own and is covered only through the planes above.
- **Caveats:** one session; real "pure" voxels still carry partial volume; matching classical maps can fit the simulator to acquisition artifacts as well as to tissue (WM STE deficit, H5), so held-out prediction stays the arbiter. **Next:** train a v4 network and compare it with v2 on held-out prediction, simulated coverage, FastSurfer regions and the FA_t gap.

## 2026-11-06: v4 trained — the white-matter STE deficit was mostly the prior; longer training matters on real data

**Held-out prediction tests the prior directly** (`heldout_hhs.py`, k-NN on a 200k bank; v2 rerun after the Dirichlet fix, tag `_v2_reseeded`). WM error in noise sd (median bias):

| bank | STE b1500, repeat noise, slice 20 | STE b1500, noise model, classical regions, slice 22 | LTE b2000, noise model |
|---|---|---|---|
| v2 | 1.80 (−1.49) | 1.67 (−1.33) | 1.90 |
| v2 + r = 4 (WM only) | — | 1.28 (−0.34) | 1.86 |
| v4 | 1.31 (−0.55) | 1.33 (−0.36) | 1.93 |
| v4 + r = 1.5 (WM only) | — | 1.25 (−0.04) | 1.95 |
| v4 + r = 2 | — | 1.24 (+0.21) | 1.95 |
| v4 + r = 4 | — | 1.38 (+0.81) | 1.97 |

GM and CSF are unchanged or slightly better with v4 (GM STE 1.18 → 1.17, LTE 1.35 → 1.30, repeat noise).

- **The prior explains about two-thirds of the WM STE deficit.** The earlier conclusion that "no time-independent brain prior fits both encodings in WM" (2026-11-04) holds for v1–v3, whose WM was far too coherent; v4, matched to real classical maps, fits both. The "simulated null" (2026-11-05) used v2 voxels, for which v2 was right by construction, so it could not catch this.
- **H5 is weakened, not refuted.** With v4, a small residual remains (bias −0.36 to −0.55), and WM-only time dependence at r ≈ 1.5–2 removes it; r = 4 now overshoots. The time-dependence strength previously estimated (r ≈ 4) was mostly compensating for the prior. One session, one slice, the τ ranges still guesses; repeat across the sessions of 2026-11-05 before using either value.
- Why coherence affects a spherical-encoding prediction: STE signal does not depend on orientation, but the k-NN match uses the LTE volumes too; with a prior lacking dispersed WM, real dispersed voxels match tissues with the wrong size distribution (C_MD, free water), which then mispredict STE.

**Networks** (`posterior_smoke.py`, v2's recipe: 3000 × 512, SNR 20–150; and `_v4long`: 12,000 × 2048 on MPS, 16× the data, 22 min). Simulated test (v4 prior, SNR 30–140):
- v4 at v2's budget: errors and coverage close to v2's on its own prior (90% coverage 0.77–0.91).
- v4long: median |error| 10–45% smaller (WM µFA_t 0.017 → 0.015, GM MD_t 0.044 → 0.036, WM f_fw 0.028 → 0.020), intervals narrower (f_fw 90% width 0.146 → 0.089), **coverage not better** (f_fw 0.78, full-distribution targets 0.77–0.83 at 90%). The overconfidence is structural (flow, loss or test setup), not under-training.

**Real data** (`posterior_maps_hhs.py ... --whole-brain`, `scripts/network_regions.py`, `results/network_regions_s0101.json`), medians in voxels ≥ 80% one FastSurfer region, classical | v2 | v4 | v4long:

| | FA_t | C_c,t | µFA_t | f_fw (network) |
|---|---|---|---|---|
| cerebral WM | 0.434 \| 0.315 \| 0.384 \| 0.400 | 0.281 \| 0.140 \| 0.190 \| 0.207 | 0.840 \| 0.843 \| 0.883 \| 0.883 | — \| 0.094 \| 0.051 \| 0.036 |
| thalamus | 0.318 \| 0.187 \| 0.248 \| 0.276 | 0.197 \| 0.060 \| 0.095 \| 0.116 | 0.764 \| 0.772 \| 0.809 \| 0.812 | — \| 0.081 \| 0.060 \| 0.039 |
| cortex | 0.127 \| 0.098 \| 0.092 \| 0.090 | 0.068 \| 0.025 \| 0.026 \| 0.025 | 0.530 \| 0.625 \| 0.583 \| 0.611 | — \| 0.088 \| 0.087 \| 0.070 |

- **The FA_t gap narrows** (WM 0.32 → 0.40 vs classical 0.43); the FA_t posterior SD in WM drops 0.075 → 0.031 (no more two-mode compromise); tissue with FA_t 0.3–0.5: 11.5% → 18.9%. Not closed: in simulation the network is unbiased, so real WM still differs from v4 somewhere (the residual STE effect is one candidate).
- **WM f_fw tracks the prior's range** (v2 0–10% → 0.094, v3 0–25% → 0.128, v4 0–5% → 0.05): this protocol does not identify free water in WM. Report WM f_fw as prior-determined.
- **Training length moves real-data estimates as much as the prior change** in places (cortex µFA_t 0.583 → 0.611, C_MD,t 0.076 → 0.058, putamen C_MD,t 0.088 → 0.054). Networks trained at v2's budget were not converged on real data; earlier real-data numbers carry this uncertainty. Check convergence (seeds, longer runs) before reading real-data differences of a few hundredths.

## 2026-11-06: the v4 result replicates across five sessions — H5 is a small effect (r ≈ 1.5)

Same five sessions as the 2026-11-05 time-dependence grid (sub-01 ses-01/02/03, sub-02 ses-01, sub-06 ses-01; prepared locally with `prepare_session.py`), same settings (`heldout_hhs.py 100000 --noise model --regions classical --slice -1`, time dependence WM-only, `--tau-gm 0`). `scripts/prior_heldout_summary.py` → `results/prior_heldout_summary.json`. v2's base row reproduces the earlier grid (WM STE 1.67 vs 1.66, WM LTE 2.16 vs 2.16).

Mean over sessions, median RMS error / noise sd:

| config | STE b1500: WM / GM / CSF | LTE b2000: WM / GM / CSF | WM STE bias |
|---|---|---|---|
| v2 | 1.67 / 1.22 / 0.86 | 2.16 / 1.47 / 1.56 | −1.29 |
| v2, r = 4 | 1.28 / 1.22 / 0.88 | 2.17 / 1.46 / 1.58 | −0.26 |
| **v4** | **1.28** / 1.23 / 0.87 | **2.12** / 1.43 / 1.56 | −0.27 |
| v4, r = 1.5 | **1.23** / 1.23 / 0.87 | 2.12 / 1.44 / 1.62 | +0.07 |
| v4, r = 2 | 1.25 / 1.23 / 0.88 | 2.13 / 1.44 / 1.63 | +0.34 |
| v4, r = 4 | 1.48 / 1.24 / 0.88 | 2.17 / 1.44 / 1.61 | +1.01 |

Per session, WM STE bias: v2 −0.90 to −1.45; v4 −0.07 to −0.42 (negative in all five); v4 + r = 1.5 −0.10 to +0.28; v4 + r = 4 +0.84 to +1.29 (overshoots in all five).

- **v4 alone does what r = 4 did on v2** (WM STE 1.28 either way), and also improves WM and GM LTE extrapolation (2.16 → 2.12, 1.47 → 1.43). v4 was tuned on sub-01 ses-01's classical maps only; sub-02 and sub-06 are out of sample and behave the same.
- **H5, revised:** about 80% of the WM STE deficit (bias −1.29 → −0.27) was the prior's over-coherent WM. A small residual remains, negative in every session, and WM-only time dependence at r ≈ 1.5 removes it (1.28 → 1.23). r = 4 was fitted to compensate for the prior. Time dependence in WM is still supported as a small effect; its size is now about a fifth of the previous estimate (the scaling 1 + τ(√r − 1): √1.5 − 1 = 0.22 vs √4 − 1 = 1.0, at the same τ ranges, which are still guesses).
- CSF LTE gets slightly worse with any time dependence (1.56 → 1.62); CSF has no WM time dependence, so this is the bank composition shifting, worth a look if r is fitted per tissue.
- Implication for training (CLAUDE.md task 4): use v4 with WM-only time dependence at r ≈ 1.5, not r ≈ 4.

## 2026-11-06: the v2-era network results rerun with v4 — test–retest and LTE-only change, the simulation table holds

Three results in the Westin summary still came from v2 networks. Rerun with v4. New outputs carry the tags below; the v2 files are untouched.

**Network vs classical on simulation** (`posterior_smoke.py 0 --model ... --test-snr 20 35`, new `--model` option for evaluating a saved model under another tag; `results/posterior_smoke_hhs_{v2,v4long}_test20_35.json`). 4,000 voxels from each network's own prior, SNR 20–35. Tissue µFA median |error|, network / classical constrained:

| class | v2 network | v4long network |
|---|---|---|
| WM | 0.023 / 0.036 | 0.020 / 0.043 |
| GM | 0.065 / 0.086 | 0.060 / 0.083 |
| WM+CSF | 0.040 / 0.348 | 0.037 / 0.393 |
| GM+CSF | 0.085 / 0.299 | 0.087 / 0.269 |

- Tissue MD at CSF boundaries: classical 0.81–0.93 µm²/ms off (it reports the mixture), network 0.05–0.10.
- 90% coverage at this SNR: v4long 0.74–0.88, v2 0.79–0.88. Same picture as at SNR 30–140.
- The table previously quoted in the Westin summary as "v2" (WM 0.025 / 0.037, …) was the 2026-11-04 smoke test, prior v1 (untagged `posterior_smoke_hhs.json`).

**Test–retest** (`consistency.py B|C MODEL TAG`, new optional arguments; `results/consistency_{B,C}_v4long.json`). Same sessions and slices as before (sub-01 ses-01/02/03; sub-02/06/07/08/09 ses-01), regional medians, SD with ddof = 1 (reproduces the v2 table exactly):

| measure | v2: within / between / ratio | v4long: within / between / ratio |
|---|---|---|
| classical WM µFA | 0.013 / 0.010 / 0.8 | 0.013 / 0.011 / 0.9 |
| network WM tissue µFA | 0.0026 / 0.010 / **3.9** | 0.0069 / 0.010 / **1.5** |
| network WM tissue MD | 0.0033 / 0.012 / 3.5 | 0.0063 / 0.014 / 2.2 |
| network WM tissue C_MD | 0.0006 / 0.0019 / 3.3 | 0.0005 / 0.0006 / 1.3 |
| network WM f_fw | 0.0020 / 0.0044 / 2.2 | 0.0009 / 0.0010 / 1.1 |
| network GM tissue µFA | 0.0022 / 0.006 / 2.7 | 0.017 / 0.029 / 1.7 |
| network GM tissue MD | 0.0094 / 0.010 / 1.1 | 0.013 / 0.012 / 0.9 |
| network GM f_fw | 0.0026 / 0.0043 / 1.7 | 0.0016 / 0.0082 / 5.1 |
| network GM tissue C_MD | 0.0008 / 0.0012 / 1.5 | 0.0008 / 0.0038 / 4.7 |

- **The v2 "5× more stable, between-subject differences kept" result was mostly the prior.** With v4long, WM tissue µFA varies 2.7× more between sessions than under v2. It is still about half the classical within-subject SD, and between-subject SD is unchanged (0.010), so the ratio is 1.5 vs classical 0.9: better than classical, not 4× better.
- v2's caveat anticipated this: real WM sat in v2's FA gap, where the posterior mean is a compromise between modes and moves little with the data.
- WM f_fw and WM C_MD,t barely vary at all under v4long (SDs ≤ 0.001): both are pinned by the prior (WM f_fw range 0–5%), so their ratios carry no information. GM f_fw and GM C_MD,t now separate subjects (ratios ≈ 5), which is new.
- Classical values match the earlier run to ≤ 1e-4 except sub-08 (WM µFA 0.872 → 0.866; run-to-run alignment).
- Caveats as before: one subject's three sessions span menstrual-cycle phases; single-slice medians; posterior sampling unseeded (n = 50, negligible at these medians). Proper ICCs need the cohort (H7).

**H4, linear encoding only** (`h4_lte_only.py`, now with a v4 label; new networks `_v4longlte` = v4long's recipe with `--shapes LTE`, 12,000 × 2048 on MPS, and `_v4lte` = v4's recipe, 3,000 × 512 on CPU; `results/h4_lte_only_{v4long,v4}.{json,png}`). Tissue µFA median |error|, full → LTE-only, on each pair's training prior:

| pair | WM | GM | real WM: SD ratio, r | real GM: SD ratio, r |
|---|---|---|---|---|
| v2 (3000 × 512) | 0.022 → 0.023 (+5%) | 0.073 → 0.078 (+7%) | ×1.18, 0.93 | ×1.06, 0.84 |
| v4 (3000 × 512) | 0.016 → 0.028 (+75%) | 0.063 → 0.075 (+19%) | ×1.7, 0.80 | ×1.3, 0.72 |
| v4long (12,000 × 2048) | 0.014 → 0.025 (+79%) | 0.052 → 0.074 (+42%) | ×1.9, 0.80 | ×1.4, 0.67 |

- **Under v4, spherical encoding matters in white matter.** Dropping STE costs ~75% in WM tissue µFA error, against 5% under v2, at the same training budget, so this is the prior, not training length. v2's WM was coherent single bundles, the regime where LTE alone suffices; v4's dispersed WM is not.
- In GM the cost grows with training (19% → 42%): the longer-trained full network uses STE better.
- The v4long pair on other test priors: on v2 voxels WM 0.018 → 0.020 (the old result reappears), on v3 voxels 0.025 → 0.035.
- Real data: WM tissue µFA 0.906 (full) vs 0.886 (LTE-only), posterior SD 1.9× larger without STE, voxel r 0.80 (v2: 0.93).
- **H4 revised:** "WM µFA is recoverable from standard multi-shell data under any reasonable prior" held only for v2's over-coherent WM. Under the prior that matches real WM, LTE-only WM µFA is ~75% worse and ~2× less certain; usable, but STE earns its place in WM too. The GM conclusion (STE matters, the prior matters more) stands.
- Not rerun: the generic-prior pair (`_v0`, `_v0lte`), which does not depend on the brain prior.

## 2026-11-06: calibration — the overconfidence was the simulator's free water; training runs agree once trained long

**Calibration** (`scripts/calibration.py`, 8,000 simulated v4 voxels at the training SNR 20–150, 200 posterior samples each; `results/calibration_*.json`). 90% coverage of the long v4 network, split by whether the voxel's *true* f_fw is exactly 0:

| voxels | f_fw | MD_t | FA_t | µFA_t | C_MD,t | MD | µFA | C_MD |
|---|---|---|---|---|---|---|---|---|
| all | 0.78 | 0.87 | 0.89 | 0.88 | 0.87 | 0.83 | 0.83 | 0.78 |
| true f_fw = 0 (14%; WM 29%) | 0.00 | 0.70 | 0.89 | 0.88 | 0.86 | 0.42 | 0.39 | 0.03 |
| true f_fw > 0 | 0.91 | 0.89 | 0.89 | 0.89 | 0.87 | 0.89 | 0.90 | 0.90 |

- **Cause:** `sample_brain` drew free water as one of the M = 64 sampled micro-tensors, so true f_fw was a random count out of 64, and at a few percent of free water (WM: 0–5% in v4) often exactly 0 (clamped to the logit edge). The full-distribution moments then lose their free-water term. The signal cannot tell 0 from 1 free-water tensor in 64, so no posterior can cover that "truth". With free water present, coverage is 0.87–0.91 and the PIT histograms are flat: the network was not broadly overconfident, and longer training could not have fixed this (2026-11-06 v4 entry).
- **Fix:** `sample_brain(exact_fw=True)` (and `simulate_batch(exact_fw=)`, `posterior_smoke.py --exact-fw`, `calibration.py --exact-fw`): free water is one extra micro-tensor (3 I) carrying exactly the voxel's free-water fraction as a weight; `simulate.signal` and `posterior.targets` take the weights. True f_fw is continuous, never 0. Test in `tests/test_core.py`.
- **Long v4 with exact free water** (`_v4longfw`, 12,000 × 2048, MPS): 90% coverage f_fw 0.83, tissue targets 0.87–0.89, full-distribution 0.84–0.87 (all voxels; was 0.77–0.89). Remaining shortfall: WM f_fw 0.79 and the WM full moments that depend on it (0.81–0.85); the first f_fw PIT bin is 1.58, i.e. voxels with near-zero free water fall below the interval. That is the free water this protocol does not identify in WM.
- **Real data barely moves:** regional medians of `_v4longfw` vs `_v4long` agree within about 0.01 (seed-spread level) except brainstem (FA_t 0.498 → 0.473, C_c,t 0.31 → 0.28). The fix changes what the intervals mean, not the maps. Use `--exact-fw` for new training.

**Training-run spread** (`scripts/seed_spread.py`, three long v4 runs differing only in seed: `_v4long`, `_v4long_s1`, `_v4long_s2`; whole brain, sub-01 ses-01; `results/seed_spread_s0101.json`):
- Voxelwise, tissue voxels: the between-run SD of the posterior mean is 11–20% of the median posterior SD (e.g. µFA_t 0.0065 vs 0.058, FA_t 0.0061 vs 0.031).
- Regional medians (voxels ≥ 80% one FastSurfer region): range across runs ≤ 0.01 for nearly every target and region; largest 0.018 (pallidum FA_t) and 0.019 (brainstem full µFA).
- So the v4 → v4long shifts (cortex µFA_t 0.583 → 0.611, C_MD,t 0.076 → 0.058) were under-training, not seed noise: long runs converge to the same answer. From one long run, regional differences above about 0.01–0.02 are interpretable; short (3000 × 512) runs are not converged on real data.

**MPS:** two trainings at once on the M1 Max killed one of them at step 9,000 with "MPS command buffer execution failed: Impacting Interactivity" (macOS's GPU watchdog). Run one MPS training at a time.

## 2026-11-06: v5 — v4 plus the measured forward model

`posterior_smoke.py 12000 --batch 2048 --prior 4 --snr 10 150 --exact-fw --td-ratio 1.5 --alpha 0 0.03 --pulsation 0.15 --gain-sd 0.007 --seed 0 --device mps --tag _v5` (23 min). New `simulate_batch` options (all off by default, so earlier training reproduces): WM-only time dependence (STE at relative frequency r), signal-proportional noise (per-voxel α ~ U(0, 0.03); fitted tissue α 0.005–0.018), fluid pulsation (U(0, 0.15) × f_fw × max(0, 1 − b); CSF fit about 0.1 at b = 0.1), per-measurement gain jitter (sd 0.7%; STE period-3 offsets 1–1.5% peak to peak), SNR from 10 (was 20).

- **Calibration on its own simulator** (`calibration.py ... --exact-fw --td-ratio 1.5 ...`): 90% coverage 0.84–0.90, the same as v4 with exact free water; again only WM f_fw near zero falls short (0.79).
- **Real data, regional medians vs `_v4longfw`** (long runs; seed spread ≤ ~0.005 in WM, up to 0.02 in pallidum and brainstem; `results/network_regions_s0101.json`):
  - The **WM FA_t gap does not close**: 0.396 → 0.391 (classical 0.434). Time dependence is not its explanation.
  - **µFA_t drops where WM time dependence acts:** cerebral WM 0.882 → 0.866, cerebellar WM 0.923 → 0.911, thalamus 0.817 → 0.795, brainstem 0.898 → 0.878 (1–3× the seed spread, towards classical µFA). Without time dependence the network was absorbing part of the STE deficit as extra microscopic anisotropy. GM unchanged (cortex 0.613 → 0.608).
  - **Pallidum and brainstem anisotropy fall** (FA_t 0.424 → 0.376 and 0.473 → 0.427; C_c,t 0.21 → 0.16 and 0.28 → 0.24), beyond the seed spread there. Attributed to the SNR floor in the next entry.
  - f_fw, MD_t and C_MD,t: within about 0.01.
- **Low-signal edge voxels** (lowest 2% of b0, tissue): median posterior SD of µFA_t v2 0.040 (below mid-b0 voxels' 0.097: overconfident where it should be least sure), v4long 0.081, v4longfw 0.071, v5 0.090 (vs 0.062 mid-b0). Uncertainty now grows as signal falls.
- **Status:** v5 is the current best network: matched prior, calibrated where the data are informative, honest at low signal, and with the measured time dependence. Open: the WM FA_t gap (about 0.04) and a held-out check of the noise options (the k-NN held-out tests the prior and time dependence, not the noise model).

## 2026-11-06: v5 ablations — the pallidum/brainstem drop is the SNR floor, and it is a correction

Three long runs, each v5 with one addition removed (12,000 × 2048, seed 0, `--exact-fw`, mapped `--whole-brain`): `_v5_notd` (no time dependence), `_v5_nopuls` (no pulsation), `_v5_snr20` (SNR from 20 again). Regional medians in `results/network_regions_s0101.json`. These are single runs, so differences under the seed spread (≤ ~0.005 in WM, up to 0.02 in pallidum and brainstem) mean nothing.

| | `_v4longfw` | `_v5` | no td | no pulsation | SNR from 20 |
|---|---|---|---|---|---|
| pallidum FA_t | 0.424 | 0.376 | 0.369 | 0.361 | 0.398 |
| brainstem FA_t | 0.473 | 0.427 | 0.435 | 0.415 | 0.463 |
| brainstem C_c,t | 0.276 | 0.235 | 0.240 | 0.221 | 0.273 |
| cerebral WM µFA_t | 0.882 | 0.866 | 0.880 | 0.866 | 0.868 |
| cerebral WM MD_t | 0.858 | 0.844 | 0.859 | 0.845 | 0.843 |

- **The SNR floor explains the brainstem drop** (FA_t 0.036 of 0.046, C_c,t all of it) **and about half the pallidum drop** (FA_t 0.022 of 0.048; the rest is near the seed spread there, or the α/gain terms, which were not ablated). Time dependence and pulsation do not cause it.
- **Why there:** these are the low-SNR regions (`scripts/low_snr_bias.py`; b0 SNR = mean b0 / SD of the 15 STE b = 1500 repeats). The median is 24 in both, and 35% (pallidum) and 38% (brainstem) of voxels are below 20. Elsewhere the median is 45–80 and under 2.5% of voxels are below 20. Pallidum has a low b0 (iron, short T2: median 259 vs 430–600). Brainstem has a normal b0 but large repeat-to-repeat variation, so its noise is mostly physiological.
- **The drop is a correction.** On v4 with v5's forward model at SNR 10–20, networks trained from SNR 20 overestimate FA_t:
  - WM: +0.039 (`_v4longfw`) and +0.022 (`_v5_snr20`), against −0.007 for v5.
  - GM: +0.054 and +0.043, against +0.020 for v5.
  - Their 90% coverage is 0.82–0.84, against 0.89 for v5.
  - Below the training range they read noise as anisotropy.
  - At SNR 20–40 and 40–150 all three agree within 0.01.
  - Classical FA has the same noise bias, so classical pallidum and brainstem values (0.45, 0.46) are not a target.
- **Time dependence is what lowers WM µFA_t** (cerebral 0.882 → 0.866, cerebellar 0.923 → 0.911; without it 0.880 and 0.921). It also lowers WM MD_t by about 0.015 (cerebral 0.858 → 0.844, cerebellar 0.800 → 0.784), which the v5 entry missed. The simulated STE signal decays faster under time dependence, so the same observed signal implies a lower base diffusivity.
- **Pulsation** has no regional effect beyond about one seed spread. Without it brainstem FA_t is 0.415, slightly lower.
- **Caveat:** the brainstem's real noise is mostly physiological (structured, likely cardiac). The simulator's noise is Rician thermal plus a white signal-proportional term, so v5's low-SNR behaviour there is right in direction but not proven in size.
- **Status:** keep v5 as trained. Open: the WM FA_t gap (about 0.04), a structured physiological-noise model for the brainstem, and a held-out check of the noise options.

## 2026-11-06: cohort run, all 66 sessions — test–retest reliability (H7): the network does not raise regional ICCs

`scripts/cohort_run.sh` prepared, segmented and mapped every session. It took 1 h 45 min on the M1 Max, with no failures:
- align and fit as before (CPU, float64), in 2 CPU lanes;
- FastSurfer asegdkt (haversack, MPS) and v5's whole-brain posterior, one GPU job at a time.

`scripts/cohort_icc.py` → `results/cohort_icc_v5.json`:
- **Inputs:** regional medians in voxels ≥ 80% one FastSurfer region, per session; 22 subjects × 3 sessions.
- **ICC(1,1)** (one-way random effects) with a 2000-resample bootstrap over subjects, and the bootstrap interval of ICC(network) − ICC(classical).
- **Pairs:** classical against the network's tissue moments, and against its full-distribution µFA, MD and C_MD.

**Classical reliability is modest.** It is lowest for the QTI-specific quantities:

| | ICC range |
|---|---|
| FA | 0.27–0.74 |
| MD | 0.46–0.79 |
| µFA | −0.05–0.57 (cerebral WM 0.57; cortex 0.07) |
| C_c | 0.11–0.63 |
| C_MD | 0.06–0.56 |

Between-subject differences are small in this cohort. The between-subject CV of MD is 1.4–4.6% against a within-subject CV of 0.9–2.5%.

**Network against classical:** the bootstrap intervals of the difference exclude 0 only here:
- **FA_t, cerebral WM:** +0.23 [+0.03, +0.37] (0.43 → 0.65).
- **µFA_t, cerebellar cortex:** −0.39 [−0.73, −0.01].
- **MD_t, worse in GM:**
  - cortex −0.44 (0.46 → 0.02);
  - caudate −0.47;
  - cerebellar cortex −0.39.

  The network's full-distribution MD matches classical MD (0.50, 0.77, 0.82). So removing free water costs MD reliability in GM. This is not a session-level trade-off with f_fw: within subjects, MD_t and f_fw changes are uncorrelated (r ≈ 0 in cortex and cerebellar cortex, −0.3 in caudate).

**Near misses and the rest:**
- **Brainstem:** FA_t +0.26 [−0.03, +0.54] and µFA_t +0.26 [−0.01, +0.49]. That fits v5's low-SNR handling, but it is not established.
- **C_MD,t:** higher in every region (+0.02 to +0.27), but no interval excludes 0.
- **The 7-subject preview misled.** WM C_MD 0.22 → 0.73 at 7 subjects is 0.21 → 0.23 at 22.

**The network shrinks within- and between-subject variation together.** Comparing within-subject CV with between-subject CV of the subject means, classical against network:

| | within-subject CV | between-subject CV |
|---|---|---|
| C_MD | 17–76% → 3–10% | 12–48% → 3–12% |
| µFA | ~2× lower | 1.5–4× lower |

Their ratio, and so the ICC, barely moves. Subjects rank much the same either way:
- correlation of subject means for µFA: 0.84–0.95 in most regions (cerebellar cortex 0.20);
- for C_MD: 0.69–0.93 (putamen 0.20, pallidum 0.55).

This is the full-cohort version of the 2026-11-06 pilot lesson: a lower test–retest CV from a prior-informed estimator is not higher reliability.

**Reading:**
- **Regional medians average away the voxel noise the network is good at.** What remains between sessions is systematic: registration and partial volume against each session's FastSurfer, per-volume gains, physiology. The network cannot remove that, and it scales between-subject differences down with it.
- **The sessions are menstrual-cycle phases**, the dataset's purpose, so part of the within-subject variation may be biological. All these ICCs are lower bounds on measurement reliability.
- **H7, answered for regional medians:** the network adds accuracy (bias correction), not reliability. Exceptions are WM FA_t (better) and GM MD_t (worse).
- **Untested:** voxelwise or small-ROI reliability, where noise matters. Is the shrunken between-subject spread of C_MD closer to the truth? That needs a simulated cohort with known subject differences. Does the shrinkage depend on the prior? Rerun the posteriors with another network (`TAG=_v4long scripts/cohort_run.sh`, ~40 min; the other steps are skipped).

## 2026-11-06: Cardiff b-tensor data (b to 12): the simulator does not predict real high-b signal; the noise floor is non-central χ

**Data and preparation.** `scripts/fetch_cardiff.sh` fetches the data; `qtilab/cardiff.py` loads it; `scripts/prepare_cardiff.py` prepares it (→ `results/cardiff_{data1,data2}_checks.json`); `scripts/motion_eddy_cardiff.py` corrects it.
- **Acquisition:** two subjects, 3T Connectom, 3 mm, TE 88 ms. LTE and PTE have 439 volumes each, STE 182, at nominal b = 0–12.
- **Motion:**
  - between runs (rigid, b0 MI): 0.5–1.9°;
  - within runs: up to 1.4° and 1.3 mm.
  - tractline's eddy (one call per shape, b ≤ 6, MPS, about 20 s) halves the variation between repeated b0s: 3–4% → 1–2% of S0 in interior voxels.
- **b-values:** a cubic fit to b = 1–3 gives MD from PTE and STE that agrees within 1% (WM 0.894 / 0.889, GM 1.017 / 1.030). LTE's is 5–12% lower, consistent with its larger higher-order terms. So the nominal b is not grossly wrong. PTE bvecs are taken as plane normals (unverified).

**Noise floor** (corrected after the adversarial review, `docs/review/adversarial_review_2026-11-06.md` issue 2).
- **The plateau is the floor.** STE levels off from b ≈ 9 at the same value in CSF (true signal ≈ 0 there), WM and GM: 1.66–1.76 raw against a tissue b0 of about 217, i.e. 0.0086–0.0089 of S0 in tissue.
- **Its distribution is not Rician.** The plateau's mean/SD in CSF is 2.8 (data1) and 2.3 (data2), against 1.91 for Rician. That is non-central χ with N_eff 2.1 and 1.4 effective channels.
- **Thermal σ** is 0.0043–0.0053 of S0, so b0 SNR is about 190–230.
- **Correction:** the first version of the checks reported b0 SNR 22 and a Rician floor of 0.058. Those came from the b0 repeat SD, which is 3–5% of S0 and mostly motion and physiology. They were wrong by about 6× and are replaced.
- **New option:** `simulate.rician(..., channels=N)` and `--channels` in training. channels = 1 draws exactly as before.

**Test** (`scripts/cardiff_extrapolation.py` → `results/cardiff_extrapolation_{data1,data2}.json`; `_rician` / `_none` give the other floor models).
- **Target:** predict each voxel's b = 3, 4.5 and 6 powder averages (LTE, PTE, STE) from its b ≤ 2 volumes.
- **Classical:** the constrained QTI cumulant extrapolated.
- **Simulator banks:** 200k noise-free voxels per prior (exact free water, WM r = 1.5), k-NN (k = 20) on rotation-invariant b ≤ 2 features: shell powder averages and l = 2 SH power.
- **Floor:** predictions are passed per volume through the non-central χ mean at each voxel's σ.

Median signed error (prediction − observed) / observed, data1 / data2:

| WM-like | LTE 3 | LTE 4.5 | LTE 6 | PTE 6 | STE 3 | STE 4.5 | STE 6 |
|---|---|---|---|---|---|---|---|
| observed / floor | 25 | 19 | 16 | 5 | 9 | 3.7 | 1.9 |
| classical | +18 / +18% | +118 / +122% | +733 / +799% | +5 / +5% | +1 / +1% | −4 / −5% | −9 / −10% |
| prior v4 | −8 / −6% | −13 / −11% | −18 / −16% | 0 / −2% | −2 / −3% | −10 / −13% | −20 / −22% |
| prior v2 | −7 / −6% | −13 / −12% | −19 / −18% | −1 / −4% | −3 / −4% | −13 / −14% | −22 / −23% |
| v4, stick axons | −6 / −4% | −8 / −6% | −10 / −8% | +7 / +5% | −2 / −3% | −11 / −13% | −20 / −22% |

| GM-like | LTE 3 | LTE 4.5 | LTE 6 | PTE 4.5 | PTE 6 | STE 4.5 | STE 6 |
|---|---|---|---|---|---|---|---|
| observed / floor | 22 | 13 | 9 | 7 | 4 | 5 | 2.4 |
| classical | +22 / +20% | +126 / +110% | +571 / +475% | +23 / +30% | +72 / +89% | +12 / +23% | +34 / +56% |
| prior v4 | +4 / +5% | +11 / +10% | +15 / +13% | +17 / +18% | +26 / +29% | −1 / +2% | −2 / +3% |
| v4, stick neurites too | +6 / +7% | +19 / +19% | +32 / +31% | +21 / +22% | +37 / +39% | −2 / +2% | −2 / +2% |

- **Classical cumulant extrapolation fails beyond b ≈ 3,** as truncation predicts: the series diverges.
- **The simulator misses real tissue at high b, the same way under v2 and v4.**
  - **WM LTE is under-predicted** by 6–8% at b = 3 and 16–19% at b = 6 (15–45 noise SD). These shells are 16–25× the floor.
  - **Stick axons halve the WM LTE miss.** That is intra-axonal D_Δ 0.99–1; v1–v4 used 0.9–1.0, a radial diffusivity up to ~0.07 that only shows at high b. STE is unchanged by construction.
  - **GM LTE and PTE are over-predicted** by 4–29%. Stick neurites make it worse. The GM signal decays faster than the Gaussian-compartment priors allow, which is consistent with exchange, a hypothesis not tested.
- **What the floor did:**
  - GM STE is predicted within ±3%: the earlier "GM STE deficit" was the floor.
  - WM STE remains under-predicted at b = 4.5 (−10 to −14%) and b = 6 (−20 to −23%). That is about 3 noise SD at only 2–4× the floor, and the fitted N_eff differs between subjects. Not established.
- **Gradient nonlinearity is not the cause.** Splitting WM-like errors into the inner and outer thirds of radius from the brain centroid changes them by ≤ 3 percentage points. A uniform b-scale error is largely absorbed, since predictors match at b ≤ 2.

**Caveats:**
- two subjects, no repeats;
- nominal b with no waveforms (time dependence is only the single-ratio model, STE only);
- regions are classical-fit FA/MD classes, not anatomy;
- the plateau may hold a little true signal, which would make the floor slightly too high and the WM STE deficit an underestimate.

**What it means:**
- The prior's high-b behaviour is wrong by 5–30% in the directions above. The network's truncation correction comes from that behaviour, so it is approximate.
- How much this moves µFA and C_MD estimated from b ≤ 2 is the next test: `scripts/cardiff_networks.py`. It compares a b ≤ 2 network and classical QTI against b ≤ 6 reference networks, with two-channel noise, under v4 and v4-sticks, plus a Rician-noise reference for noise-model sensitivity.


## 2026-11-07: Cardiff networks — b ≤ 2 and b ≤ 6 networks agree in WM, disagree in GM tissue µFA; the reference is not independent

**Setup** (`scripts/cardiff_networks.py` → `results/cardiff_networks_{data1,data2}.json`).
- **Networks:** prior v4 with v5's forward model at Cardiff's SNR (`--snr 50 400`), 12,000 × 2048 steps on MPS, about 75 min each.
  - `_b2v4`: b ≤ 2, 179 volumes, single-channel noise (the floor is irrelevant there; the lowest b ≤ 2 shell is about 15× the floor).
  - Reference `_b6v4c2`: b ≤ 6, 470 volumes, two-channel noise (`--channels 2`).
  - Alternative references: `_b6v4stickc2` (stick axons, `--prior-set ax_dd=0.99,1.0`) and `_b6v4` (single-channel Rician).
- **Applied to** the motion/eddy-corrected data, each in its own b-range. Posterior means from n = 50 samples. Classes are from the b ≤ 2 classical fit: WM-like FA > 0.4 (about 7,500 voxels), GM-like FA < 0.2 and 0.5 < MD < 1.2 (about 14,000).

**In simulation the b ≤ 6 network knows little more than the b ≤ 2 one.** Median |error| against truth (`results/posterior_smoke_cardiff_*.json`, test SNR 150–350):

| | WM µFA_t | WM µFA | WM MD_t | GM µFA_t | GM MD_t | GM C_MD_t | GM f_fw |
|---|---|---|---|---|---|---|---|
| b ≤ 2 network | 0.013 | 0.021 | 0.013 | 0.039 | 0.034 | 0.020 | 0.025 |
| b ≤ 6 network | 0.011 | 0.021 | 0.012 | 0.030 | 0.027 | 0.015 | 0.021 |

- Under the prior and at this SNR, b ≤ 2 already carries most of what b ≤ 6 adds: no gain in WM and about 25% in GM.
- On real data the posterior SD of µFA is the same for both (WM 0.027, GM 0.042).
- **So the "reference" is not a better-informed truth.** It is the same prior, given more data that the prior says is nearly redundant. Agreement with it cannot validate the b ≤ 2 network.

**Real data,** regional medians, data1 / data2:

| | WM µFA_t | WM µFA | WM MD | GM µFA_t | GM µFA | GM MD | GM f_fw |
|---|---|---|---|---|---|---|---|
| reference b ≤ 6 (c2) | 0.928 / 0.926 | 0.853 / 0.851 | 0.876 / 0.893 | 0.772 / 0.755 | 0.619 / 0.597 | 1.018 / 1.010 | 0.070 / 0.072 |
| b ≤ 6, stick axons | 0.935 / 0.931 | 0.861 / 0.859 | 0.871 / 0.890 | 0.775 / 0.763 | 0.617 / 0.602 | 1.016 / 1.010 | 0.072 / 0.074 |
| b ≤ 6, Rician | 0.930 / 0.927 | 0.861 / 0.857 | 0.878 / 0.896 | 0.772 / 0.760 | 0.623 / 0.603 | 1.014 / 1.008 | 0.066 / 0.071 |
| **b ≤ 2 network** | 0.940 / 0.936 | 0.870 / 0.868 | 0.880 / 0.893 | **0.754 / 0.712** | 0.622 / 0.575 | 1.011 / 0.997 | 0.059 / 0.064 |
| classical b ≤ 2 | — | 0.869 / 0.869 | 0.839 / 0.852 | — | 0.646 / 0.624 | 0.953 / 0.943 | — |
| classical b ≤ 6 | — | 0.810 / 0.807 | 0.758 / 0.770 | — | 0.561 / 0.561 | 0.876 / 0.873 | — |

- **Reference sensitivity.** Stick axons and single-channel noise move the reference by ≤ 0.008 in µFA_t and ≤ 0.006 in MD (voxelwise median |diff| 0.005–0.010). The noise model matters little at b ≤ 6, where the floor is at most 1/2 of the signal.
- **WM.** The b ≤ 2 network is within about 0.01 of the reference: µFA_t +0.010, µFA +0.011–0.017, MD 0.000–0.004, f_fw identical, in both subjects. In simulation the expected difference is about 0 (the two networks' biases match). So 0.01 is a real-data discrepancy, but about the size of the reference's own prior/noise sensitivity.
- **GM.** The b ≤ 2 network's tissue µFA is lower, by 0.018 (data1) and 0.043 (data2), and its f_fw lower by 0.01. Simulation predicts about −0.005, and the reference variants move ≤ 0.008. This is the shell the extrapolation test found mis-modelled: the prior over-predicts GM LTE/PTE signal at b = 4.5–6 by 10–29%. When the data stop following the prior's forward model, adding b ≤ 6 data moves the posterior. **It does not say which estimate is right.** Neither variant tested changes GM (stick axons are WM only).
- **Classical b ≤ 2.** Its MD sits below the reference by 0.034–0.039 (WM) and 0.062–0.063 (GM). That matches its simulated truncation bias against the true full distribution (−0.032 WM, −0.062 GM). Its µFA is 0.012–0.015 above the reference in both classes. Its C_MD (0.05 WM, 0.10 GM) is a third to a half of the network's full C_MD, which is mostly free water (f_fw 0.026 in WM is worth about 0.12 of C_MD) and not identified at b ≤ 2 by either method.
- **Classical b ≤ 6** drifts further from the reference, the cumulant expansion's breakdown. Its MD drift (−0.12 WM, −0.14 GM) is close to its simulated drift against truth (−0.11, −0.13), and so is WM µFA (−0.04 vs −0.05). GM µFA drifts more than simulated (−0.04 to −0.06 vs −0.01), again in GM.

**What it means:**
- **This test cannot measure the network's accuracy.** The b ≤ 6 reference shares the prior, and under that prior it is barely better informed. A prior-independent reference needs an independent model of the high-b signal, or a phantom.
- **What it does show:** on real data the b ≤ 2 and b ≤ 6 networks agree to about 0.01 in WM and disagree by 0.02–0.04 in GM tissue µFA, subject-dependent, where the prior's high-b signal is known to be wrong. That bounds how much of the GM tissue µFA is set by the prior's assumed high-b behaviour: about 0.02–0.04, or 3–6%.
- **Classical b ≤ 2 behaves on real data as simulation predicts** relative to the network, consistent with, but not proof of, the truncation correction being right in direction and size for MD.
- Next, if pursued: a GM prior with faster high-b decay (an exchange-like term or a non-Gaussian neurite compartment) and check whether the b ≤ 2 vs b ≤ 6 GM gap closes. Two subjects only.

## 2026-11-07: RICE (iRICE + CWLLS) run against qtilab — same answers on a full protocol; different µFA definition

**Setup.** The authors' package (github.com/CollapsarEcho/iRICE-CWLLS, commit 8cf9c90) was cloned to `work/ext/` and run unmodified.
- **Solver:** CLARABEL, since there is no MOSEK licence. All 12 of its tests pass. Residual constraint violations are 0.03% of voxels on HHS.
- **Script:** `scripts/rice_compare.py` → `results/rice_compare.json` (literature context: `docs/review/literature_check_2026-11-07.md`).
- **iRICE** is the second-order cumulant without its ℓ = 4 covariance terms: 13 coefficients for LTE+STE. Its constraints are 0 ⪯ D ⪯ 3.1·I, Q₀ ≥ 0, T₀ ≥ 0, and a mean-dependent bound on total variance. It solves only the voxels whose least-squares estimate violates a constraint.

**Its µFA is a different quantity.**
- iRICE's `uFA` is Lasič's µFA, 1.5a / (a + 3 MD²). QTI's (Westin's) is 1.5a / (a + 3(MD² + V_MD)). Here a = ⟨|dev D|²⟩ over the micro-tensors, so the difference is whether the isotropic variance V_MD sits in the denominator.
- Lasič's µFA can exceed 1 for genuine distributions with size variance (3% of the generic prior's truth).
- On exact second-order signals with the HHS protocol, iRICE recovers the Lasič µFA within 0.002. Against Westin's it reads +0.03 (WM) to +0.05 (GM) high with no noise at all.
- Both definitions are scored below, from the same qtilab fit (`ufa`: Westin; `ufa_l`: Lasič).

**Simulation, HHS protocol** (107 volumes, LTE+STE, QTI rank 23). Median signed / absolute error against truth, brain prior v4, 3,000 voxels:

| | WM µFA_L | WM MD | WM C_MD | GM µFA_L | GM MD | GM C_MD |
|---|---|---|---|---|---|---|
| iRICE CWLLS, SNR 50 | −0.009 / 0.021 | −0.039 / 0.040 | −0.095 / 0.095 | +0.006 / 0.041 | −0.057 / 0.057 | −0.116 / 0.116 |
| qtilab constrained, SNR 50 | −0.010 / 0.020 | −0.038 / 0.038 | −0.102 / 0.102 | +0.012 / 0.042 | −0.058 / 0.058 | −0.124 / 0.124 |
| iRICE CWLLS, SNR 20 | −0.014 / 0.034 | −0.041 / 0.047 | −0.051 / 0.073 | −0.005 / 0.076 | −0.054 / 0.061 | −0.088 / 0.093 |
| qtilab constrained, SNR 20 | −0.015 / 0.029 | −0.023 / 0.038 | −0.073 / 0.080 | +0.040 / 0.070 | −0.048 / 0.055 | −0.125 / 0.126 |

- **Practically the same estimator here.** Dropping the ℓ = 4 terms costs ≤ 0.006 on exact signals (WM MD and C_MD), and nothing visible with noise.
- **The bias is shared.** The dominant error in both is cumulant truncation (C_MD −0.07 to −0.13, MD −0.04 to −0.06), which neither method corrects.
- iRICE's µFA_L bias in GM at SNR 20 is smaller (−0.005 vs +0.040). The generic prior gives the same picture.

**Simulation, their 20-volume protocol** (b0; LTE b = 1 × 10 and b = 2 × 6; STE b = 2 × 3). QTI's design rank is 18 of 28 here, so full QTI is not identifiable and qtilab's fit is constrained only by the cone. Brain v4:
- **Without noise,** both are fine: iRICE µFA_L −0.009 (WM), +0.017 (GM).
- **At SNR 50 and 20,** both over-estimate GM µFA_L badly: iRICE +0.20 / +0.25, qtilab +0.13 / +0.21. WM is +0.05 / +0.06 for iRICE and −0.003 / +0.03 for qtilab.
- **Unconstrained least squares** is far worse: GM µFA_L +0.33 / +0.52, with µFA > 1 in 36–74% of voxels. The constraints are doing most of the work.
- **Conclusion:** at this SNR, a 1.5-minute protocol's GM µFA is dominated by the positive noise bias, with either estimator. The paper's simulations (SNR 30) are qualitative and do not show this.

**Real data,** HHS sub-01 ses-01 (95,801 voxels; CPU; iRICE with 8 workers):

| | WM-like MD | WM µFA_L | WM C_MD | GM-like MD | GM µFA_L | GM C_MD | time |
|---|---|---|---|---|---|---|---|
| iRICE WLLS | 0.878 | 0.868 | 0.040 | 0.921 | 0.596 | 0.104 | 4.5 s |
| iRICE CWLLS | 0.884 | 0.858 | 0.031 | 0.921 | 0.584 | 0.103 | 51 s (36% re-solved) |
| qtilab constrained | 0.890 | 0.857 (Westin 0.849) | 0.043 | 0.923 | 0.583 (Westin 0.559) | 0.100 | 104 s (all voxels; 26–32 s on MPS) |

- **On the same definition,** regional medians agree within 0.006 (MD), 0.001 (µFA_L) and 0.012 (WM C_MD; iRICE's constraint shrinks C_MD more).
- **Their D_max bound does not stop µFA_L > 1:** 1.2% of WM voxels with iRICE vs 0.9% with qtilab.

**What it means:**
- **On a full LTE+STE protocol, iRICE and constrained QTI are interchangeable** for MD, µFA and C_MD: same accuracy, same truncation bias, comparable time on CPU. qtilab has the GPU.
- **QTI's extra is C_c and the full covariance,** which need planar encoding; LTE+STE (HHS) does not identify them anyway (rank 23).
- **iRICE's real niche is the short protocol,** where QTI is not identifiable. There GM µFA is badly biased at realistic SNR by either estimator.
- **Reporting:** iRICE's uFA is not QTI's µFA. On real GM it is 0.024 higher (0.583 vs 0.559), which is larger than most of the effects we report. Any comparison must state the definition.
- **For a tools paper,** "fast constrained QTI" is not distinguished from iRICE on speed or accuracy for LTE+STE data. Only the full model (PTE, C_c) and the GPU batching set it apart.

## 2026-11-07: batched interior-point solver for the constrained fit — 5–9× faster than ADMM and fully converged

Prompted by RICE's per-voxel interior-point solves (previous entry): the useful lesson from their solver was the algorithm class, not the code. `qtilab/ipm.py` is a batched log-barrier method for the same QTI+ (dc) problem as `fit._admm`, selected with `fit.constrained(..., solver="ipm")`. **ADMM stays the default** until we decide to switch: every existing result used it.

**Method.**
- **Central path.** It follows the central path of t f(x) − log det ⟨D⟩ − log det C (ν = 9) with one Newton step per voxel per iteration.
  - The Newton matrix is 28 × 28 and positive definite even on rank-deficient designs, so each step is one batched Cholesky. There are no eigendecompositions except a small `eigvalsh` in the line search.
  - The barrier Hessian in Mandel coordinates is the symmetric Kronecker product of the inverse block. It is checked against autograd in the tests.
- **Step length.** The exact minimizer along the Newton direction, from the eigenvalues of the step relative to D and C. This halves the Newton steps compared with the damped step 1/(1+λ).
- **Schedule.** t grows ×30 when a voxel is centred. Stop at ν/t ≤ 1e-8 × (WLS RSS/2).
- **Float32 fixes** (each found by a failure on MPS):
  1. Solve for the step from the WLS point: the gradient H d + (H p − g), with the last term computed once. Hx − g directly is a cancellation of terms on the scale of the signal energy. It left the Newton decrement too noisy to certify centring, and 2–4% of voxels ran to the iteration cap.
  2. A voxel whose factorization fails (float32's precision limit, near the end) keeps its last strictly feasible iterate. It counts as converged if ν/t was within 100× of the target.
  3. A voxel whose decrement stalls moves on after 12 steps (float32 only). Applying that rule in float64 without the line search accepted uncentred points: µFA errors up to 0.4 before it was restricted.
  4. Jacobi equilibration of the Newton matrix (did not change the failure rate; kept as harmless).
- **Tolerance in float32.** A tolerance relative to the RSS is loose in high-residual voxels. At 1e-5, Cardiff C's small eigenvalues stayed about 4e-4 inside the cone (true value 1e-9), and low-µFA voxels were off by up to 0.04. At 1e-8 that is gone, at no extra time, because float32 stops at its own limit anyway.

**Accuracy against DIPY** (`scripts/validate_dipy_hhs.py`: HHS sub-01 ses-01, 2,500 violating + 500 other voxels, CPU float64; the DIPY cache was stale after the Nov 6 re-preparation and was recomputed):

| | time | converged | µFA vs tight ADMM, p99 / max | objective vs CLARABEL, p99 |
|---|---|---|---|---|
| ADMM default | 5.4 s | 84% | 2e-4 / 7e-4 | 9.6e-10 |
| ADMM tight (1e-9/1e-7, 20k iterations) | 39 s | 99.8% | — | 7.6e-10 |
| **IPM default** | **0.64 s** | **100%** | **2e-7 / 9e-7** | **7.6e-10** |
| DIPY CLARABEL | 36 s | — | — | — |

- **Against CLARABEL,** IPM agrees exactly as well as tight ADMM: µFA p99 1e-4, the same objective. The remaining difference is CLARABEL's own tolerance, since it violates the cone by up to 1e-5 in 65% of voxels.
- **Feasibility:** IPM's output is strictly feasible (minimum eigenvalue +1.6e-11).
- **DiB_70** (full rank, 72k voxels, almost all constrained; `scripts/bench_constrained.py`; DiB_217 is missing from the local sparse checkout):
  - IPM: 15.1 s, all converged, median 26 Newton steps.
  - ADMM: 78.9 s.
  - Agreement with CLARABEL is the same as ADMM at tight tolerances (µFA p99 5.3e-5).

**Whole sessions** (`scripts/bench_ipm.py`, `results/bench_ipm_m1max.json`). All voxels, default chunking. Error against IPM at 1e-11 in float64, over violating, non-collapsed voxels:

| | HHS (95.8k vox, rank 23) | | Cardiff data1 b ≤ 2 (47.8k, full rank) | |
|---|---|---|---|---|
| | time | µFA p99 / max | time | µFA p99 / max |
| CPU ADMM | 107 s (85% conv.) | 2e-4 / 1e-3 | 7.5 s | 2e-4 / 5e-4 |
| CPU ADMM, 300 iterations | 56 s | 2e-3 / 5e-3 | 7.4 s | same |
| **CPU IPM** | **11.7 s** (23 / 46 steps) | **2e-7 / 9e-6** | **4.6 s** | **2e-6 / 3e-4** |
| MPS ADMM (float32) | 30 s (71% conv.) | 2e-4 / 1e-3 | 12.1 s | 2e-4 / 5e-4 |
| **MPS IPM** (float32) | **4.3 s** (98% conv.) | **2e-4 / 2e-3** | **1.9 s** | **5e-5 / 3e-4** |

(MPS timings in this run were about 25% slower than earlier the same day: ADMM 22.8 s then, IPM 3.0 s. Compare within a row group.)

- **Speed:** 9× on CPU and 7× on MPS for HHS. Cardiff: 1.6× on CPU (ADMM converges there in about 200 iterations) and 6× on MPS.
- **Convergence:** IPM converges every voxel in float64. ADMM leaves 15% of HHS voxels short of its own stopping rule.
- **Float32 floor:** about 2e-4 p99 in µFA on HHS for both solvers. It comes from the float32 problem data (H, g), not the solver.
- **The cheap alternative** (capping ADMM at 300 iterations) halves the time but costs 10× in accuracy. Not worth it now.

**Where the time goes, and what to optimize next** (measured, 60k systems per step; details in the `qtilab/ipm.py` docstring):
- **MPS: PyTorch's batched linear algebra is the bottleneck.** The 28 × 28 Cholesky + solve takes 39 ms, barely faster than CPU (68 ms in float64).
  - A one-thread-per-voxel Metal kernel (`torch.mps.compile_shader`) does it in 7.6 ms (5.2×, same accuracy).
  - The 6 × 6 inverses take 18 ms through torch.linalg and 1.6 ms written as column operations.
  - A Metal kernel for the whole Newton step is the obvious next step. My estimate (not measured) is HHS in under 1 s on MPS.
  - torch.compile of an unrolled Cholesky fails on MPS: the generated kernel exceeds Metal's buffer-argument limit.
- **CPU:** the cost is the Cholesky (68 ms), the barrier-Hessian gathers (55 ms) and the small inverses (28 ms). The line search costs about as much as it saves on CPU, but it is still a small net gain.
- **Algorithmic:** a primal-dual method with Nesterov–Todd scaling and Mehrotra predictor–corrector (CLARABEL's) typically takes 10–20 iterations instead of about 23.
- **CUDA** (Modal L40S, `scripts/modal_check_hhs.py`, `results/modal_check_hhs_sub01_ses01_L40S.json`, whole HHS session):
  - float32: IPM 0.8 s (98% converged) vs ADMM 4.6 s (70%). µFA vs the tight float64 fit: p99 2e-4 for both, max 7e-4 (IPM) vs 6e-4 (ADMM).
  - float64: IPM 3.0 s (100%) vs ADMM 12.6 s (85%). µFA within 8e-7 of the tight fit (ADMM 7e-4).
  - **Memory:** IPM's peak was 16 GB in float32 and 32 GB in float64 for 59k constrained voxels in one call, about 3× ADMM's 5.3 GB. The GPU chunk size had been set from ADMM's estimate, so a 16–24 GB card could have run out of memory.
  - Fixed: on GPUs the constrained voxels now go to `ipm.barrier` in sub-batches sized from free memory at 75k × element size bytes per voxel (`fit._ipm_chunk`). That gives one call for HHS on the M1 Max (74k) and the L40S in float32 (61k), and about 29k on a 24 GB card. Sub-batching is tested to give identical results. The fix is not yet re-run on CUDA.

**Open decisions:**
- Make IPM the default solver. It is faster, fully converged and strictly feasible, but every number in these notes so far is ADMM's, which agrees to ≤ 1e-3 in µFA.
- Add RICE's D_max bound. In a barrier formulation it is one more log det term.
