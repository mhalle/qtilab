# Adversarial review, first pass (2026-11-06)

**Scope.** I read `CLAUDE.md`, `README.md` and all of `docs/NOTES.md`, treating NOTES as the authors' own account. I also read the scripts and result JSONs behind the claims examined below. I ran light CPU-only checks:

- reading the JSONs;
- one new computation: the held-out "simulated null" redone under prior v4, using a scratch copy of `scripts/heldout_diagnostics.py`, run outside the repo with outputs in a scratch folder;
- noise-floor statistics on the Cardiff STE data.

Nothing in the repo was modified apart from this file.

The Cardiff networks (`results/posterior_smoke_cardiff_*`) are still training (`posterior_smoke_cardiff_b2v4.*` is untracked), so this review does not cover them.

## Summary

The engineering claims hold up: the GPU fit matches DIPY, float32 matches float64 on MPS/CUDA, and the timings check out. I found nothing that undermines them.

The scientific claims are another matter. Almost every claim about what the network gets right is measured on the network's own simulator. The real-data checks that do exist test the prior's forward model through a k-NN bank, not the network's estimates. That bank has a resolution floor, and the remaining effects are about that size.

The five most severe issues:

1. **The H5 residual (WM time dependence) is at the predictor's floor once the null uses the right prior.**
   - I reran the "simulated null" under prior v4 instead of v2. Real WM STE error is 1.31 vs a null of 1.29 (sub-01 ses-01), 1.25 vs 1.24 (sub-06 ses-01), and 1.23 vs 1.31 (sub-02 ses-01).
   - What remains is a median bias excess over the null of −0.27, −0.26 and +0.08 noise SD. The authors' sign argument ("negative in all five sessions") is void, because the null's own bias is negative too (−0.11 to −0.18).
   - Even so, r = 1.5 is now built into v5, and the shift it causes ("lowers WM µFA_t by 0.016") is reported as a finding.
2. **The Cardiff STE high-b "deficit" is probably a noise-floor artifact.**
   - STE signal plateaus at 0.008–0.009 of S0 from b = 9 to 12, about the level of air in the raw images. The plateau is 50% of the signal at b = 6 and about 25% at b = 4.5.
   - The predictions add a single-channel Rician floor computed from a σ that is too small. Shortfalls of the reported size (−14% to −33%) are what an underestimated floor produces.
   - The checks JSON also states a b0 SNR of 21.8 and a Rician floor of 0.0575 of S0. The data contradict both by about 6×.
   - None of the Cardiff work is in NOTES.
3. **No claim about the network's accuracy has a reference independent of its prior.** The network-vs-classical tables, calibration, the low-SNR "correction" and the H4 revisions are all in-distribution.
   - The headline table scores classical QTI against a target it does not estimate (tissue µFA).
   - On the target classical QTI does estimate, the full-distribution µFA, the v4long network is **worse** than the classical fit in WM: median |error| 0.055 vs 0.036, with bias −0.045. NOTES does not report this.
4. **Prior v4 was tuned on the data later used to judge it.** The matched classical scalars (µFA, C_MD) are driven by the same STE run that the held-out test then predicts, for sub-01 ses-01. The other four test sessions share scanner, protocol and population, so they are not independent tests.
   - "v4 explains ~80% of the STE deficit" is therefore partly a property of the tuning.
   - The GM free-water range was never varied (0–15% in v1, v2 and v4). Cortical µFA_t (0.61 vs classical 0.53) rests on it.
5. **The cohort ICC results are reported selectively, without a multiplicity correction.** Seven of 80 network-vs-classical intervals exclude 0; about 4 would be expected by chance (the 80 are correlated).
   - NOTES lists five of the seven. It omits two that favour the network: full-distribution µFA in caudate (+0.18) and putamen (+0.12).
   - The one "network better" exception it keeps (WM FA_t, +0.23, lower bound +0.03) would not survive any correction, and it compares different quantities (FA_t vs FA).

Lesser issues follow below: single-seed ablation attributions, SNR definitions that mix physiological and thermal noise, stale wording left uncorrected, and regions selected from the data being tested.

---

## 1. The claims as they stand

| # | Claim (current wording) | Where | Evidence offered |
|---|---|---|---|
| C1 | The GPU constrained fit matches DIPY SDPdc, and MPS/CUDA float32 match CPU float64 | README; NOTES 2026-11-05 (night), 2026-11-06 Modal | `validate_dipy*.json`, `modal_check_hhs_*.json` |
| C2 | Truncation bias, not noise, is the dominant classical-QTI error; it inflates µFA (+0.03 to +0.06 in tissue, more at CSF boundaries) (H2) | NOTES 2026-11-04 evening | Noise-free simulation, Gaussian compartments |
| C3 | The network beats classical QTI on tissue moments, most of all at CSF boundaries | NOTES 2026-11-06 "v2-era results rerun" | `posterior_smoke_hhs_v4long_test20_35.json` (own prior) |
| C4 | Prior v4 matches real WM/GM classical maps and is preferred by held-out prediction | NOTES 2026-11-06 prior v4; v4 replicates | `prior_match_s0101.json`, `prior_heldout_summary.json` |
| C5 | ~80% of the WM STE deficit was prior misspecification; a small residual remains and WM time dependence at r ≈ 1.5 removes it (H5, "supported as a small effect") | NOTES 2026-11-06 "v4 replicates"; CLAUDE.md task 1 | `prior_heldout_summary.json` (5 sessions, 3 subjects) |
| C6 | The network is calibrated where the data are informative (90% coverage 0.84–0.90); the earlier overconfidence was the free-water count | NOTES 2026-11-06 calibration, v5 | `calibration_*.json` (own simulator) |
| C7 | v5 is the best network. Its time dependence lowers WM µFA_t by ~0.016. Its pallidum/brainstem FA_t drop comes from the SNR floor and "is a correction" | NOTES 2026-11-06 v5 and v5 ablations | `network_regions_s0101.json`, `low_snr_bias_s0101.json` |
| C8 | H4: under v4, STE matters in WM too (LTE-only is ~75% worse in WM µFA_t) | NOTES 2026-11-06 v2-era rerun | `h4_lte_only_v4*.json` (simulation) |
| C9 | H7: across the cohort, the network does not raise regional ICCs. Exceptions: WM FA_t (better) and GM MD_t (worse) | NOTES 2026-11-06 cohort | `cohort_icc_v5.json` |
| C10 | WM f_fw is prior-determined. Tissue µFA/MD are robust across brain priors | NOTES 2026-11-04 night (§3), 2026-11-06 v4 trained | Prior-sensitivity table; `network_regions_s0101.json` |
| C11 | The posterior anisotropy classes agree with FastSurfer anatomy | NOTES 2026-11-06 FastSurfer | `fastsurfer_vs_qti_s0101.json` |
| C12 | Cardiff: classical extrapolation fails; the simulator under-predicts WM LTE (8–18%) and STE at b ≥ 4.5 (14–33%) and over-predicts GM LTE/PTE; stick-like axons halve the WM LTE deficit | Commit e60aaf8 message only (not in NOTES) | `cardiff_extrapolation_data{1,2}*.json` |

C1 is outside the scope of this review. I found nothing against it; its own caveats are stated (the rank-deficient stopping rule, the dc relaxation allowing µFA > 1).

---

## 2. Issues, most severe first

### Issue 1: H5's remaining support is at the predictor's resolution floor (severity: high; changes a headline)

**Claim.** C5: "A small residual remains, negative in every session, and WM-only time dependence at r ≈ 1.5 removes it (1.28 → 1.23). Time dependence in WM is still supported as a small effect" (NOTES, "the v4 result replicates across five sessions"). r = 1.5 is then built into v5's training prior (NOTES v5), and its effect on the maps is reported as a result: "time dependence is what lowers WM µFA_t" (v5 ablations).

**Strongest case against.** The only null that separates predictor error from tissue misfit is `heldout_diagnostics.py`, and it simulates prior v2 (`scripts/heldout_diagnostics.py:72,99`). NOTES itself concedes that this null "could not catch" the prior problem (2026-11-06 v4 entry). It was never rerun under v4.

I reran it under v4 (a copy of the script with `version=4`, same settings otherwise; `--bank2 100000 --k2 20` to save time, which affects only diagnostic D). The v2 rerun reproduces the committed numbers (real WM STE 1.69, bias −1.35, vs 1.73, −1.37 committed; sim 1.37, −0.11), so the copy is faithful. Held-out run 8 (STE b = 1500), median RMS / noise SD (median bias):

| session | real WM, v4 bank | simulated v4 WM (null) | real GM | simulated GM |
|---|---|---|---|---|
| sub-01 ses-01 | 1.31 (−0.38) | 1.29 (−0.11) | 1.16 (−0.27) | 1.20 (−0.23) |
| sub-06 ses-01 | 1.25 (−0.41) | 1.24 (−0.15) | 1.20 (−0.28) | 1.19 (−0.23) |
| sub-02 ses-01 | 1.23 (−0.10) | 1.31 (−0.18) | 1.36 (−0.14) | 1.29 (−0.29) |

Three things follow:
- **In RMS, real WM is already at the floor** a correct tissue model reaches under this predictor, and in sub-02 it is below it. The 1.28 → 1.23 "improvement" from r = 1.5 takes real WM *below* the floor of the time-independent model it is compared with. That is consistent with r absorbing predictor bias, or with plain bank noise, as much as with physics.
- **What remains is a bias excess (real − null) of −0.27 (sub-01) and −0.26 (sub-06), and +0.08 in sub-02, which has no excess at all.** The sign argument ("negative in all five") is void, because the null is negative too: −0.11 to −0.18 in WM, and −0.23 to −0.29 in GM, where nobody claims an effect.
- **r is not identified.** Only τ·(√r − 1) enters, and the τ ranges are acknowledged guesses. r was picked from three values ({1.5, 2, 4}) on the same five sessions it is scored on. At r = 1.5 the WM bias overshoots to +0.25 and +0.28 in two of the five (`prior_heldout_summary.json`, `v4r1.5.per_session_wm_ste`).

So H5 reduces to a −0.26 SD median offset in the most coherent half of WM, seen in two of the three sessions checked. WM is selected as classical FA > 0.4 (`heldout_hhs.py:72`), while real WM median FA is 0.43. Candidate causes of an offset that size include STE gain calibration, STE-specific eddy currents (whose constant part leave-one-out cannot see, NOTES 2026-11-05 caveats) and residual prior shape error. Time dependence is one candidate among these.

The "along-fiber excess" (diagnostic C), cited as favouring physics (NOTES 2026-11-05 motion/eddy finding 3), is also prior-dependent. Under v4 the *simulated* null itself swings from +0.82 to −0.76 across angle bins (sub-01), against v2's +0.50 to +0.35. That diagnostic mostly measures predictor structure.

**Cheapest checks.**
- Rerun the v4 null (`--prior 4` copy, ~45 s per session on CPU) for the two remaining sessions (sub-01 ses-02/03), and repeat both real and null with 3 bank seeds (`--seed`) to get a bank-resampling SE for the RMS and bias numbers.
- Test r on sessions not used to choose it. All 66 sessions are prepared (`work/`), and `heldout_hhs.py` takes ~10–40 s per run on CPU.
- Report H5 as "a −0.26 SD STE offset in coherent WM above the predictor null, cause unknown" until a rotation-aware predictor (CLAUDE.md task 2) lowers the floor.
- Treat v5's r = 1.5 as an assumption, and its −0.016 WM µFA_t shift as prior sensitivity, not a measurement.

### Issue 2: the Cardiff high-b STE deficit is very likely the noise floor; the SNR figures are inconsistent (severity: high for C12)

**Claim.** C12, in commit e60aaf8's message only: the simulator "under-predicts … STE at b ≥ 4.5 by 14–33%", and the WM LTE deficit motivates stick-like axons. CLAUDE.md records "STE reaches the noise floor (~0.01 of S0) by b ≈ 6".

**Strongest case against.** Tissue-mask medians from `work/cardiff_data1_aligned.npz`, normalized to each run's b0:

| STE shell | b = 3 | 4.5 | 6 | 9 | 12 |
|---|---|---|---|---|---|
| median signal | 0.086 | 0.033 | 0.016 | 0.0086 | 0.0078 |
| SD over repeats | 0.0057 | 0.0042 | 0.0035 | 0.0027 | 0.0026 |

- The signal flattens at about 0.008 from b = 9, which is the noise floor. Air in the raw STE image has mean ≈ 2 against a central b0 of ≈ 168, i.e. 0.012 of S0, the same order.
- At that plateau, mean/SD ≈ 3, not the 1.91 of a Rayleigh floor. That points to a multi-channel (non-central χ) or otherwise processed magnitude.
- So the floor is about 50% of the observed STE signal at b = 6 and about 25% at b = 4.5.

`cardiff_extrapolation.py` models the floor as a single-channel Rician mean with σ = the SD across STE b = 4.5 rotations (`scripts/cardiff_extrapolation.py:115,126`; median ≈ 0.004). That gives an implied floor of about 0.005, roughly 0.003 below the observed plateau. A floor shortfall of 0.003 is about 20% of the b = 6 STE signal and about 10% at b = 4.5, the same size as the reported STE under-prediction (−14% at b = 4.5, −32% at b = 6; data1, v4).

The classical and every prior show the same STE shortfall (classical −22% at b = 6, v2 −35%, v4 −32%, "sticks" −32%). That pattern fits a measurement-side cause better than a tissue model that is wrong in the same way under every prior.

`results/cardiff_data1_checks.json` reports `b0_snr_tissue_median: 21.8` and `rician_floor_rel: 0.0575`. A floor of 0.0575 cannot be right when the signal falls to 0.008. The b0 SD (0.043 of S0) is dominated by non-thermal variation (motion, physiology, run-to-run gain), so it overstates the thermal noise by about 6×. Any analysis that uses this SNR is off by that factor.

**Other Cardiff confounds.**
- The Connectom's gradient nonlinearity is not corrected. Effective b varies spatially by a few percent to over 10% away from isocentre, which at b = 6 moves WM LTE signal by ~5–15%.
- Nominal b-values (CLAUDE.md: "more than the dataset description gives") and PTE bvec semantics ("presumably plane normals, unverified") are both open.
- The WM LTE under-prediction (−8% to −19%) is less affected by the floor (LTE b = 6 signal ≈ 0.096, so the floor is ~3%). Gradient nonlinearity and the b-value question still apply to it.
- Two subjects, no repeat sessions.

**Cheapest checks.**
- Replace the Rician floor with an empirical one: the per-voxel STE b = 12 mean, or a fitted non-central-χ floor using b ≥ 9 STE.
- Rerun `cardiff_extrapolation.py` on shells whose signal is ≥ 10× that floor (LTE/PTE to b = 6; STE to b = 3 only).
- Split the errors by distance from isocentre as a gradient-nonlinearity check.
- Write the Cardiff results into NOTES so the claims have an account to be checked against. CLAUDE.md task 7 still says a loader and motion/eddy correction are needed, which the commit already did.

### Issue 3: there is no reference independent of the prior for any network accuracy claim (severity: high; it bounds every network headline)

**Claims.** C3, C6, C7 ("the drop is a correction"), C8, and the v5 status line "matched prior, calibrated where the data are informative, honest at low signal".

**Strongest case against.**
- **C3 is in-distribution, and the estimands differ.** `posterior_smoke.py` scores the network against the tissue truth of its own prior, at SNR 20–35. It scores classical QTI against the same tissue truth (`qti_ufa_vs_tissue`), a quantity classical QTI does not try to estimate. At CSF boundaries the classical "error" (0.27–0.39) is just the mixture/tissue difference.
  - On the quantity classical QTI does estimate, full-distribution µFA, the same file (`results/posterior_smoke_hhs_v4long_test20_35.json`, `by_class`) gives:
    - **WM:** network median |error| 0.055 (bias −0.045) vs classical 0.036 (bias +0.006).
    - **GM:** 0.066 vs 0.077.
  - So in WM the network is less accurate than classical for the classical target. The cause is that full µFA depends on f_fw, which NOTES says the protocol does not identify in WM. NOTES does not state this result.
- **C6 is on the training simulator.** All calibration files use v4 (+ the v5 forward model) test voxels. Coverage on the own simulator says nothing about calibration on real tissue, where the misfits of issue 1 and 4 exist. Even there it is 0.84–0.90 at 90%, not 0.90.
- **C7's "correction".** `low_snr_bias.py` evaluates all three networks on v4 + v5's forward model, which is v5's own training distribution. It is no surprise that v5 wins at SNR 10–20. NOTES' own caveat (brainstem noise is physiological, the simulator's is not) is the reason the real-data conclusion does not follow.
- **C8 (H4 under v4)** is simulation on each pair's own prior. The real-data side has no truth, only SD ratios and correlations. The H4 sentence has already flipped once with the prior (2026-11-05 → 2026-11-06), which is itself evidence that it measures the prior.
- **C11 (FastSurfer)** checks that regional ordering on the FA–µFA plane is anatomically plausible. It says nothing about whether µFA_t, C_c,t or f_fw values are right. The classical fit gives the same ordering.

**Cheapest checks.**
- A **cross-prior** matrix: evaluate v5 on test voxels from priors it was not trained on but that pass held-out prediction about as well. Candidates are v4 "stick"/"sticks" from the Cardiff commit, v3, and v4 with fw_gm 0–25%. Report bias in µFA_t, FA_t and f_fw. This needs CPU only for evaluation (`posterior_smoke.py 0 --model ... --prior ...`), though it may need a `--prior-set` dict.
- Add the classical-vs-full rows to the C3 table in NOTES, so the comparison is like-for-like.
- For real data, the only truth-bearing checks available are phantoms (DiB water and liquid crystal, through a network trained on the DiB protocol) and the Cardiff high-b signal predicted from the *network's* posterior rather than the k-NN bank. The latter awaits the Cardiff networks now training.

### Issue 4: prior v4 was tuned on the data it is validated on (severity: high/medium; weakens C4, C5, C10)

**Claim.** C4: "v4 was tuned on sub-01 ses-01's classical maps only; sub-02 and sub-06 are out of sample and behave the same" (NOTES "v4 replicates").

**Strongest case against.**
- `prior_match.py` matches the simulated classical µFA, C_MD and C_c distributions to real ones. Classical µFA and C_MD come from the LTE-vs-STE contrast, i.e. from the STE volumes, run 8 included. The held-out test then predicts run 8 of the same session. For sub-01 ses-01 the "held-out" improvement is therefore partly in-sample.
- The other four test sessions come from the same scanner, protocol, sequence and healthy-young-female population. Population-level summary statistics transfer by construction, so "out of sample" here means other voxels, not an independent test of the prior.
- **Forking paths.** v4 is the end of a hand search: a dispersion range of 0.15–2.0 was tried and rejected for a bimodal C_c, alongside other choices of bundle counts, Dirichlet α and extra-axonal ranges. Each decision was made while looking at sub-01's maps. That is acceptable for model building, but held-out prediction on the same sessions can then no longer act as an independent "arbiter", as NOTES calls it (prior_match docstring; NOTES "prior v4").
- **The GM free-water range was never varied:** `fw_gm = (0, 0.15)` in v1, v2 and v4 (`qtilab/simulate.py:127–148`). Cortex µFA_t sits 0.08 above classical µFA "because about 9% free water is removed" (NOTES FastSurfer entry). WM f_fw was shown to track its prior range (v2 0.094, v3 0.128, v4 0.05). GM f_fw has had no matching test with v4-era networks. The only data point is v3 (0–25%), whose GM µFA_t moved little, but v3 changed many other ranges at the same time.
- C10's "tissue µFA/MD robust across brain priors" comes from the v1/v2/v3 table, all short-trained networks. NOTES later shows that short networks are not converged on real data, by shifts of up to 0.03 in cortex µFA_t (2026-11-06 v4 entry). That robustness table predates the convergence finding and was never redone.

**Cheapest checks.**
- Refit v4's ranges on half the cohort (11 subjects, ses-01), then compare held-out prediction on the other 11 against v2 and against a deliberately different "v4-like" prior. All CPU work, ~1 min per session.
- One v4 variant with `fw_gm = (0, 0.3)`: compare held-out prediction (CPU) first. If it does no worse, cortex µFA_t and f_fw are prior-determined like WM f_fw, and a network (MPS, later) would show by how much.

### Issue 5: the cohort ICC differences are reported selectively, with no multiplicity control (severity: medium; changes C9's exceptions)

**Claim.** C9: "the bootstrap intervals of the difference exclude 0 only here", listing WM FA_t (+0.23), cerebellar-cortex µFA_t (−0.39) and GM MD_t (cortex, caudate, cerebellar cortex). The headline: "the network adds accuracy (bias correction), not reliability. Exceptions are WM FA_t (better) and GM MD_t (worse)."

**Strongest case against** (`results/cohort_icc_v5.json`, `pairs`):
- There are 10 regions × 8 pairs = **80** intervals, and **7** exclude 0. NOTES lists 5. The two it omits both favour the network: caudate µFA (full) **+0.18** [+0.09, +0.30] and putamen µFA (full) **+0.12** [+0.002, +0.31].
- About 4 of 80 would exclude 0 at 95% by chance, fewer if the comparisons are correlated, which they are: regions share subjects and pairs share data.
- WM FA_t's lower bound is +0.03. It would not survive Holm or FDR over 80 comparisons.
- The comparison is classical FA (apparent, with free water) against network FA_t (tissue, free water removed by a prior-pinned f_fw), so it measures different quantities.
- The GM MD_t intervals are the most robust. The upper bound for caudate is −0.13 and for cerebellar cortex −0.18. That finding (removing free water costs GM MD reliability) is the best-supported one.
- The intervals are percentile bootstraps over 22 subjects, with resampled subjects counted as distinct, and ICC(1,1) estimated from three sessions each. Intervals this wide (e.g. cortex net µFA_t [−0.22, +0.45]) mean the null result "the network does not raise ICCs" is only "no difference detectable at n = 22". That is not equivalence.

**Cheapest checks.**
- Recompute from the JSON (no new data) with Holm/BH over the 80, and report all of them.
- Add BCa or subject-level jackknife intervals.
- Pre-register the pairs that count as primary (e.g. tissue moments only), and reword "only here" and "exceptions".

### Issue 6: the v5 attributions rest on single seeds and mismatched SNR definitions (severity: medium; C7)

**Claims.** "The SNR floor explains the brainstem drop … and about half the pallidum drop", and "the drop is a correction" (NOTES v5 ablations).

**Strongest case against.**
- **Seed spread is a range over three seeds of a different recipe** (`seed_spread_s0101.json`: v4long, no exact fw, SNR from 20). It was never measured for v5 (SNR from 10, extra noise terms), and with three draws the range is about 1.7 σ.
  - Pallidum FA_t has a seed range of 0.018 (n = 176 voxels). The v5 vs v5_snr20 difference there is 0.022, about one seed range, so "about half the pallidum drop" is unsupported.
  - Brainstem (0.036 against a range of 0.005) is more convincing.
- **The SNR used to place regions in the low-SNR bins** (`low_snr_bias.py:39`) is mean b0 / SD of the 15 STE b = 1500 repeats. That SD includes physiological noise and per-volume gains. The simulator's SNR is thermal. NOTES says the brainstem's noise is "mostly physiological", so "35–38% of voxels below SNR 20" does not map onto the simulator's SNR 10–20 bin where the bias was measured.
- **All of C7 is one session** (`network_regions_s0101.json`), although v5 maps exist for all 66 sessions.

**Cheapest checks.**
- Thermal SNR from the fitted noise model (`noise.fit_alpha_groups`'s s2, CPU, seconds) for brainstem and pallidum. If thermal SNR there is ≫ 20, the SNR-floor explanation fails.
- Regional v5 vs v4longfw differences across the 66 cohort sessions. These need v4longfw cohort maps (MPS, later); meanwhile, compare v5 cohort medians in pallidum/brainstem against classical across sessions.
- Two more v5 seeds when MPS is free.

### Issue 7: revised claims still stand uncorrected in NOTES, CLAUDE.md and the docstrings (severity: medium for publication hygiene)

Each of these has been superseded by a later entry but still reads as current where it was written:
- NOTES 2026-11-04 night, "Overall assessment": "The network's tissue estimates are robust across plausible priors and more reliable than classical QTI." Reliability was refuted by the v4long rerun and the cohort ICCs.
- NOTES 2026-11-04 night §4: "The network's session-to-session stability is not just prior shrinkage". Later shown to be mostly the prior.
- NOTES 2026-11-04 late: "a third of WM voxels remain unexplained … consistent with diffusion-time effects". The surprise maps were never redone under v4 or v5 and remain v2-era; they also have a known anisotropy-dependent floor.
- NOTES 2026-11-04 late night: "No time-independent brain prior fits both encodings in WM". Superseded by v4.
- NOTES 2026-11-05: "H4 in one sentence: WM µFA is recoverable from standard multi-shell data under any reasonable prior". Superseded.
- NOTES 2026-11-05: the "H5 status" list, including "a simulated null that rules out the predictor". The null was for the wrong prior (see issue 1).
- NOTES 2026-11-05 motion/eddy: "r ≈ 4 time dependence removes it equally"; "favors physics (axial non-Gaussianity…)".
- NOTES 2026-11-05 volume offsets finding 6, "H5 is not affected". Its argument assumed v2.
- CLAUDE.md task 7 (Cardiff "needs a loader … first") is out of date, and the Cardiff results are missing from NOTES.
- NOTES v5: "calibrated where the data are informative, honest at low signal". In-simulation only (issue 3).

**Cheapest fix.** Add a one-line "superseded by <date entry>" marker under each of these. Keep a short "current claims" section at the top of NOTES; whoever drafts a paper will otherwise quote stale lines.

### Issue 8: regions are selected from the data being tested, and anatomy rests on rigid registration (severity: medium/low)

- **Held-out regions.** These are classical FA/MD thresholds from the fit to *all* volumes, including the held-out run (`heldout_hhs.py:57–58,72`). WM = FA > 0.4 keeps only the more coherent half of real WM, and the simulated null's WM-like set changes size with the prior (sub-01: 1,502 voxels under v2, 812 under v4). So real and null compare different mixtures. Cheap fix: define regions from the remaining runs only, and from FastSurfer.
- **FastSurfer regions.** Rigid T1→b0 registration (1.9 mm translation), with EPI distortion uncorrected and no reversed-PE data in HHS. 4 mm slices. 19% of the mask is FastSurfer background. "≥ 80% one region" voxels in small structures (pallidum, 176 voxels) are exactly where misregistration bites.
- **Single-slice results.** Several earlier conclusions (prior sensitivity, the consistency pilot, surprise) use slice 20 or the "fullest slice" per session. Whole-brain reruns exist for some but not all of them.

### Issue 9: reproducibility and bank variance (severity: low/medium)

- Every k-NN number uses one fixed bank seed (`--seed 21`). No bank-resampling SE is reported, yet differences of 0.02–0.05 noise SD are interpreted (r = 1.5 vs 2; v4 vs v2r4 at 1.276 vs 1.285).
- Simulation results committed before the Dirichlet fix reproduce only statistically (NOTES 2026-11-06 FA-gap entry).
- Posterior sampling was unseeded in the slice-mode runs that feed region assignment in held-out and consistency.

**Cheap check:** three bank seeds for the v4 / v4r1.5 grid on one session (CPU, a few minutes).

### Issue 10: simulator assumptions that cap every simulation claim (severity: context, not a new finding)

C2 (truncation bias as the dominant error) and all network training assume Gaussian compartments, no exchange, no restriction beyond the √ω scaling, independent Rician noise (spatially correlated in reality, NOTES 2026-11-04 evening) and the HHS waveforms as an unknown scalar ratio.

Cardiff's classical µFA does fall with b_max (WM 0.84 → 0.79, GM 0.64 → 0.53 from b_max 2 to 6; `cardiff_data1_checks.json`), which looks like real-data support for C2's direction. But a noise floor that is underestimated or ignored also lowers apparent µFA at high b: it lifts STE more than LTE in log space. Until issue 2 is fixed, that trend cannot be credited to truncation alone.

---

## 3. What looks solid

- **C1.** The engineering validation (DIPY agreement on both protocols, float32 vs float64, the CUDA/MPS consistency) is careful, and its caveats are stated.
- **Self-correction.** The authors have revised several of their own headline claims when the prior changed (H4, H5, test-retest), and the calibration diagnosis (free water as a random count) is a convincing, mechanistic fix.
- **The GM MD_t reliability loss** in the cohort is the best-supported network-vs-classical difference (issue 5).
- **The volume-offset finding** (a reproducible period-3 STE gain pattern) uses no prior and no network. It replicates across subjects and is the cleanest real-data observation in the project.
- **"WM f_fw is prior-determined"** is an honest negative result, and well supported.

## 4. Suggested order of the cheap checks

All are CPU-only unless marked.

1. v4 null for all five sessions, plus bank seeds (issue 1). I ran three sessions for this review (sub-01, sub-02 and sub-06, ses-01), using a scratch copy of `heldout_diagnostics.py` with `version=4`.
2. Cardiff with an empirical floor and floor-safe shells (issue 2).
3. Holm/BH and complete reporting of `cohort_icc_v5.json` (issue 5; JSON only).
4. Thermal SNR for brainstem and pallidum from the noise model (issue 6).
5. Classical-vs-full rows in C3, and a cross-prior evaluation of v5 (issue 3; evaluation only).
6. A split-cohort prior-tuning test and an fw_gm variant (issue 4).
7. "Superseded" markers in NOTES and CLAUDE.md (issue 7).
