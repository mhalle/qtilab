# Literature check, 2026-11-07

**Purpose.** Check whether the project's main findings are known outside our own experiments, and confirm the citations I gave from memory.

**Method.** One pass of web searches, about 15 queries, plus two abstract fetches. Most entries were checked against abstracts and bibliographic records, not full texts. The searches were not exhaustive, so read "not found" as "not found in this pass".

## 1. Learned estimators inherit the training distribution (our H5, test–retest and H4 reversals)

**Known, and published.**

- **Gyori, Palombo, Clark, Zhang & Alexander.** "Training data distribution significantly impacts the estimation of tissue microstructure with machine learning." *MRM* 87(2):932–947, 2022. doi:10.1002/mrm.29014. [UCL open access](https://discovery.ucl.ac.uk/10141866/), [bioRxiv](https://www.biorxiv.org/content/10.1101/2021.04.13.439659.full.pdf).
  - Training on fit-derived (in vivo-like) parameter combinations gives precise estimates for typical tissue but inaccurate ones for atypical combinations. Uniform training trades this the other way.
  - They warn that high precision from machine learning can hide strong bias, and that visual inspection of maps is not enough. That is our test–retest and calibration experience.
- **Epstein, Bray, Hall-Craggs & Zhang.** "Choice of training label matters: how to best use deep learning for quantitative MRI parameter estimation." *MELBA* 2:586–610, 2024. [arXiv 2205.05587](https://arxiv.org/abs/2205.05587v2); code at github.com/seancepstein/training_labels.
  - Supervised networks trained on ground-truth labels are biased toward the training distribution.
  - Training on independent maximum-likelihood labels instead recovers the low bias of self-supervised fitting. The choice of label sets the bias–variance trade-off.
  - This is a concrete, cheap alternative to our posterior network's prior-driven shrinkage.
- **μGUIDE (Jallais & Palombo, eLife reviewed preprint).** [arXiv 2312.17293](https://arxiv.org/pdf/2312.17293v2.pdf), [eLife 101069](https://elifesciences.org/reviewed-preprints/101069v1).
  - Simulation-based inference with a neural posterior estimator: the same family of method as ours.
  - It reports uncertainty and "ambiguity" relative to the prior range, rather than trying to remove degeneracies.
  - Tested on Ball&Stick, the Standard Model and extended SANDI. Not QTI.

**Status of our claim.** The general pattern is established. What isn't established is its size for tensor-valued encoding and QTI moments (H4 under v4, Cardiff's b ≤ 2 vs b ≤ 6 dependence).

## 2. Grey matter at high b: faster decay than Gaussian compartments allow (Cardiff GM over-prediction)

**Known; exchange is the leading explanation.**

- **NEXI.** Jelescu, de Skowronski, Geffroy, Palombo & Novikov. "Neurite Exchange Imaging (NEXI): a minimal model of diffusion in gray matter with inter-compartment water exchange." *NeuroImage* 256:119277, 2022. [PubMed 35523369](https://pubmed.ncbi.nlm.nih.gov/35523369/), [arXiv 2108.06121](https://arxiv.org/abs/2108.06121v2).
  - Rat cortex and hippocampus in vivo: exchange time 15–60 ms.
- **SMEX.** Olesen, Østergaard, Shemesh & Jespersen. "Diffusion time dependence, power-law scaling, and exchange in gray matter." *NeuroImage* 251:118976, 2022. [arXiv 2108.09983](https://arxiv.org/pdf/2108.09983).
  - The stick power law is not seen in grey matter; exchange (or curvature, somas) is the proposed reason.
  - A later SMEX abstract estimates an exchange time of about 2 ms in rat in vivo ([ISMRM 2022 #1426](https://archive.ismrm.org/2022/1426.html)). The two models' exchange times disagree; unresolved.
- **Exchange in tensor-valued encoding:**
  - ISMRM 2022 #0975 ([abstract](https://archive.ismrm.org/2022/0975.html)): exchange confounds µFA from multidimensional encoding, in either direction depending on the spectral tuning of linear encoding.
  - Nilsson, Chakwizira, Szczepankiewicz, Lasič & Westin, ISMRM 2024 #3470 ([abstract](https://archive.ismrm.org/2024/3470.html)): a framework unifying b-tensor encoding and exchange.
  - Both are abstracts only.
- **Low µFA in GM is expected.**
  - Lampinen et al. 2017 (CODIVIDE, [record](https://lmi.bwh.harvard.edu/lmi/publications/neurite-density-imaging-versus-imaging-microscopic-anisotropy-diffusion-mri-model)): microscopic anisotropy is high in WM, intermediate in thalamus and putamen, low in cortex.
  - A later Lampinen et al. paper ([PMC6503974](https://pmc.ncbi.nlm.nih.gov/articles/PMC6503974)) ties microscopic anisotropy to myelinated axons, not dendrites.

**Implications for us:**
- The fix for the GM prior should come from these models (Kärger-type exchange between neurites and the extracellular space), not from tuning the prior until Cardiff matches.
- Cardiff alone cannot pin an exchange time: there is one diffusion time per encoding, and no waveforms.

## 3. White matter behaves like sticks at high b (Cardiff: stick axons halve the WM miss)

**Known.**

- **McKinnon et al.** "Dependence on b-value of the direction-averaged diffusion-weighted imaging signal in brain." *Magn Reson Imaging*, 2017. The direction-averaged signal follows ≈ b^−1/2 in human WM at high b.
- **Veraart et al.** "On the scaling behavior of water diffusion in human brain white matter." *NeuroImage*, 2019.
- **Veraart et al.** "Noninvasive quantification of axon radii using diffusion MRI." *eLife* 9:e49855, 2020. [eLife](https://elifesciences.org/articles/49855). Small deviations from the 1/2 exponent give axon radii, validated against confocal microscopy in rat corpus callosum.
- **Correction to what I said before:** the eLife paper is 2020; the 2019 paper is the separate NeuroImage one.

**Status.** Our priors v1–v4 had intra-axonal D_Δ 0.9–1.0, a non-zero radial diffusivity. That is inconsistent with this literature. The stick variant (`ax_dd=0.99,1.0`) is the literature-consistent one, not a tuning choice.

## 4. Time dependence biases spherical encoding (H5, r ≈ 1.5)

**Known in principle.**

- **Jespersen, Olesen, Ianuş & Shemesh.** "Effects of nongaussian diffusion on 'isotropic diffusion' measurements: an ex-vivo microimaging and simulation study." *J Magn Reson* 300:84–94, 2019. doi:10.1016/j.jmr.2019.01.007. [arXiv 1712.02290](https://arxiv.org/abs/1712.02290).
  - Anisotropic compartments make time dependence anisotropic, so "isotropic" encodings are not isotropic.
  - This goes back to de Swiet & Mitra 1996.
- **Lasič, Just, Nilsson, Szczepankiewicz et al.** SPAS, rat brain, 2025 ([PMC12319979](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12319979/)). Time-dependent diffusion biases µA unless the b-tensors are spectrally tuned; they map µA and time dependence separately.
- An ISMRM 2019 abstract by Gyori et al. reports non-isotropic spectral content across the axes of spherical waveforms in human brain.

**Status.** That HHS STE is affected by time dependence is expected. Our single-ratio model (r) is a crude version of what SPAS does properly with waveform spectra (task 6, real waveforms).

## 5. Classical QTI and fast constrained fitting (the tools-paper angle)

- **QTI+.** Herberthson, Boito, Haije, Feragen, Westin & Özarslan. "Q-space trajectory imaging with positivity constraints (QTI+)." *NeuroImage* 238:118198, 2021. doi:10.1016/j.neuroimage.2021.118198. [PMC9596133](https://pmc.ncbi.nlm.nih.gov/articles/PMC9596133). Semidefinite constraints; the largest gains are for short protocols.
- **New competitor for a tools paper: RICE.** Yu, Gödicke, Laun, … Kuder. "Constrained estimation of rotational invariants of the cumulant expansion (RICE) for rapid tensor-valued diffusion MRI." [arXiv 2608.27212](https://arxiv.org/abs/2608.27212), August 2026, v3 November 2026.
  - Constrained weighted linear least squares on a 1.5-minute protocol, under 30 s.
  - They describe it as "160 times faster" than a constrained QTI fit (72 min).
  - No learned components.
  - **Consequence for us:** a "fast constrained QTI" paper must position itself against RICE and against the speed of a full SDP QTI+ fit. Ours is the full QTI+ "dc" problem, exact on GPU in seconds, not a reduced invariant set.

## 6. Learned estimators for QTI / b-tensor data specifically

**Sparse. No amortized posterior or normalizing-flow QTI found.**

- **Goedicke, Laun, Martin, Rauch, Neher, Rokuss, Ladd & Kuder.** "Accelerated microstructure quantification by q-space trajectory imaging using machine learning." ISMRM 2024 #3464 ([abstract](https://archive.ismrm.org/2024/3464.html)).
  - A voxelwise regression network trained on a 10-fold reduced protocol in 18 volunteers, against "the chosen ground truth", apparently the full protocol's fit.
  - Reports smoother maps and robustness at (1.7 mm)³.
  - Abstract only. Same group as RICE.
- **de Almeida Martins, Nilsson, Lampinen, Palombo, While, Westin & Szczepankiewicz.** "Neural networks for parameter estimation in microstructural MRI: application to a diffusion-relaxation model of white matter." *NeuroImage* 244:118601, 2021. doi:10.1016/j.neuroimage.2021.118601.
  - Networks on b-tensor and relaxation data. Less affected by sub-optimal protocols than least squares, but still subject to degeneracy, and they don't remove the need for protocol design.
- **Not found in this pass:** simulation-based posterior inference (flow or otherwise) for QTI moments, or protocol-invariant set encoders for b-tensor data. This is possibly a novelty claim for us. It needs a targeted search before relying on it (ISMRM 2025–2026 proceedings, MICCAI CDMRI).

## 7. Test–retest reliability of learned microstructure estimators (our H7 cohort result)

**Not found.**

- Nothing in this pass quantified shrinkage together with ICCs for a learned microstructure estimator.
- Deep CSD work claims reduced scan–rescan variability, but nothing separating reduced within-subject variance from reduced between-subject variance turned up.
- Reported µFA ICCs for comparison (DIVIDE): about 0.74 on the skeleton and 0.81 whole-brain, with pallidum about 0.36. Source: an ISMRM 2021 repeatability abstract ([#2461](https://cds.ismrm.org/protected/21MProceedings/PDFfiles/2461.html)); not checked in full.
- Our result, that the network shrinks within- and between-subject spread together and doesn't raise ICCs, looks unreported. The principle behind it is standard.

## Summary for the project

| Our finding | Outside the project? |
|---|---|
| Learned-prior gains are mostly prior (H5, test–retest pilot, H4 under v2) | **Yes**: Gyori 2022, Epstein 2024 |
| Cardiff GM high-b over-prediction | **Consistent**: NEXI/SMEX exchange; exchange biases µFA (abstracts) |
| Cardiff WM: sticks fit better | **Yes**: McKinnon 2017, Veraart 2019/2020 |
| STE affected by time dependence (H5) | **Yes in principle**: Jespersen 2019, Lasič SPAS 2025 |
| Classical truncation bias; need for constraints | **Yes**: QTI+ 2021 |
| Network doesn't raise cohort ICCs (H7) | Not found; looks new |
| Amortized flow posterior for QTI, protocol-invariant encoder | Not found; possibly new (search more) |
| Fast GPU constrained QTI | **Competitor**: RICE (arXiv 2026), constrained but reduced invariants |

**Actionable:**
1. Adopt sticks in the WM prior as the literature default, not as a tuned option.
2. Take the GM exchange model from NEXI/SMEX instead of tuning the prior to Cardiff.
3. Try Epstein-style maximum-likelihood labels as a less prior-dominated baseline.
4. Run a targeted novelty search (ISMRM 2025–26, CDMRI) before claiming the amortized-posterior QTI.
5. Read the RICE paper before scoping a tools paper.

## Addendum: RICE (iRICE + CWLLS) compared with qtilab

Read from the [arXiv HTML](https://arxiv.org/html/2608.27212), v3. Code: github.com/CollapsarEcho/iRICE-CWLLS (not downloaded). Their imaging data are not public.

| | iRICE + CWLLS (Yu et al. 2026) | qtilab |
|---|---|---|
| Model | 2nd-order cumulant (RICE, Coelho et al. 2026) without the ℓ = 4 covariance terms: **13 coefficients** for LTE+STE (`irice13`: D00, D2m ×5, S00, S2m ×5, A00), 18 with planar encoding. *(An earlier version of this table said 8, from the paper's scalar summary; the code fits 13.)* | Full QTI, **28 parameters** (S0, ⟨D⟩ 6, ℂ 21); rank 23 on LTE+STE (HHS), full rank with PTE (DiB, Cardiff) |
| Protocol | LTE+STE only. 20 volumes, 1.5 min: b0, LTE b = 1 × 10 (dodecahedron), LTE b = 2 × 6 (icosahedron), STE b = 2 × 3 | Any b-tensor table |
| Constraints | Necessary conditions on the 8 coefficients: 0 ⪯ D ⪯ D_max·I (D_max = 3.075), Q₀, T₀ ≥ 0, and a new Bhatia–Davis bound Q₀ + T₀ + ⅕D₂² ≤ D₀(D_max − D₀). Not equivalent to the QTI+ constraints | QTI+ "dc": ⟨D⟩ ⪰ 0, ℂ ⪰ 0 and the dc relaxation (6×6 and 3×3 SDP cones). **No upper diffusivity bound**; allows µFA > 1 and C_c > 1 in a few thousand voxels |
| Solver | WLLS, then CVXPY + MOSEK only on violating voxels (33% in the patient) | Batched ADMM on every voxel, GPU or CPU; matches DIPY SDPdc/CLARABEL |
| Speed | 27 s for about 40k voxels on a 24-core i9. Their constrained QTI reference took 72 min ("160×") | About 200k voxels in 26–32 s on M1 Max MPS (one chunk), 7 s float32 on an A10, about 100 s on the M1 CPU |
| Validation | Simulation (two DTDs, SNR 30) and 14 volunteers, both qualitative, with no numerical agreement vs QTI. One glioma patient (violation counts). One-volunteer resolution/SNR test. No repeatability | Numerical match to DIPY on DiB and HHS; 66-session test–retest cohort; posterior network |
| Learned components | None. The same group (Goedicke, Laun, Kuder) has a separate ML-QTI abstract (ISMRM 2024 #3464) | Posterior network (set encoder + flow) |

**Positioning:**
- RICE's speed claim is against a slow constrained QTI fit. qtilab's GPU fit solves the *full* constrained QTI problem in about the same wall time, so speed alone is no reason to drop to 8 coefficients.
- Their real contribution is the reduced protocol: 20 volumes and 1.5 min. The direction sets are 5-designs, so the discarded ℓ = 4 terms average out of the ℓ = 0 estimates. That only holds for their direction sets.
- **Borrow:** the upper diffusivity bound D_max and their mean-dependent variance bound are physically justified, and qtilab has neither. They may remove our µFA > 1 and C_c > 1 voxels and tame C_MD at CSF interfaces.
- **Open, can be tested here:**
  1. Accuracy against truth on their 20-volume protocol (simulation), RICE vs our constrained QTI vs a network trained on that protocol. Use a generic or published DTD set, not our brain prior, to avoid favouring the network.
  2. iRICE on HHS (LTE+STE, non-5-design directions), where ℓ = 4 aliasing could bias it.
  3. Adding D_max to our constraints.

### RICE, run (2026-11-07)

Their package was cloned to `work/ext/iRICE-CWLLS` (commit 8cf9c90, not committed here) and run directly, with CLARABEL standing in for MOSEK. All 12 of their tests pass. The comparison is in `scripts/rice_compare.py` → `results/rice_compare.json`; the results are in NOTES (2026-11-07, RICE entry).

**Definition correction:** iRICE's `uFA` is Lasič's µFA, which leaves the isotropic variance out of the denominator. It is not QTI's (Westin's) µFA. On real HHS data it reads 0.008 (WM) to 0.024 (GM) higher for the same tissue.

## Addendum 2: prior art for a batched GPU solver of constrained QTI (2026-11-07)

**Question:** has anyone published a batched or GPU solver for QTI+ (or a similar per-voxel SDP in diffusion MRI)? This was asked after `qtilab/ipm.py` (NOTES 2026-11-07, IPM entry).

**Method and limits:** web searches only, about eight queries. ISMRM abstracts are partly behind a login, and I read titles and summaries, not full texts. Absence here is weak evidence.

**No GPU or batched solver for constrained QTI found.** The published routes to speed are:
- **QTI+ software** (Boito et al., SoftwareX 2022; MATLAB): per-voxel SDPs through a CPU solver, sending 50 voxels at a time, run in parallel.
  - The RICE paper (Yu et al., arXiv 2608.27212) calls full-cumulant SDP fitting "several tens of minutes per dataset". It reports its own CWLLS at under 30 s, 160× faster than QTI±.
  - Our measurement: RICE CWLLS 51 s on HHS with 8 workers, CLARABEL instead of MOSEK.
- **DIPY `SDPdc`** (CVXPY, per voxel). The DIPY tutorial reports about 15 min with SCS and 2 min with MOSEK for one slice. Our measurement on HHS: 12 ms per voxel with CLARABEL, 31 ms with SCS.
- **Learned approximations of the constrained fit:**
  - Boito & Özarslan, ISMRM 2023 #3780;
  - Gödicke et al., ISMRM 2024 #3464 (an MLP trained on constrained fits; more than an hour reduced to seconds);
  - Yu et al., ISMRM 2025 (unrolled network).
  - These are fast but approximate, trained against the constrained fit rather than solving it.
- **General GPU qMRI fitting:** GACELLE (Chan et al., arXiv 2511.22094; MATLAB) uses stochastic gradient descent and MCMC. Earlier GPU work covered NODDI, CHARMED and similar models. No positivity-constrained QTI was mentioned.

**The technique itself is known outside dMRI:**
- **qpth / OptNet** (Amos & Kolter, ICML 2017): a batched dense primal-dual interior-point solver on the GPU for many small QPs, reported more than 100× faster than Gurobi on batches. It is the same idea as ours, for QPs rather than SDPs.
- **CuClarabel** (arXiv 2412.19027; Julia, callable from CVXPY): a GPU version of CLARABEL with batched cuSOLVER factorizations for many small PSD cones. It is benchmarked on FEM problems with 3 × 3 PSD constraints.
  - Restrictions: every PSD cone must have the same dimension, at most 32. QTI+ has a 3 × 3 and a 6 × 6 cone per voxel, so it would need a reformulation, for example embedding ⟨D⟩ in a 6 × 6 block with equality constraints.
  - In principle it could solve a whole brain's QTI+ as one block-separable conic problem. It is the closest general-purpose competitor.

**Assessment:**
- A batched GPU interior-point solver is a known technique (OptNet for QPs, CuClarabel for small PSD cones).
- I found no application of it to constrained QTI or other per-voxel dMRI SDPs. The published fast alternatives either change the model (RICE) or approximate the fit with a network.
- So the defensible claim is engineering, not a new algorithm: the exact QTI+ (dc) optimum with every voxel converged and strictly feasible output, validated against DIPY/CLARABEL, in about 1–12 s per session on CPU, MPS or CUDA. That is about 100× DIPY + CLARABEL on CPU (extrapolated from 3,000 voxels).
- **Oracle-based next comparisons, if wanted:**
  1. CuClarabel on the same HHS problem: a direct GPU interior-point competitor, the strongest test of whether a custom solver is needed.
  2. The QTI+ MATLAB package with MOSEK: the reference implementation's own speed.
  - Both need new installs (Julia, MATLAB and MOSEK licences).

Sources: [QTI+ (Herberthson 2021)](https://www.sciencedirect.com/science/article/pii/S1053811921004754); [QTI+ implementation (SoftwareX)](https://github.com/ElsevierSoftwareX/SOFTX-D-21-00175); [RICE](https://arxiv.org/pdf/2608.27212); [Diffusivity-limited QTI](https://www.sciencedirect.com/science/article/pii/S2772516223000013); [DIPY QTI+ tutorial](https://docs.dipy.org/dev/examples_built/reconstruction/reconst_qtiplus.html); [Boito & Özarslan 2023](https://archive.ismrm.org/2023/3780.html); [Gödicke 2024](https://archive.ismrm.org/2024/3464.html); [Yu 2025](https://archive.ismrm.org/2025/1283.html); [GACELLE](https://arxiv.org/abs/2511.22094); [OptNet](https://arxiv.org/abs/1703.00443); [qpth](https://locuslab.github.io/qpth); [CuClarabel](https://arxiv.org/abs/2412.19027).
