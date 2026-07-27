# ECG-FM-InterpBench Paper Outline

## Working title

**ECG-FM-InterpBench: Auditing ECG Foundation Models with Matched-Scale Sparse Autoencoders**

The paper evaluates ECG foundation models. Sparse autoencoders are a shared
measurement instrument, not the method being ranked.

## Central question

> When SAE capacity, sparsity, training, data, depth, and inference are matched,
> how faithfully, semantically, and reproducibly can different ECG foundation
> model representations be decomposed?

"Multi-scale" means that this matched comparison is repeated at
`E=N/d_FM in {1,4,8,16,32}`. It never means choosing a different favorable SAE
scale for each FM.

All six source layer atlases have `d=768`, so matching relative expansion also
matches absolute dictionary width exactly: `N in {768,3072,6144,12288,24576}`.
The primary arm uses `k=96` for every FM and scale. CARDIAC-FM's released
768-to-512 pooled projection belongs to the downstream stage, not this atlas.

## Claim hierarchy

1. **Primary:** each FM has a depth-by-scale interpretability profile measured
   on the complete common grid.
2. **Primary:** FM differences and rankings are metric-specific and may change
   with common SAE scale or encoder depth.
3. **Primary:** common-scale AUC summarizes the same five `E` values for every
   FM; it is not a best-cell score.
4. **Robustness:** patient-cluster bootstrap, SAE seeds, stability matching, and
   fixed-`k/N` sensitivity quantify distinct uncertainty sources.
5. **Secondary:** cross-cohort transport, controlled intervention, and waveform
   grounding test whether selected decompositions remain useful downstream.
6. **Not claimed:** a single universal "most interpretable model," biological
   mechanism, clinical utility, or SAE superiority on every endpoint.

## Abstract structure

1. ECG FM comparison is currently dominated by predictive performance; their
   internal representational accessibility is not benchmarked under matched
   measurement capacity.
2. Introduce a six-model, five-depth, five-scale, three-seed SAE atlas on
   PTB-XL: 450 preregistered cells and 49 train-selected clinical concepts.
3. State the fairness rule: identical relative expansion at every FM
   comparison, fixed `k/d` primary arm, fixed `k/N` sensitivity arm, no
   per-model scale selection.
4. Report only audited final numbers for reconstruction, semantic alignment,
   concept coverage, feature stability, patient-level uncertainty, and scale
   rank stability.
5. Summarize secondary external-cohort and intervention evidence as capability
   validation rather than the paper's organizing result.
6. End with the contribution: an auditable protocol for comparing foundation
   model representations with a shared sparse measurement family.

## 1. Introduction

### Motivation

- Predictive AUROC does not answer whether a frozen representation admits a
  faithful, sparse, clinically aligned decomposition.
- A one-scale SAE comparison confounds model quality with an arbitrary
  dictionary capacity.
- Giving each FM its best SAE scale creates an unfair and test-leaking ranking.

### Benchmark question

- Treat the frozen FM representation as the experimental object.
- Treat the SAE family as a calibrated probe scanned over common capacity.
- Compare full depth-scale response surfaces, not isolated examples.

### Contributions

1. First matched multi-scale SAE atlas over six heterogeneous ECG FMs.
2. Leakage-controlled clinical alignment with train-only atom selection and
   frozen validation/test evaluation.
3. Metric-specific common-scale profiles with patient-cluster and design-level
   uncertainty, rank-stability analysis, and sparsity-parameterization checks.
4. A staged extension from source representation audit to transport and
   intervention tests, with explicit eligibility gates.

## 2. Related work

### ECG foundation models

- Position the six FMs by pretraining objective and released input interface.
- Do not infer interpretability from architecture family alone.

### Sparse representation analysis

- SAE reconstruction, sparsity, semantic alignment, and feature stability.
- Explain why scale sweeps are calibration, not SAE hyperparameter search.

### Mechanistic interpretability benchmarks

- Contrast single-model SAE studies with cross-model matched measurement.
- Borrow the EEG-SAE paper's layerwise/scaling motivation while adding strict
  cross-model scale matching, patient resampling, negative controls, and
  external transport.

## 3. ECG-FM-InterpBench

### 3.1 Frozen models and source cohort

- Models: CARDIAC-FM, CSFM, ECG-FM, ECG-JEPA, HuBERT-ECG, and ST-MEM.
- PTB-XL patient-level train/validation/test split.
- Preserve each released preprocessing and representation API.

### 3.2 Standardized depth axis

- Map each encoder to nearest layers at relative depths
  `{0,0.25,0.5,0.75,1}`.
- Record both target and realized relative depth.
- Never imply that layer numbers are architecturally identical.

### 3.3 Matched multi-scale SAE instrument

- BatchTopK SAE per `(model, depth, E, seed)`.
- Common relative expansion `E=N/d_FM in {1,4,8,16,32}`; because every source
  layer atlas has `d=768`, absolute `N` is matched as well.
- Primary active budget `k/d_FM=1/8`.
- Same 8,000 steps, batch size 256, optimizer, splits, and seeds 4311--4313.
- Exactly 75 matched blocks, each containing all six FMs once; 450 cells total.

### 3.4 Clinical concept protocol

- Forty-nine waveform and measurement concepts.
- Standardization uses train statistics only.
- For each concept, select one SAE feature on train only.
- Freeze that feature before validation/test correlation and coverage.
- Missing values use the frozen train-mean imputation rule and are disclosed.

### 3.5 Evaluation axes

- **Measurement fidelity:** normalized reconstruction `R^2`, dead fraction,
  realized L0, and fidelity pass rate.
- **Semantic accessibility:** mean absolute train-selected correlation and
  concept coverage at `|r|>=0.20`.
- **Reproducibility:** matched decoder cosine above random and top-feature
  subspace overlap across SAE seeds.
- **Scale efficiency:** complete common-`E` curve and normalized log-scale AUC.
- **Layer localization:** depth-scale surface without selecting the best test
  layer.
- Keep the profile multidimensional; do not create an arbitrary composite
  "interpretability score."

### 3.6 Statistical inference

- Patient-cluster bootstrap over all 450 test cells, preserving every record
  from each sampled patient and pairing draws across all FMs.
- Average the three frozen SAE seeds for patient-sampling intervals.
- Separately bootstrap crossed depth/seed units for design variation.
- Paired FM contrasts only at the same `E`, or after integrating the same five
  `E` values for both FMs.
- BH-FDR within each metric family.

### 3.7 Robustness and staged capability tests

- Fixed `k/N=1/64` mid-depth sensitivity over the same six models, five scales,
  and three seeds.
- Validation-only operating-point selection for downstream experiments.
- External transport, cohort adaptation, intervention controls, and waveform
  grounding remain secondary analyses.

## 4. Experiments

### 4.1 Completeness and invariant audit

- Check 450/450 artifacts, hashes, shapes, finite metrics, concept rows, and
  exactly 75 complete six-model blocks.

### 4.2 Full depth-scale atlas

- Present reconstruction, semantic alignment, coverage, and dead-feature maps.
- Describe patterns before assigning model ranks.

### 4.3 Matched-scale FM comparison

- Plot all six FMs at each common `E` with paired patient confidence bands.
- Report pairwise differences and FDR at common-scale AUC.
- Report rank probabilities and Kendall agreement across `E` values.

### 4.4 Feature reproducibility

- Match active decoder directions across seeds.
- Compare matched cosine with random pairing and report subspace overlap.
- Test whether apparent semantic accessibility persists when atom identity is
  unstable.

### 4.5 Sparsity-parameterization sensitivity

- Compare fixed `k/d` with fixed `k/N` at relative depth 0.5.
- Report rank agreement and paired profile changes.
- Use this analysis to qualify robustness, not to select a favorable arm.

### 4.6 Downstream capability validation

- Apply validation-selected operating points to source intervention and
  external transport.
- Preserve fidelity, transport, readout-retention, random-direction, and
  off-target gates.
- Frame these results as consequences of the FM profile, not proof that SAE is
  universally superior to other explanation methods.

## 5. Results order

1. **The matched grid is complete and auditable.**
2. **ECG FMs have distinct depth-scale decomposition fingerprints.**
3. **FM ordering is evaluated at common scales and may be scale-sensitive.**
4. **Common-scale AUC gives a capacity-robust, metric-specific comparison.**
5. **Semantic accessibility must be interpreted jointly with fidelity and
   feature stability.**
6. **The fixed-`k/N` arm establishes which conclusions survive a different
   sparsity parameterization.**
7. **External transport and intervention provide secondary capability tests.**

Traditional methods remain secondary controls or appendix analyses. The
primary estimand compares ECG FMs under the shared SAE measurement instrument.

## 6. Discussion

- Interpret the depth-scale fingerprint rather than architecture labels alone.
- Distinguish sparse reconstructability, clinical semantic accessibility, and
  reproducibility; none is a synonym for the others.
- Explain when common-scale AUC is useful and when the full surface is needed.
- Discuss benchmark implications for choosing an ECG FM for interpretable
  downstream work without declaring a universal winner.

## 7. Limitations and claim boundaries

- SAE-based accessibility is one operationalization of interpretability.
- The source concepts are measurement-derived and not an exhaustive clinical
  ontology.
- BatchTopK codes depend on evaluation batching, which remains frozen.
- Patient bootstrap is conditional on frozen FM/SAE weights and concept
  selection; design variation is reported separately.
- Relative depth and relative width improve fairness but do not make encoder
  architectures isomorphic.
- Latent intervention does not establish waveform or biological causality.

## Main figure plan

1. **Benchmark design:** six FMs through a common depth-scale SAE grid; one
   matched block highlighted.
2. **Depth-scale atlas:** six-model reconstruction and semantic heatmaps.
3. **Matched-scale curves:** patient-bootstrap bands at the same five `E`
   values.
4. **Common-scale profiles:** metric-specific AUC forest plots and paired FM
   contrasts.
5. **Scale and seed robustness:** rank agreement plus feature stability above
   random.
6. **Sparsity sensitivity:** fixed `k/d` versus fixed `k/N` profile comparison.
7. **Secondary capability tests:** transport/intervention funnel with fixed
   denominators.

## Main table plan

1. Frozen model interfaces and realized layer mapping.
2. Frozen matched-scale protocol and fairness invariants.
3. Common-scale model profiles with patient 95% intervals.
4. Pairwise common-scale differences with BH-adjusted q-values.
5. Fixed-`k/N` rank agreement and secondary transport summary.

## Result insertion gate

No number enters `main.tex` until all corresponding audit JSON files report
complete status. Every reported profile must be traceable to a generated CSV,
figure, or LaTeX table. Any failed model or cell remains in the fixed
denominator and is reported as a benchmark result rather than silently removed.
