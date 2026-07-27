# Causally Anchored SAE Extension

This is an experimental post-LEACE module, not part of the v1 main causal
benchmark. The main v1 claims remain based on Probe, LEACE erasure, residual
probe validation, closure, and coupling-aware family interpretation.

## Question

For LEACE-confirmed causal ECG measurement subspaces, can the same concept be
recovered as sparse, decomposable, and selectively steerable SAE features?

The SAE layer does not discover causal concepts. It starts from cells already
confirmed by LEACE and tests whether their causal subspace is sparse and
steerable in a learned dictionary.

## Placement

```text
Probe -> LEACE causal atlas -> coupling audit -> optional SAE extension -> Closure
```

## Current Status

The repository contains a scaffold under:

```text
benchmark_v1/sae_extension/
```

`BenchmarkSAEEnvironment` wires the non-forward pieces to existing benchmark
artifacts: raw pooled features, exported LEACE/CAV directions, measurement
matrices, and task labels. `StubEnvironment` remains available as a fail-loud
template.

The remaining blocker is the forward patcher:

- the exact continuation patcher used by LEACE for full steering runs

For CSFM, `CSFMSAEEnvironment` provides an initial real continuation bridge for
smoke testing. It applies pooled SAE deltas as a mean shift to every token in the
same record, then continues the frozen CSFM model. Identity patches reproduce
the clean continuation path; non-identity results must be described as
pooled-mean interventions.

Do not run the steering module on placeholder subspaces, synthetic activations,
or surrogate forward metrics.

CSFM identity smoke:

```bash
python scripts/smoke_csfm_sae_environment.py --max-test-shards 1
```

Minimal CSFM SAE pilot invocation, after moving to a compute/GPU node:

```bash
python -m benchmark_v1.sae_extension.run_sae_layer \
  --environment csfm \
  --cells results/analysis/model_comparison/leace_coupling_top_confirmed.csv \
  --coupling results/analysis/model_comparison/leace_coupling_risk_summary.csv \
  --artifacts results/sae_artifacts \
  --out results/sae_extension/csfm_pilot_smoke \
  --pilot \
  --limit-cells 1 \
  --max-test-shards 1 \
  --E-grid 4 \
  --k0-grid 16 \
  --recon-r2-floor -1.0 \
  --min-task-retention 0.0 \
  --steps 500 \
  --f-steps 3 \
  --n-random 2 \
  --n-features 32 \
  --n90-max-features 128
```

Equivalent Slurm smoke script:

```bash
sbatch scripts/sbatch_csfm_sae_pilot_smoke.sh
```

## Artifact Export

The LEACE benchmark JSON files store residual-probe and effect summaries, but
the SAE layer also needs the actual LEACE subspace and dense-probe CAV. Export
them from existing pooled probe features with:

```bash
python scripts/export_sae_artifacts.py --pilot-only
```

This writes:

```text
results/sae_artifacts/manifest.csv
results/sae_artifacts/<model_suffix>/<concept>__<task>__LXX/
  leace_u_sae_norm.npy
  leace_u_raw.npy
  leace_u_whitened.npy
  cav_sae_norm.npy
  cav_standardized.npy
  cav_raw_covector.npy
  sae_mu.npy
  sae_sigma.npy
  metadata.json
```

Use `leace_u_sae_norm.npy` for `A_geo` and `cav_sae_norm.npy` for SAE feature
ranking, because the SAE dictionary lives in normalised activation space.

## Metrics

- `A_geo`: geometric agreement between selected SAE decoder directions and the
  LEACE causal subspace. Report the CAV-ranked value as the primary geometric
  overlap. Concept-activation-ranked `A_geo` is diagnostic only because its
  feature selection is partly circular with the measured concept.
- `decomposability`: concept ground-truth variance recovered from selected SAE
  features. This is computed after train-split target standardisation. Use
  activation-ranked features for decomposability and `n90`.
- `n90`: number of ranked features needed to reach 90% of full decomposability.
- `delta_tilde`: clamp selectivity beyond random feature selection, computed on
  damage/drop curves rather than raw AUROC/R2 scales.
- `WBI`: off-target damage divided by target effect. `WBI > 1` means clamping
  the selected feature set damages off-target readouts more than it changes the
  target, so the intervention is not selectively steerable.
- `kappa -> WBI`: optional full-set test of whether LEACE connected damage
  predicts steering damage.

## CSFM Fixed Pilot Interpretation

The fixed CSFM SAE pilot is summarised in:

```text
results/sae_extension/csfm_sae_fixed_interpretation_summary.md
results/sae_extension/csfm_sae_fixed_interpretation_summary.csv
```

The current reading is:

- LEACE causal subspaces are partially approximated by SAE dictionary directions
  (`A_geo_cav` 0.56-0.73).
- Clinical measurements are linearly decodable from SAE activations after the
  scale fix (activation-ranked decomposability 0.26-0.62).
- The concept codes are distributed rather than single-feature monosemantic:
  `n90` is 25-73 features.
- The same feature sets are not selectively steerable: `WBI > 1` in all tested
  CSFM cells, so off-target readout damage exceeds target effect.
- `qrst_angle -> ptbxl_cd` should be described as a broad distributed electrical
  axis signal: v1 causal relevance is broad, while SAE `A_geo_cav` is lower and
  `n90` is high.

## Scope Discipline

SAE results must be reported as an extension or appendix unless the full module
has been run, audited, and validated. They should not be used to define v1 causal
claims.
