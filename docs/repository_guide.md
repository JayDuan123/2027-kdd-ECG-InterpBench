# Repository and Artifact Guide

This guide maps the manuscript experiments to the implementation entry points.
The paper uses frozen encoder representations and applies the same SAE protocol
at common relative depths and expansion factors.

## Primary PTB-XL Benchmark

| Stage | Main implementation |
|---|---|
| Frozen configuration and model gates | `benchmark_v1/config.py`, `configs/` |
| Model adapters | `benchmark_v1/adapters/` |
| Manifest construction | `scripts/build_multiscale_sae_manifest.py` |
| BatchTopK SAE training | `scripts/run_multiscale_sae_task.py` |
| Cell-level audit | `scripts/audit_multiscale_sae.py` |
| Release audit | `scripts/audit_multiscale_sae_release.py` |
| Inference and operating points | `scripts/analyze_multiscale_sae_inference.py`, `scripts/select_multiscale_sae_operating_points.py` |
| Cross-seed stability | `scripts/analyze_multiscale_sae_stability.py` |
| Patient bootstrap | `scripts/run_multiscale_sae_patient_bootstrap.py`, `scripts/summarize_multiscale_sae_patient_bootstrap.py` |
| Fixed-k/N sensitivity | `scripts/analyze_multiscale_sae_sparsity_sensitivity.py` |
| Figures and paper facts | `scripts/make_multiscale_sae_figures.py`, `scripts/make_multiscale_sae_paper_facts.py` |

The primary grid contains six models, five relative-depth positions, five
expansion factors, and three seeds. Concept-feature selection is performed on
the training split and frozen before held-out evaluation.

## Calibration and Controls

| Experiment | Main implementation |
|---|---|
| Accessibility calibration | `benchmark_v1/accessibility_calibration.py`, `scripts/run_accessibility_calibration_worker.py` |
| Dictionary accessibility | `benchmark_v1/dictionary_accessibility.py`, `scripts/run_dictionary_accessibility_worker.py` |
| Dense comparison | `benchmark_v1/five_scale_dense_comparison.py`, `scripts/summarize_five_scale_dense_comparison.py` |
| PCA comparison | `scripts/run_pca768_accessibility_worker.py`, `scripts/summarize_five_scale_pca_comparison.py` |
| Final-layer matched effect | `benchmark_v1/matched_effect.py`, `scripts/run_final_layer_matched_effect_worker.py` |
| Sparse accessibility | `benchmark_v1/sparse_accessibility.py`, `scripts/run_final_layer_sparse_accessibility_worker.py` |
| Input harmonization | `benchmark_v1/input_harmonization.py`, `scripts/extract_harmonized_ptbxl_activations.py`, `scripts/evaluate_input_harmonization.py` |

## MIMIC-IV-ECG Replication

| Stage | Main implementation |
|---|---|
| Cohort and split contract | `benchmark_v1/mimic_source_benchmark.py` |
| Activation planning | `scripts/build_mimic_source_activation_plan.py` |
| Atlas materialization | `scripts/materialize_mimic_source_atlas.py` |
| SAE manifest | `scripts/build_mimic_source_sae_manifest.py` |
| Release audit | `scripts/audit_mimic_source_release.py` |
| Final-layer matched effect | `benchmark_v1/mimic_matched_effect.py`, `scripts/run_mimic_final_layer_matched_effect_worker.py` |

MIMIC-IV-ECG remains a credentialed dataset. The repository contains only code
and aggregate manuscript artifacts.

## Manuscript Artifacts

| Manuscript artifact | Source |
|---|---|
| Multiscale semantic atlas | `scripts/make_multiscale_sae_figures.py` |
| Matched-scale curves | `scripts/make_multiscale_sae_figures.py` |
| Common-scale AUC profiles | `scripts/make_multiscale_sae_figures.py` |
| Stability and sparsity-sensitivity panels | `scripts/make_multiscale_sae_figures.py` |
| Generated macros and tables | `scripts/make_multiscale_sae_paper_facts.py`, `scripts/sync_multiscale_paper_artifacts.py` |
| Input-harmonization appendix | `scripts/evaluate_input_harmonization.py` |
| Final-layer matched-effect appendix | `scripts/summarize_final_layer_matched_effect.py` |

`paper/figures/` and `paper/generated/` contain the aggregate files used by the
current manuscript. Large benchmark result directories are deliberately
excluded from Git.

## Slurm Wrappers

The `scripts/sbatch_*.sh` files document the exact scheduler topology used for
the reported runs. They are cluster-specific snapshots. Before reuse:

1. Replace NOTS allocation, partition, module, and virtual-environment settings.
2. Point all data and model paths to licensed local copies.
3. Run the corresponding smoke test and audit.
4. Submit the full dependency chain only after the smoke audit passes.
