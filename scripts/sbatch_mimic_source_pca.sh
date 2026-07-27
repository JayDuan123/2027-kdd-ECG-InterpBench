#!/usr/bin/env bash
#SBATCH -J mimic_src_pca
#SBATCH -p commons
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH --array=0-29%10
#SBATCH -o logs/mimic_source_benchmark_100k_v1/pca_%A_%a.out
#SBATCH -e logs/mimic_source_benchmark_100k_v1/pca_%A_%a.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/run_pca768_accessibility_worker.py \
  --manifest results/mimic_source_benchmark_100k_v1/training_manifest.csv \
  --group-index "${SLURM_ARRAY_TASK_ID}" \
  --concepts results/mimic_source_benchmark_100k_v1/derived/concepts_matrix.csv \
  --concept-registry results/mimic_source_benchmark_100k_v1/derived/concept_registry.csv \
  --expected-concepts 7 --complete-case-concepts \
  --output-root results/mimic_source_benchmark_100k_v1/pca/workers
