#!/usr/bin/env bash
#SBATCH -J mimic_src_fssum
#SBATCH -p commons
#SBATCH -c 6
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o logs/mimic_source_benchmark_100k_v1/final_sparse_summary_%j.out
#SBATCH -e logs/mimic_source_benchmark_100k_v1/final_sparse_summary_%j.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/summarize_final_layer_feature_yield.py \
  --dictionary-root results/mimic_source_benchmark_100k_v1/dictionary/workers \
  --output-root results/mimic_source_benchmark_100k_v1/final_sparse/feature_yield
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/summarize_final_layer_sparse_accessibility.py \
  --workers-root results/mimic_source_benchmark_100k_v1/final_sparse/workers \
  --bootstrap-root results/mimic_source_benchmark_100k_v1/final_sparse/bootstrap \
  --dense-ceiling-root results/mimic_source_benchmark_100k_v1/accessibility/workers \
  --feature-yield-root results/mimic_source_benchmark_100k_v1/final_sparse/feature_yield \
  --output-root results/mimic_source_benchmark_100k_v1/final_sparse/summary
