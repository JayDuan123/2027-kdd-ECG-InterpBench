#!/usr/bin/env bash
#SBATCH -J mimic_src_fsboot
#SBATCH -p commons
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH --array=0-5%6
#SBATCH -o logs/mimic_source_benchmark_100k_v1/final_sparse_boot_%A_%a.out
#SBATCH -e logs/mimic_source_benchmark_100k_v1/final_sparse_boot_%A_%a.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/bootstrap_final_layer_sparse_accessibility.py \
  --model-index "${SLURM_ARRAY_TASK_ID}" --bootstrap-draws 2000 \
  --workers-root results/mimic_source_benchmark_100k_v1/final_sparse/workers \
  --output-root results/mimic_source_benchmark_100k_v1/final_sparse/bootstrap
