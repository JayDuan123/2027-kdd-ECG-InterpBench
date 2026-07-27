#!/bin/bash
#SBATCH -J dict_access_sum
#SBATCH -p commons
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -o logs/dictionary_accessibility_summary_%j.out
#SBATCH -e logs/dictionary_accessibility_summary_%j.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python \
  scripts/summarize_dictionary_accessibility.py \
  --expected-groups 30 \
  --expected-feature-rows 208 \
  --expected-target-rows 6032
