#!/bin/bash
#SBATCH -J dict_access_e8
#SBATCH -p commons
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH --array=0-29%6
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/dictionary_accessibility_e8_%A_%a.out
#SBATCH -e logs/dictionary_accessibility_e8_%A_%a.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"

/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python \
  scripts/run_dictionary_accessibility_worker.py \
  --group-index "${SLURM_ARRAY_TASK_ID}" \
  --expansion 8 \
  --batch-size 256 \
  --semantic-train-limit 4096 \
  --random-replicates 20 \
  --random-seed-base 930000 \
  --matched-budget 768 \
  --budget-replicates 20 \
  --budget-seed-base 940000 \
  --device cuda
