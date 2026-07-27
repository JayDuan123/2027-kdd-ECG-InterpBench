#!/bin/bash
#SBATCH -J dict_access_e8_cpu
#SBATCH -p scavenge
#SBATCH --qos=nots_scavenge
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:59:00
#SBATCH --array=0-29%10
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/dictionary_accessibility_e8_cpu_%A_%a.out
#SBATCH -e logs/dictionary_accessibility_e8_cpu_%A_%a.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

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
  --device cpu
