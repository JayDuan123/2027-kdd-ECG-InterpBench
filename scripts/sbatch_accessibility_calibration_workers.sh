#!/bin/bash
#SBATCH -J access_e8
#SBATCH -p commons
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH --array=0-89%12
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/accessibility_calibration_%A_%a.out
#SBATCH -e logs/accessibility_calibration_%A_%a.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python \
  scripts/run_accessibility_calibration_worker.py \
  --calibration-index "${SLURM_ARRAY_TASK_ID}" \
  --expansion 8 \
  --batch-size 256 \
  --semantic-train-limit 4096 \
  --ridge-alpha 10 \
  --device cuda
