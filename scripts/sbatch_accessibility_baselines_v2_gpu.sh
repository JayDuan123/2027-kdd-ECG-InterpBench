#!/bin/bash
#SBATCH -J access_v2_gpu
#SBATCH -p commons
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH --array=0-29%6
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/accessibility_baselines_v2_gpu_%A_%a.out
#SBATCH -e logs/accessibility_baselines_v2_gpu_%A_%a.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python \
  scripts/run_accessibility_baselines_v2_worker.py \
  --baseline-index "${SLURM_ARRAY_TASK_ID}" \
  --expansion 8 \
  --batch-size 256 \
  --semantic-train-limit 4096 \
  --random-replicates 20 \
  --random-seed-base 920000 \
  --device cuda
