#!/bin/bash
#SBATCH -J mssae_smoke
#SBATCH -p scavenge
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH --array=0-5%6
#SBATCH --requeue
#SBATCH --signal=B:USR1@90
#SBATCH -o logs/multiscale_sae_v1/smoke_%A_%a.out
#SBATCH -e logs/multiscale_sae_v1/smoke_%A_%a.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
PYTHON=/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python

"${PYTHON}" scripts/run_multiscale_sae_task.py \
  --manifest results/multiscale_sae_v1_smoke/training_manifest.csv \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --device cuda \
  --checkpoint-every 50 \
  --semantic-train-limit 512
