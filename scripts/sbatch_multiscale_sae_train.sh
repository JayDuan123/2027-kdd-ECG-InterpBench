#!/bin/bash
#SBATCH -J mssae_train
#SBATCH -p scavenge
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH -t 00:59:00
#SBATCH --array=0-449%24
#SBATCH --requeue
#SBATCH --signal=B:USR1@90
#SBATCH -o logs/multiscale_sae_v1/train_%A_%a.out
#SBATCH -e logs/multiscale_sae_v1/train_%A_%a.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
PYTHON=/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python

"${PYTHON}" scripts/run_multiscale_sae_task.py \
  --manifest results/multiscale_sae_v1/training_manifest.csv \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --device cuda \
  --checkpoint-every 100 \
  --semantic-train-limit 4096
