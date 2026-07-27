#!/bin/bash
#SBATCH -J mssae_fixed_kn
#SBATCH -p scavenge
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH -t 00:59:00
#SBATCH --array=0-89%24
#SBATCH --requeue
#SBATCH --signal=B:USR1@90
#SBATCH -o logs/multiscale_sae_fixed_k_over_n_middepth_v1/train_%A_%a.out
#SBATCH -e logs/multiscale_sae_fixed_k_over_n_middepth_v1/train_%A_%a.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python \
  scripts/run_multiscale_sae_task.py \
  --manifest results/multiscale_sae_fixed_k_over_n_middepth_v1/training_manifest.csv \
  --task-index "$SLURM_ARRAY_TASK_ID" \
  --device cuda \
  --checkpoint-every 100 \
  --semantic-train-limit 4096
