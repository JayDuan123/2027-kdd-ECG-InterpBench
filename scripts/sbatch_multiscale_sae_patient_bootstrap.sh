#!/bin/bash
#SBATCH -J mssae_pboot
#SBATCH -p scavenge
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:59:00
#SBATCH --array=0-449%24
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/multiscale_sae_v1/pboot_%A_%a.out
#SBATCH -e logs/multiscale_sae_v1/pboot_%A_%a.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python \
  scripts/run_multiscale_sae_patient_bootstrap.py \
  --task-index "$SLURM_ARRAY_TASK_ID" \
  --bootstrap-samples 2000 \
  --bootstrap-chunk 100 \
  --device cuda
