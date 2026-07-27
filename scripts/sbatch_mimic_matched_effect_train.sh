#!/bin/bash
#SBATCH -J mme_train
#SBATCH -p scavenge
#SBATCH --qos=nots_scavenge
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH -t 00:59:00
#SBATCH --array=0-17%12
#SBATCH --requeue
#SBATCH --signal=B:USR1@90
#SBATCH --open-mode=append
#SBATCH -o logs/mimic_final_layer_matched_effect_v1/train_%A_%a.out
#SBATCH -e logs/mimic_final_layer_matched_effect_v1/train_%A_%a.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python \
  scripts/train_mimic_final_layer_sae.py \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --device cuda \
  --checkpoint-every 100
