#!/usr/bin/env bash
#SBATCH -J mme100k_train
#SBATCH -p commons
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH --array=0-17%12
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/mimic_final_layer_matched_effect_100k_v1/train_%A_%a.out
#SBATCH -e logs/mimic_final_layer_matched_effect_100k_v1/train_%A_%a.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python \
  scripts/train_mimic_final_layer_sae.py \
  --task-index "${SLURM_ARRAY_TASK_ID}" --device cuda --checkpoint-every 100 \
  --protocol mimic_final_layer_matched_effect_100k_v1 \
  --manifest results/mimic_final_layer_matched_effect_100k_v1/training_manifest.csv
