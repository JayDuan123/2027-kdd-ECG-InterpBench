#!/bin/bash
#SBATCH -J mme_smoke_sae
#SBATCH -p scavenge
#SBATCH --qos=nots_scavenge
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH --requeue
#SBATCH -o logs/mimic_final_layer_matched_effect_v1/smoke_train_%j.out
#SBATCH -e logs/mimic_final_layer_matched_effect_v1/smoke_train_%j.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python \
  scripts/train_mimic_final_layer_sae.py \
  --task-index 0 \
  --manifest results/mimic_final_layer_matched_effect_v1/smoke_training_manifest.csv \
  --device cuda \
  --checkpoint-every 10
