#!/usr/bin/env bash
#SBATCH -J mlive_smoke
#SBATCH -p scavenge
#SBATCH --qos=nots_scavenge
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o logs/mimic_final_layer_live_atom_matched_effect_100k_v1/smoke_%j.out
#SBATCH -e logs/mimic_final_layer_live_atom_matched_effect_100k_v1/smoke_%j.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python \
  scripts/run_mimic_final_layer_matched_effect_worker.py \
  --task-index 12 --device cuda --max-records-per-split 2048 \
  --protocol mimic_final_layer_live_atom_matched_effect_100k_v1 \
  --candidate-pool train_live --quality-gate-mode matched_live_capacity \
  --manifest results/mimic_final_layer_matched_effect_100k_v1/training_manifest.csv \
  --concepts results/mimic_final_layer_matched_effect_100k_v1/derived/concepts_standardized.csv \
  --readout-root results/mimic_final_layer_matched_effect_100k_v1/readouts \
  --output-root results/mimic_final_layer_live_atom_matched_effect_100k_v1/smoke/workers
