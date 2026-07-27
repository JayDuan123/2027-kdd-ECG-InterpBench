#!/usr/bin/env bash
#SBATCH -J mme100k_intervene
#SBATCH -p commons
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH --array=0-17%12
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/mimic_final_layer_matched_effect_100k_v1/worker_%A_%a.out
#SBATCH -e logs/mimic_final_layer_matched_effect_100k_v1/worker_%A_%a.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python \
  scripts/run_mimic_final_layer_matched_effect_worker.py \
  --task-index "${SLURM_ARRAY_TASK_ID}" --device cuda \
  --protocol mimic_final_layer_matched_effect_100k_v1 \
  --manifest results/mimic_final_layer_matched_effect_100k_v1/training_manifest.csv \
  --concepts results/mimic_final_layer_matched_effect_100k_v1/derived/concepts_standardized.csv \
  --readout-root results/mimic_final_layer_matched_effect_100k_v1/readouts \
  --output-root results/mimic_final_layer_matched_effect_100k_v1/workers
