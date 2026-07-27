#!/usr/bin/env bash
#SBATCH -J mme100k_readout
#SBATCH -p commons
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH --array=0-5%6
#SBATCH -o logs/mimic_final_layer_matched_effect_100k_v1/readout_%A_%a.out
#SBATCH -e logs/mimic_final_layer_matched_effect_100k_v1/readout_%A_%a.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python \
  scripts/fit_mimic_final_layer_matched_effect_readout.py \
  --model-index "${SLURM_ARRAY_TASK_ID}" \
  --protocol mimic_final_layer_matched_effect_100k_v1 \
  --manifest results/mimic_final_layer_matched_effect_100k_v1/training_manifest.csv \
  --concepts results/mimic_final_layer_matched_effect_100k_v1/derived/concepts_standardized.csv \
  --output-root results/mimic_final_layer_matched_effect_100k_v1/readouts
