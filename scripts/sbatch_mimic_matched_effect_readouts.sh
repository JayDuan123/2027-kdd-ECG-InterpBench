#!/bin/bash
#SBATCH -J mme_readout
#SBATCH -p commons
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH --array=0-5%6
#SBATCH -o logs/mimic_final_layer_matched_effect_v1/readout_%A_%a.out
#SBATCH -e logs/mimic_final_layer_matched_effect_v1/readout_%A_%a.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python \
  scripts/fit_mimic_final_layer_matched_effect_readout.py \
  --model-index "${SLURM_ARRAY_TASK_ID}"
