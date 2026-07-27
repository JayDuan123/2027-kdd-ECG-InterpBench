#!/bin/bash
#SBATCH -J mme_smoke_ro
#SBATCH -p commons
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o logs/mimic_final_layer_matched_effect_v1/smoke_readout_%j.out
#SBATCH -e logs/mimic_final_layer_matched_effect_v1/smoke_readout_%j.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python \
  scripts/fit_mimic_final_layer_matched_effect_readout.py \
  --model-index 0 \
  --manifest results/mimic_final_layer_matched_effect_v1/smoke_training_manifest.csv \
  --output-root results/mimic_final_layer_matched_effect_v1/smoke/readouts \
  --max-records-per-split 256
