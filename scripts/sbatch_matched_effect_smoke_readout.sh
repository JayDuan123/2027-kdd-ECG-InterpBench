#!/bin/bash
#SBATCH -J me_smoke_readout
#SBATCH -p commons
#SBATCH -c 12
#SBATCH --mem=48G
#SBATCH -t 00:30:00
#SBATCH -o logs/matched_effect_smoke_readout_%j.out
#SBATCH -e logs/matched_effect_smoke_readout_%j.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"

/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python \
  scripts/fit_final_layer_matched_effect_readout.py \
  --model-index 0 \
  --max-records-per-split 512 \
  --output-root results/final_layer_matched_effect_v1_smoke/readouts
