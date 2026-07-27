#!/bin/bash
#SBATCH -J me_sum_v3
#SBATCH -p commons
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/matched_effect_summary_v3_%j.out
#SBATCH -e logs/matched_effect_summary_v3_%j.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python \
  scripts/summarize_final_layer_matched_effect.py \
  --bootstrap-root results/final_layer_matched_effect_v1/bootstrap_v2 \
  --output-root results/final_layer_matched_effect_v1/summary_v3
