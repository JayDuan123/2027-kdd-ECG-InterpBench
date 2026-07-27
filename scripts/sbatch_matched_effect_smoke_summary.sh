#!/bin/bash
#SBATCH -J me_smoke_sum
#SBATCH -p commons
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 00:20:00
#SBATCH -o logs/matched_effect_smoke_summary_%j.out
#SBATCH -e logs/matched_effect_smoke_summary_%j.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python \
  scripts/summarize_final_layer_matched_effect.py \
  --expected-models 1 \
  --expected-workers 1 \
  --expected-readouts 1 \
  --readouts-root results/final_layer_matched_effect_v1_smoke/readouts \
  --workers-root results/final_layer_matched_effect_v1_smoke/workers \
  --bootstrap-root results/final_layer_matched_effect_v1_smoke_v2/bootstrap \
  --output-root results/final_layer_matched_effect_v1_smoke_v2/summary
