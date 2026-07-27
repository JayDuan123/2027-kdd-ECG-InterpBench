#!/bin/bash
#SBATCH -J me_smoke_boot
#SBATCH -p commons
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 00:20:00
#SBATCH -o logs/matched_effect_smoke_bootstrap_%j.out
#SBATCH -e logs/matched_effect_smoke_bootstrap_%j.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python \
  scripts/bootstrap_final_layer_matched_effect.py \
  --model-index 0 \
  --bootstrap-draws 100 \
  --expected-models 1 \
  --expected-seeds 1 \
  --workers-root results/final_layer_matched_effect_v1_smoke/workers \
  --output-root results/final_layer_matched_effect_v1_smoke_v2/bootstrap
