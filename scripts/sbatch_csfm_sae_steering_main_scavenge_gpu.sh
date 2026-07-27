#!/bin/bash
#SBATCH -J csfm_sae
#SBATCH -p scavenge
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 00:59:00
#SBATCH --array=0-10%4
#SBATCH -o logs/sae_extension/csfm_sae_steering_main_%A_%a.out
#SBATCH -e logs/sae_extension/csfm_sae_steering_main_%A_%a.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/sae_extension results/sae_extension/six_model_sae_audit/csfm_steering_main_l0clamp

PYTHON="/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python"
CELL_GRID=(0 16 23 3 15 5 20 4 11 9 6)

CELL_INDEX="${CELL_GRID[$SLURM_ARRAY_TASK_ID]}"
OUT_DIR="results/sae_extension/six_model_sae_audit/csfm_steering_main_l0clamp/cell_${CELL_INDEX}"
mkdir -p "${OUT_DIR}"

"${PYTHON}" -m benchmark_v1.sae_extension.run_sae_layer \
  --environment csfm \
  --cells results/sae_extension/six_model_sae_audit/phase0_low_coupling_cells.csv \
  --coupling results/analysis/model_comparison/leace_coupling_risk_summary.csv \
  --artifacts results/sae_artifacts_six_model \
  --out "${OUT_DIR}" \
  --cell-index "${CELL_INDEX}" \
  --selection-mode recon_band \
  --recon-target 0.90 \
  --recon-band-width 0.02 \
  --relaxed-band-width 0.04 \
  --E-grid 1 \
  --n-features-grid 8 \
  --l0-grid 1 \
  --require-matched-tier in_band \
  --feature-ranking concept \
  --n-features 1 \
  --n-random 20 \
  --bootstrap-samples 1000 \
  --selectivity-mode endpoint \
  --steps 4000 \
  --checkpoint-every 250 \
  --skip-existing \
  --device cuda
