#!/bin/bash
#SBATCH -J sae_p0_recon
#SBATCH -p scavenge
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 00:59:00
#SBATCH --array=0-50%12
#SBATCH -o logs/sae_extension/six_model_phase0_%A_%a.out
#SBATCH -e logs/sae_extension/six_model_phase0_%A_%a.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/sae_extension results/sae_extension/six_model_sae_audit/phase0_recon_curves

PYTHON="/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python"
CELL_INDEX="${SLURM_ARRAY_TASK_ID}"
OUT_DIR="results/sae_extension/six_model_sae_audit/phase0_recon_curves/cell_${CELL_INDEX}"
mkdir -p "${OUT_DIR}"

"${PYTHON}" -m benchmark_v1.sae_extension.run_sae_layer \
  --environment benchmark \
  --cells results/analysis/model_comparison/leace_coupling_top_confirmed.csv \
  --coupling results/analysis/model_comparison/leace_coupling_risk_summary.csv \
  --artifacts results/sae_artifacts \
  --out "${OUT_DIR}" \
  --cell-index "${CELL_INDEX}" \
  --recon-curve-only \
  --selection-mode recon_band \
  --recon-target 0.90 \
  --recon-band-width 0.02 \
  --relaxed-band-width 0.04 \
  --E-grid 2,4,8,16,32 \
  --l0-grid 16,32,64 \
  --steps 4000 \
  --checkpoint-every 250 \
  --skip-existing \
  --device cuda
