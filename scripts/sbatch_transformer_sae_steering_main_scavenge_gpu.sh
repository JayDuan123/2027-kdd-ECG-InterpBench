#!/bin/bash
#SBATCH -J tr_sae
#SBATCH -p scavenge
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 00:59:00
#SBATCH --array=0-15%8
#SBATCH -o logs/sae_extension/transformer_sae_steering_main_%A_%a.out
#SBATCH -e logs/sae_extension/transformer_sae_steering_main_%A_%a.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/sae_extension results/sae_extension/six_model_sae_audit/transformer_steering_main

PYTHON="/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python"
MANIFEST="results/sae_extension/six_model_sae_audit/phase0_selected_transformer_operating_points.csv"

LINE_NUMBER=$((SLURM_ARRAY_TASK_ID + 2))
LINE="$(sed -n "${LINE_NUMBER}p" "${MANIFEST}")"
if [[ -z "${LINE}" ]]; then
  echo "No manifest row for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

IFS=',' read -r CELL_INDEX MODEL CONCEPT TASK LAYER N_CAPACITY L0_TARGET CLAMP_N_FEATURES RECON_R2 SOURCE_CSV <<< "${LINE}"

OUT_DIR="results/sae_extension/six_model_sae_audit/transformer_steering_main/task_${SLURM_ARRAY_TASK_ID}_cell_${CELL_INDEX}"
mkdir -p "${OUT_DIR}"

echo "Running ${MODEL} ${CONCEPT}->${TASK}@L${LAYER} cell=${CELL_INDEX} N=${N_CAPACITY} L0=${L0_TARGET} clamp=${CLAMP_N_FEATURES} recon=${RECON_R2}"

"${PYTHON}" -m benchmark_v1.sae_extension.run_sae_layer \
  --environment transformer \
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
  --n-features-grid "${N_CAPACITY}" \
  --l0-grid "${L0_TARGET}" \
  --require-matched-tier in_band \
  --feature-ranking concept \
  --n-features "${CLAMP_N_FEATURES}" \
  --n-random 20 \
  --bootstrap-samples 1000 \
  --selectivity-mode endpoint \
  --steps 4000 \
  --checkpoint-every 250 \
  --skip-existing \
  --device cuda
