#!/bin/bash
#SBATCH -J stg4bgate
#SBATCH -p scavenge
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 00:59:00
#SBATCH --array=0-4%5
#SBATCH -o logs/sae_extension/stage4_b_gate_%A_%a.out
#SBATCH -e logs/sae_extension/stage4_b_gate_%A_%a.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/sae_extension results/sae_extension/six_model_sae_audit/stage4_replacement_b_gate

PYTHON="/rhf/allocations/wq8/yd68/venvs/ecg_fm_cu118/bin/python"
MANIFEST="results/analysis/model_comparison/orthogonal_concepts/stage4_replacement_b_gate_manifest.csv"

LINE_NUMBER=$((SLURM_ARRAY_TASK_ID + 2))
LINE="$(sed -n "${LINE_NUMBER}p" "${MANIFEST}")"
if [[ -z "${LINE}" ]]; then
  echo "No manifest row for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

IFS=',' read -r CELL_INDEX MODEL CONCEPT TASK LAYER N_CAPACITY L0_TARGET CLAMP_N_FEATURES RECON_R2 SOURCE_CSV CHECKPOINT_DIR RECON_R2_FLOOR AUDIT_MODE REASON <<< "${LINE}"

if [[ "${MODEL}" == "ST-MEM" ]]; then
  PYTHON="/rhf/allocations/wq8/yd68/venvs/st_mem_cu118/bin/python"
fi

OUT_DIR="results/sae_extension/six_model_sae_audit/stage4_replacement_b_gate/task_${SLURM_ARRAY_TASK_ID}_cell_${CELL_INDEX}"
mkdir -p "${OUT_DIR}"

echo "Stage IV B-gate: ${MODEL} ${CONCEPT}->${TASK}@L${LAYER} cell=${CELL_INDEX} N=${N_CAPACITY} L0=${L0_TARGET} clamp=${CLAMP_N_FEATURES} recon=${RECON_R2} mode=${AUDIT_MODE}"

"${PYTHON}" -m benchmark_v1.sae_extension.run_sae_layer \
  --environment transformer \
  --cells results/sae_extension/six_model_sae_audit/phase0_low_coupling_cells.csv \
  --coupling results/analysis/model_comparison/leace_coupling_risk_summary.csv \
  --artifacts results/sae_artifacts_six_model \
  --out "${OUT_DIR}" \
  --cell-index "${CELL_INDEX}" \
  --selection-mode floor \
  --recon-r2-floor "${RECON_R2_FLOOR}" \
  --max-dead-frac 1.0 \
  --min-task-retention 0.0 \
  --E-grid 1 \
  --n-features-grid "${N_CAPACITY}" \
  --l0-grid "${L0_TARGET}" \
  --feature-ranking concept \
  --n-features "${CLAMP_N_FEATURES}" \
  --n-random 20 \
  --bootstrap-samples 1000 \
  --selectivity-mode endpoint \
  --steps 4000 \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --checkpoint-every 250 \
  --skip-existing \
  --device cuda
