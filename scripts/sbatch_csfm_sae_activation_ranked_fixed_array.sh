#!/bin/bash
#SBATCH -J csfm_sae_fix
#SBATCH -p commons
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH --array=0-3%2
#SBATCH -o logs/sae_extension/csfm_sae_fix_%A_%a.out
#SBATCH -e logs/sae_extension/csfm_sae_fix_%A_%a.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/sae_extension results/sae_extension/csfm_expanded_activation_ranked_fixed

PYTHON="/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python"

CONCEPTS=(st_amp_global qrs_duration qrst_angle p_found)
TASKS=(mi_ischemia ptbxl_cd ptbxl_cd af_rhythm)
NAMES=(st_mi qrs_cd qrst_cd p_af_control)

IDX="${SLURM_ARRAY_TASK_ID}"
CONCEPT="${CONCEPTS[$IDX]}"
TASK="${TASKS[$IDX]}"
NAME="${NAMES[$IDX]}"
OUT_DIR="results/sae_extension/csfm_expanded_activation_ranked_fixed/${NAME}"

mkdir -p "${OUT_DIR}"

"${PYTHON}" -m benchmark_v1.sae_extension.run_sae_layer \
  --environment csfm \
  --cells results/analysis/model_comparison/leace_coupling_top_confirmed.csv \
  --coupling results/analysis/model_comparison/leace_coupling_risk_summary.csv \
  --artifacts results/sae_artifacts \
  --out "${OUT_DIR}" \
  --pilot \
  --only-model CSFM \
  --only-concept "${CONCEPT}" \
  --only-task "${TASK}" \
  --limit-cells 1 \
  --max-test-shards 0 \
  --E-grid 4,8 \
  --k0-grid 16,32 \
  --recon-r2-floor 0.5 \
  --min-task-retention 0.0 \
  --steps 8000 \
  --f-steps 7 \
  --n-random 10 \
  --n-features 64 \
  --n90-max-features 512 \
  --feature-ranking concept \
  --device cuda
