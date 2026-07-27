#!/bin/bash
#SBATCH -J csfm_sae_main
#SBATCH -p commons
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH --array=0-23%4
#SBATCH -o logs/sae_extension/csfm_sae_main_%A_%a.out
#SBATCH -e logs/sae_extension/csfm_sae_main_%A_%a.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/sae_extension results/sae_extension/csfm_sae_main_robustness

PYTHON="/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python"

CONCEPTS=(st_amp_global qrs_duration qrst_angle p_found)
TASKS=(mi_ischemia ptbxl_cd ptbxl_cd af_rhythm)
NAMES=(st_mi qrs_cd qrst_cd p_af_control)
E_GRID=(4 8)
SEEDS=(4311 4312 4313)

IDX="${SLURM_ARRAY_TASK_ID}"
CELL_IDX=$((IDX / 6))
REM=$((IDX % 6))
E_IDX=$((REM / 3))
SEED_IDX=$((REM % 3))

CONCEPT="${CONCEPTS[$CELL_IDX]}"
TASK="${TASKS[$CELL_IDX]}"
NAME="${NAMES[$CELL_IDX]}"
E_VALUE="${E_GRID[$E_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"
OUT_DIR="results/sae_extension/csfm_sae_main_robustness/${NAME}/E${E_VALUE}/seed${SEED}"

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
  --E-grid "${E_VALUE}" \
  --k0-grid 16 \
  --recon-r2-floor 0.5 \
  --min-task-retention 0.0 \
  --steps 8000 \
  --f-steps 7 \
  --n-random 20 \
  --n-features 64 \
  --n90-max-features 512 \
  --feature-ranking concept \
  --seed "${SEED}" \
  --skip-existing \
  --device cuda
