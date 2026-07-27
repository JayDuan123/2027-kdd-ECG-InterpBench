#!/bin/bash
#SBATCH -J csfm_sae_smoke
#SBATCH -p scavenge
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 00:59:00
#SBATCH -o logs/sae_extension/csfm_sae_smoke_%j.out
#SBATCH -e logs/sae_extension/csfm_sae_smoke_%j.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/sae_extension results/sae_extension/six_model_sae_audit/csfm_smoke

PYTHON="/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python"
N_VALUE="${N_VALUE:?set N_VALUE from Phase 0 in-band result}"
L0_VALUE="${L0_VALUE:?set L0_VALUE from Phase 0 in-band result}"
MAX_TEST_SHARDS="${MAX_TEST_SHARDS:-1}"
N_RANDOM="${N_RANDOM:-3}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-200}"
OUT_DIR="results/sae_extension/six_model_sae_audit/csfm_smoke/qrs_cd_N${N_VALUE}_L0${L0_VALUE}_shards${MAX_TEST_SHARDS}_rand${N_RANDOM}"
mkdir -p "${OUT_DIR}"

"${PYTHON}" -m benchmark_v1.sae_extension.run_sae_layer \
  --environment csfm \
  --cells results/analysis/model_comparison/leace_coupling_top_confirmed.csv \
  --coupling results/analysis/model_comparison/leace_coupling_risk_summary.csv \
  --artifacts results/sae_artifacts \
  --out "${OUT_DIR}" \
  --only-model CSFM \
  --only-concept qrs_duration \
  --only-task ptbxl_cd \
  --limit-cells 1 \
  --max-test-shards "${MAX_TEST_SHARDS}" \
  --selection-mode recon_band \
  --recon-target 0.90 \
  --recon-band-width 0.02 \
  --relaxed-band-width 0.04 \
  --require-matched-tier in_band \
  --E-grid 1 \
  --n-features-grid "${N_VALUE}" \
  --l0-grid "${L0_VALUE}" \
  --steps 4000 \
  --checkpoint-every 250 \
  --n-random "${N_RANDOM}" \
  --n-features 64 \
  --n90-max-features 512 \
  --feature-ranking concept \
  --selectivity-mode endpoint \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
  --seed 4311 \
  --skip-existing \
  --device cuda
