#!/bin/bash
#SBATCH -J sae_qrs_cd
#SBATCH -p scavenge
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -o logs/sae_extension/csfm_sae_qrs_cd_%j.out
#SBATCH -e logs/sae_extension/csfm_sae_qrs_cd_%j.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/sae_extension results/sae_extension/csfm_qrs_cd_smoke

PYTHON="/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python"

"${PYTHON}" -m benchmark_v1.sae_extension.run_sae_layer \
  --environment csfm \
  --cells results/analysis/model_comparison/leace_coupling_top_confirmed.csv \
  --coupling results/analysis/model_comparison/leace_coupling_risk_summary.csv \
  --artifacts results/sae_artifacts \
  --out results/sae_extension/csfm_qrs_cd_smoke \
  --pilot \
  --only-model CSFM \
  --only-concept qrs_duration \
  --only-task ptbxl_cd \
  --limit-cells 1 \
  --max-test-shards 1 \
  --E-grid 4 \
  --k0-grid 16 \
  --recon-r2-floor -1.0 \
  --min-task-retention 0.0 \
  --steps 500 \
  --f-steps 3 \
  --n-random 2 \
  --n-features 32 \
  --n90-max-features 128 \
  --device cuda
