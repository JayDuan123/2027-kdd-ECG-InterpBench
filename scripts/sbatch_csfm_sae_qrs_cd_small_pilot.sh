#!/bin/bash
#SBATCH -J sae_qrs_pilot
#SBATCH -p scavenge
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:59:00
#SBATCH -o logs/sae_extension/csfm_sae_qrs_cd_pilot_%j.out
#SBATCH -e logs/sae_extension/csfm_sae_qrs_cd_pilot_%j.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/sae_extension results/sae_extension/csfm_qrs_cd_small_pilot

PYTHON="/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python"

"${PYTHON}" -m benchmark_v1.sae_extension.run_sae_layer \
  --environment csfm \
  --cells results/analysis/model_comparison/leace_coupling_top_confirmed.csv \
  --coupling results/analysis/model_comparison/leace_coupling_risk_summary.csv \
  --artifacts results/sae_artifacts \
  --out results/sae_extension/csfm_qrs_cd_small_pilot \
  --pilot \
  --only-model CSFM \
  --only-concept qrs_duration \
  --only-task ptbxl_cd \
  --limit-cells 1 \
  --max-test-shards 16 \
  --E-grid 4 \
  --k0-grid 16 \
  --recon-r2-floor 0.5 \
  --min-task-retention 0.0 \
  --steps 4000 \
  --f-steps 5 \
  --n-random 5 \
  --n-features 32 \
  --n90-max-features 512 \
  --device cuda
