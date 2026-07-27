#!/bin/bash
#SBATCH -J tr_sae_smoke
#SBATCH -p scavenge
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 00:45:00
#SBATCH -o logs/sae_extension/transformer_sae_smoke_%j.out
#SBATCH -e logs/sae_extension/transformer_sae_smoke_%j.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/sae_extension results/sae_extension/six_model_sae_audit/transformer_smoke_l0clamp/cell_24

PYTHON="/rhf/allocations/wq8/yd68/venvs/ecg_jepa_cu118/bin/python"
OUT_DIR="results/sae_extension/six_model_sae_audit/transformer_smoke_l0clamp/cell_24"

"${PYTHON}" -m benchmark_v1.sae_extension.run_sae_layer \
  --environment transformer \
  --cells results/sae_extension/six_model_sae_audit/phase0_low_coupling_cells.csv \
  --coupling results/analysis/model_comparison/leace_coupling_risk_summary.csv \
  --artifacts results/sae_artifacts_six_model \
  --out "${OUT_DIR}" \
  --cell-index 24 \
  --selection-mode recon_band \
  --recon-target 0.90 \
  --recon-band-width 0.02 \
  --relaxed-band-width 0.04 \
  --E-grid 1 \
  --n-features-grid 8 \
  --l0-grid 2 \
  --require-matched-tier in_band \
  --feature-ranking concept \
  --n-features 2 \
  --n-random 2 \
  --bootstrap-samples 0 \
  --max-test-shards 20 \
  --selectivity-mode endpoint \
  --steps 4000 \
  --checkpoint-every 250 \
  --skip-existing \
  --device cuda
