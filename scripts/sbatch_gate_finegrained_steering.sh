#!/bin/bash
#SBATCH -J v21_gate
#SBATCH -p commons
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:15:00
#SBATCH -o logs/sae_reconciliation/v21_gate_%j.out
#SBATCH -e logs/sae_reconciliation/v21_gate_%j.err
set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/sae_reconciliation
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/gate_expanded_steering_targets.py \
  --base results/sae_reconciliation/steering_benchmark_multimodel_v2_1_finegrained \
  --new-source ptbxl_scp_codes_v21
