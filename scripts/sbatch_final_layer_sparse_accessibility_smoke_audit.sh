#!/bin/bash
#SBATCH -J fl_sparse_smoke_audit
#SBATCH -p commons
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:10:00
#SBATCH -o logs/final_layer_sparse_smoke_audit_%j.out
#SBATCH -e logs/final_layer_sparse_smoke_audit_%j.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python \
  scripts/audit_final_layer_sparse_accessibility_smoke.py
