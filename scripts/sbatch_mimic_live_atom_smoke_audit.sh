#!/usr/bin/env bash
#SBATCH -J mlive_audit
#SBATCH -p commons
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:15:00
#SBATCH -o logs/mimic_final_layer_live_atom_matched_effect_100k_v1/smoke_audit_%j.out
#SBATCH -e logs/mimic_final_layer_live_atom_matched_effect_100k_v1/smoke_audit_%j.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/audit_mimic_live_atom_smoke.py
