#!/usr/bin/env bash
#SBATCH --job-name=harm_smoke_audit
#SBATCH --partition=commons
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/input_harmonization/smoke_audit_%j.out
#SBATCH --error=logs/input_harmonization/smoke_audit_%j.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
/rhf/allocations/wq8/yd68/venvs/ecg_fm_cu118/bin/python scripts/audit_input_harmonization_smoke.py
