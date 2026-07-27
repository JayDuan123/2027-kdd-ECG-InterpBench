#!/bin/bash
#SBATCH -J access_sum
#SBATCH -p commons
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o logs/accessibility_calibration_summary_%j.out
#SBATCH -e logs/accessibility_calibration_summary_%j.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python \
  scripts/summarize_accessibility_calibration.py \
  --expected-cells 90 \
  --expected-concepts 49
