#!/usr/bin/env bash
#SBATCH -J mimic_src_audit
#SBATCH -p commons
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -o logs/mimic_source_benchmark_100k_v1/sae_audit_%j.out
#SBATCH -e logs/mimic_source_benchmark_100k_v1/sae_audit_%j.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/audit_multiscale_sae.py \
  --root results/mimic_source_benchmark_100k_v1 --expected-concepts 7
