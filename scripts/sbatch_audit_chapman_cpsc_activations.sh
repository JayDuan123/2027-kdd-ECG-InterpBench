#!/bin/bash
#SBATCH -J ext_audit
#SBATCH -p scavenge
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 00:59:00
#SBATCH -o logs/external_benchmark/ext_audit_%j.out
#SBATCH -e logs/external_benchmark/ext_audit_%j.err
set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/audit_external_activation_commands.py \
 --commands results/activations_external_full_v1/plan_chapman_cpsc/pooled_commands.txt \
 --out results/external_benchmark_v1/chapman_cpsc_pooled_shard_audit.csv --check-finite
