#!/bin/bash
#SBATCH -J pool_transport
#SBATCH -p scavenge
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 00:59:00
#SBATCH -o logs/external_activation_extraction/pool_transport_%j.out
#SBATCH -e logs/external_activation_extraction/pool_transport_%j.err
set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
PLAN=results/activations_external_pooled_smoke/plan
while IFS= read -r CMD; do
  [[ -z "${CMD}" ]] || eval "${CMD}"
done < "${PLAN}/index_commands.txt"
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/evaluate_pooled_external_transport.py
