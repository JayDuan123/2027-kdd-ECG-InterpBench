#!/bin/bash
#SBATCH -J ext_index
#SBATCH -p scavenge
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 00:59:00
#SBATCH -o logs/external_activation_extraction/ext_index_%j.out
#SBATCH -e logs/external_activation_extraction/ext_index_%j.err
set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
: "${INDEX_COMMANDS_FILE:?INDEX_COMMANDS_FILE is required}"
while IFS= read -r CMD; do
  [[ -z "${CMD}" ]] || eval "${CMD}"
done < "${INDEX_COMMANDS_FILE}"
