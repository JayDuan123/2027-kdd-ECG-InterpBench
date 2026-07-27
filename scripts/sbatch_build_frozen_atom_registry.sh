#!/bin/bash
#SBATCH -J frozen_atoms
#SBATCH -p scavenge
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:59:00
#SBATCH --array=0-17%12
#SBATCH -o logs/external_benchmark/frozen_atoms_%A_%a.out
#SBATCH -e logs/external_benchmark/frozen_atoms_%A_%a.err
set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/external_benchmark
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/build_matched_scale_frozen_atom_registry.py --task-index "${SLURM_ARRAY_TASK_ID}"
