#!/usr/bin/env bash
#SBATCH --job-name=harm_merge
#SBATCH --partition=commons
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/input_harmonization/materialize_%j.out
#SBATCH --error=logs/input_harmonization/materialize_%j.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
/rhf/allocations/wq8/yd68/venvs/ecg_fm_cu118/bin/python scripts/materialize_input_harmonization_features.py
