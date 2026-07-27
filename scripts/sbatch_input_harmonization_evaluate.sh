#!/usr/bin/env bash
#SBATCH --job-name=harm_eval
#SBATCH --partition=commons
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/input_harmonization/evaluate_%j.out
#SBATCH --error=logs/input_harmonization/evaluate_%j.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MPLCONFIGDIR="/tmp/mplconfig-input-harmonization-${SLURM_JOB_ID}"
/rhf/allocations/wq8/yd68/venvs/ecg_fm_cu118/bin/python scripts/evaluate_input_harmonization.py --bootstrap 2000
