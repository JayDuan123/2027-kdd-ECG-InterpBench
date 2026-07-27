#!/usr/bin/env bash
#SBATCH --job-name=concept_coupling
#SBATCH --partition=commons
#SBATCH --output=logs/concept_coupling/%j.out
#SBATCH --error=logs/concept_coupling/%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=01:00:00

set -euo pipefail

PROJECT_ROOT="/rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark"
PYTHON="/rhf/allocations/wq8/yd68/venvs/ecg_fm_cu118/bin/python"

cd "${PROJECT_ROOT}"
mkdir -p logs/concept_coupling

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"${PYTHON}" scripts/make_concept_coupling_audit.py --workers "${SLURM_CPUS_PER_TASK:-32}"
