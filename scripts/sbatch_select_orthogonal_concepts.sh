#!/bin/bash
#SBATCH -J ortho_con
#SBATCH -p scavenge
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:59:00
#SBATCH -o logs/orthogonal_concepts_%j.out
#SBATCH -e logs/orthogonal_concepts_%j.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

PYTHON="/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python"

"${PYTHON}" scripts/select_orthogonal_concepts.py \
  --refresh-coupling \
  --workers "${SLURM_CPUS_PER_TASK:-8}"
