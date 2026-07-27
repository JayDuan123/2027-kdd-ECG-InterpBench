#!/usr/bin/env bash
#SBATCH --job-name=ext2_stability
#SBATCH --partition=long
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/benchmark_extension_v2/sae_stability_%j.out
#SBATCH --error=logs/benchmark_extension_v2/sae_stability_%j.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/benchmark_extension_v2
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
/rhf/allocations/wq8/yd68/venvs/ecg_fm_cu118/bin/python \
  scripts/analyze_sae_stability_extension.py
