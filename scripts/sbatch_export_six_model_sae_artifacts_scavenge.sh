#!/bin/bash
#SBATCH -J sae_export6
#SBATCH -p scavenge
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:59:00
#SBATCH -o logs/sae_extension/export_six_model_sae_artifacts_%j.out
#SBATCH -e logs/sae_extension/export_six_model_sae_artifacts_%j.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/sae_extension

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

PYTHON="/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python"

"${PYTHON}" scripts/export_sae_artifacts.py \
  --continuation-csv results/analysis/model_comparison/cleanup_audit/continuation_canonical_strict_fdr.csv \
  --concepts-matrix results/manifest/concepts_matrix.csv \
  --probe-features-root results/probe_features \
  --out-dir results/sae_artifacts_six_model
