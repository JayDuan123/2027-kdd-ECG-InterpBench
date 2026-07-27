#!/bin/bash
#SBATCH -J sae2d_seed
#SBATCH -p scavenge
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 00:59:00
#SBATCH --array=0-53%18
#SBATCH -o logs/sae_extension/six_model_sae_2d_seed_grid_%A_%a.out
#SBATCH -e logs/sae_extension/six_model_sae_2d_seed_grid_%A_%a.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/sae_extension results/sae_extension/six_model_sae_audit/sae_2d_profile_seed_grid

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

PYTHON="/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python"

# One representative low-coupling, LEACE-confirmed cell per model from
# phase0_low_coupling_cells.csv. This seed grid is for the model-level 2D SAE
# dictionary-stability profile, not for per-cell steering claims.
CELL_GRID=(13 0 18 24 1 14)
E_GRID=(8 16 32)
SEED_GRID=(4311 4312 4313)

IDX="${SLURM_ARRAY_TASK_ID}"
MODEL_SLOT=$((IDX / 9))
REM=$((IDX % 9))
E_IDX=$((REM / 3))
SEED_IDX=$((REM % 3))

CELL_INDEX="${CELL_GRID[$MODEL_SLOT]}"
E_VALUE="${E_GRID[$E_IDX]}"
SEED_VALUE="${SEED_GRID[$SEED_IDX]}"

OUT_DIR="results/sae_extension/six_model_sae_audit/sae_2d_profile_seed_grid/cell_${CELL_INDEX}/E${E_VALUE}/seed${SEED_VALUE}"
mkdir -p "${OUT_DIR}"

"${PYTHON}" -m benchmark_v1.sae_extension.run_sae_layer \
  --environment benchmark \
  --cells results/sae_extension/six_model_sae_audit/phase0_low_coupling_cells.csv \
  --coupling results/analysis/model_comparison/leace_coupling_risk_summary.csv \
  --artifacts results/sae_artifacts_six_model \
  --out "${OUT_DIR}" \
  --cell-index "${CELL_INDEX}" \
  --recon-curve-only \
  --selection-mode recon_band \
  --recon-target 0.90 \
  --recon-band-width 0.02 \
  --relaxed-band-width 0.04 \
  --E-grid "${E_VALUE}" \
  --l0-grid 32 \
  --steps 4000 \
  --seed "${SEED_VALUE}" \
  --checkpoint-every 250 \
  --skip-existing \
  --device cuda
