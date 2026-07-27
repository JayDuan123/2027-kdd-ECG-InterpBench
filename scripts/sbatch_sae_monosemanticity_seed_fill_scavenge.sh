#!/bin/bash
#SBATCH -J mono_seed
#SBATCH -p scavenge
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 00:59:00
#SBATCH --array=0-9%10
#SBATCH -o logs/sae_extension/monosem_seed_fill_%A_%a.out
#SBATCH -e logs/sae_extension/monosem_seed_fill_%A_%a.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/sae_extension results/sae_extension/six_model_sae_audit/monosem_seed_fill

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

PYTHON="/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python"

# Representative recon-band cells used by the monosemanticity taxonomy:
# CARDIAC-FM cell13 N128/L32, CSFM cell0 N8/L1, ECG-FM cell38 N128/L32,
# ECG-JEPA cell24 N8/L2, HuBERT-ECG cell1 N128/L8.
CELL_GRID=(13 0 38 24 1)
N_GRID=(128 8 128 8 128)
L0_GRID=(32 1 32 2 8)
SEED_GRID=(4312 4313)

IDX="${SLURM_ARRAY_TASK_ID}"
MODEL_SLOT=$((IDX / 2))
SEED_SLOT=$((IDX % 2))

CELL_INDEX="${CELL_GRID[$MODEL_SLOT]}"
N_VALUE="${N_GRID[$MODEL_SLOT]}"
L0_VALUE="${L0_GRID[$MODEL_SLOT]}"
SEED_VALUE="${SEED_GRID[$SEED_SLOT]}"

OUT_DIR="results/sae_extension/six_model_sae_audit/monosem_seed_fill/cell_${CELL_INDEX}/N${N_VALUE}/L0${L0_VALUE}/seed${SEED_VALUE}"
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
  --E-grid 1 \
  --n-features-grid "${N_VALUE}" \
  --l0-grid "${L0_VALUE}" \
  --steps 4000 \
  --seed "${SEED_VALUE}" \
  --checkpoint-every 250 \
  --skip-existing \
  --device cuda
