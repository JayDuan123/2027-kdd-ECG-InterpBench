#!/bin/bash
#SBATCH -J sae_p0_lo
#SBATCH -p scavenge
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:59:00
#SBATCH --array=0-3%4
#SBATCH -o logs/sae_extension/six_model_phase0_sixart_lowN_smoke_%A_%a.out
#SBATCH -e logs/sae_extension/six_model_phase0_sixart_lowN_smoke_%A_%a.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/sae_extension results/sae_extension/six_model_sae_audit/phase0_recon_grid_sixart_lowN_smoke

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

PYTHON="/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python"
CELL_INDEX=14
N_GRID=(2 3 4 6)
L0_VALUE=1

IDX="${SLURM_ARRAY_TASK_ID}"
N_VALUE="${N_GRID[$IDX]}"

OUT_DIR="results/sae_extension/six_model_sae_audit/phase0_recon_grid_sixart_lowN_smoke/cell_${CELL_INDEX}/N${N_VALUE}/L0${L0_VALUE}"
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
  --checkpoint-every 250 \
  --skip-existing \
  --device cpu
