#!/bin/bash
#SBATCH -p scavenge
#SBATCH -J trkf_mim
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:59:00
#SBATCH --array=0-3200%96
#SBATCH -o /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark/logs/track_f_full_mimic_%A_%a.out
#SBATCH -e /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark/logs/track_f_full_mimic_%A_%a.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs
export MPLCONFIGDIR=/tmp/matplotlib-${USER}
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/rhf/allocations/wq8/yd68/venvs/st_mem_cu118/bin/python

${PYTHON} scripts/build_track_f_full_waveform_concepts.py \
  --cohort mimic \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --shard-size 250 \
  --skip-existing
