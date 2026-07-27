#!/bin/bash
#SBATCH -p scavenge
#SBATCH -J trkf_ext
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:59:00
#SBATCH --array=0-148%96
#SBATCH -o /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark/logs/track_f_full_external_%A_%a.out
#SBATCH -e /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark/logs/track_f_full_external_%A_%a.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs
export MPLCONFIGDIR=/tmp/matplotlib-${USER}
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/rhf/allocations/wq8/yd68/venvs/st_mem_cu118/bin/python
SHARD_SIZE=500
TASK_ID=${SLURM_ARRAY_TASK_ID}

if (( TASK_ID < 44 )); then
  COHORT=ptbxl
  SHARD_INDEX=${TASK_ID}
elif (( TASK_ID < 65 )); then
  COHORT=chapman
  SHARD_INDEX=$((TASK_ID - 44))
elif (( TASK_ID < 79 )); then
  COHORT=cpsc
  SHARD_INDEX=$((TASK_ID - 65))
else
  COHORT=ningbo
  SHARD_INDEX=$((TASK_ID - 79))
fi

${PYTHON} scripts/build_track_f_full_waveform_concepts.py \
  --cohort "${COHORT}" \
  --shard-index "${SHARD_INDEX}" \
  --shard-size "${SHARD_SIZE}" \
  --skip-existing
