#!/bin/bash
#SBATCH -p scavenge
#SBATCH -J trkf_sum
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:59:00
#SBATCH -o /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark/logs/track_f_full_summary_%j.out
#SBATCH -e /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark/logs/track_f_full_summary_%j.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs

/rhf/allocations/wq8/yd68/venvs/st_mem_cu118/bin/python \
  scripts/summarize_track_f_full_waveform_concepts.py
