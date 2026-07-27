#!/bin/bash
#SBATCH -p scavenge
#SBATCH -J trkf_cls
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:59:00
#SBATCH -o /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark/logs/track_f_closure_transfer_%j.out
#SBATCH -e /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark/logs/track_f_closure_transfer_%j.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs

/rhf/allocations/wq8/yd68/venvs/st_mem_cu118/bin/python \
  scripts/run_track_f_closure_transfer.py
