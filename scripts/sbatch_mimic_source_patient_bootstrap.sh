#!/usr/bin/env bash
#SBATCH -J mimic_src_pboot
#SBATCH -p commons
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH -t 04:00:00
#SBATCH --array=0-449%24
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/mimic_source_benchmark_100k_v1/pboot_%A_%a.out
#SBATCH -e logs/mimic_source_benchmark_100k_v1/pboot_%A_%a.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/run_multiscale_sae_patient_bootstrap.py \
  --manifest results/mimic_source_benchmark_100k_v1/training_manifest.csv \
  --task-index "${SLURM_ARRAY_TASK_ID}" --bootstrap-samples 2000 --bootstrap-chunk 100 \
  --device cuda --concepts results/mimic_source_benchmark_100k_v1/derived/concepts_matrix.csv \
  --complete-case-concepts \
  --patient-manifest results/mimic_source_benchmark_100k_v1/derived/records.csv \
  --output-root results/mimic_source_benchmark_100k_v1/patient_bootstrap
