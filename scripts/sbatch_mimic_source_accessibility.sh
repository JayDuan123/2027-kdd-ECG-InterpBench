#!/usr/bin/env bash
#SBATCH -J mimic_src_access
#SBATCH -p commons
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=56G
#SBATCH -t 06:00:00
#SBATCH --array=0-89%18
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/mimic_source_benchmark_100k_v1/access_%A_%a.out
#SBATCH -e logs/mimic_source_benchmark_100k_v1/access_%A_%a.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/run_accessibility_calibration_worker.py \
  --manifest results/mimic_source_benchmark_100k_v1/training_manifest.csv \
  --calibration-index "${SLURM_ARRAY_TASK_ID}" --expansion 8 --device cuda \
  --concepts results/mimic_source_benchmark_100k_v1/derived/concepts_matrix.csv \
  --concept-registry results/mimic_source_benchmark_100k_v1/derived/concept_registry.csv \
  --complete-case-concepts \
  --patient-manifest results/mimic_source_benchmark_100k_v1/derived/records.csv \
  --output-root results/mimic_source_benchmark_100k_v1/accessibility/workers
