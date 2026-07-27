#!/usr/bin/env bash
#SBATCH -J mimic_src_dict
#SBATCH -p commons
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH --array=0-29%12
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/mimic_source_benchmark_100k_v1/dictionary_%A_%a.out
#SBATCH -e logs/mimic_source_benchmark_100k_v1/dictionary_%A_%a.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/run_dictionary_accessibility_worker.py \
  --manifest results/mimic_source_benchmark_100k_v1/training_manifest.csv \
  --group-index "${SLURM_ARRAY_TASK_ID}" --expansion 8 --device cuda \
  --random-replicates 20 --budget-replicates 20 --matched-budget 768 \
  --concepts results/mimic_source_benchmark_100k_v1/derived/concepts_matrix.csv \
  --tasks results/mimic_source_benchmark_100k_v1/derived/tasks_matrix.csv \
  --concept-registry results/mimic_source_benchmark_100k_v1/derived/concept_registry.csv \
  --task-registry results/mimic_source_benchmark_100k_v1/derived/task_registry.csv \
  --complete-case-concepts \
  --output-root results/mimic_source_benchmark_100k_v1/dictionary/workers
