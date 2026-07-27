#!/usr/bin/env bash
#SBATCH -J mimic_src_fsgpu
#SBATCH -p commons
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=56G
#SBATCH -t 08:00:00
#SBATCH --array=0-137%24
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/mimic_source_benchmark_100k_v1/final_sparse_gpu_%A_%a.out
#SBATCH -e logs/mimic_source_benchmark_100k_v1/final_sparse_gpu_%A_%a.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
if (( SLURM_ARRAY_TASK_ID < 18 )); then kind=sae; index="${SLURM_ARRAY_TASK_ID}"; else kind=random; index="$((SLURM_ARRAY_TASK_ID - 18))"; fi
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/run_final_layer_sparse_accessibility_worker.py \
  --manifest results/mimic_source_benchmark_100k_v1/training_manifest.csv \
  --source-kind "${kind}" --source-index "${index}" --device cuda \
  --concepts results/mimic_source_benchmark_100k_v1/derived/concepts_matrix.csv \
  --concept-registry results/mimic_source_benchmark_100k_v1/derived/concept_registry.csv \
  --expected-concepts 7 --complete-case-concepts \
  --patient-manifest results/mimic_source_benchmark_100k_v1/derived/records.csv \
  --pca-root results/mimic_source_benchmark_100k_v1/pca/workers \
  --output-root results/mimic_source_benchmark_100k_v1/final_sparse/workers
