#!/bin/bash
#SBATCH --job-name=tack-mol-followup
#SBATCH --account=naiss2026-3-679-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-4%3
#SBATCH --output=logs/%x.%A_%a.log
set -euo pipefail
cd /nobackup/proj/disk/naiss2024-5-630/personal/ribes/TACK
unset PYTHONPATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
tasks=(dmax activity dmax dmax activity)
models=(current_xgb current_xgb catboost xgb catboost)
features=(baseline baseline morgan_r2_2048_bit morgan_r3_2048_bit morgan_r2_2048_bit)
i=$SLURM_ARRAY_TASK_ID
.venv/bin/python -u scripts/surrogate_research/benchmark_surrogates.py \
  --snapshot models/research_v1 --output "models/research_molecular_followup/${tasks[$i]}_${models[$i]}_${features[$i]}" \
  --task "${tasks[$i]}" --model "${models[$i]}" --feature "${features[$i]}" \
  --policy exact --eval-policy exact --repeats 1 --folds 5 --trials 20 --jobs "$SLURM_CPUS_PER_TASK"
