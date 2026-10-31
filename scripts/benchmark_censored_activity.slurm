#!/bin/bash
#SBATCH --job-name=tack-censored-activity
#SBATCH --account=naiss2026-3-679-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-2%2
#SBATCH --output=logs/%x.%A_%a.log
set -euo pipefail
cd /nobackup/proj/disk/naiss2024-5-630/personal/ribes/TACK
unset PYTHONPATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
models=(extratrees catboost xgb)
model=${models[$SLURM_ARRAY_TASK_ID]}
.venv/bin/python scripts/benchmark_surrogates.py \
    --snapshot models/research_v1 \
    --output "models/research_censored_activity/activity_${model}_baseline" \
    --task activity --model "$model" --feature baseline \
    --trials 0 --jobs "$SLURM_CPUS_PER_TASK" --policy censored --eval-policy exact
