#!/bin/bash
#SBATCH --job-name=tack-exact-current
#SBATCH --account=naiss2026-3-679-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-2%3
#SBATCH --output=logs/%x.%A_%a.log
set -euo pipefail
cd /nobackup/proj/disk/naiss2024-5-630/personal/ribes/TACK
unset PYTHONPATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
tasks=(dmax pdc50 activity)
task=${tasks[$SLURM_ARRAY_TASK_ID]}
.venv/bin/python -u scripts/surrogate_research/benchmark_surrogates.py \
    --snapshot models/research_v1 \
    --output "models/research_exact_current/${task}_current_xgb_baseline_exact" \
    --task "$task" --model current_xgb --feature baseline \
    --repeats 5 --folds 5 --trials 20 --jobs "$SLURM_CPUS_PER_TASK" \
    --policy exact --eval-policy exact
