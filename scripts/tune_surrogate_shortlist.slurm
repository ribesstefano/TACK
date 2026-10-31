#!/bin/bash
#SBATCH --job-name=tack-tuned-shortlist
#SBATCH --account=naiss2026-3-679-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-5%3
#SBATCH --output=logs/%x.%A_%a.log
set -euo pipefail
cd /nobackup/proj/disk/naiss2024-5-630/personal/ribes/TACK
unset PYTHONPATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
export HF_HOME=/nobackup/proj/disk/naiss2024-5-630/personal/ribes/cache/hf
export TACKAI_CACHE=/nobackup/proj/disk/naiss2024-5-630/personal/ribes/cache/TACK
export UV_CACHE_DIR=/nobackup/proj/disk/naiss2024-5-630/personal/ribes/cache/uv
tasks=(dmax dmax pdc50 pdc50 activity activity)
models=(extratrees catboost xgb xgb extratrees catboost)
features=(baseline baseline morgan_r3_4096_bit morgan_r2_2048_bit baseline baseline)
task=${tasks[$SLURM_ARRAY_TASK_ID]}
model=${models[$SLURM_ARRAY_TASK_ID]}
feature=${features[$SLURM_ARRAY_TASK_ID]}
.venv/bin/python -u scripts/benchmark_surrogates.py \
    --snapshot models/research_v1 \
    --output "models/research_tuned_shortlist/${task}_${model}_${feature}" \
    --task "$task" --model "$model" --feature "$feature" \
    --trials 20 --jobs "$SLURM_CPUS_PER_TASK" --policy exact --eval-policy exact
