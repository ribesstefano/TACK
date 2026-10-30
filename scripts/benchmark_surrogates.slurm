#!/bin/bash
#SBATCH --job-name=tack-research-pilot
#SBATCH --account=naiss2026-3-679-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-26%4
#SBATCH --output=logs/%x.%A_%a.log
set -euo pipefail
cd /nobackup/proj/disk/naiss2024-5-630/personal/ribes/TACK
unset PYTHONPATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
export HF_HOME=/nobackup/proj/disk/naiss2024-5-630/personal/ribes/cache/hf
export TACKAI_CACHE=/nobackup/proj/disk/naiss2024-5-630/personal/ribes/cache/TACK
export UV_CACHE_DIR=/nobackup/proj/disk/naiss2024-5-630/personal/ribes/cache/uv
tasks=(dmax pdc50 activity)
models=(dummy linear extratrees xgb xgb xgb xgb xgb xgb)
features=(baseline baseline baseline baseline morgan_r2_2048_bit morgan_r3_2048_bit morgan_r2_4096_bit morgan_r3_4096_bit morgan_r2_2048_count)
candidate=$((SLURM_ARRAY_TASK_ID / 3))
task=${tasks[$((SLURM_ARRAY_TASK_ID % 3))]}
model=${models[$candidate]}
feature=${features[$candidate]}
.venv/bin/python -u scripts/benchmark_surrogates.py \
    --snapshot models/research_v1 \
    --output "${PILOT_ROOT:-models/research_pilot}/${task}_${model}_${feature}" \
    --task "$task" --model "$model" --feature "$feature" \
    --trials 0 --jobs "$SLURM_CPUS_PER_TASK" --policy "${LABEL_POLICY:-compat}" --eval-policy exact
