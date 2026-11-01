#!/bin/bash
#SBATCH --job-name=tack-extra-pilot
#SBATCH --account=naiss2026-3-679-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-11%3
#SBATCH --output=logs/%x.%A_%a.log
set -euo pipefail
cd /nobackup/proj/disk/naiss2024-5-630/personal/ribes/TACK
unset PYTHONPATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
export HF_HOME=/nobackup/proj/disk/naiss2024-5-630/personal/ribes/cache/hf
export TACKAI_CACHE=/nobackup/proj/disk/naiss2024-5-630/personal/ribes/cache/TACK
export UV_CACHE_DIR=/nobackup/proj/disk/naiss2024-5-630/personal/ribes/cache/uv
tasks=(dmax pdc50 activity)
models=(catboost catboost xgb xgb)
features=(baseline morgan_r2_2048_bit baseline baseline)
ablations=(all all context molecule)
candidate=$((SLURM_ARRAY_TASK_ID / 3))
task=${tasks[$((SLURM_ARRAY_TASK_ID % 3))]}
model=${models[$candidate]}
feature=${features[$candidate]}
ablation=${ablations[$candidate]}
.venv/bin/python -u scripts/surrogate_research/benchmark_surrogates.py \
    --snapshot models/research_v1 \
    --output "models/research_extra_pilot/${task}_${model}_${feature}_${ablation}" \
    --task "$task" --model "$model" --feature "$feature" --ablation "$ablation" \
    --trials 0 --jobs "$SLURM_CPUS_PER_TASK" --policy exact --eval-policy exact
