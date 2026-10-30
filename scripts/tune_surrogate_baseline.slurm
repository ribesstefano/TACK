#!/bin/bash
#SBATCH --job-name=tack-tuned-baseline
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
export HF_HOME=/nobackup/proj/disk/naiss2024-5-630/personal/ribes/cache/hf
export TACKAI_CACHE=/nobackup/proj/disk/naiss2024-5-630/personal/ribes/cache/TACK
export UV_CACHE_DIR=/nobackup/proj/disk/naiss2024-5-630/personal/ribes/cache/uv
tasks=(dmax pdc50 activity)
task=${tasks[$SLURM_ARRAY_TASK_ID]}
.venv/bin/python -m pytest test/test_benchmark_surrogates.py -q
.venv/bin/python -u scripts/benchmark_surrogates.py \
    --snapshot models/research_v1 --output "models/research_tuned_baseline/$task" \
    --task "$task" --model current_xgb --feature baseline \
    --trials 20 --jobs "$SLURM_CPUS_PER_TASK" --policy compat --eval-policy exact
