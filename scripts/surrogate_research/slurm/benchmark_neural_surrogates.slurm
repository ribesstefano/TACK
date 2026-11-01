#!/bin/bash
#SBATCH --job-name=tack-neural-pilot
#SBATCH --account=naiss2026-3-679-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-4%3
#SBATCH --output=logs/%x.%A_%a.log
set -euo pipefail
cd /nobackup/proj/disk/naiss2024-5-630/personal/ribes/TACK
unset PYTHONPATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
export HF_HOME=/nobackup/proj/disk/naiss2024-5-630/personal/ribes/cache/hf
export TACKAI_CACHE=/nobackup/proj/disk/naiss2024-5-630/personal/ribes/cache/TACK
export UV_CACHE_DIR=/nobackup/proj/disk/naiss2024-5-630/personal/ribes/cache/uv
.venv/bin/python -m pytest test/test_neural_surrogate.py -q
if (( SLURM_ARRAY_TASK_ID < 2 )); then
    features=(baseline morgan_r2_2048_bit)
    feature=${features[$SLURM_ARRAY_TASK_ID]}
    .venv/bin/python -u scripts/surrogate_research/benchmark_neural_surrogates.py \
        --snapshot models/research_v1 --output "${NEURAL_ROOT:-models/research_neural_pilot}/joint_$feature" \
        --mode joint --feature "$feature" --epochs 100 --policy "${LABEL_POLICY:-exact}"
else
    tasks=(dmax pdc50 activity)
    task=${tasks[$((SLURM_ARRAY_TASK_ID - 2))]}
    .venv/bin/python -u scripts/surrogate_research/benchmark_neural_surrogates.py \
        --snapshot models/research_v1 --output "${NEURAL_ROOT:-models/research_neural_pilot}/single_$task" \
        --mode single --task "$task" --feature baseline --epochs 100 --policy "${LABEL_POLICY:-exact}"
fi
