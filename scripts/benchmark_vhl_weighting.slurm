#!/bin/bash
#SBATCH --job-name=tack-vhl-weighting
#SBATCH --account=naiss2026-3-679-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x.%j.log
set -euo pipefail
cd /nobackup/proj/disk/naiss2024-5-630/personal/ribes/TACK
unset PYTHONPATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
.venv/bin/python -m pytest test/test_vhl_weighting.py -q
.venv/bin/python -u scripts/benchmark_vhl_weighting.py \
    --snapshot models/research_v1 \
    --run models/research_repeated_finalists/activity_catboost_baseline_exact \
    --output models/research_vhl_weighting/activity_catboost_weighted --jobs "$SLURM_CPUS_PER_TASK"
