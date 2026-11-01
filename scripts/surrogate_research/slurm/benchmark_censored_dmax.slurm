#!/bin/bash
#SBATCH --job-name=tack-censored-dmax
#SBATCH --account=naiss2026-3-679-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x.%j.log
set -euo pipefail
cd /nobackup/proj/disk/naiss2024-5-630/personal/ribes/TACK
unset PYTHONPATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
.venv/bin/python -m pytest test/test_censored_dmax.py -q
.venv/bin/python -u scripts/surrogate_research/benchmark_censored_dmax.py \
    --snapshot models/research_v1 --output models/research_censored_dmax/dmax_xgb_interval_baseline \
    --jobs "$SLURM_CPUS_PER_TASK"
