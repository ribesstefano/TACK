#!/bin/bash
#SBATCH --job-name=tack-pdc50-conformal-study
#SBATCH --account=naiss2026-3-679-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.%j.log
set -euo pipefail
cd /nobackup/proj/disk/naiss2024-5-630/personal/ribes/TACK
unset PYTHONPATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
.venv/bin/python -u scripts/study_conformal_coverage.py \
    --snapshot models/research_v1 --bundle models/research_precal/pdc50_r2_v2 \
    --output models/research_conformal_study/pdc50_r2 --folds 5 --alpha .1 --bootstrap 1000
