#!/bin/bash
#SBATCH --job-name=tack-pdc50-precal
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
.venv/bin/python -u scripts/surrogate_research/fit_surrogate_candidate.py \
    --snapshot models/research_v1 --recipe models/research_recipes/pdc50_r2_precal.json \
    --output "${PRECAL_OUTPUT:-models/research_precal/pdc50_r2_v2}" --pre-calibration
