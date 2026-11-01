#!/bin/bash
#SBATCH --job-name=tack-conformal-study-tests
#SBATCH --account=naiss2026-3-679-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x.%j.log
set -euo pipefail
cd /nobackup/proj/disk/naiss2024-5-630/personal/ribes/TACK
unset PYTHONPATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
.venv/bin/python -m pytest -q test/test_study_conformal_coverage.py
