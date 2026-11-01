#!/bin/bash
#SBATCH --job-name=tack-dmax-bounds
#SBATCH --account=naiss2026-3-679-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x.%j.log
set -euo pipefail
cd /nobackup/proj/disk/naiss2024-5-630/personal/ribes/TACK
unset PYTHONPATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
.venv/bin/python -m pytest test/test_study_dmax_bounds.py -q
.venv/bin/python scripts/surrogate_research/study_dmax_bounds.py \
  --runs models/research_exact_current/dmax_current_xgb_baseline_exact \
         models/research_repeated_finalists/dmax_catboost_baseline_exact \
         models/research_repeated_finalists/dmax_extratrees_baseline_exact \
  --output models/research_dmax_bounds_study_v2.json --draws 1000
