#!/bin/bash
#SBATCH --job-name=tack-activity-cal
#SBATCH --account=naiss2026-3-679-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x.%j.log
set -euo pipefail
cd /nobackup/proj/disk/naiss2024-5-630/personal/ribes/TACK
unset PYTHONPATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
case "${1:?expected current_xgb or extratrees}" in
  current_xgb|extratrees) kind=$1 ;;
  *) echo 'Invalid bundle kind' >&2; exit 2 ;;
esac
.venv/bin/python -u scripts/surrogate_research/study_activity_calibration.py \
  --snapshot models/research_v1 --bundle "models/research_precal/activity_$kind" \
  --output "models/research_activity_calibration/$kind" --folds 5 --seed 20261101 --draws 1000
