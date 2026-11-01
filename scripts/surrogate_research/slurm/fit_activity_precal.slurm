#!/bin/bash
#SBATCH --job-name=tack-activity-precal
#SBATCH --account=naiss2026-3-679-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/%x.%A_%a.log
set -euo pipefail
cd /nobackup/proj/disk/naiss2024-5-630/personal/ribes/TACK
unset PYTHONPATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
runs=(models/research_exact_current/activity_current_xgb_baseline_exact models/research_repeated_finalists/activity_extratrees_baseline_exact)
names=(current_xgb extratrees)
i=$SLURM_ARRAY_TASK_ID
.venv/bin/python scripts/surrogate_research/freeze_surrogate_recipe.py \
  --run "${runs[$i]}" --output "models/research_recipes/activity_${names[$i]}_precal.json"
.venv/bin/python -u scripts/surrogate_research/fit_surrogate_candidate.py \
  --snapshot models/research_v1 --recipe "models/research_recipes/activity_${names[$i]}_precal.json" \
  --output "models/research_precal/activity_${names[$i]}" --pre-calibration
