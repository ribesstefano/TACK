#!/bin/bash
#SBATCH --job-name=tack-tuned-slices
#SBATCH --account=naiss2026-3-679-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --array=0-5%3
#SBATCH --output=logs/%x.%A_%a.log
set -euo pipefail
cd /nobackup/proj/disk/naiss2024-5-630/personal/ribes/TACK
unset PYTHONPATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
tasks=(dmax dmax pdc50 pdc50 activity activity)
candidates=(dmax_extratrees_baseline dmax_catboost_baseline
    pdc50_xgb_morgan_r2_2048_bit pdc50_xgb_morgan_r3_4096_bit
    activity_extratrees_baseline activity_catboost_baseline)
i=$SLURM_ARRAY_TASK_ID
.venv/bin/python scripts/report_surrogate_slices.py \
    --candidate "models/research_tuned_shortlist/${candidates[$i]}" \
    --reference "models/research_tuned_baseline/${tasks[$i]}" \
    --output "models/research_tuned_slices/${candidates[$i]}.json"
