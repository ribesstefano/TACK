#!/bin/bash
#SBATCH --job-name=tack-endpoint-activity
#SBATCH --account=naiss2026-3-679-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-2%3
#SBATCH --output=logs/%x.%A_%a.log
set -euo pipefail
cd /nobackup/proj/disk/naiss2024-5-630/personal/ribes/TACK
unset PYTHONPATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
runs=(pdc50_xgb_morgan_r2_2048_bit_exact dmax_catboost_baseline_exact dmax_extratrees_baseline_exact)
names=(pdc50_r2 dmax_catboost dmax_extratrees)
i=$SLURM_ARRAY_TASK_ID
.venv/bin/python -u scripts/surrogate_research/benchmark_endpoint_activity.py predict \
    --snapshot models/research_v1 --run "models/research_repeated_finalists/${runs[$i]}" \
    --output "models/research_endpoint_activity/${names[$i]}"
