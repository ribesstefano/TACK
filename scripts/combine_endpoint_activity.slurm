#!/bin/bash
#SBATCH --job-name=tack-endpoint-combine
#SBATCH --account=naiss2026-3-679-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/%x.%A_%a.log
set -euo pipefail
cd /nobackup/proj/disk/naiss2024-5-630/personal/ribes/TACK
unset PYTHONPATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
names=(dmax_catboost dmax_extratrees)
name=${names[$SLURM_ARRAY_TASK_ID]}
.venv/bin/python scripts/benchmark_endpoint_activity.py combine \
    --dmax "models/research_endpoint_activity/${name}" \
    --pdc50 models/research_endpoint_activity/pdc50_r2 \
    --output "models/research_endpoint_activity_combined/${name}"
