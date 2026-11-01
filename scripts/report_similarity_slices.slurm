#!/bin/bash
#SBATCH --job-name=tack-similarity-slices
#SBATCH --account=naiss2026-3-679-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/%x.%A_%a.log
set -euo pipefail
cd /nobackup/proj/disk/naiss2024-5-630/personal/ribes/TACK
unset PYTHONPATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
tasks=(pdc50 activity)
candidates=(pdc50_xgb_morgan_r2_2048_bit_exact activity_catboost_baseline_exact)
i=$SLURM_ARRAY_TASK_ID
.venv/bin/python -m pytest test/test_similarity_slices.py -q
.venv/bin/python -u scripts/report_similarity_slices.py \
    --candidate "models/research_repeated_finalists/${candidates[$i]}" \
    --reference "models/research_repeated_reference/${tasks[$i]}" \
    --output "models/research_similarity_slices/${candidates[$i]}.json"
