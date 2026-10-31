#!/bin/bash
#SBATCH --job-name=tack-repeat-finalists
#SBATCH --account=naiss2026-3-679-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-8%3
#SBATCH --output=logs/%x.%A_%a.log
set -euo pipefail
cd /nobackup/proj/disk/naiss2024-5-630/personal/ribes/TACK
unset PYTHONPATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
export HF_HOME=/nobackup/proj/disk/naiss2024-5-630/personal/ribes/cache/hf
export TACKAI_CACHE=/nobackup/proj/disk/naiss2024-5-630/personal/ribes/cache/TACK
export UV_CACHE_DIR=/nobackup/proj/disk/naiss2024-5-630/personal/ribes/cache/uv
# Repeat the corrected comparator on exactly the same shared outer folds. The
# final six entries are hypotheses shortlisted from the first development
# screen; the untouched confirmation cohort is needed to assess selection bias.
tasks=(dmax pdc50 activity dmax pdc50 pdc50 activity dmax activity)
models=(current_xgb current_xgb current_xgb extratrees xgb xgb extratrees catboost catboost)
features=(baseline baseline baseline baseline morgan_r2_2048_bit morgan_r3_4096_bit baseline baseline baseline)
policies=(compat compat compat exact exact exact exact exact exact)
i=$SLURM_ARRAY_TASK_ID
task=${tasks[$i]}
model=${models[$i]}
feature=${features[$i]}
policy=${policies[$i]}
.venv/bin/python -u scripts/benchmark_surrogates.py \
    --snapshot models/research_v1 \
    --output "models/research_repeated_finalists/${task}_${model}_${feature}_${policy}" \
    --task "$task" --model "$model" --feature "$feature" \
    --repeats 5 --folds 5 --trials 20 --jobs "$SLURM_CPUS_PER_TASK" \
    --policy "$policy" --eval-policy exact
