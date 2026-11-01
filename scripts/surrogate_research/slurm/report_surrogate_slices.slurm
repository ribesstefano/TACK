#!/bin/bash
#SBATCH --job-name=tack-slices
#SBATCH --account=naiss2026-3-679-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x.%j.log
set -euo pipefail
cd /nobackup/proj/disk/naiss2024-5-630/personal/ribes/TACK
unset PYTHONPATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
.venv/bin/python -m pytest test/test_surrogate_slices.py -q
tasks=(dmax pdc50 activity)
candidates=(dmax_extratrees_baseline pdc50_xgb_morgan_r3_4096_bit activity_extratrees_baseline)
for i in 0 1 2; do
    .venv/bin/python scripts/surrogate_research/report_surrogate_slices.py \
        --candidate "models/research_exact_pilot/${candidates[$i]}" \
        --reference "models/research_tuned_baseline/${tasks[$i]}" \
        --output "models/research_slices/${tasks[$i]}.json"
done
