#!/bin/bash
#SBATCH --job-name=tack-mol-summary
#SBATCH --account=naiss2026-3-679-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x.%j.log
set -euo pipefail
cd /nobackup/proj/disk/naiss2024-5-630/personal/ribes/TACK
unset PYTHONPATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
mkdir models/research_molecular_followup_reference
for task in dmax activity; do
  ln -s "../research_molecular_followup/${task}_current_xgb_baseline" \
    "models/research_molecular_followup_reference/$task"
done
.venv/bin/python scripts/surrogate_research/summarize_surrogate_benchmarks.py \
  --roots models/research_molecular_followup --reference-root models/research_molecular_followup_reference \
  --draws 1000 --output models/research_molecular_followup_summary.json
for candidate in dmax_catboost_morgan_r2_2048_bit dmax_xgb_morgan_r3_2048_bit activity_catboost_morgan_r2_2048_bit; do
  task=${candidate%%_*}
  .venv/bin/python scripts/surrogate_research/report_surrogate_slices.py \
    --candidate "models/research_molecular_followup/$candidate" \
    --reference "models/research_molecular_followup_reference/$task" \
    --output "models/research_molecular_followup_slices/$candidate.json"
done
