#!/bin/bash
#SBATCH --job-name=tack-fixed-blend
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
.venv/bin/python -m pytest test/test_fixed_blend.py -q
base=models/research_repeated_finalists
ref=models/research_exact_current_reference
out=models/research_fixed_blends
script=scripts/surrogate_research
.venv/bin/python "$script/benchmark_fixed_blend.py" --output "$out/pdc50_r2_r3" \
  --runs "$base/pdc50_xgb_morgan_r2_2048_bit_exact" "$base/pdc50_xgb_morgan_r3_4096_bit_exact"
.venv/bin/python "$script/benchmark_fixed_blend.py" --output "$out/dmax_all" \
  --runs "$ref/dmax" "$base/dmax_catboost_baseline_exact" "$base/dmax_extratrees_baseline_exact"
.venv/bin/python "$script/benchmark_fixed_blend.py" --output "$out/activity_xgb_catboost" \
  --runs "$ref/activity" "$base/activity_catboost_baseline_exact"
.venv/bin/python "$script/benchmark_fixed_blend.py" --output "$out/activity_xgb_extratrees" \
  --runs "$ref/activity" "$base/activity_extratrees_baseline_exact"
.venv/bin/python "$script/summarize_surrogate_benchmarks.py" --roots "$out" \
  --reference-root "$ref" --draws 1000 --output models/research_fixed_blends_summary.json
for candidate in pdc50_r2_r3 dmax_all activity_xgb_catboost activity_xgb_extratrees; do
  task=${candidate%%_*}
  .venv/bin/python "$script/report_surrogate_slices.py" --candidate "$out/$candidate" \
    --reference "$ref/$task" --output "models/research_fixed_blend_slices/$candidate.json"
done
