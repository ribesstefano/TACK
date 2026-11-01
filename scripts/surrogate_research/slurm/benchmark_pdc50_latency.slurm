#!/bin/bash
#SBATCH --job-name=tack-pdc50-latency
#SBATCH --account=naiss2026-3-679-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x.%j.log
set -euo pipefail
cd /nobackup/proj/disk/naiss2024-5-630/personal/ribes/TACK
unset PYTHONPATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
.venv/bin/python scripts/surrogate_research/benchmark_candidate_latency.py \
    --snapshot models/research_v1 --bundle models/research_precal/pdc50_r2_v2 \
    --output models/research_latency/pdc50_r2.json --batch 128 --repeats 10
