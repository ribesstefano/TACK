#!/bin/bash
#SBATCH --job-name=tack-graph-probe
#SBATCH --account=naiss2026-3-679-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x.%j.log
set -euo pipefail
cd /nobackup/proj/disk/naiss2024-5-630/personal/ribes/TACK
unset PYTHONPATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
export HF_HOME=/nobackup/proj/disk/naiss2024-5-630/personal/ribes/cache/hf
export TACKAI_CACHE=/nobackup/proj/disk/naiss2024-5-630/personal/ribes/cache/TACK
export UV_CACHE_DIR=/nobackup/proj/disk/naiss2024-5-630/personal/ribes/cache/uv
.venv/bin/python -m pytest test/test_graph_surrogate.py -q
.venv/bin/python -u scripts/surrogate_research/benchmark_graph_surrogate.py \
    --snapshot models/research_v1 \
    --output "${GRAPH_ROOT:-models/research_graph_pilot}/dmpnn_probe" \
    --evaluated-splits fold0 source_holdout --policy exact \
    --epochs 8 --width 128 --depth 3 --batch-size 128 --device cpu
