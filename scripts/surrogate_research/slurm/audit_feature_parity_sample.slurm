#!/bin/bash
#SBATCH --job-name=tack-parity-sample
#SBATCH --account=naiss2026-3-679-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x.%j.log
set -euo pipefail
cd /nobackup/proj/disk/naiss2024-5-630/personal/ribes/TACK
unset PYTHONPATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python -u scripts/surrogate_research/audit_feature_parity_sample.py \
  --snapshot models/research_v1 --candidate models/research_precal/pdc50_r2_v2 \
  --reference models/ensembles/autotpd4_pdc50_refit \
  --count 512 --seed 20261101 --output models/research_latency/feature_parity_sample.json
