#!/bin/bash
#SBATCH --account=berzelius-2026-299
#SBATCH --partition=berzelius-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=tack-fast
#SBATCH --output=outputs/slurm/%x_%j.out
#SBATCH --error=outputs/slurm/%x_%j.err
#
# Submit from the repo root; override account/partition for your cluster:
#   mkdir -p outputs/slurm
#   sbatch --account=ACCOUNT --partition=CPU_PARTITION \
#     scripts/training/train_fast_models.slurm dmax \
#     --dataset-revision DATA_SHA --context-revision CONTEXT_SHA
# Repeat for pdc50/activity. Additional CLI arguments override these defaults.
# Warm the HF cache on a networked node first (REPRODUCING.md).
# The default is one 5-fold repeat, 20 trials/fold, grouped inner CV with
# early stopping, uniform averaging, and local ensemble export. PUSH_TO_HUB
# optionally publishes private task subfolders. OUTPUT_DIR selects a fresh run.

set -euo pipefail

TASK="${1:-${TASK:-}}"
case "${TASK}" in
    dmax|pdc50|activity) shift || true ;;
    all) shift || true ;;
    *) echo "Usage: sbatch [-J tack-fast-<task>] $0 {dmax|pdc50|activity} [extra args...]" >&2
       exit 2 ;;
esac
TASKS=("$TASK")
if [[ "$TASK" == all ]]; then TASKS=(dmax pdc50 activity); fi
GPU_FLAGS=()
if [[ "${SLURM_JOB_GPUS:-}" != "" ]]; then GPU_FLAGS=(--nv); fi

# SLURM runs a spooled copy of this script, so ${BASH_SOURCE[0]} does not point back into the
# repo -- SLURM_SUBMIT_DIR is wherever `sbatch` was invoked from. The fallback keeps a direct
# `bash scripts/training/train_fast_models.slurm <task>` working outside SLURM.
BASE_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${BASE_DIR}"
mkdir -p outputs/slurm

# A host PYTHONPATH (an EasyBuild module, a stray site-packages) is inherited by the
# container and shadows /opt/venv's numpy. Drop it rather than debugging the import later.
unset PYTHONPATH

echo "Task: ${TASK}  Node: $(hostname)  CPUs: ${SLURM_CPUS_PER_TASK:-$(nproc)}"

# Cache paths are data mounts, outside the bind-mounted checkout. Load credentials
# without printing them; Apptainer inherits the exported environment.
set -a
if [[ -f .env ]]; then source .env; fi
set +a
export HF_HOME="${HF_HOME:-${BASE_DIR}/../cache/hf}"
export TACKAI_CACHE="${TACKAI_CACHE:-${BASE_DIR}/../cache/TACK}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
mkdir -p "$HF_HOME" "$TACKAI_CACHE"
apptainer exec "${GPU_FLAGS[@]}" $(bash apptainer/bind_live_repo.sh) --pwd /opt/repo \
    --bind "$HF_HOME:$HF_HOME" --bind "$TACKAI_CACHE:$TACKAI_CACHE" \
    --env OMP_NUM_THREADS=1 \
    apptainer/training.sif \
    python scripts/training/train_fast_models.py \
        --tasks "${TASKS[@]}" \
        --output-dir "${OUTPUT_DIR:-outputs/fast_models/${TASK}}" \
        --fp-size "${FP_SIZE:-2024}" \
        --fp-radius "${FP_RADIUS:-8}" \
        --featurize-workers "${FEATURIZE_WORKERS:-8}" \
        --test-frac "${TEST_FRAC:-0.1}" \
        --butina-cutoff "${BUTINA_CUTOFF:-0.65}" \
        --butina-seed "${BUTINA_SEED:-42}" \
        --calibration-frac "${CALIBRATION_FRAC:-0.1}" \
        --n-repeats "${N_REPEATS:-1}" \
        --n-folds "${N_FOLDS:-5}" \
        --n-trials "${N_TRIALS:-20}" \
        ${PUSH_TO_HUB:+--push-to-hub "${PUSH_TO_HUB}"} \
        "$@"
