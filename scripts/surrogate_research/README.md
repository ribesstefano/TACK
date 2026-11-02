# Surrogate research reproduction

These tools compare pDC50, Dmax and activity models using frozen molecular and biological-context features. Run commands from the TACK repository root. The [progress record](../../docs/surrogate-development-progress.md) describes the experiments, results and limitations; [candidate artifacts](../../docs/surrogate-candidate-artifacts.md) describes fitting and calibration.

## Prerequisites

- Use the repository's `uv.lock` and the `research` extra (`uv sync --extra research`); the graph and neural experiments need the corresponding dependencies and hardware. Set `OMP_NUM_THREADS=1` before starting Python and clear inherited `PYTHONPATH`.
- The frozen `models/research_v1/` snapshot, original input CSVs and context-embedding cache are local artifacts and may not be present in a fresh checkout. The manifest records input and artifact hashes, package versions, the code revision and context fingerprint. Use the recorded inputs, context and environment for comparisons with recorded research runs.
- Slurm recipes in `slurm/` use Arrhenius account, paths and resources hard-coded for this checkout. Adjust copies for your environment. Jobs write logs under `logs/`; outputs and snapshots under `models/` are ignored by Git and can be expensive to regenerate.

## Reproduce the repeated comparison

1. If the original snapshot is available, verify it (the original historical-test CSV path recorded in its manifest must still be accessible):

   ```bash
   env -u PYTHONPATH OMP_NUM_THREADS=1 .venv/bin/python \
     scripts/surrogate_research/verify_surrogate_snapshot.py \
     --snapshot models/research_v1 --report models/research_v1_verification_reproduction.json
   ```

   The original snapshot has a documented frozen/live AvgIpc discrepancy; verification can therefore fail on feature parity. See the progress record before interpreting a failure. Current preparation recomputes features without reusing block caches.

   To create a new snapshot instead, adjust and submit `slurm/prepare_surrogate_research.slurm`. Its defaults read `data/AutoTPD-TPDdb.csv`, `data/auto_tpd_plus/AutoTPDplus-PROTAC.csv`, the historical-test CSV, and `TACKAI_CACHE`. Set `SNAPSHOT_OUTPUT` to a fresh directory. Preparation excludes historical-test molecules and refuses an existing output directory. A newly generated snapshot is a new experiment; compare its manifest with the original before interpreting differences.

2. Submit `slurm/benchmark_repeated_finalists.slurm` and `slurm/benchmark_exact_current.slurm` with `sbatch`. They run 5×5 grouped CV and source holdout with 20 tuning trials per fit. These are long CPU jobs. Each benchmark refuses an existing output directory: use adjusted copies with fresh output roots to rerun without disturbing existing results. Keep all task, feature, label-policy and split settings unchanged for a matched comparison.

3. Once both arrays complete, create a reference directory containing one link per task, then summarize the runs. The example uses the original output paths; substitute the fresh roots if you changed them:

   ```bash
   mkdir models/research_exact_reference_reproduction
   for task in dmax pdc50 activity; do
     ln -s "../research_exact_current/${task}_current_xgb_baseline_exact" \
       "models/research_exact_reference_reproduction/${task}"
   done
   env -u PYTHONPATH OMP_NUM_THREADS=1 .venv/bin/python \
     scripts/surrogate_research/summarize_surrogate_benchmarks.py \
     --roots models/research_repeated_finalists \
     --reference-root models/research_exact_reference_reproduction --draws 1000 \
     --output models/research_exact_current_comparison_reproduction.json
   ```

   Use a fresh reference directory on subsequent runs. `slurm/report_repeated_slices.slurm` produces subgroup reports; set `REFERENCE_ROOT` and `OUTPUT_ROOT` to use the matched exact-label reference and a fresh report directory.

The compatibility-trained and exact-label baselines answer different questions; report which comparator is used. Keep search, stacking, calibration and internal-confirmation groups disjoint. Lock model selection and calibration choices before evaluating confirmation, and do not use the historical test for development. The activity label here is **pDC50 > 6 and Dmax > 0.80**, not the programme prose's 100 nM threshold.
