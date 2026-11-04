# Reproducing the five-training-set dataset comparison

Whiteboard question: does AutoTPD+'s automated LLM-driven PMC extraction make
better PROTAC potency models than PROTAC-DB's own PMC curation, and is any
benefit real or just an artifact of having more rows? Full background: the
docstring of `scripts/train_dataset_comparison.py` and the header markdown of
`notebooks/dataset_comparison_five_sets.ipynb`.

This file is the ordered command log for the reference run below, and doubles
as the recipe for reproducing it from scratch.

## 1. Environment

Standard project setup (see the repo's `CLAUDE.md`):

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env   # set TACKAI_CACHE to a real cache dir (Zenodo cache.zip, or ~/.cache/tackai/)
```

The five raw training CSVs must be present under `data/yaochen/`:
`PROTACDB_basedata.csv`, `AutoTPDplus-PROTAC-training.csv`,
`PROTACDB-training.csv`, `AutoTPD-PROTAC.csv`, `PROTACDB_pmc.csv`. The shared
held-out test set is `data/sets/PROTAC-test.csv` (203 manually curated
compounds, unchanged from the original PROTAC-DB-vs-AutoTPD+ run).

## 2. Sanity-check the data pipeline (seconds, no training)

Confirms every arm's cleaned pool size per task, and that the downsampled
arm's size reference resolves to something smaller than its source pool:

```bash
python - <<'EOF'
import sys; sys.path.insert(0, "scripts")
import train_dataset_comparison as exp

cleaned, test_raw = exp.load_clean_frames()
for name, df in cleaned.items():
    print(f"{name:14s} {len(df):5d} rows")

for task in exp.TASKS:
    sizes = {name: len(exp.prepare_task_pool(df, task, tmp_tag=f"chk_{name}"))
             for name, df in cleaned.items()}
    print(task, sizes)
    for arm, (source, ref) in exp.DOWNSAMPLE_ARMS.items():
        print(f"  {arm}: source={source}({sizes[source]}) -> subsample_n={sizes[ref]}")
EOF
```

## 3. (Optional) Smoke-test one cheap cell before committing to the full array

Exercises the real training path (including Optuna tuning and the per-seed
downsampling code) end to end in under two minutes:

```bash
python scripts/train_dataset_comparison.py \
    --task bin --dataset DEV-PMC-LLM-downsampled \
    --n-seeds 2 --n-tuning-trials 2 --tune \
    --out-dir /tmp/tack_smoke_test --overwrite
```

## 4. Run the full grid (18 cells: 6 arms x 3 tasks x 25 seeds)

**On a Slurm cluster** (the supported path — see
`scripts/slurm/dataset_comparison_array.sh` for why the warm-up stage exists):

```bash
scripts/slurm/submit_dataset_comparison.sh 25
```

**Without Slurm**, a plain sequential loop (safe without a separate warm-up
step since nothing runs concurrently):

```bash
for task in Dmax DC50 bin; do
  for ds in DEV-NO-PMC AutoTPD+ PROTAC-DB DEV-PMC-LLM DEV-PMC DEV-PMC-LLM-downsampled; do
    python scripts/train_dataset_comparison.py \
        --task "$task" --dataset "$ds" --tune --n-seeds 25 --overwrite
  done
done
```

## 5. Monitor

```bash
squeue -u $USER
sacct -j <warmup_jobid>,<array_jobid> --format=JobID,JobName%25,State,ExitCode
tail -f outputs/slurm/tack-dscmp_<array_jobid>_*.out
```

**If a cell times out or is killed:** every seed is checkpointed individually
under `outputs/dataset_comparison/.seed_cache/<task>_<dataset>_<confighash>/`
as soon as it finishes (`train_cell`'s `out_dir`/`resume` args, wired to
`--out-dir` in the CLI). Resubmitting the exact same cell resumes from
whichever seeds already have a checkpoint instead of redoing all 25:

```bash
sbatch --array=<comma_separated_indices> --time=20:00:00 scripts/slurm/dataset_comparison_array.sh 25
```

`--array` on the `sbatch` command line overrides the script's own
`#SBATCH --array=0-17`, so this submits only the named indices; a longer
`--time` than the default 12h buys margin for cells that were making slow
progress rather than actually stuck (check `tail` on the `.out` log for
per-seed timing before assuming a longer budget will help). `--no-resume`
disables this and retrains every seed from scratch; the checkpoint hash also
changes automatically (and old checkpoints are simply never read, though not
deleted) if you edit the feature/model config or tuning budget between runs,
so a stale checkpoint can't silently get reused under a different config.

## 6. Verify / analyze

Open and run top to bottom:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/dataset_comparison_five_sets.ipynb
```

(or open it interactively). Section 1 reports which of the 18 (task, dataset)
cells are present and refuses to proceed with a clear message if any are
missing. The original two-set comparison still works unchanged:
`notebooks/dataset_comparison_simple.ipynb` (now reading the corrected
`PROTAC-DB` source via the same shared `DATASETS` dict).

## 7. Changing a comparison without retraining

`outputs/dataset_comparison/preds_<task>_<dataset>.parquet` +
`.meta.json` are the durable artifacts — one pair per (task, dataset) cell,
written atomically. Any change to which comparisons run, which metric is
primary, or how multiple-comparison correction is applied only touches
`notebooks/dataset_comparison_five_sets.ipynb` (sections 5 onward) or
`scripts/dataset_comparison_stats.py`; re-run the notebook from section 3
onward and it re-reads the same cached predictions. Nothing needs to retrain
unless the underlying data, features, or model config actually change.

---

## Provenance of the reference run (2026-09-23 -- 2026-09-24)

- Data-cleaning decisions made before this run: `PROTAC-DB` repointed from the
  stale `data/sets/PROTACDB-baseline-cleaned-std.csv` to
  `data/yaochen/PROTACDB-training.csv` (see `train_dataset_comparison.py`'s
  `DATASETS` comment); `DEV-PMC`'s 4 exact-duplicate rows (already present in
  `DEV-NO-PMC`) dropped via `DEDUP_AGAINST_BASE`. The old
  `data/sets/PROTACDB-baseline-cleaned-std.csv` was left on disk, unreferenced.
- Slurm jobs: warm-up `17623381` (completed), array `17623382` (indices
  0-17, `berzelius-cpu`, 25 seeds/cell, `--tune`).
- Submitted via `scripts/slurm/submit_dataset_comparison.sh 25` from the repo
  root.
- **Two cells (`Dmax/PROTAC-DB` = index 2, `bin/AutoTPD+` = index 13) hit the
  12h wall-clock limit** rather than a real failure -- per-seed logs showed
  steady progress (~20-23/25 seeds done, 15-60 min/seed, likely CPU contention
  from the other 16 concurrently running elements). This is what motivated
  adding per-seed checkpointing (see section 5 above); at the time these two
  ran, checkpointing didn't exist yet, so the resubmission below retrained all
  25 seeds for each rather than resuming. Resubmitted as job `17627812`
  (`sbatch --array=2,13 --time=20:00:00 scripts/slurm/dataset_comparison_array.sh 25`),
  both completed successfully. Any *future* timeout only needs to redo the
  seeds actually missing.

## 8. Why is test performance so poor? (root-cause investigation, 2026-09-24)

Once all 18 v1 cells were in, pooled test metrics were poor across the board
(Activity ROC-AUC ~ chance, Dmax R2 negative), making the four comparisons
hard to read. Investigation (see the analysis notebook below) found this is a
**test-set problem, not primarily a model problem**: `PROTAC-test.csv`'s 173
Activity rows come from 14 source papers, one of which alone supplies half of
them, so a trivial "predict this target's training active rate" baseline
already scores AUC ~0.40 -- the benchmark's pooled-metric ceiling is set by
cross-paper target-prior mismatch. Two secondary, fixable issues were also
found: the `--tune` validation split was row-random rather than paper-grouped
(94% of validation rows shared a paper with the training split, so Optuna
tuned for same-paper interpolation, not the cross-paper generalization the
test set actually measures), and the free-text `Assay` column's ~50 near-
duplicate spellings meant 60% of test rows (mostly "capillary electrophoresis",
essentially unseen in training) got an all-zero one-hot context vector.

Two follow-ups, keeping the v1 script/notebook/checkpoints/predictions
untouched as a reference:

**Re-score the existing v1 predictions (no retraining)** --
`notebooks/dataset_comparison_analysis_v1.ipynb`. Rescopes the four
comparisons to a paper-size-weighted within-paper metric (a model is only
compared against the SAME paper's other compounds) and a paper-clustered
(not compound-clustered) bootstrap. Reads only `outputs/dataset_comparison/`
(v1); nothing to run beforehand.

**Retrain with both root causes fixed** --
`scripts/train_dataset_comparison_v2.py` (paper-grouped `--tune` split via
`sklearn.model_selection.GroupShuffleSplit`, coarse `assay_family()`
remapping), `scripts/slurm/dataset_comparison_array_v2.sh` /
`submit_dataset_comparison_v2.sh` (18 cells, writing to
`outputs/dataset_comparison_v2/`, same per-seed checkpointing as v1). Submit
with:

```bash
scripts/slurm/submit_dataset_comparison_v2.sh 25
```

Submitted as warm-up `17628470` -> array `17628471` (`berzelius-cpu`, 25
seeds/cell, `--tune`, `-t 16:00:00`). All 18 cells completed successfully with
**zero timeouts** this time (v1 needed one resubmission for 2 cells).

**Comparison notebook**: `notebooks/dataset_comparison_five_sets_v2.ipynb`
(parallel to `dataset_comparison_five_sets.ipynb`, reading
`outputs/dataset_comparison_v2/`), plus a section 8 that diffs v1 vs v2
directly, both pooled and within-paper. Headline result: 3/12 seed-level
tests are now significant (v1: 1/12) and 6/12 survive compound-level
resampling; most notably, `4 > 5` and `4-downsampled > 5` on Activity (PMC-LLM
vs PROTAC-DB's own PMC) are now significant AND support the stated hypothesis
in both cases (p_holm < 0.001), which v1 could not establish. Per-arm deltas
are mixed rather than uniformly positive (expected -- not every fix helps
every arm/task), so read section 8's full table rather than the aggregate
counts alone.
