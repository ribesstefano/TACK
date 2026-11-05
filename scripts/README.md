# 🤖 TACK Scripts

Standalone entry points around the `tackai` package: embedding generation, model
training, ensemble selection and publishing, inference, figures, and the Slurm
wrappers that drive them on a cluster.

Scripts here are **not** the main CLI. Day-to-day training, prediction and
evaluation go through the `tack` console script (`tack train`, `tack predict`,
`tack collect`, `tack evaluate` — see the repo [README](../README.md) and
[CLAUDE.md](../CLAUDE.md)). What lives in `scripts/` is either a one-off
pipeline stage, a maintainer tool, or an experiment whose configuration is the
script itself.

## 📂 Layout

| Directory | Contents |
|---|---|
| [`boltz/`](boltz/) | Boltz-2 structure prediction + pooled structural embeddings. Has its own [README](boltz/README.md). |
| [`data/`](data/) | Dataset curation and the five-training-set dataset comparison. Has its own [README](data/dataset_comparison_README.md). |
| [`embeddings/`](embeddings/) | Regenerate the cached protein / cell-line / assay context embeddings, and publish them. |
| [`ensembles/`](ensembles/) | Caruana ensemble selection, checkpoint collection, Hub publishing. |
| [`inference/`](inference/) | Batch prediction over a CSV and analysis of the result. |
| [`plotting/`](plotting/) | Publication figures (dataset distributions, evaluation, calibration, CV folds). |
| [`slurm/`](slurm/) | `sbatch` wrappers: job arrays and end-to-end notebook executions. |
| `training/` | Per-fold CV training and calibration for both model families. |
| `clean_notebook_metadata.sh` | Strip outputs/execution counts/metadata from a notebook before committing it. |

## 🔧 Prerequisites

Install the package and activate the environment as described in the repo
[README](../README.md), then set `TACKAI_CACHE` (see "Environment Setup" in
[CLAUDE.md](../CLAUDE.md)). Every command below is run from the repo root.

Scripts that read the dataset or push to the Hub need an `HF_TOKEN` in `.env`;
scripts that touch embeddings need `TACKAI_CACHE` to point at a populated cache.

> **Regenerating a cached embedding table changes the embedding space.** The
> pretrained ensembles were fitted against the published tables — a mismatched
> one gives a feature-shape or silent-accuracy failure on every later
> prediction. **A run of these scripts overwrites the cache**, so check with
> `--verify` (or `--dry-run`, assay only) first when the old space still matters.

## 🎯 `training/` — model training

### `train_fast_models.py` / `calibrate_fast_models.py` (fusion surrogates)

The current fast-surrogate path. Repeated 5×5 CV with per-fold Optuna tuning,
then a separate calibration pass that turns the fold models into a calibrated
`XGBStackedEnsemble`.

```bash
# Train. Resumable: a fold whose model+params JSONs exist is skipped.
python scripts/training/train_fast_models.py \
    --dev-files data/yaochen/AutoTPDplus-PROTAC-training.csv \
    --test-file data/yaochen/PROTAC-test.csv \
    --tasks dmax pdc50 activity \
    --output-dir outputs/fast_models

# Fit stacking weights + conformal/temperature calibration on a held-out CSV,
# then save (locally, or straight to the Hub with --repo-id).
python scripts/training/calibrate_fast_models.py \
    --model-dir outputs/fast_models/dmax --task dmax \
    --test-file data/yaochen/test.csv \
    --output-dir ensembles/dmax --plots-dir plots/dmax
```

Per-fold metrics land in `<output-dir>/fold_metrics.csv`; models are cached as
`<task>_repeat<i>_fold<j>.json` / `.params.json`. Tasks: `dmax`, `pdc50`,
`activity`. CV shape is tunable (`--n-repeats`, `--n-folds`, `--n-trials`,
`--n-inner`, `--seed`). `calibrate_fast_models.py` takes one `--task` at a
time, and `--output-dir` without `--repo-id` saves locally without touching the
network.

### `train_models.py` (legacy)

The original argparse trainer for the MLP / BERT / XGBoost architectures,
superseded by `tack train`'s Hydra composition. It takes **flags, not Hydra
`key=value` overrides**:

```bash
python scripts/training/train_models.py \
    --model_type xgboost --task dc50 --group scaffold \
    --checkpoint_dir ./checkpoints --predictions_dir ./predictions
```

- **Model types:** `mlp`, `xgboost` (hyperparameter tuning supported), `bert`
- **Tasks:** `dmax`, `dc50`, `bin`, `dmax_bin`, `dc50_bin`, `multitask`
- **Splits (`--group`):** `random`, `scaffold` (Bemis-Murcko), `butina` (Tanimoto)
- **Configs:** `--data_config configs/data/<stem>.yaml --model_config configs/model/<stem>.yaml`

Prefer `tack train model=xgboost data=fp task=dc50 group=scaffold` for new runs.

## 🏆 `ensembles/` — selection and publishing

### Caruana's greedy forward selection

`ensemble_comparison.py` is the **driver**; `ensemble_selection.py` is an
importable module of selector classes (`NestedCVSelector`,
`HeldOutTestSelector`, `OutOfFoldSelector`, `ArchitectureLevelSelector`) that it
uses — running the module directly only prints a banner.

```bash
python scripts/ensembles/ensemble_comparison.py \
    --task dmax \
    --prediction_dir ./predictions \
    --output_dir ./ensemble_results \
    --hillclimb_perc 0.2
```

Tasks: `dmax`, `dc50`, `bin`. How it works:

1. **Held-out split** — `--hillclimb_perc` (default 20 %) for selection, the rest for evaluation
2. **Candidate pool** — all CV models (25 folds × feature configs)
3. **Greedy selection** — iteratively add the model that improves the metric
4. **Bagging** — repeated random subsamples for robustness
5. **Evaluation** — scored on the untouched evaluation split

Outputs per task, under `--output_dir`: `ensemble_weights_<task>_*.json` (the
deployable weights), `ensemble_comparison_results_<task>.csv` + `.log`,
`ensemble_uncertainty_metrics_<task>.csv`, and comparison / uncertainty PNGs.

### Packaging an ensemble

```bash
# Copy only the checkpoints a weights file references, plus their
# _hparams.yaml / _state.pt datamodule state, into one self-contained directory.
python scripts/ensembles/collect_ensemble_checkpoints.py \
    --weights ensemble_results/ensemble_weights_dmax_caruana_ensemble.json \
    --checkpoints checkpoints/ \
    --output ensembles/dmax_caruana_ensemble/

# Publish ensembles + shared cache assets. Two phases, so the staged tree can
# be reviewed before anything is uploaded.
python scripts/ensembles/publish_ensemble_hf.py stage  --all
python scripts/ensembles/publish_ensemble_hf.py upload --all
```

`publish_ensemble_hf.py upload` needs `huggingface_hub` and write access to the
`ailab-bio` org (`hf auth login`, or `HF_TOKEN`).

## 🧬 `embeddings/` — context tables

Each script regenerates the npz tables of its context blocks, writing into
`TACKAI_CACHE`. They are the generation half of the
corresponding notebook with the sweep and analysis removed, so the published
tables have a reproducible recipe.

The vocabulary is read from the public [`ailab-bio/AutoTPDplus`](https://huggingface.co/datasets/ailab-bio/AutoTPDplus)
dataset on the Hugging Face Hub, loaded by config name — no token, no local
`data/` folder needed:

```python
protac = load_dataset("ailab-bio/AutoTPDplus", "autotpdplus-protac")
```

Each script loads the `autotpdplus-protac` config (the living PROTAC release,
6,766 rows) and hands the split straight to its reader as a DataFrame. Nothing
is written to or read from a CSV: `build_table` accepts frames as well as paths,
so no file sits between the Hub and the embedding tables.

```bash
# POI / E3 protein embeddings (ESM); runs only the winning (layer, pooling) combos
python scripts/embeddings/generate_protein_embeddings.py

# Cell-line embeddings from Cellosaurus descriptions; vocabulary taken from the
# production FusionData.build_table key resolution, not from a single column
python scripts/embeddings/generate_cell_embeddings.py

# Assay-type embeddings; one table over tackai.fusion.context.ASSAY_VOCAB
python scripts/embeddings/generate_assay_embeddings.py

# Pin a revision for a reproducible regeneration
python scripts/embeddings/generate_assay_embeddings.py --revision <sha>
```

Compute nodes have no reliable outbound network. `datasets` caches the download
under `HF_HOME`, so run any one of the scripts on a login node first to populate
that cache; the job then reuses it without touching the Hub. Export
`HF_DATASETS_OFFLINE=1` in the job so an accidental fetch fails loudly instead
of hanging.

### Running them on a GPU

The encoders default to CPU, so a GPU needs three things *together*: an
allocation, `--nv` on `apptainer exec` to expose the host driver, and
`--device cuda` to tell the script to use it. With any one missing the pass
silently stays on CPU — check with
`python -c "import torch; print(torch.cuda.is_available())"` inside the same
`apptainer exec` if in doubt.

```bash
# Protein (ESM) -- the one that genuinely benefits: ~450 sequences, some >1000 aa
srun -A <account> -p <gpu-partition> --gpus=1 -t 01:00:00 \
  apptainer exec --nv $(bash apptainer/bind_live_repo.sh) --pwd /opt/repo \
    apptainer/training.sif \
    python scripts/embeddings/generate_protein_embeddings.py --device cuda

# Cell lines -- ~360 short descriptions; the GPU saves seconds, CPU is fine
    ... python scripts/embeddings/generate_cell_embeddings.py --device cuda --batch-size 256
```

The assay script has no `--device`: it encodes 78 short sentences, where loading
the model costs more than the forward pass.

No `--bind` for the cache is needed above, for two reasons worth knowing
separately. The *variable* comes from `.env`, which the scripts load themselves
and which `bind_live_repo.sh` mounts into the container (it binds dotfiles — see
its comment; before that it did not, and `get_cache_dir()` silently fell back to
`~/.cache/tackai` on a full `$HOME`). The *path* it names resolves because this
cluster auto-mounts `/proj`; on a site that does not, add
`--bind "$TACKAI_CACHE"` so the path exists inside the container under its own
name. `HF_HOME` is **not** in `.env`, so export it (or add it) or the dataset and
model downloads go to `$HOME`.

### The assay vocabulary

The assay block is a **closed vocabulary**, `tackai.fusion.context.ASSAY_VOCAB`, and
`normalize_assay` folds every raw string into one of its keys. There is one
scheme — the finer "open" one; the 14-category "closed" scheme the notebook
also wrote is gone. Two keys are not readouts but outcomes of the fold:

- **`other`** — a real but uncategorised readout (`"cell-based degradation assay"`).
- **`unknown`** — reserved for a missing value, or text saying the method was
  not recorded (`"not specified"`, `"(method not specified)"`).

Keeping them apart matters: "we measured something this vocabulary has no word
for" and "nobody wrote down what was measured" are different facts, and the
surrogates see them as different vectors.

**To add a category:**

1. add the key to `ASSAY_VOCAB` and its pattern to `ASSAY_RULES`, both in
   [`tackai/fusion/context.py`](../tackai/fusion/context.py) — order matters,
   first match wins, so a compound rule goes above its parts;
2. add three descriptions under that key in `ASSAY_DESCRIPTIONS` in
   `generate_assay_embeddings.py`; the script refuses to start otherwise;
3. rerun the script. The vocabulary size is part of the file name
   (`vocab=open<N>`), so the table lands under a new name. The script warns that
   the cache's `manifest.json` still names the old table until the new generation
   is staged and installed.

Adding a key refits the PCA over one more row, so **every** compressed vector
moves, not only the new one. The full (pre-PCA) vectors of existing keys do not
change, since they depend only on their own descriptions.

A plain run overwrites the cache. The safe modes before that: `--verify`
recomputes and diffs against what is already there, writing nothing, and the
assay script also takes `--dry-run` to report the vocabulary and file names
only. `--verify` agreement is numerical, not bit-exact (~1e-7 on full tables, ~1e-5
after PCA): the sentence-transformer/ESM forward pass is float32 and varies with
library version and thread count. A genuine logic change shows up orders of
magnitude larger.

Two properties worth knowing before enlarging a vocabulary: the E3 table's
`mean_rm_pc2` pooling projects out the block's own leading principal components,
so **adding one sequence rewrites every E3 vector**; and the PCA component count
is chosen from the data, so every regenerated table is a new embedding space.

### Publishing

Each generator records what it wrote in its own part manifest in `TACKAI_CACHE`:
`manifest_protein.json`, `manifest_cell.json` and `manifest_assay.json`. They hold
each table's and PCA side file's sha256, size and dimensions, plus the model,
dataset revision and PCA settings behind each block. `--verify` and `--dry-run`
write none. `generate_protein_embeddings.py --blocks poi` replaces only the `poi`
entries.

```bash
# Aggregate the three part manifests into manifest.json and copy the tables they
# name to scripts/hf_staging/fusion_context/
python scripts/embeddings/publish_fusion_context.py stage

# Review that folder, then upload it
python scripts/embeddings/publish_fusion_context.py upload
```

Staging publishes the generation the part manifests name, even when older PCA
widths are still in the cache. It refuses to write anything if a part manifest is
missing, or if a file it lists is missing or was rewritten after its generator
recorded it; the error names the script to rerun. The cache's own `manifest.json`,
which names the tables `ContextEncoder` reads, is not changed by generating or
staging.

## 🔮 `inference/` — batch prediction and analysis

```bash
# Append Dmax / DC50 / binary-activity predictions as columns to a CSV.
# Accepts either the raw curated column names or the canonical ailab-bio/TACK schema.
python scripts/inference/predict_tack_v2.py \
    --input-csv tack_2.csv \
    --dmax-ensemble-dir ensembles/dmax_caruana_ensemble \
    --dc50-ensemble-dir ensembles/dc50_caruana_ensemble \
    --bin-ensemble-dir ensembles/bin_caruana_ensemble \
    --output-csv tack_v2_with_predictions.csv

# Score that output against ground truth.
python scripts/inference/analyze_predictions.py \
    --predictions-csv tack_v2_with_predictions.csv \
    --output-dir prediction_analysis
```

`analyze_predictions.py` covers regression quality and 95 % CI coverage,
threshold-based active/inactive separation, the binary model against a
threshold-derived label, self-consistency between the binary and regression
heads, how many missing readouts got filled in, and the true-vs-predicted
Dmax/DC50 correlation. Results go to the log, to Markdown tables, to a tidy
`analysis_summary.csv` (`section, task, metric, value`), and to PNGs.

## 📊 `plotting/` — figures

| Script | Figure |
|---|---|
| `plot_data_distributions.py` | pDC50 / Dmax distributions (train vs held-out) and active-inactive stacked bars by POI, E3 and cell line |
| `plot_evaluation_figures.py` | The paper's dataset-distribution and model-evaluation figures, each sub-panel saved individually and combined |
| `plot_calibration_parity.py` | Fast-ensemble test predictions before vs after calibration — parity plots for `dmax`/`pdc50`, a reliability curve for `activity` |
| `plot_folds_distributions.py` | Label distributions across the CV folds a given split strategy produces |

```bash
python scripts/plotting/plot_calibration_parity.py --run-tag autotpd4
```

Activity throughout is DC50 < 100 nM **and** Dmax > 80 %;
$\mathrm{pDC50} = -\log_{10}(\mathrm{DC50\ in\ M}) = 9 - \log_{10}(\mathrm{DC50_{nM}})$.

## 🖥️ `slurm/` — cluster wrappers

The `#SBATCH` headers carry a **Berzelius** account and partition
(`Berzelius-2026-62`, `berzelius` / `berzelius-cpu`). Translate the account,
partition and paths for another cluster before submitting.

**Notebook executions** run a notebook end-to-end and overwrite it in place with
its own filled-in outputs and plots — the equivalent of "Run All":

| Script | Notebook | Resource |
|---|---|---|
| `run_fast_tack.sh` | `fast_tack.ipynb` — context-aware CheMeleon evidential regressor, 5-fold scaffold CV | 1 GPU |
| `run_fast_tack_v6.sh` | `fast_tack_v6.ipynb` — frozen RetroTAC fingerprints, merged assay families, censored Gaussian NN + censor-aware pseudo-Huber XGBoost | 1 GPU |
| `run_fast_tack_v7.sh` | `fast_tack_v7.ipynb` — XGBRanker learning-to-rank on v6 features/folds | CPU |
| `run_fast_tack_v8.sh` | `fast_tack_v8.ipynb` — v6 plus Dmax ≥ 80 % and DC50 ≤ 100 nM binary tasks | CPU |
| `run_fusion_comparison.sh` | `fusion_comparison.ipynb` — 6(+1) fusion methods × 3 tasks × 5×5 scaffold CV, with repeated-measures statistics | CPU |

```bash
sbatch scripts/slurm/run_fast_tack_v8.sh
sbatch scripts/slurm/run_fast_tack_v8.sh --evaluate-only   # redo tables/plots, retrain nothing
sbatch scripts/slurm/run_fast_tack_v7.sh --smoke           # 2 folds, tiny budgets, on a copy
sbatch scripts/slurm/run_fusion_comparison.sh --fast       # 1 repeat × 3 folds

squeue -u "$USER"
tail -f outputs/slurm/<job-name>_<jobid>.out
```

`run_fast_tack_v7.sh` / `v8.sh` run the relevant unit tests first and abort the
job before training if one fails. `run_fusion_comparison.sh` is resumable from
its `results.csv` — a job that hit the wall clock just needs resubmitting.

**Dataset-comparison job arrays** — one array element per (task, dataset) cell,
submitted as two dependent stages (a serial featurization warm-up, then the
array). Use the `_v2` pair, which adds a paper-grouped (DOI) tuning validation
split and a coarse Assay-family remap; v1 is kept so the reference run stays
reproducible:

```bash
bash scripts/slurm/submit_dataset_comparison_v2.sh
```

See [`data/dataset_comparison_README.md`](data/dataset_comparison_README.md) for
the ordered command log of the reference run.

## 🧹 `clean_notebook_metadata.sh`

Strips outputs, execution counts and metadata from a notebook in place (needs
`jq`). Run it before committing a notebook the Slurm wrappers have filled in.

```bash
scripts/clean_notebook_metadata.sh notebooks/fast_tack_v8.ipynb
```
