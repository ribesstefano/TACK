# Command-Line Interface

For batch inference from a CSV without writing any Python, the `tack
predict` subcommand wraps the same
[`EnsemblePredictor.from_pretrained`][tackai.ensemble_predictor.EnsemblePredictor.from_pretrained] /
[`predict_dataframe`][tackai.ensemble_predictor.EnsemblePredictor.predict_dataframe]
path documented on this site.

```bash
tack predict \
    --repo-id ailab-bio/TACK-ensembles \
    --subfolder dmax_caruana \
    --input-csv compounds.csv \
    --output-csv predictions.csv
```

The input CSV must contain a `SMILES` column. Context columns (`POI_Name`,
`POI_Sequence`, `Ligase_Name`, `Cell_Line_ID`/`Cell_Line`, `Assay_Time`) are
auto-detected when present; missing ones fall back to the same feature
defaults `predict()` uses, with a logged warning.

## Arguments

| Argument | Description | Default |
|---|---|---|
| `--repo-id` | Hugging Face Hub repo id, e.g. `ailab-bio/TACK-ensembles` (mutually exclusive with `--checkpoints-dir`). | — |
| `--subfolder` | Subfolder within `--repo-id` for one ensemble, e.g. `dmax_caruana`. Ignored with `--checkpoints-dir`. | — |
| `--checkpoints-dir` | Local directory with model checkpoints + datamodule states, instead of the Hub. | — |
| `--input-csv` | Input CSV; must contain a SMILES column (**required**). | — |
| `--output-csv` | Where to write predictions (**required**). | — |
| `--weights` | Ensemble weights JSON; restricts inference to the listed models. | auto-discovered from the Hub repo / all models |
| `--device` | `cuda` / `cpu`. | `cuda` if available, else `cpu` |
| `--n-jobs` | Threads for XGBoost inference. | all cores |
| `--smiles-col` | Name of the SMILES column. | `SMILES` |
| `--poi-col`, `--poi-sequence-col`, `--ligase-col`, `--cell-line-col`, `--treatment-time-col` | Context column overrides. | auto-detected |

`--checkpoints-dir` and `--repo-id` are mutually exclusive — pick one.

The output CSV appends `prediction`, `uncertainty`, and percentile/SEM 95%
confidence-interval columns (suffixed per task when the ensemble spans
multiple tasks) — the same columns `predict_dataframe` produces, see
[Making Predictions](making-predictions.md#predicting-over-a-dataframe).

!!! info "Python API vs. CLI"
    This site otherwise documents the Python API (`EnsemblePredictor` and
    friends) rather than the CLI. Reach for `tack predict` for one-off batch
    scoring of a CSV; reach for the Python API when you need per-sample
    control, uncertainty fields beyond what's in the output CSV, or the
    context-reuse fast path from [Fast Screening](screening.md). The
    `tack train`/`tack collect`/`tack evaluate` subcommands cover model
    training and evaluation and are out of scope for this site — see the
    main repository README.
