# Making Predictions

[`EnsemblePredictor.predict`][tackai.ensemble_predictor.EnsemblePredictor.predict]
is the core inference method. It accepts a single sample or a list, runs
every loaded fold-model, denormalizes each model's raw output, and combines
them into a weighted ensemble with uncertainty estimates.

## Describing a sample

Two equivalent input forms are accepted: a [`SampleInput`][tackai.ensemble_predictor.SampleInput]
dataclass, or a plain dict.

=== "SampleInput"

    ```python
    from tackai.ensemble_predictor import SampleInput

    sample = SampleInput(
        smiles="CCO",
        poi_name="BRD4",
        poi_sequence="MSAESGP...",       # full UniProt amino-acid sequence
        ligase_name="VHL",
        ligase_sequence="MPRRAEN...",
        cell_line="HeLa",
        assay_type="HiBiT",
        treatment_time=24.0,
    )
    ```

=== "dict"

    ```python
    sample = {
        "SMILES": "CCO",
        "POI_Name": "BRD4",
        "POI_Sequence": "MSAESGP...",
        "Ligase_Name": "VHL",
        "Ligase_Sequence": "MPRRAEN...",
        "Cell_Line_ID": "HeLa",
        "Assay": "HiBiT",
        "Assay_Time": 24.0,
    }
    ```

With a dict, **keys must match the dataset's column names** (as used to
train the models), not the `SampleInput` field names:

| `SampleInput` field | dict / column key |
|---|---|
| `smiles` | `SMILES` |
| `poi_name` | `POI_Name` |
| `poi_sequence` | `POI_Sequence` |
| `ligase_name` | `Ligase_Name` |
| `ligase_sequence` | `Ligase_Sequence` |
| `cell_line` | `Cell_Line_ID` |
| `assay_type` | `Assay` |
| `treatment_time` | `Assay_Time` |
| `degrader_type` | `Degrader_Type` |

`predict()` also accepts keys matching a loaded datamodule's *actual*
configured column names when they differ from the defaults above (it
normalizes case-insensitively against each model's own `smiles_col`,
`poi_col`, etc. before validating).

## Required vs. optional fields

Which fields are actually required depends on how each model was trained
(e.g. a model using `poi_features: name` only needs `POI_Name`, while one
using `poi_features: sequence` needs `POI_Sequence`). `predict()` inspects
every loaded model's datamodule and:

- **Raises** `ValueError` if a field a model strictly needs (e.g. `SMILES`,
  or `POI_Sequence` for a sequence-based model) is missing.
- **Fills a default and warns** for optional fields the model was trained
  with but that support a sensible fallback:

| Column | Default |
|---|---|
| `Cell_Line_ID` | `"Unknown cell line."` |
| `Assay` | `"Unknown"` |
| `Assay_Time` | `24.0` |
| `Degrader_Type` | `"PROTAC"` |

!!! tip "Only `smiles` is guaranteed optional-safe"
    Every other field's requirement depends on the specific ensemble you
    loaded. When in doubt, pass everything you have — extra fields a model
    doesn't need are simply ignored by that model.

## Single vs. batch

```python
# Single sample -> Dict[task_name, EnsemblePrediction]
task_results = predictor.predict(sample)

# List of samples -> List[Dict[task_name, EnsemblePrediction]]
batch_results = predictor.predict([sample_1, sample_2, sample_3], verbose=True)
```

Useful `predict()` keyword arguments:

| Argument | Default | Purpose |
|---|---|---|
| `tasks` | `None` | Restrict to a subset of tasks (e.g. `tasks=["dmax"]`) when an ensemble spans several. `None` predicts every available task. |
| `return_individual` | `True` | Include each contributing model's raw prediction in the result. |
| `verbose` | `False` | Show a progress bar over models (useful for large ensembles / lazy loading). |
| `return_timings` | `False` | Return `(result, timings)` instead of just `result` — a dict of stage → seconds, useful for profiling. |

## Reading an `EnsemblePrediction`

```python
task_results = predictor.predict(sample, tasks=["dmax"])
result = task_results["dmax"]

print(result.summary())
```

```text
Task: DMAX
Label: Dmax (%)
Number of models: 23
Prediction: 71.4032 ± 4.1187
95% CI (percentile): [61.2044, 78.9501]
95% CI (SEM):        [69.6857, 73.1207]
```

All values are in the **original (denormalized) scale** — each model's
output is inverse-transformed through its own datamodule before the
ensemble average is computed. Key fields on
[`EnsemblePrediction`][tackai.ensemble_predictor.EnsemblePrediction]:

| Field | Meaning |
|---|---|
| `weighted_mean` | Ensemble prediction — weighted average of every contributing model. |
| `uncertainty_std` | Standard deviation across contributing models' predictions. |
| `ci_percentile_lower_95` / `_upper_95` | Non-parametric 95% CI from the 2.5th/97.5th percentiles of individual predictions — "most models predict within this range". |
| `ci_sem_lower_95` / `_upper_95` | Parametric 95% CI from the standard error of the mean — "the true ensemble average is likely in this range"; narrows as more models are added. |
| `prediction_variance`, `prediction_range`, `prediction_iqr` | Additional spread statistics across individual predictions. |
| `predictive_entropy` | Only set for `task == "bin"` — binary predictive entropy of the ensemble mean probability. |
| `individual_predictions` | `{model_name: prediction}` for every contributing model. |
| `weights` | `{model_name: ensemble_weight}` actually used. |
| `model_names` | Names of the fold-models that contributed. |

`result.to_dict()` returns a JSON-serializable dict of all of the above
(numpy arrays converted to lists) — handy for logging or API responses.

## Predicting over a DataFrame

[`predict_dataframe`][tackai.ensemble_predictor.EnsemblePredictor.predict_dataframe]
wraps `predict()` for a whole `pandas.DataFrame` at once and appends
prediction columns:

```python
import pandas as pd

df = pd.read_csv("compounds.csv")  # needs at least a SMILES column

result_df = predictor.predict_dataframe(
    df,
    smiles_col="SMILES",
    poi_col="POI_Name",
    poi_sequence_col="POI_Sequence",
    ligase_col="Ligase_Name",
    cell_line_col="Cell_Line_ID",
    treatment_time_col="Assay_Time",
)
```

For each task the ensemble covers, `result_df` gains `prediction`,
`uncertainty`, `ci_pctl_lower`, `ci_pctl_upper`, `ci_sem_lower`,
`ci_sem_upper` columns (suffixed `_<task>` when the ensemble spans more than
one task).

!!! warning "One bad row fails the whole call"
    `predict_dataframe` builds one `SampleInput` per row and calls
    `predict()` **once** on the full batch. If any row is missing a field a
    loaded model strictly requires, `predict()` raises for the entire
    DataFrame rather than skipping that row — clean your input columns (or
    pre-filter rows) before calling it on a large batch.

!!! note "`assay_type` / `degrader_type` are not DataFrame columns here"
    `predict_dataframe`'s column arguments cover SMILES, POI name/sequence,
    ligase name, cell line, and treatment time only. Assay type and degrader
    type always fall back to their defaults (`"Unknown"` / `"PROTAC"`) in
    this helper. Use `predict()` directly with `SampleInput`/dicts if you
    need to vary those per row.

For predicting many SMILES against one fixed biological context (e.g.
virtual screening), see [Fast Screening](screening.md) instead — it avoids
re-encoding the same protein/cell-line context for every compound.
