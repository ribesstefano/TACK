# Fast Screening: Reusing a Fixed Context

When scoring many compounds against the **same** biological context (same
POI, E3 ligase, cell line, treatment time) — e.g. virtual screening a
candidate library against one target — re-encoding the protein embeddings,
cell-line embeddings, and categorical encoders for every single molecule is
wasted work. `EnsemblePredictor` splits featurization into a SMILES-dependent
**molecular** pass and a SMILES-independent **context** pass so the context
can be encoded once and reused.

## 1. Encode the context once

[`transform_context`][tackai.ensemble_predictor.EnsemblePredictor.transform_context]
runs the heavy encoders — protein embeddings, cell-line embeddings,
categorical encoders, treatment-time scaler — a single time, for every
loaded model:

```python
from tackai.ensemble_predictor import SampleInput

context_sample = SampleInput(
    poi_name="AR",
    poi_sequence=AR_SEQ,
    ligase_name="CRBN",
    ligase_sequence=E3_SEQ,
    cell_line="Unknown",
    assay_type="Unknown",
    treatment_time=24.0,
    # smiles is ignored here — only POI/ligase/cell-line/time matter
)

context = predictor.transform_context(context_sample, verbose=True)
```

`transform_context` also accepts a plain dict, using the same column-name
keys described in [Making Predictions](making-predictions.md). It returns a
[`PreprocessedContext`][tackai.ensemble_predictor.PreprocessedContext] —
treat it as an opaque handle to pass back into `predict()`.

## 2. Screen SMILES against it

Pass the `PreprocessedContext` to `predict()` via `context=`, with `samples`
now being a SMILES string or list of SMILES strings instead of
`SampleInput`/dict objects:

```python
smiles_list = [
    "Cn1c(=O)n(C2CCC(=O)NC2=O)c2cccc(C#CCCN3CCC4(CC3)...",
    "Cc1ncsc1C1=CCC([C@H](C)NC(=O)[C@@H]2C[C@@H](O)CN2C(=O)...",
    # ... hundreds or thousands more
]

results, timings = predictor.predict(
    smiles_list,
    context=context,
    verbose=True,
    return_timings=True,
)

for smi, task_dict in zip(smiles_list, results):
    for task, pred in task_dict.items():
        print(f"{smi[:30]:30s}  {task}: {pred.weighted_mean[0]:.3f} ± {pred.uncertainty_std[0]:.3f}")
```

Only the SMILES-dependent features (fingerprints / RDKit descriptors) are
computed per call; the cached context features are merged in directly. The
return shape matches the normal batch path — a list of `{task: EnsemblePrediction}`.

!!! note "A `PreprocessedContext` is tied to the predictor instance that made it"
    `context.predictor_id` is checked against `id(self)` on every `predict()`
    call — calling `predict(context=...)` on a *different* `EnsemblePredictor`
    instance than the one that produced the context raises `ValueError`. Call
    `transform_context` on the same object you'll call `predict` on.

!!! warning "Not for tokenizer-based models"
    `transform_context` raises if a loaded model uses a tokenizer
    (`use_tokenizer=True` on its datamodule) — those models can't split
    context/molecular featurization this way. Use `predict()` directly with
    full `SampleInput`/dict objects instead for such ensembles.

## Why this is faster

Both `predict()` code paths ultimately call the same per-model
`datamodule.transform(...)`, but the context path skips re-fitting/re-running
the categorical encoders, protein-embedding lookups, and treatment-time
scaler on every call — those are looked up once from
`context.context_features[model_name]`. For XGBoost models specifically, a
per-model feature-layout template (`context.xgb_row_template`) is
precomputed too, so only the fingerprint/descriptor slots need to be filled
in per SMILES. See section 8 ("Fast Screening") of the
[tutorial notebook](../tutorial/ensemble_predictor_tutorial.ipynb) for a
runnable end-to-end example with real timings.
