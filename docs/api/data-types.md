# Data Types

Supporting dataclasses returned by, or passed into,
[`EnsemblePredictor`][tackai.ensemble_predictor.EnsemblePredictor] methods.

## `SampleInput`

The input container for a single sample — see
[Making Predictions](../guide/making-predictions.md#describing-a-sample) for
the mapping between its fields and raw dict/column keys.

::: tackai.ensemble_predictor.SampleInput

## `EnsemblePrediction`

Returned by [`predict()`][tackai.ensemble_predictor.EnsemblePredictor.predict]
(one per requested task). See
[Reading an EnsemblePrediction](../guide/making-predictions.md#reading-an-ensembleprediction)
for a field-by-field walkthrough.

::: tackai.ensemble_predictor.EnsemblePrediction

## `PreprocessedContext`

Returned by [`transform_context()`][tackai.ensemble_predictor.EnsemblePredictor.transform_context];
pass it back into `predict(..., context=...)` — see
[Fast Screening](../guide/screening.md).

::: tackai.ensemble_predictor.PreprocessedContext
