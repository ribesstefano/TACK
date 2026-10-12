# Fusion training: `fit` only trains — design

**Date:** 2026-10-05
**Branch:** `fast-surrogates`
**Status:** awaiting review
**Supersedes parts of:** `2026-10-03-fusion-surrogates-design.md` (estimator contract, calibration,
hyper-parameter selection)

## Goal

Make the fusion training code readable by giving every piece one job.

`FusionEstimator.fit` trains a model on the rows it is given, in the units it is given, with the
hyper-parameters it was constructed with. It does not split data, does not search
hyper-parameters, does not rescale labels and does not calibrate. Everything else moves to a
caller that is honest about doing it, and every degenerate input that used to be absorbed
silently now raises.

## Why

### `fit` does six jobs in fourteen lines

`models.py:86-106` standardises the target, fits the preprocessor, fits the model, searches
hyper-parameters (inside `_fit_model`), fits a calibrator on out-of-fold scores, and then runs a
hook to repair state the calibration fits overwrote. A reader following one `fit` call cannot
tell which of those they are inside.

### Three cross-validations are nested in one call, none named

| Level | Where | Purpose |
|---|---|---|
| outer | `ensemble.py:138` `FusionEnsemble.fit` | one member per fold |
| inner | `models.py:310-316`, inside a nested comprehension | pick XGBoost hyper-parameters |
| inner | `models.py:128-135` `_oof_scores` | out-of-fold scores for Platt calibration |

All three use the same `inner_group_splits` helper, so the reader cannot tell the levels apart
from the call site either.

### The subclass hook has five parameters and two modes

`_fit_model(Z, ys, groups, y_raw, hyper=None)`. `hyper=None` means "search"; `hyper=dict` means
"reuse frozen". `y_raw` exists only because binary XGBoost wants raw labels while the GP wants
standardised ones. `groups` exists only because the hook does its own splitting. Three of the
five parameters are there to serve responsibilities that do not belong to a fit.

### `_after_fit` exists only to repair a re-entrancy bug

`_oof_scores` re-enters `_fit_model`, so `GPInteraction` has to re-record `kernel_report_`
afterwards (`models.py:120`, `models.py:233`). The hook's entire reason to exist is damage caused
by its sibling. `test_kernel_report_describes_the_fitted_model_not_an_inner_fold` is a regression
test for that damage.

### Five degenerate cases are absorbed silently, each producing a quietly wrong model

| Location | Fallback | What it actually produces |
|---|---|---|
| `models.py:33-35` | `< 2` groups → `[(idx, idx)]` | train == test; the "out-of-fold" scores the calibrator is fitted on are in-fold, so the probabilities are overconfident |
| `models.py:110-116` | single class → `single_class_` | a constant predictor wearing a GP's clothes, reporting probability 0.0 or 1.0 for every molecule |
| `models.py:293-297` | `< 2` groups → `eval_set=[(A, target)]` | early stopping against the training set, i.e. no early stopping, i.e. all 400 rounds |
| `models.py:97` | `y.std() or 1.0` | a constant target is accepted and fitted |
| `models.py:67-71` | `__setstate__` dtype default | back-compat for artifacts predating this branch |

Each is a data defect dressed as a model feature. The comment at `models.py:110-113` states the
reasoning explicitly — "which would otherwise kill an ensemble fit after minutes of work" — and
that reasoning is the problem: the cost of a late failure was paid for with a wrong answer. The
fix for a slow failure is a fast one, not a silent one.

### CV lives on the inference class and does not cross-validate

`FusionEnsemble.fit` (`ensemble.py:137-186`) builds splits, flattens repeats, dispatches a
factory per fold, handles `factory` being either a callable or a list, and parallelises with
joblib — then discards every fold's *test* indices. It is a cross-validation harness that never
validates, bolted onto the class whose job is inference. `fusion_surrogates.ipynb` has to define
its own `holdout_splits` to get an honest score.

## Design

### 1. The estimator contract

```python
def fit(self, X, y, *, validation=None) -> "FusionEstimator":
    """Fit the block preprocessing and the model on the rows given.

    Args:
        X: Design matrix of the rows to train on.
        y: Labels, in the units the model should report.
        validation: Optional ``(X_val, y_val)`` for a model that early-stops on it. A model
            that does not early-stop ignores it.

    Returns:
        self
    """
    self.pre_ = BlockPreprocessor(blocks=self.blocks, dtype=self.dtype).fit(X)
    self.calibrator_ = None
    self.model_ = self._fit_model(self.pre_.transform(X), np.asarray(y, dtype=float),
                                  self._prepare_validation(validation))
    return self
```

`validation` is keyword-only so that the old three-positional call
`fit(X[train], y[train], groups[train])` fails loudly with a `TypeError` instead of silently
treating a group vector as a validation set.

`fit` resets `calibrator_` to `None`: a refit invalidates any calibration.

The subclass hook becomes single-purpose:

```python
_fit_model(self, Z, y, validation) -> model
```

`validation` is `None` or `(Z_val, y_val)`, preprocessed by a three-line base-class helper:

```python
def _prepare_validation(self, validation):
    """Processed blocks and labels of an early-stopping set, or None."""
    if validation is None:
        return None
    X_val, y_val = validation
    return self.pre_.transform(X_val), np.asarray(y_val, dtype=float)
```

**Deleted from `models.py`:** `INNER_FOLDS`, `inner_group_splits`, `slice_blocks`, `_oof_scores`,
`_fit_calibrator`, `_after_fit`, `_to_original_units`, `_platt`, `y_mean_`, `y_std_`,
`single_class_`, `hyper_`, `__setstate__`, `XGBoostFusion.GRID`, `XGBoostFusion._fit_cfg`, and the
`grid` and `hyper=` parameters.

**`XGBoostFusion`** takes plain `max_depth=5, reg_lambda=20.0` — the configuration the 25-fold
comparison selected in 75/75 folds for `max_depth` and 43/75 for `reg_lambda`
(`notebooks/fusion_results/results.csv`). Hyper-parameter search, if wanted, happens outside the
package. `_fit_model` early-stops when given a validation set and otherwise fits
`n_estimators` rounds.

**`GPInteraction`** records `kernel_report_` at the end of its own `_fit_model`, where no inner
fit can clobber it.

### 2. No target standardisation

`AdditiveProductGP` already learns a constant mean as a free kernel parameter (`gp.py:156`,
`gp.py:338`), so centring is absorbed by the model. XGBoost is scale-indifferent. All three
targets are already O(1): Dmax σ≈0.3, pDC50 σ≈1.1, activity σ=0.5. Standardisation was
therefore buying nothing while costing two fitted attributes on every member and a
`* y_std_ + y_mean_` branch in three separate places.

Predictions come back in whatever units `y` was in. `ensemble.py:588` and `ensemble.py:621` lose
their rescaling branches.

### 3. One reported-units method, called by both `predict` paths

With standardisation gone, the only score transform left is binary calibration. It is written
once:

```python
def report(self, score, std=None):
    """Map a model score to the reported quantity, with its uncertainty.

    For a regression task the score is already the reported quantity. For a binary task a
    native classifier's score is a probability and only needs clipping, while a latent score
    is pushed through the fitted calibrator — and so is its interval, as half the width of
    ``[calibrate(score - std), calibrate(score + std)]``.

    Args:
        score: Model score for each row.
        std: Optional predictive standard deviation on the score's own scale.

    Returns:
        ``(value, std)``, where ``std`` is zeros when none was given.

    Raises:
        ValueError: If this member needs a calibrator and has none.
    """
```

`FusionEstimator.predict`, `FusionEnsemble._member_scores` and `FusionEnsemble._folded_scores`
all call it, replacing three hand-written copies of the same mapping. A binary non-native member
with `calibrator_ is None` raises
`"member is uncalibrated; call FusionEnsemble.calibrate before predicting"` rather than
returning a latent score dressed as a probability.

This was not part of the approved scope ("unify the units path" was declined), but it falls out:
once standardisation is gone the three copies collapse to one short method, and keeping three
copies of it would be more code than removing them.

### 4. `tackai/fusion/training.py` — three functions, no classes

```python
def check_labels(y, task_type, what="labels") -> None
def validation_split(groups, test_size=0.2, random_state=0) -> tuple[np.ndarray, np.ndarray]
def fit_member(factory, X, y, train=None, *, groups=None, early_stopping=False,
               random_state=0) -> FusionEstimator
```

**`check_labels`** is the one place the data-level invariants are stated. Note that *constant
target* and *single-class binary* are the same condition — `len(np.unique(y)) < 2` — so one check
replaces two of the old guards:

```python
y = np.asarray(y, dtype=float)
if len(y) == 0:                  raise ValueError(f"no {what} to fit on")
if not np.isfinite(y).all():     raise ValueError(f"{what} contain {n} non-finite value(s)")
if len(np.unique(y)) < 2:        raise ValueError(f"{what} are all {value}; nothing to learn")
if task_type == "binary" and not set(np.unique(y)) <= {0.0, 1.0}:
                                 raise ValueError(f"binary {what} must be 0 or 1, got {bad}")
```

**`validation_split`** is the grouped early-stopping split that used to hide in
`XGBoostFusion._fit_cfg`, and it raises on fewer than two groups instead of degrading to
`eval_set=[(A, target)]`.

**`fit_member`** is the whole per-member recipe, readable top to bottom:

```python
train = np.arange(len(y)) if train is None else np.asarray(train)
est = factory(random_state=random_state)
check_labels(y[train], est.task_type, "training labels")
if not early_stopping:
    return est.fit(X[train], y[train])
if groups is None:
    raise ValueError("early_stopping needs groups")
inner, val = validation_split(groups[train], random_state=random_state)
return est.fit(X[train[inner]], y[train[inner]],
               validation=(X[train[val]], y[train[val]]))
```

`factory` is called with `random_state` alone; `task_type` and every other setting is baked into
the factory by the caller. `check_labels` reads `task_type` off the constructed estimator, so the
caller never passes it twice.

### 5. `FusionEnsemble.calibrate`

```python
def calibrate(self, X, y) -> "FusionEnsemble":
    """Fit each member's score-to-probability map on rows no member was fitted on.

    A member whose model does not itself output probabilities scores on a latent scale, and
    only a held-out set can turn that into a probability. The set must be held out from every
    member's training rows: fitted on scores the members have already seen, the map comes out
    overconfident.

    Args:
        X: Design matrix of the calibration rows, in this ensemble's block layout.
        y: Binary labels (0 or 1) for those rows.

    Returns:
        self

    Raises:
        ValueError: If the task is not binary, if every member outputs probabilities natively
            and there is nothing to calibrate, or if ``y`` does not hold both classes.
    """
```

The scores **must** be taken through `self._iter_member_blocks(X)`, the same path `predict` uses.
Scoring through `member.pre_.transform(X)` instead would put the calibration scores on a
slightly different scale from the scores being calibrated whenever `shared_context` is on, since
the shared context preprocessor is the consensus of the members' own and not identical to any of
them.

```python
check_labels(y, "binary", "calibration labels")
targets = [m for m in self.members if not m.native_binary]
if not targets:
    raise ValueError(...)
labels = np.asarray(y).astype(int)
for member, Z in zip(self.members, self._iter_member_blocks(X)):
    if member.native_binary:
        continue
    score = member._predict_model(member.model_, Z)
    member.calibrator_ = LogisticRegression(C=1e4).fit(score[:, None], labels)
return self
```

`FusionEnsemble.fit` is deleted. `__init__`, `from_pretrained`, `save`, `astype`,
`set_weights`, `transform_context`, `predict`, `predict_matrix` stay.

### 6. Refusing stale artifacts

`ensembles/fusion_dmax`, `ensembles/fusion_pdc50` and `ensembles/fusion_activity` hold members
pickled **with** `y_mean_`/`y_std_`. Loaded by the new code, their predictions would come back in
standardised units — no exception, just numbers that are wrong by a factor of σ. `from_pretrained`
therefore raises when a loaded member carries `y_std_`:

```
this ensemble was fitted with target standardisation, which this version no longer applies;
refit it (its predictions would be in standardised units)
```

This is the same class of guard as the existing `_check_layout` and belongs next to it.

## Error policy

Every previously-silent degeneracy, and where it now raises:

| Condition | Old behaviour | New |
|---|---|---|
| constant target | `y.std() or 1.0`, fit proceeds | `check_labels` raises |
| single-class binary labels | `single_class_` constant predictor | `check_labels` raises (same condition as above) |
| binary labels not in {0, 1} | undefined | `check_labels` raises |
| non-finite labels | undefined | `check_labels` raises |
| `< 2` groups for an early-stopping split | early-stops on the training set | `validation_split` raises |
| `< n_splits` groups for inner CV | `[(idx, idx)]`, train == test | n/a — there is no inner CV left |
| binary non-native member, no calibrator | n/a | `report` raises |
| calibrating a non-binary task | n/a | `calibrate` raises |
| member pickled with `y_std_` | n/a | `from_pretrained` raises |
| member pickled without `dtype` | `__setstate__` defaults to float64 | removed; such a member fails on load |

## Out of scope

- **`fit_ensemble`** — the user will implement the ensemble-level fitting loop separately. This
  design leaves `fit_member` as the unit it will call.
- **Notebook updates** — `fusion_surrogates.ipynb` and `ensemble_inference_speed.ipynb` reference
  `FusionEnsemble.fit` and will be updated and re-executed by the user.
  `fusion_comparison.ipynb` embeds its own copy of the old code and is unaffected.
- **The ensemble's remaining reach into member internals** (`_predict_model`, `pre_`) stays;
  only the units mapping is unified, via `report`.
- **`cross_validate`** — nothing calls one today, and the notebook's held-out scoring is three
  lines. Not added.

## Consequences

**Accuracy.** XGBoost members no longer search 4 configurations over 3 inner folds, so a member
costs 1 fit instead of 13 and uses `max_depth=5, reg_lambda=20.0` outright. Dropping target
standardisation re-initialises the GP's kernel-scale parameters against raw units, so Dmax and
pDC50 numbers will shift slightly from the published 5×5 run. This is accepted: the user is
retraining with external hyper-parameter optimisation regardless.

**Artifacts.** The three saved ensembles under `ensembles/` become unloadable by design and must
be refitted.

**Line counts.** `models.py` 330 → ~205. `ensemble.py` 649 → ~590 (`fit` out, `calibrate` in,
three units branches collapsed). `training.py` new at ~110. Net: roughly 95 fewer lines across the
package, with the three CV levels reduced to one that lives outside the estimators.

## Testing

**New** `test/test_fusion_training.py`:
- `check_labels` raises on each of: empty, non-finite, constant, single-class binary, binary
  labels outside {0, 1}; and accepts a healthy regression and a healthy binary vector.
- `validation_split` raises below two groups; otherwise never splits a group across the pair.
- `fit_member` fits on `train` only, is reproducible for a given `random_state`, raises on bad
  training labels before constructing anything expensive, and raises when `early_stopping` is
  set without `groups`.
- `fit_member(early_stopping=True)` produces an XGBoost member whose `n_trees_` is below
  `n_estimators`, proving early stopping actually engaged.

**Inverted** in `test/test_fusion_models.py` — each of these asserted a silent fallback and now
asserts the raise:
- `test_single_class_binary_fold_does_not_crash` → `test_single_class_labels_raise`
- `test_constant_target_does_not_divide_by_zero` → `test_constant_target_raises`

**Deleted**: `test_xgboost_selects_a_grid_configuration` (no grid).
**Simplified**: `test_kernel_report_describes_the_fitted_model_not_an_inner_fold` keeps asserting
the invariant but no longer needs the inner-fit scenario, since nothing re-enters `_fit_model`.
**Changed**: the two binary tests (`test_binary_task_returns_probabilities`,
`test_gp_binary_std_is_a_probability_interval`) now calibrate explicitly before predicting, which
is the real path a binary member takes.

**At risk** — `test_predictions_are_in_original_target_units` fits on `y * 100 + 500`. Today the
target is standardised first, so the GP always sees O(1) values; without standardisation it must
recover a mean of 500 through its own learnable mean parameter and fit a kernel scale of ~100
from a log-space initialisation. The test may need a tolerance adjustment, or the GP may need its
mean parameter initialised to `y.mean()` instead of 0.0 (`gp.py:156`) — a one-line change that is
legitimate, since initialising a free parameter at the data's own mean is not preprocessing the
labels. Decide this when the test is run, not before.

**New** in `test/test_fusion_ensemble.py`: `calibrate` raises on a non-binary task, on an
all-native ensemble and on single-class labels; a calibrated GP ensemble returns probabilities in
[0, 1] with AUC above chance; and `from_pretrained` raises on a member carrying `y_std_`.

**Test-side ensemble builder.** With `FusionEnsemble.fit` deleted and `fit_ensemble` out of scope,
the ~25 existing call sites across `test_fusion_ensemble.py`, `test_fusion_ensemble_astype.py`,
`test_fusion_shared_context.py`, `test_fusion_ensemble_speed.py`, `test_fusion_review_fixes.py`,
`test_fusion_dtype.py` and `test_inference_bench.py` need a replacement. A helper goes in
`test/fusion_fixtures.py` (already imported by those modules):

```python
def build_ensemble(factory, data, task, n_members=3, n_folds=3, early_stopping=False):
    """Fit one member per fold, pending the package's own ensemble-fitting entry point."""
```

It loops `fit_member` over `data.splits(task, 1, n_folds)[0]` and constructs
`FusionEnsemble(members, data, task)`. Deliberately test-only: it must not pre-empt the
`fit_ensemble` the user intends to write.

## Rulings made on the user's behalf

- `validation` is keyword-only in `fit`, so the old positional `groups` argument raises a
  `TypeError` rather than being misread. Cost if wrong: a one-word signature change.
- `factory` in `fit_member` is called with `random_state` alone, with `task_type` baked in by the
  caller. Cost if wrong: callers pass `partial(GPInteraction, task_type=...)` instead of relying
  on `fit_member` to forward it.
- `calibrator_` stores a fitted `LogisticRegression` directly rather than a new wrapper class.
  Cost if wrong: a 10-line class later.
- `report` is public (no leading underscore) because `FusionEnsemble` calls it across the class
  boundary. Cost if wrong: a rename.
- `fit` resets `calibrator_` to `None`, so a refit drops a calibration rather than keeping a stale
  one. Cost if wrong: nothing; this is the safe direction.
- `check_labels` treats a constant regression target and single-class binary labels as one
  condition. Cost if wrong: two separate messages instead of one.
- Early stopping stays available (via `fit_member(early_stopping=True)`) rather than being dropped
  with the grid search, because removing it would silently cost M7 accuracy by fitting all 400
  rounds. Cost if wrong: delete `validation_split` and the `validation` parameter.
