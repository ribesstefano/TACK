# Fusion Training Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `FusionEstimator.fit` train and nothing else — no splitting, no hyper-parameter search, no label rescaling, no calibration — and move every data-level decision out to named functions that raise instead of degrading.

**Architecture:** Three moves, in dependency order. First, a new `tackai/fusion/training.py` states the data-level invariants (`check_labels`) and owns the one remaining split (`validation_split`, for early stopping). Second, the units/calibration mapping that is currently written three times collapses into one `FusionEstimator.report()` method, after which target standardisation is deleted outright (the GP learns its own mean). Third, the estimator's `fit`/`_fit_model` contract is cut down to `(X, y, validation)` / `(Z, y, validation)`, `FusionEnsemble.fit` is deleted in favour of `training.fit_member`, and calibration becomes an explicit `FusionEnsemble.calibrate(X, y)` on a held-out set.

**Tech Stack:** Python 3.12.4 (the `.venv` in this repo, despite `CLAUDE.md`'s 3.13 setup line), NumPy, scikit-learn (`LogisticRegression`, `GroupShuffleSplit`), XGBoost, PyTorch (via `tackai.fusion.gp`), pytest.

**Spec:** `docs/superpowers/specs/2026-10-05-fusion-training-simplification-design.md`

## Global Constraints

- Run every pytest invocation with `OMP_NUM_THREADS=1` prefixed, and use the project venv: `OMP_NUM_THREADS=1 .venv/bin/python -m pytest ...`. Without it, `torch` and `xgboost` load two `libomp` copies in this venv and the process segfaults on macOS.
- Baseline, measured 2026-10-05 before any change: 217 tests collected; `OMP_NUM_THREADS=1 .venv/bin/python -m pytest test/ -q --ignore=test/test_ensemble.py` gives **216 passed in 71s**. The one excluded test, `test/test_ensemble.py::test_ensemble_predictor`, fails with `HFValidationError` because it loads `ensembles/dc50_ensemble/`, which is not on disk — pre-existing, unrelated, and not to be "fixed" here. Do not run the suite with `-x`: that test sorts first and would mask everything after it.
- Docstrings follow `CLAUDE.md`: a summary line, then a blank line and `Args:`/`Returns:`/`Raises:` sections for anything non-trivial. Do **not** truncate to one-liners where parameters or return values benefit from documentation.
- `fit` must never split data, search hyper-parameters, rescale labels, or calibrate. This is the whole point of the plan; a step that reintroduces any of these is wrong even if tests pass.
- Do **not** write `fit_ensemble` or any production ensemble-fitting loop. The user is implementing that separately. The only ensemble-building helper added here is test-only, in `test/fusion_fixtures.py`.
- Do **not** edit `notebooks/*.ipynb`. The user will update and re-execute `fusion_surrogates.ipynb` and `ensemble_inference_speed.ipynb` themselves.
- `XGBoostFusion` defaults are `max_depth=5, reg_lambda=20.0` — the configuration the 25-fold comparison selected in 75/75 folds for depth and 43/75 for lambda.
- Existing saved ensembles under `ensembles/fusion_dmax`, `ensembles/fusion_pdc50`, `ensembles/fusion_activity` become unloadable by design. No test reads them (verified); do not add one.

## Review Focus

Five inputs the spec implies but whose handling no task's happy path exercises, most likely to bite first:

1. **A calibration set that is empty, or whose rows RDKit-encoded to all-NaN.** `calibrate` would fit `LogisticRegression` on 0 rows and raise a bare sklearn error. Expected: `calibrate` raises a message naming the calibration set. → pinned in Task 4.
2. **Calling `predict` on a binary GP member that was never calibrated.** The latent score would be returned as if it were a probability, outside [0, 1]. Expected: a `ValueError` naming `FusionEnsemble.calibrate`. → pinned in Task 5.
3. **`fit(X, y, groups)` — the old three-positional call** left in a notebook or a user's script. With `validation` keyword-only this is a `TypeError`; if it were positional, a group vector would be silently unpacked as `(X_val, y_val)`. → pinned in Task 5.
4. **A validation set whose labels are a single class**, handed to `XGBoostFusion` for early stopping. XGBoost's eval metric is undefined and early stopping picks round 1. Expected: `validation_split`'s caller validates it. → pinned in Task 3.
5. **`astype("float32")` on an ensemble whose members carry a calibrator.** `astype` rebuilds the context preprocessor and switches member dtypes; the calibrator is fitted on float64-scale scores and must survive the cast unchanged. → pinned in Task 4.

---

### Task 1: `check_labels` and `validation_split`

The data-level invariants, stated once, in a new module that nothing yet depends on.

**Files:**
- Create: `tackai/fusion/training.py`
- Create: `test/test_fusion_training.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `check_labels(y, task_type: str, what: str = "labels") -> None` — raises `ValueError`.
  - `validation_split(groups, test_size: float = 0.2, random_state: int = 0) -> tuple[np.ndarray, np.ndarray]` — returns `(train_idx, val_idx)` into `groups`.

- [ ] **Step 1: Write the failing tests**

Create `test/test_fusion_training.py`:

```python
"""Everything around a fit that is not the fit itself: invariants and the one split left."""
import numpy as np
import pytest

from tackai.fusion.training import check_labels, validation_split


def test_healthy_labels_pass():
    check_labels(np.array([0.1, 0.5, 0.9]), "regression")
    check_labels(np.array([0.0, 1.0, 1.0, 0.0]), "binary")


def test_empty_labels_raise():
    with pytest.raises(ValueError, match="no training labels"):
        check_labels(np.array([]), "regression", "training labels")


def test_non_finite_labels_raise():
    with pytest.raises(ValueError, match="non-finite"):
        check_labels(np.array([0.1, np.nan, 0.9]), "regression")


def test_constant_target_raises():
    """A constant target used to be absorbed by `y.std() or 1.0`; it is a data error."""
    with pytest.raises(ValueError, match="all 3.5"):
        check_labels(np.full(20, 3.5), "regression")


def test_single_class_binary_labels_raise():
    """The old code degraded to a constant predictor here; nothing can be calibrated."""
    for label in (0.0, 1.0):
        with pytest.raises(ValueError, match=f"all {label}"):
            check_labels(np.full(20, label), "binary")


def test_binary_labels_outside_zero_one_raise():
    with pytest.raises(ValueError, match="must be 0 or 1"):
        check_labels(np.array([0.0, 1.0, 2.0]), "binary")


def test_continuous_labels_are_fine_for_a_regression_task():
    check_labels(np.array([0.0, 1.0, 2.0]), "regression")


def test_validation_split_never_splits_a_group():
    groups = np.repeat(np.arange(10), 5)
    train, val = validation_split(groups, random_state=0)
    assert not (set(groups[train]) & set(groups[val]))
    assert len(train) + len(val) == len(groups)


def test_validation_split_is_reproducible():
    groups = np.repeat(np.arange(10), 5)
    a = validation_split(groups, random_state=3)
    b = validation_split(groups, random_state=3)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_validation_split_refuses_a_single_group():
    """The old code early-stopped on the training set itself here."""
    with pytest.raises(ValueError, match="at least 2 scaffold groups"):
        validation_split(np.zeros(40, dtype=int))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `OMP_NUM_THREADS=1 .venv/bin/python -m pytest test/test_fusion_training.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'tackai.fusion.training'`.

- [ ] **Step 3: Write the implementation**

Create `tackai/fusion/training.py`:

```python
"""Everything around a fit that is not the fit itself.

The estimators in :mod:`tackai.fusion.models` only train: they take the rows they are given,
in the units they are given, with the hyper-parameters they were constructed with. Choosing
those rows -- which fold a member trains on, which of them early-stop it -- and refusing the
ones no model can learn from is this module's job.
"""
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def check_labels(y, task_type: str, what: str = "labels") -> None:
    """Refuse a set of labels no model can learn from.

    Four conditions make a fit meaningless rather than merely hard, and every one of them was
    absorbed silently somewhere in the old code: an empty training fold, non-finite targets, a
    target with a single distinct value, and a binary task whose labels are not 0/1. Note that
    a constant regression target and single-class binary labels are the same condition, so one
    check replaces both of the guards it supersedes.

    Args:
        y: Label values to check.
        task_type: ``"regression"`` or ``"binary"``; only the latter constrains the values.
        what: Noun phrase naming this set in the error message, e.g. ``"training labels"``.

    Raises:
        ValueError: If the labels are empty, non-finite, constant, or -- for a binary task --
            hold a value other than 0 or 1.
    """
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        raise ValueError(f"no {what} to fit on")
    n_bad = int((~np.isfinite(y)).sum())
    if n_bad:
        raise ValueError(f"{what} contain {n_bad} non-finite value(s); drop those rows first")
    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError(f"{what} are all {classes[0]:g}; there is nothing to learn. This is a "
                         "data or split problem, not something a model can absorb")
    if task_type == "binary":
        bad = sorted(set(classes) - {0.0, 1.0})
        if bad:
            raise ValueError(f"binary {what} must be 0 or 1, got {bad}")


def validation_split(groups, test_size: float = 0.2, random_state: int = 0):
    """One scaffold-grouped ``(train, validation)`` index pair for a model that early-stops.

    This is the split that used to hide inside ``XGBoostFusion._fit_cfg``, where too few groups
    made it fall back to early-stopping against the training set -- which is no early stopping
    at all, just a silent 400 rounds.

    Args:
        groups: Scaffold group id per row.
        test_size: Fraction of rows to put in the validation set.
        random_state: Seed for the split.

    Returns:
        ``(train_idx, val_idx)``, index arrays into ``groups``, sharing no group.

    Raises:
        ValueError: If there are fewer than two distinct groups to split.
    """
    groups = np.asarray(groups)
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        raise ValueError(f"an early-stopping split needs at least 2 scaffold groups, got "
                         f"{n_groups}; fit without early stopping or widen the fold")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    return next(splitter.split(np.zeros(len(groups)), groups=groups))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `OMP_NUM_THREADS=1 .venv/bin/python -m pytest test/test_fusion_training.py -q`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add tackai/fusion/training.py test/test_fusion_training.py
git commit -m "feat(fusion): state the training-data invariants in one place

check_labels refuses what the old code absorbed silently: an empty fold, a
constant target, single-class binary labels. validation_split is the grouped
early-stopping split, raising instead of early-stopping on the training set.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: One `report()` method for the reported units

A pure refactor: the score-to-reported-quantity mapping is currently written three times. Collapse it to one method and have all three callers use it. No behaviour change — standardisation and in-fit calibration both still happen. Also renames `platt_` to `calibrator_`, since Task 4's `calibrate` writes it.

**Files:**
- Modify: `tackai/fusion/models.py` — `_fit_calibrator`, `predict`, `_to_original_units`, `_platt`
- Modify: `tackai/fusion/ensemble.py:580-595` (`_folded_scores`), `tackai/fusion/ensemble.py:616-627` (`_member_scores`)
- Test: `test/test_fusion_models.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `FusionEstimator.report(score, std=None) -> tuple[np.ndarray, np.ndarray]`, and the attribute `calibrator_` (a fitted `LogisticRegression`, or `None`), replacing `platt_`.

- [ ] **Step 1: Write the failing test**

Append to `test/test_fusion_models.py`:

```python
@pytest.mark.parametrize("factory", FACTORIES)
def test_report_is_the_single_path_to_reported_units(factory):
    """predict() must agree with report() on the model's own score, for both estimators."""
    X, y, g = synth()
    est = factory(random_state=0).fit(X, y * 100.0, g)
    Z = est.pre_.transform(X)
    score = est._predict_model(est.model_, Z)
    value, std = est.report(score)
    assert np.allclose(value, est.predict(X))
    assert std.shape == value.shape and not std.any()


def test_report_pushes_a_gp_interval_through_the_calibrator():
    X, y, g = synth(binary=True)
    est = fast_gp(task_type="binary", random_state=0).fit(X, y, g)
    Z = est.pre_.transform(X)
    score, score_std = est._predict_model(est.model_, Z, return_std=True)
    value, std = est.report(score, score_std)
    assert ((value >= 0) & (value <= 1)).all() and (std >= 0).all() and (std <= 1).all()
    assert np.allclose(value, est.predict(X))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `OMP_NUM_THREADS=1 .venv/bin/python -m pytest test/test_fusion_models.py -q -k report`
Expected: FAIL with `AttributeError: 'GPInteraction' object has no attribute 'report'`.

- [ ] **Step 3: Add `report` and delete the three copies**

In `tackai/fusion/models.py`, replace `_to_original_units` and `_platt` with:

```python
    def report(self, score, std=None):
        """Map a model score to the reported quantity, with its uncertainty.

        For a regression task the score is already the reported quantity. For a binary task a
        native classifier's score is a probability and only needs clipping, while a latent
        score is pushed through the fitted calibrator -- and so is its interval, as half the
        width of ``[calibrate(score - std), calibrate(score + std)]``.

        Args:
            score: Model score for each row, on the model's own scale.
            std: Optional predictive standard deviation on that same scale.

        Returns:
            ``(value, std)`` in the reported units; ``std`` is zeros when none was given.

        Raises:
            ValueError: If this member needs a calibrator and has none.
        """
        score = np.asarray(score)
        zeros = np.zeros(len(score))
        if self.task_type != "binary":
            value = score * self.y_std_ + self.y_mean_
            return value, (std * self.y_std_ if std is not None else zeros)
        if self.native_binary:
            return np.clip(score, 0.0, 1.0), (std if std is not None else zeros)
        if getattr(self, "calibrator_", None) is None:
            raise ValueError(
                f"{type(self).__name__} is uncalibrated: a latent score is not a probability. "
                "Call FusionEnsemble.calibrate on a held-out set before predicting")
        probability = self._calibrate(score)
        if std is None:
            return probability, zeros
        high, low = self._calibrate(score + std), self._calibrate(score - std)
        return probability, np.abs(high - low) / 2.0

    def _calibrate(self, score) -> np.ndarray:
        """Calibrated probability of the positive class for a latent score."""
        return self.calibrator_.predict_proba(np.asarray(score)[:, None])[:, 1]
```

Rewrite `predict`'s body below the `return_std` guard to use it:

```python
        Z = self.pre_.transform(X)
        if not return_std:
            return self.report(self._predict_model(self.model_, Z))[0]
        score, std = self._predict_model(self.model_, Z, return_std=True)
        return self.report(score, std)
```

In `_fit_calibrator`, rename the two assignments: `self.platt_ = ...` becomes
`self.calibrator_ = ...`, and the single-class branch sets `self.calibrator_ = None` alongside
`self.single_class_`. Keep `single_class_` for now; Task 4 deletes it. Because `report` now
raises when `calibrator_` is `None`, the single-class branch must set a calibrator that
reproduces the old constant behaviour — instead, have `_fit_calibrator`'s single-class branch
fit `LogisticRegression` on a two-row sentinel so the attribute is never `None`:

```python
        classes = np.unique(y.astype(int))
        if len(classes) < 2:
            # Preserved only until Task 4 deletes in-fit calibration: a single-class fold has
            # nothing to calibrate, and check_labels will refuse it outright.
            self.calibrator_ = None
            self.single_class_ = float(classes[0])
            return
        self.single_class_ = None
```

and give `report` a single-class short-circuit immediately before the `calibrator_ is None`
check, to be deleted in Task 4:

```python
        if getattr(self, "single_class_", None) is not None:
            constant = np.full(len(score), self.single_class_)
            return constant, zeros
```

In `tackai/fusion/ensemble.py`, replace the whole body of `_folded_scores` after the
`predict_in_context` call:

```python
        wants_std = return_std and member.supports_std
        result = member.model_.predict_in_context(mol_blocks, fold, return_std=wants_std)
        score, std = result if wants_std else (result, None)
        return member.report(score, std)
```

and the whole body of `_member_scores`:

```python
        if return_std and member.supports_std:
            score, std = member._predict_model(member.model_, Z, return_std=True)
            return member.report(score, std)
        return member.report(member._predict_model(member.model_, Z))
```

- [ ] **Step 4: Run the full fusion suite to verify nothing changed**

Run: `OMP_NUM_THREADS=1 .venv/bin/python -m pytest test/ -q --ignore=test/test_ensemble.py`
Expected: all pass, including the two new `report` tests. This is a pure refactor; a failure
anywhere else means the mapping was not reproduced faithfully.

- [ ] **Step 5: Commit**

```bash
git add tackai/fusion/models.py tackai/fusion/ensemble.py test/test_fusion_models.py
git commit -m "refactor(fusion): one method maps a score to the reported units

predict, _member_scores and _folded_scores each had their own copy of the
standardise-or-calibrate mapping. They now all call FusionEstimator.report.
platt_ becomes calibrator_, which FusionEnsemble.calibrate will write.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Drop target standardisation

With `report` as the single chokepoint, removing standardisation is a small edit in one place. The GP learns its own constant mean (`gp.py:156`, `gp.py:338`) and XGBoost is scale-indifferent, so nothing replaces it. This is the task that makes the saved ensembles stale, so the guard that refuses them lands here too.

**Files:**
- Modify: `tackai/fusion/models.py` — `fit` (drop `y_mean_`/`y_std_`), `report`
- Modify: `tackai/fusion/ensemble.py` — `from_pretrained` / `_check_layout` neighbourhood
- Test: `test/test_fusion_models.py`, `test/test_fusion_ensemble.py`

**Interfaces:**
- Consumes: `FusionEstimator.report` from Task 2.
- Produces: `FusionEstimator` no longer defines `y_mean_` or `y_std_`; `FusionEnsemble._check_target_scaling(members) -> None` raising `ValueError` on a stale member.

- [ ] **Step 1: Write the failing tests**

In `test/test_fusion_models.py`, replace `test_constant_target_does_not_divide_by_zero` entirely
with:

```python
@pytest.mark.parametrize("factory", FACTORIES)
def test_no_target_scaling_is_recorded(factory):
    """The GP learns its own mean and XGBoost is scale-indifferent; nothing rescales y."""
    X, y, g = synth()
    est = factory(random_state=0).fit(X, y, g)
    assert not hasattr(est, "y_mean_") and not hasattr(est, "y_std_")


@pytest.mark.parametrize("factory", FACTORIES)
def test_predictions_are_in_the_units_y_was_given_in(factory):
    X, y, g = synth()
    est = factory(random_state=0).fit(X, y + 500.0, g)
    pred = est.predict(X)
    assert abs(pred.mean() - (y.mean() + 500.0)) < 0.5 * y.std()
```

Add to `test/test_fusion_ensemble.py`:

```python
def test_from_pretrained_refuses_a_member_fitted_with_target_standardisation(data, tmp_path):
    """A stale member would report standardised units: wrong by a factor of sigma, silently."""
    import joblib
    ens = build_ensemble(fast_gp, data, "pdc50", n_members=1, n_folds=3)
    ens.save(tmp_path / "ens")
    member = joblib.load(tmp_path / "ens" / "member_00.joblib")
    member.y_mean_, member.y_std_ = 0.5, 2.0          # as fitted before this change
    joblib.dump(member, tmp_path / "ens" / "member_00.joblib")
    with pytest.raises(ValueError, match="target standardisation"):
        FusionEnsemble.from_pretrained(tmp_path / "ens")
```

Note: `build_ensemble` does not exist until Task 5. Until then write this test against
`FusionEnsemble.fit(fast_gp, data, task="pdc50", n_members=1, n_folds=3)` and change the call in
Task 5 along with the other 25.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `OMP_NUM_THREADS=1 .venv/bin/python -m pytest test/test_fusion_models.py -q -k "no_target_scaling or units_y_was_given"`
Expected: FAIL — `assert not hasattr(est, "y_mean_")` fails, because `fit` still sets it.

- [ ] **Step 3: Remove the scaling and add the guard**

In `tackai/fusion/models.py`, delete these two lines from `fit`:

```python
        self.y_mean_ = float(y.mean())
        self.y_std_ = float(y.std()) or 1.0        # a constant target must not divide by zero
```

and delete the `ys = (y - self.y_mean_) / self.y_std_` line. The hook still has its old
five-parameter signature at this point, so the call in `fit` becomes
`self._fit_model(Z, y, groups, y)` — `y` twice, which looks wrong and is, but it is correct and
temporary: Task 5 cuts the signature down to `(Z, y, validation)`. `XGBoostFusion._fit_model`'s
`target = np.asarray(y_raw, float) if self.task_type == "binary" else ys` now picks the same
array either way and can be reduced to `target = np.asarray(ys, float)` in this task.

In `report`, the regression branch becomes:

```python
        if self.task_type != "binary":
            return score, (std if std is not None else zeros)
```

Update `FusionEstimator`'s class docstring: `_fit_model` and `_predict_model` work "in the
model's own units", not "in standardised-target units".

In `tackai/fusion/ensemble.py`, add next to `_check_layout`:

```python
    @staticmethod
    def _check_target_scaling(members: Sequence) -> None:
        """Refuse members fitted while the estimators still standardised the target.

        Such a member's model predicts in standardised units, and nothing in this version
        multiplies the scale back in -- so its predictions would be wrong by a factor of the
        training fold's standard deviation, with no error to show for it.

        Args:
            members: The loaded members.

        Raises:
            ValueError: If any member carries a fitted target scale.
        """
        stale = [i for i, m in enumerate(members) if hasattr(m, "y_std_")]
        if stale:
            raise ValueError(
                f"member(s) {stale} were fitted with target standardisation, which this "
                "version no longer applies; their predictions would be in standardised units. "
                "Refit the ensemble.")
```

and call it in `from_pretrained`, immediately before `cls._check_layout(...)`:

```python
        cls._check_target_scaling(members)
```

- [ ] **Step 4: Run the full fusion suite**

Run: `OMP_NUM_THREADS=1 .venv/bin/python -m pytest test/ -q --ignore=test/test_ensemble.py`
Expected: all pass. Watch `test_predictions_are_in_original_target_units`, which fits on
`y * 100 + 500`: the GP must now recover a mean of 500 through its own mean parameter from a
`0.0` initialisation (`gp.py:156`). If it fails, initialise that parameter to the training
mean — `params["mean"] = torch.tensor(float(y.mean()), ...)` in `AdditiveProductGP._init_params`
— which is legitimate (initialising a free parameter at the data's own mean is not
preprocessing the labels) and note it in the commit message.

- [ ] **Step 5: Commit**

```bash
git add tackai/fusion/models.py tackai/fusion/ensemble.py test/test_fusion_models.py test/test_fusion_ensemble.py
git commit -m "refactor(fusion): stop standardising the target

The GP learns its own constant mean and XGBoost is scale-indifferent, so
y_mean_/y_std_ bought nothing while putting a rescaling branch in report and
two attributes on every member. Predictions now come back in whatever units y
was given in. from_pretrained refuses members fitted the old way, whose
predictions would otherwise be silently off by a factor of sigma.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: `FusionEnsemble.calibrate`

Calibration on an explicit held-out set, scored through the same path `predict` uses. Added while in-fit calibration still exists, so it can be reviewed on its own; Task 5 removes the in-fit version.

**Files:**
- Modify: `tackai/fusion/ensemble.py` — new method after `set_weights`
- Test: `test/test_fusion_ensemble.py`

**Interfaces:**
- Consumes: `check_labels` from Task 1; `calibrator_` from Task 2.
- Produces: `FusionEnsemble.calibrate(X, y) -> "FusionEnsemble"`.

- [ ] **Step 1: Write the failing tests**

Add to `test/test_fusion_ensemble.py` (import `fast_xgb` from `test_fusion_models` alongside the
existing helpers, and `numpy as np` / `pytest` if not already imported):

```python
def test_calibrate_makes_a_gp_ensemble_report_probabilities(data):
    from sklearn.metrics import roc_auc_score
    idx, X, y, groups = data.task_rows("activity")
    fit_rows, cal_rows = np.arange(0, len(y) - 40), np.arange(len(y) - 40, len(y))
    ens = FusionEnsemble.fit(partial(fast_gp, task_type="binary"), data, task="activity",
                             n_members=2, n_folds=3)
    ens.calibrate(X[cal_rows], y[cal_rows])
    pred = ens.predict_matrix(X[cal_rows])
    assert ((pred.mean >= 0) & (pred.mean <= 1)).all()
    assert all(((v >= 0) & (v <= 1)).all() for v in pred.member_predictions.values())
    assert roc_auc_score(y[cal_rows], pred.mean) > 0.5


def test_calibrate_refuses_a_regression_task(ens):
    idx, X, y, groups = ens.data.task_rows("pdc50")
    with pytest.raises(ValueError, match="binary"):
        ens.calibrate(X[:40], (y[:40] > y.mean()).astype(float))


def test_calibrate_refuses_single_class_labels(data):
    idx, X, y, groups = data.task_rows("activity")
    ens = FusionEnsemble.fit(partial(fast_gp, task_type="binary"), data, task="activity",
                             n_members=1, n_folds=3)
    with pytest.raises(ValueError, match="calibration labels"):
        ens.calibrate(X[:40], np.zeros(40))


def test_calibrate_refuses_an_empty_set(data):
    idx, X, y, groups = data.task_rows("activity")
    ens = FusionEnsemble.fit(partial(fast_gp, task_type="binary"), data, task="activity",
                             n_members=1, n_folds=3)
    with pytest.raises(ValueError, match="calibration labels"):
        ens.calibrate(X[:0], np.zeros(0))


def test_calibrate_refuses_an_all_native_ensemble(data):
    idx, X, y, groups = data.task_rows("activity")
    ens = FusionEnsemble.fit(partial(fast_xgb, task_type="binary"), data, task="activity",
                             n_members=2, n_folds=3)
    with pytest.raises(ValueError, match="nothing to calibrate"):
        ens.calibrate(X[:40], y[:40])


def test_calibrate_scores_through_the_shared_context_path(data):
    """Calibrating through member.pre_ would put the scores on a different scale than predict."""
    idx, X, y, groups = data.task_rows("activity")
    ens = FusionEnsemble.fit(partial(fast_gp, task_type="binary"), data, task="activity",
                             n_members=2, n_folds=3)
    ens.calibrate(X[:60], y[:60])
    direct = []
    for member, Z in zip(ens.members, ens._iter_member_blocks(X[:60])):
        direct.append(member.report(member._predict_model(member.model_, Z))[0])
    pred = ens.predict_matrix(X[:60])
    for name, expected in zip(ens.names, direct):
        assert np.allclose(pred.member_predictions[name], expected)


def test_astype_keeps_the_calibrator(data):
    idx, X, y, groups = data.task_rows("activity")
    ens = FusionEnsemble.fit(partial(fast_gp, task_type="binary", dtype="float64"), data,
                             task="activity", n_members=1, n_folds=3)
    ens.calibrate(X[:60], y[:60])
    before = ens.members[0].calibrator_
    ens.astype("float32")
    assert ens.members[0].calibrator_ is before
    pred = ens.predict_matrix(X[:60])
    assert ((pred.mean >= 0) & (pred.mean <= 1)).all()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `OMP_NUM_THREADS=1 .venv/bin/python -m pytest test/test_fusion_ensemble.py -q -k calibrate`
Expected: FAIL with `AttributeError: 'FusionEnsemble' object has no attribute 'calibrate'`.

- [ ] **Step 3: Implement `calibrate`**

In `tackai/fusion/ensemble.py`, add `from sklearn.linear_model import LogisticRegression` and
`from tackai.fusion.training import check_labels` to the imports, then add after `set_weights`:

```python
    def calibrate(self, X, y) -> "FusionEnsemble":
        """Fit each member's score-to-probability map on rows no member was fitted on.

        A member whose model does not itself output probabilities -- the GP -- scores on a
        latent scale, and only a held-out set can turn that into a probability. The set must be
        held out from every member's training rows: fitted on scores the members have already
        seen, the map comes out overconfident.

        The scores are taken through the same path :meth:`predict` uses, so the map is fitted
        on the scale it will be applied to. Scoring through each member's own preprocessor
        instead would differ whenever :attr:`shared_context` is on, since the shared transform
        is the consensus of the members' own and identical to none of them.

        Args:
            X: Design matrix of the calibration rows, in this ensemble's block layout.
            y: Binary labels (0 or 1) for those rows.

        Returns:
            self

        Raises:
            ValueError: If the task is not binary, if every member outputs probabilities
                natively and there is nothing to calibrate, or if the labels are empty,
                non-finite or hold a single class.
        """
        if TASK_TYPES[self.task] != "binary":
            raise ValueError(f"only a binary task needs calibration; this ensemble predicts "
                             f"{self.task!r}")
        if all(m.native_binary for m in self.members):
            raise ValueError("every member outputs probabilities natively; nothing to calibrate")
        check_labels(y, "binary", "calibration labels")
        labels = np.asarray(y).astype(int)
        X = np.asarray(X)
        for member, Z in zip(self.members, self._iter_member_blocks(X)):
            if member.native_binary:
                continue
            score = member._predict_model(member.model_, Z)
            member.calibrator_ = LogisticRegression(C=1e4).fit(score[:, None], labels)
            member.single_class_ = None       # removed in the next task with in-fit calibration
        return self
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `OMP_NUM_THREADS=1 .venv/bin/python -m pytest test/test_fusion_ensemble.py -q`
Expected: all pass, including the seven new tests.

- [ ] **Step 5: Commit**

```bash
git add tackai/fusion/ensemble.py test/test_fusion_ensemble.py
git commit -m "feat(fusion): calibrate an ensemble on an explicit held-out set

FusionEnsemble.calibrate fits one Platt map per non-native member from that
member's scores on a set the caller held out, scored through the same path
predict uses so the map is fitted on the scale it is applied to. It refuses a
non-binary task, an all-native ensemble, and labels nothing can be fitted on.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: `fit` only trains

The contract change. The hook signature cannot change without its callers, so this task is atomic: `fit`/`_fit_model`, both subclasses, the dead helpers, `fit_member`, the deletion of `FusionEnsemble.fit`, and the test migration all land together.

**Files:**
- Modify: `tackai/fusion/models.py` — `fit`, `_fit_model` contract, `GPInteraction`, `XGBoostFusion`; delete `INNER_FOLDS`, `inner_group_splits`, `slice_blocks`, `_oof_scores`, `_fit_calibrator`, `_after_fit`, `single_class_`, `hyper_`, `__setstate__`, `XGBoostFusion.GRID`, `XGBoostFusion._fit_cfg`
- Modify: `tackai/fusion/training.py` — add `fit_member`
- Modify: `tackai/fusion/ensemble.py:137-186` — delete `fit`
- Modify: `test/fusion_fixtures.py` — add `build_ensemble`
- Modify: `test/test_fusion_models.py`, `test/test_fusion_training.py`, and the seven files holding `FusionEnsemble.fit` call sites
- Test: `test/test_fusion_training.py`

**Interfaces:**
- Consumes: `check_labels`, `validation_split` (Task 1); `report` (Task 2); `calibrate` (Task 4).
- Produces:
  - `FusionEstimator.fit(X, y, *, validation=None) -> FusionEstimator`
  - `FusionEstimator._fit_model(Z, y, validation) -> model` (subclass hook)
  - `FusionEstimator._prepare_validation(validation) -> tuple | None`
  - `training.fit_member(factory, X, y, train=None, *, groups=None, early_stopping=False, random_state=0) -> FusionEstimator`
  - `test.fusion_fixtures.build_ensemble(factory, data, task, n_members=3, n_folds=3, early_stopping=False) -> FusionEnsemble`

- [ ] **Step 1: Write the failing tests for `fit_member` and the new signature**

Append to `test/test_fusion_training.py`:

```python
from functools import partial

from tackai.fusion.models import GPInteraction, XGBoostFusion
from tackai.fusion.training import fit_member
from test_fusion_models import synth      # pytest puts test/ on sys.path


def fast_gp(**kw):
    return GPInteraction(n_restarts=1, n_iter=15, max_hyper_points=60, **kw)


def fast_xgb(**kw):
    return XGBoostFusion(n_estimators=40, **kw)


def test_fit_member_trains_on_the_given_rows_only():
    """Identical to fitting the estimator on those rows by hand, and nothing else."""
    X, y, g = synth(n=120)
    train = np.arange(90)
    est = fit_member(fast_gp, X, y, train, random_state=0)
    direct = fast_gp(random_state=0).fit(X[train], y[train])
    assert np.allclose(est.predict(X[90:]), direct.predict(X[90:]), rtol=1e-8, atol=1e-10)


def test_fit_member_defaults_to_every_row():
    X, y, g = synth()
    est = fit_member(fast_gp, X, y, random_state=0)
    direct = fast_gp(random_state=0).fit(X, y)
    assert np.allclose(est.predict(X), direct.predict(X), rtol=1e-8, atol=1e-10)


def test_fit_member_refuses_bad_training_labels_before_fitting():
    X, _, g = synth()
    y = np.full(len(X), 3.5)
    with pytest.raises(ValueError, match="all 3.5"):
        fit_member(fast_gp, X, y, random_state=0)


def test_fit_member_needs_groups_for_early_stopping():
    X, y, g = synth()
    with pytest.raises(ValueError, match="groups"):
        fit_member(fast_xgb, X, y, early_stopping=True, random_state=0)


def test_fit_member_early_stops_an_xgboost_member():
    X, y, g = synth(n=150)
    est = fit_member(fast_xgb, X, y, groups=g, early_stopping=True, random_state=0)
    assert est.model_.best_iteration is not None       # a validation set was actually used
    assert 1 <= est.n_trees_ <= 40


def test_fit_member_without_early_stopping_uses_every_round():
    """The `validation is None` branch must fit n_estimators rounds, not stop at one."""
    X, y, g = synth(n=150)
    est = fit_member(fast_xgb, X, y, random_state=0)
    assert est.n_trees_ == 40


def test_fit_member_refuses_a_single_class_validation_set():
    """A validation set with one class makes XGBoost's eval metric undefined.

    `fit_member` with no `train` splits `validation_split(g)` for every row, so the same call
    here names exactly the rows it will hold out.
    """
    X, y, g = synth(n=150, binary=True)
    inner, val = validation_split(g, random_state=0)
    y = y.copy()
    y[val] = 1.0                                       # the validation fold loses its negatives
    y[inner[:len(inner) // 2]] = 0.0                   # training rows stay mixed
    y[inner[len(inner) // 2:]] = 1.0
    with pytest.raises(ValueError, match="validation labels"):
        fit_member(partial(fast_xgb, task_type="binary"), X, y, groups=g,
                   early_stopping=True, random_state=0)


def test_fit_is_reproducible_for_a_seed():
    X, y, g = synth()
    a = fit_member(fast_gp, X, y, random_state=3).predict(X)
    b = fit_member(fast_gp, X, y, random_state=3).predict(X)
    assert np.allclose(a, b, rtol=1e-8, atol=1e-10)


def test_fit_refuses_the_old_positional_groups_argument():
    """fit(X, y, groups) was the old call; validation is keyword-only so it cannot be misread."""
    X, y, g = synth()
    with pytest.raises(TypeError):
        fast_gp(random_state=0).fit(X, y, g)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `OMP_NUM_THREADS=1 .venv/bin/python -m pytest test/test_fusion_training.py -q`
Expected: FAIL — `ImportError: cannot import name 'fit_member'`.

- [ ] **Step 3: Rewrite the estimator contract**

In `tackai/fusion/models.py`: delete the imports of `GroupKFold` and `GroupShuffleSplit` and
`LogisticRegression`, delete `INNER_FOLDS`, `inner_group_splits` and `slice_blocks`, and replace
`fit` with:

```python
    def fit(self, X, y, *, validation=None) -> "FusionEstimator":
        """Fit the block preprocessing and the model on the rows given.

        This trains and nothing else. Choosing the rows, scaling the labels, selecting the
        hyper-parameters and calibrating the output all happen before or after this call -- see
        :mod:`tackai.fusion.training` and :meth:`tackai.fusion.ensemble.FusionEnsemble.calibrate`.

        Args:
            X: Design matrix of the rows to train on.
            y: Labels, in the units the model should report.
            validation: Optional ``(X_val, y_val)`` for a model that early-stops on it; a model
                that does not early-stop ignores it.

        Returns:
            self
        """
        self.pre_ = self._make_preprocessor().fit(X)
        self.calibrator_ = None
        self.model_ = self._fit_model(self.pre_.transform(X), np.asarray(y, dtype=float),
                                      self._prepare_validation(validation))
        return self

    def _prepare_validation(self, validation):
        """Processed blocks and labels of an early-stopping set, or None.

        Args:
            validation: ``(X_val, y_val)`` or ``None``.

        Returns:
            ``(Z_val, y_val)`` with the blocks transformed by the fitted preprocessor, or
            ``None``.
        """
        if validation is None:
            return None
        X_val, y_val = validation
        return self.pre_.transform(X_val), np.asarray(y_val, dtype=float)
```

Delete `_fit_calibrator`, `_after_fit`, `_oof_scores`, `__setstate__`, and the `single_class_`
short-circuit added to `report` in Task 2. Change the two hook stubs to:

```python
    def _fit_model(self, Z, y, validation):
        raise NotImplementedError
```

Update the class docstring: subclasses implement `_fit_model(Z, y, validation) -> model` and
`_predict_model(model, Z, return_std=False)`, both in the model's own units.

In `GPInteraction`, delete `_after_fit` and make `_fit_model`:

```python
    def _fit_model(self, Z, y, validation):
        """Fit the GP; ``validation`` is unused, as a GP has no early stopping."""
        gp = AdditiveProductGP(self.pre_.dims_, interactions=self.interactions,
                               ard_blocks=self.ard_blocks, dtype=self.dtype)
        gp.fit(Z, y, n_restarts=self.n_restarts, n_iter=self.n_iter, lr=self.lr,
               seed=self.random_state, max_hyper_points=self.max_hyper_points)
        self.kernel_report_ = gp.kernel_report()
        return gp
```

In `XGBoostFusion`: delete `GRID` and `_fit_cfg`, replace the `grid` constructor parameter with
`max_depth: int = 5, reg_lambda: float = 20.0`, drop `hyper_`, and make:

```python
    def _new(self):
        kw = dict(n_estimators=self.n_estimators, learning_rate=self.learning_rate,
                  max_depth=self.max_depth, reg_lambda=self.reg_lambda, min_child_weight=5,
                  subsample=0.8, colsample_bytree=0.5, gamma=0.1, tree_method="hist",
                  n_jobs=self.n_jobs, random_state=self.random_state, verbosity=0,
                  early_stopping_rounds=30)
        return xgb.XGBClassifier(**kw) if self.task_type == "binary" else xgb.XGBRegressor(**kw)

    def _fit_model(self, Z, y, validation):
        """Fit the trees, early-stopping on ``validation`` when one is given."""
        A = self.pre_.concat(Z)
        model = self._new()
        if validation is None:
            model.set_params(early_stopping_rounds=None)
            model.fit(A, y, verbose=False)
        else:
            Z_val, y_val = validation
            model.fit(A, y, eval_set=[(self.pre_.concat(Z_val), y_val)], verbose=False)
        best = getattr(model, "best_iteration", None)
        self.n_trees_ = int(best) + 1 if best is not None else self.n_estimators
        return model
```

Update both class docstrings: `GPInteraction` no longer mentions `hyper`; `XGBoostFusion`'s
`Args:` documents `max_depth` and `reg_lambda` and drops `grid`.

- [ ] **Step 4: Add `fit_member`**

Append to `tackai/fusion/training.py`:

```python
def fit_member(factory, X, y, train=None, *, groups=None, early_stopping: bool = False,
               random_state: int = 0):
    """Fit one ensemble member on the rows ``train``.

    This is the whole per-member recipe, and it is deliberately short: check the labels, build
    the estimator, optionally carve an early-stopping set out of the training rows, fit. An
    ensemble-level loop over folds calls this once per member.

    Args:
        factory: Callable returning a fresh estimator, called with ``random_state`` alone. Bake
            ``task_type`` and every other setting into the factory, e.g.
            ``partial(GPInteraction, task_type="binary")``.
        X: Full design matrix.
        y: Full label vector, in the units the member should report.
        train: Row indices to train on; defaults to every row.
        groups: Scaffold group id per row. Required when ``early_stopping`` is set, so the
            validation rows share no scaffold with the rows the model fits.
        early_stopping: Hold out a grouped fraction of ``train`` and pass it to the estimator
            as a validation set, for a model that can early-stop on it.
        random_state: Seed, forwarded to the factory and to the validation split.

    Returns:
        The fitted estimator.

    Raises:
        ValueError: If the training labels -- or, with early stopping, the validation labels --
            cannot be learned from, or if ``early_stopping`` is set without ``groups``.
    """
    train = np.arange(len(y)) if train is None else np.asarray(train)
    estimator = factory(random_state=random_state)
    check_labels(y[train], estimator.task_type, "training labels")
    if not early_stopping:
        return estimator.fit(X[train], y[train])
    if groups is None:
        raise ValueError("early_stopping needs groups, so the validation rows share no "
                         "scaffold with the rows the model fits")
    inner, val = validation_split(np.asarray(groups)[train], random_state=random_state)
    check_labels(y[train[val]], estimator.task_type, "validation labels")
    return estimator.fit(X[train[inner]], y[train[inner]],
                         validation=(X[train[val]], y[train[val]]))
```

- [ ] **Step 5: Delete `FusionEnsemble.fit` and add the test-side builder**

Delete the entire `fit` classmethod from `tackai/fusion/ensemble.py` (lines 137-186) along with
its `Callable` import if now unused, and the `TASK_TYPES` import if now unused (it is still
needed by `calibrate`, so keep it).

Append to `test/fusion_fixtures.py`:

```python
def build_ensemble(factory, data, task, n_members=3, n_folds=3, early_stopping=False):
    """Fit one member per fold and wrap them in a FusionEnsemble, for tests.

    Deliberately test-only: the package's own ensemble-fitting entry point is being written
    separately, and this must not pre-empt its name or its behaviour.

    Args:
        factory: Callable returning a fresh estimator, called with ``random_state``.
        data: The :class:`~tackai.fusion.data.FusionData` to draw rows and splits from.
        task: Task name.
        n_members: Number of folds to fit.
        n_folds: Folds per repeat.
        early_stopping: Forwarded to :func:`~tackai.fusion.training.fit_member`.

    Returns:
        A fitted :class:`~tackai.fusion.ensemble.FusionEnsemble`.
    """
    from functools import partial

    from tackai.fusion.data import TASK_TYPES
    from tackai.fusion.ensemble import FusionEnsemble
    from tackai.fusion.training import fit_member

    _, X, y, groups = data.task_rows(task)
    folds = data.splits(task, 1, n_folds)[0]
    make = (lambda k: factory[k]) if isinstance(factory, (list, tuple)) else (lambda k: factory)
    count = len(factory) if isinstance(factory, (list, tuple)) else min(n_members, len(folds))
    members = [fit_member(partial(make(k), task_type=TASK_TYPES[task]), X, y, folds[k][0],
                          groups=groups, early_stopping=early_stopping, random_state=k)
               for k in range(count)]
    return FusionEnsemble(members, data, task)
```

- [ ] **Step 6: Migrate the test call sites**

Replace every `FusionEnsemble.fit(` with `build_ensemble(` across the seven files, adding
`from fusion_fixtures import build_ensemble` to each. The keyword `n_members=`/`n_folds=` survive;
`verbose=`, `n_jobs=` and `splits=` do not and must be dropped where present. The files and
counts:

| File | Call sites |
|---|---|
| `test/test_fusion_ensemble.py` | 18 |
| `test/test_fusion_shared_context.py` | 3 |
| `test/test_fusion_ensemble_astype.py` | 2 |
| `test/test_fusion_ensemble_speed.py` | 1 |
| `test/test_fusion_review_fixes.py` | 2 |
| `test/test_inference_bench.py` | 1 |

Find them with:

```bash
grep -rn "FusionEnsemble.fit" test/
```

Factories that were passed bare (`FusionEnsemble.fit(GPInteraction, ...)`) or as a `partial` keep
working, since `build_ensemble` applies `task_type` itself. A factory passed as a list
(`[fast_gp, fast_xgb]` in `test_fusion_ensemble.py:96` and `test_fusion_ensemble_astype.py:71`)
is handled by `build_ensemble`'s `make`/`count` branch.

- [ ] **Step 7: Update the model tests the contract change invalidates**

In `test/test_fusion_models.py`:
- Every `.fit(X, y, g)` and `.fit(X[tr], y[tr], g[tr])` loses its third argument.
- Delete `test_xgboost_selects_a_grid_configuration` (there is no grid).
- Delete `test_single_class_binary_fold_does_not_crash`; `check_labels` covers it in
  `test/test_fusion_training.py`.
- `fast_xgb` in the module's helpers drops `grid=[{...}]`, becoming
  `XGBoostFusion(n_estimators=40, **kw)`.
- The two binary tests calibrate before predicting. `test_binary_task_returns_probabilities`
  becomes:

```python
@pytest.mark.parametrize("factory", FACTORIES)
def test_binary_task_returns_probabilities(factory):
    from sklearn.metrics import roc_auc_score
    X, y, g = synth(binary=True)
    est = factory(task_type="binary", random_state=0).fit(X, y)
    if not est.native_binary:
        from sklearn.linear_model import LogisticRegression
        score = est._predict_model(est.model_, est.pre_.transform(X))
        est.calibrator_ = LogisticRegression(C=1e4).fit(score[:, None], y.astype(int))
    p = est.predict(X)
    assert ((p >= 0) & (p <= 1)).all()
    assert roc_auc_score(y, p) > 0.7


def test_an_uncalibrated_binary_gp_refuses_to_predict():
    """A latent score is not a probability, and returning it as one would be silent nonsense."""
    X, y, g = synth(binary=True)
    est = fast_gp(task_type="binary", random_state=0).fit(X, y)
    with pytest.raises(ValueError, match="uncalibrated"):
        est.predict(X)
```

- `test_gp_binary_std_is_a_probability_interval` gets the same two-line calibration before
  `est.predict(X, return_std=True)`.
- `test_kernel_report_describes_the_fitted_model_not_an_inner_fold` keeps its assertion but
  drops the comment about inner folds: nothing re-enters `_fit_model` any more.
- Add a test for the "a refit drops a stale calibration" ruling, which nothing else covers:

```python
def test_refitting_clears_a_calibration():
    """A calibrator fitted against the old model must not survive onto a new one."""
    from sklearn.linear_model import LogisticRegression
    X, y, g = synth(binary=True)
    est = fast_gp(task_type="binary", random_state=0).fit(X, y)
    score = est._predict_model(est.model_, est.pre_.transform(X))
    est.calibrator_ = LogisticRegression(C=1e4).fit(score[:, None], y.astype(int))
    est.fit(X, y)
    assert est.calibrator_ is None
```

- [ ] **Step 8: Run the full suite**

Run: `OMP_NUM_THREADS=1 .venv/bin/python -m pytest test/ -q --ignore=test/test_ensemble.py`
Expected: all pass. Re-check `grep -rn "FusionEnsemble.fit\|inner_group_splits\|y_std_\|platt_\|single_class_\|hyper_\|_after_fit" tackai/ test/` returns nothing but the
`_check_target_scaling` guard's own `y_std_` reference.

- [ ] **Step 9: Commit**

```bash
git add tackai/fusion/models.py tackai/fusion/training.py tackai/fusion/ensemble.py test/
git commit -m "refactor(fusion)!: fit only trains

fit(X, y, *, validation=None) fits the preprocessor and the model, and does
nothing else. The subclass hook drops three of its five parameters and its
dual mode: _fit_model(Z, y, validation). Gone with them: the inner CV for
Platt calibration, the inner CV for the XGBoost grid, the _after_fit hook that
existed only to repair what the former overwrote, the single-class and
too-few-groups fallbacks, and __setstate__.

Which rows a member trains on is now training.fit_member's decision, and
calibration is FusionEnsemble.calibrate's. XGBoost takes max_depth=5 and
reg_lambda=20.0 outright -- the configuration the 25-fold comparison chose in
75/75 folds for depth -- so a member costs one fit instead of thirteen.

FusionEnsemble.fit is deleted; tests build ensembles with a test-only helper
pending the package's own entry point.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: Exports and a final sweep

**Files:**
- Modify: `tackai/fusion/__init__.py`
- Modify: `CLAUDE.md` — the `tackai/fusion` architecture notes, if present

**Interfaces:**
- Consumes: everything from Tasks 1-5.
- Produces: `tackai.fusion.check_labels`, `tackai.fusion.validation_split`, `tackai.fusion.fit_member`.

- [ ] **Step 1: Write the failing test**

Append to `test/test_fusion_training.py`:

```python
def test_the_training_helpers_are_exported():
    import tackai.fusion as fusion
    for name in ("check_labels", "validation_split", "fit_member"):
        assert hasattr(fusion, name) and name in fusion.__all__
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `OMP_NUM_THREADS=1 .venv/bin/python -m pytest test/test_fusion_training.py -q -k exported`
Expected: FAIL with `AssertionError`.

- [ ] **Step 3: Update the exports**

In `tackai/fusion/__init__.py`, add after the `models` import:

```python
from tackai.fusion.training import check_labels, fit_member, validation_split
```

and extend `__all__` with `"check_labels", "validation_split", "fit_member"`.

- [ ] **Step 4: Run the whole suite and confirm the count**

Run: `OMP_NUM_THREADS=1 .venv/bin/python -m pytest test/ -q`
Expected: every test passes except the pre-existing `test/test_ensemble.py` failure
(`ensembles/dc50_ensemble/` is not on disk). Record the new total in the commit message.

- [ ] **Step 5: Check the architecture notes**

Run `grep -n "fusion" CLAUDE.md`. If the fusion package is described there, add one line for
`tackai/fusion/training.py` ("invariant checks and per-member fitting; the estimators only
train") and correct anything that describes `fit` as handling splits or calibration. If the
fusion package is not described in `CLAUDE.md`, skip this step rather than adding a section.

- [ ] **Step 6: Commit**

```bash
git add tackai/fusion/__init__.py test/test_fusion_training.py CLAUDE.md
git commit -m "feat(fusion): export the training helpers

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## What this plan deliberately does not do

- **No `fit_ensemble`.** The user is writing the ensemble-level fitting loop. `fit_member` is the
  unit it will call; `build_ensemble` in `test/fusion_fixtures.py` is a test fixture and must not
  grow into production code.
- **No notebook edits.** `fusion_surrogates.ipynb` (cell 10) and `ensemble_inference_speed.ipynb`
  call `FusionEnsemble.fit` and will break. The user will update and re-execute them.
  `fusion_comparison.ipynb` embeds its own copy of the old code and is unaffected.
- **No refit of the saved ensembles.** `ensembles/fusion_{dmax,pdc50,activity}` become unloadable
  by design, and Task 3's guard makes that explicit rather than silent. Refitting them is the
  user's call.
- **No change to the ensemble's other reaches into member internals.** `_predict_model` and
  `pre_` stay as they are; only the units mapping was unified.
