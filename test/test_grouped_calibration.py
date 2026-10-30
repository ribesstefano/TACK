"""Group-disjoint lambda selection and calibration-pool splitting."""
import numpy as np
import pytest

import tackai.fusion.xgb_ensemble as stacked
from scripts.calibrate_fast_models import split_fit_cal_by_group
from tackai.fusion.xgb_ensemble import XGBStackedEnsemble


class _Regressor:
    def get_xgb_params(self):
        return {"objective": "reg:squarederror"}

    def predict(self, X):
        return np.asarray(X)[:, 0]


def test_fit_uses_group_disjoint_folds_for_lambda_selection(monkeypatch):
    X = np.arange(24, dtype=float).reshape(-1, 1)
    y = np.sin(X[:, 0])
    groups = np.repeat(np.arange(8), 3)
    ensemble = XGBStackedEnsemble([_Regressor(), _Regressor()], n_folds=4,
                                  n_restarts=1, lambdas=(0.0, 1.0))
    train_groups = []
    held_groups = []

    def optimize(preds, targets, lam, rng):
        train_groups.append(set(groups[preds[:, 0].astype(int)]))
        return np.zeros(4)

    def dummy_loss(params, preds, targets, lam):
        held_groups.append(set(groups[preds[:, 0].astype(int)]))
        return 0.0, np.zeros_like(params)

    monkeypatch.setattr(ensemble, "_optimise", optimize)
    monkeypatch.setitem(stacked.LOSSES, "regression", dummy_loss)
    ensemble.fit(X, y, groups=groups)

    # Every lambda is evaluated on each same fold; the final full-data optimization has 24 rows.
    cv_train = [seen for seen in train_groups if len(seen) < len(np.unique(groups))]
    assert cv_train
    assert len(cv_train) == len(held_groups)
    assert all(train.isdisjoint(held) for train, held in zip(cv_train, held_groups))


def test_fit_accepts_legacy_call_without_groups():
    X = np.arange(12, dtype=float).reshape(-1, 1)
    ensemble = XGBStackedEnsemble([_Regressor()], n_folds=3, n_restarts=1, lambdas=(0.1,))
    assert ensemble.fit(X, X[:, 0]).weights_.shape == (1,)


def test_grouped_calibration_split_keeps_groups_disjoint():
    groups = np.repeat(np.arange(10), 2)
    X = np.arange(20).reshape(-1, 1)
    y = np.tile([0, 1], 10)
    (X_fit, y_fit, fit_groups), (X_cal, y_cal, cal_groups) = split_fit_cal_by_group(
        X, y, groups, seed=31)
    assert len(X_fit) == len(y_fit) == len(fit_groups)
    assert len(X_cal) == len(y_cal) == len(cal_groups)
    assert set(fit_groups).isdisjoint(cal_groups)


def test_grouped_calibration_rejects_missing_or_single_group():
    X = np.arange(4).reshape(-1, 1)
    y = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="missing"):
        split_fit_cal_by_group(X, y, np.array(["a", None, "b", "c"], dtype=object), 0)
    with pytest.raises(ValueError, match="two distinct"):
        split_fit_cal_by_group(X, y, np.array(["a"] * 4), 0)
