"""One dtype knob on the estimators reaches the preprocessor and the GP."""
import numpy as np
import pytest
import torch

from tackai.fusion.blocks import BLOCK_DIMS, block_index
from tackai.fusion.models import GPInteraction, XGBoostFusion


def design(n=40, seed=0):
    rng = np.random.default_rng(seed)
    idx = block_index(BLOCK_DIMS)
    X = np.zeros((n, sum(BLOCK_DIMS.values())))
    X[:, idx["fingerprint"]] = rng.integers(0, 2, (n, 1024))
    X[:, idx["descriptors"]] = rng.normal(size=(n, 217))
    for b in ("e3", "cell", "poi", "assay"):
        X[:, idx[b]] = rng.normal(size=(n, BLOCK_DIMS[b]))
    X[:, idx["assay_time"]] = rng.choice([12.0, 24.0], (n, 1))
    y = X[:, idx["poi"][0]] + 0.1 * rng.normal(size=n)
    return X, y


def fast_gp(**kw):
    return GPInteraction(n_restarts=1, n_iter=5, max_hyper_points=40, **kw)


def test_estimators_default_to_float32():
    assert GPInteraction().dtype == "float32" and XGBoostFusion().dtype == "float32"


def test_gp_member_is_float32_end_to_end():
    X, y = design()
    est = fast_gp().fit(X, y)
    assert est.pre_.dtype == "float32" and est.model_.dtype is torch.float32
    assert est.predict(X).dtype in (np.float32, np.float64) and np.isfinite(est.predict(X)).all()


def test_gp_member_can_be_float64_end_to_end():
    X, y = design()
    est = fast_gp(dtype="float64").fit(X, y)
    assert est.pre_.dtype == "float64" and est.model_.dtype is torch.float64


def test_xgboost_member_preprocesses_in_the_requested_dtype():
    X, y = design()
    est = XGBoostFusion(n_estimators=10, max_depth=3, reg_lambda=5.0).fit(X, y)
    assert est.pre_.dtype == "float32"
    assert est.pre_.transform(X)["fingerprint"].dtype == np.float32
    assert np.isfinite(est.predict(X)).all()


def test_dtype_survives_sklearn_clone():
    from sklearn.base import clone
    assert clone(fast_gp(dtype="float64")).dtype == "float64"
