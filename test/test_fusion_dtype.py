"""Everything runs in float32; there is no precision option left to set."""
import inspect

import numpy as np
import torch
from sklearn.base import clone

from fusion_fixtures import BLOCK_DIMS, DEFAULT_BLOCKS, block_index
from tackai.fusion.gp import GPInteraction


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
    return GPInteraction(**{"blocks": DEFAULT_BLOCKS, **kw}, n_restarts=1, n_iter=5,
                         max_hyper_points=40)


def test_no_estimator_takes_a_dtype():
    assert "dtype" not in inspect.signature(GPInteraction.__init__).parameters


def test_the_gp_is_float32_end_to_end():
    X, y = design()
    est = fast_gp().fit(X, y)
    assert est.model_.dtype is torch.float32
    assert est.model_.Z_train_["poi"].dtype == torch.float32
    assert est.model_.alpha_.dtype == torch.float32
    assert np.isfinite(est.predict(X)).all()


def test_a_float64_design_matrix_is_accepted():
    X, y = design()
    assert X.dtype == np.float64
    assert np.isfinite(fast_gp().fit(X, y).predict(X)).all()


def test_the_gp_estimator_survives_sklearn_clone():
    assert set(clone(fast_gp()).blocks) == set(DEFAULT_BLOCKS)
