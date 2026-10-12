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
        with pytest.raises(ValueError, match=f"all {label:g}"):
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


from fusion_fixtures import DEFAULT_BLOCKS
from tackai.fusion.gp import GPInteraction
from tackai.fusion.training import fit_member
from test_fusion_models import synth      # pytest puts test/ on sys.path


def fast_gp(**kw):
    return GPInteraction(**{"blocks": DEFAULT_BLOCKS, **kw}, n_restarts=1, n_iter=15, max_hyper_points=60)


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


def test_fit_is_reproducible_for_a_seed():
    X, y, g = synth()
    a = fit_member(fast_gp, X, y, random_state=3).predict(X)
    b = fit_member(fast_gp, X, y, random_state=3).predict(X)
    assert np.allclose(a, b, rtol=1e-8, atol=1e-10)



def test_the_training_helpers_are_exported():
    import tackai.fusion as fusion
    for name in ("check_labels", "validation_split", "fit_member"):
        assert hasattr(fusion, name) and name in fusion.__all__


def test_fit_member_does_not_starve_a_model_that_cannot_early_stop():
    """A GP ignores a validation set, so carving one out would only throw rows away."""
    X, y, g = synth(n=150)
    est = fit_member(fast_gp, X, y, groups=g, early_stopping=True, random_state=0)
    assert est.model_.n_train_ == len(y)


