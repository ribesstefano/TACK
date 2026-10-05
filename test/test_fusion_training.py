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
