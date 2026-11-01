"""Interval targets preserve censoring direction and never use hidden rows."""
import numpy as np
import pytest

from scripts.surrogate_research.benchmark_censored_dmax import interval_gradient, read_bound_rows


def test_interval_gradient_matches_exact_and_censored_constraints():
    bounds = np.array([[.4, .4], [.2, .8], [.2, .8], [.2, .8]])
    grad, hess = interval_gradient(np.array([.6, .5, .1, .9]), bounds)
    np.testing.assert_allclose(grad, [.2, 0., -.1, .1])
    assert hess[0] == hess[2] == hess[3] == 1. and 0 < hess[1] < .001
    with pytest.raises(ValueError, match="reversed"):
        interval_gradient(np.array([.5]), np.array([[.8, .2]]))


def test_bound_reader_selects_only_requested_rows(tmp_path):
    values = np.array([[.1, .1], [.2, .8], [99., 100.], [.3, .9]])
    path = tmp_path / "bounds.npz"
    np.savez(path, dmax=values)
    np.testing.assert_array_equal(read_bound_rows(path, "dmax", [3, 0]), values[[3, 0]])
