"""Edge cases in development-only benchmark metric reporting."""
import json
import numpy as np

from scripts.benchmark_surrogates import estimator, fit_learner, scores


def test_constant_prediction_has_null_rank_correlation():
    result = scores(np.linspace(0, 1, 100), np.full(100, .37), False)
    assert result["spearman"] is None
    json.dumps(result, allow_nan=False)


def test_constant_target_has_null_rank_correlation():
    result = scores(np.ones(100), np.linspace(0, 1, 100), False)
    assert result["spearman"] is None
    json.dumps(result, allow_nan=False)


def test_single_class_reports_undefined_auc():
    result = scores(np.zeros(10), np.full(10, .2), True)
    assert result["roc_auc"] is None and result["ap"] is None
    assert result["brier"] > 0
    json.dumps(result, allow_nan=False)


def test_current_recipe_zero_density_exponent_gives_uniform_weights():
    class RecordingModel:
        def fit(self, X, y, **kwargs):
            self.weights = kwargs["sample_weight"]
            return self
    model = RecordingModel()
    fit_learner(model, "current_xgb", np.zeros((20, 2)), np.linspace(0, 1, 20),
                {"alpha_w": 0.}, False)
    np.testing.assert_array_equal(model.weights, np.ones(20))


def test_density_exponent_is_not_passed_to_xgboost():
    model = estimator("current_xgb", False, {"alpha_w": .5}, 1, 0)
    assert "alpha_w" not in model.get_params()
