"""Edge cases in development-only benchmark metric reporting."""
import json
import numpy as np

from scripts.benchmark_surrogates import scores


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
