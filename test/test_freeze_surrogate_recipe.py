"""Freezing a candidate preserves each fold's selected parameters."""
import json

import pytest

from scripts.surrogate_research.freeze_surrogate_recipe import recipe_from_run


def _run(path):
    path.mkdir()
    (path / "COMPLETE").write_text("complete\n")
    (path / "config.json").write_text(json.dumps({
        "model": "xgb", "task": "pdc50", "feature": "baseline", "policy": "exact",
        "eval_policy": "exact", "ablation": "all", "trials": 20, "repeats": 1, "folds": 2,
        "snapshot_hash": "a" * 64}))
    metrics = [{"split": "repeat0_fold1", "params": {"max_depth": 3}},
               {"split": "source_holdout", "params": {"max_depth": 99}},
               {"split": "repeat0_fold0", "params": {"max_depth": 5}}]
    (path / "metrics.json").write_text(json.dumps(metrics))
    return metrics


def test_freeze_never_uses_source_holdout_tuning(tmp_path):
    run = tmp_path / "run"
    _run(run)
    recipe = recipe_from_run(run)
    assert recipe["cv"]["seeds"] == [1000]
    assert recipe["snapshot_hash"] == "a" * 64
    assert [x["params"]["max_depth"] for x in recipe["members"]] == [5, 3]
    assert [x["seed"] for x in recipe["members"]] == [20261030, 20261031]
    assert recipe_from_run(run, unit="group_max")["calibration"]["unit"] == "group_max"


def test_missing_or_duplicate_fold_is_never_frozen(tmp_path):
    run = tmp_path / "run"
    metrics = _run(run)
    (run / "metrics.json").write_text(json.dumps(metrics[:2]))
    with pytest.raises(ValueError, match="missing"):
        recipe_from_run(run)
    (run / "metrics.json").write_text(json.dumps(metrics + [metrics[0]]))
    with pytest.raises(ValueError, match="duplicate"):
        recipe_from_run(run)
