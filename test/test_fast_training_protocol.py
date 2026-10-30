"""Protocol invariants for the fast XGBoost training entry point."""
import argparse
import sys

import numpy as np
import pytest

from scripts import train_fast_models as training


@pytest.mark.parametrize("task,y", [
    ("activity", np.tile([0, 1], 20)),
    ("dmax", np.linspace(0.0, 1.0, 40)),
    ("pdc50", np.linspace(4.0, 9.0, 40)),
])
def test_inner_folds_keep_groups_disjoint_and_cover_rows(task, y):
    groups = np.repeat(np.arange(10), 4)
    a = training.grouped_inner_splits(y, groups, task, None, n_inner=3, seed=17)
    b = training.grouped_inner_splits(y, groups, task, None, n_inner=3, seed=17)

    held_out = []
    for (tr, va), (tr_again, va_again) in zip(a, b):
        assert np.array_equal(tr, tr_again)
        assert np.array_equal(va, va_again)
        assert set(groups[tr]).isdisjoint(groups[va])
        held_out.extend(va.tolist())
    assert sorted(held_out) == list(range(len(y)))


def test_inner_folds_require_enough_distinct_groups():
    with pytest.raises(ValueError, match="at least that many distinct groups"):
        training.grouped_inner_splits(np.arange(6), np.repeat([0, 1], 3),
                                      "pdc50", None, n_inner=3, seed=0)


def test_tail_cutoff_is_computed_from_the_supplied_training_rows():
    outer_train = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    inner_train = outer_train[2:]
    assert training.tail_cutoff(inner_train) == pytest.approx(np.quantile(inner_train, 0.1))
    assert training.tail_cutoff(inner_train) != training.tail_cutoff(outer_train)


def test_resume_provenance_mismatch_is_rejected(tmp_path):
    path = tmp_path / "run_provenance.json"
    training.check_resume_provenance(path, {"input": "first"})
    training.check_resume_provenance(path, {"input": "first"})
    with pytest.raises(ValueError, match="provenance mismatch"):
        training.check_resume_provenance(path, {"input": "changed"})


def test_existing_checkpoints_without_provenance_are_rejected(tmp_path):
    model = tmp_path / "task_repeat0_fold0.json"
    model.touch()
    with pytest.raises(ValueError, match="no provenance record"):
        training.check_resume_provenance(tmp_path / "run_provenance.json", {}, [model])


def test_test_evaluation_is_opt_in(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_fast_models.py", "--dev-files", "dev.csv",
                                       "--test-file", "test.csv", "--output-dir", "out"])
    args = training.parse_args()
    assert args.evaluate_test is False
    assert args.n_jobs == 1


def test_model_worker_count_is_explicit_and_bounded():
    model = training.make_model({}, "dmax", n_estimators=5, seed=0, n_jobs=3)
    assert model.get_params()["n_jobs"] == 3
    with pytest.raises(argparse.ArgumentTypeError, match="between 1 and"):
        training.positive_int("0")
    with pytest.raises(argparse.ArgumentTypeError, match="between 1 and"):
        training.positive_int(str(training.MAX_N_JOBS + 1))
