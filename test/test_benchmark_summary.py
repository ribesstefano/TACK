"""Prevent misleading paired comparisons from incompatible OOF ledgers."""
import numpy as np
import pandas as pd
import pytest

from scripts.summarize_surrogate_benchmarks import paired_interval, paired_view


def ledger():
    return pd.DataFrame({"row": [0, 1, 0, 1], "group": [10, 11, 10, 11],
                         "split": ["repeat0_fold0", "repeat0_fold1",
                                   "repeat1_fold1", "repeat1_fold0"],
                         "y": [0., 1., 0., 1.], "prediction": [.1, .9, .3, .7]})


def test_repeated_oof_pools_only_matched_predictions():
    frame = ledger()
    candidate, baseline = paired_view(frame.iloc[::-1], frame, "cv")
    np.testing.assert_allclose(candidate.prediction, [.2, .8])
    pd.testing.assert_frame_equal(candidate, baseline)


@pytest.mark.parametrize("change", ["missing_repeat", "duplicate", "label", "group", "fold"])
def test_noncomparable_ledgers_are_rejected(change):
    reference = ledger()
    candidate = reference.copy()
    if change == "missing_repeat":
        candidate = candidate.iloc[:2]
    elif change == "duplicate":
        candidate = pd.concat([candidate, candidate.iloc[:1]])
    elif change == "label":
        candidate.loc[candidate.row == 0, "y"] = .5
    elif change == "group":
        candidate.loc[candidate.row == 0, "group"] = 12
    else:
        candidate.loc[0, "split"] = "repeat0_fold1"
    with pytest.raises(ValueError):
        paired_view(candidate, reference, "cv")


def test_group_split_leakage_rejected_even_when_both_ledgers_agree():
    frame = ledger()
    frame["group"] = 10
    with pytest.raises(ValueError, match="crosses"):
        paired_view(frame, frame, "cv")


def test_identical_models_have_zero_paired_interval():
    result = paired_interval(np.array([0., 1., .5]), np.array([.1, .9, .6]),
                             np.array([.1, .9, .6]), np.array([0, 0, 1]), False, 20)
    assert result == {"low": 0., "high": 0., "valid_draws": 20}


def test_single_class_bootstraps_do_not_invent_ap_evidence():
    result = paired_interval(np.zeros(3), np.full(3, .2), np.full(3, .1),
                             np.arange(3), True, 20)
    assert result == {"low": None, "high": None, "valid_draws": 0}
