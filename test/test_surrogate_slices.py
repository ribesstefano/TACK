"""Slice reports preserve pairing and refuse confirmation rows."""
import pandas as pd
import pytest

from scripts.surrogate_research.report_surrogate_slices import METADATA, slice_report


def inputs():
    predictions = pd.DataFrame({"row": [0, 1, 2], "group": [0, 1, 2],
                                "split": ["source_holdout"] * 3,
                                "y": [0., 1., 1.], "prediction": [.2, .8, .7]})
    metadata = pd.DataFrame({key: ["A"] * 3 for key in METADATA})
    metadata["partition"] = "search"
    metadata["source_partition"] = "eval"
    metadata["assay_time"] = [4., 24., 72.]
    return predictions, metadata


def test_small_slices_are_flagged_and_identical_models_match():
    predictions, metadata = inputs()
    result = slice_report(predictions, predictions, metadata, "activity", "source_holdout")
    assert all(x["small_slice"] for x in result)
    assert all(x["candidate"] == x["reference"] for x in result)
    times = [x for x in result if x["field"] == "time_bin"]
    assert len(times) == 3 and all(x["candidate"] is None for x in times)


def test_confirmation_is_not_accepted_as_development():
    predictions, metadata = inputs()
    metadata.loc[0, "partition"] = "confirmation"
    with pytest.raises(ValueError, match="non-search"):
        slice_report(predictions, predictions, metadata, "activity", "source_holdout")
