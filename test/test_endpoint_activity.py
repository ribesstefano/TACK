"""The joint ranking uses paired endpoints and rejects misaligned rows."""
import json

import numpy as np
import pandas as pd
import pytest

from scripts.surrogate_research.benchmark_endpoint_activity import bottleneck_score, combine


def test_bottleneck_score_respects_both_thresholds():
    score = bottleneck_score([.9, .6, .9], [7., 7., 5.])
    assert score[0] > .5
    assert score[1] < .5 and score[2] < .5
    assert score[1] == pytest.approx(score[2])


def test_combine_rejects_unpaired_observations(tmp_path):
    for task in ("dmax", "pdc50"):
        directory = tmp_path / task
        directory.mkdir()
        (directory / "COMPLETE").write_text("complete")
        (directory / "config.json").write_text(json.dumps({
            "task": task, "snapshot": "models/research_v1", "snapshot_hash": "hash",
            "repeats": 1, "folds": 2}))
        pd.DataFrame({"row": [1, 2], "group": [1, 2], "split": ["repeat0_fold0"] * 2,
                      "y": [1., 0.], "prediction": [.9, .7]}).to_csv(
                          directory / "predictions.csv", index=False)
    combine(tmp_path / "dmax", tmp_path / "pdc50", tmp_path / "joined")
    values = pd.read_csv(tmp_path / "joined/predictions.csv").prediction.to_numpy()
    assert np.isfinite(values).all()
    data = pd.read_csv(tmp_path / "pdc50/predictions.csv")
    data.loc[0, "group"] = 99
    data.to_csv(tmp_path / "pdc50/predictions.csv", index=False)
    with pytest.raises(ValueError, match="align"):
        combine(tmp_path / "dmax", tmp_path / "pdc50", tmp_path / "bad")
