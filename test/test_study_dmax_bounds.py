import json

import numpy as np
import pandas as pd
import pytest

from scripts.surrogate_research.study_dmax_bounds import study
from tackai.fusion.research import sha256


def make_run(tmp_path, *, bad_label=False, bad_role=False):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    np.save(snapshot / "partition.npy", np.array(["search", "search", "confirmation" if bad_role else "search"]))
    np.save(snapshot / "source_partition.npy", np.array(["train", "eval", "eval"]))
    np.save(snapshot / "groups.npy", np.array([1, 2, 3]))
    np.savez(snapshot / "labels.npz", exact_dmax=np.array([.1, .9, 1.1 if bad_label else 1.]))
    pd.DataFrame({"publication": ["p1", "p2", "p3"]}).to_csv(snapshot / "rows.csv", index=False)
    names = ("partition.npy", "source_partition.npy", "groups.npy", "labels.npz", "rows.csv")
    (snapshot / "manifest.json").write_text(json.dumps({"artifact_hashes": {n: sha256(snapshot / n) for n in names}}))
    run = tmp_path / "run"
    run.mkdir()
    config = {"task": "dmax", "snapshot": str(snapshot), "snapshot_hash": sha256(snapshot / "manifest.json"),
              "eval_policy": "exact", "repeats": 1, "folds": 2}
    (run / "config.json").write_text(json.dumps(config))
    (run / "COMPLETE").touch()
    pd.DataFrame({"row": [0, 1, 2, 1, 2], "group": [1, 2, 3, 2, 3],
                  "split": ["repeat0_fold0", "repeat0_fold1", "repeat0_fold1", "source_holdout", "source_holdout"],
                  "y": [.1, .9, 1.1 if bad_label else 1., .9, 1.1 if bad_label else 1.],
                  "prediction": [-.2, 1.2, 1.5, 1.2, 1.5]}).to_csv(run / "predictions.csv", index=False)
    return run


def test_clipping_is_evaluated_after_pooling(tmp_path):
    report = study([make_run(tmp_path)], tmp_path / "report.json", draws=10)
    cv, source = report["results"]
    assert cv["fold_prediction_outside_bounds"] == 3
    assert source["fold_prediction_outside_bounds"] == 2
    assert cv["pooled_prediction_outside_bounds"] == 3
    assert cv["clipped"]["rmse"] < cv["unclipped"]["rmse"]
    assert source["clipped"]["rmse"] < source["unclipped"]["rmse"]
    with pytest.raises(ValueError, match="fresh"):
        study([tmp_path / "run"], tmp_path / "report.json", draws=10)


def test_rejects_reserved_rows_and_out_of_bounds_labels(tmp_path):
    for name, kwargs, message in [("role", {"bad_role": True}, "roles"),
                                  ("label", {"bad_label": True}, "outside")]:
        root = tmp_path / name
        root.mkdir()
        with pytest.raises(ValueError, match=message):
            study([make_run(root, **kwargs)], root / "report.json", draws=10)
        assert not (root / "report.json").exists()
