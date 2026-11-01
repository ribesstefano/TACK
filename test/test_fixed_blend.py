import json

import numpy as np
import pandas as pd
import pytest

from scripts.surrogate_research.benchmark_fixed_blend import blend_runs
from tackai.fusion.research import sha256


def make_run(path, offset=0, group=1):
    path.mkdir()
    snapshot = path.parent / "snapshot"
    if not snapshot.exists():
        snapshot.mkdir()
        np.save(snapshot / "groups.npy", np.array([1, 2]))
        np.save(snapshot / "partition.npy", np.array(["search", "search"]))
        np.save(snapshot / "source_partition.npy", np.array(["train", "eval"]))
        np.savez(snapshot / "labels.npz", exact_activity=np.array([1., 0.]))
        names = ("groups.npy", "partition.npy", "source_partition.npy", "labels.npz")
        (snapshot / "manifest.json").write_text(json.dumps({"artifact_hashes": {n: sha256(snapshot / n) for n in names}}))
    config = {"task": "activity", "snapshot": str(snapshot), "snapshot_hash": sha256(snapshot / "manifest.json"),
              "eval_policy": "exact", "repeats": 1, "folds": 1}
    (path / "config.json").write_text(json.dumps(config))
    (path / "COMPLETE").touch()
    pd.DataFrame({"split": ["repeat0_fold0", "source_holdout"], "row": [0, 1],
                  "group": [group, 2], "y": [1, 0],
                  "prediction": [.2 + offset, .4 + offset]}).to_csv(path / "predictions.csv", index=False)
    return path


def test_fixed_blend_and_provenance(tmp_path):
    a = make_run(tmp_path / "a")
    b = make_run(tmp_path / "b", .2)
    out = tmp_path / "out"
    blend_runs([a, b], out)
    assert pd.read_csv(out / "predictions.csv").prediction.tolist() == pytest.approx([.3, .5])
    config = json.loads((out / "config.json").read_text())
    assert config["weights"] == [.5, .5]
    assert len(config["inputs"][0]["predictions_sha256"]) == 64
    assert (out / "COMPLETE").exists()
    with pytest.raises(FileExistsError):
        blend_runs([a, b], out)


def test_blend_rejects_mismatched_groups_before_writing(tmp_path):
    a = make_run(tmp_path / "a")
    b = make_run(tmp_path / "b", group=3)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="groups"):
        blend_runs([a, b], out)
    assert not out.exists()
