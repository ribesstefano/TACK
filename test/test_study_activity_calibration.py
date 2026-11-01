"""Activity calibration is group-held-out and never reads reserved labels."""
import hashlib
import importlib.metadata
import json

import numpy as np
import pandas as pd
import pytest

from scripts.surrogate_research import fit_surrogate_candidate as fit_cli
from scripts.surrogate_research import study_activity_calibration as study_cli
from scripts.surrogate_research.study_conformal_coverage import _validate_bundle
from tackai.fusion.mol_encoder import DESCRIPTOR_NAMES


def make_bundle(snapshot, bundle, *, flip_reserved=False):
    snapshot.mkdir()
    roles = np.array(["search"] * 60 + ["stacking"] * 90
                     + ["calibration"] * 12 + ["confirmation"] * 12)
    rng = np.random.default_rng(712)
    X = rng.normal(size=(len(roles), 6)).astype(np.float32)
    y = (X[:, 0] + rng.normal(size=len(roles)) > 0).astype(float)
    if flip_reserved:
        mask = np.isin(roles, ["calibration", "confirmation"])
        y[mask] = 1 - y[mask]
    np.save(snapshot / "synthetic.npy", X)
    np.save(snapshot / "groups.npy", np.arange(len(roles), dtype=np.int64))
    np.save(snapshot / "partition.npy", roles)
    np.savez(snapshot / "labels.npz", exact_activity=y)
    pd.DataFrame({"e3_raw": ["VHL" if i % 2 else "CRBN" for i in range(len(roles))]}).to_csv(snapshot / "rows.csv", index=False)
    names = ("synthetic.npy", "groups.npy", "partition.npy", "labels.npz", "rows.csv")
    hashes = {name: hashlib.sha256((snapshot / name).read_bytes()).hexdigest() for name in names}
    manifest = {"format": "tack-surrogate-research/v1", "artifact_hashes": hashes,
                "features": {"synthetic": {"radius": 2, "fp_size": 2, "counts": False,
                    "descriptors": [DESCRIPTOR_NAMES[0]],
                    "dims": {"fingerprint": 2, "descriptors": 1, "poi": 1, "e3": 1, "assay_time": 1}}},
                "context_fingerprint": "synthetic-context-activity",
                "versions": {"rdkit": importlib.metadata.version("rdkit")}}
    (snapshot / "manifest.json").write_text(json.dumps(manifest))
    recipe = {"schema_version": 1, "snapshot_hash": fit_cli.sha256(snapshot / "manifest.json"),
              "task": "activity", "feature": "synthetic", "label_policy": "exact",
              "cv": {"repeats": 1, "folds": 3, "seeds": [210]},
              "members": [{"family": "extratrees", "repeat": 0, "fold": fold,
                           "seed": 40 + fold, "params": {"n_estimators": 5, "min_samples_leaf": 1}}
                          for fold in range(3)],
              "stacking": {"weights": "uniform", "l2": .1}, "calibration": {"alpha": .1}}
    fit_cli._fit_one(snapshot, recipe, bundle, pre_calibration=True)


def test_activity_study_uses_only_stacking_labels_and_disjoint_groups(tmp_path, monkeypatch):
    snapshot, bundle = tmp_path / "snapshot", tmp_path / "bundle"
    make_bundle(snapshot, bundle)
    seen = []
    original = study_cli.read_npz_rows

    def record(path, key, rows):
        seen.append((key, set(rows)))
        return original(path, key, rows)

    monkeypatch.setattr(study_cli, "read_npz_rows", record)
    report = study_cli.study(snapshot, bundle, tmp_path / "study", folds=3, draws=10)
    stacking = set(np.flatnonzero(np.load(snapshot / "partition.npy") == "stacking"))
    assert seen and all(key == "exact_activity" and rows <= stacking for key, rows in seen)
    assert report["only_stacking_labels_read"] and not report["final_calibration_performed"]
    assert not report["confirmation_labels_read"]
    assert set(report["slices"]["all"]["metrics"]) == set(study_cli.METHODS)
    assert "e3:VHL" in report["slices"] and "e3:CRBN" in report["slices"]
    for fold in report["folds_detail"]:
        assert not (set(fold["train_groups"]) & set(fold["evaluation_groups"]))
        assert fold["temperature"] > 0 and fold["platt_slope"] >= 0
    with pytest.raises(ValueError, match="regression"):
        _validate_bundle(snapshot, bundle)
    with pytest.raises(FileExistsError):
        study_cli.study(snapshot, bundle, tmp_path / "study", folds=3, draws=0)


def test_reserved_label_changes_do_not_change_results(tmp_path):
    snap_a, snap_b = tmp_path / "snap_a", tmp_path / "snap_b"
    bundle_a, bundle_b = tmp_path / "bundle_a", tmp_path / "bundle_b"
    make_bundle(snap_a, bundle_a)
    make_bundle(snap_b, bundle_b, flip_reserved=True)
    a = study_cli.study(snap_a, bundle_a, tmp_path / "a", folds=3, draws=10)
    b = study_cli.study(snap_b, bundle_b, tmp_path / "b", folds=3, draws=10)
    assert a["slices"] == b["slices"]


def test_single_class_and_invalid_probabilities_rejected():
    with pytest.raises(ValueError, match="both classes"):
        study_cli._fit_calibrators(np.array([[.1], [.3]]), np.array([0, 0]))
    with pytest.raises(ValueError, match="both classes"):
        study_cli._fit_calibrators(np.array([[.1], [1.2]]), np.array([0, 1]))
