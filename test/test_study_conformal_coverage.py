"""Conformal coverage study stays inside stacking groups and uses group splits."""
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts import fit_surrogate_candidate as fit_cli
from scripts import study_conformal_coverage as coverage_cli
from tackai.fusion.mol_encoder import DESCRIPTOR_NAMES


def _make_snapshot(path, *, change_role=None, change_amount=0.):
    path.mkdir()
    roles = np.array(["search"] * 30 + ["stacking"] * 60
                     + ["calibration"] * 12 + ["confirmation"] * 12)
    rng = np.random.default_rng(71)
    X = rng.normal(size=(len(roles), 6)).astype(np.float32)
    y = (0.4 * X[:, 0] - 0.2 * X[:, 1] + rng.normal(0, 0.25, len(roles))).astype(float)
    if change_role:
        y[roles == change_role] += change_amount
    np.save(path / "synthetic.npy", X)
    np.save(path / "groups.npy", np.arange(len(roles), dtype=np.int64))
    np.save(path / "partition.npy", roles)
    np.savez(path / "labels.npz", exact_dmax=y)
    names = ("synthetic.npy", "groups.npy", "partition.npy", "labels.npz")
    hashes = {name: hashlib.sha256((path / name).read_bytes()).hexdigest() for name in names}
    manifest = {
        "format": "tack-surrogate-research/v1", "artifact_hashes": hashes,
        "features": {"synthetic": {
            "radius": 2, "fp_size": 2, "counts": False,
            "descriptors": [DESCRIPTOR_NAMES[0]],
            "dims": {"fingerprint": 2, "descriptors": 1,
                     "poi": 1, "e3": 1, "assay_time": 1},
        }},
        "context_fingerprint": "synthetic-context-42",
        "versions": {"rdkit": importlib.metadata.version("rdkit")},
    }
    (path / "manifest.json").write_text(json.dumps(manifest))


def _recipe(snapshot):
    return {
        "schema_version": 1, "snapshot_hash": fit_cli.sha256(snapshot / "manifest.json"),
        "task": "dmax", "feature": "synthetic", "label_policy": "exact",
        "cv": {"repeats": 1, "folds": 3, "seeds": [210]},
        "members": [
            {"family": "extratrees", "repeat": 0, "fold": fold,
             "seed": 40 + fold,
             "params": {"n_estimators": 5, "min_samples_leaf": 1}}
            for fold in range(3)
        ],
        "stacking": {"weights": "uniform", "l2": 0.1},
        "calibration": {"alpha": 0.1},
    }


def _bundle(snapshot, bundle, *, change_role=None, change_amount=0.):
    _make_snapshot(snapshot, change_role=change_role, change_amount=change_amount)
    fit_cli._fit_one(snapshot, _recipe(snapshot), bundle, pre_calibration=True)


def test_group_max_quantile_uses_group_maxima_not_rows():
    scores = np.array([1., 100., 2., 3.])
    groups = np.array(["large", "large", "single_a", "single_b"])
    row_quantile = coverage_cli._finite_sample_quantile(scores, alpha=.4)
    group_scores = coverage_cli._group_max_scores(scores, groups)
    group_quantile = coverage_cli._finite_sample_quantile(group_scores, alpha=.4)
    assert row_quantile == 3.
    assert group_quantile == 100.


def test_study_reads_stacking_labels_only_and_keeps_nested_groups_disjoint(tmp_path, monkeypatch):
    snapshot_a, snapshot_b = tmp_path / "snapshot_a", tmp_path / "snapshot_b"
    bundle_a, bundle_b = tmp_path / "bundle_a", tmp_path / "bundle_b"
    _bundle(snapshot_a, bundle_a)
    _bundle(snapshot_b, bundle_b, change_role="calibration", change_amount=500.)

    seen = []
    original_reader = coverage_cli.read_npz_rows

    def record_reads(path, key, indices):
        seen.append((key, tuple(np.asarray(indices, dtype=int))))
        return original_reader(path, key, indices)

    monkeypatch.setattr(coverage_cli, "read_npz_rows", record_reads)
    result_a = coverage_cli.study(snapshot_a, bundle_a, tmp_path / "study_a", folds=3,
                                  n_bootstrap=30, alpha=.1)
    result_b = coverage_cli.study(snapshot_b, bundle_b, tmp_path / "study_b", folds=3,
                                  n_bootstrap=30, alpha=.1)

    roles = np.load(snapshot_a / "partition.npy", allow_pickle=False)
    stacking_rows = set(np.flatnonzero(roles == "stacking"))
    assert seen and all(key == "exact_dmax" and set(rows).issubset(stacking_rows)
                        for key, rows in seen)
    assert result_a["only_stacking_labels_read"] is True
    assert result_a["calibration_labels_read"] is False
    assert result_a["confirmation_labels_read"] is False
    assert result_a["final_calibration_performed"] is False
    assert result_a["metrics"] == result_b["metrics"]
    for fold in result_a["fold_metrics"]:
        scale = set(fold["scale_group_ids"])
        quantile = set(fold["quantile_group_ids"])
        evaluation = set(fold["evaluation_group_ids"])
        assert not (scale & quantile or scale & evaluation or quantile & evaluation)
        assert fold["row_metrics"]["rows"] == fold["evaluation_rows"]
        assert fold["group_max_metrics"]["groups"] == fold["evaluation_groups"]
    assert result_a["paired_group_bootstrap"]["n_bootstraps"] == 30
    assert result_a["paired_group_bootstrap"]["paired_group_max_minus_row_ci95"][
        "row_coverage"]["valid_bootstraps"] == 30


def test_study_rejects_calibrated_bundle_and_wrong_snapshot(tmp_path):
    snapshot = tmp_path / "snapshot"
    bundle = tmp_path / "bundle"
    _make_snapshot(snapshot)
    recipe = _recipe(snapshot)
    fit_cli._fit_one(snapshot, recipe, bundle)
    with pytest.raises(ValueError, match="pre-calibration"):
        coverage_cli.study(snapshot, bundle, tmp_path / "bad_study", folds=3)


def test_study_rejects_snapshot_artifact_hash_mismatch(tmp_path):
    snapshot, bundle = tmp_path / "snapshot", tmp_path / "bundle"
    _bundle(snapshot, bundle)
    with (snapshot / "groups.npy").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        coverage_cli.study(snapshot, bundle, tmp_path / "bad_study", folds=3)


def test_study_cli_help_runs_as_direct_script():
    repository = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/study_conformal_coverage.py", "--help"],
        cwd=repository, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "--alpha" in result.stdout and "--bundle" in result.stdout
