"""Stacking studies use only grouped stacking labels and frozen model outputs."""
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts import fit_surrogate_candidate as fit_cli
from scripts import study_surrogate_stacking as study_cli
from tackai.fusion.mol_encoder import DESCRIPTOR_NAMES


def _make_snapshot(path, change_role=None, change_amount=0.):
    path.mkdir()
    roles = np.array(["search"] * 30 + ["stacking"] * 12
                     + ["calibration"] * 12 + ["confirmation"] * 12)
    rng = np.random.default_rng(71)
    X = rng.normal(size=(len(roles), 6)).astype(np.float32)
    y = (0.4 * X[:, 0] - 0.2 * X[:, 1] + 0.1 * X[:, 2]
         + rng.normal(0, 0.2, size=len(roles))).astype(float)
    if change_role:
        y[roles == change_role] += change_amount
    np.save(path / "synthetic.npy", X)
    np.save(path / "groups.npy", np.arange(len(roles), dtype=np.int64))
    np.save(path / "partition.npy", roles)
    np.savez(path / "labels.npz", exact_dmax=y)
    artifact_names = ("synthetic.npy", "groups.npy", "partition.npy", "labels.npz")
    hashes = {name: hashlib.sha256((path / name).read_bytes()).hexdigest()
              for name in artifact_names}
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


def _recipe():
    return {
        "schema_version": 1, "snapshot_hash": "a" * 64, "task": "dmax",
        "feature": "synthetic", "label_policy": "exact",
        "cv": {"repeats": 1, "folds": 3, "seeds": [210]},
        "members": [
            {"family": "extratrees", "repeat": 0, "fold": fold,
             "seed": 40 + fold,
             "params": {"n_estimators": 5, "min_samples_leaf": 1}}
            for fold in range(3)
        ],
        "stacking": {"weights": "uniform", "l2": 0.1},
        "calibration": {"alpha": 0.2},
    }
def _bundle(snapshot, output, *, changed_role=None, change_amount=0.):
    _make_snapshot(snapshot, changed_role, change_amount)
    recipe = _recipe()
    recipe["snapshot_hash"] = fit_cli.sha256(snapshot / "manifest.json")
    fit_cli._fit_one(snapshot, recipe, output, pre_calibration=True)


def test_precalibration_bundle_and_group_study_never_read_calibration_or_confirmation(
        tmp_path, monkeypatch):
    snapshot_a, snapshot_b = tmp_path / "snapshot_a", tmp_path / "snapshot_b"
    bundle_a, bundle_b = tmp_path / "bundle_a", tmp_path / "bundle_b"
    _bundle(snapshot_a, bundle_a)
    _bundle(snapshot_b, bundle_b, changed_role="calibration", change_amount=200.)

    seen_fit = []
    original_fit_reader = fit_cli.read_npz_rows

    def record_fit_reads(path, key, indices):
        seen_fit.append(tuple(np.asarray(indices, dtype=int)))
        return original_fit_reader(path, key, indices)

    # Pre-calibration bundle construction must read search labels only.
    monkeypatch.setattr(fit_cli, "read_npz_rows", record_fit_reads)
    # Run bundle construction again with a fresh snapshot to observe label reads.
    snapshot_c, bundle_c = tmp_path / "snapshot_c", tmp_path / "bundle_c"
    _bundle(snapshot_c, bundle_c)
    roles = np.load(snapshot_c / "partition.npy")
    search_rows = set(np.flatnonzero(roles == "search"))
    assert seen_fit and all(set(rows).issubset(search_rows) for rows in seen_fit)

    artifact = fit_cli.CandidateArtifact.from_pretrained(bundle_a)
    assert artifact.recipe["fit_stage"] == "pre_calibration"
    assert artifact.q_hat_ is None and artifact.temperature_ is None
    assert artifact.residual_scale_ is None

    seen_study = []
    original_study_reader = study_cli.read_npz_rows

    def record_study_reads(path, key, indices):
        seen_study.append((key, tuple(np.asarray(indices, dtype=int))))
        return original_study_reader(path, key, indices)

    monkeypatch.setattr(study_cli, "read_npz_rows", record_study_reads)
    result_a = study_cli.study(snapshot_a, bundle_a, tmp_path / "study_a", folds=3,
                               l2_values=(.1, 1.), n_bootstrap=25)
    result_b = study_cli.study(snapshot_b, bundle_b, tmp_path / "study_b", folds=3,
                               l2_values=(.1, 1.), n_bootstrap=25)

    stacking_rows = set(np.flatnonzero(np.load(snapshot_a / "partition.npy") == "stacking"))
    assert seen_study
    assert all(key == "exact_dmax" and set(rows).issubset(stacking_rows)
               for key, rows in seen_study)
    assert result_a["selection_performed"] is False
    assert result_a["calibration_labels_read"] is False
    assert result_a["confirmation_labels_read"] is False
    assert result_a["metrics"] == result_b["metrics"]
    assert (tmp_path / "study_a" / "oof_predictions.csv").read_text() == (
        tmp_path / "study_b" / "oof_predictions.csv").read_text()

    for fold in result_a["fold_metrics"]:
        assert not (set(fold["training_group_ids"]) & set(fold["validation_group_ids"]))
        assert set(fold) >= {"uniform", "convex_l2_0.1", "convex_l2_1"}


def test_study_rejects_calibrated_or_wrong_snapshot_bundle(tmp_path):
    snapshot = tmp_path / "snapshot"
    bundle = tmp_path / "bundle"
    _make_snapshot(snapshot)
    recipe = _recipe()
    recipe["snapshot_hash"] = fit_cli.sha256(snapshot / "manifest.json")
    fit_cli._fit_one(snapshot, recipe, bundle)
    import pytest

    with pytest.raises(ValueError, match="pre-calibration"):
        study_cli.study(snapshot, bundle, tmp_path / "study")


def test_study_cli_help_runs_as_direct_script():
    repository = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/study_surrogate_stacking.py", "--help"],
        cwd=repository, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "--snapshot" in result.stdout and "--bundle" in result.stdout
