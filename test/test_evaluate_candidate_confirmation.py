"""Synthetic-only checks for locked internal confirmation evaluation."""
import hashlib
import importlib.metadata
import json

import numpy as np
import pytest
from rdkit import rdBase
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from scripts import evaluate_candidate_confirmation as eval_cli
from tackai.fusion.candidate_artifact import CandidateArtifact
from tackai.fusion.molecular_spec import MolecularFeatureSpec
from tackai.fusion.mol_encoder import DESCRIPTOR_NAMES


def _snapshot_and_artifact(root, task="dmax", *, calibrated=True):
    snapshot = root / "snapshot"
    snapshot.mkdir(parents=True)
    roles = np.array(["search"] * 30 + ["stacking"] * 12
                     + ["calibration"] * 12 + ["confirmation"] * 12)
    rng = np.random.default_rng(913)
    X = rng.normal(size=(len(roles), 6)).astype(np.float32)
    y_dmax = .5 * X[:, 0] - .25 * X[:, 1] + rng.normal(0, .15, len(roles))
    y_pdc50 = 6.5 + X[:, 0] + rng.normal(0, .2, len(roles))
    y_activity = (np.arange(len(roles)) % 2).astype(float)
    groups = np.arange(len(roles), dtype=np.int64)
    np.save(snapshot / "synthetic.npy", X)
    np.save(snapshot / "groups.npy", groups)
    np.save(snapshot / "partition.npy", roles)
    np.savez(snapshot / "labels.npz", exact_dmax=y_dmax, exact_pdc50=y_pdc50,
             exact_activity=y_activity)
    dims = {"fingerprint": 2, "descriptors": 1, "poi": 1, "e3": 1, "assay_time": 1}
    hashes = {name: hashlib.sha256((snapshot / name).read_bytes()).hexdigest()
              for name in ("synthetic.npy", "groups.npy", "partition.npy", "labels.npz")}
    manifest = {
        "format": "tack-surrogate-research/v1",
        "artifact_hashes": hashes,
        "features": {"synthetic": {"radius": 2, "fp_size": 2, "counts": False,
                                     "descriptors": [DESCRIPTOR_NAMES[0]], "dims": dims}},
        "context_fingerprint": "synthetic-context-confirmation-v1",
        "versions": {"rdkit": importlib.metadata.version("rdkit")},
    }
    manifest_path = snapshot / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    train_rows = np.flatnonzero(roles == "search")
    stack_rows = np.flatnonzero(roles == "stacking")
    cal_rows = np.flatnonzero(roles == "calibration")
    if task == "activity":
        estimator = ExtraTreesClassifier(n_estimators=8, min_samples_leaf=2,
                                         random_state=7, n_jobs=1)
        y = y_activity
        kind = "classification"
    else:
        estimator = ExtraTreesRegressor(n_estimators=8, min_samples_leaf=2,
                                        random_state=7, n_jobs=1)
        y = y_dmax if task == "dmax" else y_pdc50
        kind = "regression"
    estimator.fit(X[train_rows], y[train_rows])
    spec = MolecularFeatureSpec(radius=2, size=2, count=False,
                                descriptor_names=(DESCRIPTOR_NAMES[0],),
                                rdkit_version=rdBase.rdkitVersion)
    recipe_hash = "b" * 64
    recipe = {"task": task, "feature": "synthetic", "label_policy": "exact",
              "snapshot_hash": manifest_hash, "recipe_sha256": recipe_hash,
              "fit_stage": "calibrated" if calibrated else "pre_calibration"}
    contract = {
        "context_fingerprint": manifest["context_fingerprint"],
        "snapshot_manifest_sha256": manifest_hash,
        "snapshot_artifact_hashes": {name: hashes[name]
                                     for name in ("synthetic.npy", "labels.npz",
                                                  "groups.npy", "partition.npy")},
        "feature_name": "synthetic", "feature_dims": dims, "feature_width": 6,
        "molecular_feature_spec": spec.to_dict(),
    }
    artifact = CandidateArtifact([estimator], kind, spec, recipe=recipe,
                                 context_contract=contract)
    if calibrated:
        if kind == "regression":
            artifact.fit_residual_scale(X[stack_rows], y[stack_rows])
        artifact.calibrate(X[cal_rows], y[cal_rows], alpha=.1)
    artifact_dir = root / "artifact"
    artifact.save_pretrained(artifact_dir)
    return snapshot, artifact_dir, recipe_hash, roles


@pytest.mark.parametrize("task", ["dmax", "activity"])
def test_confirmation_reads_only_exact_confirmation_offsets(tmp_path, monkeypatch, task):
    snapshot, artifact, recipe_hash, roles = _snapshot_and_artifact(tmp_path, task)
    seen = []
    original = eval_cli.read_npz_rows

    def record(path, key, indices):
        seen.append((key, np.asarray(indices).copy()))
        return original(path, key, indices)

    monkeypatch.setattr(eval_cli, "read_npz_rows", record)
    result = eval_cli.evaluate(snapshot, artifact, tmp_path / "report", task=task,
                               selected_recipe_sha256=recipe_hash,
                               n_bootstrap=30, seed=19)
    output = tmp_path / "report"
    confirmation_rows = np.flatnonzero(roles == "confirmation")
    assert len(seen) == 1
    assert seen[0][0] == f"exact_{task}"
    assert np.array_equal(seen[0][1], confirmation_rows)
    assert result["n_rows"] == len(confirmation_rows)
    assert result["n_groups"] == len(confirmation_rows)
    assert result["evaluation_scope"] == "internal_confirmation_only"
    assert result["candidate_selection_or_fitting_performed"] is False
    assert result["historical_test_accessed"] is False
    assert (output / "COMPLETE").is_file()
    assert not (output / "IN_PROGRESS").exists()
    if task == "activity":
        assert {"ap", "roc_auc", "brier", "log_loss"}.issubset(result["metrics"])
    else:
        assert {"rmse", "mae", "r2"}.issubset(result["metrics"])
        assert result["metrics"]["interval_90"]["nominal_coverage"] == .9
        assert 0 <= result["metrics"]["interval_90"]["group_all_coverage"] <= 1
        assert "group_all_coverage_ci95" in result["group_bootstrap"]["metrics"]["interval_90"]
    with pytest.raises(FileExistsError):
        eval_cli.evaluate(snapshot, artifact, tmp_path / "report", task=task,
                          selected_recipe_sha256=recipe_hash, n_bootstrap=0)


def test_confirmation_refuses_wrong_recipe_and_uncalibrated_artifact(tmp_path):
    snapshot, artifact, recipe_hash, _ = _snapshot_and_artifact(tmp_path)
    wrong_output = tmp_path / "wrong_hash"
    with pytest.raises(ValueError, match="recipe hash"):
        eval_cli.evaluate(snapshot, artifact, wrong_output, task="dmax",
                          selected_recipe_sha256="c" * 64, n_bootstrap=0)
    assert not wrong_output.exists()

    uncal_root = tmp_path / "uncalibrated"
    uncal_root.mkdir()
    uncal_snapshot, uncal_artifact, uncal_hash, _ = _snapshot_and_artifact(
        uncal_root, calibrated=False)
    with pytest.raises(ValueError, match="not marked as fully calibrated"):
        eval_cli.evaluate(uncal_snapshot, uncal_artifact, tmp_path / "uncal_report",
                          task="dmax", selected_recipe_sha256=uncal_hash, n_bootstrap=0)


def test_failure_after_confirmation_read_starts_leaves_nonreusable_lock(tmp_path, monkeypatch):
    snapshot, artifact, recipe_hash, _ = _snapshot_and_artifact(tmp_path)
    output = tmp_path / "interrupted"

    def fail_after_lock(path, key, indices):
        assert (output / "IN_PROGRESS").is_file()
        raise RuntimeError("synthetic read interruption")

    monkeypatch.setattr(eval_cli, "read_npz_rows", fail_after_lock)
    with pytest.raises(RuntimeError, match="synthetic read interruption"):
        eval_cli.evaluate(snapshot, artifact, output, task="dmax",
                          selected_recipe_sha256=recipe_hash, n_bootstrap=0)
    assert (output / "IN_PROGRESS").is_file()
    assert not (output / "COMPLETE").exists()
    with pytest.raises(FileExistsError):
        eval_cli.evaluate(snapshot, artifact, output, task="dmax",
                          selected_recipe_sha256=recipe_hash, n_bootstrap=0)
