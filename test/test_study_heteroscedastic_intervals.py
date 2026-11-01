"""Heteroscedastic study reads only stacking labels and respects group roles."""
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts import fit_surrogate_candidate as fit_cli
from scripts import study_heteroscedastic_intervals as study_cli
from tackai.fusion.mol_encoder import DESCRIPTOR_NAMES


def _make_snapshot(path, *, changed_role=None, change_amount=0.):
    path.mkdir()
    roles = np.array(["search"] * 30 + ["stacking"] * 60
                     + ["calibration"] * 12 + ["confirmation"] * 12)
    rng = np.random.default_rng(94)
    X = rng.normal(size=(len(roles), 6)).astype(np.float32)
    y = (6.0 + 0.8 * X[:, 0] + 0.3 * X[:, 1] ** 2
         + rng.normal(0, 0.25, len(roles))).astype(float)
    if changed_role:
        y[roles == changed_role] += change_amount
    np.save(path / "synthetic.npy", X)
    np.save(path / "groups.npy", np.arange(len(roles), dtype=np.int64))
    np.save(path / "partition.npy", roles)
    np.savez(path / "labels.npz", exact_pdc50=y)
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
        "context_fingerprint": "synthetic-context-71",
        "versions": {"rdkit": importlib.metadata.version("rdkit")},
    }
    (path / "manifest.json").write_text(json.dumps(manifest))


def _bundle(snapshot, bundle, *, changed_role=None, change_amount=0.):
    _make_snapshot(snapshot, changed_role=changed_role, change_amount=change_amount)
    recipe = {
        "schema_version": 1, "snapshot_hash": fit_cli.sha256(snapshot / "manifest.json"),
        "task": "pdc50", "feature": "synthetic", "label_policy": "exact",
        "cv": {"repeats": 1, "folds": 3, "seeds": [410]},
        "members": [
            {"family": "extratrees", "repeat": 0, "fold": fold,
             "seed": 70 + fold,
             "params": {"n_estimators": 8, "min_samples_leaf": 1}}
            for fold in range(3)
        ],
        "stacking": {"weights": "uniform", "l2": 0.1},
        "calibration": {"alpha": 0.1},
    }
    fit_cli._fit_one(snapshot, recipe, bundle, pre_calibration=True)


def test_heteroscedastic_study_uses_only_stacking_labels_and_disjoint_groups(
        tmp_path, monkeypatch):
    snapshot, bundle = tmp_path / "snapshot", tmp_path / "bundle"
    _bundle(snapshot, bundle)
    seen = []
    original_reader = study_cli.read_npz_rows

    def record_reads(path, key, indices):
        seen.append((key, tuple(np.asarray(indices, dtype=int))))
        return original_reader(path, key, indices)

    monkeypatch.setattr(study_cli, "read_npz_rows", record_reads)
    result = study_cli.study(snapshot, bundle, tmp_path / "study", folds=3,
                             n_bootstrap=30, alpha=.1)
    stack_rows = set(np.flatnonzero(
        np.load(snapshot / "partition.npy", allow_pickle=False) == "stacking"))
    assert seen and all(key == "exact_pdc50" and set(rows).issubset(stack_rows)
                        for key, rows in seen)
    assert result["only_stacking_labels_read"] is True
    assert result["calibration_labels_read"] is False
    assert result["confirmation_labels_read"] is False
    assert result["final_calibration_performed"] is False
    assert result["scale_model"]["features"] == [
        "uniform pooled prediction", "member disagreement"]
    assert set(result["metrics"]) == {"global_group_max", "heteroscedastic_group_max"}
    assert result["paired_group_bootstrap"]["n_bootstraps"] == 30
    for fold in result["fold_metrics"]:
        scale = set(fold["scale_group_ids"])
        quantile = set(fold["quantile_group_ids"])
        evaluation = set(fold["evaluation_group_ids"])
        assert not (scale & quantile or scale & evaluation or quantile & evaluation)
        assert fold["global_group_max"]["rows"] == fold["evaluation_rows"]
        assert fold["heteroscedastic_group_max"]["groups"] == fold["evaluation_groups"]


def test_calibration_and_confirmation_label_changes_do_not_change_study_metrics(tmp_path):
    snap_a, snap_b = tmp_path / "snap_a", tmp_path / "snap_b"
    bundle_a, bundle_b = tmp_path / "bundle_a", tmp_path / "bundle_b"
    _bundle(snap_a, bundle_a)
    _bundle(snap_b, bundle_b, changed_role="calibration", change_amount=300.)
    report_a = study_cli.study(snap_a, bundle_a, tmp_path / "study_a", folds=3,
                               n_bootstrap=20)
    report_b = study_cli.study(snap_b, bundle_b, tmp_path / "study_b", folds=3,
                               n_bootstrap=20)
    assert report_a["metrics"] == report_b["metrics"]
    assert report_a["paired_group_bootstrap"] == report_b["paired_group_bootstrap"]


def test_study_cli_help_runs_as_direct_script():
    repository = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/study_heteroscedastic_intervals.py", "--help"],
        cwd=repository, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "--alpha" in result.stdout and "--scale-fraction" in result.stdout
