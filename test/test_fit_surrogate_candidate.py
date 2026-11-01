"""Synthetic checks for recipe-driven fitting and data-partition boundaries."""
import hashlib
import importlib.metadata
import json

import numpy as np
import pytest

from scripts.surrogate_research import fit_surrogate_candidate as fit_cli
from tackai.fusion.mol_encoder import DESCRIPTOR_NAMES


def _make_snapshot(path, change_role=None, change_amount=0.0):
    path.mkdir()
    n_search, n_stacking, n_calibration, n_confirmation = 30, 12, 12, 12
    roles = np.array(((["search"] * n_search) + (["stacking"] * n_stacking)
                      + (["calibration"] * n_calibration)
                      + (["confirmation"] * n_confirmation)))
    n = len(roles)
    rng = np.random.default_rng(71)
    X = rng.normal(size=(n, 6)).astype(np.float32)
    groups = np.arange(n, dtype=np.int64)
    # Numeric target with a small deterministic residual, ensuring scale and
    # interval calibration change when only their authorized labels change.
    labels = 0.4 * X[:, 0] - 0.2 * X[:, 1] + 0.1 * X[:, 2]
    labels += rng.normal(0, 0.2, size=n)
    labels = labels.astype(float)
    if change_role:
        labels[roles == change_role] += change_amount
    np.save(path / "synthetic.npy", X)
    np.save(path / "groups.npy", groups)
    np.save(path / "partition.npy", roles)
    np.savez(path / "labels.npz", exact_dmax=labels)
    artifact_names = ("synthetic.npy", "groups.npy", "partition.npy", "labels.npz")
    hashes = {name: hashlib.sha256((path / name).read_bytes()).hexdigest()
              for name in artifact_names}
    manifest = {
        "format": "tack-surrogate-research/v1",
        "artifact_hashes": hashes,
        "features": {"synthetic": {
            "radius": 2, "fp_size": 2, "counts": False,
            "descriptors": [DESCRIPTOR_NAMES[0]],
            "dims": {"fingerprint": 2, "descriptors": 1,
                     "poi": 1, "e3": 1, "assay_time": 1},
        }},
        "context_fingerprint": "synthetic-context-42",
        "versions": {"rdkit": importlib.metadata.version("rdkit")},
        "code_revision": "synthetic",
    }
    (path / "manifest.json").write_text(json.dumps(manifest))
    return {role: np.flatnonzero(roles == role) for role in fit_cli.ROLES}


def _recipe():
    return {
        "schema_version": 1,
        "snapshot_hash": "a" * 64,
        "task": "dmax",
        "feature": "synthetic",
        "label_policy": "exact",
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


def _fit(path, output, change_role=None, change_amount=0.0):
    roles = _make_snapshot(path, change_role, change_amount)
    recipe = _recipe()
    recipe["snapshot_hash"] = fit_cli.sha256(path / "manifest.json")
    recipe_path = path.parent / f"{path.name}.recipe.json"
    recipe_path.write_text(json.dumps(recipe))
    fit_cli._fit_one(path, recipe, output)
    return roles


def test_fit_uses_only_authorized_labels_and_freezes_base_models(tmp_path, monkeypatch):
    seen_role_indices = []
    original_read = fit_cli.read_npz_rows

    def record_reads(path, key, indices):
        seen_role_indices.append(tuple(np.asarray(indices, dtype=int)))
        return original_read(path, key, indices)

    monkeypatch.setattr(fit_cli, "read_npz_rows", record_reads)
    roles_a = _fit(tmp_path / "snapshot_a", tmp_path / "artifact_a")
    roles_stack = _fit(tmp_path / "snapshot_stack", tmp_path / "artifact_stack",
                       "stacking", 8.0)
    roles_cal = _fit(tmp_path / "snapshot_cal", tmp_path / "artifact_calibration",
                    "calibration", 20.0)
    roles_confirm = _fit(tmp_path / "snapshot_confirm", tmp_path / "artifact_confirmation",
                         "confirmation", 10000.0)
    for roles in (roles_stack, roles_cal, roles_confirm):
        assert all(np.array_equal(roles_a[role], roles[role]) for role in fit_cli.ROLES)

    authorized = set(roles_a["search"]) | set(roles_a["stacking"]) | set(roles_a["calibration"])
    assert all(set(rows).issubset(authorized) for rows in seen_role_indices)
    assert not any(set(rows) & set(roles_a["confirmation"]) for rows in seen_role_indices)

    artifacts = [fit_cli.CandidateArtifact.from_pretrained(tmp_path / name)
                 for name in ("artifact_a", "artifact_stack", "artifact_calibration",
                              "artifact_confirmation")]
    baseline, stack_changed, cal_changed, confirm_changed = artifacts
    probe_X = np.random.default_rng(4).normal(size=(5, 6))
    for artifact in artifacts[1:]:
        for base_model, changed_model in zip(baseline.models, artifact.models):
            assert base_model.get_params() == changed_model.get_params()
            assert np.array_equal(base_model.predict(probe_X), changed_model.predict(probe_X))
    # Stacking labels can change only residual scale under the uniform-weight recipe.
    assert stack_changed.residual_scale_ != baseline.residual_scale_
    # Calibration labels can change the interval multiplier, not the fitted point models.
    assert cal_changed.q_hat_ != baseline.q_hat_
    # Confirmation labels have no influence on any fitted/calibrated parameter.
    assert confirm_changed.residual_scale_ == baseline.residual_scale_
    assert confirm_changed.q_hat_ == baseline.q_hat_


def test_recipe_rejects_negative_params_and_incomplete_shared_folds():
    recipe = _recipe()
    recipe["members"][0]["params"]["min_samples_leaf"] = -1
    with pytest.raises(ValueError, match="negative"):
        fit_cli.validate_recipe(recipe)
    recipe = _recipe()
    recipe["members"].pop()
    with pytest.raises(ValueError, match="exactly one model"):
        fit_cli.validate_recipe(recipe)


def test_recipe_cannot_override_seed_or_thread_limits():
    recipe = _recipe()
    recipe["members"][0]["params"]["n_jobs"] = -1
    with pytest.raises(ValueError, match="controlled resources"):
        fit_cli.validate_recipe(recipe)


def test_recipe_rejects_wrong_snapshot_before_training(tmp_path):
    path = tmp_path / "snapshot"
    _make_snapshot(path)
    with pytest.raises(ValueError, match="different snapshot"):
        fit_cli._fit_one(path, _recipe(), tmp_path / "artifact")


def test_fit_uses_frozen_calibration_groups_for_group_max(tmp_path):
    snapshot = tmp_path / "snapshot"
    roles = _make_snapshot(snapshot)
    recipe = _recipe()
    recipe["snapshot_hash"] = fit_cli.sha256(snapshot / "manifest.json")
    recipe["calibration"]["unit"] = "group_max"
    fit_cli._fit_one(snapshot, recipe, tmp_path / "artifact")
    artifact = fit_cli.CandidateArtifact.from_pretrained(tmp_path / "artifact")
    assert artifact.calibration_unit_ == "group_max"
    assert artifact.n_calibration_units_ == len(roles["calibration"])


def test_catboost_accepts_tuned_parameters_not_present_in_default_get_params():
    pytest.importorskip("catboost")
    member = {"family": "catboost", "seed": 3, "params": {"l2_leaf_reg": 4.5}}
    assert fit_cli._estimator(member, "dmax").get_params()["l2_leaf_reg"] == 4.5


def test_current_baseline_weight_parameter_is_not_forwarded_to_xgboost():
    member = {"family": "current_xgb", "seed": 3,
              "params": {"n_estimators": 2, "alpha_w": .5}}
    model = fit_cli._estimator(member, "dmax")
    assert "alpha_w" not in model.get_params()
    X = np.arange(60).reshape(30, 2)
    y = np.linspace(0, 1, 30)
    fit_cli.fit_learner(model, member["family"], X, y, member["params"], False)
    assert np.isfinite(model.predict(X)).all()


def test_role_limited_npz_reader_loads_only_requested_offsets(tmp_path):
    labels = np.array([0., 1., np.nan, 999999.])
    path = tmp_path / "labels.npz"
    np.savez(path, exact_activity=labels)
    selected = fit_cli.read_npz_rows(path, "exact_activity", np.array([0, 2]))
    assert np.array_equal(selected, labels[[0, 2]], equal_nan=True)
