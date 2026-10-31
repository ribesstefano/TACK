"""Portable non-XGBoost fast-candidate artifact contract."""
import json

import numpy as np
import pytest
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from tackai.fusion.candidate_artifact import CandidateArtifact
from tackai.fusion.molecular_spec import MolecularFeatureSpec


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(41)
    X = rng.normal(size=(140, 8))
    y_reg = X[:, 0] - 0.4 * X[:, 1] + rng.normal(0, 0.25, len(X))
    y_cls = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y_reg, y_cls


def test_regression_residual_scale_is_fitted_before_conformal_calibration(tmp_path, sample_data):
    X, y, _ = sample_data
    train, stacking, calibration = np.arange(60), np.arange(60, 100), np.arange(100, 140)
    model = ExtraTreesRegressor(n_estimators=12, min_samples_leaf=2,
                                random_state=3, n_jobs=1).fit(X[train], y[train])
    contract = {"snapshot_manifest_sha256": "frozen-test-hash", "feature_space": "context-v1"}
    recipe = {"family": "extratrees", "params": {"n_estimators": 12}}
    artifact = CandidateArtifact([model], "regression", MolecularFeatureSpec.preset(
        "morgan_r3_4096"), recipe=recipe, context_contract=contract)

    with pytest.raises(ValueError, match="fit_residual_scale"):
        artifact.calibrate(X[calibration], y[calibration])

    # A single estimator is the member: its public predict() remains the point
    # estimate, and internal ExtraTrees members are not silently substituted.
    raw = artifact.predict(X[calibration])
    assert np.allclose(raw["mean"], model.predict(X[calibration]))
    assert np.all(raw["std_disagreement"] == 0)
    assert "std_noise" not in raw and "std" not in raw

    artifact.fit_residual_scale(X[stacking], y[stacking])
    frozen_scale = artifact.residual_scale_
    artifact.calibrate(X[calibration], y[calibration])
    out = artifact.predict(X[calibration])
    assert artifact.residual_scale_ == frozen_scale
    assert set(out) == {"mean", "std", "std_noise", "std_disagreement", "lower", "upper"}
    assert np.allclose(out["std"], np.hypot(out["std_noise"], out["std_disagreement"]))
    assert np.all(out["std_noise"] == frozen_scale)
    assert np.all(out["lower"] <= out["mean"])
    assert np.all(out["upper"] >= out["mean"])
    assert artifact.help(True)["explanations"]["std_noise"].startswith(
        "Global RMS residual scale fitted on stacking rows")

    # Changing calibration labels cannot refit the stacking scale.
    other = CandidateArtifact([model], "regression", artifact.molecular_spec,
                              recipe=recipe, context_contract=contract)
    other.fit_residual_scale(X[stacking], y[stacking])
    other.calibrate(X[calibration], y[calibration] + 1.0)
    assert other.residual_scale_ == frozen_scale

    artifact.save_pretrained(tmp_path / "artifact")
    loaded = CandidateArtifact.from_pretrained(tmp_path / "artifact")
    for key, values in out.items():
        assert np.allclose(loaded.predict(X[calibration])[key], values)
    assert loaded.molecular_spec == artifact.molecular_spec
    assert loaded.recipe == recipe
    assert loaded.context_contract == contract


def test_activity_temperature_and_entropy_terms_are_coherent(sample_data):
    X, _, y = sample_data
    train, calibration = np.arange(80), np.arange(100, 140)
    first = ExtraTreesClassifier(n_estimators=12, min_samples_leaf=2,
                                 random_state=5, n_jobs=1).fit(X[train], y[train])
    second = ExtraTreesClassifier(n_estimators=12, min_samples_leaf=3,
                                  random_state=9, n_jobs=1).fit(X[train], y[train])
    artifact = CandidateArtifact([first, second], "classification",
                                 MolecularFeatureSpec.preset("morgan_r2_2048"),
                                 weights=[0.7, 0.3])
    artifact.calibrate(X[calibration], y[calibration])
    out = artifact.predict(X[calibration])
    assert set(out) == {"proba", "entropy_total", "entropy_aleatoric", "entropy_epistemic"}
    assert np.all((out["proba"] >= 0) & (out["proba"] <= 1))
    assert np.all(out["entropy_epistemic"] >= 0)
    assert np.allclose(out["entropy_total"],
                       out["entropy_aleatoric"] + out["entropy_epistemic"])
    assert artifact.temperature_ > 0
    original_temperature = artifact.temperature_
    with pytest.raises(ValueError, match="both labels"):
        artifact.calibrate(X[calibration], np.zeros(len(calibration)))
    assert artifact.temperature_ == original_temperature


def test_rejects_bad_weights_width_and_nonfinite_predictions(sample_data):
    X, y, _ = sample_data
    model = ExtraTreesRegressor(n_estimators=2, random_state=1).fit(X, y)
    spec = MolecularFeatureSpec.preset("default")
    with pytest.raises(ValueError, match="nonnegative"):
        CandidateArtifact([model], "regression", spec, weights=[-1])
    artifact = CandidateArtifact([model], "regression", spec)
    with pytest.raises(ValueError, match="feature width"):
        artifact.predict(X[:, :-1])

    class NonfiniteRegressor:
        n_features_in_ = X.shape[1]

        def predict(self, rows):
            return np.full(len(rows), np.nan)

    bad = CandidateArtifact([NonfiniteRegressor()], "regression", spec)
    with pytest.raises(ValueError, match="non-finite"):
        bad.predict(X[:2])


def test_empty_batch_and_model_integrity_validation(tmp_path, sample_data):
    X, y, _ = sample_data
    model = ExtraTreesRegressor(n_estimators=2, random_state=1).fit(X, y)
    artifact = CandidateArtifact([model], "regression", MolecularFeatureSpec.preset("default"))
    empty = artifact.predict(X[:0])
    assert all(values.shape == (0,) for values in empty.values())
    artifact.save_pretrained(tmp_path / "artifact")
    model_path = tmp_path / "artifact" / "candidate.joblib"
    model_path.write_bytes(model_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="integrity"):
        CandidateArtifact.from_pretrained(tmp_path / "artifact")


def test_context_contract_tampering_is_detected(tmp_path, sample_data):
    X, y, _ = sample_data
    model = ExtraTreesRegressor(n_estimators=2, random_state=1).fit(X, y)
    artifact = CandidateArtifact([model], "regression", MolecularFeatureSpec.preset("default"),
                                 context_contract={"context_hash": "expected"})
    artifact.save_pretrained(tmp_path / "artifact")
    manifest_path = tmp_path / "artifact" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["context_contract"]["context_hash"] = "changed"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest"):
        CandidateArtifact.from_pretrained(tmp_path / "artifact")
