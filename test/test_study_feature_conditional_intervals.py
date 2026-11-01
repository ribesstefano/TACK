"""Feature-conditioned scales use only permitted labels and disjoint groups."""
import numpy as np
import pytest

from scripts.surrogate_research import study_feature_conditional_intervals as study


def specification():
    return {"dims": {"fingerprint": 2, "descriptors": 4, "e3": 1, "cell": 1,
                     "poi": 1, "assay": 1, "assay_time": 1},
            "descriptors": ["TPSA", "MolLogP", "NumRotatableBonds", "MolWt"]}


def test_fixed_features_are_named_and_schema_checked():
    matrix = np.arange(33, dtype=float).reshape(3, 11)
    values = study._feature_columns(matrix, specification())
    np.testing.assert_array_equal(values[0], matrix[0, [5, 2, 3, 4, 6, 7, 8, 9, 10]])
    with pytest.raises(ValueError, match="width"):
        study._feature_columns(matrix[:, :-1], specification())
    bad = specification()
    bad["descriptors"][0] = "wrong"
    with pytest.raises(ValueError, match="absent"):
        study._feature_columns(matrix, bad)
    matrix[0, 5] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        study._feature_columns(matrix, specification())


def test_scales_ignore_nonfitting_labels():
    rng = np.random.default_rng(53)
    mean = rng.normal(size=100)
    disagreement = rng.uniform(.01, .3, size=100)
    fixed = rng.normal(size=(100, 9))
    y = rng.normal(size=100)
    fit = np.arange(45)
    a = study._fit_scales(mean, disagreement, fixed, y, fit, 4)
    y[45:] += 1000
    b = study._fit_scales(mean, disagreement, fixed, y, fit, 4)
    for name in a:
        np.testing.assert_array_equal(a[name], b[name])
        assert np.all(np.isfinite(a[name])) and np.all(a[name] > 0)


def test_study_uses_only_stacking_labels_and_disjoint_groups(tmp_path, monkeypatch):
    snapshot, bundle = tmp_path / "snapshot", tmp_path / "bundle"
    snapshot.mkdir()
    bundle.mkdir()
    (snapshot / "manifest.json").write_text("snapshot")
    (bundle / "manifest.json").write_text("bundle")
    rng = np.random.default_rng(51)
    X = rng.normal(size=(160, 11))
    groups = np.arange(len(X))
    roles = np.array(["search"] * 20 + ["stacking"] * 100 + ["calibration"] * 20 + ["confirmation"] * 20)
    labels = 6 + X[:, 2] + rng.normal(0, .3, len(X))
    manifest = {"features": {"synthetic": specification()}}
    class Artifact:
        recipe = {"task": "pdc50", "feature": "synthetic"}
        def _predictions(self, features):
            return np.column_stack((6 + features[:, 2], 6 + features[:, 2] + .1))
    monkeypatch.setattr(study, "_validate_bundle", lambda *args: (
        snapshot, bundle, manifest, {"model_sha256": "model"}, Artifact(), X, groups, roles))
    seen = []
    def reader(path, key, indices):
        seen.append(indices.copy())
        return labels[indices]
    monkeypatch.setattr(study, "read_npz_rows", reader)
    report = study.study(snapshot, bundle, tmp_path / "out", folds=3, n_bootstrap=3)
    assert len(seen) == 1 and np.all(roles[seen[0]] == "stacking")
    assert set(report["metrics"]) == {"global", "two_feature", "fixed_features"}
    assert report["final_calibration_performed"] is False
    assert report["descriptor_names"] == list(study.DESCRIPTORS)
    for fold in report["fold_metrics"]:
        scale, quantile, evaluation = (set(fold[key]) for key in (
            "scale_group_ids", "quantile_group_ids", "evaluation_group_ids"))
        assert not (scale & quantile or scale & evaluation or quantile & evaluation)
    with pytest.raises(FileExistsError):
        study.study(snapshot, bundle, tmp_path / "out", folds=3)
