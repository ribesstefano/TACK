"""Regression checks for grouped tuning, provenance, and portable fast ensembles."""
import json
from argparse import Namespace
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "training"))
import train_fast_models as fast
from tackai.fusion.data import FusionData, make_targets
from tackai.fusion.xgb_ensemble import XGBStackedEnsemble


def test_activity_uses_strict_100nm_threshold():
    labels = make_targets([90, 90, 90, 80, np.nan], [99, 100, 101, 10, 100])["activity"]
    np.testing.assert_array_equal(labels, [1, 0, 0, 0, 0])


def test_tuning_never_splits_scaffolds(monkeypatch):
    X = np.arange(60, dtype=np.float32).reshape(-1, 1)
    y = np.linspace(0, 1, 60)
    groups = np.arange(60) // 3
    seen = []

    class Model:
        best_iteration = 4

        def set_params(self, **kwargs):
            return self

        def fit(self, X_train, y_train, *, eval_set, **kwargs):
            X_val, _ = eval_set[0]
            assert not set(groups[X_train[:, 0].astype(int)]) & set(groups[X_val[:, 0].astype(int)])
            seen.append(len(X_val))
            return self

        def predict(self, X_val):
            return np.zeros(len(X_val))

    monkeypatch.setattr(fast, "make_model", lambda *a, **kw: Model())
    trial = optuna.trial.FixedTrial(dict(max_depth=3, learning_rate=.05, subsample=1.,
                                        colsample_bytree=1., min_child_weight=5.,
                                        reg_lambda=10., alpha_w=0.))
    score = fast.objective(trial, X, y, "dmax", .1, 42, 3, groups=groups)
    assert len(seen) == 3 and np.isfinite(score)


def test_resume_rejects_changed_training_contract(tmp_path):
    path = tmp_path / "provenance.json"
    record = {"source": {"revision": "abc"}, "training": {"seed": 42}}
    fast.check_provenance(path, record)
    with pytest.raises(SystemExit, match="training"):
        fast.check_provenance(path, {**record, "training": {"seed": 43}})


def test_ensemble_restores_features_and_ignores_unlisted_models(fake_cache, tiny_csv, tmp_path):
    data = FusionData.from_csv([tiny_csv], cache=False)
    _, X, y, _ = data.task_rows("dmax")
    model = xgb.XGBRegressor(n_estimators=3, n_jobs=1).fit(X, y)
    ensemble = XGBStackedEnsemble([model])
    ensemble.target_name = "dmax"
    ensemble.feature_spec = dict(
        context_repo=str(fake_cache), context_revision="local",
        context_fingerprint=data.encoder.fingerprint(), protein_space=data.encoder.protein_space,
        fingerprint={"radius": data.featurizer.radius, "size": data.featurizer.fp_size},
        descriptors=data.featurizer.descriptors, block_dims=data.dims)
    out = tmp_path / "ensemble"
    ensemble.save_pretrained(out)
    model.save_model(out / "stale.ubj")
    (out / "provenance.json").write_text(json.dumps({"metadata": True}))
    restored = XGBStackedEnsemble.from_pretrained(str(out))
    assert len(restored.models) == 1
    assert restored.feature_spec == ensemble.feature_spec
    assert restored.target_name == "dmax"
    assert restored.load_data().dims == data.dims
    np.testing.assert_array_equal(restored.predict(X)["mean"], ensemble.predict(X)["mean"])
    from tackai.serve.app import load_service
    service = load_service(model=str(out), tasks=["dmax"])
    assert service.data.dims == data.dims
    with pytest.raises(ValueError, match="ensemble target"):
        load_service(model=str(out), tasks=["pdc50"])


@pytest.mark.parametrize("external_test", [False, True])
def test_training_exports_without_publishing(fake_cache, tiny_csv, tmp_path, monkeypatch, external_test):
    out = tmp_path / "run"
    monkeypatch.setattr(fast, "load_frames", lambda args: [pd.read_csv(tiny_csv)])
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *a, **kw: str(fake_cache))
    argv = ["train_fast_models.py", "--output-dir", str(out),
                                     "--tasks", "dmax", "--n-folds", "2", "--n-inner", "2",
                                     "--n-trials", "1", "--calibration-frac", "0",
                                     "--test-frac", "0", "--n-jobs", "1"]
    if external_test:
        test = pd.read_csv(tiny_csv).iloc[:2].copy()
        test["SMILES"] = ["CCCCCCC", "OCCCCCCO"]
        test_file = tmp_path / "external_test.csv"
        test.to_csv(test_file, index=False)
        argv += ["--test-file", str(test_file)]
    monkeypatch.setattr(sys, "argv", argv)
    fast.main()
    ensemble = XGBStackedEnsemble.from_pretrained(str(out / "ensembles" / "dmax"))
    assert len(ensemble.models) == 2
    assert ensemble.feature_spec["activity_thresholds"]["pdc50"] == 7.0
    assert ensemble.target_name == "dmax"
    assert ensemble.training_provenance["training"]["n_trials"] == 1
    assert "rdkit" in ensemble.training_provenance["training"]["versions"]
    assert (out / "fold_metrics.csv").exists()
    if external_test:
        saved = np.load(out / "dmax" / "test.npz")
        assert saved["X"].shape[0] == len(saved["y"]) == 2
        assert len(saved["scaffold_keys"]) == 2
        assert "ensemble_test" in set(pd.read_csv(out / "fold_metrics.csv").split)
    # A matching resume reloads the same folds and exports a valid ensemble again.
    fast.main()


def test_calibration_fit_and_conformal_rows_never_share_scaffolds(tmp_path, monkeypatch):
    seen = {}

    class Ensemble:
        task = "regression"
        models = [object()]

        def _base_predictions(self, X):
            seen["fit_groups"] = set(X[:, 0].astype(int) // 4)
            return np.zeros((len(X), 1))

        def calibrate(self, X, y):
            assert not seen["fit_groups"] & set(X[:, 0].astype(int) // 4)
            assert (self.sigmas_ > 0).all()
            seen["calibrated"] = True

    monkeypatch.setattr(fast.XGBStackedEnsemble, "from_pretrained", lambda *a, **kw: Ensemble())
    fast.build_ensemble(tmp_path, np.arange(80).reshape(-1, 1), np.ones(80), 42,
                        groups_cal=np.arange(80) // 4)
    assert seen["calibrated"]


def test_explicit_hf_training_files_use_pinned_local_reader(monkeypatch):
    import _dataset

    seen = {}

    class Split:
        def to_pandas(self):
            return pd.DataFrame({"SMILES": ["CC"]})

    def load(repo, config=None, **kwargs):
        seen.update(repo=repo, config=config, **kwargs)
        return {"train": Split()}

    monkeypatch.setattr("datasets.load_dataset", load)
    downloads = []
    def download(repo, filename, **kwargs):
        downloads.append((repo, filename, kwargs))
        return "/cached/train.csv"
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    args = Namespace(dataset="example/public-protac", dataset_config="default",
                     dataset_revision="pinned", dataset_files=["train.csv"])
    _dataset.load_frames(args)
    assert seen["config"] is None
    assert seen["repo"] == "csv"
    assert seen["data_files"] == {"train": ["/cached/train.csv"]}
    assert downloads == [("example/public-protac", "train.csv",
                           {"repo_type": "dataset", "revision": "pinned"})]


def test_external_scaffold_exclusion_removes_related_molecules(tmp_path):
    test_csv = tmp_path / "held_out.csv"
    pd.DataFrame({"SMILES": ["c1ccccc1"]}).to_csv(test_csv, index=False)
    frames = [pd.DataFrame({"SMILES": ["Cc1ccccc1", "Oc1ccccc1", "c1ccc2ccccc2c1"]})]
    assert fast.test_scaffold_overlap_smiles(frames, str(test_csv)) == {"Cc1ccccc1", "Oc1ccccc1"}


def test_fixed_test_export_preserves_features_and_labels(fake_cache, tiny_csv, tmp_path, monkeypatch):
    import hashlib
    import audit_fast_models as audit

    frame = pd.read_csv(tiny_csv)
    data = FusionData.from_csv([frame], cache=False)
    model = xgb.XGBRegressor(n_estimators=2, n_jobs=1).fit(data.X, np.zeros(len(data.X)))
    bundle = XGBStackedEnsemble([model])
    bundle.feature_spec = dict(
        context_repo=str(fake_cache), context_revision="local",
        context_fingerprint=data.encoder.fingerprint(), protein_space=data.encoder.protein_space,
        fingerprint={"radius": data.featurizer.radius, "size": data.featurizer.fp_size},
        descriptors=data.featurizer.descriptors, block_dims=data.dims)
    run = tmp_path / "run"
    bundle.save_pretrained(run / "ensembles" / "dmax")
    held_out = np.arange(0, len(frame), 3)
    for task in ("dmax", "pdc50", "activity"):
        idx, X, y, _ = data.task_rows(task)
        keep = np.isin(idx, held_out)
        (run / task).mkdir()
        np.savez(run / task / "test.npz", X=X[keep], y=y[keep], row_index=idx[keep])
    digest = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()
    (run / "provenance.json").write_text(json.dumps({
        "source": {"dataset": "mock/source", "config": "default", "revision": "pinned"},
        "dataset_identity": {"frames": [digest]}, "test_split": {"source": "butina"}}))
    monkeypatch.setattr(audit, "load_frames", lambda args: [frame])
    output = tmp_path / "benchmark.csv"
    manifest = audit.export_fixed_test(run, output)
    assert manifest["n_rows"] == len(held_out)
    assert manifest["development_row_index"] == held_out.tolist()
    assert output.exists() and output.with_suffix(".json").exists()


def test_hub_download_reads_only_the_requested_task_config(tmp_path, monkeypatch):
    """Task bundles must not require a config.json at the multi-task repo root."""
    X = np.arange(24, dtype=np.float32).reshape(-1, 2)
    model = xgb.XGBRegressor(n_estimators=2, n_jobs=1).fit(X, np.arange(12))
    bundle = XGBStackedEnsemble([model])
    bundle.target_name = "pdc50"
    bundle.training_provenance = {"source": {"revision": "dataset-commit"}}
    bundle.feature_spec = {"context_revision": "pinned-context"}
    bundle.save_pretrained(tmp_path / "pdc50")
    card = (tmp_path / "pdc50" / "README.md").read_text()
    assert "load_data()" in card and "local_files_only=True" in card
    assert "pinned-context" in card
    assert "dataset-commit" in card
    requests = []

    def snapshot(repo_id, **kwargs):
        requests.append((repo_id, kwargs))
        return str(tmp_path)

    monkeypatch.setattr("tackai.fusion.xgb_ensemble.snapshot_download", snapshot)
    restored = XGBStackedEnsemble.from_pretrained(
        pretrained_model_name_or_path="org/ensemble", subfolder="pdc50",
        revision="model-commit", local_files_only=True)
    assert len(requests) == 1
    assert requests[0][1]["allow_patterns"] == "pdc50/*"
    assert requests[0][1]["revision"] == "model-commit"
    assert requests[0][1]["local_files_only"]
    assert restored.feature_spec == bundle.feature_spec
    assert restored.target_name == "pdc50"
    assert restored.training_provenance == bundle.training_provenance
    np.testing.assert_array_equal(restored.predict(X)["mean"], bundle.predict(X)["mean"])


def test_capacity_selection_keeps_outer_validation_and_test_out_of_training(tmp_path, monkeypatch):
    import compare_fast_capacity as capacity
    from types import SimpleNamespace
    from tackai.fusion.data import make_splits

    source, output = tmp_path / "source", tmp_path / "result"
    (source / "pdc50").mkdir(parents=True)
    X = np.column_stack([np.arange(60), np.ones(60)]).astype(np.float32)
    y, groups = np.arange(60) * .01, np.arange(60) // 2
    np.savez(source / "pdc50" / "split_rows.npz", cv_row_index=np.arange(50),
             calibration_row_index=np.arange(50, 60))
    np.savez(source / "pdc50" / "calibration.npz", X=X[50:], y=y[50:], groups=groups[50:])
    np.savez(source / "pdc50" / "test.npz", X=np.array([[100, 1]]), y=np.array([999.]))
    provenance = {"source": {"dataset": "ailab-bio/AutoTPDplus"},
                  "test_file": "public-test.csv", "training": {}}
    (source / "provenance.json").write_text(json.dumps(provenance))
    data = SimpleNamespace(X=X, groups=groups, target=lambda task: y)
    monkeypatch.setattr(capacity, "_load_run_source", lambda path: (provenance, [], []))
    monkeypatch.setattr(capacity, "test_overlap_smiles", lambda *args: set())
    monkeypatch.setattr(capacity.FusionData, "from_csv", lambda *args, **kwargs: data)
    fitted_rows = []

    class Model:
        best_iteration, best_score = 4, .1
        def __init__(self, params):
            self.params = params
        def set_params(self, **kwargs):
            return self
        def fit(self, train_X, train_y, **kwargs):
            ids = train_X[:, 0].astype(int)
            assert (ids < 50).all()  # neither calibration nor test rows
            fitted_rows.append(set(ids))
            if "eval_set" in kwargs:
                stop = kwargs["eval_set"][0][0][:, 0].astype(int)
                assert not set(groups[ids]) & set(groups[stop])
            return self
        def predict(self, matrix):
            return matrix[:, 0] * .01 + (self.params["max_depth"] - 3) * .01
        def save_model(self, path):
            path.write_text("{}")

    monkeypatch.setattr(capacity, "make_model", lambda params, *a, **kw: Model(params))
    class Ensemble:
        feature_spec = {"test": True}
        training_provenance = {"source": provenance["source"]}
        def load_data(self, **kwargs):
            return SimpleNamespace(encoder=None, featurizer=None)
        def save_pretrained(self, path):
            path.mkdir(parents=True, exist_ok=True)
        def predict(self, matrix):
            return {"mean": np.zeros(len(matrix))}

    monkeypatch.setattr(capacity.XGBStackedEnsemble, "from_pretrained", lambda *a, **kw: Ensemble())
    monkeypatch.setattr(capacity, "build_ensemble", lambda *a, **kw: Ensemble())
    load = np.load
    def checked_load(path, *args, **kwargs):
        if Path(path).name == "test.npz":
            assert (output / "selection.json").exists()
        return load(path, *args, **kwargs)
    monkeypatch.setattr(capacity.np, "load", checked_load)
    selected = capacity.compare(source, output)
    assert selected["candidate"] == "shallow"
    folds = make_splits(y[:50], groups[:50], n_repeats=1, n_folds=5)[0]
    for candidate in range(len(capacity.CAPACITIES)):
        for fold, (train, valid) in enumerate(folds):
            early, refit = fitted_rows[10*candidate+2*fold:10*candidate+2*fold+2]
            assert not early & set(valid) and refit == set(train)


def test_capacity_pool_selects_complementary_oof_predictions_and_checks_alignment(tmp_path):
    import compare_fast_capacity as capacity

    rows = [{"candidate": "a", "rmse": 1.}, {"candidate": "b", "rmse": 1.}]
    (tmp_path / "selection.json").write_text(json.dumps({"candidates": rows}))
    for name, prediction in (("a", [0., 2., 0.]), ("b", [2., 0., 2.])):
        directory = tmp_path / "candidates" / name
        directory.mkdir(parents=True)
        np.savez(directory / "oof.npz", y=np.ones(3), prediction=prediction,
                 row_index=np.arange(3))
    selection = capacity.select_pool(tmp_path)
    assert selection["selected"]["profiles"] == ["a", "b"]
    assert selection["selected"]["rmse"] == 0.
    np.savez(tmp_path / "candidates" / "b" / "oof.npz", y=np.ones(3),
             prediction=[2., 0., 2.], row_index=[2, 1, 0])
    with pytest.raises(AssertionError):
        capacity.select_pool(tmp_path)
