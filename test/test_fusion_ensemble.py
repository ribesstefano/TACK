"""The generic ensemble: shared input for every member, cached context, from_pretrained."""
import json

import numpy as np
import pytest

from fusion_fixtures import CELLS, SEQS, SMILES
from tackai.fusion.data import FusionData
from tackai.fusion.ensemble import FusionEnsemble, FusionPrediction
from tackai.fusion.models import GPInteraction, XGBoostFusion

CTX = {"poi_seq": SEQS["poi"][0], "e3_seq": SEQS["e3"][0], "cell_id": CELLS[0],
       "assay": "western blot", "assay_time": 24.0}


@pytest.fixture
def data(fake_cache, tiny_csv):
    return FusionData.from_csv([tiny_csv], cache=False)


def fast_gp(**kw):
    return GPInteraction(n_restarts=1, n_iter=5, max_hyper_points=40, **kw)


def fast_xgb(**kw):
    return XGBoostFusion(n_estimators=20, grid=[{"max_depth": 3, "reg_lambda": 5.0}], **kw)


def test_fit_produces_the_requested_number_of_members(data):
    ens = FusionEnsemble.fit(fast_gp, data, task="pdc50", n_members=3, n_folds=3)
    assert len(ens.members) == 3 and ens.task == "pdc50"
    assert sum(ens.weights.values()) == pytest.approx(1.0)


def test_predict_is_the_weighted_mean_of_the_members(data):
    ens = FusionEnsemble.fit(fast_gp, data, task="pdc50", n_members=2, n_folds=3)
    pred = ens.predict([{"smiles": SMILES[0], **CTX}])
    assert isinstance(pred, FusionPrediction)
    members = np.array(list(pred.member_predictions.values()))
    expected = np.average(members, axis=0, weights=list(ens.weights.values()))
    assert np.allclose(pred.mean, expected)


def test_context_path_equals_the_ordinary_path_exactly(data):
    """The whole point of the context cache: identical numbers, less work."""
    ens = FusionEnsemble.fit(fast_gp, data, task="pdc50", n_members=2, n_folds=3)
    ctx = ens.transform_context(CTX)
    fast = ens.predict(SMILES[:4], context=ctx)
    slow = ens.predict([{"smiles": s, **CTX} for s in SMILES[:4]])
    assert np.allclose(fast.mean, slow.mean, rtol=0, atol=0)
    assert np.allclose(fast.std, slow.std, rtol=0, atol=0)


def test_std_combines_member_variance_and_member_spread(data):
    ens = FusionEnsemble.fit(fast_gp, data, task="pdc50", n_members=3, n_folds=3)
    pred = ens.predict(SMILES[:3], context=ens.transform_context(CTX))
    means = np.array(list(pred.member_predictions.values()))
    varis = np.array(list(pred.member_std.values())) ** 2
    w = np.array(list(ens.weights.values()))[:, None]
    expected = np.sqrt((w * varis).sum(0) + (w * (means - pred.mean) ** 2).sum(0))
    assert np.allclose(pred.std, expected)


def test_std_falls_back_to_member_spread_without_predictive_variance(data):
    ens = FusionEnsemble.fit(fast_xgb, data, task="pdc50", n_members=3, n_folds=3)
    pred = ens.predict(SMILES[:3], context=ens.transform_context(CTX))
    means = np.array(list(pred.member_predictions.values()))
    w = np.array(list(ens.weights.values()))[:, None]
    assert np.allclose(pred.std, np.sqrt((w * (means - pred.mean) ** 2).sum(0)))


def test_confidence_interval_brackets_the_mean(data):
    ens = FusionEnsemble.fit(fast_gp, data, task="pdc50", n_members=2, n_folds=3)
    pred = ens.predict(SMILES[:3], context=ens.transform_context(CTX))
    assert (pred.ci_lower_95 <= pred.mean).all() and (pred.mean <= pred.ci_upper_95).all()


def test_members_may_mix_estimator_classes(data):
    """Generic over members: the same input serves all of them."""
    ens = FusionEnsemble.fit([fast_gp, fast_xgb, fast_gp], data, task="pdc50", n_folds=3)
    assert len(ens.members) == 3
    pred = ens.predict(SMILES[:2], context=ens.transform_context(CTX))
    assert pred.mean.shape == (2,) and np.isfinite(pred.mean).all()


def test_binary_task_predicts_probabilities(data):
    ens = FusionEnsemble.fit(fast_xgb, data, task="activity", n_members=2, n_folds=3)
    pred = ens.predict(SMILES[:3], context=ens.transform_context(CTX))
    assert ((pred.mean >= 0) & (pred.mean <= 1)).all()
    assert pred.label_name


def test_save_and_from_pretrained_round_trip(data, tmp_path):
    ens = FusionEnsemble.fit(fast_gp, data, task="pdc50", n_members=2, n_folds=3)
    before = ens.predict(SMILES[:3], context=ens.transform_context(CTX)).mean
    ens.save(tmp_path / "ens")
    loaded = FusionEnsemble.from_pretrained(tmp_path / "ens")
    assert loaded.task == "pdc50" and len(loaded.members) == 2
    after = loaded.predict(SMILES[:3], context=loaded.transform_context(CTX)).mean
    assert np.allclose(before, after, rtol=0, atol=0)


def test_manifest_records_the_block_layout(data, tmp_path):
    FusionEnsemble.fit(fast_gp, data, task="pdc50", n_members=1, n_folds=3).save(tmp_path / "ens")
    manifest = json.loads((tmp_path / "ens" / "manifest.json").read_text())
    assert manifest["task"] == "pdc50"
    assert manifest["block_dims"]["fingerprint"] == 1024
    assert manifest["n_members"] == 1 and "tackai" in manifest["versions"]


def test_invalid_smiles_yields_nan_not_an_exception(data):  # Review Focus 2
    ens = FusionEnsemble.fit(fast_gp, data, task="pdc50", n_members=1, n_folds=3)
    pred = ens.predict(["not_a_molecule", SMILES[0]], context=ens.transform_context(CTX))
    assert list(pred.ok) == [False, True]
    assert np.isnan(pred.mean[0]) and np.isfinite(pred.mean[1])


def test_empty_batch_returns_empty_arrays(data):  # Review Focus 5
    ens = FusionEnsemble.fit(fast_gp, data, task="pdc50", n_members=1, n_folds=3)
    pred = ens.predict([], context=ens.transform_context(CTX))
    assert pred.mean.shape == (0,) and pred.std.shape == (0,)


def test_unseen_sequence_in_a_context_raises(data):  # Review Focus 1
    ens = FusionEnsemble.fit(fast_gp, data, task="pdc50", n_members=1, n_folds=3)
    with pytest.raises(KeyError, match="poi"):
        ens.transform_context({**CTX, "poi_seq": "MKKKWWWNOTINTABLE"})


def test_to_frame_has_one_row_per_molecule(data):
    ens = FusionEnsemble.fit(fast_gp, data, task="pdc50", n_members=2, n_folds=3)
    df = ens.predict(SMILES[:3], context=ens.transform_context(CTX)).to_frame()
    assert len(df) == 3
    assert {"prediction", "std", "ci_lower_95", "ci_upper_95"} <= set(df.columns)


def test_weights_can_be_given_explicitly(data):
    ens = FusionEnsemble.fit(fast_gp, data, task="pdc50", n_members=2, n_folds=3)
    ens.set_weights([0.25, 0.75])
    pred = ens.predict(SMILES[:2], context=ens.transform_context(CTX))
    members = np.array(list(pred.member_predictions.values()))
    assert np.allclose(pred.mean, 0.25 * members[0] + 0.75 * members[1])


def test_predict_matrix_scores_an_encoded_design_matrix(data):
    """Evaluating on a held-out fold needs the already-encoded rows, not records."""
    ens = FusionEnsemble.fit(fast_gp, data, task="pdc50", n_members=2, n_folds=3)
    idx, X, y, groups = data.task_rows("pdc50")
    pred = ens.predict_matrix(X[:5])
    assert pred.mean.shape == (5,) and np.isfinite(pred.mean).all()
    assert (pred.std > 0).all()


def test_predict_matrix_agrees_with_the_record_path(data):
    ens = FusionEnsemble.fit(fast_gp, data, task="pdc50", n_members=2, n_folds=3)
    row = data.table.iloc[0]
    record = {"smiles": row["smiles"], "poi_seq": row["poi_seq"], "e3_seq": row["e3_seq"],
              "cell_id": row["cell_key"], "assay": row["assay_raw"],
              "assay_time": row["assay_time"]}
    from_matrix = ens.predict_matrix(data.X[:1])
    from_record = ens.predict([record])
    assert np.allclose(from_matrix.mean, from_record.mean, rtol=0, atol=0)
    assert np.allclose(from_matrix.std, from_record.std, rtol=0, atol=0)
