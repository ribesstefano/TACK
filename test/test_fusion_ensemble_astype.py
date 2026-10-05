"""Saved float64 ensembles can be served in float32 without refitting."""
from functools import partial

import numpy as np
import pytest
import torch

from fusion_fixtures import CELLS, SEQS, SMILES
from tackai.fusion.data import FusionData
from tackai.fusion.ensemble import FusionEnsemble
from tackai.fusion.models import GPInteraction, XGBoostFusion

CTX = {"poi_seq": SEQS["poi"][0], "e3_seq": SEQS["e3"][0], "cell_id": CELLS[0],
       "assay": "western blot", "assay_time": 24.0}
RECORDS = [{"smiles": s, **CTX} for s in SMILES[:4]]


@pytest.fixture
def data(fake_cache, tiny_csv):
    return FusionData.from_csv([tiny_csv], cache=False)


def fast_gp(**kw):
    return GPInteraction(n_restarts=1, n_iter=5, max_hyper_points=40, **kw)


def fast_xgb(**kw):
    return XGBoostFusion(n_estimators=20, grid=[{"max_depth": 3, "reg_lambda": 5.0}], **kw)


@pytest.fixture
def ens64(data):
    return FusionEnsemble.fit(partial(fast_gp, dtype="float64"), data, task="pdc50",
                              n_members=2, n_folds=3)


def test_astype_casts_members_preprocessors_and_the_shared_context(ens64):
    before = ens64.predict(RECORDS)
    assert ens64.astype("float32") is ens64
    for member in ens64.members:
        assert member.model_.dtype is torch.float32 and member.pre_.dtype == "float32"
        assert member.dtype == "float32"
    assert ens64.context_pre_.dtype == "float32"
    after = ens64.predict(RECORDS)
    assert np.allclose(before.mean, after.mean, rtol=1e-2, atol=1e-2)


def test_promoted_reports_one_flag_per_member(ens64):
    ens64.astype("float32")
    assert len(ens64.promoted) == len(ens64.members)
    assert all(isinstance(flag, bool) for flag in ens64.promoted)


def test_from_pretrained_can_serve_in_another_precision(ens64, tmp_path):
    before = ens64.predict(RECORDS).mean
    ens64.save(tmp_path / "ens")
    loaded = FusionEnsemble.from_pretrained(tmp_path / "ens", dtype="float32")
    assert all(m.model_.dtype is torch.float32 for m in loaded.members)
    assert np.allclose(loaded.predict(RECORDS).mean, before, rtol=1e-2, atol=1e-2)
    kept = FusionEnsemble.from_pretrained(tmp_path / "ens")           # default: as stored
    assert all(m.model_.dtype is torch.float64 for m in kept.members)


def test_the_manifest_records_the_precision(ens64, tmp_path):
    import json
    ens64.save(tmp_path / "ens")
    assert json.loads((tmp_path / "ens" / "manifest.json").read_text())["dtype"] == "float64"


def test_astype_leaves_xgboost_members_working(data):
    ens = FusionEnsemble.fit([fast_gp, fast_xgb], data, task="pdc50", n_folds=3)
    ens.astype("float64")
    assert np.isfinite(ens.predict(RECORDS).mean).all()
    assert ens.promoted == [False, False]


def test_an_artifact_written_before_this_change_loads_and_scores_identically(ens64, tmp_path):
    """Review Focus 1: no preprocessor dtype, no train-side cache, no promoted_ flag."""
    before = ens64.predict(RECORDS)
    for member in ens64.members:
        del member.pre_.dtype
        del member.model_._train_side_
        del member.model_.promoted_
        del member.dtype
    ens64.save(tmp_path / "old")
    loaded = FusionEnsemble.from_pretrained(tmp_path / "old")
    after = loaded.predict(RECORDS)
    assert np.array_equal(before.mean, after.mean) and np.array_equal(before.std, after.std)
    assert loaded.promoted == [False, False]
