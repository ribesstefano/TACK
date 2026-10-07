"""The context path shares its work: featurise once, build the full matrix only if needed."""
import numpy as np
import pytest

from fusion_fixtures import CELLS, DEFAULT_BLOCKS, SEQS, SMILES, build_ensemble
from tackai.fusion.data import FusionData
from tackai.fusion.gp import GPInteraction

CTX = {"poi_seq": SEQS["poi"][0], "e3_seq": SEQS["e3"][0], "cell_id": CELLS[0],
       "assay": "western blot", "assay_time": 24.0}


@pytest.fixture
def data(fake_cache, tiny_csv):
    return FusionData.from_csv([tiny_csv], cache=False)


def fast_gp(**kw):
    return GPInteraction(**{"blocks": DEFAULT_BLOCKS, **kw}, n_restarts=1, n_iter=5,
                         max_hyper_points=40)


@pytest.fixture
def ens(data):
    return build_ensemble(fast_gp, data, task="pdc50", n_members=3, n_folds=3)


def count_calls(monkeypatch, obj, name):
    calls = []
    original = getattr(obj, name)

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(obj, name, spy)
    return calls


def test_invalid_neighbours_do_not_disturb_the_valid_rows(ens):
    ctx = ens.transform_context(CTX)
    clean = ens.predict(SMILES[:4], context=ctx)
    mixed = ens.predict([SMILES[0], "bad", SMILES[1], SMILES[2], "worse", SMILES[3]], context=ctx)
    keep = mixed.ok
    assert np.allclose(mixed.mean[keep], clean.mean, rtol=1e-4, atol=1e-4)
    assert np.isnan(mixed.mean[~keep]).all()


def test_an_all_invalid_batch_is_all_nan_and_not_an_error(ens):
    pred = ens.predict(["bad", "worse"], context=ens.transform_context(CTX))
    assert np.isnan(pred.mean).all() and not pred.ok.any()


def test_the_context_path_agrees_with_the_records_path(ens):
    ctx = ens.transform_context(CTX)
    fast = ens.predict(SMILES[:4], context=ctx)
    slow = ens.predict([{"smiles": s, **CTX} for s in SMILES[:4]])
    assert np.allclose(fast.mean, slow.mean, rtol=1e-3, atol=1e-3)
    assert np.allclose(fast.std, slow.std, rtol=1e-3, atol=1e-3)


def test_folded_members_never_assemble_a_full_matrix(ens, monkeypatch):
    ctx = ens.transform_context(CTX)
    calls = count_calls(monkeypatch, ens.data, "assemble_features")
    ens.predict(SMILES[:4], context=ctx)
    assert calls == []


def test_the_records_path_featurises_once(ens, monkeypatch):
    records = [{"smiles": s, **CTX} for s in SMILES[:4]]
    ens.predict(records)
    calls, real = [], ens.data.featurizer.featurize
    monkeypatch.setattr(ens.data.featurizer, "featurize", lambda s: calls.append(1) or real(s))
    ens.predict(records)
    assert len(calls) == 1


def test_encode_can_also_return_which_rows_parsed(data):
    records = [{"smiles": s, **CTX} for s in (SMILES[0], "not a molecule")]
    X = data.encode(records)
    assert isinstance(X, np.ndarray)
    X2, ok = data.encode(records, return_ok=True)
    assert np.array_equal(X, X2) and ok.tolist() == [True, False]
    X0, ok0 = data.encode([], return_ok=True)
    assert X0.shape == (0, data.n_columns) and ok0.shape == (0,)
