"""The context path does its shared work once, and changes no returned bit by doing so."""
import numpy as np
import pytest

from fusion_fixtures import CELLS, SEQS, SMILES
from tackai.fusion.blocks import BlockPreprocessor
from tackai.fusion.data import FusionData
from tackai.fusion.ensemble import FusionEnsemble
from tackai.fusion.models import GPInteraction

CTX = {"poi_seq": SEQS["poi"][0], "e3_seq": SEQS["e3"][0], "cell_id": CELLS[0],
       "assay": "western blot", "assay_time": 24.0}


@pytest.fixture
def data(fake_cache, tiny_csv):
    return FusionData.from_csv([tiny_csv], cache=False)


def fast_gp(**kw):
    return GPInteraction(n_restarts=1, n_iter=5, max_hyper_points=40, **kw)


@pytest.fixture
def ens(data):
    return FusionEnsemble.fit(fast_gp, data, task="pdc50", n_members=3, n_folds=3)


def count_calls(monkeypatch, cls, name):
    calls = []
    original = getattr(cls, name)

    def spy(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(cls, name, spy)
    return calls


def unhoisted_context_predict(ens, smiles, ctx):
    """The context path as it was: a full-width scratch row and one [valid] copy per member."""
    n = len(smiles)
    fp, desc, ok = ens.data.featurizer.featurize(smiles)
    valid = np.flatnonzero(ok)
    mol_row = np.zeros((n, ens.data.n_columns), dtype=np.float64)
    mol_row[:, ens.data.index["fingerprint"]] = fp
    mol_row[:, ens.data.index["descriptors"]] = desc
    per_member = []
    for member, fold in zip(ens.members, ctx.folds):
        mean, std = np.full(n, np.nan), np.zeros(n)
        Z = dict(member.pre_.transform_blocks(mol_row[valid],
                                              only=["fingerprint", "descriptors"]))
        mean[valid], std[valid] = ens._folded_scores(member, Z, fold, True)
        per_member.append((mean, std))
    return ens._aggregate(per_member, ok, smiles, True)


def test_context_path_is_bit_identical_to_the_unhoisted_reference(ens):
    ctx = ens.transform_context(CTX)
    smiles = SMILES[:4] + ["not a molecule"] + SMILES[4:6]
    got, ref = ens.predict(smiles, context=ctx), unhoisted_context_predict(ens, smiles, ctx)
    assert np.array_equal(got.mean, ref.mean, equal_nan=True)
    assert np.array_equal(got.std, ref.std, equal_nan=True)
    assert got.ok.tolist() == ref.ok.tolist() == [True] * 4 + [False] + [True] * 2


def test_invalid_neighbours_do_not_disturb_the_valid_rows(ens):  # Review Focus 2
    ctx = ens.transform_context(CTX)
    clean = ens.predict(SMILES[:4], context=ctx)
    mixed = ens.predict([SMILES[0], "bad", SMILES[1], SMILES[2], "worse", SMILES[3]], context=ctx)
    keep = mixed.ok
    assert np.allclose(mixed.mean[keep], clean.mean, rtol=1e-4, atol=1e-4)
    assert np.isnan(mixed.mean[~keep]).all()


def test_an_all_invalid_batch_is_all_nan_and_not_an_error(ens):  # Review Focus 2
    pred = ens.predict(["bad", "worse"], context=ens.transform_context(CTX))
    assert np.isnan(pred.mean).all() and not pred.ok.any()


def test_members_with_identical_molecular_transforms_share_one_result(ens, monkeypatch):
    ctx = ens.transform_context(CTX)
    ens.predict(SMILES[:4], context=ctx)                       # warm the featuriser memo
    calls = count_calls(monkeypatch, BlockPreprocessor, "transform_block")
    ens.predict(SMILES[:4], context=ctx)
    assert len(calls) == 2, "fingerprint and descriptors should be transformed once for all members"


def test_a_nan_feature_switches_sharing_off(ens, monkeypatch):  # Review Focus 2
    """With no NaN the imputer is a no-op and sharing is exact; with one it is not."""
    ctx = ens.transform_context(CTX)
    real = ens.data.featurizer.featurize

    def with_nan(smiles):
        fp, desc, ok = real(smiles)
        desc = desc.copy()
        desc[0, 3] = np.nan
        return fp, desc, ok
    monkeypatch.setattr(ens.data.featurizer, "featurize", with_nan)
    calls = count_calls(monkeypatch, BlockPreprocessor, "transform_block")
    pred = ens.predict(SMILES[:4], context=ctx)
    assert len(calls) == 2 * len(ens.members)
    assert np.isfinite(pred.mean).all()


def test_the_context_path_never_builds_a_full_width_row(ens, monkeypatch):
    ctx = ens.transform_context(CTX)
    calls = count_calls(monkeypatch, BlockPreprocessor, "transform_blocks")
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
