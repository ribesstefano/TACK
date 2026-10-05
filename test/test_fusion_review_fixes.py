"""Regressions found by the whole-branch review of the float32 / shared-context work."""
import pickle
from functools import partial

import numpy as np
import pytest
import torch
from sklearn.base import clone

from fusion_fixtures import CELLS, SEQS, SMILES, build_ensemble
from tackai.fusion.blocks import BlockPreprocessor, block_index
from tackai.fusion.data import FusionData
from tackai.fusion.ensemble import FusionEnsemble
from tackai.fusion.gp import AdditiveProductGP
from tackai.fusion.models import GPInteraction

from test_fusion_blocks import make_X
from test_fusion_dtype import design
from test_fusion_gp import dims_of, toy

CTX = {"poi_seq": SEQS["poi"][0], "e3_seq": SEQS["e3"][0], "cell_id": CELLS[0],
       "assay": "western blot", "assay_time": 24.0}
RECORDS = [{"smiles": s, **CTX} for s in SMILES[:4]]


def fast_gp(**kw):
    return GPInteraction(n_restarts=1, n_iter=5, max_hyper_points=40, **kw)


@pytest.fixture
def data(fake_cache, tiny_csv):
    return FusionData.from_csv([tiny_csv], cache=False)


@pytest.fixture
def ens64(data):
    return build_ensemble(partial(fast_gp, dtype="float64"), data, task="pdc50",
                              n_members=2, n_folds=3)


# -- an artifact pickled before ``dtype`` existed must stay a well-behaved sklearn object --------

def test_a_preprocessor_pickled_before_dtype_existed_supports_repr_and_clone():  # Review Focus 1
    X, _ = make_X()
    pre = BlockPreprocessor().fit(X)
    del pre.dtype                                     # what the old code wrote
    loaded = pickle.loads(pickle.dumps(pre))
    assert loaded.dtype == "float64" and loaded.get_params()["dtype"] == "float64"
    assert "BlockPreprocessor" in repr(loaded)
    assert clone(loaded).dtype == "float64"


# -- an unsupported precision must be refused before anything is changed --------------------------

def test_gp_refuses_an_unsupported_dtype_before_changing_anything():
    Z, y = toy()
    with pytest.raises(ValueError, match="float32"):
        AdditiveProductGP(dims_of(Z), dtype="float16")
    gp = AdditiveProductGP(dims_of(Z))
    gp.fit(Z, y, n_restarts=1, n_iter=5, seed=0)
    before = gp.predict(Z)
    with pytest.raises(ValueError, match="float32"):
        gp.astype("float16")
    assert gp.dtype is torch.float32 and gp.Z_train_["poi"].dtype == torch.float32
    assert np.array_equal(gp.predict(Z), before)


def test_ensemble_astype_refuses_an_unsupported_dtype_before_changing_anything(ens64):
    before = ens64.predict(RECORDS).mean
    with pytest.raises(ValueError, match="float32"):
        ens64.astype("float16")
    for member in ens64.members:
        assert member.dtype == "float64" and member.pre_.dtype == "float64"
        assert member.model_.dtype is torch.float64
    assert ens64.context_pre_.dtype == "float64"
    assert np.array_equal(ens64.predict(RECORDS).mean, before)


# -- a member with its own column layout is scored through that layout on every path --------------

def test_a_member_with_its_own_descriptor_order_is_scored_correctly_from_a_cached_context(data):
    """The descriptor kernel is ARD, so the order of its columns changes the prediction."""
    blocks = block_index(data.dims)
    blocks["descriptors"] = blocks["descriptors"][::-1].copy()
    ens = build_ensemble(partial(fast_gp, blocks=blocks, dtype="float64"), data,
                             task="pdc50", n_members=2, n_folds=3)
    from_context = ens.predict(SMILES[:4], context=ens.transform_context(CTX))
    from_records = ens.predict(RECORDS)
    assert np.allclose(from_context.mean, from_records.mean, rtol=1e-8, atol=1e-8)
    assert np.allclose(from_context.std, from_records.std, rtol=1e-6, atol=1e-8)


# -- save / load must not silently change which context transform is used -------------------------

def test_save_records_whether_the_context_is_shared(data, ens64, tmp_path):
    own = FusionEnsemble(ens64.members, data, "pdc50", shared_context=False)
    own.save(tmp_path / "own")
    assert FusionEnsemble.from_pretrained(tmp_path / "own").shared_context is False
    ens64.save(tmp_path / "shared")
    assert FusionEnsemble.from_pretrained(tmp_path / "shared").shared_context is True
    assert FusionEnsemble.from_pretrained(tmp_path / "own", shared_context=True).shared_context
    assert FusionEnsemble.from_pretrained(tmp_path / "shared",
                                          shared_context=False).shared_context is False


def test_a_manifest_without_shared_context_defaults_to_shared(data, ens64, tmp_path):
    import json
    ens64.save(tmp_path / "old")
    manifest = json.loads((tmp_path / "old" / "manifest.json").read_text())
    manifest.pop("shared_context", None)
    (tmp_path / "old" / "manifest.json").write_text(json.dumps(manifest))
    assert FusionEnsemble.from_pretrained(tmp_path / "old").shared_context is True
