"""Contracts of the stateless design: a member's layout, the GP's own scaling, the assay-time default."""
from functools import partial

import numpy as np
import pytest

from fusion_fixtures import CELLS, DEFAULT_BLOCKS, SEQS, SMILES, build_ensemble
from tackai.fusion.context import ASSAY_TIME_DEFAULT
from tackai.fusion.data import FusionData
from tackai.fusion.ensemble import FusionEnsemble
from tackai.fusion.gp import AdditiveProductGP
from tackai.fusion.gp import GPInteraction

from test_fusion_gp import dims_of, toy

CTX = {"poi_seq": SEQS["poi"][0], "e3_seq": SEQS["e3"][0], "cell_id": CELLS[0],
       "assay": "western blot", "assay_time": 24.0}
RECORDS = [{"smiles": s, **CTX} for s in SMILES[:4]]


def fast_gp(**kw):
    return GPInteraction(**{"blocks": DEFAULT_BLOCKS, **kw}, n_restarts=1, n_iter=5,
                         max_hyper_points=40)


@pytest.fixture
def data(fake_cache, tiny_csv):
    return FusionData.from_csv([tiny_csv], cache=False)


@pytest.fixture
def ens(data):
    return build_ensemble(fast_gp, data, task="pdc50", n_members=2, n_folds=3)


# -- the GP applies 1/sqrt(width) itself, so callers hand it raw blocks ----------------------------

def test_the_gp_divides_every_non_linear_block_by_sqrt_width():
    Z, y = toy()
    gp = AdditiveProductGP(dims_of(Z))
    gp.fit(Z, y, n_restarts=1, n_iter=5, seed=0)
    for block, width in dims_of(Z).items():
        want = np.asarray(Z[block], dtype=np.float32)
        if block not in gp.linear_blocks:
            want = want / np.float32(np.sqrt(width))
        assert np.allclose(gp.Z_train_[block].numpy(), want, rtol=1e-6), block


def test_the_linear_block_is_left_at_its_own_scale():
    Z, y = toy()
    gp = AdditiveProductGP(dims_of(Z))
    gp.fit(Z, y, n_restarts=1, n_iter=5, seed=0)
    assert "assay_time" in gp.linear_blocks
    assert np.array_equal(gp.Z_train_["assay_time"].numpy(), np.asarray(Z["assay_time"], np.float32))


# -- a member must have been fitted on the layout of the data the ensemble is built on -------------

def test_an_ensemble_refuses_a_member_fitted_on_another_layout(data, ens):
    from tackai.fusion.training import fit_member
    other = data.blocks_indexes
    other["descriptors"] = other["descriptors"][::-1].copy()
    _, X, y, _ = data.task_rows("pdc50")
    odd = fit_member(partial(fast_gp, blocks=other), X, y)
    with pytest.raises(ValueError, match="block layout"):
        FusionEnsemble(list(ens.members) + [odd], data, "pdc50")


def test_the_estimators_take_their_layout_from_the_data(data):
    _, X, y, _ = data.task_rows("pdc50")
    est = fast_gp(blocks=data.blocks_indexes).fit(X, y)
    assert est.dims_ == data.dims


def test_a_gp_cannot_be_built_without_a_layout():
    with pytest.raises(TypeError):
        GPInteraction()


# -- a missing assay duration is the constant 24 h, never learned ----------------------------------

def test_a_missing_assay_time_becomes_the_constant_default(data):
    col = data.index["assay_time"][0]
    rows = [{"smiles": SMILES[0], **{k: v for k, v in CTX.items() if k != "assay_time"}},
            {"smiles": SMILES[0], **{**CTX, "assay_time": float("nan")}},
            {"smiles": SMILES[0], **{**CTX, "assay_time": 6.0}}]
    X = data.encode(rows)
    assert X[0, col] == X[1, col] == ASSAY_TIME_DEFAULT == 24.0 and X[2, col] == 6.0
    assert np.isfinite(X).all()


def test_the_training_matrix_holds_no_nan(data):
    assert np.isfinite(data.X).all()


def test_an_estimator_refuses_a_matrix_with_nan(data):
    _, X, y, _ = data.task_rows("pdc50")
    X = X.copy()
    X[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        fast_gp(blocks=data.blocks_indexes).fit(X, y)
