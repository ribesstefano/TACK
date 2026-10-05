"""Every member scores a context through one transform, whatever fold it was fitted on."""
import numpy as np
import pytest

from fusion_fixtures import CELLS, SEQS, SMILES
from tackai.fusion.blocks import LEGACY_SCALE_BLOCKS, BlockPreprocessor
from tackai.fusion.data import FusionData
from tackai.fusion.ensemble import FusionEnsemble
from tackai.fusion.models import GPInteraction, XGBoostFusion

CTX = {"poi_seq": SEQS["poi"][0], "e3_seq": SEQS["e3"][0], "cell_id": CELLS[0],
       "assay": "western blot", "assay_time": 24.0}
TASK = "pdc50"


@pytest.fixture
def data(fake_cache, tiny_csv):
    return FusionData.from_csv([tiny_csv], cache=False)


def fast_gp(**kw):
    return GPInteraction(n_restarts=1, n_iter=5, max_hyper_points=40, **kw)


def fast_xgb(**kw):
    return XGBoostFusion(n_estimators=20, grid=[{"max_depth": 3, "reg_lambda": 5.0}], **kw)


@pytest.fixture
def fitted(data):
    return FusionEnsemble.fit(fast_gp, data, task=TASK, n_members=3, n_folds=3)


def make_members_disagree(members, data, order=None):
    """Give each member a legacy-scaled preprocessor fitted on a different part of the rows,
    as members fitted on different folds have. Predictions become meaningless; the context
    each member sees is what is under test."""
    _, X, _, _ = data.task_rows(TASK)
    # A seeded permutation, not contiguous thirds: the tiny fixture's rows cycle through the same
    # cell / POI / ligase pattern, so contiguous parts have identical statistics and the members
    # would not actually disagree.
    parts = np.array_split(np.random.default_rng(0).permutation(len(X)), len(members))
    for k, member in enumerate(members):
        rows = parts[k if order is None else order[k]]
        member.pre_ = BlockPreprocessor(scale_blocks=LEGACY_SCALE_BLOCKS,
                                        dtype=member.dtype).fit(X[rows])
    return members


def count_calls(monkeypatch, cls, name):
    calls, original = [], getattr(cls, name)

    def spy(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(cls, name, spy)
    return calls


def test_every_member_is_handed_the_same_context(fitted, data):
    members = make_members_disagree(fitted.members, data)
    shared = FusionEnsemble(members, data, TASK, shared_context=True)
    own = FusionEnsemble(members, data, TASK, shared_context=False)

    ctx = shared.transform_context(CTX)
    assert all(blocks is ctx.per_member[0] for blocks in ctx.per_member)

    per_member = own.transform_context(CTX).per_member          # the defect this removes
    assert not np.allclose(per_member[0]["cell"], per_member[1]["cell"])


def test_the_shared_flag_off_keeps_per_member_contexts(fitted, data):
    ens = FusionEnsemble(fitted.members, data, TASK, shared_context=False)
    assert ens.context_pre_ is None and ens.shared_context is False


def test_the_context_is_transformed_once_not_once_per_member(fitted, data, monkeypatch):
    shared = FusionEnsemble(fitted.members, data, TASK, shared_context=True)
    own = FusionEnsemble(fitted.members, data, TASK, shared_context=False)
    calls = count_calls(monkeypatch, BlockPreprocessor, "transform_blocks")
    shared.transform_context(CTX)
    assert len(calls) == 1
    calls.clear()
    own.transform_context(CTX)
    assert len(calls) == len(own.members)


def test_all_three_paths_use_the_shared_context(fitted, data):
    members = make_members_disagree(fitted.members, data)
    ens = FusionEnsemble(members, data, TASK, shared_context=True)
    records = [{"smiles": s, **CTX} for s in SMILES[:4]]
    from_records = ens.predict(records)
    from_matrix = ens.predict_matrix(data.encode(records))
    from_context = ens.predict(SMILES[:4], context=ens.transform_context(CTX))
    assert np.array_equal(from_records.mean, from_matrix.mean)
    assert np.allclose(from_records.mean, from_context.mean, rtol=1e-3, atol=1e-3)
    unshared = FusionEnsemble(members, data, TASK, shared_context=False).predict(records)
    assert not np.allclose(from_records.mean, unshared.mean)     # the toggle does something


def test_the_context_does_not_depend_on_which_member_saw_which_rows(fitted, data):
    members = fitted.members
    make_members_disagree(members, data, order=[0, 1, 2])
    a = FusionEnsemble(members, data, TASK).transform_context(CTX).per_member[0]
    a = {b: arr.copy() for b, arr in a.items()}
    make_members_disagree(members, data, order=[2, 0, 1])
    b = FusionEnsemble(members, data, TASK).transform_context(CTX).per_member[0]
    for block in a:
        assert np.allclose(a[block], b[block], rtol=1e-5, atol=1e-6), block


def test_members_that_disagree_on_scaling_are_refused_by_name(fitted, data):  # Review Focus 5
    _, X, _, _ = data.task_rows(TASK)
    fitted.members[1].pre_ = BlockPreprocessor(scale_blocks=LEGACY_SCALE_BLOCKS).fit(X)
    with pytest.raises(ValueError, match="e3"):
        FusionEnsemble(fitted.members, data, TASK)


def test_a_missing_assay_time_is_imputed_from_the_consensus(fitted):  # Review Focus 5
    record = {k: v for k, v in CTX.items() if k != "assay_time"}
    ctx = fitted.transform_context(record)
    assert np.isfinite(ctx.per_member[0]["assay_time"]).all()
    assert np.isfinite(fitted.predict(SMILES[:3], context=ctx).mean).all()


def test_gp_and_xgboost_members_share_the_context(data):  # Review Focus 5
    ens = FusionEnsemble.fit([fast_gp, fast_xgb], data, task=TASK, n_folds=3)
    ctx = ens.transform_context(CTX)
    assert ctx.per_member[0] is ctx.per_member[1]
    assert ctx.folds[0] is not None and ctx.folds[1] is None
    pred = ens.predict(SMILES[:4], context=ctx)
    assert np.isfinite(pred.mean).all()
    assert np.allclose(pred.mean, np.average(list(pred.member_predictions.values()), axis=0,
                                             weights=list(ens.weights.values())))


def test_scaled_float64_members_from_the_old_code_still_load_and_score(data):  # Review Focus 1
    ens = FusionEnsemble.fit(lambda **kw: fast_gp(dtype="float64", **kw), data, task=TASK,
                             n_members=2, n_folds=3)
    make_members_disagree(ens.members, data)
    for member in ens.members:
        del member.pre_.dtype                                    # written before dtype existed
    old = FusionEnsemble(ens.members, data, TASK)
    assert old.context_pre_.steps_["cell"][1] is not None        # a scaling consensus
    records = [{"smiles": s, **CTX} for s in SMILES[:3]]
    assert np.isfinite(old.predict(records).mean).all()
