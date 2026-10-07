"""The train-side cache and the one-time block weights change nothing but the work done."""
import numpy as np
import pytest
import torch

from tackai.fusion.gp import MOL_KERNEL_BLOCKS, AdditiveProductGP

from test_fusion_gp import dims_of, toy


def uncached_predict(gp, Z, return_std=False):
    """predict() as it was before the cache: every train-side term recomputed per call."""
    with torch.no_grad():
        Zt = gp._to_tensor(Z)
        cross = gp._assemble(gp.params_, Zt, gp.Z_train_, gp._cache_distances(Zt, gp.Z_train_))
        mean = (gp.params_["mean"] + (cross @ gp.alpha_).squeeze(1)).numpy()
        if not return_std:
            return mean
        v = torch.linalg.solve_triangular(gp.chol_, cross.T.to(gp.chol_.dtype), upper=False)
        var = gp._diag(gp.params_, Zt) - (v * v).sum(0)
        return mean, torch.sqrt(var.clamp_min(0.0)).numpy()


def uncached_predict_in_context(gp, mol_blocks, fold, return_std=False):
    """predict_in_context() as it was: train-side terms recomputed, every block weight re-tested."""
    with torch.no_grad():
        Zm = {b: gp._block_tensor(b, mol_blocks[b]) for b in MOL_KERNEL_BLOCKS}
        cached = {b: gp._sq_dists(Zm[b], gp.Z_train_[b])
                  for b in MOL_KERNEL_BLOCKS if b not in gp.ard_blocks}
        rbf = {b: gp._rbf(b, gp.params_, Zm, gp.Z_train_, cached) for b in MOL_KERNEL_BLOCKS}
        scaled = None
        for b in MOL_KERNEL_BLOCKS:
            term = torch.exp(gp.params_[f"sc:rbf:{b}"]) * rbf[b]
            scaled = term if scaled is None else scaled + term
        mol = sum(rbf[b] for b in MOL_KERNEL_BLOCKS)
        cross = scaled + mol * fold["mol_weight"][None, :] + fold["const"][None, :]
        for block, weight in fold.get("block_weight", {}).items():
            if bool(torch.any(weight != 0)):
                cross = cross + rbf[block] * weight[None, :]
        mean = (gp.params_["mean"] + (cross @ gp.alpha_).squeeze(1)).numpy()
        if not return_std:
            return mean
        v = torch.linalg.solve_triangular(gp.chol_, cross.T.to(gp.chol_.dtype), upper=False)
        var = fold["prior_var"] - (v * v).sum(0)
        return mean, torch.sqrt(var.clamp_min(0.0)).numpy()


def fitted(interactions=None, n=80):
    Z, y = toy(n=n)
    Z["e3"] = np.random.default_rng(1).normal(size=(n, 3))
    kw = {} if interactions is None else {"interactions": interactions}
    gp = AdditiveProductGP(dims_of(Z), **kw)
    gp.fit(Z, y, n_restarts=1, n_iter=15, seed=0)
    return gp, Z


def context_of(Z):
    return {b: Z[b][:1] for b in ("e3", "poi", "cell", "assay_time")}


def test_cached_predict_is_bit_identical_to_the_uncached_formula():
    gp, Z = fitted()
    q = {b: a[:23] for b, a in Z.items()}
    for return_std in (False, True):
        got, ref = gp.predict(q, return_std=return_std), uncached_predict(gp, q, return_std)
        pairs = zip(got, ref) if return_std else [(got, ref)]
        for g, r in pairs:
            assert np.array_equal(g, r)


@pytest.mark.parametrize("interactions", [None, ("fingerprint*poi", "descriptors*cell", "poi*cell")])
def test_cached_predict_in_context_is_bit_identical_to_the_uncached_formula(interactions):
    gp, Z = fitted(interactions)
    fold = gp.fold_context(context_of(Z))
    mols = {b: Z[b][:17] for b in MOL_KERNEL_BLOCKS}
    got = gp.predict_in_context(mols, fold, return_std=True)
    ref = uncached_predict_in_context(gp, mols, fold, return_std=True)
    assert np.array_equal(got[0], ref[0]) and np.array_equal(got[1], ref[1])


def test_the_train_side_cache_is_built_once_per_conditioning(monkeypatch):
    gp, Z = fitted()
    calls = []
    original = AdditiveProductGP._build_train_side
    monkeypatch.setattr(AdditiveProductGP, "_build_train_side",
                        lambda self: calls.append(1) or original(self))
    q = {b: a[:5] for b, a in Z.items()}
    for _ in range(3):
        gp.predict(q, return_std=True)
    assert calls == []                                 # served from the cache built at load_state
    gp.load_state(gp.state_, Z, gp.y_train_.numpy())   # a new conditioning rebuilds it
    assert calls == [1]


def test_a_gp_pickled_before_the_cache_existed_still_predicts_the_same():  # Review Focus 1
    gp, Z = fitted()
    q = {b: a[:9] for b, a in Z.items()}
    before = gp.predict(q, return_std=True)
    del gp._train_side_                                # what an old artifact looks like
    after = gp.predict(q, return_std=True)
    assert np.array_equal(before[0], after[0]) and np.array_equal(before[1], after[1])


def test_fold_context_resolves_the_active_block_weights_once():
    gp, Z = fitted()                          # default interactions are all "mol*..."
    assert gp.fold_context(context_of(Z))["active_blocks"] == []
    gp2, Z2 = fitted(interactions=("fingerprint*poi",))
    assert gp2.fold_context(context_of(Z2))["active_blocks"] == ["fingerprint"]


def test_a_fold_without_active_blocks_still_works():
    gp, Z = fitted(interactions=("fingerprint*poi", "descriptors*cell"))
    fold = gp.fold_context(context_of(Z))
    mols = {b: Z[b][:6] for b in MOL_KERNEL_BLOCKS}
    expected = gp.predict_in_context(mols, fold)
    fold.pop("active_blocks")
    assert np.array_equal(gp.predict_in_context(mols, fold), expected)
