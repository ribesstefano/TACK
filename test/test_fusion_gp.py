"""The exact GP: additive block kernels, product interactions, cached distances, exact variance."""
import numpy as np
import pytest

from tackai.fusion.gp import AdditiveProductGP


def toy(n=60, seed=0):
    """Two molecular blocks with a genuine product interaction, plus a linear small block."""
    rng = np.random.default_rng(seed)
    Z = {"fingerprint": rng.normal(size=(n, 4)), "descriptors": rng.normal(size=(n, 3)),
         "poi": rng.normal(size=(n, 2)), "cell": rng.normal(size=(n, 2)),
         "assay_time": rng.normal(size=(n, 1))}
    y = (np.sin(Z["fingerprint"][:, 0]) + Z["poi"][:, 0] * Z["fingerprint"][:, 1]
         + 0.5 * Z["assay_time"][:, 0] + 0.01 * rng.normal(size=n))
    return Z, y


def dims_of(Z):
    return {b: a.shape[1] for b, a in Z.items()}


def test_fit_then_predict_recovers_training_signal():
    Z, y = toy()
    gp = AdditiveProductGP(dims_of(Z))
    gp.fit(Z, y, n_restarts=1, n_iter=40, seed=0)
    pred = gp.predict(Z)
    assert pred.shape == (60,)
    assert np.corrcoef(pred, y)[0, 1] > 0.9


def test_predict_returns_std_and_it_is_smaller_on_training_points():
    Z, y = toy()
    gp = AdditiveProductGP(dims_of(Z))
    gp.fit(Z, y, n_restarts=1, n_iter=40, seed=0)
    _, std_train = gp.predict(Z, return_std=True)
    far = {b: a + 50.0 for b, a in Z.items()}
    _, std_far = gp.predict(far, return_std=True)
    assert (std_train > 0).all()
    assert std_far.mean() > std_train.mean() * 2


def test_variance_matches_the_textbook_formula():
    """var = k** - ks' (K + sI)^-1 ks, assembled independently from the fitted kernel."""
    Z, y = toy(n=40)
    gp = AdditiveProductGP(dims_of(Z))
    gp.fit(Z, y, n_restarts=1, n_iter=20, seed=0)
    _, std = gp.predict(Z, return_std=True)
    K = gp.kernel_matrix(Z, Z)
    Kn = K + np.eye(len(y)) * gp.noise_
    naive = np.sqrt(np.clip(np.diag(K) - np.einsum("ij,jk,ki->i", K, np.linalg.inv(Kn), K),
                            0, None))
    assert np.allclose(std, naive, rtol=1e-5, atol=1e-7)


def test_cached_distances_equal_a_naive_recomputation():
    Z, y = toy()
    gp = AdditiveProductGP(dims_of(Z))
    state = gp.fit(Z, y, n_restarts=1, n_iter=20, seed=0)
    cached = gp.predict(Z)
    fresh = AdditiveProductGP(dims_of(Z))
    fresh.load_state(state, Z, y)
    assert np.allclose(cached, fresh.predict(Z), rtol=1e-10, atol=1e-12)


def test_ard_block_learns_one_lengthscale_per_column():
    Z, y = toy()
    gp = AdditiveProductGP(dims_of(Z), ard_blocks=("descriptors",))
    gp.fit(Z, y, n_restarts=1, n_iter=20, seed=0)
    report = gp.kernel_report()
    assert np.shape(report["lengthscale"]["descriptors"]) == (3,)
    assert np.shape(report["lengthscale"]["fingerprint"]) == ()


def test_ard_absorbs_a_wildly_scaled_column():
    """A column 1e18 larger must not swamp the kernel when ARD is on."""
    Z, y = toy()
    Z["descriptors"][:, 0] *= 1e18
    gp = AdditiveProductGP(dims_of(Z), ard_blocks=("descriptors",))
    gp.fit(Z, y, n_restarts=1, n_iter=60, seed=0)
    assert np.corrcoef(gp.predict(Z), y)[0, 1] > 0.85


def test_product_kernels_can_be_disabled():
    Z, y = toy()
    gp = AdditiveProductGP(dims_of(Z), interactions=())
    gp.fit(Z, y, n_restarts=1, n_iter=20, seed=0)
    assert "prod:mol*poi" not in gp.kernel_report()["weight"]


def test_interactions_are_configurable_including_mol_x_e3():
    Z, y = toy()
    Z["e3"] = Z["cell"].copy()
    gp = AdditiveProductGP(dims_of(Z), interactions=("mol*e3",))
    gp.fit(Z, y, n_restarts=1, n_iter=20, seed=0)
    assert "prod:mol*e3" in gp.kernel_report()["weight"]


def test_fit_is_reproducible_for_a_seed():
    Z, y = toy()
    a = AdditiveProductGP(dims_of(Z))
    a.fit(Z, y, n_restarts=2, n_iter=20, seed=7)
    b = AdditiveProductGP(dims_of(Z))
    b.fit(Z, y, n_restarts=2, n_iter=20, seed=7)
    assert np.allclose(a.predict(Z), b.predict(Z), rtol=1e-10, atol=1e-12)


def test_hyper_fit_subsamples_but_conditions_on_all_rows():
    Z, y = toy(n=80)
    gp = AdditiveProductGP(dims_of(Z))
    gp.fit(Z, y, n_restarts=1, n_iter=10, seed=0, max_hyper_points=30)
    assert gp.n_train_ == 80 and gp.n_hyper_ == 30


def test_cholesky_recovers_from_an_ill_conditioned_kernel():
    Z, y = toy(n=30)
    for b in Z:
        Z[b][5] = Z[b][4]          # an exactly duplicated row
    gp = AdditiveProductGP(dims_of(Z), jitter=1e-10)
    gp.fit(Z, y, n_restarts=1, n_iter=10, seed=0)
    assert np.isfinite(gp.predict(Z)).all()


def test_fit_survives_many_duplicated_rows():
    """Real data repeats measurements of one compound in one context: K is then singular.

    Without a noise floor the Cholesky inside the marginal likelihood fails outright, which
    is how this first showed up on the development table (leading minor of order 1066).
    """
    Z, y = toy(n=60)
    for b in Z:                       # 20 distinct rows, each measured three times
        Z[b] = np.repeat(Z[b][:20], 3, axis=0)
    y = np.repeat(y[:20], 3) + 1e-9 * np.arange(60)
    gp = AdditiveProductGP(dims_of(Z))
    gp.fit(Z, y, n_restarts=2, n_iter=30, seed=0)
    assert np.isfinite(gp.predict(Z)).all()
    assert gp.noise_ >= 1e-3


def test_noise_has_a_floor():
    Z, y = toy()
    gp = AdditiveProductGP(dims_of(Z))
    gp.fit(Z, y, n_restarts=1, n_iter=30, seed=0)
    assert gp.noise_ >= 1e-3


def test_singular_kernel_with_vanishing_noise_still_fits(monkeypatch):
    """The real failure: a rank-deficient K while the noise parameter sits near zero.

    On the development table the Cholesky inside the marginal likelihood gave
    "leading minor of order 1066 is not positive-definite". A noise floor is what the
    GPyTorch original had (GreaterThan(1e-3)) and what keeps K + noise.I factorisable.
    """
    import torch
    Z, y = toy(n=60)
    for b in Z:                       # 20 distinct rows, each measured three times
        Z[b] = np.repeat(Z[b][:20], 3, axis=0)
    y = np.repeat(y[:20], 3)

    original = AdditiveProductGP._init_params

    def near_zero_noise(self, rng):
        params = original(self, rng)
        params["noise"] = torch.tensor(-25.0, dtype=self.dtype, requires_grad=True)
        return params

    monkeypatch.setattr(AdditiveProductGP, "_init_params", near_zero_noise)
    gp = AdditiveProductGP(dims_of(Z), jitter=0.0)   # no jitter: only the floor can save it
    gp.fit(Z, y, n_restarts=1, n_iter=10, seed=0)
    assert np.isfinite(gp.predict(Z)).all()


def test_rbf_self_covariance_is_one_at_any_column_scale():
    """RBF(x, x) must be exactly 1 however large the raw column values are.

    The descriptor block is raw by design and Ipc reaches 1e20. Computing squared distances
    as |a|^2 + |b|^2 - 2a.b then loses the diagonal: its true value is 0, but the absolute
    error of the cancellation is ~1e19, so exp(-d2/2l^2) returns 0 where it must return 1.
    On the real dmax data that gave rbf:descriptors a minimum eigenvalue of -3.7 and the
    conditioning Cholesky failed outright.
    """
    import torch

    for scale in (1.0, 1e9, 1e18):
        Z, y = toy(n=50)
        Z["descriptors"] = Z["descriptors"] * scale
        gp = AdditiveProductGP(dims_of(Z))
        gp.fit(Z, y, n_restarts=1, n_iter=5, seed=0)
        Zt = gp._to_tensor(Z)
        cached = gp._cache_distances(Zt, Zt)
        with torch.no_grad():
            for block in gp.rbf_blocks:
                diag = gp._rbf(block, gp.params_, Zt, Zt, cached).diagonal().numpy()
                assert np.allclose(diag, 1.0, atol=1e-12), (
                    f"scale {scale:.0e}: RBF({block}, x, x) = {diag.min():.6f}, not 1")


def test_raw_descriptor_scale_gives_a_positive_definite_kernel():
    """The assembled kernel of a 1e18-scale block must still be factorisable."""
    Z, y = toy(n=120)
    Z["descriptors"] = Z["descriptors"] * 6e18
    gp = AdditiveProductGP(dims_of(Z))
    gp.fit(Z, y, n_restarts=1, n_iter=20, seed=0)
    K = gp.kernel_matrix(Z, Z)
    assert np.isfinite(K).all()
    eigenvalues = np.linalg.eigvalsh(0.5 * (K + K.T))
    assert eigenvalues.min() > -1e-8 * eigenvalues.max(), "kernel is indefinite"
    mean, std = gp.predict(Z, return_std=True)
    assert np.isfinite(mean).all() and np.isfinite(std).all()


def test_folded_context_matches_the_full_kernel():
    """With a fixed context, the context-only kernel terms collapse to two vectors.

    Six of the ten terms (four context RBFs, the linear block, and poi*cell) do not depend on
    the molecule at all, and mol*poi / mol*cell are the molecular kernel times a per-training-
    row scalar. Folding them means a batch only pays for the two molecular distance matrices.
    """
    Z, y = toy(n=80)
    Z["e3"] = np.random.default_rng(1).normal(size=(80, 3))
    gp = AdditiveProductGP(dims_of(Z))
    gp.fit(Z, y, n_restarts=1, n_iter=20, seed=0)

    context = {b: Z[b][:1] for b in ("e3", "poi", "cell", "assay_time")}
    mols = {b: Z[b][:12] for b in ("fingerprint", "descriptors")}
    fold = gp.fold_context(context)

    folded_mean, folded_std = gp.predict_in_context(mols, fold, return_std=True)

    full = {**{b: np.repeat(context[b], 12, axis=0) for b in context}, **mols}
    mean, std = gp.predict(full, return_std=True)
    assert np.allclose(folded_mean, mean, rtol=1e-10, atol=1e-12)
    assert np.allclose(folded_std, std, rtol=1e-10, atol=1e-12)


def test_folded_context_rejects_a_molecule_only_interaction():
    """A product of two molecular kernels cannot be folded into a per-row scalar."""
    Z, y = toy(n=40)
    gp = AdditiveProductGP(dims_of(Z), interactions=("mol*mol",))
    gp.fit(Z, y, n_restarts=1, n_iter=5, seed=0)
    with pytest.raises(ValueError, match="mol"):
        gp.fold_context({b: Z[b][:1] for b in ("poi", "cell", "assay_time")})


@pytest.mark.parametrize("interactions", [
    ("mol*poi", "mol*cell", "poi*cell"),
    ("fingerprint*poi",),
    ("descriptors*cell",),
    ("fingerprint*poi", "mol*cell", "poi*cell"),
    ("mol*e3",),
    ("poi*poi",),
    (),
])
def test_folded_context_matches_the_full_kernel_for_any_interactions(interactions):
    """The fold must be algebraically exact for every interaction, not just the defaults.

    A side naming a single molecular block (fingerprint*poi) is not the same term as mol*poi:
    the molecular kernel of a fold is RBF_fp + RBF_desc, so folding a bare block against the
    shared molecular weight silently adds the other block's kernel to the term.
    """
    Z, y = toy(n=80)
    Z["e3"] = np.random.default_rng(1).normal(size=(80, 3))
    gp = AdditiveProductGP(dims_of(Z), interactions=interactions)
    gp.fit(Z, y, n_restarts=1, n_iter=15, seed=0)

    context = {b: Z[b][:1] for b in ("e3", "poi", "cell", "assay_time")}
    mols = {b: Z[b][:12] for b in ("fingerprint", "descriptors")}
    folded_mean, folded_std = gp.predict_in_context(mols, gp.fold_context(context),
                                                    return_std=True)
    full = {**{b: np.repeat(context[b], 12, axis=0) for b in context}, **mols}
    mean, std = gp.predict(full, return_std=True)
    assert np.allclose(folded_mean, mean, rtol=1e-10, atol=1e-12)
    assert np.allclose(folded_std, std, rtol=1e-10, atol=1e-12)


def test_cholesky_with_zero_jitter_raises_instead_of_hanging():
    """jitter=0 must still escalate: `0 * 10` is 0, so the loop would never terminate."""
    import torch
    gp = AdditiveProductGP({"fingerprint": 2, "descriptors": 2, "assay_time": 1}, jitter=0.0)
    indefinite = torch.tensor([[1.0, 2.0], [2.0, 1.0]], dtype=torch.float64)
    with pytest.raises(RuntimeError, match="positive definite"):
        gp._cholesky(indefinite)
