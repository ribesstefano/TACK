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
