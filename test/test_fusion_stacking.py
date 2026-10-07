"""Stacking math: mixture MLE, conformal calibration, entropy decomposition.

Every test here uses small synthetic arrays — no FusionData, no real model fits — per the
design spec's instruction to keep these tests fast and focused on the stacking math itself.
"""
import numpy as np
import pytest

from tackai.fusion.stacking import (binary_entropy, conformal_quantile,
                                    entropy_decomposition, fit_mixture_weights,
                                    fit_pooled_weights, mixture_nll, mixture_nll_grad, softmax)


def test_softmax_sums_to_one_and_matches_definition():
    theta = np.array([0.5, -1.0, 2.0])
    w = softmax(theta)
    assert w.sum() == pytest.approx(1.0)
    expected = np.exp(theta) / np.exp(theta).sum()
    assert np.allclose(w, expected)


def test_softmax_single_element_is_one():
    assert softmax(np.array([3.7])) == pytest.approx([1.0])


def test_mixture_nll_two_identical_models_equals_one_model_nll():
    """Averaging two copies of the same (f, sigma) column should give the same density
    as scoring the single model directly."""
    rng = np.random.default_rng(0)
    y = rng.normal(size=20)
    f_col = y + rng.normal(scale=0.1, size=20)
    F = np.column_stack([f_col, f_col])
    S = np.ones_like(F)
    theta = np.zeros(2)   # uniform weights
    phi = np.zeros(2)     # s_i = 1
    nll_two = mixture_nll(theta, phi, F, S, y, sigma_min=1e-6, lam=0.0)

    single_nll = -np.mean(
        -0.5 * np.log(2 * np.pi) - np.log(1.0) - 0.5 * ((y - f_col) / 1.0) ** 2
    )
    assert nll_two == pytest.approx(single_nll, abs=1e-6)


def test_mixture_nll_penalty_increases_loss_away_from_uniform():
    rng = np.random.default_rng(1)
    y = rng.normal(size=10)
    F = rng.normal(size=(10, 3))
    S = np.ones_like(F)
    theta_uniform = np.zeros(3)
    theta_skewed = np.array([3.0, -3.0, 0.0])
    phi = np.zeros(3)
    unpenalized_uniform = mixture_nll(theta_uniform, phi, F, S, y, sigma_min=1e-6, lam=0.0)
    unpenalized_skewed = mixture_nll(theta_skewed, phi, F, S, y, sigma_min=1e-6, lam=0.0)
    penalized_uniform = mixture_nll(theta_uniform, phi, F, S, y, sigma_min=1e-6, lam=10.0)
    penalized_skewed = mixture_nll(theta_skewed, phi, F, S, y, sigma_min=1e-6, lam=10.0)
    # The penalty term is 0 at uniform weights and positive away from it.
    assert penalized_uniform == pytest.approx(unpenalized_uniform, abs=1e-9)
    assert penalized_skewed > unpenalized_skewed


def test_mixture_nll_respects_sigma_floor():
    """A model that reproduces one label exactly must not blow up the loss to -inf;
    the sigma_min floor keeps the density finite."""
    y = np.array([0.0, 1.0, 2.0])
    F = np.column_stack([y, y + 5.0])     # first model is exact
    S = np.zeros_like(F)                   # raw sigma is 0 everywhere
    theta = np.zeros(2)
    phi = np.zeros(2)
    loss = mixture_nll(theta, phi, F, S, y, sigma_min=1e-3, lam=0.0)
    assert np.isfinite(loss)


def test_mixture_nll_grad_matches_finite_differences():
    rng = np.random.default_rng(2)
    F = rng.normal(size=(15, 4))
    S = np.abs(rng.normal(size=(15, 4))) + 0.1
    y = rng.normal(size=15)
    theta = rng.normal(scale=0.5, size=4)
    phi = rng.normal(scale=0.5, size=4)
    sigma_min, lam = 1e-3, 0.1

    grad_theta, grad_phi = mixture_nll_grad(theta, phi, F, S, y, sigma_min, lam)

    eps = 1e-5
    fd_theta = np.zeros(4)
    for i in range(4):
        tp, tm = theta.copy(), theta.copy()
        tp[i] += eps
        tm[i] -= eps
        fd_theta[i] = (mixture_nll(tp, phi, F, S, y, sigma_min, lam)
                      - mixture_nll(tm, phi, F, S, y, sigma_min, lam)) / (2 * eps)
    fd_phi = np.zeros(4)
    for i in range(4):
        pp, pm = phi.copy(), phi.copy()
        pp[i] += eps
        pm[i] -= eps
        fd_phi[i] = (mixture_nll(theta, pp, F, S, y, sigma_min, lam)
                    - mixture_nll(theta, pm, F, S, y, sigma_min, lam)) / (2 * eps)

    assert np.allclose(grad_theta, fd_theta, rtol=1e-4, atol=1e-6)
    assert np.allclose(grad_phi, fd_phi, rtol=1e-4, atol=1e-6)


def test_fit_mixture_weights_favors_the_accurate_model():
    """One model equals the truth plus small noise, the others are random noise:
    its weight should dominate at lam=0 (acceptance test from the design spec)."""
    rng = np.random.default_rng(3)
    y = rng.normal(size=200)
    good = y + rng.normal(scale=0.01, size=200)
    bad1 = rng.normal(size=200)
    bad2 = rng.normal(size=200)
    F = np.column_stack([good, bad1, bad2])
    S = np.ones_like(F)
    w, s, obj = fit_mixture_weights(F, S, y, lam=0.0, sigma_min=1e-3, n_restarts=5, seed=0)
    assert w[0] > 0.95
    assert np.isfinite(obj)


def test_fit_mixture_weights_identical_models_gives_uniform_weight_and_zero_disagreement():
    rng = np.random.default_rng(4)
    y = rng.normal(size=50)
    f = y + rng.normal(scale=0.2, size=50)
    F = np.column_stack([f, f, f])
    S = np.ones_like(F)
    w, s, obj = fit_mixture_weights(F, S, y, lam=0.0, sigma_min=1e-3, n_restarts=3, seed=0)
    assert w.sum() == pytest.approx(1.0)
    mean = F @ w
    disagreement = np.sum(w[None, :] * (F - mean[:, None]) ** 2, axis=1)
    assert np.allclose(disagreement, 0.0, atol=1e-8)


def test_fit_mixture_weights_is_deterministic_for_a_fixed_seed():
    rng = np.random.default_rng(5)
    y = rng.normal(size=30)
    F = rng.normal(size=(30, 3))
    S = np.ones_like(F)
    w1, s1, _ = fit_mixture_weights(F, S, y, lam=0.1, sigma_min=1e-3, n_restarts=5, seed=42)
    w2, s2, _ = fit_mixture_weights(F, S, y, lam=0.1, sigma_min=1e-3, n_restarts=5, seed=42)
    assert np.array_equal(w1, w2)
    assert np.array_equal(s1, s2)


def test_binary_entropy_is_zero_at_extremes_and_max_at_half():
    q = np.array([0.0, 1.0, 0.5])
    h = binary_entropy(q)
    assert h[0] == pytest.approx(0.0)
    assert h[1] == pytest.approx(0.0)
    assert h[2] == pytest.approx(np.log(2))


def test_fit_pooled_weights_favors_the_accurate_model():
    rng = np.random.default_rng(6)
    y = rng.integers(0, 2, size=200).astype(float)
    good = np.clip(y * 0.9 + (1 - y) * 0.1 + rng.normal(scale=0.02, size=200), 1e-6, 1 - 1e-6)
    bad = np.clip(rng.uniform(size=200), 1e-6, 1 - 1e-6)
    P = np.column_stack([good, bad])
    w, obj = fit_pooled_weights(P, y, lam=0.0, seed=0)
    assert w[0] > 0.95
    assert np.isfinite(obj)


def test_entropy_decomposition_sums_exactly_and_epistemic_is_nonnegative():
    rng = np.random.default_rng(7)
    P = np.clip(rng.uniform(size=(30, 4)), 1e-6, 1 - 1e-6)
    w = softmax(rng.normal(size=4))
    total, aleatoric, epistemic = entropy_decomposition(P, w)
    assert np.allclose(total, aleatoric + epistemic, atol=1e-10)
    assert np.all(epistemic >= -1e-12)


def test_entropy_decomposition_identical_models_has_zero_epistemic():
    p = np.array([0.3, 0.6, 0.9])
    P = np.column_stack([p, p, p])
    w = np.array([0.2, 0.3, 0.5])
    total, aleatoric, epistemic = entropy_decomposition(P, w)
    assert np.allclose(epistemic, 0.0, atol=1e-10)
    assert np.allclose(total, aleatoric, atol=1e-10)


def test_conformal_quantile_basic():
    residuals = np.array([0.1, 0.5, 0.2, 0.9, 0.3])
    q = conformal_quantile(residuals, alpha=0.2)
    # ceil((5+1)*0.8) = 5th smallest (1-indexed) of the 5 values -> the max
    assert q == pytest.approx(0.9)


def test_conformal_quantile_too_few_samples_warns_and_returns_inf():
    residuals = np.array([0.1, 0.2])
    with pytest.warns(UserWarning, match="too small"):
        q = conformal_quantile(residuals, alpha=0.01)
    assert q == np.inf
