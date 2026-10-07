"""Stacking math: mixture MLE, conformal calibration, entropy decomposition.

Every test here uses small synthetic arrays — no FusionData, no real model fits — per the
design spec's instruction to keep these tests fast and focused on the stacking math itself.
"""
import numpy as np
import pytest
import xgboost as xgb

from fusion_fixtures import DEFAULT_BLOCKS
from tackai.fusion.data import FusionData
from tackai.fusion.ensemble import FusionEnsemble
from tackai.fusion.gp import GPInteraction
from tackai.fusion.stacking import (binary_entropy, conformal_quantile,
                                    entropy_decomposition, fit_mixture_weights,
                                    fit_pooled_weights, mixture_nll, mixture_nll_grad,
                                    select_lambda_classification, select_lambda_regression,
                                    softmax)


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


def test_select_lambda_regression_returns_a_value_from_the_grid_and_a_log_per_lambda():
    rng = np.random.default_rng(8)
    y = rng.normal(size=60)
    F = np.column_stack([y + rng.normal(scale=0.1, size=60), rng.normal(size=60)])
    S = np.ones_like(F)
    lambdas = (0.0, 0.1, 1.0)
    best, log = select_lambda_regression(F, S, y, lambdas=lambdas, sigma_min=1e-3, n_restarts=2, seed=0)
    assert best in lambdas
    assert set(log.keys()) == set(lambdas)
    assert all(np.isfinite(v) for v in log.values())


def test_select_lambda_regression_handles_tiny_fit_set_without_crashing():
    """D_fit smaller than n_splits must not raise a sklearn fold-count error."""
    y = np.array([0.1, 0.5, 0.9])
    F = np.column_stack([y, y[::-1]])
    S = np.ones_like(F)
    best, log = select_lambda_regression(F, S, y, lambdas=(0.0, 1.0), sigma_min=1e-3,
                                         n_restarts=1, seed=0, n_splits=5)
    assert best in (0.0, 1.0)


def test_select_lambda_classification_returns_a_value_from_the_grid():
    rng = np.random.default_rng(9)
    y = rng.integers(0, 2, size=60).astype(float)
    good = np.clip(y * 0.8 + (1 - y) * 0.2 + rng.normal(scale=0.05, size=60), 1e-6, 1 - 1e-6)
    P = np.column_stack([good, np.clip(rng.uniform(size=60), 1e-6, 1 - 1e-6)])
    lambdas = (0.0, 0.1, 1.0)
    best, log = select_lambda_classification(P, y, lambdas=lambdas, seed=0)
    assert best in lambdas
    assert set(log.keys()) == set(lambdas)


def _tiny_design_matrix(n=20, seed=0):
    """A design matrix shaped like FusionData's layout, with a trivial GP-friendly signal."""
    rng = np.random.default_rng(seed)
    width = sum(len(idx) for idx in DEFAULT_BLOCKS.values())
    X = rng.normal(size=(n, width)).astype(np.float32)
    y = X[:, 0] * 0.5 + rng.normal(scale=0.05, size=n)
    return X, y


def test_stacking_predict_dispatches_gp_member():
    X, y = _tiny_design_matrix()
    gp = GPInteraction(blocks=DEFAULT_BLOCKS, task_type="regression", n_restarts=1, n_iter=5,
                       max_hyper_points=20).fit(X, y)
    ens = FusionEnsemble([gp], data=_fake_data(), task="dmax")
    mean, sigma = ens._stacking_predict(gp, X)
    assert mean.shape == (20,)
    assert sigma is not None and sigma.shape == (20,) and np.all(sigma >= 0)


def test_stacking_predict_dispatches_xgb_regressor():
    X, y = _tiny_design_matrix()
    model = xgb.XGBRegressor(n_estimators=5, max_depth=2).fit(X, y)
    ens = FusionEnsemble([model], data=_fake_data(), task="dmax")
    mean, sigma = ens._stacking_predict(model, X)
    assert mean.shape == (20,)
    assert sigma is None


def test_stacking_predict_dispatches_xgb_classifier():
    X, y = _tiny_design_matrix()
    labels = (y > np.median(y)).astype(int)
    model = xgb.XGBClassifier(n_estimators=5, max_depth=2).fit(X, labels)
    ens = FusionEnsemble([model], data=_fake_data(), task="activity")
    mean, sigma = ens._stacking_predict(model, X)
    assert mean.shape == (20,)
    assert np.all((mean >= 0) & (mean <= 1))
    assert sigma is None


def test_stacking_predict_raises_for_unknown_member_type():
    X, _ = _tiny_design_matrix()
    ens = FusionEnsemble([object()], data=_fake_data(), task="dmax")
    with pytest.raises(TypeError, match="object"):
        ens._stacking_predict(ens.members[0], X)


def _fake_data():
    """A minimal stand-in exposing just what FusionEnsemble.__init__/_check_members read.

    FusionEnsemble only reads `.blocks_indexes` (for the GP block-layout check, which a
    bare XGBoost member skips entirely since it has no `.blocks` attribute); it does not
    need a real FusionData for these dispatch-only tests.
    """
    class _Data:
        blocks_indexes = DEFAULT_BLOCKS
    return _Data()


def test_fit_stacking_rejects_both_calling_conventions():
    X, y = _tiny_design_matrix(n=30)
    model = xgb.XGBRegressor(n_estimators=5, max_depth=2).fit(X, y)
    ens = FusionEnsemble([model], data=_fake_data(), task="dmax")
    with pytest.raises(ValueError, match="both"):
        ens.fit_stacking(X=X, y=y, X_fit=X[:10], y_fit=y[:10], X_cal=X[10:20], y_cal=y[10:20])


def test_fit_stacking_rejects_neither_calling_convention():
    X, y = _tiny_design_matrix(n=30)
    model = xgb.XGBRegressor(n_estimators=5, max_depth=2).fit(X, y)
    ens = FusionEnsemble([model], data=_fake_data(), task="dmax")
    with pytest.raises(ValueError, match="neither"):
        ens.fit_stacking()


def test_fit_stacking_rejects_nan_in_y():
    X, y = _tiny_design_matrix(n=30)
    y[0] = np.nan
    model = xgb.XGBRegressor(n_estimators=5, max_depth=2).fit(X[1:], y[1:])
    ens = FusionEnsemble([model], data=_fake_data(), task="dmax")
    with pytest.raises(ValueError, match="finite"):
        ens.fit_stacking(X=X, y=y)


def test_fit_stacking_regression_with_explicit_split_sets_fitted_attributes():
    X, y = _tiny_design_matrix(n=60, seed=1)
    good = xgb.XGBRegressor(n_estimators=20, max_depth=2).fit(X[:30], y[:30])
    bad = xgb.XGBRegressor(n_estimators=1, max_depth=1).fit(X[:30], np.random.default_rng(2).normal(size=30))
    ens = FusionEnsemble([good, bad], data=_fake_data(), task="dmax")
    ens.fit_stacking(X_fit=X[30:45], y_fit=y[30:45], X_cal=X[45:55], y_cal=y[45:55],
                     X_test=X[55:], y_test=y[55:], n_restarts=2)
    assert set(ens.weights_.keys()) == set(ens.names)
    assert sum(ens.weights_.values()) == pytest.approx(1.0, abs=1e-6)
    assert ens.weights_[ens.names[0]] > ens.weights_[ens.names[1]]  # "good" outweighs "bad"
    assert ens.lambda_ in (0.0, 0.01, 0.1, 1.0, 10.0)
    assert ens.X_cal_.shape == X[45:55].shape
    assert ens.X_test_.shape == X[55:].shape


def test_fit_stacking_regression_auto_split_without_groups():
    X, y = _tiny_design_matrix(n=90, seed=3)
    model = xgb.XGBRegressor(n_estimators=10, max_depth=2).fit(X[:30], y[:30])
    ens = FusionEnsemble([model], data=_fake_data(), task="dmax")
    ens.fit_stacking(X=X[30:], y=y[30:], n_restarts=2)
    assert ens.weights_[ens.names[0]] == pytest.approx(1.0, abs=1e-6)  # single member
    assert ens.X_cal_ is not None and ens.X_test_ is not None
    # 60 rows split 60/20/20 -> roughly 36/12/12; allow rounding slack
    assert 8 <= len(ens.X_cal_) <= 16


def test_fit_stacking_classification_with_explicit_split():
    X, y = _tiny_design_matrix(n=60, seed=4)
    labels = (y > np.median(y)).astype(float)
    good = xgb.XGBClassifier(n_estimators=20, max_depth=2).fit(X[:30], labels[:30])
    ens = FusionEnsemble([good], data=_fake_data(), task="activity")
    ens.fit_stacking(X_fit=X[30:40], y_fit=labels[30:40], X_cal=X[40:50], y_cal=labels[40:50])
    assert ens.weights_[ens.names[0]] == pytest.approx(1.0, abs=1e-6)
    assert ens.scales_ == {}


def test_fit_stacking_runs_end_to_end_with_mixed_gp_and_xgboost_membership():
    """Acceptance test from the design spec §9: fit_stacking must handle an ensemble
    whose members are a mix of GPInteraction and XGBRegressor."""
    X, y = _tiny_design_matrix(n=60, seed=13)
    gp = GPInteraction(blocks=DEFAULT_BLOCKS, task_type="regression", n_restarts=1, n_iter=5,
                       max_hyper_points=20).fit(X[:20], y[:20])
    tree = xgb.XGBRegressor(n_estimators=10, max_depth=2).fit(X[:20], y[:20])
    ens = FusionEnsemble([gp, tree], data=_fake_data(), task="dmax")
    ens.fit_stacking(X_fit=X[20:35], y_fit=y[20:35], X_cal=X[35:45], y_cal=y[35:45], n_restarts=2)
    assert set(ens.weights_.keys()) == set(ens.names)
    assert sum(ens.weights_.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(np.isfinite(v) for v in ens.scales_.values())
    out = ens.predict_stacked(X[45:])
    assert out["mean"].shape == (15,)
    assert np.all(np.isfinite(out["mean"])) and np.all(out["std"] >= 0)


def test_calibrate_stacking_raises_before_fit_stacking():
    X, y = _tiny_design_matrix(n=10)
    model = xgb.XGBRegressor(n_estimators=5, max_depth=2).fit(X, y)
    ens = FusionEnsemble([model], data=_fake_data(), task="dmax")
    with pytest.raises(RuntimeError, match="fit_stacking"):
        ens.calibrate_stacking()


def test_calibrate_stacking_regression_sets_q_hat_and_c():
    X, y = _tiny_design_matrix(n=90, seed=5)
    model = xgb.XGBRegressor(n_estimators=15, max_depth=2).fit(X[:30], y[:30])
    ens = FusionEnsemble([model], data=_fake_data(), task="dmax")
    ens.fit_stacking(X=X[30:], y=y[30:], n_restarts=2)
    ens.calibrate_stacking(alpha=0.1)
    assert ens.q_hat_ is not None and ens.q_hat_ > 0
    assert ens.c_ is not None and ens.c_ > 0


def test_calibrate_stacking_regression_warns_when_cal_set_too_small():
    X, y = _tiny_design_matrix(n=20, seed=6)
    model = xgb.XGBRegressor(n_estimators=5, max_depth=2).fit(X[:5], y[:5])
    ens = FusionEnsemble([model], data=_fake_data(), task="dmax")
    ens.fit_stacking(X_fit=X[5:10], y_fit=y[5:10], X_cal=X[10:12], y_cal=y[10:12], n_restarts=1)
    with pytest.warns(UserWarning, match="too small"):
        ens.calibrate_stacking(alpha=0.01)
    assert ens.q_hat_ == np.inf


def test_calibrate_stacking_classification_sets_temperature():
    X, y = _tiny_design_matrix(n=60, seed=7)
    labels = (y > np.median(y)).astype(float)
    model = xgb.XGBClassifier(n_estimators=15, max_depth=2).fit(X[:20], labels[:20])
    ens = FusionEnsemble([model], data=_fake_data(), task="activity")
    ens.fit_stacking(X=X[20:], y=labels[20:])
    ens.calibrate_stacking()
    assert ens.temperature_ > 0


def test_predict_stacked_raises_before_fit_stacking():
    X, _ = _tiny_design_matrix(n=10)
    model = xgb.XGBRegressor(n_estimators=5, max_depth=2).fit(X, np.zeros(10))
    ens = FusionEnsemble([model], data=_fake_data(), task="dmax")
    with pytest.raises(RuntimeError, match="fit_stacking"):
        ens.predict_stacked(X)


def test_predict_stacked_regression_without_calibration_omits_interval():
    X, y = _tiny_design_matrix(n=60, seed=8)
    model = xgb.XGBRegressor(n_estimators=15, max_depth=2).fit(X[:20], y[:20])
    ens = FusionEnsemble([model], data=_fake_data(), task="dmax")
    ens.fit_stacking(X=X[20:], y=y[20:])
    out = ens.predict_stacked(X[:5])
    assert set(out) == {"mean", "std", "std_noise", "std_disagreement", "lower", "upper"}
    assert out["mean"].shape == (5,)
    assert np.all(out["std"] >= 0)
    assert out["lower"] is None and out["upper"] is None


def test_predict_stacked_regression_with_calibration_gives_interval():
    X, y = _tiny_design_matrix(n=90, seed=9)
    model = xgb.XGBRegressor(n_estimators=15, max_depth=2).fit(X[:30], y[:30])
    ens = FusionEnsemble([model], data=_fake_data(), task="dmax")
    ens.fit_stacking(X=X[30:], y=y[30:])
    ens.calibrate_stacking(alpha=0.1)
    out = ens.predict_stacked(X[:5])
    assert out["lower"] is not None and out["upper"] is not None
    assert np.all(out["upper"] >= out["lower"])
    assert np.allclose(out["upper"] - out["mean"], ens.q_hat_ * out["std"])


def test_predict_stacked_classification_gives_entropy_decomposition():
    X, y = _tiny_design_matrix(n=60, seed=10)
    labels = (y > np.median(y)).astype(float)
    model = xgb.XGBClassifier(n_estimators=15, max_depth=2).fit(X[:20], labels[:20])
    ens = FusionEnsemble([model], data=_fake_data(), task="activity")
    ens.fit_stacking(X=X[20:], y=labels[20:])
    out = ens.predict_stacked(X[:5])
    assert set(out) == {"proba", "entropy_total", "entropy_aleatoric", "entropy_epistemic"}
    assert np.allclose(out["entropy_total"], out["entropy_aleatoric"] + out["entropy_epistemic"],
                       atol=1e-9)
    assert np.all(out["entropy_epistemic"] >= -1e-12)


def test_fit_stacking_classification_single_class_in_fit_set_raises():
    X, y = _tiny_design_matrix(n=30, seed=11)
    labels = np.ones(30)  # single class
    model = xgb.XGBClassifier(n_estimators=5, max_depth=2).fit(
        X[:10], (np.arange(10) % 2).astype(float))  # trained on a valid, disjoint slice
    ens = FusionEnsemble([model], data=_fake_data(), task="activity")
    with pytest.raises(ValueError, match="class"):
        ens.fit_stacking(X_fit=X[10:20], y_fit=labels[10:20], X_cal=X[20:], y_cal=labels[20:])


def test_fit_stacking_rejects_nan_in_explicit_split():
    X, y = _tiny_design_matrix(n=30, seed=12)
    X_fit = X[:10].copy()
    X_fit[0, 0] = np.nan
    model = xgb.XGBRegressor(n_estimators=5, max_depth=2).fit(X[10:20], y[10:20])
    ens = FusionEnsemble([model], data=_fake_data(), task="dmax")
    with pytest.raises(ValueError, match="finite"):
        ens.fit_stacking(X_fit=X_fit, y_fit=y[:10], X_cal=X[20:25], y_cal=y[20:25])
