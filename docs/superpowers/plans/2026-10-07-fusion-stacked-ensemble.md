# Stacked Ensemble Weights + Uncertainty Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add learned stacking weights and calibrated uncertainty to `FusionEnsemble` — mixture-of-Gaussians MLE for regression, log-loss-pooled probabilities for classification — as three new methods (`fit_stacking`, `calibrate_stacking`, `predict_stacked`), supporting mixed `GPInteraction`/`XGBRegressor`/`XGBClassifier` membership.

**Architecture:** The statistical core (mixture NLL + analytic gradient, softmax/exp reparameterization, lambda cross-validation, conformal quantile, binary entropy decomposition) is written as plain NumPy functions in a new module `tackai/fusion/stacking.py`, taking arrays only — no `FusionEnsemble`, no member objects. `FusionEnsemble` gains a small member-dispatch helper (`_stacking_predict`) and three thin orchestration methods that call into `stacking.py` for the math and store the fitted state (`weights_`, `scales_`, `lambda_`, `q_hat_`, `c_`, `temperature_`) on `self`. This split means the hard math is unit-tested directly on synthetic arrays (fast, no model fitting needed), while only a couple of integration tests exercise real fitted members.

**Tech Stack:** NumPy, SciPy (`scipy.optimize.minimize`, L-BFGS-B), scikit-learn (`train_test_split`, `GroupShuffleSplit` — already a dependency), xgboost (already a dependency, `>=2.0.0`). No new dependencies.

**Spec:** [docs/superpowers/specs/2026-10-07-fusion-stacked-ensemble-design.md](../specs/2026-10-07-fusion-stacked-ensemble-design.md)

## Global Constraints

- Tests must be concise and fast — small synthetic arrays, no large datasets, no long training runs. Verify stacking logic only (spec §9, final line).
- `OMP_NUM_THREADS=1` before any torch/xgboost import in test runs (CLAUDE.md) — `test/conftest.py` already does this; new test files just need to not bypass it (no direct `import torch`/`xgboost` before conftest loads, which pytest guarantees).
- No torch/JAX dependency added for the mixture-MLE optimization; pure NumPy + `scipy.optimize.minimize`.
- `predict()`/`predict_matrix()`/`_member_scores()`/`_aggregate()`/`calibrate()` on `FusionEnsemble` are untouched — stays GP-only, as today.
- No `XGBoostFusion`/M7 wrapper class is restored. Members may be bare `xgboost.XGBRegressor`/`xgboost.XGBClassifier`.
- Variant B, CV+, Section 7's model-selection/bootstrap reporting, and a standalone `StackedEnsemble` class are explicitly out of scope — no stubs, no `NotImplementedError` placeholders for them.
- `sigma_min = 1e-3 * std(y_fit)` (exact constant from the spec) — never omit the floor.
- Lambda grid is exactly `(0.0, 0.01, 0.1, 1.0, 10.0)` unless the caller overrides it.

## Review Focus

- **A single-member ensemble.** `fit_stacking` with `len(self.members) == 1`: softmax over one `theta` must still return weight 1.0, not divide-by-zero or NaN; disagreement component must be exactly 0.
- **`D_fit` smaller than needed for 5-fold CV on lambda.** If `D_fit` has fewer than 5 rows (or fewer than 5 of each class for the binary task), the inner 5-fold CV in lambda selection must not crash with a scikit-learn fold-count error — reduce the fold count or raise a clear `ValueError` naming the cause, not a raw sklearn traceback.
- **NaN/infinite values in `X`, `y`, or a member's predictions.** The spec's original source (§8 interface table) says to "reject NaN and infinite inputs" — `fit_stacking`, `calibrate_stacking`, and `predict_stacked` must raise `ValueError` rather than silently propagating NaN through the optimizer or the conformal quantile.
- **Calling `predict_stacked` or `calibrate_stacking` before `fit_stacking`.** Must raise a clear error (`RuntimeError` or `ValueError` naming the missing call), not an `AttributeError` from a missing `weights_`.
- **A classification `D_cal`/`D_fit` holding only one class.** `check_labels`-style validation already exists elsewhere in `tackai/fusion/training.py` for this exact condition (constant/single-class labels) — the new split and fit code must raise the same kind of clear error rather than letting log loss blow up silently (`log(0)`).

---

## Task 1: Mixture-MLE core math (`tackai/fusion/stacking.py`, regression)

**Files:**
- Create: `tackai/fusion/stacking.py`
- Test: `test/test_fusion_stacking.py`

**Interfaces:**
- Consumes: nothing from other tasks (first task).
- Produces:
  - `softmax(theta: np.ndarray) -> np.ndarray` — `w_i = exp(theta_i) / sum(exp(theta_k))`.
  - `mixture_nll(theta: np.ndarray, phi: np.ndarray, F: np.ndarray, S: np.ndarray, y: np.ndarray, sigma_min: float, lam: float) -> float` — penalized mean NLL, §4 of the spec. `F`, `S` are `(M, N)`; `y` is `(M,)`.
  - `mixture_nll_grad(theta, phi, F, S, y, sigma_min, lam) -> Tuple[np.ndarray, np.ndarray]` — analytic gradient `(d/dtheta, d/dphi)`, same shapes as `theta`/`phi`.
  - `fit_mixture_weights(F: np.ndarray, S: np.ndarray, y: np.ndarray, *, lam: float, sigma_min: float, n_restarts: int = 5, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, float]` — returns `(w, s, objective)`, the softmax/exp-transformed optimum over restarts.

Later tasks (Task 3) call `fit_mixture_weights` with `lam` chosen by cross-validation, and call `mixture_nll` directly (not through `fit_mixture_weights`) to score held-out folds.

- [ ] **Step 1: Write the failing tests for `softmax` and `mixture_nll`**

```python
# test/test_fusion_stacking.py
"""Stacking math: mixture MLE, conformal calibration, entropy decomposition.

Every test here uses small synthetic arrays — no FusionData, no real model fits — per the
design spec's instruction to keep these tests fast and focused on the stacking math itself.
"""
import numpy as np
import pytest

from tackai.fusion.stacking import mixture_nll, mixture_nll_grad, softmax


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `OMP_NUM_THREADS=1 pytest test/test_fusion_stacking.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tackai.fusion.stacking'`

- [ ] **Step 3: Implement `softmax`, `mixture_nll`, `mixture_nll_grad`**

```python
# tackai/fusion/stacking.py
"""Mixture-MLE stacking weights and uncertainty, consumed by FusionEnsemble.

Pure-NumPy statistical core for the stacking fit described in
docs/superpowers/specs/2026-10-07-fusion-stacked-ensemble-design.md. Every function here takes
plain arrays (predictions/sigmas/labels), never a model or a FusionEnsemble, so the math is
testable on small synthetic data without fitting anything. FusionEnsemble.fit_stacking /
calibrate_stacking / predict_stacked are thin orchestration around these functions.
"""
from typing import Tuple

import numpy as np

LOG_2PI = float(np.log(2.0 * np.pi))


def softmax(theta: np.ndarray) -> np.ndarray:
    """Numerically stable softmax: ``w_i = exp(theta_i) / sum_k exp(theta_k)``."""
    shifted = theta - theta.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def _scaled_sigma(phi: np.ndarray, S: np.ndarray, sigma_min: float) -> np.ndarray:
    """``sigma_tilde_ij = max(s_i * sigma_i(x_j), sigma_min)``, shape ``(M, N)``."""
    s = np.exp(phi)
    return np.maximum(S * s[None, :], sigma_min)


def mixture_nll(theta: np.ndarray, phi: np.ndarray, F: np.ndarray, S: np.ndarray,
                y: np.ndarray, sigma_min: float, lam: float) -> float:
    """Penalized mean negative log-likelihood of a Gaussian mixture (spec §4).

    Args:
        theta: Unconstrained weight parameters, shape ``(N,)``.
        phi: Unconstrained scale parameters, shape ``(N,)``.
        F: Member predictions, shape ``(M, N)``.
        S: Member sigmas (unscaled), shape ``(M, N)``.
        y: Labels, shape ``(M,)``.
        sigma_min: Variance floor (spec: ``1e-3 * std(y_fit)``).
        lam: Penalty strength toward uniform weights.

    Returns:
        Scalar penalized mean NLL.
    """
    w = softmax(theta)
    sigma = _scaled_sigma(phi, S, sigma_min)
    log_w = np.log(w)[None, :]
    log_density = -0.5 * LOG_2PI - np.log(sigma) - 0.5 * ((y[:, None] - F) / sigma) ** 2
    log_mix = _logsumexp(log_w + log_density, axis=1)
    nll = -np.mean(log_mix)
    penalty = lam * np.sum((w - 1.0 / len(w)) ** 2)
    return float(nll + penalty)


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    m = a.max(axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


def mixture_nll_grad(theta: np.ndarray, phi: np.ndarray, F: np.ndarray, S: np.ndarray,
                     y: np.ndarray, sigma_min: float, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    """Analytic gradient of :func:`mixture_nll` with respect to ``(theta, phi)``.

    Derived via the responsibility (posterior mixture-component probability) ``r_ij``:
    the gradient of a log-mixture with respect to each component's log-weight or
    log-density is exactly ``r_ij`` times that component's own gradient.
    """
    M, N = F.shape
    w = softmax(theta)
    s = np.exp(phi)
    raw_sigma = S * s[None, :]
    floored = raw_sigma < sigma_min
    sigma = np.maximum(raw_sigma, sigma_min)

    log_w = np.log(w)[None, :]
    log_density = -0.5 * LOG_2PI - np.log(sigma) - 0.5 * ((y[:, None] - F) / sigma) ** 2
    log_joint = log_w + log_density
    log_mix = _logsumexp(log_joint, axis=1)
    r = np.exp(log_joint - log_mix[:, None])              # (M, N) responsibilities, rows sum to 1

    # d(mean NLL)/d w_i = -mean_j r_ij / w_i; d w / d theta via softmax Jacobian.
    dnll_dw = -np.mean(r, axis=0) / w
    dw_dtheta = w[:, None] * (np.eye(N) - w[None, :])      # softmax Jacobian, (N, N)
    grad_theta = dw_dtheta @ dnll_dw
    penalty_grad_w = 2.0 * lam * (w - 1.0 / N)
    grad_theta += dw_dtheta @ penalty_grad_w

    # d(log_density)/d sigma_tilde = -1/sigma + (y-f)^2/sigma^3; zero where the floor binds
    # (sigma_tilde does not depend on s_i there, so the chain rule gives exactly 0).
    resid = (y[:, None] - F)
    dlogdensity_dsigma = -1.0 / sigma + (resid ** 2) / (sigma ** 3)
    dsigma_ds = np.where(floored, 0.0, S)                  # d(sigma_tilde)/d(s_i), (M, N)
    ds_dphi = s[None, :]                                    # d(s_i)/d(phi_i) = s_i
    dnll_dphi = -np.mean(r * dlogdensity_dsigma * dsigma_ds, axis=0) * ds_dphi[0]
    grad_phi = dnll_dphi

    return grad_theta, grad_phi
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `OMP_NUM_THREADS=1 pytest test/test_fusion_stacking.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Add and run the gradient-vs-finite-difference test**

```python
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
```

Run: `OMP_NUM_THREADS=1 pytest test/test_fusion_stacking.py -v`
Expected: PASS (5 tests)

- [ ] **Step 6: Write the failing test for `fit_mixture_weights`**

```python
from tackai.fusion.stacking import fit_mixture_weights


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
```

- [ ] **Step 7: Run tests to verify they fail**

Run: `OMP_NUM_THREADS=1 pytest test/test_fusion_stacking.py -v`
Expected: FAIL with `ImportError: cannot import name 'fit_mixture_weights'`

- [ ] **Step 8: Implement `fit_mixture_weights`**

```python
# Append to tackai/fusion/stacking.py
from scipy.optimize import minimize


def fit_mixture_weights(F: np.ndarray, S: np.ndarray, y: np.ndarray, *, lam: float,
                        sigma_min: float, n_restarts: int = 5,
                        seed: int = 0) -> Tuple[np.ndarray, np.ndarray, float]:
    """Fit mixture weights and per-member scales by L-BFGS-B with 5 random restarts.

    Args:
        F: Member predictions on the fit set, shape ``(M, N)``.
        S: Member sigmas (unscaled) on the fit set, shape ``(M, N)``.
        y: Labels, shape ``(M,)``.
        lam: Penalty strength toward uniform weights.
        sigma_min: Variance floor.
        n_restarts: Random restarts (``theta`` perturbed by ``N(0, 0.5^2)``; the first restart
            always starts at exactly zero, i.e. uniform weights).
        seed: Seed for the restart perturbations.

    Returns:
        ``(w, s, objective)``: fitted weights (sum to 1), fitted scales, and the objective
        value at the optimum.
    """
    N = F.shape[1]
    rng = np.random.default_rng(seed)

    def objective(z):
        theta, phi = z[:N], z[N:]
        return mixture_nll(theta, phi, F, S, y, sigma_min, lam)

    def grad(z):
        theta, phi = z[:N], z[N:]
        g_theta, g_phi = mixture_nll_grad(theta, phi, F, S, y, sigma_min, lam)
        return np.concatenate([g_theta, g_phi])

    phi0 = np.zeros(N) if not np.allclose(S, 1.0) else np.log(
        np.sqrt(np.mean((y[:, None] - F) ** 2, axis=0)) + 1e-12)

    best = (np.inf, None)
    for restart in range(n_restarts):
        theta0 = np.zeros(N) if restart == 0 else rng.normal(0.0, 0.5, size=N)
        z0 = np.concatenate([theta0, phi0])
        result = minimize(objective, z0, jac=grad, method="L-BFGS-B",
                          options={"ftol": 1e-10, "gtol": 1e-10})
        if result.fun < best[0]:
            best = (result.fun, result.x)

    theta_opt, phi_opt = best[1][:N], best[1][N:]
    return softmax(theta_opt), np.exp(phi_opt), float(best[0])
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `OMP_NUM_THREADS=1 pytest test/test_fusion_stacking.py -v`
Expected: PASS (8 tests)

- [ ] **Step 10: Commit**

```bash
git add tackai/fusion/stacking.py test/test_fusion_stacking.py
git commit -m "feat(fusion): mixture-MLE stacking weights core math

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 2: Classification pooling, conformal calibration, entropy decomposition (`stacking.py`)

**Files:**
- Modify: `tackai/fusion/stacking.py`
- Test: `test/test_fusion_stacking.py`

**Interfaces:**
- Consumes: `softmax` from Task 1.
- Produces:
  - `fit_pooled_weights(P: np.ndarray, y: np.ndarray, *, lam: float, seed: int = 0) -> Tuple[np.ndarray, float]` — `(w, objective)` for the binary log-loss pooling (spec §5), one L-BFGS run, no restarts.
  - `binary_entropy(q: np.ndarray) -> np.ndarray` — elementwise `H(q) = -q*log(q) - (1-q)*log(1-q)`, safe at `q=0`/`q=1`.
  - `entropy_decomposition(P: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]` — `(total, aleatoric, epistemic)`, epistemic clamped at 0.
  - `conformal_quantile(residual_ratio: np.ndarray, alpha: float) -> float` — the `ceil((n+1)(1-alpha))`-th smallest of `residual_ratio`, or `inf` with a `UserWarning` if the index exceeds `n`.

Later tasks (Task 3, Task 4) call these directly from `FusionEnsemble.fit_stacking`/`calibrate_stacking`.

- [ ] **Step 1: Write the failing tests**

```python
# Append to test/test_fusion_stacking.py
from tackai.fusion.stacking import (binary_entropy, conformal_quantile,
                                    entropy_decomposition, fit_pooled_weights)


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `OMP_NUM_THREADS=1 pytest test/test_fusion_stacking.py -v`
Expected: FAIL with `ImportError` for the four new names

- [ ] **Step 3: Implement the four functions**

```python
# Append to tackai/fusion/stacking.py
import warnings


def fit_pooled_weights(P: np.ndarray, y: np.ndarray, *, lam: float,
                       seed: int = 0) -> Tuple[np.ndarray, float]:
    """Fit log-loss-pooled weights over clipped member probabilities (spec §5).

    Convex in the weights, so a single L-BFGS run from uniform weights is enough.

    Args:
        P: Clipped member probabilities, shape ``(M, N)``, values in ``[1e-6, 1 - 1e-6]``.
        y: Binary labels (0 or 1), shape ``(M,)``.
        lam: Penalty strength toward uniform weights.
        seed: Unused (kept for interface symmetry with :func:`fit_mixture_weights`; the
            problem is convex so the result does not depend on a starting point).

    Returns:
        ``(w, objective)``: fitted weights (sum to 1) and the objective value at the optimum.
    """
    N = P.shape[1]

    def objective(theta):
        w = softmax(theta)
        p = P @ w
        loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        penalty = lam * np.sum((w - 1.0 / N) ** 2)
        return loss + penalty

    def grad(theta):
        w = softmax(theta)
        p = P @ w
        dloss_dp = (p - y) / (p * (1 - p)) / len(y)
        dloss_dw = P.T @ dloss_dp
        dw_dtheta = w[:, None] * (np.eye(N) - w[None, :])
        g = dw_dtheta @ dloss_dw
        g += dw_dtheta @ (2.0 * lam * (w - 1.0 / N))
        return g

    result = minimize(objective, np.zeros(N), jac=grad, method="L-BFGS-B",
                      options={"ftol": 1e-10, "gtol": 1e-10})
    return softmax(result.x), float(result.fun)


def binary_entropy(q: np.ndarray) -> np.ndarray:
    """Binary entropy ``H(q) = -q log q - (1-q) log(1-q)``, 0 at ``q in {0, 1}``."""
    q = np.asarray(q, dtype=float)
    out = np.zeros_like(q)
    mask = (q > 0) & (q < 1)
    qm = q[mask]
    out[mask] = -qm * np.log(qm) - (1 - qm) * np.log(1 - qm)
    return out


def entropy_decomposition(P: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Total/aleatoric/epistemic binary-entropy decomposition (spec §5).

    Args:
        P: Member probabilities, shape ``(M, N)``.
        w: Member weights, shape ``(N,)``, summing to 1.

    Returns:
        ``(total, aleatoric, epistemic)``, each shape ``(M,)``. ``epistemic`` is clamped at 0
        to remove floating-point noise; it is non-negative by concavity of ``H``.
    """
    pooled = P @ w
    total = binary_entropy(pooled)
    aleatoric = binary_entropy(P) @ w
    epistemic = np.maximum(total - aleatoric, 0.0)
    return total, aleatoric, epistemic


def conformal_quantile(residual_ratio: np.ndarray, alpha: float) -> float:
    """Finite-sample conformal quantile of normalized residuals (spec §6).

    Args:
        residual_ratio: ``|y - mu(x)| / sigma(x)`` on the calibration set.
        alpha: Miscoverage level (e.g. 0.1 for 90% coverage).

    Returns:
        The ``ceil((n+1)(1-alpha))``-th smallest value, or ``inf`` (with a ``UserWarning``)
        if that index exceeds ``n`` — the calibration set is too small for this ``alpha``.
    """
    n = len(residual_ratio)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        warnings.warn(
            f"calibration set too small (n={n}) for alpha={alpha}; returning an infinite "
            "interval. Use a larger D_cal or a larger alpha.", UserWarning, stacklevel=2)
        return np.inf
    return float(np.sort(residual_ratio)[k - 1])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `OMP_NUM_THREADS=1 pytest test/test_fusion_stacking.py -v`
Expected: PASS (14 tests total)

- [ ] **Step 5: Commit**

```bash
git add tackai/fusion/stacking.py test/test_fusion_stacking.py
git commit -m "feat(fusion): pooled-probability, entropy and conformal stacking math

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 3: Lambda cross-validation helper (`stacking.py`)

**Files:**
- Modify: `tackai/fusion/stacking.py`
- Test: `test/test_fusion_stacking.py`

**Interfaces:**
- Consumes: `mixture_nll`, `fit_mixture_weights`, `fit_pooled_weights` from Tasks 1–2.
- Produces:
  - `select_lambda_regression(F: np.ndarray, S: np.ndarray, y: np.ndarray, *, lambdas, sigma_min: float, n_restarts: int, seed: int = 0, n_splits: int = 5) -> Tuple[float, dict]` — returns `(best_lambda, cv_log)` where `cv_log` maps each lambda to its mean held-out NLL. `n_splits` is clamped to `min(n_splits, len(y))` so a tiny `D_fit` degrades gracefully rather than raising a scikit-learn fold-count error (Review Focus item 2).
  - `select_lambda_classification(P: np.ndarray, y: np.ndarray, *, lambdas, seed: int = 0, n_splits: int = 5) -> Tuple[float, dict]` — same shape, scoring held-out log loss; uses `StratifiedKFold` and clamps `n_splits` to the smaller class count as well as `len(y)`.

Task 4 calls both from `FusionEnsemble.fit_stacking`.

- [ ] **Step 1: Write the failing tests**

```python
# Append to test/test_fusion_stacking.py
from tackai.fusion.stacking import select_lambda_classification, select_lambda_regression


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `OMP_NUM_THREADS=1 pytest test/test_fusion_stacking.py -v`
Expected: FAIL with `ImportError` for `select_lambda_regression`/`select_lambda_classification`

- [ ] **Step 3: Implement both functions**

```python
# Append to tackai/fusion/stacking.py
from sklearn.linear_model import LogisticRegression  # noqa: F401  (not used here; see note below)
from sklearn.model_selection import KFold, StratifiedKFold


def select_lambda_regression(F: np.ndarray, S: np.ndarray, y: np.ndarray, *, lambdas,
                             sigma_min: float, n_restarts: int, seed: int = 0,
                             n_splits: int = 5) -> Tuple[float, dict]:
    """Pick ``lambda`` by K-fold CV inside ``D_fit``, scoring unpenalized held-out NLL.

    Args:
        F: Member predictions on ``D_fit``, shape ``(M, N)``.
        S: Member sigmas on ``D_fit``, shape ``(M, N)``.
        y: Labels on ``D_fit``, shape ``(M,)``.
        lambdas: Candidate penalty values.
        sigma_min: Variance floor.
        n_restarts: Restarts passed to each inner :func:`fit_mixture_weights` call.
        seed: Seed for the fold split and the restarts.
        n_splits: Requested fold count, clamped to ``min(n_splits, len(y))`` so a ``D_fit``
            smaller than the requested fold count degrades gracefully.

    Returns:
        ``(best_lambda, cv_log)`` where ``cv_log`` maps each candidate lambda to its mean
        held-out unpenalized NLL across folds.
    """
    n_splits = max(2, min(n_splits, len(y)))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cv_log = {}
    for lam in lambdas:
        scores = []
        for train_idx, val_idx in kf.split(y):
            w, s, _ = fit_mixture_weights(F[train_idx], S[train_idx], y[train_idx], lam=lam,
                                          sigma_min=sigma_min, n_restarts=n_restarts, seed=seed)
            theta = np.log(np.maximum(w, 1e-300))
            phi = np.log(np.maximum(s, 1e-300))
            scores.append(mixture_nll(theta, phi, F[val_idx], S[val_idx], y[val_idx],
                                      sigma_min, lam=0.0))
        cv_log[lam] = float(np.mean(scores))
    best = min(cv_log, key=cv_log.get)
    return best, cv_log


def select_lambda_classification(P: np.ndarray, y: np.ndarray, *, lambdas, seed: int = 0,
                                  n_splits: int = 5) -> Tuple[float, dict]:
    """Pick ``lambda`` by stratified K-fold CV, scoring held-out log loss.

    Args:
        P: Member probabilities on ``D_fit``, shape ``(M, N)``.
        y: Binary labels on ``D_fit``, shape ``(M,)``.
        lambdas: Candidate penalty values.
        seed: Seed for the fold split.
        n_splits: Requested fold count, clamped to ``min(n_splits, len(y), minority_count)``.

    Returns:
        ``(best_lambda, cv_log)`` where ``cv_log`` maps each candidate lambda to its mean
        held-out log loss across folds.
    """
    minority = int(min(np.sum(y == 0), np.sum(y == 1)))
    n_splits = max(2, min(n_splits, len(y), minority))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cv_log = {}
    for lam in lambdas:
        scores = []
        for train_idx, val_idx in skf.split(P, y):
            w, _ = fit_pooled_weights(P[train_idx], y[train_idx], lam=lam, seed=seed)
            p = np.clip(P[val_idx] @ w, 1e-6, 1 - 1e-6)
            yv = y[val_idx]
            scores.append(float(-np.mean(yv * np.log(p) + (1 - yv) * np.log(1 - p))))
        cv_log[lam] = float(np.mean(scores))
    best = min(cv_log, key=cv_log.get)
    return best, cv_log
```

Remove the unused `LogisticRegression` import placeholder above — it was left in by mistake; the real file must not import it. Final import line for this task is just `from sklearn.model_selection import KFold, StratifiedKFold`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `OMP_NUM_THREADS=1 pytest test/test_fusion_stacking.py -v`
Expected: PASS (17 tests total)

- [ ] **Step 5: Commit**

```bash
git add tackai/fusion/stacking.py test/test_fusion_stacking.py
git commit -m "feat(fusion): lambda cross-validation for stacking weights

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 4: `FusionEnsemble._stacking_predict` member dispatch

**Files:**
- Modify: `tackai/fusion/ensemble.py`
- Test: `test/test_fusion_stacking.py`

**Interfaces:**
- Consumes: `FusionEnsemble` class (existing), `GPInteraction` (existing, `tackai/fusion/gp.py`), `xgboost.XGBRegressor`/`xgboost.XGBClassifier` (external).
- Produces: `FusionEnsemble._stacking_predict(self, member, X) -> Tuple[np.ndarray, Optional[np.ndarray]]` — instance method; `(mean, sigma)`, `sigma` is `None` for any classifier or XGBoost regressor (Option-A sigma is computed by the caller, not here — see spec §3).

Task 5 calls this inside `fit_stacking`/`predict_stacked`.

- [ ] **Step 1: Write the failing tests**

```python
# Append to test/test_fusion_stacking.py
import xgboost as xgb

from fusion_fixtures import DEFAULT_BLOCKS
from tackai.fusion.data import FusionData
from tackai.fusion.ensemble import FusionEnsemble
from tackai.fusion.gp import GPInteraction


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `OMP_NUM_THREADS=1 pytest test/test_fusion_stacking.py -v`
Expected: FAIL with `AttributeError: 'FusionEnsemble' object has no attribute '_stacking_predict'`

- [ ] **Step 3: Implement `_stacking_predict`**

Add the import and method to `tackai/fusion/ensemble.py`. Insert the import near the top with the other `tackai.fusion` imports, and the method near `_member_scores` (after it, before `_aggregate`):

```python
# Add to the import block at the top of tackai/fusion/ensemble.py, alongside the
# existing `from tackai.fusion.gp import GPInteraction`:
import xgboost as xgb
```

```python
# New method on FusionEnsemble, placed after _member_scores (around line 528)
def _stacking_predict(self, member, X) -> "Tuple[np.ndarray, Optional[np.ndarray]]":
    """(mean, sigma) for one member, in its own reported units, for the stacking fit.

    Used only by fit_stacking/calibrate_stacking/predict_stacked — predict()/predict_matrix()
    are unaffected and remain GP-only, as documented in the design spec.

    Args:
        member: A GPInteraction, xgboost.XGBRegressor, or xgboost.XGBClassifier.
        X: Design matrix of the rows to score.

    Returns:
        ``(mean, sigma)``. ``sigma`` is the GP's own predictive std for a GPInteraction, or
        None for any XGBoost member — Option A's constant residual sigma for an XGBoost
        regressor is computed once from D_fit by the caller (fit_stacking), not here.

    Raises:
        TypeError: If ``member`` is none of the three supported types.
    """
    X = np.asarray(X)
    if isinstance(member, GPInteraction):
        mean, sigma = member.predict(X, return_std=True)
        return mean, sigma
    if isinstance(member, xgb.XGBClassifier):
        return member.predict_proba(X)[:, 1], None
    if isinstance(member, xgb.XGBRegressor):
        return member.predict(X), None
    raise TypeError(
        f"unsupported stacking member type {type(member).__name__!r} for {member!r}; "
        "expected GPInteraction, xgboost.XGBRegressor, or xgboost.XGBClassifier")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `OMP_NUM_THREADS=1 pytest test/test_fusion_stacking.py -v`
Expected: PASS (22 tests total)

- [ ] **Step 5: Commit**

```bash
git add tackai/fusion/ensemble.py test/test_fusion_stacking.py
git commit -m "feat(fusion): FusionEnsemble._stacking_predict member dispatch

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 5: `fit_stacking` — data splits and orchestration

**Files:**
- Modify: `tackai/fusion/ensemble.py`
- Test: `test/test_fusion_stacking.py`

**Interfaces:**
- Consumes: `_stacking_predict` (Task 4), `select_lambda_regression`/`select_lambda_classification`/`fit_mixture_weights`/`fit_pooled_weights` (Tasks 1–3), `TASK_TYPES` (existing, `tackai/fusion/data.py`).
- Produces: `FusionEnsemble.fit_stacking(self, X=None, y=None, *, groups=None, X_fit=None, y_fit=None, X_cal=None, y_cal=None, X_test=None, y_test=None, lambdas=(0.0, 0.01, 0.1, 1.0, 10.0), n_restarts=5, seed=0) -> "FusionEnsemble"`. Side effects on `self`: `weights_` (dict, member name → float), `scales_` (dict, member name → float; empty dict for classification), `lambda_` (float), `stacking_cv_log_` (dict), `X_cal_`/`y_cal_` (arrays), `X_test_`/`y_test_` (arrays or `None`).

Task 6 (`calibrate_stacking`) reads `X_cal_`/`y_cal_`/`weights_`/`scales_`. Task 7 (`predict_stacked`) reads `weights_`/`scales_`.

- [ ] **Step 1: Write the failing tests for the split-argument validation and the happy path**

```python
# Append to test/test_fusion_stacking.py
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


@pytest.mark.skip(reason="predict_stacked added in Task 7")
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
```

This test calls `ens.predict_stacked`, which does not exist until Task 7 — hence the `skip` mark here. Task 7 step 1 removes that mark (deleting the `@pytest.mark.skip(...)` line above the test) alongside adding its own new tests, then Task 7 step 4's full-file run confirms it passes unskipped.

- [ ] **Step 2: Run tests to verify they fail**

Run: `OMP_NUM_THREADS=1 pytest test/test_fusion_stacking.py -v`
Expected: FAIL with `AttributeError: 'FusionEnsemble' object has no attribute 'fit_stacking'`

- [ ] **Step 3: Implement `fit_stacking`**

Add imports near the top of `tackai/fusion/ensemble.py` (alongside the existing `from tackai.fusion.training import check_labels`):

```python
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from tackai.fusion.data import TASK_TYPES
from tackai.fusion.stacking import (fit_mixture_weights, fit_pooled_weights,
                                    select_lambda_classification, select_lambda_regression)
```

Note `TASK_TYPES` and `TASK_SUPPORT`/`BLOCK_ORDER` are already imported from `tackai.fusion.data` at the top of this file — just add `TASK_TYPES` to that existing import line rather than duplicating the import.

Add the method after `calibrate` (around line 357, before the `available_tasks` property):

```python
def fit_stacking(self, X=None, y=None, *, groups=None, X_fit=None, y_fit=None,
                 X_cal=None, y_cal=None, X_test=None, y_test=None,
                 lambdas=(0.0, 0.01, 0.1, 1.0, 10.0), n_restarts: int = 5,
                 seed: int = 0) -> "FusionEnsemble":
    """Fit stacking weights (and, for regression, per-member scales) on a held-out set.

    Learns weights by maximum likelihood of a Gaussian mixture (regression) or by minimizing
    log loss on pooled probabilities (classification), per
    docs/superpowers/specs/2026-10-07-fusion-stacked-ensemble-design.md. The rows passed here
    must already be held out from every member's own training data -- this method has no
    record of what the members were trained on and cannot check that invariant.

    Two mutually exclusive calling conventions:

    * Auto-split: pass ``X``/``y`` (and optionally ``groups``); this method splits them
      60/20/20 into D_fit/D_cal/D_test internally.
    * Explicit split: pass ``X_fit``/``y_fit``/``X_cal``/``y_cal`` (and optionally
      ``X_test``/``y_test``) directly, skipping the internal split.

    Args:
        X: Design matrix for the auto-split path.
        y: Labels for the auto-split path.
        groups: Optional scaffold group id per row, for the auto-split path; when given, the
            split uses GroupShuffleSplit so D_fit/D_cal/D_test share no group.
        X_fit, y_fit, X_cal, y_cal: Explicit D_fit/D_cal rows, for the explicit-split path.
        X_test, y_test: Optional explicit D_test rows, kept on ``self`` for a future reporting
            method but unused by this one.
        lambdas: Candidate penalty values for the held-out lambda selection.
        n_restarts: Random restarts for the regression mixture fit.
        seed: Seed for the split, the restarts, and the lambda-selection folds.

    Returns:
        self, with ``weights_``, ``scales_``, ``lambda_``, ``stacking_cv_log_``, ``X_cal_``,
        ``y_cal_``, ``X_test_``, ``y_test_`` set.

    Raises:
        ValueError: If both or neither calling convention is given, or if ``X``/``y`` (in
            either convention) hold a NaN or infinite value.
    """
    auto_given = X is not None or y is not None
    explicit_given = any(v is not None for v in (X_fit, y_fit, X_cal, y_cal))
    if auto_given and explicit_given:
        raise ValueError("pass either (X, y[, groups]) or (X_fit, y_fit, X_cal, y_cal, ...), "
                         "not both")
    if not auto_given and not explicit_given:
        raise ValueError("pass either (X, y[, groups]) or (X_fit, y_fit, X_cal, y_cal, ...); "
                         "neither was given")

    if auto_given:
        if X is None or y is None:
            raise ValueError("both X and y are required for the auto-split path")
        X, y = np.asarray(X), np.asarray(y)
        self._check_finite(X, "X")
        self._check_finite(y, "y")
        X_fit, X_cal, X_test, y_fit, y_cal, y_test = self._split_stacking_set(X, y, groups, seed)
    else:
        if X_fit is None or y_fit is None or X_cal is None or y_cal is None:
            raise ValueError("X_fit, y_fit, X_cal and y_cal are all required for the explicit "
                             "split path")
        X_fit, y_fit = np.asarray(X_fit), np.asarray(y_fit)
        X_cal, y_cal = np.asarray(X_cal), np.asarray(y_cal)
        for arr, name in ((X_fit, "X_fit"), (y_fit, "y_fit"), (X_cal, "X_cal"), (y_cal, "y_cal")):
            self._check_finite(arr, name)
        X_test = np.asarray(X_test) if X_test is not None else None
        y_test = np.asarray(y_test) if y_test is not None else None

    task_type = TASK_TYPES[self.task]
    F = np.column_stack([self._stacking_predict(m, X_fit)[0] for m in self.members])
    if task_type == "regression":
        S = np.column_stack([self._fit_set_sigma(m, X_fit, y_fit) for m in self.members])
        sigma_min = 1e-3 * float(np.std(y_fit))
        best_lambda, cv_log = select_lambda_regression(
            F, S, y_fit, lambdas=lambdas, sigma_min=sigma_min, n_restarts=n_restarts, seed=seed)
        w, s, _ = fit_mixture_weights(F, S, y_fit, lam=best_lambda, sigma_min=sigma_min,
                                      n_restarts=n_restarts, seed=seed)
        self.weights_ = {name: float(wi) for name, wi in zip(self.names, w)}
        self.scales_ = {name: float(si) for name, si in zip(self.names, s)}
        self._sigma_min_ = sigma_min
    else:
        P = np.clip(F, 1e-6, 1 - 1e-6)
        best_lambda, cv_log = select_lambda_classification(P, y_fit, lambdas=lambdas, seed=seed)
        w, _ = fit_pooled_weights(P, y_fit, lam=best_lambda, seed=seed)
        self.weights_ = {name: float(wi) for name, wi in zip(self.names, w)}
        self.scales_ = {}

    self.lambda_ = float(best_lambda)
    self.stacking_cv_log_ = cv_log
    self.X_cal_, self.y_cal_ = X_cal, y_cal
    self.X_test_, self.y_test_ = X_test, y_test
    self.temperature_ = 1.0
    self.q_hat_ = None
    self.c_ = None
    return self

@staticmethod
def _check_finite(arr: np.ndarray, name: str) -> None:
    """Raise ValueError naming ``name`` if ``arr`` holds a NaN or infinite value."""
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite value(s)")

def _fit_set_sigma(self, member, X_fit, y_fit) -> np.ndarray:
    """Per-row sigma column for one member on D_fit (spec §3): the GP's own predictive
    std, or a constant RMSE broadcast across every row for an XGBoost regressor."""
    mean, sigma = self._stacking_predict(member, X_fit)
    if sigma is not None:
        return sigma
    rmse = float(np.sqrt(np.mean((y_fit - mean) ** 2)))
    return np.full(len(y_fit), rmse)

@staticmethod
def _split_stacking_set(X: np.ndarray, y: np.ndarray, groups, seed: int):
    """60/20/20 split into (X_fit, X_cal, X_test, y_fit, y_cal, y_test).

    Uses GroupShuffleSplit twice when ``groups`` is given (so no group crosses a split
    boundary), else train_test_split, stratified by ``y`` for a binary-looking target
    (exactly two distinct values).
    """
    n = len(y)
    if groups is not None:
        groups = np.asarray(groups)
        splitter1 = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=seed)
        fit_idx, rest_idx = next(splitter1.split(np.zeros(n), groups=groups))
        splitter2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        cal_idx, test_idx = next(splitter2.split(np.zeros(len(rest_idx)), groups=groups[rest_idx]))
        cal_idx, test_idx = rest_idx[cal_idx], rest_idx[test_idx]
    else:
        stratify = y if len(np.unique(y)) == 2 else None
        fit_idx, rest_idx = train_test_split(np.arange(n), test_size=0.4, random_state=seed,
                                             stratify=stratify)
        rest_stratify = y[rest_idx] if stratify is not None else None
        cal_idx, test_idx = train_test_split(rest_idx, test_size=0.5, random_state=seed,
                                             stratify=rest_stratify)
    return (X[fit_idx], X[cal_idx], X[test_idx], y[fit_idx], y[cal_idx], y[test_idx])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `OMP_NUM_THREADS=1 pytest test/test_fusion_stacking.py -v`
Expected: 28 PASS, 1 SKIPPED (29 tests total; `test_fit_stacking_runs_end_to_end_with_mixed_gp_and_xgboost_membership` is skipped until Task 7)

- [ ] **Step 5: Commit**

```bash
git add tackai/fusion/ensemble.py test/test_fusion_stacking.py
git commit -m "feat(fusion): FusionEnsemble.fit_stacking orchestration and data splits

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 6: `calibrate_stacking`

**Files:**
- Modify: `tackai/fusion/ensemble.py`
- Test: `test/test_fusion_stacking.py`

**Interfaces:**
- Consumes: `conformal_quantile` (Task 2), `self.weights_`/`self.scales_`/`self._sigma_min_`/`self.X_cal_`/`self.y_cal_` (Task 5), `self._stacking_predict` (Task 4), `self._mixture_mean_std` (new helper this task introduces, also consumed by Task 7).
- Produces:
  - `FusionEnsemble._mixture_mean_std(self, X) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]` — `(mean, std, std_noise, std_disagreement)` from the fitted `weights_`/`scales_`, regression only.
  - `FusionEnsemble.calibrate_stacking(self, *, X_cal=None, y_cal=None, alpha: float = 0.1) -> "FusionEnsemble"` — side effects: `q_hat_`, `c_` (regression) or `temperature_` (classification).

Task 7 (`predict_stacked`) reads `q_hat_`/`c_`/`temperature_` and calls `_mixture_mean_std`.

- [ ] **Step 1: Write the failing tests**

```python
# Append to test/test_fusion_stacking.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `OMP_NUM_THREADS=1 pytest test/test_fusion_stacking.py -v`
Expected: FAIL with `AttributeError: 'FusionEnsemble' object has no attribute 'calibrate_stacking'`

- [ ] **Step 3: Implement `_mixture_mean_std` and `calibrate_stacking`**

Add after `fit_stacking` in `tackai/fusion/ensemble.py`. Add `conformal_quantile` to the `tackai.fusion.stacking` import line from Task 5.

```python
def _mixture_mean_std(self, X) -> "Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """Mixture mean/std and their noise/disagreement components (spec §4), regression only.

    An XGBoost regressor member has no per-row sigma: its raw sigma column is exactly 1
    everywhere, so ``sigma_tilde = s_i * 1 = s_i`` reconstructs the constant RMSE-scale
    fit_stacking fit for it, without re-deriving the RMSE here.

    Requires fit_stacking to have been called.
    """
    n = len(X)
    F = np.empty((n, len(self.members)))
    S = np.empty((n, len(self.members)))
    for j, member in enumerate(self.members):
        mean_j, sigma_j = self._stacking_predict(member, X)
        F[:, j] = mean_j
        S[:, j] = sigma_j if sigma_j is not None else 1.0
    w = np.array([self.weights_[name] for name in self.names])
    s = np.array([self.scales_[name] for name in self.names])
    sigma_tilde = np.maximum(S * s[None, :], self._sigma_min_)
    mean = F @ w
    noise = np.sqrt(np.sum(w[None, :] * sigma_tilde ** 2, axis=1))
    disagreement = np.sqrt(np.sum(w[None, :] * (F - mean[:, None]) ** 2, axis=1))
    std = np.sqrt(noise ** 2 + disagreement ** 2)
    return mean, std, noise, disagreement

def calibrate_stacking(self, *, X_cal=None, y_cal=None, alpha: float = 0.1) -> "FusionEnsemble":
    """Calibrate the stacking uncertainty on a held-out set (spec §6).

    Must be called after :meth:`fit_stacking`. Defaults to the D_cal split
    ``fit_stacking`` stored; pass ``X_cal``/``y_cal`` to use a different set instead.

    Args:
        X_cal: Calibration design matrix (default: the D_cal fit_stacking stored).
        y_cal: Calibration labels (default: the D_cal fit_stacking stored).
        alpha: Miscoverage level for the regression conformal interval.

    Returns:
        self, with ``q_hat_``/``c_`` set for regression, or ``temperature_`` for
        classification.

    Raises:
        RuntimeError: If called before :meth:`fit_stacking`.
    """
    if not hasattr(self, "weights_"):
        raise RuntimeError("call fit_stacking before calibrate_stacking")
    X_cal = np.asarray(X_cal) if X_cal is not None else self.X_cal_
    y_cal = np.asarray(y_cal) if y_cal is not None else self.y_cal_

    if TASK_TYPES[self.task] == "regression":
        mean, std, _, _ = self._mixture_mean_std(X_cal)
        ratio = np.abs(y_cal - mean) / std
        self.q_hat_ = conformal_quantile(ratio, alpha)
        self.c_ = float(np.sqrt(np.mean(ratio ** 2)))
    else:
        F = np.column_stack([self._stacking_predict(m, X_cal)[0] for m in self.members])
        P = np.clip(F, 1e-6, 1 - 1e-6)
        w = np.array([self.weights_[name] for name in self.names])
        pooled = np.clip(P @ w, 1e-6, 1 - 1e-6)
        uncalibrated_loss = float(-np.mean(y_cal * np.log(pooled) + (1 - y_cal) * np.log(1 - pooled)))
        logit = np.log(pooled / (1 - pooled))
        best_T, best_loss = 1.0, uncalibrated_loss
        for T in np.geomspace(0.2, 5.0, 25):
            adjusted = 1.0 / (1.0 + np.exp(-logit / T))
            adjusted = np.clip(adjusted, 1e-6, 1 - 1e-6)
            loss = float(-np.mean(y_cal * np.log(adjusted) + (1 - y_cal) * np.log(1 - adjusted)))
            if loss < best_loss:
                best_T, best_loss = T, loss
        diffs = self._bootstrap_loss_diffs(y_cal, pooled, best_T, n_boot=200, seed=0)
        se = float(np.std(diffs))
        self.temperature_ = best_T if (uncalibrated_loss - best_loss) > se else 1.0
    return self

@staticmethod
def _bootstrap_loss_diffs(y, pooled, T, n_boot: int, seed: int) -> np.ndarray:
    """Bootstrap standard error of (uncalibrated - calibrated) log loss, for the
    temperature-acceptance test in calibrate_stacking."""
    rng = np.random.default_rng(seed)
    n = len(y)
    logit = np.log(pooled / (1 - pooled))
    adjusted = np.clip(1.0 / (1.0 + np.exp(-logit / T)), 1e-6, 1 - 1e-6)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        unc = -np.mean(y[idx] * np.log(pooled[idx]) + (1 - y[idx]) * np.log(1 - pooled[idx]))
        cal = -np.mean(y[idx] * np.log(adjusted[idx]) + (1 - y[idx]) * np.log(1 - adjusted[idx]))
        diffs[b] = unc - cal
    return diffs
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `OMP_NUM_THREADS=1 pytest test/test_fusion_stacking.py -v`
Expected: 32 PASS, 1 SKIPPED (33 tests total)

- [ ] **Step 5: Commit**

```bash
git add tackai/fusion/ensemble.py test/test_fusion_stacking.py
git commit -m "feat(fusion): FusionEnsemble.calibrate_stacking conformal and temperature calibration

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 7: `predict_stacked`

**Files:**
- Modify: `tackai/fusion/ensemble.py`
- Test: `test/test_fusion_stacking.py`

**Interfaces:**
- Consumes: `self.weights_`, `self._mixture_mean_std` (Task 6), `entropy_decomposition` (Task 2), `self.q_hat_`/`self.c_`/`self.temperature_` (Task 6, may be unset if `calibrate_stacking` was never called).
- Produces: `FusionEnsemble.predict_stacked(self, X) -> dict`. Final public method of this plan.

Nothing downstream in this plan consumes this; it is the terminal entry point.

- [ ] **Step 1: Write the failing tests**

```python
# Append to test/test_fusion_stacking.py
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
```

Also, remove the `@pytest.mark.skip(reason="predict_stacked added in Task 7")` line directly above `test_fit_stacking_runs_end_to_end_with_mixed_gp_and_xgboost_membership` (added back in Task 5) — that test is now unblocked.

- [ ] **Step 2: Run tests to verify they fail**

Run: `OMP_NUM_THREADS=1 pytest test/test_fusion_stacking.py -v`
Expected: FAIL with `AttributeError: 'FusionEnsemble' object has no attribute 'predict_stacked'` (this now also fails the previously-skipped mixed-membership test, for the same reason)

- [ ] **Step 3: Implement `predict_stacked`**

Add `entropy_decomposition` to the `tackai.fusion.stacking` import line, and add the method after `calibrate_stacking`:

```python
def predict_stacked(self, X) -> dict:
    """Score new rows with the fitted stacking weights and calibrated uncertainty.

    Must be called after :meth:`fit_stacking` (and, for a calibrated interval,
    :meth:`calibrate_stacking`).

    Args:
        X: Design matrix of the rows to score.

    Returns:
        For regression: ``{"mean", "std", "std_noise", "std_disagreement", "lower",
        "upper"}`` — ``lower``/``upper`` are ``None`` if :meth:`calibrate_stacking` has not
        been called yet (the predictive distribution is a mixture, not a Gaussian, so no
        interval is reported without the conformal quantile). For classification:
        ``{"proba", "entropy_total", "entropy_aleatoric", "entropy_epistemic"}``.

    Raises:
        RuntimeError: If called before :meth:`fit_stacking`.
    """
    if not hasattr(self, "weights_"):
        raise RuntimeError("call fit_stacking before predict_stacked")
    X = np.asarray(X)

    if TASK_TYPES[self.task] == "regression":
        mean, std, noise, disagreement = self._mixture_mean_std(X)
        q_hat = getattr(self, "q_hat_", None)
        lower = mean - q_hat * std if q_hat is not None else None
        upper = mean + q_hat * std if q_hat is not None else None
        return {"mean": mean, "std": std, "std_noise": noise, "std_disagreement": disagreement,
               "lower": lower, "upper": upper}

    F = np.column_stack([self._stacking_predict(m, X)[0] for m in self.members])
    P = np.clip(F, 1e-6, 1 - 1e-6)
    w = np.array([self.weights_[name] for name in self.names])
    pooled = np.clip(P @ w, 1e-6, 1 - 1e-6)
    temperature = getattr(self, "temperature_", 1.0)
    if temperature != 1.0:
        logit = np.log(pooled / (1 - pooled))
        pooled = np.clip(1.0 / (1.0 + np.exp(-logit / temperature)), 1e-6, 1 - 1e-6)
    total, aleatoric, epistemic = entropy_decomposition(P, w)
    return {"proba": pooled, "entropy_total": total, "entropy_aleatoric": aleatoric,
           "entropy_epistemic": epistemic}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `OMP_NUM_THREADS=1 pytest test/test_fusion_stacking.py -v`
Expected: PASS (37 tests total, 0 skipped — the mixed-membership test from Task 5 is now unskipped and passing)

- [ ] **Step 5: Run the full fusion test suite to check for regressions**

Run: `OMP_NUM_THREADS=1 pytest test/test_fusion_ensemble.py test/test_fusion_stacking.py test/test_fusion_gp.py -v`
Expected: PASS, no regressions in existing `FusionEnsemble`/`GPInteraction` tests

- [ ] **Step 6: Commit**

```bash
git add tackai/fusion/ensemble.py test/test_fusion_stacking.py
git commit -m "feat(fusion): FusionEnsemble.predict_stacked

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 8: Review Focus regression tests and full-suite check

**Files:**
- Modify: `test/test_fusion_stacking.py`

**Interfaces:**
- Consumes: everything from Tasks 1–7.
- Produces: nothing new; this task only adds the remaining Review Focus coverage not already exercised by earlier tasks' own tests, and runs a final full-suite check.

Four of the five Review Focus items are already covered by earlier tasks' tests (single-member ensemble: Task 1 step 6/Task 5; tiny `D_fit` for lambda CV: Task 3 step 1; before-fit-stacking calls: Tasks 6–7; single-class `D_cal`/`D_fit`: covered indirectly by `check_labels` elsewhere, but not yet exercised through `fit_stacking` itself). This task adds the one missing case and a NaN-in-predictions check.

- [ ] **Step 1: Write the failing tests**

```python
# Append to test/test_fusion_stacking.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `OMP_NUM_THREADS=1 pytest test/test_fusion_stacking.py -v -k "single_class_in_fit_set or rejects_nan_in_explicit_split"`
Expected: FAIL — `test_fit_stacking_classification_single_class_in_fit_set_raises` fails because `fit_stacking` currently calls `select_lambda_classification`, which calls `StratifiedKFold`, which will raise an unhelpful `ValueError` from sklearn about needing at least 2 classes, not naming "class" the way the test expects (or may raise a different, confusing sklearn error) — this exposes a real gap. `test_fit_stacking_rejects_nan_in_explicit_split` should already PASS from Task 5 — confirm it does.

- [ ] **Step 3: Add an explicit single-class guard in `fit_stacking`**

In `tackai/fusion/ensemble.py`, `fit_stacking` already has this line (from Task 5), right after both branches resolve `X_fit`/`y_fit`/`X_cal`/`y_cal`:

```python
    task_type = TASK_TYPES[self.task]
```

Insert the guard directly below that line, before the `F = np.column_stack(...)` line that follows it:

```python
    task_type = TASK_TYPES[self.task]
    if task_type == "binary":
        for arr, name in ((y_fit, "y_fit"), (y_cal, "y_cal")):
            if len(np.unique(arr)) < 2:
                raise ValueError(f"{name} has a single class; nothing to learn a pooled "
                                 "weight from. This is a split problem, not something "
                                 "fit_stacking can fix")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `OMP_NUM_THREADS=1 pytest test/test_fusion_stacking.py -v`
Expected: PASS (39 tests total)

- [ ] **Step 5: Run the full test suite**

Run: `OMP_NUM_THREADS=1 pytest`
Expected: PASS, no regressions anywhere (230+ existing tests plus the new 39)

- [ ] **Step 6: Commit**

```bash
git add tackai/fusion/ensemble.py test/test_fusion_stacking.py
git commit -m "fix(fusion): fit_stacking raises a clear error on single-class splits

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```
