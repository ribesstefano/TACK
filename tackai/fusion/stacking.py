"""Mixture-MLE stacking weights and uncertainty, consumed by FusionEnsemble.

Pure-NumPy statistical core for the stacking fit described in
docs/superpowers/specs/2026-10-07-fusion-stacked-ensemble-design.md. Every function here takes
plain arrays (predictions/sigmas/labels), never a model or a FusionEnsemble, so the math is
testable on small synthetic data without fitting anything. FusionEnsemble.fit_stacking /
calibrate_stacking / predict_stacked are thin orchestration around these functions.
"""
import warnings
from typing import Tuple

import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import KFold, StratifiedKFold

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
