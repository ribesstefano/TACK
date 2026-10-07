"""Mixture-MLE stacking weights and uncertainty, consumed by FusionEnsemble.

Pure-NumPy statistical core for the stacking fit described in
docs/superpowers/specs/2026-10-07-fusion-stacked-ensemble-design.md. Every function here takes
plain arrays (predictions/sigmas/labels), never a model or a FusionEnsemble, so the math is
testable on small synthetic data without fitting anything. FusionEnsemble.fit_stacking /
calibrate_stacking / predict_stacked are thin orchestration around these functions.
"""
from typing import Tuple

import numpy as np
from scipy.optimize import minimize

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
