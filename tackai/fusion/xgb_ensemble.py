"""Stacked XGBoost ensemble with uncertainty, for regression and binary classification.

Weights over fitted XGBoost models start uniform, so the ensemble predicts
without fitting. `fit` learns simplex weights on a held-out stacking set: a
Gaussian mixture likelihood for regression, a linear pool with log loss for
classification. The task is inferred from the models' objective.

    ens = XGBStackedEnsemble.from_pretrained("path/to/models")     # *.json / *.ubj
    out = ens.predict(X_new)                 # uniform weights until fitted
    ens.fit(X_fit, y_fit).calibrate(X_cal, y_cal)
    out = ens.predict(X_new)                 # dict of arrays
    ens.push_to_hub("user/my-ensemble")
    ens = XGBStackedEnsemble.from_pretrained("user/my-ensemble")
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xgboost as xgb
from huggingface_hub import ModelHubMixin, snapshot_download
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, log_expit, log_softmax, logit, logsumexp, softmax, xlog1py, xlogy

EPS = 1e-6  # probability clipping
CONFIG = "config.json"
MODEL_SUFFIXES = {".json", ".ubj"}
HPARAMS = ("lambdas", "n_restarts", "n_folds", "sigma_floor_ratio", "alpha", "seed")
STATE = ("weights_", "sigmas_", "lambda_", "q_hat_", "temperature_")


# --------------------------------------------------------------------------- #
# Objectives: each returns (loss, gradient) for scipy's `jac=True`.
# --------------------------------------------------------------------------- #
def _shrinkage(w, lam):
    """Penalty lam * ||w - 1/N||^2 and its gradient with respect to w."""
    g = w - 1 / w.size
    return lam * g @ g, 2 * lam * g


def _through_softmax(w, grad_w):
    """Chain rule from a gradient in w to one in theta, with w = softmax(theta)."""
    return w * (grad_w - w @ grad_w)


def _regression_loss(params, F, y, lam):
    """Penalised mean NLL of sum_i w_i N(y; f_i, sigma_i^2); params = [theta, log sigma]."""
    theta, log_sigma = np.split(params, 2)
    w = softmax(theta)
    z = (y[:, None] - F) / np.exp(log_sigma)
    log_comp = log_softmax(theta) - 0.5 * z**2 - log_sigma - 0.5 * np.log(2 * np.pi)
    log_mix = logsumexp(log_comp, axis=1, keepdims=True)
    resp = np.exp(log_comp - log_mix)  # responsibilities
    penalty, penalty_grad = _shrinkage(w, lam)
    grad_theta = w - resp.mean(0) + _through_softmax(w, penalty_grad)
    grad_log_sigma = -(resp * (z**2 - 1)).mean(0)
    return penalty - log_mix.mean(), np.concatenate([grad_theta, grad_log_sigma])


def _classification_loss(theta, P, y, lam):
    """Penalised log loss of the linear pool p = P w."""
    w = softmax(theta)
    p = P @ w
    penalty, penalty_grad = _shrinkage(w, lam)
    loss = -np.mean(xlogy(y, p) + xlog1py(1 - y, -p)) + penalty
    grad_w = P.T @ ((p - y) / (p * (1 - p))) / y.size + penalty_grad
    return loss, _through_softmax(w, grad_w)


def _entropy(q):
    """Binary entropy in nats."""
    return -(xlogy(q, q) + xlog1py(1 - q, -q))


LOSSES = {"regression": _regression_loss, "classification": _classification_loss}


def _load_model(path):
    """Load a saved model as an `XGBRegressor` or `XGBClassifier`, whichever it was saved as."""
    for estimator in (xgb.XGBRegressor, xgb.XGBClassifier):
        model = estimator()
        try:
            model.load_model(path)
            return model
        except TypeError:  # XGBoost refuses to load a file saved by the other estimator type
            continue
    raise TypeError(f"{path} is neither an XGBoost regressor nor a classifier.")


def _infer_task(models):
    objectives = {m.get_xgb_params()["objective"] for m in models}
    if objectives == {"binary:logistic"}:
        return "classification"
    if all(str(o).startswith("reg:") for o in objectives):
        return "regression"
    raise ValueError(f"Need all 'binary:logistic' or all 'reg:*' models, got {objectives}")


# --------------------------------------------------------------------------- #
# Ensemble
# --------------------------------------------------------------------------- #
class XGBStackedEnsemble(ModelHubMixin, library_name="xgboost", tags=["xgboost", "ensemble", "uncertainty"]):
    """Weighted ensemble of scikit-learn API XGBoost models (`XGBRegressor`,
    `XGBClassifier`, `XGBModel`) returning a mean and an uncertainty.

    Weights are uniform until `fit` is called. Before fitting, the regression
    noise term is unknown and set to zero, so `std` is model disagreement only.

    `X` is anything `xgboost.DMatrix` accepts. Fit on data disjoint from the
    models' training data, and calibrate on data disjoint from both.
    """

    def __init__(self, models, *, sigma="constant", lambdas=(0.0, 0.01, 0.1, 1.0, 10.0),
                 n_restarts=5, n_folds=5, sigma_floor_ratio=1e-3, alpha=0.1, seed=0):
        if sigma != "constant":
            # TODO: input-dependent sigma_i(x): quantile pair (B), residual model (C),
            # distributional boosting (D). They would enter as s_i * sigma_i(x).
            raise NotImplementedError("Only sigma='constant' (option A) is implemented.")
        self.models = list(models)
        self.task = _infer_task(self.models)
        self.lambdas, self.n_restarts, self.n_folds = list(lambdas), n_restarts, n_folds
        self.sigma_floor_ratio, self.alpha, self.seed = sigma_floor_ratio, alpha, seed
        n = len(self.models)
        self.weights_, self.sigmas_ = np.full(n, 1 / n), np.zeros(n)  # uniform, unknown noise
        self.lambda_ = self.q_hat_ = self.temperature_ = None

    # ------------------------------ fitting -------------------------------- #
    def fit(self, X, y):
        """Learn the weights (and per-model sigmas for regression) on the stacking set."""
        preds, y = self._base_predictions(X), np.asarray(y, float)
        rng = np.random.default_rng(self.seed)
        folds = np.array_split(rng.permutation(y.size), self.n_folds)
        loss = LOSSES[self.task]

        def cv_loss(lam):  # unpenalised held-out loss
            held_out = []
            for val in folds:
                train = np.setdiff1d(np.arange(y.size), val)
                params = self._optimise(preds[train], y[train], lam, rng)
                held_out.append(loss(params, preds[val], y[val], 0.0)[0])
            return np.mean(held_out)

        self.lambda_ = min(self.lambdas, key=cv_loss) if len(self.lambdas) > 1 else self.lambdas[0]
        params = self._optimise(preds, y, self.lambda_, rng)
        n = len(self.models)
        self.weights_, self.sigmas_ = softmax(params[:n]), np.exp(params[n:])
        self.q_hat_ = self.temperature_ = None
        return self

    def _optimise(self, preds, y, lam, rng):
        n = preds.shape[1]
        if self.task == "regression":  # non-convex in sigma: restarts and a variance floor
            floor = max(self.sigma_floor_ratio * y.std(), 1e-12)
            rmse = np.sqrt(np.mean((y[:, None] - preds) ** 2, axis=0))
            tail, n_starts = np.log(np.maximum(rmse, floor)), self.n_restarts
            bounds = [(None, None)] * n + [(np.log(floor), None)] * n
        else:  # convex: one run from uniform weights
            tail, n_starts, bounds = np.empty(0), 1, None
        thetas = [np.zeros(n)] + [rng.normal(0, 0.5, n) for _ in range(n_starts - 1)]
        runs = [minimize(LOSSES[self.task], np.concatenate([theta, tail]), args=(preds, y, lam),
                         jac=True, method="L-BFGS-B", bounds=bounds) for theta in thetas]
        return min(runs, key=lambda run: run.fun).x

    def calibrate(self, X, y):
        """Conformal quantile (regression) or temperature (classification) on a calibration set."""
        self.q_hat_ = self.temperature_ = None
        out, y = self.predict(X), np.asarray(y, float)
        if self.task == "regression":
            scores = np.sort(np.abs(y - out["mean"]) / out["std"])
            k = int(np.ceil((y.size + 1) * (1 - self.alpha)))
            if k > y.size:
                raise ValueError(f"Calibration set too small for alpha={self.alpha}.")
            self.q_hat_ = float(scores[k - 1])
        else:
            z, sign = logit(out["proba"]), 2 * y - 1
            log_loss = lambda log_t: -np.mean(log_expit(sign * z / np.exp(log_t)))
            self.temperature_ = float(np.exp(minimize_scalar(log_loss, bounds=(-3, 3), method="bounded").x))
        return self

    # ----------------------------- prediction ------------------------------ #
    def _base_predictions(self, X):
        """Matrix (samples, models) of base predictions; probabilities are clipped."""
        # data = xgb.DMatrix(X)
        # preds = np.column_stack([m.get_booster().predict(data) for m in self.models]).astype(float)
        preds = np.column_stack([m.predict(X) for m in self.models]).astype(float)
        return np.clip(preds, EPS, 1 - EPS) if self.task == "classification" else preds

    def predict(self, X):
        """Return a dict of arrays: the mean prediction and its uncertainty decomposition."""
        preds, w = self._base_predictions(X), self.weights_
        if self.task == "regression":
            mean = preds @ w
            var_noise = np.full_like(mean, w @ self.sigmas_**2)
            var_disagreement = (preds - mean[:, None]) ** 2 @ w
            out = dict(mean=mean, std=np.sqrt(var_noise + var_disagreement),
                       std_noise=np.sqrt(var_noise), std_disagreement=np.sqrt(var_disagreement))
            if self.q_hat_ is not None:  # conformal interval at level 1 - alpha
                half_width = self.q_hat_ * out["std"]
                out.update(lower=mean - half_width, upper=mean + half_width)
            return out
        pooled = preds @ w
        proba = pooled if self.temperature_ is None else expit(logit(pooled) / self.temperature_)
        aleatoric = _entropy(preds) @ w
        # Aleatoric and epistemic terms come from the uncalibrated pool, where the split is exact.
        return dict(proba=proba, entropy_total=_entropy(proba), entropy_aleatoric=aleatoric,
                    entropy_epistemic=np.maximum(_entropy(pooled) - aleatoric, 0.0))

    # ------------------------- Hugging Face Hub ---------------------------- #
    # ModelHubMixin builds save_pretrained, from_pretrained and push_to_hub on these two hooks.
    def _save_pretrained(self, save_directory: Path) -> None:
        for i, model in enumerate(self.models):
            model.save_model(save_directory / f"model_{i:03d}.ubj")
        config = {"task": self.task,
                  "init": {k: getattr(self, k) for k in HPARAMS},
                  "state": {k: getattr(self, k) for k in STATE}}
        (save_directory / CONFIG).write_text(json.dumps(config, indent=2, default=lambda a: a.tolist()))

    @classmethod
    def _from_pretrained(cls, *, model_id, revision=None, cache_dir=None, force_download=False,
                         local_files_only=False, token=None, **kwargs):
        """Load from a local directory of models, a saved ensemble, or a Hub repo."""
        path = Path(model_id)
        if not path.is_dir():
            path = Path(snapshot_download(model_id, revision=revision, cache_dir=cache_dir, token=token,
                                          force_download=force_download, local_files_only=local_files_only))
        files = sorted(f for f in path.iterdir() if f.suffix in MODEL_SUFFIXES and f.name != CONFIG)
        if not files:
            raise FileNotFoundError(f"No model files ({', '.join(sorted(MODEL_SUFFIXES))}) in {path}")
        config = json.loads((path / CONFIG).read_text()) if (path / CONFIG).exists() else {}
        overrides = {k: v for k, v in kwargs.items() if k in HPARAMS}
        ensemble = cls([_load_model(f) for f in files], **{**config.get("init", {}), **overrides})
        for key, value in config.get("state", {}).items():
            setattr(ensemble, key, np.asarray(value) if isinstance(value, list) else value)
        return ensemble