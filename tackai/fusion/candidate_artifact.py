"""Portable artifact wrapper for calibrated non-XGBoost tree candidates.

This wrapper accepts already-fitted sklearn-compatible estimators. Recipe
selection and training-data restrictions belong to the explicit fit tool.
Regression residual scale is fitted on stacking rows, then frozen before
independent conformal calibration. Reported uncertainty is empirical, not an
exact aleatoric/epistemic decomposition.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Optional, Sequence

import joblib
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn import __version__ as sklearn_version

from tackai.fusion.molecular_spec import MolecularFeatureSpec

ARTIFACT_SCHEMA_VERSION = 1
_EPS = 1e-7


def _entropy(probabilities):
    """Binary entropy in nats, stable at clipped probability endpoints."""
    p = np.clip(np.asarray(probabilities, dtype=float), _EPS, 1 - _EPS)
    return -(p * np.log(p) + (1 - p) * np.log1p(-p))


def _predict_one(model, X, task):
    """Use the estimator's public prediction API as its candidate prediction."""
    if task == "regression":
        values = np.asarray(model.predict(X), dtype=float).reshape(-1)
        if not np.all(np.isfinite(values)):
            raise ValueError("regression candidate returned non-finite predictions")
        return values
    if not hasattr(model, "predict_proba"):
        raise TypeError("classification candidates must provide predict_proba")
    classes = np.asarray(getattr(model, "classes_", [0, 1]))
    positive = np.flatnonzero(classes == 1)
    if positive.size != 1:
        raise ValueError("classification candidate must expose class label 1")
    probabilities = np.asarray(model.predict_proba(X), dtype=float)
    values = probabilities[:, int(positive[0])].reshape(-1)
    if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("classification candidate returned invalid probabilities")
    return values


class CandidateArtifact:
    """Calibrated, torch-free inference artifact for fitted tree candidates.

    Args:
        models: One or more fitted sklearn-compatible estimators. Each model is
            one ensemble member; internal tree arrays are not treated as models.
        task: ``regression`` or ``classification`` (activity).
        molecular_spec: Exact molecular featurization contract used in training.
        weights: Optional nonnegative simplex weights, normally fitted on the
            reserved stacking groups. Defaults to uniform weights.
        recipe: JSON metadata pinning model family, parameters, task and feature
            preset selected by the fit tool.
        context_contract: JSON metadata describing the frozen context feature
            space and source snapshot used to build the design matrix.
    """

    def __init__(self, models: Sequence, task: str,
                 molecular_spec: MolecularFeatureSpec,
                 weights: Optional[Sequence[float]] = None,
                 recipe: Optional[dict] = None,
                 context_contract: Optional[dict] = None):
        if task not in {"regression", "classification"}:
            raise ValueError("task must be 'regression' or 'classification'")
        if len(models) == 0:
            raise ValueError("at least one fitted estimator is required")
        if not isinstance(molecular_spec, MolecularFeatureSpec):
            raise TypeError("molecular_spec must be a MolecularFeatureSpec")
        self.models = list(models)
        self.task = task
        self.molecular_spec = molecular_spec
        self.recipe = self._json_object(recipe, "recipe")
        self.context_contract = self._json_object(context_contract, "context_contract")
        self.weights_ = self._validate_weights(weights)
        widths = {int(model.n_features_in_) for model in self.models
                  if hasattr(model, "n_features_in_")}
        if len(widths) > 1:
            raise ValueError("candidate models disagree on input feature width")
        self.feature_width_ = next(iter(widths), None)
        self.residual_scale_ = None
        self.q_hat_ = None
        self.alpha_ = None
        self.temperature_ = None

    @staticmethod
    def _json_object(value, name):
        value = {} if value is None else value
        if not isinstance(value, dict):
            raise TypeError(f"{name} must be a JSON-compatible object")
        try:
            json.dumps(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be JSON serializable") from exc
        return value

    def _validate_weights(self, weights):
        values = (np.ones(len(self.models), dtype=float) if weights is None
                  else np.asarray(weights, dtype=float))
        if values.shape != (len(self.models),) or not np.all(np.isfinite(values)):
            raise ValueError("weights must be a finite vector aligned with models")
        if np.any(values < 0) or values.sum() <= 0:
            raise ValueError("weights must be nonnegative with positive sum")
        return values / values.sum()

    def _validate_X(self, X):
        shape = getattr(X, "shape", None)
        if shape is None or len(shape) != 2:
            raise ValueError("X must be a two-dimensional design matrix")
        if self.feature_width_ is not None and shape[1] != self.feature_width_:
            raise ValueError("X feature width does not match the fitted candidate")
        return shape[0]

    def _predictions(self, X):
        n_rows = self._validate_X(X)
        if n_rows == 0:
            return np.empty((0, len(self.models)), dtype=float)
        predictions = np.column_stack([_predict_one(model, X, self.task) for model in self.models])
        if predictions.shape != (n_rows, len(self.models)):
            raise ValueError("candidate predictions are not row aligned")
        if not np.all(np.isfinite(predictions)):
            raise ValueError("candidate returned non-finite predictions")
        return predictions

    def fit_residual_scale(self, X_stacking, y_stacking):
        """Fit and freeze the global RMS residual scale on reserved stacking rows.

        Call this before regression ``calibrate`` and use rows disjoint from both
        base-model training and conformal calibration. The method does not fit or
        refit model weights.
        """
        if self.task != "regression":
            raise ValueError("residual scale is only defined for regression")
        predictions = self._predictions(X_stacking)
        y = np.asarray(y_stacking, dtype=float)
        if y.ndim != 1 or y.size != predictions.shape[0] or not np.all(np.isfinite(y)):
            raise ValueError("stacking labels must be finite and aligned with X_stacking")
        if y.size == 0:
            raise ValueError("stacking set must not be empty")
        residual = y - predictions @ self.weights_
        self.residual_scale_ = max(float(np.sqrt(np.mean(residual ** 2))), 1e-12)
        self.q_hat_ = self.alpha_ = None
        return self

    def calibrate(self, X, y, alpha: float = 0.1):
        """Calibrate intervals or probabilities after all model choices are fixed.

        Regression requires a residual scale already fitted on independent
        stacking rows. Its scale is frozen while the calibration set supplies
        only the conformal quantile. Classification fits one temperature to the
        weighted pool of equally temperature-scaled member probabilities.
        """
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between zero and one")
        predictions = self._predictions(X)
        y = np.asarray(y, dtype=float)
        n_rows = predictions.shape[0]
        if y.ndim != 1 or y.size != n_rows or not np.all(np.isfinite(y)):
            raise ValueError("y must be finite, one-dimensional, and aligned with X")
        if n_rows == 0:
            raise ValueError("calibration set must not be empty")
        if self.task == "regression":
            if self.residual_scale_ is None:
                raise ValueError(
                    "fit_residual_scale on stacking rows before regression calibration")
            mean = predictions @ self.weights_
            scale = self._predictive_scale(predictions)
            scores = np.sort(np.abs(y - mean) / scale)
            rank = int(np.ceil((n_rows + 1) * (1 - alpha)))
            if rank > n_rows:
                raise ValueError("calibration set is too small for requested alpha")
            self.q_hat_ = float(scores[rank - 1])
            self.alpha_ = float(alpha)
        else:
            if not np.isin(y, [0, 1]).all():
                raise ValueError("classification calibration labels must be 0 or 1")
            if np.unique(y).size != 2:
                raise ValueError("classification calibration requires both labels")
            member_probs = np.clip(predictions, _EPS, 1 - _EPS)
            signs = 2 * y - 1

            def objective(log_temperature):
                temperature = np.exp(log_temperature)
                calibrated = expit(logit(member_probs) / temperature)
                pooled = np.clip(calibrated @ self.weights_, _EPS, 1 - _EPS)
                return float(np.mean(np.logaddexp(
                    0, -signs * logit(pooled))))

            result = minimize_scalar(objective, bounds=(-3, 3), method="bounded")
            if not result.success or not np.isfinite(result.fun):
                raise ValueError("temperature calibration failed")
            self.temperature_ = float(np.exp(result.x))
        return self

    def _disagreement(self, predictions):
        mean = predictions @ self.weights_
        variance = np.sum(self.weights_[None, :] * (predictions - mean[:, None]) ** 2, axis=1)
        return np.sqrt(np.maximum(variance, 0.0))

    def _predictive_scale(self, predictions):
        disagreement = self._disagreement(predictions)
        return np.sqrt(self.residual_scale_ ** 2 + disagreement ** 2)

    def predict(self, X):
        """Return the TACK fast-model prediction keys for regression or activity."""
        predictions = self._predictions(X)
        pooled = predictions @ self.weights_
        if self.task == "regression":
            disagreement = self._disagreement(predictions)
            result = {"mean": pooled, "std_disagreement": disagreement}
            # An unfitted residual scale is unknown, not zero uncertainty.
            if self.residual_scale_ is None:
                return result
            noise = np.full_like(pooled, self.residual_scale_)
            std = np.sqrt(noise ** 2 + disagreement ** 2)
            result.update(std=std, std_noise=noise)
            if self.q_hat_ is not None:
                result["lower"] = pooled - self.q_hat_ * std
                result["upper"] = pooled + self.q_hat_ * std
            return result

        member_probs = np.clip(predictions, _EPS, 1 - _EPS)
        if self.temperature_ is not None:
            member_probs = expit(logit(member_probs) / self.temperature_)
        proba = member_probs @ self.weights_
        total = _entropy(proba)
        aleatoric = _entropy(member_probs) @ self.weights_
        epistemic = total - aleatoric
        return {"proba": proba, "entropy_total": total,
                "entropy_aleatoric": aleatoric, "entropy_epistemic": epistemic}

    def help(self, show_extras: bool = False):
        """Describe prediction keys and empirical uncertainty interpretations."""
        if self.task == "regression":
            keys = ["mean", "std_disagreement"]
            if self.residual_scale_ is not None:
                keys += ["std", "std_noise"]
            if self.q_hat_ is not None:
                keys += ["lower", "upper"]
            explanations = {
                "std_noise": ("Global RMS residual scale fitted on stacking rows; empirical, "
                              "not a learned aleatoric model."),
                "std_disagreement": ("Weighted spread of candidate predictions around the "
                                     "pooled mean."),
                "std": ("Quadrature sum of the stacking residual scale and model "
                        "disagreement."),
            }
            if self.q_hat_ is not None:
                explanations.update({
                    "lower": (f"mean - q_hat_ * std; finite-sample conformal interval targeting "
                              f"{1 - self.alpha_:.0%} calibration coverage under exchangeability."),
                    "upper": "mean + q_hat_ * std.",
                })
        else:
            keys = ["proba", "entropy_total", "entropy_aleatoric", "entropy_epistemic"]
            explanations = {
                "entropy_total": "Binary entropy of the calibrated pooled probability, in nats.",
                "entropy_aleatoric": ("Weighted mean entropy of calibrated member probabilities, "
                                      "in nats."),
                "entropy_epistemic": ("Total minus member entropy; exact entropy decomposition, "
                                      "in nats."),
            }
        result = {"task": self.task, "output_keys": keys}
        if show_extras:
            result["hyperparameters"] = {"weights_": self.weights_.tolist(),
                                          "residual_scale_": self.residual_scale_,
                                          "q_hat_": self.q_hat_,
                                          "alpha_": self.alpha_,
                                          "temperature_": self.temperature_}
            result["explanations"] = explanations
        return result

    def save_pretrained(self, directory):
        """Write a local artifact with feature, context, version and integrity contracts."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=False)
        model_path = path / "candidate.joblib"
        joblib.dump(self, model_path)
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        manifest = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "format": "tack-tree-candidate/v1",
            "task": self.task,
            "weights": self.weights_.tolist(),
            "feature_width": self.feature_width_,
            "residual_scale": self.residual_scale_,
            "q_hat": self.q_hat_,
            "alpha": self.alpha_,
            "temperature": self.temperature_,
            "molecular_feature_spec": self.molecular_spec.to_dict(),
            "recipe": self.recipe,
            "context_contract": self.context_contract,
            "versions": {"scikit-learn": sklearn_version, "joblib": joblib.__version__,
                         "numpy": np.__version__},
            "model_sha256": digest,
            "uncertainty_contract": self.help(show_extras=True)["explanations"],
        }
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        return path

    @classmethod
    def from_pretrained(cls, directory):
        """Load a trusted local artifact after validating schema, versions and integrity.

        Joblib uses pickle; hashes detect corruption but do not authenticate the
        source. Only load artifact directories from trusted producers.
        """
        path = Path(directory)
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise ValueError("unsupported candidate artifact schema")
        if manifest.get("format") != "tack-tree-candidate/v1":
            raise ValueError("unrecognized candidate artifact format")
        for package, runtime in (("scikit-learn", sklearn_version),
                                 ("joblib", joblib.__version__), ("numpy", np.__version__)):
            saved = manifest["versions"][package].split(".")[:2]
            if saved != runtime.split(".")[:2]:
                raise ValueError(f"{package} minor version mismatch: "
                                 f"artifact={manifest['versions'][package]}, runtime={runtime}")
        packages = manifest.get("context_contract", {}).get("runtime_model_package_versions", {})
        for package in ("xgboost", "catboost"):
            if package in packages:
                try:
                    installed = importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError as exc:
                    raise ValueError(f"{package} is required by this candidate artifact") from exc
                if installed.split(".")[:2] != packages[package].split(".")[:2]:
                    raise ValueError(f"{package} minor version mismatch: "
                                     f"artifact={packages[package]}, runtime={installed}")
        model_path = path / "candidate.joblib"
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if digest != manifest.get("model_sha256"):
            raise ValueError("candidate model integrity check failed")
        spec = MolecularFeatureSpec.from_dict(manifest["molecular_feature_spec"])
        artifact = joblib.load(model_path)
        if not isinstance(artifact, cls):
            raise TypeError("candidate.joblib does not contain a CandidateArtifact")
        expected = (artifact.molecular_spec == spec
                    and artifact.task == manifest["task"]
                    and artifact.recipe == manifest.get("recipe", {})
                    and artifact.context_contract == manifest.get("context_contract", {})
                    and artifact.feature_width_ == manifest.get("feature_width")
                    and np.allclose(artifact.weights_, manifest["weights"])
                    and _same_optional_float(artifact.residual_scale_, manifest["residual_scale"])
                    and _same_optional_float(artifact.q_hat_, manifest["q_hat"])
                    and _same_optional_float(artifact.alpha_, manifest["alpha"])
                    and _same_optional_float(artifact.temperature_, manifest["temperature"])
                    and artifact.help(show_extras=True)["explanations"]
                    == manifest["uncertainty_contract"])
        if not expected:
            raise ValueError("candidate manifest does not match serialized artifact")
        return artifact


def _same_optional_float(left, right):
    """Compare nullable serialized scalar state without accepting NaN mismatches."""
    if left is None or right is None:
        return left is None and right is None
    return bool(np.isclose(float(left), float(right), rtol=1e-12, atol=1e-15))
