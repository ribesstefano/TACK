"""A generic ensemble of fusion members with a cached biological context.

Every member consumes the *same* design matrix, which is what makes the class generic: a
member may be a :class:`~tackai.fusion.gp.GPInteraction` or any other estimator sharing its
``fit``/``predict``/``report`` contract, and the ensemble never needs to know which. It also
makes the context cache possible: encode one experimental context once, fold it into every
member that can, and then every new batch of molecules only pays for its own featurisation.

The API mirrors :class:`tackai.ensemble_predictor.EnsemblePredictor`: build with
:meth:`FusionEnsemble.from_pretrained`, score with :meth:`FusionEnsemble.predict`.
"""
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from tackai.fusion.context import ContextEncoder
from tackai.fusion.data import BLOCK_ORDER, TASK_LABELS, TASK_SUPPORT, TASK_TYPES, FusionData
from tackai.fusion.mol_encoder import DESCRIPTOR_NAMES, MolEncoder
from tackai.fusion.gp import GPInteraction
from tackai.fusion.stacking import (conformal_quantile, fit_mixture_weights, fit_pooled_weights,
                                    select_lambda_classification, select_lambda_regression)
from tackai.fusion.training import check_labels

Z95 = 1.959963984540054          # two-sided 95% normal quantile


@dataclass
class FusionPrediction:
    """Ensemble prediction for a batch of molecules in one context.

    Attributes:
        mean: Weighted ensemble prediction, in the task's own units (a probability for the
            binary task). NaN where the SMILES could not be parsed.
        std: Total uncertainty by the law of total variance — the weighted mean of the
            members' predictive variances plus the weighted spread of their means.
        member_predictions: Per-member prediction.
        member_std: Per-member predictive standard deviation (zeros for a member without one).
        weights: Member weights used for the average.
        task: Task name.
        label_name: Human-readable label of the predicted quantity.
        ok: False where the input SMILES could not be parsed.
        ci_lower_95 / ci_upper_95: ``mean +/- 1.96 * std``, clipped to the task's own support
            where it has one (Dmax is a fraction, activity a probability), so an interval
            never reports an impossible value. This is an interval on the latent mean: it
            covers the ensemble's disagreement and each member's posterior variance, not the
            assay's own measurement noise, which the GP fits separately and excludes.
    """

    mean: np.ndarray
    std: np.ndarray
    member_predictions: Dict[str, np.ndarray]
    member_std: Dict[str, np.ndarray]
    weights: Dict[str, float]
    task: str
    label_name: str
    ok: np.ndarray
    ci_lower_95: np.ndarray = field(default=None)
    ci_upper_95: np.ndarray = field(default=None)
    smiles: Optional[List[str]] = None

    def __post_init__(self):
        support = TASK_SUPPORT.get(self.task)
        if self.ci_lower_95 is None:
            self.ci_lower_95 = self.mean - Z95 * self.std
            if support is not None:
                self.ci_lower_95 = np.clip(self.ci_lower_95, support[0], support[1])
        if self.ci_upper_95 is None:
            self.ci_upper_95 = self.mean + Z95 * self.std
            if support is not None:
                self.ci_upper_95 = np.clip(self.ci_upper_95, support[0], support[1])

    def to_frame(self) -> pd.DataFrame:
        """One row per molecule, with the prediction, its uncertainty and the interval."""
        frame = pd.DataFrame({"prediction": self.mean, "std": self.std,
                              "ci_lower_95": self.ci_lower_95, "ci_upper_95": self.ci_upper_95,
                              "ok": self.ok})
        if self.smiles is not None:
            frame.insert(0, "smiles", self.smiles)
        return frame

    def summary(self) -> str:
        """One-line description of the batch."""
        valid = np.isfinite(self.mean)
        if not valid.any():
            return f"{self.label_name}: no valid molecule in {len(self.mean)} input(s)"
        return (f"{self.label_name}: {np.mean(self.mean[valid]):.3f} mean over {valid.sum()} "
                f"molecule(s), typical uncertainty {np.mean(self.std[valid]):.3f} "
                f"({len(self.weights)} members)")


@dataclass
class FusionContext:
    """A biological context encoded once and folded into every member that can.

    Attributes:
        values: The encoded context columns, shape ``(1, n_context_columns)``.
        blocks: The same context as ``{block: (1, dim) array}``.
        source: The record the context was built from.
        folds: For each member that supports it, the context-only kernel terms collapsed to
            two vectors, so a batch pays only for the molecular kernels.
    """

    values: np.ndarray
    blocks: Dict[str, np.ndarray]
    source: dict
    folds: List[Optional[dict]] = field(default_factory=list)


class FusionEnsemble:
    """Weighted ensemble of fusion members sharing one design matrix and one context.

    Args:
        members: Fitted estimators, all consuming the design matrix of ``data``.
        data: The :class:`FusionData` whose encoder and layout the members were fitted with.
        task: Task name (``"dmax"``, ``"pdc50"`` or ``"activity"``).
        weights: Member weights (default: equal).

    Raises:
        ValueError: If a member that carries a block layout (a GP) was fitted with one other
            than ``data.blocks_indexes``.
    """

    def __init__(self, members: Sequence, data: FusionData, task: str,
                 weights: Optional[Union[Sequence[float], Dict[str, float]]] = None):
        if task not in TASK_TYPES:
            raise ValueError(f"task must be one of {sorted(TASK_TYPES)}, got {task!r}")
        self.members = list(members)
        self.data = data
        self.task = task
        self.names = [f"member_{i:02d}" for i in range(len(self.members))]
        self._check_members()
        self.set_weights(weights)

    def _check_members(self) -> None:
        """Refuse a member whose block layout differs from the data's.

        The ensemble hands every member the one design matrix of ``data`` and, on the cached
        context path, raw blocks cut by ``data``'s layout. A member fitted with another layout
        would read the wrong columns without any error.
        """
        expected = self.data.blocks_indexes
        for name, member in zip(self.names, self.members):
            layout = getattr(member, "blocks", None)
            if layout is None:
                continue
            if list(layout) != list(expected) or any(
                    not np.array_equal(layout[b], expected[b]) for b in expected):
                raise ValueError(
                    f"{name} was fitted with a different block layout than the data this "
                    "ensemble is built on; pass blocks=data.blocks_indexes when fitting")

    # ---------------------------------------------------------------- construction

    @classmethod
    def from_pretrained(cls, model_id: Union[str, Path], *, revision: Optional[str] = None,
                        token: Optional[str] = None,
                        data: Optional[FusionData] = None) -> "FusionEnsemble":
        """Load a saved ensemble from a local directory or a Hugging Face Hub repo.

        Args:
            model_id: Directory written by :meth:`save`, or a Hub repo id.
            revision: Hub revision, for a repo id.
            token: Hub token, for a private repo.
            data: Reuse this :class:`FusionData` instead of rebuilding an encoder-only one
                from the manifest (inference needs only the encoder and the layout).

        Returns:
            The loaded ensemble.
        """
        import joblib

        path = Path(model_id)
        if not path.exists():
            from huggingface_hub import snapshot_download
            path = Path(snapshot_download(repo_id=str(model_id), revision=revision, token=token))
        manifest = json.loads((path / "manifest.json").read_text())
        members = [joblib.load(path / name) for name in manifest["member_files"]]
        if data is None:
            encoder = ContextEncoder(protein_space=manifest.get("protein_space", "per_block"))
            featurizer = MolEncoder(radius=manifest["fingerprint"][0],
                                       fp_size=manifest["fingerprint"][1],
                                    descriptors=manifest.get("descriptors", DESCRIPTOR_NAMES))
            data = FusionData(encoder=encoder, featurizer=featurizer)
        cls._check_layout(manifest, data)
        return cls(members, data, manifest["task"], weights=manifest.get("weights"))

    @staticmethod
    def _check_layout(manifest: dict, data: FusionData) -> None:
        """Refuse to score with members whose block layout no longer matches the encoder.

        Re-fitting a context PCA changes a cached table's width. The members still slice the
        columns they were fitted with, so every context block after the changed one lands at
        the wrong offset and the predictions are quietly wrong rather than absent.

        Args:
            manifest: The saved manifest.
            data: The encoder the predictions would be made with.

        Raises:
            ValueError: If any block's width differs between the manifest and the encoder.
        """
        expected = manifest.get("block_dims")
        if not expected:
            return
        bad = {b: (expected[b], data.dims[b]) for b in expected
               if b in data.dims and expected[b] != data.dims[b]}
        if bad:
            detail = ", ".join(f"{b}: trained with {e}, the embedding cache has {a}"
                               for b, (e, a) in sorted(bad.items()))
            raise ValueError(
                f"block layout mismatch between this ensemble and the embedding cache "
                f"({detail}). The cached embedding tables have changed since the members were "
                "fitted; refit the ensemble or restore the tables it was trained on.")

    def save(self, path: Union[str, Path]) -> Path:
        """Write every member plus a manifest describing the layout it expects.

        Args:
            path: Target directory (created if needed).

        Returns:
            The directory written to.
        """
        import joblib
        import sklearn
        import torch
        import xgboost

        import tackai
        from tackai.fusion.mol_encoder import FP_RADIUS, FP_SIZE

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        files = []
        for name, member in zip(self.names, self.members):
            filename = f"{name}.joblib"
            joblib.dump(member, path / filename)
            files.append(filename)
        (path / "manifest.json").write_text(json.dumps({
            "task": self.task,
            "n_members": len(self.members),
            "member_files": files,
            "member_types": [type(m).__name__ for m in self.members],
            "weights": self.weights,
            "block_dims": self.data.dims,
            "block_order": BLOCK_ORDER,
            "protein_space": self.data.encoder.protein_space,
            "fingerprint": [self.data.featurizer.radius, self.data.featurizer.fp_size],
            "descriptors": self.data.featurizer.descriptors,
            "versions": {"tackai": getattr(tackai, "__version__", "unknown"),
                         "numpy": np.__version__, "torch": torch.__version__,
                         "xgboost": xgboost.__version__, "sklearn": sklearn.__version__},
        }, indent=1))
        return path

    def push_to_hub(self, repo_id: str, *, private: bool = False,
                    commit_message: str = "Update fusion ensemble",
                    staging_dir: Optional[Union[str, Path]] = None) -> str:
        """Save this ensemble and upload it to a Hugging Face Hub model repo.

        The inverse of :meth:`from_pretrained` with a Hub ``model_id``.

        Args:
            repo_id: Hugging Face Hub model repo id to create (if needed) and upload to.
            private: Create the repo as private if it does not exist yet.
            commit_message: Commit message for the upload.
            staging_dir: Directory to write the saved ensemble into before uploading.
                Default: a temporary directory removed once the upload finishes.

        Returns:
            The commit sha ``upload_folder`` reports.
        """
        if staging_dir is not None:
            directory = self.save(staging_dir)
        else:
            with tempfile.TemporaryDirectory() as tmp:
                return self._upload(self.save(Path(tmp) / "fusion_ensemble"), repo_id,
                                    private, commit_message)
        return self._upload(directory, repo_id, private, commit_message)

    @staticmethod
    def _upload(directory: Path, repo_id: str, private: bool, commit_message: str) -> str:
        """Create (if needed) the model repo and upload every file in ``directory``."""
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id, repo_type="model", exist_ok=True, private=private)
        commit = api.upload_folder(repo_id=repo_id, repo_type="model",
                                   folder_path=str(directory), commit_message=commit_message)
        return getattr(commit, "oid", str(commit))

    def set_weights(self, weights: Optional[Union[Sequence[float], Dict[str, float]]]) -> None:
        """Set the member weights, normalised to sum to one (default: equal).

        Args:
            weights: Sequence aligned with the members, or a mapping of member name to weight.
        """
        if weights is None:
            values = np.ones(len(self.members))
        elif isinstance(weights, dict):
            unknown = sorted(set(weights) - set(self.names))
            if unknown:
                raise ValueError(f"unknown member name(s) in weights: {unknown}; "
                                 f"this ensemble has {self.names}")
            missing = sorted(set(self.names) - set(weights))
            if missing:
                raise ValueError(f"no weight given for member(s): {missing}")
            values = np.array([float(weights[name]) for name in self.names])
        else:
            values = np.asarray(list(weights), dtype=float)
            if len(values) != len(self.members):
                raise ValueError(f"expected {len(self.members)} weights, got {len(values)}")
        total = values.sum()
        if total <= 0:
            raise ValueError("member weights must sum to a positive number")
        self.weights = {name: float(w) for name, w in zip(self.names, values / total)}

    def calibrate(self, X, y) -> "FusionEnsemble":
        """Fit each member's score-to-probability map on rows no member was fitted on.

        A member whose model does not itself output probabilities -- the GP -- scores on a
        latent scale, and only a held-out set can turn that into a probability. The set must be
        held out from every member's training rows: fitted on scores the members have already
        seen, the map comes out overconfident.

        The scores are taken from the same member scores :meth:`predict` uses, so the map is
        fitted on the scale it will be applied to.

        Args:
            X: Design matrix of the calibration rows, in this ensemble's block layout.
            y: Binary labels (0 or 1) for those rows.

        Returns:
            self

        Raises:
            ValueError: If the task is not binary, if every member outputs probabilities
                natively and there is nothing to calibrate, or if the labels are empty,
                non-finite or hold a single class.
        """
        if TASK_TYPES[self.task] != "binary":
            raise ValueError(f"only a binary task needs calibration; this ensemble predicts "
                             f"{self.task!r}")
        if all(m.native_binary for m in self.members):
            raise ValueError("every member outputs probabilities natively; nothing to calibrate")
        check_labels(y, "binary", "calibration labels")
        labels = np.asarray(y).astype(int)
        X = np.asarray(X)
        for member in self.members:
            if member.native_binary:
                continue
            score = member._predict_model(member.model_, X)
            member.calibrator_ = LogisticRegression(C=1e4).fit(score[:, None], labels)
        return self

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
        if task_type == "binary":
            for arr, name in ((y_fit, "y_fit"), (y_cal, "y_cal")):
                if len(np.unique(arr)) < 2:
                    raise ValueError(f"{name} has a single class; nothing to learn a pooled "
                                     "weight from. This is a split problem, not something "
                                     "fit_stacking can fix")
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

    @property
    def available_tasks(self) -> List[str]:
        """The tasks this ensemble can predict (one, the task it was fitted on)."""
        return [self.task]

    @property
    def promoted(self) -> List[bool]:
        """Per member: whether its GP's factorisation had to be promoted to float64."""
        return [bool(getattr(getattr(m, "model_", None), "promoted_", False))
                for m in self.members]

    # ---------------------------------------------------------------- context cache

    def transform_context(self, record: dict) -> FusionContext:
        """Encode one experimental context and fold it into every member that can.

        Scoring many molecules against one context is the common case in a screening or
        reinforcement-learning loop; this does the context's share of the work once.

        Args:
            record: Mapping with ``poi_seq``, ``e3_seq``, ``cell_id``, ``assay`` and
                optionally ``assay_time``.

        Returns:
            A :class:`FusionContext` to pass to :meth:`predict`.

        Raises:
            KeyError: If a POI or E3 sequence is not in the cached embedding tables.
        """
        blocks = self.data.encoder.encode_context(record)
        values = self.data.encode_context(record)
        folds = [self._fold_member(member, blocks) for member in self.members]
        return FusionContext(values=values, blocks=blocks, source=dict(record), folds=folds)

    @staticmethod
    def _fold_member(member, context_blocks: Dict[str, np.ndarray]) -> Optional[dict]:
        """Collapse a member's context-only kernel terms, when its model can do that.

        Six of the GP's ten kernel terms depend on the context alone and are the same for
        every molecule in a batch; folding them is what makes the context cache pay for
        itself. A member whose model has no such shortcut simply gets None.
        """
        model = getattr(member, "model_", None)
        if not hasattr(model, "fold_context"):
            return None
        try:
            return model.fold_context(context_blocks)
        except ValueError:
            return None        # an interaction this context cannot fold; use the ordinary path

    # ---------------------------------------------------------------- prediction

    def predict(self, samples: Union[Sequence[str], pd.DataFrame, Sequence[dict]],
                context: Optional[FusionContext] = None,
                return_individual: bool = True,
                return_std: bool = True) -> FusionPrediction:
        """Score molecules, either against a cached context or from full records.

        Args:
            samples: SMILES string(s) when ``context`` is given; otherwise dicts (or a
                DataFrame) carrying both the molecule and its context.
            context: A :class:`FusionContext` from :meth:`transform_context`. The fast path.
            return_individual: Keep the per-member predictions in the result.
            return_std: Ask every member that can for a predictive standard deviation.

        Returns:
            A :class:`FusionPrediction`. Rows whose SMILES RDKit cannot parse come back as
            NaN with ``ok=False`` rather than aborting the batch.
        """
        if context is not None:
            smiles = [samples] if isinstance(samples, str) else list(samples)
            ok, scores = self._predict_with_context(smiles, context, return_std)
        else:
            if isinstance(samples, list):
                smiles = samples
            else:
                records = samples.to_dict("records") if isinstance(samples, pd.DataFrame) else list(
                    [samples] if isinstance(samples, dict) else samples)
                smiles = [r.get("smiles") for r in records]
            ok, scores = self._predict_from_records(records, return_std)
        return self._aggregate(scores, ok, smiles, return_individual)

    def predict_matrix(self, X, return_individual: bool = True,
                       return_std: bool = True) -> FusionPrediction:
        """Score rows that are already encoded as a design matrix.

        This is the path an evaluation takes: a held-out fold of a :class:`FusionData` is
        already featurised, and re-encoding it from records would only repeat the work.

        Args:
            X: Design matrix with this ensemble's block layout.
            return_individual: Keep the per-member predictions in the result.
            return_std: Ask every member that can for a predictive standard deviation.

        Returns:
            A :class:`FusionPrediction` whose ``ok`` is all True (an encoded row is valid by
            construction).
        """
        X = np.asarray(X)
        n = len(X)
        ok = np.ones(n, dtype=bool)
        per_member = [self._member_scores(member, X, return_std) if n
                      else (np.empty(0), np.empty(0)) for member in self.members]
        return self._aggregate(per_member, ok, [None] * n, return_individual)

    def _predict_with_context(self, smiles: List[str], context: FusionContext,
                              return_std: bool):
        """Fast path: featurise the molecules once, score through each member's folded context."""
        n = len(smiles)
        if n == 0:
            return np.zeros(0, dtype=bool), [(np.empty(0), np.empty(0)) for _ in self.members]

        fp, desc, ok = self.data.featurizer.featurize(smiles)
        valid = np.flatnonzero(ok)
        if len(valid) < n:                       # copy only when something is invalid
            fp, desc = fp[valid], desc[valid]
        mol = {"fingerprint": fp, "descriptors": desc}

        folds = context.folds or [None] * len(self.members)
        X = None                                 # built once, only if some member needs it
        per_member = []
        for member, fold in zip(self.members, folds):
            mean, std = np.full(n, np.nan), np.zeros(n)
            if len(valid):
                if fold is not None:
                    mean[valid], std[valid] = self._member_scores(
                        member, mol, return_std, fold=fold)
                else:
                    if X is None:
                        X = self.data.assemble_features(context.values, fp, desc)
                    mean[valid], std[valid] = self._member_scores(member, X, return_std)
            per_member.append((mean, std))
        return ok, per_member

    def _predict_from_records(self, records: List[dict], return_std: bool):
        """Ordinary path: encode each record in full, then score it with every member."""
        n = len(records)
        if n == 0:
            return np.zeros(0, dtype=bool), [(np.empty(0), np.empty(0)) for _ in self.members]

        X, ok = self.data.encode(records, return_ok=True)
        valid = np.flatnonzero(ok)
        Xv = X if len(valid) == n else X[valid]
        per_member = []
        for member in self.members:
            mean, std = np.full(n, np.nan), np.zeros(n)
            if len(valid):
                mean[valid], std[valid] = self._member_scores(member, Xv, return_std)
            per_member.append((mean, std))
        return ok, per_member

    def _member_scores(self, member, X, return_std: bool, fold: Optional[dict] = None):
        """One member's prediction, in the reported units.

        Args:
            X: Design-matrix rows, or -- with ``fold`` given -- the raw molecular blocks to
                score through that member's folded context.
            return_std: Ask for a predictive standard deviation, if this member has one.
            fold: A :meth:`FusionEnsemble._fold_member` result for this member, to take the
                fast context-folded path instead of the ordinary design-matrix one.
        """
        wants_std = return_std and isinstance(member, GPInteraction)
        if fold is not None:
            result = member.model_.predict_in_context(X, fold, return_std=wants_std)
        elif wants_std:
            result = member._predict_model(member.model_, X, return_std=True)
        else:
            return member.predict(X), None
        score, std = result if wants_std else (result, None)
        return member.report(score, std)

    def _stacking_predict(self, member, X) -> "Tuple[np.ndarray, Optional[np.ndarray]]":
        """(mean, sigma) for one member, in its own reported units, for the stacking fit.

        Used only by fit_stacking/calibrate_stacking/predict_stacked — predict()/predict_matrix()
        are unaffected and remain GP-only, as documented in the design spec.

        Args:
            member: A GPInteraction, xgboost.XGBRegressor, or xgboost.XGBClassifier.
            X: Design matrix of the rows to score.

        Returns:
            ``(mean, sigma)``. ``sigma`` is the GP's own predictive std for a GPInteraction, or
            None for any XGBoost member -- Option A's constant residual sigma for an XGBoost
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

    def _aggregate(self, per_member, ok, smiles, return_individual: bool) -> FusionPrediction:
        """Weighted mean, and the law of total variance over the members."""
        weights = np.array([self.weights[name] for name in self.names])[:, None]
        means = np.array([m for m, _ in per_member], dtype=float)
        stds = np.array([s for _, s in per_member], dtype=float)
        if means.size == 0:
            mean = np.empty(0)
            std = np.empty(0)
        else:
            mean = (weights * means).sum(axis=0)
            within = (weights * stds ** 2).sum(axis=0)
            between = (weights * (means - mean) ** 2).sum(axis=0)
            std = np.sqrt(within + between)
        member_predictions = ({name: means[i] for i, name in enumerate(self.names)}
                              if return_individual and means.size else {})
        member_std = ({name: stds[i] for i, name in enumerate(self.names)}
                      if return_individual and stds.size else {})
        return FusionPrediction(mean=mean, std=std, member_predictions=member_predictions,
                                member_std=member_std, weights=dict(self.weights),
                                task=self.task, label_name=TASK_LABELS[self.task],
                                ok=np.asarray(ok, dtype=bool), smiles=list(smiles))
