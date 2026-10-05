"""The two fusion estimators: M4 (interaction GP) and M7 (gradient-boosted trees).

Both share :class:`FusionEstimator`, which does the one thing every fit has in common: fit the
block preprocessing on the rows it is given, then hand the processed blocks to a model. It
trains and nothing else -- choosing the rows, tuning hyper-parameters, rescaling labels and
calibrating a latent score all happen outside, in :mod:`tackai.fusion.training` and
:meth:`tackai.fusion.ensemble.FusionEnsemble.calibrate`. Subclasses only have to fit a model
to processed blocks and score new ones.
"""
from typing import Dict, Optional, Sequence

import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin

from tackai.fusion.blocks import BlockPreprocessor
from tackai.fusion.gp import DEFAULT_INTERACTIONS, AdditiveProductGP


class FusionEstimator(BaseEstimator, RegressorMixin):
    """Block preprocessing plus a model, trained on exactly the rows it is given.

    Subclasses implement ``_fit_model(Z, y, validation) -> model`` and
    ``_predict_model(model, Z, return_std=False)``, both in the labels' own units: the target
    is never rescaled. ``predict`` returns those units. For ``task_type="binary"`` a model that
    does not itself output probabilities (``native_binary`` False) returns a latent score, which
    only becomes a probability once ``calibrator_`` is set -- see
    :meth:`tackai.fusion.ensemble.FusionEnsemble.calibrate`.

    Args:
        task_type: ``"regression"`` or ``"binary"``.
        blocks: Block column indices (default: the standard contiguous layout).
        random_state: Seed for every stochastic component.
        dtype: Floating-point type of the processed blocks and, for a GP, of the model.
    """

    native_binary = False      # True when the model itself outputs probabilities

    def __init__(self, task_type: str = "regression",
                 blocks: Optional[Dict[str, np.ndarray]] = None, random_state: int = 0,
                 dtype: str = "float32"):
        self.task_type = task_type
        self.blocks = blocks
        self.random_state = random_state
        self.dtype = dtype

    @property
    def supports_std(self) -> bool:
        """Whether this estimator can report a predictive standard deviation."""
        return False

    def _make_preprocessor(self) -> BlockPreprocessor:
        return BlockPreprocessor(blocks=self.blocks, dtype=self.dtype)

    def fit(self, X, y, *, validation=None) -> "FusionEstimator":
        """Fit the block preprocessing and the model on the rows given.

        This trains and nothing else. Choosing the rows, scaling the labels, selecting the
        hyper-parameters and calibrating the output all happen before or after this call -- see
        :mod:`tackai.fusion.training` and :meth:`tackai.fusion.ensemble.FusionEnsemble.calibrate`.

        Args:
            X: Design matrix of the rows to train on.
            y: Labels, in the units the model should report.
            validation: Optional ``(X_val, y_val)`` for a model that early-stops on it; a model
                that does not early-stop ignores it.

        Returns:
            self
        """
        self.pre_ = self._make_preprocessor().fit(X)
        self.calibrator_ = None
        self.model_ = self._fit_model(self.pre_.transform(X), np.asarray(y, dtype=float),
                                      self._prepare_validation(validation))
        return self

    def _prepare_validation(self, validation):
        """Processed blocks and labels of an early-stopping set, or None.

        Args:
            validation: ``(X_val, y_val)`` or ``None``.

        Returns:
            ``(Z_val, y_val)`` with the blocks transformed by the fitted preprocessor, or
            ``None``.
        """
        if validation is None:
            return None
        X_val, y_val = validation
        return self.pre_.transform(X_val), np.asarray(y_val, dtype=float)

    def predict(self, X, return_std: bool = False):
        """Predict for new rows.

        Args:
            X: Design matrix.
            return_std: Also return the predictive standard deviation; only available when
                :attr:`supports_std` is True.

        Returns:
            Predictions in original units (a probability for ``task_type="binary"``), or
            ``(prediction, std)`` when ``return_std`` is set.
        """
        if return_std and not self.supports_std:
            raise NotImplementedError(
                f"{type(self).__name__} has no predictive variance; use a GP member or read the "
                "ensemble's member spread instead")
        Z = self.pre_.transform(X)
        if not return_std:
            return self.report(self._predict_model(self.model_, Z))[0]
        score, std = self._predict_model(self.model_, Z, return_std=True)
        return self.report(score, std)

    def report(self, score, std=None):
        """Map a model score to the reported quantity, with its uncertainty.

        For a regression task the score is already the reported quantity. For a binary task a
        native classifier's score is a probability and only needs clipping, while a latent
        score is pushed through the fitted calibrator -- and so is its interval, as half the
        width of ``[calibrate(score - std), calibrate(score + std)]``.

        Args:
            score: Model score for each row, on the model's own scale.
            std: Optional predictive standard deviation on that same scale.

        Returns:
            ``(value, std)`` in the reported units; ``std`` is zeros when none was given.

        Raises:
            ValueError: If this member needs a calibrator and has none.
        """
        score = np.asarray(score)
        zeros = np.zeros(len(score))
        if self.task_type != "binary":
            return score, (std if std is not None else zeros)
        if self.native_binary:
            return np.clip(score, 0.0, 1.0), (std if std is not None else zeros)
        if self.calibrator_ is None:
            raise ValueError(
                f"{type(self).__name__} is uncalibrated: a latent score is not a probability. "
                "Call FusionEnsemble.calibrate on a held-out set before predicting")
        probability = self._calibrate(score)
        if std is None:
            return probability, zeros
        high, low = self._calibrate(score + std), self._calibrate(score - std)
        return probability, np.abs(high - low) / 2.0

    def _calibrate(self, score) -> np.ndarray:
        """Calibrated probability of the positive class for a latent score."""
        return self.calibrator_.predict_proba(np.asarray(score)[:, None])[:, 1]

    # subclass hooks -----------------------------------------------------------

    def _fit_model(self, Z, y, validation):
        raise NotImplementedError

    def _predict_model(self, model, Z, return_std: bool = False):
        raise NotImplementedError


class GPInteraction(FusionEstimator):
    """M4: additive-kernel GP with cross-block product kernels.

    The kernel is a sum of per-block RBFs, a linear term on the small blocks and the product
    kernels named by ``interactions`` — the three benchmarked ones by default. ``mol*e3`` is
    also expressible now that the ligase is an embedding rather than a one-hot, but stays off
    by default so results remain comparable with the published comparison.

    Args:
        task_type: ``"regression"`` or ``"binary"``.
        blocks: Block column indices.
        random_state: Seed.
        interactions: Product kernel terms.
        ard_blocks: Blocks given one lengthscale per column. The descriptor block needs this:
            its columns are raw and span ~20 orders of magnitude (``Ipc``), which no single
            lengthscale can fit.
        n_restarts: Random restarts of the marginal-likelihood optimisation.
        n_iter: Adam steps per restart.
        lr: Adam learning rate.
        max_hyper_points: Rows used to fit the hyper-parameters; the exact GP that follows
            conditions on every training row.
        dtype: Precision of the processed blocks and of the GP (``"float32"`` or ``"float64"``).
    """

    def __init__(self, task_type: str = "regression",
                 blocks: Optional[Dict[str, np.ndarray]] = None, random_state: int = 0,
                 interactions: Sequence[str] = DEFAULT_INTERACTIONS,
                 ard_blocks: Sequence[str] = ("descriptors",), n_restarts: int = 3,
                 n_iter: int = 60, lr: float = 0.1, max_hyper_points: int = 1200,
                 dtype: str = "float32"):
        super().__init__(task_type=task_type, blocks=blocks, random_state=random_state,
                         dtype=dtype)
        self.interactions = interactions
        self.ard_blocks = ard_blocks
        self.n_restarts = n_restarts
        self.n_iter = n_iter
        self.lr = lr
        self.max_hyper_points = max_hyper_points

    @property
    def supports_std(self) -> bool:
        return True

    def _fit_model(self, Z, y, validation):
        """Fit the GP; ``validation`` is unused, as a GP has no early stopping."""
        gp = AdditiveProductGP(self.pre_.dims_, interactions=self.interactions,
                               ard_blocks=self.ard_blocks, dtype=self.dtype)
        gp.fit(Z, y, n_restarts=self.n_restarts, n_iter=self.n_iter, lr=self.lr,
               seed=self.random_state, max_hyper_points=self.max_hyper_points)
        self.kernel_report_ = gp.kernel_report()
        return gp

    def _predict_model(self, model, Z, return_std: bool = False):
        return model.predict(Z, return_std=return_std)


class XGBoostFusion(FusionEstimator):
    """M7: regularised gradient-boosted trees on the concatenated blocks.

    The molecular blocks reach the trees raw, which is the point of giving them no scaler: a
    tree splits on an individual Morgan bit or descriptor, and any rotation or rescaling would
    smear that signal across columns.

    The defaults are the configuration the 25-fold comparison chose (depth 5 in every fold,
    ``reg_lambda`` 20 in 43 of 75). Tuning them is a separate step outside the estimator.

    Args:
        task_type: ``"regression"`` or ``"binary"``.
        blocks: Block column indices.
        random_state: Seed.
        max_depth: Maximum tree depth.
        reg_lambda: L2 regularisation of the leaf weights.
        n_estimators: Boosting rounds, or the ceiling when early stopping on a validation set.
        learning_rate: Boosting learning rate.
        n_jobs: XGBoost threads.
        dtype: Precision the blocks are prepared in (XGBoost itself works in float32).
    """

    native_binary = True

    def __init__(self, task_type: str = "regression",
                 blocks: Optional[Dict[str, np.ndarray]] = None, random_state: int = 0,
                 max_depth: int = 5, reg_lambda: float = 20.0, n_estimators: int = 400,
                 learning_rate: float = 0.05, n_jobs: int = 1, dtype: str = "float32"):
        super().__init__(task_type=task_type, blocks=blocks, random_state=random_state,
                         dtype=dtype)
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.n_jobs = n_jobs

    def _new(self, early_stopping: bool):
        kw = dict(n_estimators=self.n_estimators, learning_rate=self.learning_rate,
                  max_depth=self.max_depth, reg_lambda=self.reg_lambda, min_child_weight=5,
                  subsample=0.8, colsample_bytree=0.5, gamma=0.1, tree_method="hist",
                  n_jobs=self.n_jobs, random_state=self.random_state, verbosity=0)
        if early_stopping:
            kw["early_stopping_rounds"] = 30
        return xgb.XGBClassifier(**kw) if self.task_type == "binary" else xgb.XGBRegressor(**kw)

    def _score(self, model, A) -> np.ndarray:
        if self.task_type == "binary":
            return model.predict_proba(A)[:, 1]
        return model.predict(A)

    def _fit_model(self, Z, y, validation):
        """Fit the trees, early-stopping on ``validation`` when one is given."""
        model = self._new(early_stopping=validation is not None)
        if validation is None:
            model.fit(self.pre_.concat(Z), y, verbose=False)
        else:
            Z_val, y_val = validation
            model.fit(self.pre_.concat(Z), y, eval_set=[(self.pre_.concat(Z_val), y_val)],
                      verbose=False)
        best = getattr(model, "best_iteration", None)
        self.n_trees_ = int(best) + 1 if best is not None else self.n_estimators
        return model

    def _predict_model(self, model, Z, return_std: bool = False):
        if return_std:
            raise NotImplementedError("XGBoostFusion has no predictive variance")
        return self._score(model, self.pre_.concat(Z))
