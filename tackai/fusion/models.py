"""The two fusion estimators: M4 (interaction GP) and M7 (gradient-boosted trees).

Both share :class:`FusionEstimator`, which owns everything that must happen *inside* a fold:
the block preprocessing is fitted on the training rows only, the target is standardised, and
a regression model used for the binary task is calibrated on scaffold-grouped out-of-fold
scores. Subclasses only have to fit a model to processed blocks and score new ones.
"""
from typing import Dict, Optional, Sequence

import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from tackai.fusion.blocks import BlockPreprocessor
from tackai.fusion.gp import DEFAULT_INTERACTIONS, AdditiveProductGP

INNER_FOLDS = 3


def inner_group_splits(groups, n_splits: int = INNER_FOLDS):
    """Scaffold-grouped inner CV splits of a training fold.

    Args:
        groups: Group id per training row.
        n_splits: Requested number of folds (reduced if there are fewer groups).

    Returns:
        List of ``(train_idx, test_idx)`` pairs.
    """
    n_splits = int(min(n_splits, len(np.unique(groups))))
    if n_splits < 2:
        idx = np.arange(len(groups))
        return [(idx, idx)]
    return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=groups))


def slice_blocks(Z: Dict[str, np.ndarray], idx) -> Dict[str, np.ndarray]:
    """Rows ``idx`` of every processed block."""
    return {b: a[idx] for b, a in Z.items()}


class FusionEstimator(BaseEstimator, RegressorMixin):
    """Fold-internal preprocessing and binary handling.

    Subclasses implement ``_fit_model(Z, ys, groups, y_raw, hyper=None) -> (model, hyper)``
    and ``_predict_model(model, Z, return_std=False)``, both in the labels' own units (the
    target is not rescaled). ``fit`` must be given the *training rows only*; ``predict``
    returns those units, or a calibrated probability of the positive class when
    ``task_type="binary"``.

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

    def __setstate__(self, state):
        """Restore an object pickled before ``dtype`` existed, which was fitted in float64."""
        state = dict(state)
        state.setdefault("dtype", "float64")
        super().__setstate__(state)

    @property
    def supports_std(self) -> bool:
        """Whether this estimator can report a predictive standard deviation."""
        return False

    def _make_preprocessor(self) -> BlockPreprocessor:
        return BlockPreprocessor(blocks=self.blocks, dtype=self.dtype)

    def fit(self, X, y, groups=None) -> "FusionEstimator":
        """Fit on the training rows of one fold.

        Args:
            X: Design matrix of the training rows.
            y: Targets in original units.
            groups: Scaffold group ids, used for inner splits and early stopping.

        Returns:
            self
        """
        y = np.asarray(y, dtype=float)
        groups = np.arange(len(y)) if groups is None else np.asarray(groups)
        self.pre_ = self._make_preprocessor().fit(X)
        Z = self.pre_.transform(X)
        self.model_, self.hyper_ = self._fit_model(Z, y, groups, y)
        if self.task_type == "binary" and not self.native_binary:
            self._fit_calibrator(Z, y, groups, y)
        self._after_fit()
        return self

    def _fit_calibrator(self, Z, ys, groups, y) -> None:
        """Calibrate scores to probabilities on scaffold-grouped out-of-fold predictions.

        A fold whose labels are all one class has nothing to calibrate -- and sklearn raises
        rather than saying so -- which would otherwise kill an ensemble fit after minutes of
        work. Such a fold predicts its single class outright.
        """
        classes = np.unique(y.astype(int))
        if len(classes) < 2:
            self.calibrator_ = None
            self.single_class_ = float(classes[0])
            return
        self.single_class_ = None
        oof = self._oof_scores(Z, ys, groups)
        self.calibrator_ = LogisticRegression(C=1e4).fit(oof[:, None], y.astype(int))

    def _after_fit(self) -> None:
        """Hook for subclasses to record state that inner calibration fits must not clobber."""

    def _oof_scores(self, Z, ys, groups) -> np.ndarray:
        """Scaffold-grouped out-of-fold scores with frozen hyper-parameters, for calibration."""
        oof = np.zeros(len(ys))
        for train, test in inner_group_splits(groups):
            model, _ = self._fit_model(slice_blocks(Z, train), ys[train], groups[train], None,
                                       hyper=self.hyper_)
            oof[test] = self._predict_model(model, slice_blocks(Z, test))
        return oof

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
        if getattr(self, "single_class_", None) is not None:
            return np.full(len(score), self.single_class_), zeros
        if getattr(self, "calibrator_", None) is None:
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

    def _fit_model(self, Z, ys, groups, y_raw, hyper=None):
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

    def _fit_model(self, Z, ys, groups, y_raw, hyper=None):
        gp = AdditiveProductGP(self.pre_.dims_, interactions=self.interactions,
                               ard_blocks=self.ard_blocks, dtype=self.dtype)
        state = gp.fit(Z, ys, n_restarts=self.n_restarts, n_iter=self.n_iter, lr=self.lr,
                       seed=self.random_state, max_hyper_points=self.max_hyper_points,
                       state=hyper)
        return gp, state

    def _after_fit(self) -> None:
        """Record the kernel of the model being kept, after any inner calibration fits."""
        self.kernel_report_ = self.model_.kernel_report()

    def _predict_model(self, model, Z, return_std: bool = False):
        return model.predict(Z, return_std=return_std)


class XGBoostFusion(FusionEstimator):
    """M7: regularised gradient-boosted trees on the concatenated blocks.

    The molecular blocks reach the trees raw, which is the point of giving them no scaler: a
    tree splits on an individual Morgan bit or descriptor, and any rotation or rescaling would
    smear that signal across columns.

    Args:
        task_type: ``"regression"`` or ``"binary"``.
        blocks: Block column indices.
        random_state: Seed.
        grid: Hyper-parameter grid (default: :attr:`GRID`); a single entry skips the search.
        n_estimators: Boosting rounds before early stopping.
        learning_rate: Boosting learning rate.
        n_jobs: XGBoost threads.
        dtype: Precision the blocks are prepared in (XGBoost itself works in float32).
    """

    native_binary = True
    GRID = [{"max_depth": d, "reg_lambda": lam} for d in (3, 5) for lam in (5.0, 20.0)]

    def __init__(self, task_type: str = "regression",
                 blocks: Optional[Dict[str, np.ndarray]] = None, random_state: int = 0,
                 grid: Optional[Sequence[dict]] = None, n_estimators: int = 400,
                 learning_rate: float = 0.05, n_jobs: int = 1, dtype: str = "float32"):
        super().__init__(task_type=task_type, blocks=blocks, random_state=random_state,
                         dtype=dtype)
        self.grid = grid
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.n_jobs = n_jobs

    def _new(self, cfg: dict):
        kw = dict(n_estimators=self.n_estimators, learning_rate=self.learning_rate,
                  max_depth=cfg["max_depth"], reg_lambda=cfg["reg_lambda"], min_child_weight=5,
                  subsample=0.8, colsample_bytree=0.5, gamma=0.1, tree_method="hist",
                  n_jobs=self.n_jobs, random_state=self.random_state,
                  early_stopping_rounds=30, verbosity=0)
        return xgb.XGBClassifier(**kw) if self.task_type == "binary" else xgb.XGBRegressor(**kw)

    def _fit_cfg(self, A, target, groups, cfg):
        """Fit one configuration, early-stopping on a scaffold-grouped 20% split."""
        n_groups = len(np.unique(groups))
        if n_groups < 2:
            model = self._new(cfg)
            model.fit(A, target, eval_set=[(A, target)], verbose=False)
            return model
        split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=self.random_state)
        train, val = next(split.split(A, groups=groups))
        model = self._new(cfg)
        model.fit(A[train], target[train], eval_set=[(A[val], target[val])], verbose=False)
        return model

    def _score(self, model, A) -> np.ndarray:
        if self.task_type == "binary":
            return model.predict_proba(A)[:, 1]
        return model.predict(A)

    def _fit_model(self, Z, ys, groups, y_raw, hyper=None):
        A = self.pre_.concat(Z)
        target = np.asarray(ys, float)
        grid = list(self.grid) if self.grid else list(self.GRID)
        if hyper is None:
            if len(grid) > 1:
                scores = []
                for cfg in grid:
                    fold_scores = [
                        np.mean((self._score(self._fit_cfg(A[tr], target[tr], groups[tr], cfg),
                                             A[te]) - target[te]) ** 2)
                        for tr, te in inner_group_splits(groups)]
                    scores.append(np.mean(fold_scores))
                hyper = grid[int(np.argmin(scores))]
            else:
                hyper = grid[0]
        model = self._fit_cfg(A, target, groups, hyper)
        best = getattr(model, "best_iteration", None)
        self.n_trees_ = int(best) + 1 if best is not None else self.n_estimators
        return model, hyper

    def _predict_model(self, model, Z, return_std: bool = False):
        if return_std:
            raise NotImplementedError("XGBoostFusion has no predictive variance")
        return self._score(model, self.pre_.concat(Z))
