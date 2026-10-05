"""Everything around a fit that is not the fit itself.

The estimators in :mod:`tackai.fusion.models` only train: they take the rows they are given,
in the units they are given, with the hyper-parameters they were constructed with. Choosing
those rows -- which fold a member trains on, which of them early-stop it -- and refusing the
ones no model can learn from is this module's job.
"""
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def check_labels(y, task_type: str, what: str = "labels") -> None:
    """Refuse a set of labels no model can learn from.

    Four conditions make a fit meaningless rather than merely hard, and every one of them was
    absorbed silently somewhere in the old code: an empty training fold, non-finite targets, a
    target with a single distinct value, and a binary task whose labels are not 0/1. Note that
    a constant regression target and single-class binary labels are the same condition, so one
    check replaces both of the guards it supersedes.

    Args:
        y: Label values to check.
        task_type: ``"regression"`` or ``"binary"``; only the latter constrains the values.
        what: Noun phrase naming this set in the error message, e.g. ``"training labels"``.

    Raises:
        ValueError: If the labels are empty, non-finite, constant, or -- for a binary task --
            hold a value other than 0 or 1.
    """
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        raise ValueError(f"no {what} to fit on")
    n_bad = int((~np.isfinite(y)).sum())
    if n_bad:
        raise ValueError(f"{what} contain {n_bad} non-finite value(s); drop those rows first")
    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError(f"{what} are all {classes[0]:g}; there is nothing to learn. This is a "
                         "data or split problem, not something a model can absorb")
    if task_type == "binary":
        bad = sorted(set(classes) - {0.0, 1.0})
        if bad:
            raise ValueError(f"binary {what} must be 0 or 1, got {bad}")


def validation_split(groups, test_size: float = 0.2, random_state: int = 0):
    """One scaffold-grouped ``(train, validation)`` index pair for a model that early-stops.

    This is the split that used to hide inside ``XGBoostFusion._fit_cfg``, where too few groups
    made it fall back to early-stopping against the training set -- which is no early stopping
    at all, just a silent 400 rounds.

    Args:
        groups: Scaffold group id per row.
        test_size: Fraction of rows to put in the validation set.
        random_state: Seed for the split.

    Returns:
        ``(train_idx, val_idx)``, index arrays into ``groups``, sharing no group.

    Raises:
        ValueError: If there are fewer than two distinct groups to split.
    """
    groups = np.asarray(groups)
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        raise ValueError(f"an early-stopping split needs at least 2 scaffold groups, got "
                         f"{n_groups}; fit without early stopping or widen the fold")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    return next(splitter.split(np.zeros(len(groups)), groups=groups))


def fit_member(factory, X, y, train=None, *, groups=None, early_stopping: bool = False,
               random_state: int = 0):
    """Fit one ensemble member on the rows ``train``.

    This is the whole per-member recipe, and it is deliberately short: check the labels, build
    the estimator, optionally carve an early-stopping set out of the training rows, fit. An
    ensemble-level loop over folds calls this once per member.

    Args:
        factory: Callable returning a fresh estimator, called with ``random_state`` alone. Bake
            ``task_type`` and every other setting into the factory, e.g.
            ``partial(GPInteraction, task_type="binary")``.
        X: Full design matrix.
        y: Full label vector, in the units the member should report.
        train: Row indices to train on; defaults to every row.
        groups: Scaffold group id per row. Required when ``early_stopping`` is set, so the
            validation rows share no scaffold with the rows the model fits.
        early_stopping: Hold out a grouped fraction of ``train`` and pass it to the estimator
            as a validation set, for a model that can early-stop on it.
        random_state: Seed, forwarded to the factory and to the validation split.

    Returns:
        The fitted estimator.

    Raises:
        ValueError: If the training labels -- or, with early stopping, the validation labels --
            cannot be learned from, or if ``early_stopping`` is set without ``groups``.
    """
    train = np.arange(len(y)) if train is None else np.asarray(train)
    estimator = factory(random_state=random_state)
    check_labels(y[train], estimator.task_type, "training labels")
    if not early_stopping:
        return estimator.fit(X[train], y[train])
    if groups is None:
        raise ValueError("early_stopping needs groups, so the validation rows share no "
                         "scaffold with the rows the model fits")
    inner, val = validation_split(np.asarray(groups)[train], random_state=random_state)
    check_labels(y[train[val]], estimator.task_type, "validation labels")
    return estimator.fit(X[train[inner]], y[train[inner]],
                         validation=(X[train[val]], y[train[val]]))
