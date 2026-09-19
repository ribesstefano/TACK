"""
Omnibus statistical comparison of methods, backed by the ``autorank`` package.

``autorank`` runs the recommended pipeline for comparing many methods over
paired measurements (CV folds): parametric (repeated-measures ANOVA + Tukey
HSD) or non-parametric (Friedman + Nemenyi), applies multiple-comparison
correction, and yields a ranking plus a critical-difference (CD) diagram. The
parametric-vs-nonparametric route is decided by :func:`_check_parametric`
(Levene's test + a variance-ratio criterion, matching the TACK analysis
notebooks) rather than left to autorank's own Shapiro-Wilk-gated
auto-selection — pass an explicit ``force_mode`` to bypass it.

Statistical testing (:func:`find_equivalent_best_set`, :func:`build_full_report`)
is restricted to ``subset="val"``: the ``"test"`` predictions are the same
fixed held-out rows scored by every fold's model, so "fold" isn't an
independent repeated measure there. Use :func:`all_fold_metrics` for
descriptive numbers on the ``"test"`` set.

This module wraps ``autorank`` for TACK's prediction tables and resurrects the
API the notebooks expect from the (missing) ``best_equivalent_models`` module:

- :func:`build_full_report` — ranked summary across tasks × group strategies ×
  model families.
- :func:`find_equivalent_best_set` — the best method and the set statistically
  indistinguishable from it, for one slice.

Custom plots (MCS grids, parity, ROC/PR) remain in
:mod:`tackai.models_comparison`; this module only adds the omnibus ranking + CD.
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import levene as _levene_test
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

logger = logging.getLogger("tackai.evaluation.compare")

# Optimization direction per metric (higher-is-better vs lower-is-better).
DIRECTIONS: Dict[str, str] = {
    "rmse": "minimize", "mae": "minimize", "mse": "minimize",
    "r2": "maximize", "rho": "maximize",
    "roc_auc": "maximize", "pr_auc": "maximize", "mcc": "maximize",
    "accuracy": "maximize", "recall": "maximize", "prec": "maximize",
}

# Default ranking metric per task family.
DEFAULT_METRIC = {"bin": "roc_auc", "Dmax": "rmse", "DC50": "rmse"}

# All metrics computed per task (used in the full report).
ALL_METRICS: Dict[str, List[str]] = {
    "Dmax": ["rmse", "mae", "r2", "rho"],
    "DC50": ["rmse", "mae", "r2", "rho"],
    "bin": ["roc_auc", "pr_auc", "mcc", "accuracy"],
}


def _check_parametric(pivot: pd.DataFrame, alpha: float = 0.05) -> str:
    """Return ``'parametric'`` or ``'nonparametric'`` based on Levene's test.

    Parametric is the default; non-parametric is chosen only when Levene's
    p-value is significant (p < alpha) AND the maximum inter-method variance
    ratio exceeds 9 — the same dual criterion used in the TACK analysis
    notebooks. This is what :func:`find_equivalent_best_set` uses by default
    (as ``force_mode``) instead of autorank's own Shapiro-Wilk-gated
    auto-selection, to stay consistent with that established methodology.
    """
    groups = [pivot[col].dropna().values for col in pivot.columns]
    if any(len(g) < 2 for g in groups):
        return "parametric"
    _, pvalue = _levene_test(*groups)
    variances = pivot.var()
    min_var = variances.min()
    max_ratio = float(variances.max() / min_var) if min_var > 0 else float("inf")
    if pvalue < alpha and max_ratio > 9:
        logger.info(
            "Levene p=%.3g, max variance ratio=%.2f → non-parametric", pvalue, max_ratio
        )
        return "nonparametric"
    logger.info(
        "Levene p=%.3g, max variance ratio=%.2f → parametric", pvalue, max_ratio
    )
    return "parametric"


def _require_autorank():
    """Import autorank lazily with a helpful error if it is missing."""
    try:
        import autorank  # noqa: F401
        return autorank
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "The 'autorank' package is required for statistical comparison. "
            "Install with:  pip install autorank"
        ) from exc


# ---------------------------------------------------------------------------
# Per-fold metric computation (all metrics for a group)
# ---------------------------------------------------------------------------

def _all_fold_scores(grp: pd.DataFrame, task: str) -> Dict[str, float]:
    """Compute all relevant metrics for one (method, fold) group."""
    y_true = grp["target"].to_numpy()
    y_pred = grp["pred"].to_numpy()

    if task in ("Dmax", "DC50"):
        rho = spearmanr(y_true, y_pred).correlation if len(y_true) > 1 else np.nan
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
            "rho": float(rho) if rho is not None else np.nan,
        }
    else:  # bin
        prob = grp["prob"].to_numpy() if "prob" in grp.columns else y_pred
        has_both = len(np.unique(y_true)) >= 2
        roc = float(roc_auc_score(y_true, prob)) if has_both else np.nan
        pr = float(average_precision_score(y_true, prob)) if has_both else np.nan
        mcc = float(matthews_corrcoef(y_true, y_pred))
        acc = float(accuracy_score(y_true, y_pred))
        return {"roc_auc": roc, "pr_auc": pr, "mcc": mcc, "accuracy": acc}


def metric_pivot(
    df: pd.DataFrame,
    task: str,
    subset: str,
    metric: Optional[str] = None,
    model_family: Optional[str] = None,
    cv_group: Optional[str] = None,
) -> Tuple[pd.DataFrame, str]:
    """Build a methods×folds matrix of per-fold metric values.

    Args:
        df: Long predictions table from
            :func:`tackai.evaluation.loader.load_predictions`.
        task: Normalized task (``"Dmax"``/``"DC50"``/``"bin"``).
        subset: ``"val"`` or ``"test"``.
        metric: Metric key; defaults per task (RMSE / ROC-AUC).
        model_family: Optional display prefix filter (``"XGB"``/``"MLP"``).
        cv_group: Optional CV group strategy filter (``"scaffold"``/``"random"``).

    Returns:
        ``(pivot, metric)`` where ``pivot`` has one column per method and one
        row per fold, values = the primary metric.
    """
    metric = metric or DEFAULT_METRIC.get(task, "rmse")

    sub = df[(df["set"] == subset) & (df["task"] == task)].copy()
    if model_family is not None:
        sub = sub[sub["method"].str.upper().str.startswith(model_family.upper())]
    if cv_group is not None and "cv_group" in sub.columns:
        sub = sub[sub["cv_group"] == cv_group]
    if sub.empty:
        raise ValueError(
            f"No rows for task={task!r} subset={subset!r} "
            f"model_family={model_family!r} cv_group={cv_group!r}."
        )

    records: List[Dict[str, Any]] = []
    for (method, fold), grp in sub.groupby(["method", "fold"]):
        scores = _all_fold_scores(grp, task)
        records.append({"method": method, "fold": int(fold), **scores})
    scores_df = pd.DataFrame(records)

    pivot = scores_df.pivot(index="fold", columns="method", values=metric)
    incomplete = pivot.columns[pivot.isna().any()].tolist()
    if incomplete:
        logger.warning(
            "Dropping %d method(s) from the %s/%s comparison: at least one fold "
            "produced a NaN '%s' (paired tests require complete columns): %s",
            len(incomplete), task, subset, metric, incomplete,
        )
    pivot = pivot.dropna(axis=1)  # paired tests require complete columns
    return pivot, metric


# ---------------------------------------------------------------------------
# Full per-fold metrics table (all metrics, for the report CSV)
# ---------------------------------------------------------------------------

def all_fold_metrics(
    df: pd.DataFrame,
    task: str,
    subset: str,
) -> pd.DataFrame:
    """Per-(method, fold) table of ALL metrics for one task/subset."""
    sub = df[(df["set"] == subset) & (df["task"] == task)].copy()
    if sub.empty:
        return pd.DataFrame()
    records: List[Dict[str, Any]] = []
    for (method, fold), grp in sub.groupby(["method", "fold"]):
        row: Dict[str, Any] = {"method": method, "fold": int(fold)}
        cv_grp = grp["cv_group"].iloc[0] if "cv_group" in grp.columns else None
        if cv_grp is not None:
            row["cv_group"] = cv_grp
        row.update(_all_fold_scores(grp, task))
        records.append(row)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class EquivalenceResult:
    """Best method and its statistically-equivalent set for one slice."""
    task: str
    subset: str
    metric: str
    model_family: Optional[str]
    cv_group: Optional[str]
    best_method: str
    equivalent_methods: List[str]
    rank_table: pd.DataFrame
    omnibus_pvalue: float
    autorank_result: Any = field(repr=False, default=None)

    def __str__(self) -> str:
        scope = f"{self.task}/{self.subset}"
        if self.cv_group:
            scope += f"/{self.cv_group}"
        if self.model_family:
            scope += f"/{self.model_family}"
        return (
            f"[{scope}] metric={self.metric} | best={self.best_method!r} "
            f"| {len(self.equivalent_methods)} equivalent | "
            f"omnibus p={self.omnibus_pvalue:.3g}"
        )

    @property
    def per_method_table(self) -> pd.DataFrame:
        """Ranking table with an ``equivalent_to_best`` flag (notebook parity)."""
        out = self.rank_table.copy()
        out["equivalent_to_best"] = out.index.isin(self.equivalent_methods)
        return out


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------

def _equivalent_to_best(result: Any, best: str) -> List[str]:
    """Derive the set of methods not significantly different from ``best``."""
    rankdf = result.rankdf
    equivalent = {best}

    cd = getattr(result, "cd", None)
    if cd is not None and "meanrank" in rankdf.columns:
        best_rank = rankdf.loc[best, "meanrank"]
        for method in rankdf.index:
            if abs(rankdf.loc[method, "meanrank"] - best_rank) <= cd:
                equivalent.add(method)
        return list(equivalent)

    if {"ci_lower", "ci_upper"}.issubset(rankdf.columns):
        b_lo, b_hi = rankdf.loc[best, "ci_lower"], rankdf.loc[best, "ci_upper"]
        for method in rankdf.index:
            lo, hi = rankdf.loc[method, "ci_lower"], rankdf.loc[method, "ci_upper"]
            if not (hi < b_lo or lo > b_hi):
                equivalent.add(method)
    return list(equivalent)


def find_equivalent_best_set(
    df: pd.DataFrame,
    task: str,
    subset: str = "val",
    metric: Optional[str] = None,
    model_family: Optional[str] = None,
    cv_group: Optional[str] = None,
    alpha: float = 0.05,
    correction: Optional[str] = None,
    force_mode: Optional[str] = None,
) -> EquivalenceResult:
    """Run the omnibus comparison for one slice and identify the best set.

    Statistical testing is restricted to ``subset="val"``: the ``"test"``
    predictions are the *same* fixed held-out rows scored by every fold's
    model, so treating "fold" as a repeated-measures block there tests
    training-instance stability on one fixed sample, not generalization across
    data splits — not what a "which method is statistically best" claim should
    rest on. Use :func:`all_fold_metrics` to report descriptive test-set
    numbers instead.

    Args:
        df: Long predictions table.
        task: Normalized task name.
        subset: Must be ``"val"``.
        metric: Metric key (default per task).
        model_family: Optional display-prefix filter (``"XGB"``/``"MLP"``).
        cv_group: Optional CV group strategy filter (``"scaffold"``/``"random"``).
        alpha: Significance level.
        correction: Accepted for API compatibility; ignored (autorank auto-selects).
        force_mode: ``"parametric"``, ``"nonparametric"``, or ``None``. When
            ``None`` (default), :func:`_check_parametric` decides via Levene's
            test (dual criterion: p < alpha and max variance ratio > 9),
            matching the TACK analysis notebooks. Pass an explicit
            ``"parametric"``/``"nonparametric"`` to bypass that check, or let
            autorank auto-select via Shapiro-Wilk/Bartlett/Levene by calling
            ``autorank.autorank`` directly instead of this wrapper.

    Raises:
        ValueError: If ``subset != "val"``.
    """
    if subset != "val":
        raise ValueError(
            f"Statistical testing is only valid on 'val' predictions (got subset={subset!r}); "
            "the 'test' set is identical across folds, so folds aren't independent "
            "measurements there. Use all_fold_metrics() for descriptive test-set numbers."
        )
    if correction is not None:
        logger.debug("Ignoring correction=%r; autorank selects the post-hoc test.", correction)
    autorank = _require_autorank()
    pivot, metric = metric_pivot(
        df, task=task, subset=subset, metric=metric,
        model_family=model_family, cv_group=cv_group,
    )
    if pivot.shape[1] < 2:
        raise ValueError(
            f"Need >=2 methods to compare for task={task!r} subset={subset!r}; "
            f"got {list(pivot.columns)}."
        )

    if force_mode is None:
        force_mode = _check_parametric(pivot, alpha=alpha)

    order = "ascending" if DIRECTIONS.get(metric) == "minimize" else "descending"
    with redirect_stdout(io.StringIO()):
        result = autorank.autorank(
            pivot, alpha=alpha, verbose=False, order=order, force_mode=force_mode
        )

    # rankdf is not ordered best-first; meanrank=1 is always best regardless of order.
    rankdf = result.rankdf.sort_values("meanrank")
    best = rankdf["meanrank"].idxmin()
    equivalent = _equivalent_to_best(result, best)

    return EquivalenceResult(
        task=task, subset=subset, metric=metric,
        model_family=model_family, cv_group=cv_group,
        best_method=best, equivalent_methods=equivalent,
        rank_table=rankdf, omnibus_pvalue=float(result.pvalue),
        autorank_result=result,
    )


def build_full_report(
    df: pd.DataFrame,
    subsets: Sequence[str] = ("val",),
    tasks: Optional[Sequence[str]] = None,
    metric: Optional[str] = None,
    alpha: float = 0.05,
    correction: Optional[str] = None,
    force_mode: Optional[str] = None,
    return_equivalence: bool = False,
) -> "pd.DataFrame | tuple[pd.DataFrame, dict]":
    """Ranked comparison across tasks × cv_groups × model families, on ``val``.

    Groups automatically by the ``cv_group`` column (scaffold/random/butina)
    and by model family prefix (XGB/MLP/BERT), using one ranking metric per task
    while reporting all metrics.

    Args:
        df: Long predictions table (must have ``cv_group`` column).
        subsets: Which sets to statistically rank. Only ``"val"`` is valid (see
            :func:`find_equivalent_best_set`); any other entry is dropped up
            front with a warning rather than attempted and skipped per-slice.
        tasks: Tasks to evaluate (default: all present).
        metric: Ranking metric key (default per task).
        alpha: Significance level.
        correction: Accepted for legacy-notebook compatibility; ignored.
        force_mode: ``"parametric"``, ``"nonparametric"``, or ``None``.  When
            ``None`` (default) each slice auto-detects via Levene's test.

    Returns:
        One row per ``(task, subset, cv_group, model_family, method)`` with
        ``meanrank``, ``best``, ``equivalent_to_best``, ``omnibus_p``, and
        the central tendency of the ranking metric.
    """
    invalid_subsets = [s for s in subsets if s != "val"]
    if invalid_subsets:
        logger.warning(
            "Statistical ranking only supports subset='val'; dropping %s. "
            "Use all_fold_metrics() for descriptive numbers on other subsets.",
            invalid_subsets,
        )
    subsets = [s for s in subsets if s == "val"]

    tasks = list(tasks) if tasks is not None else sorted(df["task"].unique())
    cv_groups: List[Optional[str]] = (
        sorted(df["cv_group"].dropna().unique()) if "cv_group" in df.columns else [None]
    )
    # Build the family list: always compare all methods together (None), then
    # add within-family slices for any family that has >=2 distinct methods.
    method_names = list(df["method"].dropna().unique())
    detected = sorted({
        fam for fam in ("XGB", "MLP", "BERT")
        if sum(1 for m in method_names if m.upper().startswith(fam)) >= 2
    })
    families: List[Optional[str]] = [None] + detected  # None = all methods together

    rows: List[Dict[str, Any]] = []
    equiv_map: Dict[tuple, EquivalenceResult] = {}
    for subset in subsets:
        for cv_grp in cv_groups:
            for fam in families:
                for task in tasks:
                    try:
                        res = find_equivalent_best_set(
                            df, task=task, subset=subset, metric=metric,
                            model_family=fam, cv_group=cv_grp,
                            alpha=alpha, correction=correction,
                            force_mode=force_mode,
                        )
                    except (ValueError, KeyError) as exc:
                        logger.info(
                            "Skipping task=%s subset=%s cv_group=%s family=%s: %s",
                            task, subset, cv_grp, fam, exc,
                        )
                        continue

                    equiv_map[(subset, cv_grp, fam, task)] = res
                    rank = res.rank_table
                    central = next(
                        (c for c in ("median", "mean") if c in rank.columns), None
                    )
                    for method in rank.index:
                        rows.append({
                            "subset": subset,
                            "cv_group": cv_grp,
                            "model_family": fam,
                            "task": task,
                            "metric": res.metric,
                            "method": method,
                            "meanrank": rank.loc[method, "meanrank"],
                            "score": rank.loc[method, central] if central else np.nan,
                            "best": method == res.best_method,
                            "equivalent_to_best": method in res.equivalent_methods,
                            "omnibus_p": res.omnibus_pvalue,
                        })

    ranking_df = pd.DataFrame(rows)
    if return_equivalence:
        return ranking_df, equiv_map
    return ranking_df


def plot_cd_diagram(result: EquivalenceResult, ax=None):
    """Draw the critical-difference / CI diagram for a slice."""
    autorank = _require_autorank()
    return autorank.plot_stats(result.autorank_result, ax=ax)
