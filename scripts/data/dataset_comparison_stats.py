"""Shared statistics for the dataset-comparison notebooks.

Two independent per-seed metric samples in, a full comparison table out: Welch's
t-test or Mann-Whitney U (selected by Shapiro-Wilk normality), effect sizes, a
bootstrap CI on the mean difference, and a compound-clustered bootstrap that
isolates test-set sampling uncertainty from model-instance uncertainty. Used by
``notebooks/dataset_comparison_simple.ipynb`` and
``notebooks/dataset_comparison_five_sets.ipynb`` so the two analyses can't drift
apart on how "different" is decided.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from tackai.evaluation.compare import DIRECTIONS


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta: P(a > b) - P(a < b), a non-parametric effect size in [-1, 1]."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.sign(a[:, None] - b[None, :]).mean())


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d: standardized mean difference using the pooled standard deviation."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else np.nan


def _magnitude(value: float, thresholds: Sequence[float], labels: Sequence[str]) -> str:
    """Bucket an effect size into a plain-language magnitude label."""
    for t, lab in zip(thresholds, labels):
        if abs(value) < t:
            return lab
    return labels[-1]


def compare_two(a: np.ndarray, b: np.ndarray, metric: str, name_a: str, name_b: str,
                alpha: float = 0.05, n_boot: int = 10_000) -> dict:
    """Compare two independent samples of per-seed metric values.

    Runs the normality check, selects Welch's t-test or Mann-Whitney U accordingly,
    and adds effect sizes, a bootstrap CI on the mean difference, and the paired
    tests as a secondary cross-check. Unpaired is the primary comparison: the two
    seed sequences share only an RNG stream, not training data, so pairing seed i
    of A with seed i of B is not a real correspondence (see the paired tests as a
    cross-check only).

    Args:
        a: Per-seed metric values for the first dataset (``name_a``).
        b: Per-seed metric values for the second dataset (``name_b``).
        metric: Metric key, used to look up the optimization direction.
        name_a: Display name of the first dataset (for ``favours``).
        name_b: Display name of the second dataset (for ``favours``).
        alpha: Significance level for the normality gate and the tests.
        n_boot: Bootstrap resamples for the confidence interval.

    Returns:
        A flat dict of every statistic, ready to become one row of a summary table.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    direction = DIRECTIONS.get(metric, "minimize")

    sw_a, sw_b = stats.shapiro(a).pvalue, stats.shapiro(b).pvalue
    both_normal = (sw_a > alpha) and (sw_b > alpha)

    if both_normal:
        test_name, res = "Welch t-test", stats.ttest_ind(a, b, equal_var=False)
    else:
        test_name, res = "Mann-Whitney U", stats.mannwhitneyu(a, b, alternative="two-sided")

    boot = stats.bootstrap(
        (a, b), lambda x, y, axis=-1: x.mean(axis=axis) - y.mean(axis=axis),
        n_resamples=n_boot, method="percentile",
        rng=np.random.default_rng(0), vectorized=True,
    )
    ci_low, ci_high = boot.confidence_interval

    try:
        wilcoxon_p = float(stats.wilcoxon(a, b).pvalue)
    except ValueError:      # all differences zero
        wilcoxon_p = np.nan

    d, delta = cohens_d(a, b), cliffs_delta(a, b)
    favours = name_a if ((a.mean() < b.mean()) == (direction == "minimize")) else name_b

    return dict(
        metric=metric, direction=direction, n_per_group=len(a),
        **{f"mean[{name_a}]": a.mean(), f"std[{name_a}]": a.std(ddof=1),
           f"mean[{name_b}]": b.mean(), f"std[{name_b}]": b.std(ddof=1)},
        both_normal=both_normal, test=test_name,
        statistic=float(res.statistic), p_value=float(res.pvalue),
        mean_diff=float(a.mean() - b.mean()), ci_low=float(ci_low), ci_high=float(ci_high),
        cohens_d=d, d_magnitude=_magnitude(d, (0.2, 0.5, 0.8), ("negligible", "small", "medium", "large")),
        cliffs_delta=delta,
        delta_magnitude=_magnitude(delta, (0.147, 0.33, 0.474), ("negligible", "small", "medium", "large")),
        paired_t_p=float(stats.ttest_rel(a, b).pvalue), wilcoxon_p=wilcoxon_p,
        favours=favours,
    )


def metric_vectors(metrics_df: pd.DataFrame, task: str, metric: str,
                   name_a: str, name_b: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return the two per-seed metric vectors (name_a, name_b) for one task, seed-aligned.

    Args:
        metrics_df: One task's ``all_fold_metrics`` output (a ``method``/``fold`` table).
        task: Task key, only used for error messages.
        metric: Metric column to pull.
        name_a: ``method`` value for the first sample.
        name_b: ``method`` value for the second sample.
    """
    m = metrics_df.sort_values("fold")
    return (m[m["method"] == name_a][metric].to_numpy(),
            m[m["method"] == name_b][metric].to_numpy())


def metric_from_arrays(y_true: np.ndarray, y_pred: np.ndarray, task: str, metric: str) -> float:
    """Compute one metric from raw target/prediction arrays."""
    from sklearn.metrics import (roc_auc_score, average_precision_score, matthews_corrcoef,
                                 accuracy_score, mean_absolute_error, mean_squared_error, r2_score)
    if task == "bin":
        if len(np.unique(y_true)) < 2:
            return np.nan
        if metric == "roc_auc":  return roc_auc_score(y_true, y_pred)
        if metric == "pr_auc":   return average_precision_score(y_true, y_pred)
        if metric == "mcc":      return matthews_corrcoef(y_true, (y_pred >= 0.5).astype(int))
        if metric == "accuracy": return accuracy_score(y_true, (y_pred >= 0.5).astype(int))
    else:
        if metric == "rmse": return float(np.sqrt(mean_squared_error(y_true, y_pred)))
        if metric == "mae":  return float(mean_absolute_error(y_true, y_pred))
        if metric == "r2":   return float(r2_score(y_true, y_pred))
        if metric == "rho":  return float(stats.spearmanr(y_true, y_pred).correlation)
    return np.nan


def seed_averaged_test_frame(results_df: pd.DataFrame, task: str,
                             names: Iterable[str]) -> pd.DataFrame:
    """One row per test row: compound, target, and each dataset's seed-averaged prediction.

    Args:
        results_df: The concatenated per-row prediction table (``task``/``method``/
            ``fold``/``set``/``compound``/``cell_seen``/``target``/``pred``/``prob``).
        task: Task key to filter on.
        names: ``method`` values to include as columns, in order.
    """
    names = list(names)
    sub = results_df[(results_df["task"] == task) & (results_df["set"] == "test")]
    col = "prob" if task == "bin" else "pred"
    frames = []
    for name in names:
        s = sub[sub["method"] == name].copy()
        s["row_pos"] = s.groupby("fold").cumcount()
        agg = s.groupby("row_pos").agg(compound=("compound", "first"),
                                       target=("target", "first"),
                                       cell_seen=("cell_seen", "first"),
                                       **{name: (col, "mean")})
        frames.append(agg)
    out = frames[0]
    for extra in frames[1:]:
        out = out.join(extra[[extra.columns[-1]]])
    return out.reset_index(drop=True)


def clustered_bootstrap(results_df: pd.DataFrame, task: str, metric: str,
                        names: Sequence[str], diff_pair: Optional[Tuple[str, str]] = None,
                        n_boot: int = 5000, seed: int = 0) -> dict:
    """Bootstrap a metric and a between-dataset difference, resampling whole compounds.

    Clustering by compound is essential: rows sharing a compound are not independent,
    so a naive row-level bootstrap would understate the interval width.

    Args:
        results_df: The concatenated per-row prediction table.
        task: Task key.
        metric: Metric to compute.
        names: ``method`` values to include (their seed-averaged predictions are
            all computed on the same resampled rows).
        diff_pair: ``(name_a, name_b)`` to report a ``diff`` CI for; defaults to
            ``(names[0], names[1])``.
        n_boot: Number of bootstrap resamples.
        seed: RNG seed for reproducibility.

    Returns:
        Point estimates and 95% percentile CIs for each dataset and their difference.
    """
    names = list(names)
    name_a, name_b = diff_pair if diff_pair is not None else (names[0], names[1])
    df = seed_averaged_test_frame(results_df, task, names)
    groups = df["compound"].to_numpy()
    uniq = pd.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)

    point = {n: metric_from_arrays(df["target"].to_numpy(), df[n].to_numpy(), task, metric)
             for n in names}
    draws: Dict[str, list] = {n: [] for n in names}
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in pick])
        yt = df["target"].to_numpy()[idx]
        vals = {n: metric_from_arrays(yt, df[n].to_numpy()[idx], task, metric) for n in names}
        if any(np.isnan(v) for v in vals.values()):
            continue
        for n in names:
            draws[n].append(vals[n])
        diffs.append(vals[name_a] - vals[name_b])

    out = dict(task=task, metric=metric, n_compounds=len(uniq), n_rows=len(df))
    for n in names:
        lo, hi = np.percentile(draws[n], [2.5, 97.5])
        out[n] = point[n]; out[f"{n}_lo"] = lo; out[f"{n}_hi"] = hi
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    out.update(diff=point[name_a] - point[name_b], diff_lo=lo, diff_hi=hi,
               diff_excludes_zero=not (lo <= 0 <= hi))
    return out
