"""
Analyze the output of predict_tack_v2.py against ground truth.

Evaluates, for the Dmax/DC50/binary-activity ensembles:
  1. Regression quality (MAE/RMSE/R2/Spearman) and 95% CI coverage.
  2. Threshold-based active/inactive separation (Dmax > 80%, DC50 < 100nM).
  3. The binary-activity model against a true "active" label derived from
     the Dmax/DC50 thresholds (accuracy, ROC-AUC, PR-AUC, F1).
  4. Self-consistency: does the binary model agree with the Dmax/DC50
     *predictions* (independent of ground truth)?
  5. Data completeness: how many rows missing Dmax/DC50 got filled in by a
     prediction, and how many are still missing?
  6. Dmax-vs-DC50 correlation, true and predicted (potency consistency).

Results are logged, printed as Markdown tables, saved to a tidy
``analysis_summary.csv`` (columns: section, task, metric, value), and
plotted as PNGs -- all under --output-dir.

Usage:
    python scripts/analyze_predictions.py \\
        --predictions-csv tack_v2_with_predictions.csv \\
        --output-dir prediction_analysis
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger("analyze_predictions")
REPO_ROOT = Path(__file__).resolve().parent.parent
sns.set_style("whitegrid")

DMAX_TRUE_COL, DMAX_PRED_COL, DMAX_UNC_COL = "Dmax", "Dmax_prediction", "Dmax_uncertainty"
DC50_TRUE_COL, DC50_PRED_COL, DC50_UNC_COL = "DC50", "DC50_prediction", "DC50_uncertainty"
BINARY_PRED_COL = "Binary_prediction"

COLOR_WITHIN_CI = "#2a9d8f"
COLOR_OUTSIDE_CI = "#e76f51"
COLOR_NEUTRAL = "#457b9d"
COLOR_NO_CI = "#c9c9c9"


def _setup_logging(verbose: bool) -> None:
    """Configure root logging to stdout (mirrors predict_tack_v2.py)."""
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.addHandler(handler)


def _log_section(title: str) -> None:
    logger.info("=" * 78)
    logger.info(title)
    logger.info("=" * 78)


def kleene_active(df: pd.DataFrame, gt_col: str, lt_col: str,
                   gt_threshold: float, lt_threshold: float) -> pd.Series:
    """Combine ``gt_col > gt_threshold`` AND ``lt_col < lt_threshold`` with 3-valued logic.

    Uses pandas' nullable ``Float64``/``boolean`` dtypes so a missing value
    only makes the result unknown (``pd.NA``) when it isn't already decided
    by the other column (e.g. Dmax known and below threshold -> inactive
    regardless of a missing DC50).
    """
    gt = df[gt_col].astype("Float64") > gt_threshold
    lt = df[lt_col].astype("Float64") < lt_threshold
    return gt & lt


def _plot_confusion(cm: np.ndarray, labels: List[str], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels,
                ax=ax, cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_roc_pr(y_true: np.ndarray, y_score: np.ndarray, title: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    fpr, tpr, _ = roc_curve(y_true, y_score)
    axes[0].plot(fpr, tpr, color=COLOR_NEUTRAL)
    axes[0].plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title(f"ROC (AUC={roc_auc_score(y_true, y_score):.3f})")

    prec, rec, _ = precision_recall_curve(y_true, y_score)
    axes[1].plot(rec, prec, color=COLOR_OUTSIDE_CI)
    axes[1].axhline(float(np.mean(y_true)), linestyle="--", color="gray", linewidth=1, label="baseline")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"PR (AUC={average_precision_score(y_true, y_score):.3f})")
    axes[1].legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_true_vs_pred(df: pd.DataFrame, valid: pd.Series, true_col: str, pred_col: str,
                        ci_lo_col: str, ci_hi_col: str, task: str, threshold: float,
                        log_scale: bool, path: Path) -> None:
    sub = df.loc[valid]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 6))

    has_ci = ci_lo_col in df.columns and ci_hi_col in df.columns
    if has_ci:
        ci_ok = valid & df[ci_lo_col].notna() & df[ci_hi_col].notna()
        within = (df.loc[ci_ok, true_col] >= df.loc[ci_ok, ci_lo_col]) & \
                 (df.loc[ci_ok, true_col] <= df.loc[ci_ok, ci_hi_col])
        colors = within.map({True: COLOR_WITHIN_CI, False: COLOR_OUTSIDE_CI})
        ax.scatter(df.loc[ci_ok, true_col], df.loc[ci_ok, pred_col], c=colors, s=14, alpha=0.65)
        no_ci = valid & ~ci_ok
        if no_ci.any():
            ax.scatter(df.loc[no_ci, true_col], df.loc[no_ci, pred_col], c=COLOR_NO_CI, s=14, alpha=0.5)
        handles = [
            Line2D([0], [0], marker="o", linestyle="", color=COLOR_WITHIN_CI, label="within 95% CI"),
            Line2D([0], [0], marker="o", linestyle="", color=COLOR_OUTSIDE_CI, label="outside 95% CI"),
        ]
        ax.legend(handles=handles, loc="best", fontsize=9)
    else:
        ax.scatter(sub[true_col], sub[pred_col], s=14, alpha=0.65, color=COLOR_NEUTRAL)

    lo = float(min(sub[true_col].min(), sub[pred_col].min()))
    hi = float(max(sub[true_col].max(), sub[pred_col].max()))
    pad = (hi - lo) * 0.05 or 1.0
    lims = [max(lo - pad, 1e-6 if log_scale else lo - pad), hi + pad]
    ax.plot(lims, lims, "--", color="gray", linewidth=1)
    ax.axhline(threshold, color="black", linewidth=0.8, linestyle=":")
    ax.axvline(threshold, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlabel(f"True {task}")
    ax.set_ylabel(f"Predicted {task}")
    ax.set_title(f"{task}: true vs. predicted (n={len(sub)})")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_completeness(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    tasks = [r["task"] for r in rows]
    present = [r["pct_true_present"] * 100 for r in rows]
    filled = [r["pct_filled_by_prediction_of_total"] * 100 for r in rows]
    missing = [r["pct_still_missing_of_total"] * 100 for r in rows]

    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(len(tasks))
    ax.bar(x, present, label="true value present", color=COLOR_WITHIN_CI)
    ax.bar(x, filled, bottom=present, label="filled by prediction", color="#e9c46a")
    ax.bar(x, missing, bottom=np.array(present) + np.array(filled),
           label="still missing", color=COLOR_OUTSIDE_CI)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylabel("% of rows")
    ax.set_ylim(0, 100)
    ax.set_title("Data completeness before/after prediction")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def analyze_regression(df: pd.DataFrame, task: str, true_col: str, pred_col: str, unc_col: str,
                        ci_lo_col: str, ci_hi_col: str, threshold: float, log_scale: bool,
                        summary: List[Dict[str, Any]], plots_dir: Path) -> None:
    if pred_col not in df.columns:
        logger.warning(f"[{task}] '{pred_col}' not in predictions CSV -- skipping regression analysis.")
        return

    valid = df[true_col].notna() & df[pred_col].notna()
    n = int(valid.sum())
    if n < 2:
        logger.warning(f"[{task}] fewer than 2 rows with both true & predicted values -- skipping.")
        return

    y_true = df.loc[valid, true_col].to_numpy(dtype=float)
    y_pred = df.loc[valid, pred_col].to_numpy(dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    rho, _ = spearmanr(y_true, y_pred)

    rows = [("n_evaluated", n), ("mae", mae), ("rmse", rmse), ("r2", r2), ("spearman_r", rho)]
    logger.info(f"[{task}] n={n}  MAE={mae:.3g}  RMSE={rmse:.3g}  R2={r2:.3f}  Spearman={rho:.3f}")

    if ci_lo_col in df.columns and ci_hi_col in df.columns:
        ci_ok = valid & df[ci_lo_col].notna() & df[ci_hi_col].notna()
        n_ci = int(ci_ok.sum())
        if n_ci:
            within = (df.loc[ci_ok, true_col] >= df.loc[ci_ok, ci_lo_col]) & \
                     (df.loc[ci_ok, true_col] <= df.loc[ci_ok, ci_hi_col])
            coverage = float(within.mean())
            rows.append(("ci95_coverage", coverage))
            rows.append(("ci95_n_evaluated", n_ci))
            logger.info(f"[{task}] 95% CI coverage: {coverage:.1%} ({int(within.sum())}/{n_ci})")

    if unc_col in df.columns:
        unc_ok = valid & df[unc_col].notna()
        if int(unc_ok.sum()) > 1:
            abs_err = (df.loc[unc_ok, true_col] - df.loc[unc_ok, pred_col]).abs()
            unc_rho, _ = spearmanr(abs_err, df.loc[unc_ok, unc_col])
            rows.append(("uncertainty_vs_abs_error_spearman", unc_rho))
            logger.info(f"[{task}] uncertainty vs. |error| Spearman r = {unc_rho:.3f} "
                        "(sanity check: higher should mean less reliable)")

    for metric, value in rows:
        summary.append({"section": "regression", "task": task, "metric": metric, "value": value})

    _plot_true_vs_pred(df, valid, true_col, pred_col, ci_lo_col, ci_hi_col, task, threshold,
                        log_scale, plots_dir / f"scatter_true_vs_pred_{task.lower()}.png")


def analyze_threshold_separation(df: pd.DataFrame, task: str, true_col: str, pred_col: str,
                                  threshold: float, direction: str,
                                  summary: List[Dict[str, Any]], plots_dir: Path) -> None:
    """direction='gt': active if value > threshold (Dmax); 'lt': active if value < threshold (DC50)."""
    if pred_col not in df.columns:
        logger.warning(f"[{task}] '{pred_col}' not in predictions CSV -- skipping threshold separation.")
        return

    valid = df[true_col].notna() & df[pred_col].notna()
    n = int(valid.sum())
    if n == 0:
        logger.warning(f"[{task}] no rows with both true & predicted values -- skipping.")
        return

    if direction == "gt":
        true_active = df.loc[valid, true_col] > threshold
        pred_active = df.loc[valid, pred_col] > threshold
    else:
        true_active = df.loc[valid, true_col] < threshold
        pred_active = df.loc[valid, pred_col] < threshold

    acc = accuracy_score(true_active, pred_active)
    two_classes = true_active.nunique() > 1
    prec = precision_score(true_active, pred_active, zero_division=0) if two_classes else np.nan
    rec = recall_score(true_active, pred_active, zero_division=0) if two_classes else np.nan
    f1 = f1_score(true_active, pred_active, zero_division=0) if two_classes else np.nan
    cm = confusion_matrix(true_active, pred_active, labels=[False, True])

    logger.info(f"[{task}] threshold separation @ {threshold}: n={n}  accuracy={acc:.1%}  "
                f"true active rate={true_active.mean():.1%}  predicted active rate={pred_active.mean():.1%}")

    for metric, value in [("n_evaluated", n), ("accuracy", acc), ("precision", prec),
                          ("recall", rec), ("f1", f1), ("true_active_rate", float(true_active.mean()))]:
        summary.append({"section": "threshold_separation", "task": task, "metric": metric, "value": value})

    _plot_confusion(cm, ["inactive", "active"], f"{task} threshold separation (@ {threshold})",
                     plots_dir / f"confusion_{task.lower()}_threshold.png")


def analyze_binary_vs_active(active_label: pd.Series, df: pd.DataFrame, binary_col: str,
                              binary_threshold: float, section: str, title: str,
                              summary: List[Dict[str, Any]], plots_dir: Path) -> None:
    if binary_col not in df.columns:
        logger.warning(f"'{binary_col}' not in predictions CSV -- skipping {section}.")
        return

    valid = active_label.notna() & df[binary_col].notna()
    n = int(valid.sum())
    if n == 0 or active_label[valid].astype(bool).nunique() < 2:
        logger.warning(f"{section}: not enough data / class variety (n={n}) -- skipping.")
        return

    y_true = active_label[valid].astype(bool).to_numpy()
    y_score = df.loc[valid, binary_col].to_numpy(dtype=float)
    y_pred = y_score >= binary_threshold

    acc = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[False, True])

    logger.info(f"{section}: n={n}  accuracy={acc:.1%}  ROC-AUC={roc_auc:.3f}  "
                f"PR-AUC={pr_auc:.3f}  F1={f1:.3f}  (true active rate={y_true.mean():.1%})")

    for metric, value in [("n_evaluated", n), ("accuracy", acc), ("roc_auc", roc_auc),
                          ("pr_auc", pr_auc), ("f1", f1), ("true_active_rate", float(y_true.mean()))]:
        summary.append({"section": section, "task": "combined", "metric": metric, "value": value})

    stem = section.replace(" ", "_")
    _plot_confusion(cm, ["inactive", "active"], f"{title} (@ {binary_threshold})",
                     plots_dir / f"confusion_{stem}.png")
    _plot_roc_pr(y_true, y_score, title, plots_dir / f"roc_pr_{stem}.png")


def analyze_model_agreement(pred_active: pd.Series, df: pd.DataFrame, binary_col: str,
                             binary_threshold: float, summary: List[Dict[str, Any]],
                             plots_dir: Path) -> None:
    if binary_col not in df.columns:
        logger.warning(f"'{binary_col}' not in predictions CSV -- skipping model-agreement check.")
        return

    valid = pred_active.notna() & df[binary_col].notna()
    n = int(valid.sum())
    if n == 0:
        logger.warning("No overlapping rows for regression-vs-binary agreement check.")
        return

    a = pred_active[valid].astype(bool).to_numpy()
    b = (df.loc[valid, binary_col] >= binary_threshold).to_numpy()

    agreement = float((a == b).mean())
    kappa = cohen_kappa_score(a, b) if len(set(a)) > 1 and len(set(b)) > 1 else np.nan
    cm = confusion_matrix(a, b, labels=[False, True])

    logger.info(f"Dmax&DC50-predicted-active vs. Binary model: n={n}  agreement={agreement:.1%}  "
                f"Cohen's kappa={kappa:.3f}")

    for metric, value in [("n_evaluated", n), ("agreement_rate", agreement), ("cohen_kappa", kappa)]:
        summary.append({"section": "regression_vs_binary_agreement", "task": "combined",
                        "metric": metric, "value": value})

    _plot_confusion(cm, ["inactive", "active"], "Predicted-active (Dmax & DC50) vs. Binary model",
                     plots_dir / "confusion_regression_vs_binary_agreement.png")


def analyze_completeness(df: pd.DataFrame, task: str, true_col: str, pred_col: str,
                          summary: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    n = len(df)
    pred_present = df[pred_col].notna() if pred_col in df.columns else pd.Series(False, index=df.index)
    true_present = df[true_col].notna()

    n_true_present = int(true_present.sum())
    n_true_missing = n - n_true_present
    n_filled = int((~true_present & pred_present).sum())
    n_still_missing = int((~true_present & ~pred_present).sum())

    logger.info(
        f"[{task}] {n_true_present}/{n} ({n_true_present / n:.1%}) true values present. "
        f"Of the {n_true_missing} missing: {n_filled} "
        f"({n_filled / max(n_true_missing, 1):.1%}) filled by prediction, "
        f"{n_still_missing} ({n_still_missing / max(n_true_missing, 1):.1%}) still missing "
        f"({n_still_missing / n:.1%} of all rows)."
    )

    result = {
        "task": task,
        "n_total": n,
        "n_true_present": n_true_present,
        "pct_true_present": n_true_present / n,
        "n_true_missing": n_true_missing,
        "n_filled_by_prediction": n_filled,
        "pct_filled_by_prediction_of_missing": n_filled / max(n_true_missing, 1),
        "pct_filled_by_prediction_of_total": n_filled / n,
        "n_still_missing": n_still_missing,
        "pct_still_missing_of_missing": n_still_missing / max(n_true_missing, 1),
        "pct_still_missing_of_total": n_still_missing / n,
    }
    for metric, value in result.items():
        if metric == "task":
            continue
        summary.append({"section": "completeness", "task": task, "metric": metric, "value": value})
    return result


def analyze_dmax_dc50_consistency(df: pd.DataFrame, summary: List[Dict[str, Any]]) -> None:
    valid_true = df[DMAX_TRUE_COL].notna() & df[DC50_TRUE_COL].notna()
    if int(valid_true.sum()) > 1:
        rho, _ = spearmanr(df.loc[valid_true, DMAX_TRUE_COL], df.loc[valid_true, DC50_TRUE_COL])
        summary.append({"section": "consistency", "task": "dmax_vs_dc50_true",
                        "metric": "spearman_r", "value": rho})
        summary.append({"section": "consistency", "task": "dmax_vs_dc50_true",
                        "metric": "n_evaluated", "value": int(valid_true.sum())})
        logger.info(f"True Dmax vs. true DC50: Spearman r = {rho:.3f} (n={int(valid_true.sum())})")

    if DMAX_PRED_COL in df.columns and DC50_PRED_COL in df.columns:
        valid_pred = df[DMAX_PRED_COL].notna() & df[DC50_PRED_COL].notna()
        if int(valid_pred.sum()) > 1:
            rho, _ = spearmanr(df.loc[valid_pred, DMAX_PRED_COL], df.loc[valid_pred, DC50_PRED_COL])
            summary.append({"section": "consistency", "task": "dmax_vs_dc50_pred",
                            "metric": "spearman_r", "value": rho})
            summary.append({"section": "consistency", "task": "dmax_vs_dc50_pred",
                            "metric": "n_evaluated", "value": int(valid_pred.sum())})
            logger.info(f"Predicted Dmax vs. predicted DC50: Spearman r = {rho:.3f} "
                        f"(n={int(valid_pred.sum())})")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze predict_tack_v2.py predictions against ground truth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--predictions-csv", type=Path, default=REPO_ROOT / "tack_v2_with_predictions.csv",
                        help="CSV produced by predict_tack_v2.py.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "prediction_analysis",
                        help="Directory to write analysis_summary.csv and plots to.")
    parser.add_argument("--dmax-threshold", type=float, default=80.0,
                        help="Dmax (%%) active/inactive threshold (active if Dmax > threshold).")
    parser.add_argument("--dc50-threshold", type=float, default=100.0,
                        help="DC50 (nM) active/inactive threshold (active if DC50 < threshold).")
    parser.add_argument("--binary-threshold", type=float, default=0.5,
                        help="Probability threshold for the binary-activity model.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG-level logging.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    _setup_logging(args.verbose)

    logger.info(f"Loading predictions from {args.predictions_csv}")
    df = pd.read_csv(args.predictions_csv)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    for col in (DMAX_TRUE_COL, DC50_TRUE_COL):
        if col not in df.columns:
            raise ValueError(
                f"Expected ground-truth column '{col}' not found in {args.predictions_csv}. "
                f"Available columns: {list(df.columns)}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.output_dir
    summary: List[Dict[str, Any]] = []

    _log_section("1. Regression quality + 95% CI coverage")
    analyze_regression(df, "Dmax", DMAX_TRUE_COL, DMAX_PRED_COL, DMAX_UNC_COL,
                        "Dmax_ci_lower_95", "Dmax_ci_upper_95", args.dmax_threshold,
                        log_scale=False, summary=summary, plots_dir=plots_dir)
    analyze_regression(df, "DC50", DC50_TRUE_COL, DC50_PRED_COL, DC50_UNC_COL,
                        "DC50_ci_lower_95", "DC50_ci_upper_95", args.dc50_threshold,
                        log_scale=True, summary=summary, plots_dir=plots_dir)

    _log_section(f"2. Threshold-based separation (Dmax > {args.dmax_threshold}, DC50 < {args.dc50_threshold})")
    analyze_threshold_separation(df, "Dmax", DMAX_TRUE_COL, DMAX_PRED_COL,
                                  args.dmax_threshold, "gt", summary, plots_dir)
    analyze_threshold_separation(df, "DC50", DC50_TRUE_COL, DC50_PRED_COL,
                                  args.dc50_threshold, "lt", summary, plots_dir)

    _log_section("3. Binary-activity model vs. true combined-active label (Dmax & DC50 thresholds)")
    true_active = kleene_active(df, DMAX_TRUE_COL, DC50_TRUE_COL, args.dmax_threshold, args.dc50_threshold)
    logger.info(f"True combined-active label decidable for {int(true_active.notna().sum())}/{len(df)} rows "
                "(both present, or one side already decides it regardless of the other)")
    analyze_binary_vs_active(true_active, df, BINARY_PRED_COL, args.binary_threshold,
                              "true_active_vs_binary_model", "True active vs. Binary model",
                              summary, plots_dir)

    _log_section("4. Self-consistency: Binary model vs. Dmax/DC50 regression predictions")
    if DMAX_PRED_COL in df.columns and DC50_PRED_COL in df.columns:
        pred_active = kleene_active(df, DMAX_PRED_COL, DC50_PRED_COL, args.dmax_threshold, args.dc50_threshold)
        analyze_model_agreement(pred_active, df, BINARY_PRED_COL, args.binary_threshold, summary, plots_dir)
    else:
        logger.warning("Dmax_prediction and/or DC50_prediction missing -- skipping self-consistency check.")

    _log_section("5. Data completeness: how much missing Dmax/DC50 did predictions fill in?")
    completeness_rows = []
    for task, true_col, pred_col in [("Dmax", DMAX_TRUE_COL, DMAX_PRED_COL),
                                      ("DC50", DC50_TRUE_COL, DC50_PRED_COL)]:
        result = analyze_completeness(df, task, true_col, pred_col, summary)
        if result:
            completeness_rows.append(result)
    _plot_completeness(completeness_rows, plots_dir / "completeness.png")

    dmax_missing_after = df[DMAX_TRUE_COL].isna() & (
        df[DMAX_PRED_COL].isna() if DMAX_PRED_COL in df.columns else pd.Series(True, index=df.index))
    dc50_missing_after = df[DC50_TRUE_COL].isna() & (
        df[DC50_PRED_COL].isna() if DC50_PRED_COL in df.columns else pd.Series(True, index=df.index))
    both_missing = dmax_missing_after & dc50_missing_after
    either_missing = dmax_missing_after | dc50_missing_after
    logger.info(f"Combined: {int(both_missing.sum())} ({both_missing.mean():.1%}) rows still miss BOTH "
                f"Dmax and DC50; {int(either_missing.sum())} ({either_missing.mean():.1%}) still miss AT LEAST ONE.")
    for metric, value in [("n_still_missing_both", int(both_missing.sum())),
                          ("pct_still_missing_both", float(both_missing.mean())),
                          ("n_still_missing_either", int(either_missing.sum())),
                          ("pct_still_missing_either", float(either_missing.mean()))]:
        summary.append({"section": "completeness", "task": "combined", "metric": metric, "value": value})

    _log_section("6. Extra: Dmax-vs-DC50 potency consistency (true and predicted)")
    analyze_dmax_dc50_consistency(df, summary)

    summary_df = pd.DataFrame(summary)
    summary_csv = args.output_dir / "analysis_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    _log_section("Summary")
    with pd.option_context("display.max_rows", None, "display.width", 120):
        for section, group in summary_df.groupby("section", sort=False):
            print(f"\n### {section}")
            print(group[["task", "metric", "value"]].to_markdown(index=False, floatfmt=".4g"))

    logger.info(f"Saved {len(summary_df)} summary metrics to {summary_csv}")
    logger.info(f"Saved plots to {plots_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.getLogger("analyze_predictions").exception("analyze_predictions.py FAILED")
        sys.exit(1)
