"""
Plot the test-set predictions of the fast fusion ensembles before and after calibration.

"Before" is the ensemble as cross-validation left it: 25 members, **uniform** weights, no
conformal quantile and no temperature. "After" is the same members with the stacking weights
and the calibration fitted on the held-out development split by
``scripts/calibrate_fast_models.py``. Both are scored on the whole test CSV, so the pair shows
what calibration actually bought.

    python scripts/plot_calibration_parity.py --run-tag autotpd4

One figure per task, two panels side by side:

* ``dmax`` / ``pdc50`` (regression) -- a parity plot, observed against predicted, with the
  identity line, a +/-1 sigma band and, on the calibrated panel, the conformal interval. A
  parity plot is the right form here: the question is whether predictions track observations
  one-for-one, which is a comparison against y = x rather than a magnitude ranking.
* ``activity`` (classification) -- a **reliability curve**, not a parity plot. Parity is
  undefined against a 0/1 label, and the thing calibration changes for a classifier is
  exactly the agreement between predicted probability and observed frequency, which is what
  a reliability curve shows. The diagonal plays the same role the identity line does.

Also writes ``calibration_metrics.json``: the before/after metrics per task, including the
conformal interval's empirical coverage, which is the number that says whether the
calibration holds on this test set rather than merely on the set it was fitted to.

Author: Stefano Ribes
"""
import os
import sys

# torch and xgboost each bundle their own libomp and segfault when both run multi-threaded in
# one process. Must be set before numpy/xgboost are imported.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                     # a file, not a window: this runs on a login node
import matplotlib.pyplot as plt
import numpy as np
from dotenv import find_dotenv, load_dotenv
from sklearn.metrics import (average_precision_score, brier_score_loss, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")
load_dotenv(find_dotenv(usecwd=True))

from tackai.fusion.context import ContextEncoder
from tackai.fusion.data import FusionData
from tackai.fusion.xgb_ensemble import XGBStackedEnsemble

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

TASKS = ("dmax", "pdc50", "activity")
REGRESSION_TASKS = ("dmax", "pdc50")

# Slots 1 and 2 of the reference categorical palette, in its fixed order and unaltered, so no
# re-validation is required. Identity is carried by marker shape as well as hue, so the panels
# stay readable in greyscale and under colour-vision deficiency.
SERIES = {"before": {"color": "#2a78d6", "marker": "o", "label": "before calibration"},
          "after": {"color": "#eb6834", "marker": "^", "label": "after calibration"}}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#dfdfdd"

#: Axis label per task, so the plot says what the number is rather than just "y".
AXIS_LABEL = {"dmax": "Dmax (fraction degraded)",
              "pdc50": r"pDC50  ($-\log_{10}$ DC50/M)",
              "activity": "P(active)"}


def style_axes(ax) -> None:
    """Recessive grid and spines: the data should be the only prominent thing in the frame."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)


def regression_metrics(y: np.ndarray, out: dict, nominal: float | None) -> dict:
    """Point-prediction error plus, when the ensemble is calibrated, interval coverage."""
    pred = out["mean"]
    metrics = {"r2": float(r2_score(y, pred)),
               "rmse": float(np.sqrt(mean_squared_error(y, pred))),
               "mae": float(mean_absolute_error(y, pred)),
               "mean_std": float(np.mean(out["std"]))}
    if "lower" in out and "upper" in out:
        inside = (y >= out["lower"]) & (y <= out["upper"])
        metrics["interval_coverage"] = float(inside.mean())
        metrics["mean_interval_width"] = float(np.mean(out["upper"] - out["lower"]))
        if nominal is not None:
            metrics["nominal_coverage"] = float(nominal)
    return metrics


def classification_metrics(y: np.ndarray, out: dict) -> dict:
    proba = out["proba"]
    return {"roc_auc": float(roc_auc_score(y, proba)),
            "pr_auc": float(average_precision_score(y, proba)),
            "brier": float(brier_score_loss(y, proba)),
            "accuracy": float(((proba >= 0.5).astype(int) == y).mean()),
            "mean_proba": float(proba.mean())}


def plot_parity_panel(ax, y: np.ndarray, out: dict, which: str, metrics: dict) -> None:
    """One parity panel: identity line, per-point uncertainty, then the points on top.

    Uncertainty is drawn as per-point error bars, not a band across the observed axis. A
    filled band would join points ordered by observed value and so draw a jagged ribbon that
    implies the prediction is a function of the observation -- it is a scatter, and each point
    carries its own interval.
    """
    style = SERIES[which]
    pred, std = out["mean"], out["std"]
    lo = float(min(y.min(), pred.min()))
    hi = float(max(y.max(), pred.max()))
    pad = 0.04 * (hi - lo)
    span = hi - lo

    # A sigma far wider than the target's own range is a diverged fit, not an honest interval;
    # drawing it would fill the panel and say nothing. Say so in words instead.
    sigma_usable = float(np.mean(std)) <= span
    if "lower" in out and "upper" in out:
        ax.errorbar(y, pred, yerr=[pred - out["lower"], out["upper"] - pred], fmt="none",
                    ecolor=style["color"], elinewidth=3, alpha=0.18,
                    label=f"conformal interval ({metrics['interval_coverage']:.0%} covered)",
                    zorder=1)
    if sigma_usable:
        ax.errorbar(y, pred, yerr=std, fmt="none", ecolor=style["color"], elinewidth=1.2,
                    alpha=0.45, label=r"$\pm1\sigma$", zorder=2)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle=(0, (4, 3)),
            color=INK_2, linewidth=1.5, label="identity", zorder=3)
    # >=8px markers with a thin surface-coloured ring, so overlapping points stay countable.
    ax.scatter(y, pred, s=34, marker=style["marker"], color=style["color"],
               edgecolor=SURFACE, linewidth=0.7, alpha=0.9, zorder=4)
    if not sigma_usable:
        # Top-right: the metrics box holds the top-left and the legend the bottom-right.
        ax.text(0.97, 0.97, f"$\\pm1\\sigma$ not drawn:\nmean $\\sigma$ = {np.mean(std):.3g}"
                            "\n(diverged fit)", transform=ax.transAxes, va="top",
                ha="right", fontsize=8, color=INK_2,
                bbox=dict(facecolor=SURFACE, edgecolor=GRID, boxstyle="round,pad=0.35"))

    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    note = (f"R$^2$ = {metrics['r2']:.3f}\nRMSE = {metrics['rmse']:.3f}\n"
            f"MAE = {metrics['mae']:.3f}")
    ax.text(0.03, 0.97, note, transform=ax.transAxes, va="top", ha="left",
            fontsize=9, color=INK, linespacing=1.5,
            bbox=dict(facecolor=SURFACE, edgecolor=GRID, boxstyle="round,pad=0.4"))
    ax.legend(loc="lower right", fontsize=8, frameon=True, facecolor=SURFACE,
              edgecolor=GRID, labelcolor=INK_2)


def reliability_curve(y: np.ndarray, proba: np.ndarray, n_bins: int):
    """Observed frequency against mean predicted probability, per equal-count bin.

    Equal-count rather than equal-width bins: the predictions pile up in the middle of [0, 1],
    and equal-width bins would put most of the test set in two of them and leave the rest
    carrying one point each.
    """
    edges = np.quantile(proba, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(proba, edges[1:-1])
    xs, ys, ns = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        xs.append(float(proba[m].mean()))
        ys.append(float(y[m].mean()))
        ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def plot_reliability(ax, y: np.ndarray, outs: dict, metrics: dict, n_bins: int) -> None:
    """Both ensembles on one axes: calibration is a comparison, so it wants a shared frame."""
    ax.plot([0, 1], [0, 1], linestyle=(0, (4, 3)), color=INK_2, linewidth=1.5,
            label="perfectly calibrated", zorder=3)
    for which in ("before", "after"):
        style = SERIES[which]
        xs, ys, ns = reliability_curve(y, outs[which]["proba"], n_bins)
        ax.plot(xs, ys, color=style["color"], linewidth=2, zorder=4)
        ax.scatter(xs, ys, s=np.clip(ns * 6, 40, 220), marker=style["marker"],
                   color=style["color"], edgecolor=SURFACE, linewidth=0.9, zorder=5,
                   label=f"{style['label']} (Brier {metrics[which]['brier']:.3f})")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("mean predicted P(active)", color=INK_2, fontsize=10)
    ax.set_ylabel("observed fraction active", color=INK_2, fontsize=10)
    ax.legend(loc="upper left", fontsize=8, frameon=True, facecolor=SURFACE,
              edgecolor=GRID, labelcolor=INK_2)


def load_pair(uncalibrated: Path, calibrated: Path) -> dict:
    """The same 25 members, before and after their weights and calibration were fitted."""
    for path in (uncalibrated, calibrated):
        if not path.is_dir():
            raise FileNotFoundError(f"not a directory: {path}")
    return {"before": XGBStackedEnsemble.from_pretrained(str(uncalibrated)),
            "after": XGBStackedEnsemble.from_pretrained(str(calibrated))}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parity / reliability plots of the test predictions, before vs after calibration.")
    parser.add_argument("--run-tag", default="autotpd4",
                        help="run tag naming the model directories (default: autotpd4)")
    parser.add_argument("--models-root", type=Path, default=REPO_ROOT / "models",
                        help="directory holding fast_xgb_<tag>_<task>/ and ensembles/")
    parser.add_argument("--test-file", type=Path,
                        default=REPO_ROOT / "data" / "auto_tpd_plus" / "PROTAC-test.csv",
                        help="test CSV, scored in full")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "calibration_parity",
                        help="directory for the figures and calibration_metrics.json")
    parser.add_argument("--protein-space", default="combined", choices=["per_block", "combined"],
                        help="must match what training used (default: combined)")
    parser.add_argument("--bins", type=int, default=8,
                        help="equal-count bins in the reliability curve (default: 8)")
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building the test table from %s", args.test_file)
    encoder = ContextEncoder(protein_space=args.protein_space)
    test_dt = FusionData.from_csv([args.test_file], encoder=encoder, verbose=True)

    report: dict = {"test_file": str(args.test_file), "run_tag": args.run_tag, "tasks": {}}
    for task in args.tasks:
        uncal = args.models_root / f"fast_xgb_{args.run_tag}_{task}" / task
        cal = args.models_root / "ensembles" / f"{args.run_tag}_{task}_calibrated"
        ensembles = load_pair(uncal, cal)
        _, X, y, _ = test_dt.task_rows(task=task)
        outs = {k: e.predict(X) for k, e in ensembles.items()}
        is_classification = ensembles["after"].task == "classification"
        nominal = getattr(ensembles["after"], "alpha", None)
        nominal = None if nominal is None else 1.0 - float(nominal)

        if is_classification:
            metrics = {k: classification_metrics(y, out) for k, out in outs.items()}
            fig, ax = plt.subplots(figsize=(6.2, 6.2), facecolor=SURFACE)
            style_axes(ax)
            plot_reliability(ax, y, outs, metrics, args.bins)
            ax.set_title(f"{task}: reliability on {len(y)} test rows\n"
                         f"ROC-AUC {metrics['before']['roc_auc']:.3f} -> "
                         f"{metrics['after']['roc_auc']:.3f}",
                         color=INK, fontsize=11)
        else:
            metrics = {k: regression_metrics(y, out, nominal) for k, out in outs.items()}
            fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.9), facecolor=SURFACE)
            for ax, which in zip(axes, ("before", "after")):
                style_axes(ax)
                plot_parity_panel(ax, y, outs[which], which, metrics[which])
                ax.set_title(SERIES[which]["label"], color=INK, fontsize=11)
                ax.set_xlabel(f"observed {AXIS_LABEL[task]}", color=INK_2, fontsize=10)
            axes[0].set_ylabel(f"predicted {AXIS_LABEL[task]}", color=INK_2, fontsize=10)
            fig.suptitle(f"{task}: {len(y)} test rows, "
                         f"R$^2$ {metrics['before']['r2']:+.3f} -> {metrics['after']['r2']:+.3f}",
                         color=INK, fontsize=12)

        path = args.output_dir / f"calibration_parity_{task}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight", facecolor=SURFACE)
        plt.close(fig)

        report["tasks"][task] = {"n_test_rows": int(len(y)), "figure": path.name,
                                 "members": len(ensembles["after"].models), **metrics}
        logger.info("%s: %d rows -> %s", task, len(y), path)
        for which in ("before", "after"):
            logger.info("    %-6s %s", which,
                        "  ".join(f"{k}={v:.3f}" for k, v in metrics[which].items()))

    out_json = args.output_dir / "calibration_metrics.json"
    out_json.write_text(json.dumps(report, indent=2) + "\n")
    logger.info("Wrote %s", out_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
