"""
Fit and calibrate a directory of fusion-surrogate XGBoost models into an
``XGBStackedEnsemble``, on a held-out test set, then save the result.

A directory produced by ``scripts/train_fast_models.py`` (``<task>/*.json`` model files)
holds models with uniform ensemble weights and no calibration. This script loads them,
splits the given test CSV into disjoint fit / calibration / evaluation subsets, learns the
stacking weights (and regression sigmas) on the fit subset, conformal/temperature-calibrates
on the calibration subset, reports metrics on the evaluation subset, and saves the resulting
calibrated ``XGBStackedEnsemble``:

    python scripts/calibrate_fast_models.py \\
        --model-dir outputs/fast_models/dmax --task dmax \\
        --test-file data/yaochen/test.csv \\
        --output-dir ensembles/dmax --plots-dir plots/dmax \\
        --repo-id ailab-bio/TACK-fast-dmax --private

``--output-dir`` alone (no ``--repo-id``) saves the calibrated ensemble locally via
``XGBStackedEnsemble.save_pretrained`` without touching the network.

Author: Stefano Ribes
"""
import os
import sys

# macOS: torch and xgboost each bundle their own libomp; running both multi-threaded in one
# process segfaults regardless of import order. Must be set before numpy/xgboost are imported.
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dotenv import find_dotenv, load_dotenv
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from tackai.fusion.context import DEFAULT_CONTEXT_REPO
from tackai.fusion.data import FusionData
from tackai.fusion.xgb_ensemble import XGBStackedEnsemble

# Load HF_TOKEN / TACKAI_CACHE from the repo's .env before any tackai/huggingface_hub call.
# Anchored on this file rather than the cwd so it also works from a job submitted elsewhere;
# variables already in the environment take precedence.
REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")
load_dotenv(find_dotenv(usecwd=True))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

TASKS = ("dmax", "pdc50", "activity")


def split_fit_cal_eval(X: np.ndarray, y: np.ndarray, fit_frac: float, cal_frac: float,
                       seed: int, stratify: bool):
    """Three disjoint subsets of (X, y): fit, calibration, and the rest for evaluation."""
    strat = y if stratify else None
    X_pool, X_eval, y_pool, y_eval = train_test_split(
        X, y, test_size=1.0 - fit_frac - cal_frac, random_state=seed, stratify=strat)
    strat_pool = y_pool if stratify else None
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_pool, y_pool, test_size=cal_frac / (fit_frac + cal_frac),
        random_state=seed, stratify=strat_pool)
    return (X_fit, y_fit), (X_cal, y_cal), (X_eval, y_eval)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(y_true: np.ndarray, proba: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "accuracy": float((pred == y_true).mean()),
    }


def verify_pushed_ensemble(repo_id: str, X_eval: np.ndarray, y_eval: np.ndarray,
                           reference_out: dict, is_classification: bool, metrics: dict, task: str) -> dict:
    """Round-trip check: reload the just-pushed repo and confirm it reproduces predictions."""
    reloaded = XGBStackedEnsemble.from_pretrained(repo_id, subfolder=task, force_download=True)
    out = reloaded.predict(X_eval)
    pred_keys = ("proba",) if is_classification else ("mean", "std")
    mismatched = [k for k in pred_keys if not np.allclose(out[k], reference_out[k])]
    if mismatched:
        raise ValueError(f"Reloaded ensemble from {repo_id} does not reproduce predictions "
                         f"for {mismatched} — the push did not round-trip correctly.")
    if is_classification:
        pred = (out["proba"] >= 0.5).astype(int)
        reloaded_metrics = classification_metrics(y_eval, out["proba"], pred)
    else:
        reloaded_metrics = regression_metrics(y_eval, out["mean"])
    mismatched_metrics = {k: (metrics[k], reloaded_metrics[k]) for k in metrics
                          if not np.isclose(metrics[k], reloaded_metrics[k])}
    if mismatched_metrics:
        raise ValueError(f"Reloaded ensemble from {repo_id} gives different metrics: "
                         f"{mismatched_metrics}")
    logger.info("Verified %s: reloaded ensemble reproduces predictions and metrics exactly.",
               repo_id)
    return reloaded_metrics


def plot_parity(y_true: np.ndarray, out: dict, task: str, metrics: dict, save_path: Path) -> None:
    """Parity plot (true vs. predicted) with std-dev and conformal-interval bands."""
    yp, std = out["mean"], out["std"]
    order = np.argsort(y_true)
    y_sorted, yp_sorted, std_sorted = y_true[order], yp[order], std[order]

    fig, ax = plt.subplots(figsize=(6, 6))
    if "lower" in out and "upper" in out:
        ax.fill_between(y_sorted, out["lower"][order], out["upper"][order],
                        color="lightblue", alpha=0.5, label="conformal interval")
    ax.fill_between(y_sorted, yp_sorted - std_sorted, yp_sorted + std_sorted,
                    color="lightgreen", alpha=0.5, label="±1 std dev")
    ax.scatter(y_true, yp, s=20, alpha=0.8, edgecolor="none")
    lo, hi = float(y_true.min()), float(y_true.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5)
    ax.legend()
    ax.set_title(f"Parity plot for {task}\nRMSE={metrics['rmse']:.3f}, R2={metrics['r2']:.3f}")
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.grid(alpha=0.5)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(y_true: np.ndarray, pred: np.ndarray, task: str, metrics: dict,
                          save_path: Path) -> None:
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(cm, display_labels=["inactive", "active"]).plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion matrix for {task}\n"
                f"ROC-AUC={metrics['roc_auc']:.3f}, PR-AUC={metrics['pr_auc']:.3f}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit and calibrate a directory of fusion XGBoost models on a test set.",
    )
    parser.add_argument("--model-dir", required=True,
                        help="Local directory of *.json/*.ubj XGBoost models (or a Hub "
                             "repo id) to load as an XGBStackedEnsemble.")
    parser.add_argument("--task", required=True, choices=list(TASKS),
                        help="Task whose target to pull from --test-file.")
    parser.add_argument("--test-file", required=True,
                        help="Held-out test CSV (AutoTPD+/TACKv2 layout) to split into "
                             "fit/calibration/evaluation subsets.")
    parser.add_argument("--output-dir", default=None,
                        help="Save the calibrated ensemble locally (XGBStackedEnsemble."
                             "save_pretrained) to this directory.")
    parser.add_argument("--repo-id", default=None,
                        help="Also push the calibrated ensemble to this Hugging Face Hub repo id.")
    parser.add_argument("--private", type=lambda s: s.lower() not in {"false", "0", "no"},
                        default=True, help="Create the Hub repo as private (default: True).")
    parser.add_argument("--plots-dir", default=None,
                        help="Save a parity plot (regression tasks) or confusion matrix "
                             "(activity) to this directory.")
    parser.add_argument("--context-repo-id", default=DEFAULT_CONTEXT_REPO,
                        help="Hub repo id (or local staged directory) of the context "
                             "embedding tables used to encode --test-file.")
    parser.add_argument("--fit-frac", type=float, default=0.1,
                        help="Fraction of the test set used to fit the stacking weights.")
    parser.add_argument("--cal-frac", type=float, default=0.1,
                        help="Fraction of the test set used for calibration.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the fit/cal/eval split.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.fit_frac <= 0 or args.cal_frac <= 0 or args.fit_frac + args.cal_frac >= 1:
        raise ValueError("--fit-frac and --cal-frac must each be positive and sum to < 1.")
    if not os.environ.get("HF_TOKEN"):
        logger.warning("HF_TOKEN is not set (looked in %s and the cwd); pushing to a private "
                       "Hub repo will fail.", REPO_ROOT / ".env")

    logger.info("Loading context tables from %s", args.context_repo_id)
    context_dt = FusionData.from_pretrained(args.context_repo_id)

    logger.info("Building the test table from %s", args.test_file)
    test_dt = FusionData.from_csv([args.test_file], encoder=context_dt.encoder, verbose=True)
    _, X, y, _ = test_dt.task_rows(task=args.task)
    logger.info("%d rows with a defined %s target", len(y), args.task)

    logger.info("Loading ensemble from %s", args.model_dir)
    ensemble = XGBStackedEnsemble.from_pretrained(args.model_dir)
    is_classification = ensemble.task == "classification"
    expected = "classification" if args.task == "activity" else "regression"
    if ensemble.task != expected:
        raise ValueError(f"--task {args.task} expects {expected} models, but {args.model_dir} "
                         f"holds {ensemble.task} models.")

    (X_fit, y_fit), (X_cal, y_cal), (X_eval, y_eval) = split_fit_cal_eval(
        X, y, args.fit_frac, args.cal_frac, args.seed, stratify=is_classification)
    logger.info("Split: %d fit, %d calibration, %d evaluation", len(y_fit), len(y_cal), len(y_eval))

    ensemble.fit(X_fit, y_fit)
    ensemble.calibrate(X_cal, y_cal)
    out = ensemble.predict(X_eval)

    if is_classification:
        pred = (out["proba"] >= 0.5).astype(int)
        metrics = classification_metrics(y_eval, out["proba"], pred)
    else:
        metrics = regression_metrics(y_eval, out["mean"])
    logger.info("Evaluation metrics: %s", json.dumps(metrics, indent=2))

    if args.plots_dir:
        plots_dir = Path(args.plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)
        if is_classification:
            path = plots_dir / f"confusion_matrix_{args.task}.png"
            plot_confusion_matrix(y_eval, pred, args.task, metrics, path)
        else:
            path = plots_dir / f"parity_{args.task}.png"
            plot_parity(y_eval, out, args.task, metrics, path)
        logger.info("Saved plot to %s", path)

    if args.output_dir:
        logger.info("Saving calibrated ensemble to %s", args.output_dir)
        ensemble.save_pretrained(args.output_dir)
        (Path(args.output_dir) / "eval_metrics.json").write_text(json.dumps(metrics, indent=2))

    if args.repo_id:
        logger.info("Pushing calibrated ensemble to %s (private=%s)", args.repo_id, args.private)
        ensemble.push_to_hub(args.repo_id, private=args.private, subfolder=args.task)
        verify_pushed_ensemble(args.repo_id, X_eval, y_eval, out, is_classification, metrics, args.task)


if __name__ == "__main__":
    main()
