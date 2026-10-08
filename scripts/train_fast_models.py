"""
Train the fusion-surrogate XGBoost models with repeated 5x5 cross-validation and
per-fold Optuna tuning, mirroring the top of notebooks/fast_ensemble.ipynb.

Trained models (and their best hyperparameters) are cached under --output-dir as
``<task>_repeat<i>_fold<j>.json`` / ``.params.json``; re-running the script skips any
fold whose pair of files already exists. Per-fold validation/test metrics are written
to ``<output-dir>/fold_metrics.csv``.

Author: Stefano Ribes
"""
import os
import sys

# macOS: torch and xgboost each bundle their own libomp; running both multi-threaded in
# one process segfaults regardless of import order. Must be set before numpy/torch/xgboost
# are imported.
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import scipy.stats as st
import xgboost as xgb
from dotenv import load_dotenv, find_dotenv
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold

from tackai.fusion.context import ContextEncoder
from tackai.fusion.data import DEFAULT_CONTEXT_REPO, FusionData

# Load HF_TOKEN / TACKAI_CACHE from the repo's .env before any tackai/huggingface_hub call.
# Anchored on this file rather than the cwd so it also works from a SLURM job submitted
# from elsewhere; variables already in the environment take precedence.
REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")
load_dotenv(find_dotenv(usecwd=True))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

TAIL = "low"  # which end of the regression target is the rare region
TAIL_Q = 0.10  # quantile defining it
TASKS = ("activity", "dmax", "pdc50")


def inverse_density_weights(y: np.ndarray, alpha: float = 1.0, max_ratio: float = 10.0) -> np.ndarray:
    """Sample weights inversely proportional to the KDE density of ``y``, mean-normalised."""
    y = np.asarray(y, dtype=float)
    if np.std(y) < 1e-12 or len(np.unique(y)) < 3:
        return np.ones_like(y)
    try:
        dens = st.gaussian_kde(y)(y)
    except np.linalg.LinAlgError:
        return np.ones_like(y)
    w = np.clip(dens, 1e-12, None) ** -alpha
    w = np.minimum(w, max_ratio * np.median(w))
    return w / w.mean()


def suggest_params(trial: "optuna.Trial", task: str) -> dict:
    p = dict(
        max_depth=trial.suggest_int("max_depth", 3, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        min_child_weight=trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    )
    if task != "activity":
        p["alpha_w"] = trial.suggest_float("alpha_w", 0.0, 1.0)
    return p


def make_model(params: dict, task: str, n_estimators: int, seed: int, scale_pos_weight=None):
    kw = {k: v for k, v in params.items() if k != "alpha_w"}
    kw.update(n_jobs=-1, random_state=seed, n_estimators=n_estimators)
    if task == "activity":
        return xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric="aucpr", **kw)
    return xgb.XGBRegressor(eval_metric="rmse", **kw)


def tail_mask(y: np.ndarray, cut: float) -> np.ndarray:
    return y <= cut if TAIL == "low" else y >= cut


def regression_scores(y_true: np.ndarray, y_pred: np.ndarray, cut) -> dict:
    rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
    sc = dict(
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=rmse(y_true, y_pred),
        r2=float(r2_score(y_true, y_pred)),
    )
    if cut is not None:
        m = tail_mask(y_true, cut)
        sc["rmse_rare"] = rmse(y_true[m], y_pred[m]) if m.sum() > 1 else np.nan
        sc["rmse_common"] = rmse(y_true[~m], y_pred[~m]) if (~m).sum() > 1 else np.nan
    return sc


def binary_scores(y_true: np.ndarray, proba: np.ndarray, pred: np.ndarray) -> dict:
    return dict(
        roc_auc=float(roc_auc_score(y_true, proba)),
        pr_auc=float(average_precision_score(y_true, proba)),
        f1=float(f1_score(y_true, pred)),
        precision=float(precision_score(y_true, pred, zero_division=0)),
        recall=float(recall_score(y_true, pred)),
    )


def objective(trial: "optuna.Trial", X: np.ndarray, y: np.ndarray, task: str, cut, seed: int, n_inner: int) -> float:
    params = suggest_params(trial, task)
    if task == "activity":
        strat = y
    elif cut is not None:
        strat = tail_mask(y, cut).astype(int)
    else:
        strat = None
    cv = StratifiedKFold(n_inner, shuffle=True, random_state=seed)
    splitter = (
        cv.split(X, strat) if strat is not None
        else KFold(n_inner, shuffle=True, random_state=seed).split(X)
    )
    scores, iters = [], []
    for tr, va in splitter:
        if task == "activity":
            spw = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
            model = make_model(params, task, 2000, seed, scale_pos_weight=spw)
            model.set_params(early_stopping_rounds=50)
            model.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
            s = -average_precision_score(y[va], model.predict_proba(X[va])[:, 1])
        else:
            w = inverse_density_weights(y[tr], alpha=params["alpha_w"])
            model = make_model(params, task, 2000, seed)
            model.set_params(early_stopping_rounds=50)
            model.fit(X[tr], y[tr], sample_weight=w, eval_set=[(X[va], y[va])], verbose=False)
            sc = regression_scores(y[va], model.predict(X[va]), cut)
            s = np.nanmean([sc["rmse_rare"], sc["rmse_common"]]) if cut is not None else sc["rmse"]
        scores.append(s)
        iters.append(model.best_iteration + 1)
    trial.set_user_attr("n_estimators", int(np.mean(iters)))
    return float(np.mean(scores))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train fusion-surrogate XGBoost models with 5x5 CV and per-fold Optuna tuning.",
    )
    parser.add_argument("--dev-files", nargs="+", required=True,
                         help="Development CSV(s) (AutoTPD+/TACKv2 layout) to build the training table from.")
    parser.add_argument("--test-file", required=True,
                         help="Held-out test CSV used for the reported test metrics.")
    parser.add_argument("--output-dir", required=True,
                         help="Directory to save trained models, their params, and fold_metrics.csv.")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS),
                         help="Tasks to train (default: all of activity, dmax, pdc50).")
    parser.add_argument("--protein-space", default="combined", choices=["per_block", "combined"],
                         help="Passed to ContextEncoder.")
    parser.add_argument("--n-repeats", type=int, default=5, help="Outer CV repeats.")
    parser.add_argument("--n-folds", type=int, default=5, help="Outer CV folds per repeat.")
    parser.add_argument("--n-trials", type=int, default=50, help="Optuna trials per fold.")
    parser.add_argument("--n-inner", type=int, default=3, help="Inner CV folds used inside the Optuna objective.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--push-context-to-hub", action="store_true",
                         help="Push the development context encoder's cached tables to the "
                              "Hugging Face Hub (the inverse of FusionData.from_pretrained) "
                              "after building dev_dt.")
    parser.add_argument("--context-repo-id", default=DEFAULT_CONTEXT_REPO,
                         help="Hub repo id to push the context encoder to (with --push-context-to-hub).")
    parser.add_argument("--private", type=lambda s: s.lower() not in {"false", "0", "no"}, default=True,
                         help="Create the context repo as private (default: True).")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.environ.get("HF_TOKEN"):
        logger.warning("HF_TOKEN is not set (looked in %s and the cwd); private Hub repos will fail.",
                       REPO_ROOT / ".env")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    ctx = ContextEncoder(protein_space=args.protein_space)
    dev_dt = FusionData.from_csv(args.dev_files, encoder=ctx, verbose=True)
    test_dt = FusionData.from_csv([args.test_file], encoder=ctx, verbose=True)

    if args.push_context_to_hub:
        logger.info("Pushing development context encoder to %s (private=%s)",
                     args.context_repo_id, args.private)
        dev_dt.push_to_hub(args.context_repo_id, private=args.private)

    records = []

    for task in args.tasks:
        task_dir = outdir / task
        task_dir.mkdir(parents=True, exist_ok=True)

        _, X, y, _ = dev_dt.task_rows(task=task)
        _, X_test, y_test, _ = test_dt.task_rows(task=task)
        # The tail/cut-based rare-vs-common RMSE split only applies to dmax (a bounded
        # fraction, where a tail quantile is meaningful); skip it for activity and pdc50.
        cut = float(np.quantile(y, TAIL_Q if TAIL == "low" else 1 - TAIL_Q)) if task == "dmax" else None

        for i, repeat in enumerate(dev_dt.splits(task=task, n_repeats=args.n_repeats, n_folds=args.n_folds)):
            for j, (train_idx, val_idx) in enumerate(repeat):
                key = f"{task}_repeat{i}_fold{j}"
                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]

                model_file = task_dir / f"{key}.json"
                params_file = task_dir / f"{key}.params.json"
                if model_file.exists() and params_file.exists():
                    logger.info("Loading existing model for %s from %s", key, model_file)
                    model = xgb.XGBClassifier() if task == "activity" else xgb.XGBRegressor()
                    model.load_model(model_file)
                    with open(params_file, "r") as f:
                        params_data = json.load(f)
                    best = params_data.get("best_params", {})
                else:
                    logger.info("Training new model for %s", key)
                    seed = args.seed + 100 * i + j
                    study = optuna.create_study(
                        direction="minimize",
                        sampler=optuna.samplers.TPESampler(seed=seed),
                    )
                    study.optimize(
                        lambda t: objective(t, X_train, y_train, task, cut, seed, args.n_inner),
                        n_trials=args.n_trials, show_progress_bar=True,
                    )
                    best = study.best_params
                    n_est = study.best_trial.user_attrs["n_estimators"]

                    if task == "activity":
                        spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
                        model = make_model(best, task, n_est, args.seed, scale_pos_weight=spw).fit(X_train, y_train)
                    else:
                        w = inverse_density_weights(y_train, alpha=best.get("alpha_w", 1.0))
                        model = make_model(best, task, n_est, args.seed).fit(X_train, y_train, sample_weight=w)

                    model.save_model(model_file)
                    params_file.write_text(json.dumps(
                        dict(best_params=best, n_estimators=n_est,
                             inner_score=study.best_value, cut=cut), indent=2))

                for split, Xe, ye in (("val", X_val, y_val), ("test", X_test, y_test)):
                    if task == "activity":
                        proba = model.predict_proba(Xe)[:, 1]
                        pred = (proba >= 0.5).astype(int)
                        sc = binary_scores(ye, proba, pred)
                    else:
                        pred = model.predict(Xe)
                        sc = regression_scores(ye, pred, cut)
                    records.append(dict(task=task, repeat=i, fold=j, split=split,
                                         n_train=len(train_idx), **sc))

    results = pd.DataFrame(records)
    results.to_csv(outdir / "fold_metrics.csv", index=False)
    logger.info("Wrote %s", outdir / "fold_metrics.csv")


if __name__ == "__main__":
    main()
