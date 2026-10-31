"""Development-only, grouped nested evaluation on a frozen research snapshot.

Never reads confirmation, stacking or calibration labels for model selection.
Each candidate writes row-aligned out-of-fold predictions and its complete run
configuration. Use --trials 0 for a fixed-parameter resource pilot only.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import optuna
import pandas as pd
from scipy.stats import gaussian_kde, spearmanr
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             mean_absolute_error, mean_squared_error, r2_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from tackai.fusion.data import make_splits
from tackai.fusion.research import sha256, write_once_json


def scores(y, pred, classification):
    """Point/ranking metrics, retaining undefined single-class AUC as null."""
    if classification:
        return {"ap": float(average_precision_score(y, pred)) if np.any(y == 1) else None,
                "roc_auc": float(roc_auc_score(y, pred)) if len(np.unique(y)) == 2 else None,
                "brier": float(brier_score_loss(y, pred)),
                "log_loss": float(log_loss(y, np.clip(pred, 1e-7, 1 - 1e-7), labels=[0, 1])),
                "prevalence": float(y.mean())}
    correlation = (float(spearmanr(y, pred).statistic)
                   if np.ptp(pred) > 0 and np.ptp(y) > 0 else None)
    if correlation is not None and not np.isfinite(correlation):
        correlation = None
    return {"rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "mae": float(mean_absolute_error(y, pred)), "r2": float(r2_score(y, pred)),
            "spearman": correlation}


def estimator(family, classification, params, jobs, seed):
    """Construct a learner; preprocessing is always fitted inside its train fold."""
    if family == "dummy":
        return DummyClassifier(strategy="prior") if classification else DummyRegressor()
    if family == "linear":
        model = (LogisticRegression(C=params.get("C", 1.), max_iter=2000,
                                    solver="liblinear", random_state=seed) if classification else
                 Ridge(alpha=params.get("alpha", 100.)))
        return make_pipeline(SimpleImputer(), StandardScaler(), model)
    if family == "extratrees":
        cls = ExtraTreesClassifier if classification else ExtraTreesRegressor
        options = dict(n_estimators=300, min_samples_leaf=3, max_features=.7,
                       n_jobs=jobs, random_state=seed)
        options.update(params)
        return cls(**options)
    if family == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor
        cls = CatBoostClassifier if classification else CatBoostRegressor
        options = dict(iterations=500, depth=6, learning_rate=.05, thread_count=jobs,
                       random_seed=seed, verbose=False, allow_writing_files=False)
        options.update(params)
        return cls(**options)
    cls = xgb.XGBClassifier if classification else xgb.XGBRegressor
    options = dict(n_estimators=400, max_depth=4, learning_rate=.05,
                   subsample=.85, colsample_bytree=.8, min_child_weight=5.,
                   reg_lambda=10., tree_method="hist", n_jobs=jobs, random_state=seed)
    options.update({k: v for k, v in params.items() if k != "alpha_w"})
    return cls(**options)


def fit_learner(model, family, X, y, params, classification):
    """Reproduce current recipe weighting only for the corrected B1 comparator."""
    if family != "current_xgb":
        return model.fit(X, y)
    if classification:
        model.set_params(scale_pos_weight=float((y == 0).sum() / max((y == 1).sum(), 1)))
        return model.fit(X, y)
    weights = np.ones(len(y))
    if np.std(y) > 1e-12 and len(np.unique(y)) >= 3:
        try:
            density = gaussian_kde(y)(y)
            weights = np.clip(density, 1e-12, None) ** -params.get("alpha_w", 1.)
            weights = np.minimum(weights, 10 * np.median(weights))
            weights /= weights.mean()
        except np.linalg.LinAlgError:
            pass
    return model.fit(X, y, sample_weight=weights)


def predict(model, X, classification):
    """Return aligned probabilities, including constant-class dummy predictions."""
    if not classification:
        return model.predict(X)
    if len(model.classes_) == 1:
        return np.full(len(X), float(model.classes_[0]))
    return model.predict_proba(X)[:, 1]


def suggest(trial, family, classification):
    if family == "extratrees":
        return dict(min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 16, log=True),
                    max_features=trial.suggest_float("max_features", .25, 1.),
                    max_depth=trial.suggest_categorical("max_depth", [None, 8, 16, 24]))
    if family == "catboost":
        return dict(iterations=trial.suggest_int("iterations", 200, 1000, log=True),
                    depth=trial.suggest_int("depth", 4, 8),
                    learning_rate=trial.suggest_float("learning_rate", .015, .15, log=True),
                    l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1., 30., log=True))
    if family == "current_xgb":
        params = dict(max_depth=trial.suggest_int("max_depth", 3, 10),
                      n_estimators=trial.suggest_int("n_estimators", 50, 2000, log=True),
                      learning_rate=trial.suggest_float("learning_rate", .001, .3, log=True),
                      subsample=trial.suggest_float("subsample", .6, 1.),
                      colsample_bytree=trial.suggest_float("colsample_bytree", .6, 1.),
                      min_child_weight=trial.suggest_float("min_child_weight", 1., 20., log=True),
                      reg_lambda=trial.suggest_float("reg_lambda", .001, 10., log=True))
        if not classification:
            params["alpha_w"] = trial.suggest_float("alpha_w", 0., 1.)
        return params
    if family == "xgb":
        params = dict(max_depth=trial.suggest_int("max_depth", 2, 8),
                      n_estimators=trial.suggest_int("n_estimators", 100, 1200, log=True),
                      learning_rate=trial.suggest_float("learning_rate", .015, .15, log=True),
                      min_child_weight=trial.suggest_float("min_child_weight", 2., 40., log=True),
                      reg_lambda=trial.suggest_float("reg_lambda", 1., 100., log=True),
                      reg_alpha=trial.suggest_float("reg_alpha", .001, 10., log=True),
                      subsample=trial.suggest_float("subsample", .6, 1.),
                      colsample_bytree=trial.suggest_float("colsample_bytree", .5, 1.))
        if not classification:
            params["objective"] = trial.suggest_categorical(
                "objective", ["reg:squarederror", "reg:pseudohubererror"])
        return params
    if family == "linear":
        name = "C" if classification else "alpha"
        return {name: trial.suggest_float(name, .001 if classification else .1,
                                          100. if classification else 10000., log=True)}
    return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=["dmax", "pdc50", "activity"], required=True)
    parser.add_argument("--feature", default="baseline")
    parser.add_argument("--model", choices=["dummy", "linear", "xgb", "current_xgb", "extratrees", "catboost"], default="xgb")
    parser.add_argument("--policy", choices=["compat", "exact"], default="compat")
    parser.add_argument("--eval-policy", choices=["compat", "exact"], default="exact")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--ablation", choices=["all", "molecule", "context"], default="all")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Use a fresh candidate output directory: {args.output}")
    manifest = json.loads((args.snapshot / "manifest.json").read_text())
    required = [f"{args.feature}.npy", "labels.npz", "groups.npy", "partition.npy", "source_partition.npy"]
    for filename in required:
        if sha256(args.snapshot / filename) != manifest["artifact_hashes"][filename]:
            raise ValueError(f"Frozen snapshot hash mismatch: {filename}")
    X = np.load(args.snapshot / f"{args.feature}.npy", mmap_mode="r")
    dims = manifest["features"][args.feature]["dims"]
    mol_width = dims["fingerprint"] + dims["descriptors"]
    if args.ablation == "molecule":
        X = X[:, :mol_width]
    elif args.ablation == "context":
        X = X[:, mol_width:]
    labels = np.load(args.snapshot / "labels.npz")
    y = labels[f"{args.policy}_{args.task}"]
    yeval = labels[f"{args.eval_policy}_{args.task}"]
    groups = np.load(args.snapshot / "groups.npy")
    roles = np.load(args.snapshot / "partition.npy")
    source_roles = np.load(args.snapshot / "source_partition.npy")
    # Masks are applied only after the shared group assignment.
    search = np.flatnonzero(roles == "search")
    classification = args.task == "activity"
    args.output.mkdir(parents=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(snapshot_hash=sha256(args.snapshot / "manifest.json"),
                  runner_hash=sha256(__file__),
                  versions={k: importlib.metadata.version(k) for k in ("scikit-learn", "xgboost", "optuna")})
    if args.model == "catboost":
        config["versions"]["catboost"] = importlib.metadata.version("catboost")
    write_once_json(args.output / "config.json", config)
    # Shared outer folds are independent of endpoint label availability and values.
    shared = make_splits(np.zeros(len(search)), groups[search], args.repeats, args.folds)
    splits = [(f"repeat{r}_fold{f}", search[tr], search[va])
              for r, repeat in enumerate(shared) for f, (tr, va) in enumerate(repeat)]
    splits.append(("source_holdout", np.flatnonzero(source_roles == "train"),
                   np.flatnonzero(source_roles == "eval")))
    all_predictions, metrics = [], []
    for number, (name, train, val) in enumerate(splits):
        train, val = train[np.isfinite(y[train])], val[np.isfinite(yeval[val])]
        if len(train) < 30 or len(val) < 10 or (classification and len(np.unique(y[train])) < 2):
            raise ValueError(f"Insufficient labels for {name}: {len(train)} train/{len(val)} eval")
        if set(groups[train]) & set(groups[val]):
            raise AssertionError("Group leakage")
        seed = 20261030 + number
        start = time.monotonic()
        best = {}
        if args.trials and args.model in {"xgb", "current_xgb", "linear", "extratrees", "catboost"}:
            inner = make_splits(y[train], groups[train], 1, 3, seeds=[seed])[0]
            def objective(trial):
                params = suggest(trial, args.model, classification)
                losses = []
                for it, iv in inner:
                    ti, vi = train[it], train[iv]
                    vi = vi[np.isfinite(yeval[vi])]
                    if not len(vi) or (classification and len(np.unique(y[ti])) < 2):
                        raise ValueError("Insufficient labels in grouped inner fold")
                    model = estimator(args.model, classification, params, args.jobs, seed)
                    fit_learner(model, args.model, X[ti], y[ti], params, classification)
                    pred = predict(model, X[vi], classification)
                    if args.model == "current_xgb" and args.task == "dmax":
                        rare = yeval[vi] <= np.quantile(y[ti], .1)
                        losses.append(float(np.mean([
                            np.sqrt(mean_squared_error(yeval[vi][mask], pred[mask]))
                            for mask in (rare, ~rare) if mask.sum() > 1])))
                    else:
                        losses.append(-average_precision_score(yeval[vi], pred) if classification
                                      else np.sqrt(mean_squared_error(yeval[vi], pred)))
                return float(np.mean(losses))
            study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
            study.optimize(objective, n_trials=args.trials)
            best = study.best_params
            study.trials_dataframe().to_csv(args.output / f"{name}_trials.csv", index=False)
        model = estimator(args.model, classification, best, args.jobs, seed)
        fit_learner(model, args.model, X[train], y[train], best, classification)
        pred = predict(model, X[val], classification)
        if not np.isfinite(pred).all():
            raise ValueError("Nonfinite model predictions")
        fold_scores = {"split": name, "n_train": len(train), "n_eval": len(val),
                       "seconds": time.monotonic() - start, "params": best,
                       **scores(yeval[val], pred, classification)}
        metrics.append(fold_scores)
        all_predictions.append(pd.DataFrame({"row": val, "group": groups[val],
                                            "split": name, "y": yeval[val], "prediction": pred}))
        pd.concat(all_predictions, ignore_index=True).to_csv(args.output / "predictions.csv", index=False)
        (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
        print(json.dumps(fold_scores), flush=True)
    (args.output / "COMPLETE").write_text("All planned development folds completed.\n")


if __name__ == "__main__":
    main()
