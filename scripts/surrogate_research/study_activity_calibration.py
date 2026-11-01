"""Group-held-out activity calibration study on stacking labels only.

Compare fixed methods; the saved artifact remains uncalibrated and no method is
selected. Bootstrap intervals are exploratory because folds share fitted models.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.model_selection import GroupKFold

from tackai.fusion.research import read_npz_rows, sha256

try:
    from .study_conformal_coverage import _validate_bundle
except ImportError:
    from study_conformal_coverage import _validate_bundle

EPS = 1e-7
PLATT_L2 = .01
METHODS = ("raw", "temperature", "platt")


def _fit_calibrators(predictions, y):
    """Fit prescribed transformations on the provided training rows only."""
    probabilities = np.asarray(predictions, dtype=float)
    y = np.asarray(y, dtype=float)
    if (probabilities.ndim != 2 or len(probabilities) != len(y)
            or not len(y) or not np.isfinite(probabilities).all()
            or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2
            or np.any((probabilities < 0) | (probabilities > 1))):
        raise ValueError("Calibration needs aligned probabilities and both classes")
    member_logits = logit(np.clip(probabilities, EPS, 1 - EPS))
    pooled_logits = logit(np.clip(probabilities.mean(axis=1), EPS, 1 - EPS))
    signs = 2 * y - 1

    def temp_objective(log_temperature):
        pooled = expit(member_logits / np.exp(log_temperature)).mean(axis=1)
        return float(np.mean(np.logaddexp(0, -signs * logit(np.clip(pooled, EPS, 1 - EPS)))))

    temp = minimize_scalar(temp_objective, bounds=(-3, 3), method="bounded")
    if not temp.success or not np.isfinite(temp.fun):
        raise ValueError("Temperature optimization failed")

    def platt_objective(params):
        slope, intercept = params
        return float(np.mean(np.logaddexp(0, -signs * (slope * pooled_logits + intercept)))
                     + PLATT_L2 * ((slope - 1) ** 2 + intercept ** 2))

    platt = minimize(platt_objective, x0=[1., 0.], method="L-BFGS-B",
                     bounds=[(0., 20.), (-20., 20.)])
    if not platt.success or not np.isfinite(platt.fun):
        raise ValueError("Platt optimization failed")
    return float(np.exp(temp.x)), float(platt.x[0]), float(platt.x[1])


def _apply_calibrators(predictions, fitted):
    p = np.asarray(predictions, dtype=float)
    temperature, slope, intercept = fitted
    pooled = p.mean(axis=1)
    return {"raw": pooled,
            "temperature": expit(logit(np.clip(p, EPS, 1 - EPS)) / temperature).mean(axis=1),
            "platt": expit(slope * logit(np.clip(pooled, EPS, 1 - EPS)) + intercept)}


def _metrics(y, p, weights=None):
    if weights is None:
        weights = np.ones(len(y), dtype=float)
    total = weights.sum()
    if total == 0:
        raise ValueError("Empty weighted sample")
    positive = np.dot(weights, y)
    ap = (float(average_precision_score(y, p, sample_weight=weights))
          if 0 < positive < total else None)
    return {"ap": ap, "brier": float(brier_score_loss(y, p, sample_weight=weights)),
            "log_loss": float(log_loss(y, np.clip(p, EPS, 1 - EPS),
                                       labels=[0, 1], sample_weight=weights)),
            "prevalence": float(positive / total)}


def _report_slice(y, predictions, groups, *, draws, seed):
    """Paired bootstrap samples whole chemical groups, shared by all methods."""
    unique, inverse = np.unique(groups, return_inverse=True)
    result = {"rows": int(len(y)), "groups": int(len(unique)),
              "metrics": {name: _metrics(y, predictions[name]) for name in METHODS}}
    deltas = {name: {metric: [] for metric in ("ap", "brier", "log_loss")}
              for name in METHODS if name != "raw"}
    rng = np.random.default_rng(seed)
    for _ in range(draws):
        weights = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))[inverse]
        base = _metrics(y, predictions["raw"], weights)
        for method in deltas:
            value = _metrics(y, predictions[method], weights)
            for metric in deltas[method]:
                if value[metric] is not None and base[metric] is not None:
                    deltas[method][metric].append(value[metric] - base[metric])
    result["paired_delta_vs_raw_ci95"] = {
        method: {metric: {"interval": np.quantile(values, [.025, .975]).tolist() if values else None,
                          "valid_draws": len(values)}
                 for metric, values in metrics.items()}
        for method, metrics in deltas.items()}
    return result


def study(snapshot, bundle, output, *, folds=5, seed=20261101, draws=1000):
    snapshot, bundle, output = Path(snapshot), Path(bundle), Path(output)
    if output.exists():
        raise FileExistsError(f"Use a fresh output directory: {output}")
    if folds < 2 or draws < 0:
        raise ValueError("Need at least two folds and nonnegative bootstrap draws")
    (snapshot, bundle, manifest, bundle_manifest, artifact, X, groups_all,
     roles) = _validate_bundle(snapshot, bundle, task_type="classification")
    if artifact.recipe.get("label_policy") != "exact":
        raise ValueError("Study requires an exact-label activity bundle")
    rows = np.flatnonzero(roles == "stacking")
    labels = read_npz_rows(snapshot / "labels.npz", "exact_activity", rows)
    finite = np.isfinite(labels)
    rows, y = rows[finite], labels[finite]
    groups = groups_all[rows]
    if (not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2
            or len(np.unique(groups)) < folds):
        raise ValueError("Insufficient labeled activity stacking groups or classes")
    members = artifact._predictions(X[rows])
    if not np.allclose(artifact.weights_, np.full(len(artifact.weights_), 1 / len(artifact.weights_))):
        raise ValueError("Study requires actual uniform ensemble weights")
    oof = {method: np.full(len(rows), np.nan) for method in METHODS}
    fold_reports = []
    split = GroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (train, evaluation) in enumerate(split.split(members, y, groups)):
        train_groups, test_groups = set(groups[train]), set(groups[evaluation])
        if train_groups & test_groups:
            raise ValueError("Calibration and evaluation groups overlap")
        fitted = _fit_calibrators(members[train], y[train])
        for method, values in _apply_calibrators(members[evaluation], fitted).items():
            oof[method][evaluation] = values
        fold_reports.append({"fold": fold, "train_groups": sorted(map(str, train_groups)),
                             "evaluation_groups": sorted(map(str, test_groups)),
                             "temperature": fitted[0], "platt_slope": fitted[1],
                             "platt_intercept": fitted[2]})
    if any(not np.isfinite(values).all() for values in oof.values()):
        raise ValueError("Incomplete held-out predictions")
    if sha256(snapshot / "rows.csv") != manifest["artifact_hashes"]["rows.csv"]:
        raise ValueError("Snapshot metadata hash mismatch")
    metadata = pd.read_csv(snapshot / "rows.csv", usecols=["e3_raw"]).fillna("")
    if len(metadata) != len(roles):
        raise ValueError("Snapshot metadata length mismatch")
    e3 = metadata.e3_raw.to_numpy(dtype=str)[rows]
    slices = {"all": _report_slice(y, oof, groups, draws=draws, seed=seed + 701)}
    for value in sorted(set(e3)):
        mask = e3 == value
        if mask.sum() < 30 or len(np.unique(groups[mask])) < 5:
            continue
        slices[f"e3:{value or '(missing)'}"] = _report_slice(
            y[mask], {m: oof[m][mask] for m in METHODS}, groups[mask], draws=draws, seed=seed + 701)
    output.mkdir(parents=True)
    with (output / "oof_predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["row", "group", "y", "e3", *METHODS])
        writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow({"row": int(row), "group": str(groups[i]), "y": float(y[i]),
                             "e3": e3[i], **{method: float(oof[method][i]) for method in METHODS}})
    report = {"format": "tack-activity-calibration-study/v1", "folds": folds, "seed": seed,
              "bootstrap_draws": draws, "methods": list(METHODS), "platt_l2": PLATT_L2,
              "snapshot_manifest_sha256": sha256(snapshot / "manifest.json"),
              "bundle_manifest_sha256": sha256(bundle / "manifest.json"),
              "bundle_model_sha256": bundle_manifest["model_sha256"],
              "study_code_sha256": sha256(__file__), "only_stacking_labels_read": True,
              "final_calibration_performed": False, "confirmation_labels_read": False,
              "folds_detail": fold_reports, "slices": slices,
              "note": "Exploratory held-out stacking study; intervals resample fixed OOF predictions, not retraining."}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--draws", type=int, default=1000)
    args = parser.parse_args()
    report = study(args.snapshot, args.bundle, args.output,
                   folds=args.folds, seed=args.seed, draws=args.draws)
    print(json.dumps(report["slices"]["all"]["metrics"], indent=2))


if __name__ == "__main__":
    main()
