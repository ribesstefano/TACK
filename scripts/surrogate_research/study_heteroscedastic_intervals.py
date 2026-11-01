"""Development-only test of feature-conditional pDC50 interval scales.

This experiment compares the existing global RMS-residual plus member-
disagreement scale with a fixed ExtraTrees scale model. Both use the same
frozen search-only uniform predictions, nested group splits, and group-maximum
conformal quantile. It does not fit final calibration or select a method.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from tackai.fusion.research import read_npz_rows, sha256

try:
    from .study_conformal_coverage import (
        _finite_sample_quantile,
        _group_max_scores,
        _interval_metrics,
        _validate_bundle,
    )
except ImportError:
    from study_conformal_coverage import (
        _finite_sample_quantile,
        _group_max_scores,
        _interval_metrics,
        _validate_bundle,
    )


SCALE_MODEL_PARAMS = {
    "n_estimators": 200,
    "min_samples_leaf": 10,
    "max_features": 1.0,
    "bootstrap": False,
    "n_jobs": 1,
}
SCALE_FLOOR = 1e-4


def _bootstrap_comparison(y, mean, widths, groups, *, n_bootstrap, seed):
    """Paired chemical-group bootstrap CIs for coverage and interval width."""
    methods = ("global_group_max", "heteroscedastic_group_max")
    metrics = ("row_coverage", "group_all_rows_coverage", "mean_width")
    unique = np.unique(groups)
    by_group = {group: np.flatnonzero(groups == group) for group in unique}
    point = {}
    for method in methods:
        covered = np.abs(y - mean) <= widths[method] / 2
        point[method] = {
            "row_coverage": float(np.mean(covered)),
            "group_all_rows_coverage": float(np.mean([
                np.all(covered[groups == group]) for group in unique])),
            "mean_width": float(np.mean(widths[method])),
        }
    values = {method: {metric: [] for metric in metrics} for method in methods}
    delta = {metric: [] for metric in metrics}
    rng = np.random.default_rng(seed)
    for _ in range(n_bootstrap):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        chunks, relabeled_groups = [], []
        for copy_id, group in enumerate(sampled):
            rows = by_group[group]
            chunks.append(rows)
            relabeled_groups.append(np.full(len(rows), copy_id, dtype=np.int64))
        rows = np.concatenate(chunks)
        bootstrap_groups = np.concatenate(relabeled_groups)
        fold_scores = {}
        for method in methods:
            covered = np.abs(y[rows] - mean[rows]) <= widths[method][rows] / 2
            scores = {
                "row_coverage": float(np.mean(covered)),
                "group_all_rows_coverage": float(np.mean([
                    np.all(covered[bootstrap_groups == group])
                    for group in np.unique(bootstrap_groups)])),
                "mean_width": float(np.mean(widths[method][rows])),
            }
            fold_scores[method] = scores
            for metric in metrics:
                values[method][metric].append(scores[metric])
        for metric in metrics:
            delta[metric].append(fold_scores["heteroscedastic_group_max"][metric]
                                 - fold_scores["global_group_max"][metric])

    def ci(samples):
        return ([float(value) for value in np.quantile(samples, [.025, .975])]
                if samples else None)

    return {
        "n_bootstraps": int(n_bootstrap),
        "per_method_ci95": {
            method: {metric: ci(values[method][metric]) for metric in metrics}
            for method in methods
        },
        "paired_heteroscedastic_minus_global_ci95": {
            metric: {
                "estimate": float(point["heteroscedastic_group_max"][metric]
                                   - point["global_group_max"][metric]),
                "ci95": ci(delta[metric]), "valid_bootstraps": len(delta[metric]),
            }
            for metric in metrics
        },
    }


def study(snapshot, bundle, output, *, alpha=0.1, folds=5, seed=20261101,
          n_bootstrap=1000, scale_fraction=0.5):
    """Compare scale estimators with identical outer and quantile group splits."""
    snapshot, bundle, output = Path(snapshot), Path(bundle), Path(output)
    if output.exists():
        raise FileExistsError(f"study output already exists: {output}")
    if not 0 < alpha < 1 or folds < 2 or n_bootstrap < 0:
        raise ValueError("alpha must be in (0,1), folds >= 2, and bootstrap >= 0")
    if not 0 < scale_fraction < 1:
        raise ValueError("scale_fraction must lie strictly between zero and one")
    (snapshot, bundle, manifest, bundle_manifest, artifact, X, groups_all,
     roles) = _validate_bundle(snapshot, bundle)
    if artifact.recipe.get("task") != "pdc50":
        raise ValueError("heteroscedastic interval study requires a pDC50 bundle")

    stack_rows = np.flatnonzero(roles == "stacking")
    labels = read_npz_rows(snapshot / "labels.npz", "exact_pdc50", stack_rows)
    keep = np.isfinite(labels)
    rows, y = stack_rows[keep], labels[keep]
    groups = groups_all[rows]
    unique_groups = np.unique(groups)
    if len(unique_groups) < max(folds, int(np.ceil(1 / alpha)) + 2):
        raise ValueError("too few labeled stacking groups for requested folds and alpha")

    # Frozen model outputs are evaluated once. The scale model sees only their
    # pooled mean and between-member disagreement, never raw assay/context data.
    member_predictions = artifact._predictions(X[rows])
    mean = member_predictions.mean(axis=1)
    disagreement = np.sqrt(np.mean((member_predictions - mean[:, None]) ** 2, axis=1))
    scale_features = np.column_stack((mean, disagreement))
    methods = ("global_group_max", "heteroscedastic_group_max")
    widths = {name: np.full(len(rows), np.nan) for name in methods}
    fold_reports = []

    outer = GroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(outer.split(mean, y, groups)):
        eval_groups = set(groups[test_idx])
        nested = GroupShuffleSplit(n_splits=1, test_size=1 - scale_fraction,
                                   random_state=seed + 1009 * (fold + 1))
        scale_local, quantile_local = next(nested.split(
            np.arange(len(train_idx)), groups=groups[train_idx]))
        scale_idx, quantile_idx = train_idx[scale_local], train_idx[quantile_local]
        scale_groups, quantile_groups = set(groups[scale_idx]), set(groups[quantile_idx])
        if (scale_groups & quantile_groups or scale_groups & eval_groups
                or quantile_groups & eval_groups):
            raise AssertionError("scale, quantile, and evaluation groups must be disjoint")
        if len(quantile_groups) < int(np.ceil(1 / alpha)):
            raise ValueError(f"fold {fold} has too few quantile groups for alpha={alpha}")

        residual = np.abs(y[scale_idx] - mean[scale_idx])
        global_residual_rms = max(float(np.sqrt(np.mean(residual ** 2))), SCALE_FLOOR)
        global_scale = np.sqrt(global_residual_rms ** 2 + disagreement ** 2)

        scale_model = ExtraTreesRegressor(
            **SCALE_MODEL_PARAMS, random_state=seed + 1009 * (fold + 1))
        scale_model.fit(scale_features[scale_idx], residual)
        conditional_residual_scale = np.maximum(
            np.asarray(scale_model.predict(scale_features), dtype=float), SCALE_FLOOR)
        heteroscedastic_scale = np.sqrt(conditional_residual_scale ** 2 + disagreement ** 2)

        scores = {
            "global_group_max": np.abs(y - mean) / global_scale,
            "heteroscedastic_group_max": np.abs(y - mean) / heteroscedastic_scale,
        }
        quantiles = {
            name: _finite_sample_quantile(
                _group_max_scores(scores[name][quantile_idx], groups[quantile_idx]), alpha)
            for name in methods
        }
        scales = {"global_group_max": global_scale,
                  "heteroscedastic_group_max": heteroscedastic_scale}
        fold_report = {
            "fold": fold,
            "scale_rows": int(len(scale_idx)), "scale_groups": int(len(scale_groups)),
            "quantile_rows": int(len(quantile_idx)), "quantile_groups": int(len(quantile_groups)),
            "evaluation_rows": int(len(test_idx)), "evaluation_groups": int(len(eval_groups)),
            "global_residual_rms": global_residual_rms,
            "global_group_max_quantile": quantiles["global_group_max"],
            "heteroscedastic_group_max_quantile": quantiles["heteroscedastic_group_max"],
            "global_scale_mean_eval": float(np.mean(global_scale[test_idx])),
            "heteroscedastic_scale_mean_eval": float(np.mean(heteroscedastic_scale[test_idx])),
            "scale_group_ids": sorted(map(str, scale_groups)),
            "quantile_group_ids": sorted(map(str, quantile_groups)),
            "evaluation_group_ids": sorted(map(str, eval_groups)),
        }
        for name in methods:
            widths[name][test_idx] = 2 * quantiles[name] * scales[name][test_idx]
            fold_report[name] = _interval_metrics(
                y[test_idx], mean[test_idx], scales[name][test_idx],
                quantiles[name], groups[test_idx])
        fold_reports.append(fold_report)

    if any(not np.isfinite(values).all() for values in widths.values()):
        raise RuntimeError("outer folds did not produce complete held-out intervals")
    metrics = {}
    for name in methods:
        covered = np.abs(y - mean) <= widths[name] / 2
        metrics[name] = {
            "rows": int(len(y)), "groups": int(len(unique_groups)),
            "row_coverage": float(np.mean(covered)),
            "group_all_rows_coverage": float(np.mean([
                np.all(covered[groups == group]) for group in unique_groups])),
            "mean_width": float(np.mean(widths[name])),
        }

    output.mkdir(parents=True)
    with (output / "oof_intervals.csv").open("w", newline="") as stream:
        fields = ["row", "group", "label", "uniform_mean", "member_disagreement",
                  "global_group_max_width", "heteroscedastic_group_max_width",
                  "global_covered", "heteroscedastic_covered"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow({
                "row": int(row), "group": str(groups[i]), "label": float(y[i]),
                "uniform_mean": float(mean[i]),
                "member_disagreement": float(disagreement[i]),
                "global_group_max_width": float(widths["global_group_max"][i]),
                "heteroscedastic_group_max_width": float(widths["heteroscedastic_group_max"][i]),
                "global_covered": bool(abs(y[i] - mean[i]) <= widths["global_group_max"][i] / 2),
                "heteroscedastic_covered": bool(
                    abs(y[i] - mean[i]) <= widths["heteroscedastic_group_max"][i] / 2),
            })

    report = {
        "format": "tack-heteroscedastic-interval-study/v1",
        "selection_performed": False, "final_calibration_performed": False,
        "calibration_labels_read": False, "confirmation_labels_read": False,
        "only_stacking_labels_read": True,
        "snapshot_manifest_sha256": sha256(snapshot / "manifest.json"),
        "snapshot_artifact_hashes": {
            name: manifest["artifact_hashes"][name]
            for name in (f"{artifact.recipe['feature']}.npy", "labels.npz", "groups.npy", "partition.npy")},
        "bundle": str(bundle), "bundle_model_sha256": bundle_manifest["model_sha256"],
        "bundle_manifest_sha256": sha256(bundle / "manifest.json"),
        "study_code_sha256": sha256(Path(__file__)),
        "task": "pdc50", "label_policy": "exact", "alpha": float(alpha),
        "folds": int(folds), "seed": int(seed),
        "scale_fraction_of_outer_training_groups": float(scale_fraction),
        "stacking_rows": int(len(rows)), "stacking_groups": int(len(unique_groups)),
        "uniform_member_count": int(member_predictions.shape[1]),
        "scale_model": {"estimator": "ExtraTreesRegressor", **SCALE_MODEL_PARAMS,
                        "random_state_per_fold": "seed + 1009 * (fold + 1)",
                        "target": "absolute residual on scale-fit groups only",
                        "features": ["uniform pooled prediction", "member disagreement"],
                        "positive_floor": SCALE_FLOOR},
        "metrics": metrics,
        "paired_group_bootstrap": _bootstrap_comparison(
            y, mean, widths, groups, n_bootstrap=n_bootstrap, seed=seed + 701),
        "fold_metrics": fold_reports,
        "assumptions_and_limits": (
            "The group-maximum quantile targets marginal coverage of all observed rows "
            "within a new exchangeable chemical group. Chemical groups may not be "
            "exchangeable under source or biological-context shift; coverage is not "
            "conditional on assay, source, or group size. The heteroscedastic model "
            "uses only pooled prediction and member disagreement, learns absolute "
            "residual scale on scale-fit groups, and adds disagreement in quadrature. "
            "Outer-fold results reuse a finite stacking partition and are a development "
            "comparison, not final calibration or a method-selection result."),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--scale-fraction", type=float, default=0.5)
    args = parser.parse_args(argv)
    report = study(args.snapshot, args.bundle, args.output, alpha=args.alpha,
                   folds=args.folds, seed=args.seed, n_bootstrap=args.bootstrap,
                   scale_fraction=args.scale_fraction)
    print(json.dumps({"output": str(args.output), "metrics": report["metrics"],
                      "final_calibration_performed": False}, indent=2))


if __name__ == "__main__":
    main()
