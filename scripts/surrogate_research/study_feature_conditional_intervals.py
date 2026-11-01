"""Compare fixed feature-conditioned pDC50 scales on held-out stacking groups.

This development study does not fit final calibration or read confirmation labels.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from tackai.fusion.context import CONTEXT_BLOCKS
from tackai.fusion.research import read_npz_rows, sha256

try:
    from .study_conformal_coverage import (_finite_sample_quantile, _group_max_scores,
                                           _interval_metrics, _validate_bundle)
    from .study_heteroscedastic_intervals import (_bootstrap_comparison, SCALE_FLOOR,
                                                   SCALE_MODEL_PARAMS)
except ImportError:
    from study_conformal_coverage import (_finite_sample_quantile, _group_max_scores,
                                          _interval_metrics, _validate_bundle)
    from study_heteroscedastic_intervals import (_bootstrap_comparison, SCALE_FLOOR,
                                                  SCALE_MODEL_PARAMS)

DESCRIPTORS = ("MolWt", "TPSA", "MolLogP", "NumRotatableBonds")
FEATURE_PARAMS = {**SCALE_MODEL_PARAMS, "min_samples_leaf": 20}


def _feature_columns(matrix, feature):
    """Select fixed named descriptors and all context columns in model order."""
    dims, names = feature["dims"], feature["descriptors"]
    if len(names) != dims["descriptors"] or len(set(names)) != len(names):
        raise ValueError("Descriptor names do not match the frozen feature contract")
    if any(name not in names for name in DESCRIPTORS):
        raise ValueError("Required fixed descriptor is absent")
    if matrix.ndim != 2 or matrix.shape[1] != sum(dims.values()):
        raise ValueError("Feature matrix width does not match the frozen contract")
    expected = {"fingerprint", "descriptors", *CONTEXT_BLOCKS}
    if set(dims) != expected or any(not isinstance(dims[k], int) or dims[k] < 1 for k in expected):
        raise ValueError("Unexpected feature blocks")
    fp, desc = dims["fingerprint"], dims["descriptors"]
    indices = [fp + names.index(name) for name in DESCRIPTORS]
    indices += list(range(fp + desc, matrix.shape[1]))
    result = np.asarray(matrix[:, indices], dtype=float)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite fixed scale features")
    return result


def _fit_scales(mean, disagreement, fixed, y, scale_idx, seed):
    """Fit two fixed residual models using only supplied scale-fit indices."""
    residual = np.abs(y[scale_idx] - mean[scale_idx])
    global_rms = max(float(np.sqrt(np.mean(residual ** 2))), SCALE_FLOOR)
    base = np.column_stack([mean, disagreement])
    feature = np.column_stack([base, fixed])
    scales = {"global": np.sqrt(global_rms ** 2 + disagreement ** 2)}
    for name, predictors, params in (("two_feature", base, SCALE_MODEL_PARAMS),
                                      ("fixed_features", feature, FEATURE_PARAMS)):
        model = ExtraTreesRegressor(**params, random_state=seed)
        model.fit(predictors[scale_idx], residual)
        noise = np.maximum(model.predict(predictors), SCALE_FLOOR)
        scales[name] = np.sqrt(noise ** 2 + disagreement ** 2)
    if any(not np.isfinite(scale).all() or np.any(scale <= 0) for scale in scales.values()):
        raise ValueError("Scale predictions must be positive and finite")
    return scales


def study(snapshot, bundle, output, *, alpha=.1, folds=5, seed=20261101,
          n_bootstrap=1000, scale_fraction=.5):
    snapshot, bundle, output = Path(snapshot), Path(bundle), Path(output)
    if output.exists():
        raise FileExistsError(f"Study output already exists: {output}")
    if not 0 < alpha < 1 or folds < 2 or n_bootstrap < 0 or not 0 < scale_fraction < 1:
        raise ValueError("Invalid study parameters")
    (snapshot, bundle, manifest, bundle_manifest, artifact, X, groups_all,
     roles) = _validate_bundle(snapshot, bundle)
    if artifact.recipe.get("task") != "pdc50":
        raise ValueError("Study requires a pDC50 bundle")
    stack_rows = np.flatnonzero(roles == "stacking")
    labels = read_npz_rows(snapshot / "labels.npz", "exact_pdc50", stack_rows)
    finite = np.isfinite(labels)
    rows, y = stack_rows[finite], labels[finite]
    groups = groups_all[rows]
    unique_groups = np.unique(groups)
    if len(unique_groups) < max(folds, int(np.ceil(1 / alpha)) + 2):
        raise ValueError("Too few labeled stacking groups")
    member_predictions = artifact._predictions(X[rows])
    mean = member_predictions.mean(axis=1)
    disagreement = np.sqrt(np.mean((member_predictions - mean[:, None]) ** 2, axis=1))
    spec = manifest["features"][artifact.recipe["feature"]]
    fixed = _feature_columns(X[rows], spec)
    names = ("global", "two_feature", "fixed_features")
    widths = {name: np.full(len(rows), np.nan) for name in names}
    reports = []
    outer = GroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (train_idx, eval_idx) in enumerate(outer.split(mean, y, groups)):
        fold_seed = seed + 1009 * (fold + 1)
        split = GroupShuffleSplit(n_splits=1, test_size=1 - scale_fraction, random_state=fold_seed)
        scale_local, quantile_local = next(split.split(np.arange(len(train_idx)), groups=groups[train_idx]))
        scale_idx, quantile_idx = train_idx[scale_local], train_idx[quantile_local]
        sg, qg, eg = (set(groups[idx]) for idx in (scale_idx, quantile_idx, eval_idx))
        if sg & qg or sg & eg or qg & eg:
            raise ValueError("Scale, quantile and evaluation groups must be disjoint")
        if len(qg) < int(np.ceil(1 / alpha)):
            raise ValueError("Too few quantile groups")
        scales = _fit_scales(mean, disagreement, fixed, y, scale_idx, fold_seed)
        report = {"fold": fold, "scale_group_ids": sorted(map(str, sg)),
                  "quantile_group_ids": sorted(map(str, qg)), "evaluation_group_ids": sorted(map(str, eg))}
        for name in names:
            scores = np.abs(y[quantile_idx] - mean[quantile_idx]) / scales[name][quantile_idx]
            q = _finite_sample_quantile(_group_max_scores(scores, groups[quantile_idx]), alpha)
            widths[name][eval_idx] = 2 * q * scales[name][eval_idx]
            report[name] = _interval_metrics(y[eval_idx], mean[eval_idx], scales[name][eval_idx], q, groups[eval_idx])
            report[name]["quantile"] = q
        reports.append(report)
    if any(not np.isfinite(v).all() for v in widths.values()):
        raise RuntimeError("Incomplete outer-fold intervals")
    metrics = {}
    for name in names:
        covered = np.abs(y - mean) <= widths[name] / 2
        metrics[name] = {"rows": len(y), "groups": len(unique_groups),
                         "row_coverage": float(covered.mean()),
                         "group_all_rows_coverage": float(np.mean([np.all(covered[groups == group]) for group in unique_groups])),
                         "mean_width": float(widths[name].mean())}
    comparisons = {}
    for reference in ("global", "two_feature"):
        comparisons[reference] = _bootstrap_comparison(
            y, mean, {"global_group_max": widths[reference], "heteroscedastic_group_max": widths["fixed_features"]},
            groups, n_bootstrap=n_bootstrap, seed=seed + 701)
    output.mkdir(parents=True)
    with (output / "oof_intervals.csv").open("w", newline="") as stream:
        fields = ["row", "group", "label", "uniform_mean", "member_disagreement", *[f"{name}_width" for name in names]]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow({"row": int(row), "group": str(groups[i]), "label": float(y[i]),
                             "uniform_mean": float(mean[i]), "member_disagreement": float(disagreement[i]),
                             **{f"{name}_width": float(widths[name][i]) for name in names}})
    report = {"format": "tack-feature-conditional-study/v1", "snapshot_manifest_sha256": sha256(snapshot / "manifest.json"),
              "bundle_manifest_sha256": sha256(bundle / "manifest.json"), "bundle_model_sha256": bundle_manifest["model_sha256"],
              "study_code_sha256": sha256(__file__), "alpha": alpha, "folds": folds, "seed": seed,
              "scale_fraction": scale_fraction, "only_stacking_labels_read": True,
              "final_calibration_performed": False, "confirmation_labels_read": False,
              "descriptor_names": list(DESCRIPTORS), "context_blocks": list(CONTEXT_BLOCKS),
              "prediction_features": ["uniform mean", "member disagreement"],
              "two_feature_params": SCALE_MODEL_PARAMS, "fixed_feature_params": FEATURE_PARAMS,
              "metrics": metrics, "paired_comparisons": comparisons, "fold_metrics": reports,
              "note": "Development-only nested grouped comparison; coverage assumes exchangeable groups."}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=.1)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--scale-fraction", type=float, default=.5)
    args = parser.parse_args()
    report = study(args.snapshot, args.bundle, args.output, alpha=args.alpha, folds=args.folds,
                   seed=args.seed, n_bootstrap=args.bootstrap, scale_fraction=args.scale_fraction)
    print(json.dumps({"metrics": report["metrics"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
