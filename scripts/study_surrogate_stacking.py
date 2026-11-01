"""Compare uniform and convex pooling with grouped CV on stacking groups only.

The input bundle must be fitted on search groups and remain uncalibrated. This
study reads stacking labels only, never reads calibration or confirmation
labels, and reports alternatives without selecting one or refitting final
weights.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from tackai.fusion.candidate_artifact import CandidateArtifact
from tackai.fusion.research import read_npz_rows, sha256

try:
    from .fit_surrogate_candidate import _fit_convex_weights
except ImportError:
    from fit_surrogate_candidate import _fit_convex_weights


def _score(task, y, prediction):
    """Return task-appropriate point and probability metrics."""
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if task != "activity":
        residual = y - prediction
        return {"rmse": float(np.sqrt(np.mean(residual ** 2))),
                "mae": float(np.mean(np.abs(residual)))}
    p = np.clip(prediction, 1e-7, 1 - 1e-7)
    if not np.isin(y, [0., 1.]).all():
        raise ValueError("activity stacking labels must be binary")
    ap = None
    if np.any(y == 1):
        from sklearn.metrics import average_precision_score
        ap = float(average_precision_score(y, p))
    return {"ap": ap, "brier": float(np.mean((y - p) ** 2)),
            "log_loss": float(np.mean(-y * np.log(p) - (1 - y) * np.log1p(-p)))}


def _macro_group_scores(task, y, prediction, groups):
    """Average within-group scores so large groups do not dominate this view."""
    values = [_score(task, y[groups == group], prediction[groups == group])
              for group in np.unique(groups)]
    result = {}
    for key in values[0]:
        if key == "ap":
            # Average precision is a ranking metric across examples; per-group
            # AP is unstable or undefined for the small within-series slices.
            continue
        observed = [row[key] for row in values if row[key] is not None]
        result[f"macro_group_{key}"] = (float(np.mean(observed)) if observed else None)
    return result


def _group_bootstrap_deltas(task, y, uniform, candidate, groups, *, n_bootstrap, seed):
    """Paired bootstrap differences by group; candidate minus uniform."""
    unique_groups = np.unique(groups)
    by_group = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    point_uniform = _score(task, y, uniform)
    point_candidate = _score(task, y, candidate)
    draws = {key: [] for key, value in point_uniform.items()
             if value is not None and point_candidate[key] is not None}
    for _ in range(n_bootstrap):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        rows = np.concatenate([by_group[group] for group in sampled])
        left, right = _score(task, y[rows], uniform[rows]), _score(task, y[rows], candidate[rows])
        for key in draws:
            if left[key] is not None and right[key] is not None:
                draws[key].append(right[key] - left[key])
    result = {}
    for key, samples in draws.items():
        point = point_candidate[key] - point_uniform[key]
        result[key] = {"estimate": float(point),
                       "ci95": ([float(x) for x in np.quantile(samples, [.025, .975])]
                                if samples else None),
                       "valid_bootstraps": len(samples)}
    return result


def study(snapshot, bundle, output, *, folds=5, seed=20261101,
          l2_values=(.01, .1, 1., 10.), n_bootstrap=1000):
    """Run a no-selection, group-held-out comparison on stacking rows.

    Args:
        snapshot: Frozen research snapshot used to fit ``bundle``.
        bundle: Uncalibrated CandidateArtifact trained on search groups only.
        output: New directory for the report and paired out-of-fold predictions.
        folds: Number of group folds.
        seed: Seed used only to shuffle group-to-fold assignment.
        l2_values: Prespecified convex-weight shrinkage strengths.
        n_bootstrap: Paired group bootstrap repetitions for score differences.
    """
    snapshot, bundle, output = Path(snapshot), Path(bundle), Path(output)
    if output.exists():
        raise FileExistsError(f"study output already exists: {output}")
    if folds < 2 or n_bootstrap < 0:
        raise ValueError("folds must be >= 2 and n_bootstrap must be nonnegative")
    l2_values = tuple(float(value) for value in l2_values)
    if (not l2_values or any(not np.isfinite(value) or value < 0 for value in l2_values)
            or len(set(l2_values)) != len(l2_values)
            or len({f"convex_l2_{value:g}" for value in l2_values}) != len(l2_values)):
        raise ValueError("l2_values must be a nonempty sequence of finite nonnegative values")

    manifest_path = snapshot / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "tack-surrogate-research/v1":
        raise ValueError("unsupported research snapshot format")
    artifact = CandidateArtifact.from_pretrained(bundle)
    recipe = artifact.recipe
    context = artifact.context_contract
    if recipe.get("fit_stage") != "pre_calibration":
        raise ValueError("stacking study requires a pre-calibration model bundle")
    if context.get("snapshot_manifest_sha256") != sha256(manifest_path):
        raise ValueError("model bundle and snapshot manifest differ")
    if recipe.get("snapshot_hash") != sha256(manifest_path):
        raise ValueError("bundle recipe is pinned to a different snapshot")
    if recipe.get("task") not in {"dmax", "pdc50", "activity"}:
        raise ValueError("bundle recipe has an unsupported task")
    if artifact.q_hat_ is not None or artifact.alpha_ is not None or artifact.temperature_ is not None:
        raise ValueError("stacking study requires an uncalibrated model bundle")
    if artifact.residual_scale_ is not None:
        raise ValueError("pre-calibration bundle must not contain a stacking residual scale")

    feature_name = recipe["feature"]
    if feature_name not in manifest.get("features", {}):
        raise ValueError("bundle feature is absent from the snapshot")
    files = {"feature": snapshot / f"{feature_name}.npy",
             "labels": snapshot / "labels.npz",
             "groups": snapshot / "groups.npy",
             "partition": snapshot / "partition.npy"}
    for name, path in files.items():
        expected = manifest["artifact_hashes"].get(path.name)
        if not expected or sha256(path) != expected:
            raise ValueError(f"snapshot hash mismatch: {path.name}")
        if context.get("snapshot_artifact_hashes", {}).get(path.name) != expected:
            raise ValueError(f"bundle does not match snapshot artifact: {path.name}")

    X = np.load(files["feature"], mmap_mode="r", allow_pickle=False)
    groups_all = np.load(files["groups"], allow_pickle=False)
    roles = np.load(files["partition"], allow_pickle=False)
    if X.ndim != 2 or X.shape[0] != len(groups_all) or len(roles) != len(groups_all):
        raise ValueError("snapshot features, groups and roles are misaligned")
    if X.shape[1] != artifact.feature_width_:
        raise ValueError("bundle feature width differs from snapshot matrix")
    for group in np.unique(groups_all):
        if len(set(roles[groups_all == group])) != 1:
            raise ValueError(f"group {group!r} crosses snapshot roles")
    rows = np.flatnonzero(roles == "stacking")
    if len(rows) == 0:
        raise ValueError("snapshot has no stacking rows")
    labels_path = files["labels"]
    y = read_npz_rows(labels_path, f"{recipe['label_policy']}_{recipe['task']}", rows)
    finite = np.isfinite(y)
    rows, y = rows[finite], y[finite]
    groups = groups_all[rows]
    unique_groups = np.unique(groups)
    if len(unique_groups) < folds:
        raise ValueError(f"need at least {folds} labeled stacking groups, found {len(unique_groups)}")
    if recipe["task"] == "activity" and set(np.unique(y)) != {0., 1.}:
        raise ValueError("activity stacking partition must contain both labels")

    # Base members are frozen from search. This matrix is the only model output
    # used during the study; no model is fit or updated here.
    predictions = artifact._predictions(X[rows])
    uniform_weights = np.full(predictions.shape[1], 1. / predictions.shape[1])
    uniform_oof = predictions @ uniform_weights
    oof = {float(l2): np.full(len(rows), np.nan) for l2 in l2_values}
    fold_reports = []
    if recipe["task"] == "activity":
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    else:
        splitter = GroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(predictions, y, groups)):
        train_groups, valid_groups = set(groups[train_idx]), set(groups[valid_idx])
        if train_groups & valid_groups:
            raise AssertionError("stacking CV split a group")
        if recipe["task"] == "activity" and set(np.unique(y[train_idx])) != {0., 1.}:
            raise ValueError(f"activity fold {fold} training subset lacks both labels")
        report = {"fold": fold, "training_groups": len(train_groups),
                  "validation_groups": len(valid_groups),
                  "training_group_ids": sorted(map(str, train_groups)),
                  "validation_group_ids": sorted(map(str, valid_groups)),
                  "uniform": _score(recipe["task"], y[valid_idx], uniform_oof[valid_idx])}
        for l2 in l2_values:
            weights = _fit_convex_weights(predictions[train_idx], y[train_idx],
                                          recipe["task"], float(l2))
            oof[float(l2)][valid_idx] = predictions[valid_idx] @ weights
            report[f"convex_l2_{l2:g}"] = {
                "metrics": _score(recipe["task"], y[valid_idx], oof[float(l2)][valid_idx]),
                "weights": weights.tolist(),
            }
        fold_reports.append(report)

    strategies = {"uniform": uniform_oof}
    strategies.update({f"convex_l2_{l2:g}": values for l2, values in oof.items()})
    summary = {}
    for name, prediction in strategies.items():
        if not np.all(np.isfinite(prediction)):
            raise RuntimeError(f"strategy {name} did not produce complete out-of-fold predictions")
        summary[name] = {"pooled": _score(recipe["task"], y, prediction),
                         "group_macro": _macro_group_scores(recipe["task"], y, prediction, groups)}
        if name != "uniform":
            summary[name]["paired_group_bootstrap_delta_vs_uniform"] = _group_bootstrap_deltas(
                recipe["task"], y, uniform_oof, prediction, groups,
                n_bootstrap=n_bootstrap, seed=seed)

    output.mkdir(parents=True)
    with (output / "oof_predictions.csv").open("w", newline="") as stream:
        fields = ["row", "group", "label", "uniform"] + list(strategies.keys())[1:]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(rows):
            item = {"row": int(row), "group": str(groups[i]), "label": float(y[i]),
                    "uniform": float(uniform_oof[i])}
            item.update({name: float(strategies[name][i]) for name in fields[4:]})
            writer.writerow(item)
    report = {
        "format": "tack-stacking-study/v1",
        "selection_performed": False,
        "calibration_labels_read": False,
        "confirmation_labels_read": False,
        "snapshot_manifest_sha256": sha256(manifest_path),
        "snapshot_artifact_hashes": {path.name: manifest["artifact_hashes"][path.name]
                                      for path in files.values()},
        "bundle": str(bundle),
        "bundle_model_sha256": sha256(bundle / "candidate.joblib"),
        "bundle_manifest_sha256": sha256(bundle / "manifest.json"),
        "study_code_sha256": sha256(Path(__file__)),
        "bundle_task": recipe["task"],
        "label_policy": recipe["label_policy"],
        "folds": folds,
        "seed": seed,
        "stacking_rows": int(len(rows)),
        "stacking_groups": int(len(unique_groups)),
        "l2_values": [float(value) for value in l2_values],
        "n_group_bootstraps": int(n_bootstrap),
        "metrics": summary,
        "fold_metrics": fold_reports,
        "note": "Compare these development-only results; no strategy was selected or refit.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--l2", type=float, nargs="+", default=[.01, .1, 1., 10.])
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args(argv)
    report = study(args.snapshot, args.bundle, args.output, folds=args.folds, seed=args.seed,
                   l2_values=args.l2, n_bootstrap=args.bootstrap)
    print(json.dumps({"output": str(args.output), "strategies": list(report["metrics"]),
                      "selection_performed": False}, indent=2))


if __name__ == "__main__":
    main()
