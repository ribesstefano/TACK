"""Evaluate one locked calibrated candidate on internal confirmation groups.

This command does not train, calibrate, compare or select candidates. It reads
only exact task labels at rows assigned to the frozen ``confirmation`` role.
The result is an internal confirmation report, not an external validation claim.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, mean_absolute_error, mean_squared_error,
                             r2_score, roc_auc_score)

from tackai.fusion.candidate_artifact import CandidateArtifact
from tackai.fusion.research import read_npz_rows, sha256


def _metrics(task, y, prediction):
    """Calculate prespecified point and calibration metrics."""
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if task != "activity":
        r2 = float(r2_score(y, prediction)) if len(y) > 1 and np.ptp(y) > 0 else None
        return {"rmse": float(np.sqrt(mean_squared_error(y, prediction))),
                "mae": float(mean_absolute_error(y, prediction)), "r2": r2}
    p = np.clip(prediction, 1e-7, 1 - 1e-7)
    has_both = np.unique(y).size == 2
    return {
        "ap": float(average_precision_score(y, p)) if np.any(y == 1) else None,
        "roc_auc": float(roc_auc_score(y, p)) if has_both else None,
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }


def _bootstrap(task, y, prediction, groups, *, n_bootstrap, seed,
               covered=None):
    """Compute percentile intervals by resampling whole confirmation groups."""
    unique_groups = np.unique(groups)
    group_rows = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    metric_samples = {}
    row_coverage_samples = []
    group_coverage_samples = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        rows = np.concatenate([group_rows[group] for group in sampled])
        values = _metrics(task, y[rows], prediction[rows])
        for name, value in values.items():
            if value is not None and np.isfinite(value):
                metric_samples.setdefault(name, []).append(float(value))
        if covered is not None:
            row_coverage_samples.append(float(np.mean(covered[rows])))
            group_hits = [bool(np.all(covered[group_rows[group]])) for group in sampled]
            group_coverage_samples.append(float(np.mean(group_hits)))

    result = {name: {"ci95": _percentile_interval(samples),
                     "valid_bootstraps": len(samples)}
              for name, samples in metric_samples.items()}
    if covered is not None:
        result["row_coverage"] = {
            "ci95": _percentile_interval(row_coverage_samples),
            "valid_bootstraps": len(row_coverage_samples),
        }
        result["group_all_coverage"] = {
            "ci95": _percentile_interval(group_coverage_samples),
            "valid_bootstraps": len(group_coverage_samples),
        }
    return result


def _percentile_interval(values):
    if not values:
        return None
    return [float(value) for value in np.quantile(values, [.025, .975])]


def _verify_artifact(snapshot, artifact_dir, task, selected_recipe_sha256):
    """Validate provenance, task, calibration and frozen feature layout."""
    manifest_path = snapshot / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "tack-surrogate-research/v1":
        raise ValueError("unsupported research snapshot format")
    artifact = CandidateArtifact.from_pretrained(artifact_dir)
    recipe = artifact.recipe
    contract = artifact.context_contract
    if recipe.get("task") != task:
        raise ValueError("selected task does not match the candidate recipe")
    if recipe.get("recipe_sha256") != selected_recipe_sha256:
        raise ValueError("selected recipe hash does not match the candidate artifact")
    if recipe.get("fit_stage") != "calibrated":
        raise ValueError("candidate artifact is not marked as fully calibrated")
    if artifact.task != ("classification" if task == "activity" else "regression"):
        raise ValueError("candidate prediction type does not match the requested task")
    if task == "activity":
        if artifact.temperature_ is None:
            raise ValueError("activity candidate is not temperature calibrated")
    elif (artifact.residual_scale_ is None or artifact.q_hat_ is None
          or artifact.alpha_ is None):
        raise ValueError("regression candidate lacks residual-scale or conformal calibration")
    elif not np.isclose(artifact.alpha_, .1):
        raise ValueError("regression candidate must carry nominal 90% intervals")

    manifest_hash = sha256(manifest_path)
    if (recipe.get("snapshot_hash") != manifest_hash
            or contract.get("snapshot_manifest_sha256") != manifest_hash):
        raise ValueError("candidate and confirmation snapshot manifests differ")
    if contract.get("context_fingerprint") != manifest.get("context_fingerprint"):
        raise ValueError("candidate context fingerprint differs from snapshot")
    feature = recipe.get("feature")
    if feature not in manifest.get("features", {}):
        raise ValueError("candidate feature is absent from the snapshot")
    feature_meta = manifest["features"][feature]
    expected_dims = feature_meta.get("dims")
    if contract.get("feature_name") != feature or contract.get("feature_dims") != expected_dims:
        raise ValueError("candidate feature layout differs from snapshot manifest")
    expected_width = sum(expected_dims.values())
    if contract.get("feature_width") != expected_width or artifact.feature_width_ != expected_width:
        raise ValueError("candidate feature width differs from snapshot layout")
    spec = artifact.molecular_spec
    if (spec.radius != feature_meta["radius"] or spec.size != feature_meta["fp_size"]
            or spec.count != feature_meta["counts"]
            or list(spec.descriptor_names) != feature_meta["descriptors"]):
        raise ValueError("candidate molecular feature spec differs from snapshot")
    if manifest["versions"]["rdkit"] != importlib.metadata.version("rdkit"):
        raise ValueError("snapshot RDKit distribution differs from the runtime")
    if contract.get("molecular_feature_spec") != spec.to_dict():
        raise ValueError("candidate molecular spec metadata is inconsistent")

    files = {
        "feature": snapshot / f"{feature}.npy",
        "labels": snapshot / "labels.npz",
        "groups": snapshot / "groups.npy",
        "partition": snapshot / "partition.npy",
    }
    for path in files.values():
        expected = manifest.get("artifact_hashes", {}).get(path.name)
        if not expected or sha256(path) != expected:
            raise ValueError(f"frozen snapshot hash mismatch: {path.name}")
        if contract.get("snapshot_artifact_hashes", {}).get(path.name) != expected:
            raise ValueError(f"candidate artifact does not match snapshot file: {path.name}")
    return manifest, manifest_hash, files, artifact


def evaluate(snapshot, artifact_dir, output, *, task, selected_recipe_sha256,
             n_bootstrap=2000, seed=20261101):
    """Evaluate one explicitly selected candidate once on confirmation groups.

    The function reads the exact endpoint label vector only at rows whose frozen
    role is ``confirmation``. It never reads historical test inputs or labels.
    """
    snapshot, artifact_dir, output = map(Path, (snapshot, artifact_dir, output))
    if output.exists():
        raise FileExistsError(f"confirmation output already exists: {output}")
    if task not in {"dmax", "pdc50", "activity"}:
        raise ValueError("task must be dmax, pdc50, or activity")
    if n_bootstrap < 0:
        raise ValueError("n_bootstrap must be nonnegative")
    if (not isinstance(selected_recipe_sha256, str) or len(selected_recipe_sha256) != 64
            or any(c not in "0123456789abcdef" for c in selected_recipe_sha256)):
        raise ValueError("selected_recipe_sha256 must be a lowercase SHA256 digest")

    manifest, manifest_hash, files, artifact = _verify_artifact(
        snapshot, artifact_dir, task, selected_recipe_sha256)
    X = np.load(files["feature"], mmap_mode="r", allow_pickle=False)
    groups_all = np.load(files["groups"], allow_pickle=False)
    roles = np.load(files["partition"], allow_pickle=False)
    if X.ndim != 2 or X.shape[0] != len(groups_all) or len(roles) != len(groups_all):
        raise ValueError("snapshot features, groups and roles are misaligned")
    if X.shape[1] != artifact.feature_width_:
        raise ValueError("snapshot design matrix width differs from candidate")
    for group in np.unique(groups_all):
        if len(set(roles[groups_all == group])) != 1:
            raise ValueError(f"group {group!r} crosses frozen partitions")

    confirmation_rows = np.flatnonzero(roles == "confirmation")
    if len(confirmation_rows) == 0:
        raise ValueError("snapshot has no confirmation rows")

    # Lock this output immediately before the only operation that reads
    # confirmation label values. A failure after this point leaves an explicit
    # marker and the output path cannot be reused silently.
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    in_progress = output / "IN_PROGRESS"
    with in_progress.open("x") as stream:
        stream.write("Confirmation labels may have been read; do not retry this output path.\n")

    # Exact labels only; role-limited reads keep all other label offsets out of
    # the evaluator's numeric path.
    label_key = f"exact_{task}"
    y = read_npz_rows(files["labels"], label_key, confirmation_rows).astype(float)
    labeled = np.isfinite(y)
    rows, y = confirmation_rows[labeled], y[labeled]
    groups = groups_all[rows]
    if len(y) == 0:
        raise ValueError("confirmation partition has no exact labels for this task")
    if task == "activity" and not np.isin(y, [0., 1.]).all():
        raise ValueError("exact activity labels must be binary")

    prediction = artifact.predict(X[rows])
    if task == "activity":
        point = np.asarray(prediction["proba"], dtype=float)
        metrics = _metrics(task, y, point)
        intervals = None
    else:
        point = np.asarray(prediction["mean"], dtype=float)
        metrics = _metrics(task, y, point)
        lower, upper = np.asarray(prediction["lower"]), np.asarray(prediction["upper"])
        covered = (y >= lower) & (y <= upper)
        metrics["interval_90"] = {
            "coverage": float(np.mean(covered)),
            "mean_width": float(np.mean(upper - lower)),
            "group_all_coverage": float(np.mean([
                np.all(covered[groups == group]) for group in np.unique(groups)])),
            "nominal_coverage": .9,
            "alpha": float(artifact.alpha_),
            "calibration_unit": getattr(artifact, "calibration_unit_", None) or "row",
        }
        intervals = covered

    unique_groups = np.unique(groups)
    bootstrap = _bootstrap(task, y, point, groups, n_bootstrap=n_bootstrap,
                           seed=seed, covered=intervals)
    if intervals is not None:
        bootstrap["interval_90"] = {
            "row_coverage_ci95": bootstrap.pop("row_coverage"),
            "group_all_coverage_ci95": bootstrap.pop("group_all_coverage"),
        }
    report = {
        "format": "tack-candidate-internal-confirmation/v1",
        "evaluation_scope": "internal_confirmation_only",
        "candidate_selection_or_fitting_performed": False,
        "historical_test_accessed": False,
        "task": task,
        "label_key": label_key,
        "recipe_sha256": selected_recipe_sha256,
        "artifact": str(artifact_dir),
        "artifact_model_sha256": sha256(artifact_dir / "candidate.joblib"),
        "artifact_manifest_sha256": sha256(artifact_dir / "manifest.json"),
        "snapshot_manifest_sha256": manifest_hash,
        "snapshot_artifact_hashes": {path.name: manifest["artifact_hashes"][path.name]
                                      for path in files.values()},
        "evaluator_code_sha256": sha256(Path(__file__)),
        "n_rows": int(len(rows)),
        "n_groups": int(len(unique_groups)),
        "metrics": metrics,
        "group_bootstrap": {
            "method": "resample confirmation chemical groups with replacement; include all rows per sampled group",
            "seed": seed,
            "n_bootstrap": n_bootstrap,
            "metrics": bootstrap,
        },
        "interpretation_note": (
            "This is one internal confirmation evaluation. Group separation does not prove "
            "exchangeability or external coverage. Row calibration targets marginal row "
            "coverage; group-max calibration targets all observed rows in an exchangeable "
            "new group. Both coverage estimates here are descriptive."),
    }
    report_path = output / "report.json"
    temporary_report = output / "report.json.tmp"
    temporary_report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary_report.replace(report_path)
    in_progress.replace(output / "COMPLETE")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=["dmax", "pdc50", "activity"], required=True)
    parser.add_argument("--recipe-sha256", required=True,
                        help="Exact recipe_sha256 recorded in the selected candidate artifact")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261101)
    args = parser.parse_args(argv)
    result = evaluate(args.snapshot, args.artifact, args.output, task=args.task,
                      selected_recipe_sha256=args.recipe_sha256,
                      n_bootstrap=args.bootstrap, seed=args.seed)
    print(json.dumps({"output": str(args.output), "scope": result["evaluation_scope"],
                      "rows": result["n_rows"], "groups": result["n_groups"]}, indent=2))


if __name__ == "__main__":
    main()
