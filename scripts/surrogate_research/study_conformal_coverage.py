"""Compare row- and chemical-group conformal intervals on stacking groups only.

The candidate models are frozen and must come from a search-only,
pre-calibration CandidateArtifact bundle. Stacking groups are divided into
outer evaluation folds; each outer-training partition is split again into
disjoint residual-scale and conformal-quantile groups. This is a development
study, not final calibration and not a strategy-selection procedure.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from tackai.fusion.candidate_artifact import CandidateArtifact
from tackai.fusion.research import read_npz_rows, sha256


def _finite_sample_quantile(scores, alpha):
    """Return the split-conformal order statistic, rejecting undersized sets."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("conformal scores must be a nonempty finite vector")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    rank = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    if rank > len(scores):
        raise ValueError(
            f"quantile set too small (n={len(scores)}) for alpha={alpha}; "
            "the finite-sample conformal quantile is infinite")
    return float(np.partition(scores, rank - 1)[rank - 1])


def _group_max_scores(scores, groups):
    """Reduce row scores to one simultaneous score for each chemical group."""
    scores = np.asarray(scores, dtype=float)
    groups = np.asarray(groups)
    if scores.ndim != 1 or groups.shape != scores.shape or not len(scores):
        raise ValueError("scores and groups must be nonempty aligned vectors")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    return np.asarray([np.max(scores[groups == group]) for group in np.unique(groups)])


def _interval_metrics(y, mean, scale, q, groups):
    """Compute row coverage, all-rows group coverage, and average interval width."""
    lower, upper = mean - q * scale, mean + q * scale
    covered = (y >= lower) & (y <= upper)
    unique_groups = np.unique(groups)
    group_covered = np.asarray([np.all(covered[groups == group]) for group in unique_groups])
    return {
        "rows": int(len(y)),
        "groups": int(len(unique_groups)),
        "row_coverage": float(np.mean(covered)),
        "group_all_rows_coverage": float(np.mean(group_covered)),
        "mean_width": float(np.mean(upper - lower)),
    }


def _validate_bundle(snapshot, bundle, *, task_type="regression"):
    """Load and cross-check the frozen snapshot and uncalibrated bundle."""
    snapshot, bundle = Path(snapshot), Path(bundle)
    manifest_path = snapshot / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "tack-surrogate-research/v1":
        raise ValueError("unsupported research snapshot format")

    bundle_manifest_path = bundle / "manifest.json"
    bundle_manifest = json.loads(bundle_manifest_path.read_text())
    artifact = CandidateArtifact.from_pretrained(bundle)
    recipe, contract = artifact.recipe, artifact.context_contract
    if recipe.get("fit_stage") != "pre_calibration":
        raise ValueError("conformal study requires a pre-calibration bundle")
    if recipe.get("snapshot_hash") != sha256(manifest_path):
        raise ValueError("bundle recipe is pinned to a different snapshot")
    if contract.get("snapshot_manifest_sha256") != sha256(manifest_path):
        raise ValueError("bundle context contract is pinned to a different snapshot")
    if bundle_manifest.get("recipe") != recipe or bundle_manifest.get("context_contract") != contract:
        raise ValueError("candidate bundle provenance differs from its serialized artifact")
    expected_tasks = {"regression": {"dmax", "pdc50"}, "classification": {"activity"}}
    if task_type not in expected_tasks or artifact.task != task_type or recipe.get("task") not in expected_tasks[task_type]:
        raise ValueError(f"study requires a {task_type} candidate")
    if (artifact.residual_scale_ is not None or artifact.q_hat_ is not None
            or artifact.alpha_ is not None or artifact.temperature_ is not None):
        raise ValueError("candidate bundle must be uncalibrated")
    if recipe.get("stacking", {}).get("weights") != "uniform":
        raise ValueError("candidate recipe must specify uniform weights")
    if len(artifact.models) != len(recipe.get("members", [])) or not artifact.models:
        raise ValueError("bundle members do not match its pinned recipe")

    role_counts = contract.get("role_counts")
    if not isinstance(role_counts, dict) or not {"search", "stacking", "calibration", "confirmation"}.issubset(role_counts):
        raise ValueError("bundle lacks frozen partition-count provenance")
    fitted_members = recipe["members"]
    search_rows = int(role_counts["search"]["rows"])
    search_groups = int(role_counts["search"]["groups"])
    if any(not isinstance(m.get("training_rows_sha256"), str)
           or len(m["training_rows_sha256"]) != 64
           or not isinstance(m.get("training_rows"), int)
           or not isinstance(m.get("training_groups"), int)
           or not 0 < m["training_rows"] <= search_rows
           or not 0 < m["training_groups"] <= search_groups
           for m in fitted_members):
        raise ValueError("bundle lacks fitted-member training provenance")

    feature_name = recipe.get("feature")
    if feature_name not in manifest.get("features", {}):
        raise ValueError("bundle feature is absent from the frozen snapshot")
    filenames = (f"{feature_name}.npy", "labels.npz", "groups.npy", "partition.npy")
    for name in filenames:
        expected = manifest.get("artifact_hashes", {}).get(name)
        if not expected or sha256(snapshot / name) != expected:
            raise ValueError(f"snapshot hash mismatch: {name}")
        if contract.get("snapshot_artifact_hashes", {}).get(name) != expected:
            raise ValueError(f"bundle does not match frozen snapshot artifact: {name}")

    feature = np.load(snapshot / f"{feature_name}.npy", mmap_mode="r", allow_pickle=False)
    groups = np.load(snapshot / "groups.npy", allow_pickle=False)
    roles = np.load(snapshot / "partition.npy", allow_pickle=False)
    if feature.ndim != 2 or feature.shape[0] != len(groups) or len(roles) != len(groups):
        raise ValueError("snapshot feature, group, and role arrays are misaligned")
    if feature.shape[1] != artifact.feature_width_:
        raise ValueError("bundle feature width differs from frozen snapshot")
    for group in np.unique(groups):
        if len(set(roles[groups == group])) != 1:
            raise ValueError("a chemical group crosses frozen snapshot roles")
    if not set(np.unique(roles)).issubset({"search", "stacking", "calibration", "confirmation"}):
        raise ValueError("snapshot has an unknown partition role")
    actual_counts = {
        role: {"rows": int(np.sum(roles == role)),
               "groups": int(len(np.unique(groups[roles == role])))}
        for role in ("search", "stacking", "calibration", "confirmation")
    }
    if role_counts != actual_counts:
        raise ValueError("bundle partition counts differ from frozen snapshot")
    return snapshot, bundle, manifest, bundle_manifest, artifact, feature, groups, roles


def study(snapshot, bundle, output, *, alpha=0.1, folds=5, seed=20261101,
          n_bootstrap=1000, scale_fraction=0.5):
    """Run grouped nested split conformal coverage comparisons on stacking rows."""
    snapshot, bundle, output = Path(snapshot), Path(bundle), Path(output)
    if output.exists():
        raise FileExistsError(f"study output already exists: {output}")
    if not 0 < alpha < 1 or folds < 2 or n_bootstrap < 0:
        raise ValueError("alpha must be in (0,1), folds >= 2, and bootstrap >= 0")
    if not 0 < scale_fraction < 1:
        raise ValueError("scale_fraction must lie strictly between zero and one")
    (snapshot, bundle, manifest, bundle_manifest, artifact, X, groups_all,
     roles) = _validate_bundle(snapshot, bundle)

    stacking_rows = np.flatnonzero(roles == "stacking")
    # This role-limited reader receives stacking row offsets only. No labels from
    # search, calibration, or confirmation are requested by this study.
    y_all = read_npz_rows(snapshot / "labels.npz",
                          f"{artifact.recipe['label_policy']}_{artifact.recipe['task']}",
                          stacking_rows)
    finite = np.isfinite(y_all)
    rows, y = stacking_rows[finite], y_all[finite]
    groups = groups_all[rows]
    unique_groups = np.unique(groups)
    if len(unique_groups) < max(folds, int(math.ceil(1 / alpha)) + 2):
        raise ValueError("too few labeled stacking groups for requested folds and alpha")
    if len(y) < max(folds, int(math.ceil(1 / alpha)) + 2):
        raise ValueError("too few labeled stacking rows for requested folds and alpha")

    # Freeze uniform predictions and disagreement once; no model or weight fitting.
    member_predictions = artifact._predictions(X[rows])
    uniform_mean = member_predictions.mean(axis=1)
    disagreement = np.sqrt(np.mean((member_predictions - uniform_mean[:, None]) ** 2,
                                   axis=1))
    oof = {name: np.full(len(rows), np.nan) for name in ("row", "group_max")}
    widths = {name: np.full(len(rows), np.nan) for name in ("row", "group_max")}
    fold_reports = []
    outer = GroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(outer.split(uniform_mean, y, groups)):
        train_groups, test_groups = set(groups[train_idx]), set(groups[test_idx])
        if train_groups & test_groups:
            raise AssertionError("outer GroupKFold split a chemical group")
        train_local = np.arange(len(train_idx))
        nested = GroupShuffleSplit(n_splits=1, test_size=1 - scale_fraction,
                                   random_state=seed + 1009 * (fold + 1))
        scale_local, quantile_local = next(nested.split(
            train_local, groups=groups[train_idx]))
        scale_idx, quantile_idx = train_idx[scale_local], train_idx[quantile_local]
        scale_groups, quantile_groups = set(groups[scale_idx]), set(groups[quantile_idx])
        if scale_groups & quantile_groups or scale_groups & test_groups or quantile_groups & test_groups:
            raise AssertionError("scale, quantile, and outer evaluation groups must be disjoint")
        if len(quantile_groups) < int(math.ceil(1 / alpha)):
            raise ValueError(f"fold {fold} has too few quantile groups for alpha={alpha}")

        residual_scale = max(float(np.sqrt(np.mean((y[scale_idx] - uniform_mean[scale_idx]) ** 2))),
                             1e-12)
        scale = np.sqrt(residual_scale ** 2 + disagreement ** 2)
        scores = np.abs(y - uniform_mean) / scale
        row_q = _finite_sample_quantile(scores[quantile_idx], alpha)
        group_q = _finite_sample_quantile(
            _group_max_scores(scores[quantile_idx], groups[quantile_idx]), alpha)
        fold_quantiles = {"row": row_q, "group_max": group_q}
        for name, q in fold_quantiles.items():
            oof[name][test_idx] = q
            widths[name][test_idx] = 2 * q * scale[test_idx]

        fold_report = {
            "fold": fold,
            "scale_rows": int(len(scale_idx)), "scale_groups": int(len(scale_groups)),
            "quantile_rows": int(len(quantile_idx)), "quantile_groups": int(len(quantile_groups)),
            "evaluation_rows": int(len(test_idx)), "evaluation_groups": int(len(test_groups)),
            "residual_scale": residual_scale,
            "row_quantile": row_q, "group_max_quantile": group_q,
            "row_metrics": _interval_metrics(y[test_idx], uniform_mean[test_idx],
                                              scale[test_idx], row_q, groups[test_idx]),
            "group_max_metrics": _interval_metrics(y[test_idx], uniform_mean[test_idx],
                                                    scale[test_idx], group_q, groups[test_idx]),
            "scale_group_ids": sorted(map(str, scale_groups)),
            "quantile_group_ids": sorted(map(str, quantile_groups)),
            "evaluation_group_ids": sorted(map(str, test_groups)),
        }
        fold_reports.append(fold_report)

    if not all(np.isfinite(oof[name]).all() for name in oof):
        raise RuntimeError("outer folds did not produce complete held-out predictions")
    # The per-row widths were frozen with each fold's residual scale and q.
    # Summarize coverage from those interval widths (fold q values can differ).
    coverage = {}
    for name in ("row", "group_max"):
        half_width = widths[name] / 2
        lower, upper = uniform_mean - half_width, uniform_mean + half_width
        covered = (y >= lower) & (y <= upper)
        group_covered = [bool(np.all(covered[groups == g])) for g in unique_groups]
        coverage[name] = {
            "rows": int(len(y)), "groups": int(len(unique_groups)),
            "row_coverage": float(np.mean(covered)),
            "group_all_rows_coverage": float(np.mean(group_covered)),
            "mean_width": float(np.mean(widths[name])),
        }
    bootstrap = _bootstrap_oof(y, uniform_mean, widths, groups,
                              n_bootstrap=n_bootstrap, seed=seed + 701)

    output.mkdir(parents=True)
    with (output / "oof_intervals.csv").open("w", newline="") as stream:
        fields = ["row", "group", "label", "uniform_mean", "member_disagreement",
                  "row_width", "group_max_width", "row_covered", "group_max_covered"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow({
                "row": int(row), "group": str(groups[i]), "label": float(y[i]),
                "uniform_mean": float(uniform_mean[i]),
                "member_disagreement": float(disagreement[i]),
                "row_width": float(widths["row"][i]),
                "group_max_width": float(widths["group_max"][i]),
                "row_covered": bool(abs(y[i] - uniform_mean[i]) <= widths["row"][i] / 2),
                "group_max_covered": bool(abs(y[i] - uniform_mean[i]) <= widths["group_max"][i] / 2),
            })

    report = {
        "format": "tack-conformal-coverage-study/v1",
        "selection_performed": False,
        "final_calibration_performed": False,
        "calibration_labels_read": False,
        "confirmation_labels_read": False,
        "only_stacking_labels_read": True,
        "snapshot_manifest_sha256": sha256(snapshot / "manifest.json"),
        "snapshot_artifact_hashes": {
            name: manifest["artifact_hashes"][name]
            for name in (f"{artifact.recipe['feature']}.npy", "labels.npz", "groups.npy", "partition.npy")},
        "bundle": str(bundle),
        "bundle_model_sha256": bundle_manifest["model_sha256"],
        "bundle_manifest_sha256": sha256(bundle / "manifest.json"),
        "study_code_sha256": sha256(Path(__file__)),
        "task": artifact.recipe["task"], "label_policy": artifact.recipe["label_policy"],
        "alpha": float(alpha), "folds": int(folds), "seed": int(seed),
        "scale_fraction_of_outer_training_groups": float(scale_fraction),
        "stacking_rows": int(len(rows)), "stacking_groups": int(len(unique_groups)),
        "uniform_member_count": int(member_predictions.shape[1]),
        "metrics": coverage, "paired_group_bootstrap": bootstrap,
        "fold_metrics": fold_reports,
        "interpretation": {
            "row_method": "Finite-sample quantile of row-normalized absolute residuals.",
            "group_max_method": ("Finite-sample quantile of each quantile group's maximum "
                                  "row-normalized absolute residual; targets marginal coverage "
                                  "of all observed rows in a new exchangeable chemical group."),
            "limits": ("Coverage is marginal over groups from this stacking population. "
                       "It is not conditional coverage by endpoint, assay, source, or group size; "
                       "exchangeability of chemical groups may fail under domain shift. "
                       "The outer folds reuse a limited set of stacking groups and are a "
                       "development comparison, not an independent final calibration."),
        },
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def _bootstrap_oof(y, mean, widths, groups, *, n_bootstrap, seed):
    """Paired group bootstrap for OOF absolute metrics and method differences."""
    unique = np.unique(groups)
    by_group = {group: np.flatnonzero(groups == group) for group in unique}
    methods = ("row", "group_max")
    metrics = ("row_coverage", "group_all_rows_coverage", "mean_width")
    point = {}
    for method in methods:
        covered = np.abs(y - mean) <= widths[method] / 2
        point[method] = {
            "row_coverage": float(np.mean(covered)),
            "group_all_rows_coverage": float(np.mean([
                np.all(covered[groups == group]) for group in unique])),
            "mean_width": float(np.mean(widths[method])),
        }
    draws = {method: {metric: [] for metric in metrics} for method in methods}
    delta_draws = {metric: [] for metric in metrics}
    rng = np.random.default_rng(seed)
    for _ in range(n_bootstrap):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        chunks, group_chunks = [], []
        for copy_id, group in enumerate(sampled):
            idx = by_group[group]
            chunks.append(idx)
            group_chunks.append(np.full(len(idx), copy_id, dtype=np.int64))
        idx = np.concatenate(chunks)
        boot_groups = np.concatenate(group_chunks)
        boot_metrics = {}
        for method in methods:
            covered = np.abs(y[idx] - mean[idx]) <= widths[method][idx] / 2
            values = {
                "row_coverage": float(np.mean(covered)),
                "group_all_rows_coverage": float(np.mean([
                    np.all(covered[boot_groups == group])
                    for group in np.unique(boot_groups)])),
                "mean_width": float(np.mean(widths[method][idx])),
            }
            boot_metrics[method] = values
            for metric in metrics:
                draws[method][metric].append(values[metric])
        for metric in metrics:
            delta_draws[metric].append(boot_metrics["group_max"][metric]
                                       - boot_metrics["row"][metric])

    def interval(values):
        return [float(x) for x in np.quantile(values, [.025, .975])] if values else None

    return {
        "n_bootstraps": int(n_bootstrap),
        "per_method_ci95": {
            method: {metric: interval(draws[method][metric]) for metric in metrics}
            for method in methods
        },
        "paired_group_max_minus_row_ci95": {
            metric: {"estimate": float(point["group_max"][metric] - point["row"][metric]),
                     "ci95": interval(delta_draws[metric]),
                     "valid_bootstraps": len(delta_draws[metric])}
            for metric in metrics
        },
    }


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
    result = study(args.snapshot, args.bundle, args.output, alpha=args.alpha,
                   folds=args.folds, seed=args.seed, n_bootstrap=args.bootstrap,
                   scale_fraction=args.scale_fraction)
    print(json.dumps({"output": str(args.output), "metrics": result["metrics"],
                      "final_calibration_performed": False}, indent=2))


if __name__ == "__main__":
    main()
