"""Summarize development runs with paired chemical/publication bootstrap.

The fixed-parameter pilot is a screen, not the tuned B1 promotion baseline.
Incomplete candidates are listed explicitly and never ranked as completed.
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, mean_squared_error

try:
    from .benchmark_surrogates import scores
except ImportError:
    from benchmark_surrogates import scores


def paired_view(candidate, baseline, view):
    """Require identical held-out observations and folds before pooling repeats.

    Comparing a five-repeat average with a single-repeat prediction would mix
    model quality with variance reduction. Fail rather than silently taking the
    intersection or averaging an incomplete prediction ledger.
    """
    subsets = []
    for predictions in (candidate, baseline):
        mask = (predictions.split.eq("source_holdout") if view == "source_holdout"
                else predictions.split.str.startswith("repeat"))
        frame = predictions.loc[mask].sort_values(["split", "row"]).reset_index(drop=True)
        if frame.empty or frame.duplicated(["split", "row"]).any():
            raise ValueError("Missing or duplicate evaluation observations")
        if not np.isfinite(frame[["y", "prediction"]].to_numpy()).all():
            raise ValueError("Nonfinite evaluation observations")
        if (frame.groupby("row")[["y", "group"]].nunique() > 1).any().any():
            raise ValueError("Inconsistent target or group across repeats")
        if view == "cv":
            frame["repeat"] = frame.split.str.extract(r"^(repeat\d+)_fold\d+$", expand=False)
            if frame["repeat"].isna().any() or frame.duplicated(["repeat", "row"]).any():
                raise ValueError("Invalid fold name or duplicate row within CV repeat")
            if (frame.groupby(["repeat", "group"]).split.nunique() > 1).any():
                raise ValueError("Chemical group crosses evaluation folds within a repeat")
            if frame.groupby("row")["repeat"].nunique().ne(frame["repeat"].nunique()).any():
                raise ValueError("Incomplete observations across CV repeats")
        subsets.append(frame)
    candidate, baseline = subsets
    if (not candidate[["split", "row", "group"]].equals(baseline[["split", "row", "group"]])
            or not np.array_equal(candidate.y.to_numpy(), baseline.y.to_numpy())):
        raise ValueError("Noncomparable evaluation rows, labels, groups or split assignments")
    return tuple(frame.groupby("row", sort=True).agg(
        group=("group", "first"), y=("y", "first"), prediction=("prediction", "mean"))
        for frame in subsets)


def read_candidates(roots):
    """Load completed candidates; keep unfinished work visible in the report."""
    candidates, incomplete = {}, []
    for root in roots:
        for directory in sorted(root.iterdir()):
            if not directory.is_dir() or not (directory / "config.json").exists():
                continue
            if not (directory / "COMPLETE").exists():
                incomplete.append(str(directory))
                continue
            config = json.loads((directory / "config.json").read_text())
            pred = pd.read_csv(directory / "predictions.csv")
            expected = set(config.get("evaluated_splits", [
                f"repeat{r}_fold{f}" for r in range(config.get("repeats", 1))
                for f in range(config.get("folds", 5))] + ["source_holdout"]))
            if set(pred.split) != expected:
                raise ValueError(f"Completed run has missing/unexpected splits: {directory}")
            candidates[str(directory)] = (config, pred)
    return candidates, incomplete


def paired_interval(y, baseline, candidate, groups, classification, draws=1000):
    """Positive values mean candidate improvement; resample whole groups together."""
    unique, index = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(20261030)
    improvements = []
    for _ in range(draws):
        counts = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        weights = counts[index]
        if classification:
            if np.sum(weights * y) == 0 or np.sum(weights * (1 - y)) == 0:
                continue
            delta = (average_precision_score(y, candidate, sample_weight=weights)
                     - average_precision_score(y, baseline, sample_weight=weights))
        else:
            delta = (np.sqrt(mean_squared_error(y, baseline, sample_weight=weights))
                     - np.sqrt(mean_squared_error(y, candidate, sample_weight=weights)))
        improvements.append(float(delta))
    if not improvements:
        return {"low": None, "high": None, "valid_draws": 0}
    low, high = np.quantile(improvements, [.025, .975])
    return {"low": float(low), "high": float(high), "valid_draws": len(improvements)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--reference-root", type=Path,
                        help="Explicit completed comparator runs (one per task/snapshot).")
    args = parser.parse_args()
    if args.draws < 1:
        parser.error("--draws must be positive")
    candidates, incomplete = read_candidates(args.roots)
    if args.reference_root:
        references, unfinished_references = read_candidates([args.reference_root])
        if unfinished_references:
            raise ValueError(f"Reference runs incomplete: {unfinished_references}")
    else:
        references = {path: (c, p) for path, (c, p) in candidates.items()
                      if c["model"] == "xgb" and c["feature"] == "baseline"
                      and c["policy"] == "compat" and c["ablation"] == "all"}
    results, publication_cache = [], {}
    for path, (config, pred) in candidates.items():
        task = config["task"]
        if config["snapshot"] not in publication_cache:
            publication_cache[config["snapshot"]] = pd.read_csv(
                Path(config["snapshot"]) / "rows.csv", usecols=["publication"])
        publications = publication_cache[config["snapshot"]]
        reference = [(path, c, p) for path, (c, p) in references.items() if c["task"] == task
                     and c["snapshot_hash"] == config["snapshot_hash"]
                     and c.get("eval_policy", "exact") == config.get("eval_policy", "exact")]
        if len(reference) != 1:
            raise ValueError(f"Need exactly one compatible reference for {path}; found {len(reference)}")
        for view in ("cv", "source_holdout"):
            summary, base = paired_view(pred, reference[0][2], view)
            if view == "source_holdout":
                publication_ids = publications.iloc[summary.index].publication
                if publication_ids.isna().any():
                    raise ValueError("Source holdout contains unknown publication identity")
                summary["group"] = publication_ids.to_numpy()
            row = {"path": path, "task": task, "model": config["model"],
                   "feature": config["feature"], "policy": config["policy"], "view": view,
                   "n": len(summary), "groups": summary.group.nunique(),
                   "resampling_unit": "publication" if view == "source_holdout" else "chemical_component",
                   **scores(summary.y.to_numpy(), summary.prediction.to_numpy(), task == "activity")}
            if config.get("score_note"):
                row["score_note"] = config["score_note"]
            row["reference_path"] = reference[0][0]
            row["reference_metrics"] = scores(base.y.to_numpy(), base.prediction.to_numpy(), task == "activity")
            key = "ap" if task == "activity" else "rmse"
            row["improvement"] = (None if row[key] is None or row["reference_metrics"][key] is None
                                  else (row[key] - row["reference_metrics"][key]) if task == "activity"
                                  else row["reference_metrics"][key] - row[key])
            row["improvement_ci"] = paired_interval(
                summary.y.to_numpy(), base.prediction.to_numpy(), summary.prediction.to_numpy(),
                summary.group.to_numpy(), task == "activity", args.draws)
            results.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"results": results, "incomplete": incomplete,
                                      "bootstrap_draws": args.draws, "bootstrap_seed": 20261030,
                                      "reference_root": str(args.reference_root) if args.reference_root else None,
                                      "note": "Development-only comparisons; confirmation/test are not evaluated."},
                                     indent=2, allow_nan=False) + "\n")
    print(f"Summarized {len(candidates)} completed candidates; {len(incomplete)} incomplete")


if __name__ == "__main__":
    main()
