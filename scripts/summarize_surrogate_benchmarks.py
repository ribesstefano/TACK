"""Summarize completed development runs with paired chemical-group bootstrap.

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

from benchmark_surrogates import scores


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
    args = parser.parse_args()
    candidates, incomplete = {}, []
    for root in args.roots:
        for directory in sorted(root.iterdir()):
            if not directory.is_dir() or not (directory / "config.json").exists():
                continue
            if not (directory / "COMPLETE").exists():
                incomplete.append(str(directory))
                continue
            config = json.loads((directory / "config.json").read_text())
            pred = pd.read_csv(directory / "predictions.csv")
            candidates[str(directory)] = (config, pred)
    results = []
    for path, (config, pred) in candidates.items():
        task = config["task"]
        reference = [(c, p) for c, p in candidates.values() if c["task"] == task
                     and c["model"] == "xgb" and c["feature"] == "baseline"
                     and c["policy"] == "compat" and c["ablation"] == "all"
                     and c["snapshot_hash"] == config["snapshot_hash"]]
        for view in ("cv", "source_holdout"):
            subset = pred[pred.split.eq("source_holdout") if view == "source_holdout"
                          else pred.split.str.startswith("repeat")]
            # For repeated CV, one mean OOF prediction per row. No member trained
            # on that row contributes to its prediction in any repeat.
            summary = subset.groupby("row", sort=True).agg(
                group=("group", "first"), y=("y", "first"), prediction=("prediction", "mean"))
            row = {"path": path, "task": task, "model": config["model"],
                   "feature": config["feature"], "policy": config["policy"], "view": view,
                   "n": len(summary), "groups": summary.group.nunique(),
                   **scores(summary.y.to_numpy(), summary.prediction.to_numpy(), task == "activity")}
            if len(reference) == 1:
                base = reference[0][1]
                base = base[base.split.eq("source_holdout") if view == "source_holdout"
                            else base.split.str.startswith("repeat")]
                base = base.groupby("row", sort=True).agg(y=("y", "first"),
                                                          prediction=("prediction", "mean"))
                if not base.index.equals(summary.index) or not np.allclose(base.y, summary.y):
                    raise ValueError(f"Noncomparable eligible observations: {path}")
                row["improvement_ci"] = paired_interval(
                    summary.y.to_numpy(), base.prediction.to_numpy(), summary.prediction.to_numpy(),
                    summary.group.to_numpy(), task == "activity", args.draws)
            results.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"results": results, "incomplete": incomplete,
                                      "note": "Development-only fixed-parameter pilot; not final model selection."},
                                     indent=2, allow_nan=False) + "\n")
    print(f"Summarized {len(candidates)} completed candidates; {len(incomplete)} incomplete")


if __name__ == "__main__":
    main()
