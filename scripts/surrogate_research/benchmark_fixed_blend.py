"""Combine completed development-fold predictions with fixed uniform weights."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tackai.fusion.research import read_npz_rows, sha256, write_once_json

try:
    from .summarize_surrogate_benchmarks import paired_view
except ImportError:
    from summarize_surrogate_benchmarks import paired_view


def blend_runs(runs, output):
    if len(runs) < 2 or len({p.resolve() for p in runs}) != len(runs):
        raise ValueError("At least two distinct completed runs are required")
    if output.exists():
        raise FileExistsError(f"Use a fresh output directory: {output}")
    configs, predictions = [], []
    for run in runs:
        if not (run / "COMPLETE").is_file():
            raise ValueError(f"Incomplete run: {run}")
        config = json.loads((run / "config.json").read_text())
        pred = pd.read_csv(run / "predictions.csv").sort_values(["split", "row"]).reset_index(drop=True)
        if (pred[["row", "group", "split"]].isna().any().any()
                or pred.duplicated(["split", "row"]).any()
                or not np.isfinite(pred[["y", "prediction"]].to_numpy()).all()):
            raise ValueError(f"Invalid prediction ledger: {run}")
        if config["task"] == "activity" and (not pred.prediction.between(0, 1).all()
                                              or not pred.y.isin([0, 1]).all()):
            raise ValueError(f"Invalid activity labels or probabilities: {run}")
        expected = {f"repeat{r}_fold{f}" for r in range(config["repeats"]) for f in range(config["folds"])} | {"source_holdout"}
        if set(pred.split) != expected:
            raise ValueError(f"Incomplete split ledger: {run}")
        configs.append(config)
        predictions.append(pred)
    first, base = configs[0], predictions[0]
    snapshot = Path(first["snapshot"])
    if sha256(snapshot / "manifest.json") != first["snapshot_hash"]:
        raise ValueError("Snapshot manifest does not match input runs")
    manifest = json.loads((snapshot / "manifest.json").read_text())
    for name in ("groups.npy", "partition.npy", "source_partition.npy", "labels.npz"):
        if sha256(snapshot / name) != manifest["artifact_hashes"][name]:
            raise ValueError(f"Snapshot artifact mismatch: {name}")
    groups = np.load(snapshot / "groups.npy", allow_pickle=False)
    roles = np.load(snapshot / "partition.npy", allow_pickle=False)
    source_roles = np.load(snapshot / "source_partition.npy", allow_pickle=False)
    row = base.row.to_numpy()
    if (not np.issubdtype(row.dtype, np.integer) or np.any(row < 0) or np.any(row >= len(roles))
            or np.any(roles[row] != "search") or np.any(groups[row] != base.group.to_numpy())
            or np.any(source_roles[row[base.split.eq("source_holdout").to_numpy()]] != "eval")):
        raise ValueError("Ledger rows do not match development snapshot partitions or groups")
    unique_rows, inverse = np.unique(row, return_inverse=True)
    labels = read_npz_rows(snapshot / "labels.npz", f"exact_{first['task']}", unique_rows)[inverse]
    if not np.array_equal(labels, base.y.to_numpy()):
        raise ValueError("Ledger labels do not match frozen exact labels")
    for config, pred in zip(configs[1:], predictions[1:]):
        for key in ("task", "snapshot_hash", "eval_policy", "repeats", "folds"):
            if config.get(key) != first.get(key):
                raise ValueError(f"Incompatible {key}")
        if not base[["split", "row", "group", "y"]].equals(pred[["split", "row", "group", "y"]]):
            raise ValueError("Prediction ledger rows, groups or labels differ")
        for view in ("cv", "source_holdout"):
            paired_view(base, pred, view)
    output.mkdir(parents=True)
    result = base[["row", "group", "split", "y"]].copy()
    result["prediction"] = np.mean([p.prediction.to_numpy() for p in predictions], axis=0)
    config = {"task": first["task"], "model": "uniform_blend", "feature": "mixed",
              "policy": "fixed_blend", "eval_policy": first.get("eval_policy", "exact"),
              "ablation": "all", "snapshot": first["snapshot"], "snapshot_hash": first["snapshot_hash"],
              "repeats": first["repeats"], "folds": first["folds"],
              "weights": [1 / len(runs)] * len(runs), "runner_sha256": sha256(__file__),
              "inputs": [{"path": str(run), "config_sha256": sha256(run / "config.json"),
                          "predictions_sha256": sha256(run / "predictions.csv")} for run in runs],
              "note": "Fixed uniform prediction blend; development-only, no fitted weights."}
    write_once_json(output / "config.json", config)
    result.to_csv(output / "predictions.csv", index=False)
    (output / "COMPLETE").write_text("Completed fixed-weight development blend.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    blend_runs(args.runs, args.output)


if __name__ == "__main__":
    main()
