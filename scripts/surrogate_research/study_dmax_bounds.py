"""Study fixed [0,1] clipping of pooled Dmax development predictions.

This is an exploratory postprocessing comparison only: calibration and serving
must be rebuilt and revalidated if the prediction rule is ever changed.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tackai.fusion.research import read_npz_rows, sha256

try:
    from .benchmark_surrogates import scores
    from .summarize_surrogate_benchmarks import paired_interval, paired_view
except ImportError:
    from benchmark_surrogates import scores
    from summarize_surrogate_benchmarks import paired_interval, paired_view


def check_ledger(snapshot, config, predictions):
    """Validate exact Dmax ledger against frozen search-only snapshot labels."""
    manifest_path = snapshot / "manifest.json"
    if sha256(manifest_path) != config.get("snapshot_hash"):
        raise ValueError("Snapshot manifest mismatch")
    if config.get("task") != "dmax" or config.get("eval_policy", "exact") != "exact":
        raise ValueError("Only exact-label Dmax evaluation is supported")
    manifest = json.loads(manifest_path.read_text())
    for name in ("partition.npy", "source_partition.npy", "groups.npy", "labels.npz", "rows.csv"):
        if sha256(snapshot / name) != manifest["artifact_hashes"][name]:
            raise ValueError(f"Snapshot artifact mismatch: {name}")
    roles = np.load(snapshot / "partition.npy", allow_pickle=False)
    source = np.load(snapshot / "source_partition.npy", allow_pickle=False)
    groups = np.load(snapshot / "groups.npy", allow_pickle=False)
    if (predictions[["split", "row", "group"]].isna().any().any()
            or predictions.duplicated(["split", "row"]).any()
            or not np.isfinite(predictions[["y", "prediction"]].to_numpy()).all()):
        raise ValueError("Invalid prediction ledger")
    expected = {f"repeat{r}_fold{f}" for r in range(config["repeats"])
                for f in range(config["folds"])} | {"source_holdout"}
    if set(predictions.split) != expected:
        raise ValueError("Incomplete ledger splits")
    rows = predictions.row.to_numpy()
    if (not np.issubdtype(rows.dtype, np.integer) or np.any(rows < 0)
            or np.any(rows >= len(roles)) or np.any(roles[rows] != "search")
            or np.any(groups[rows] != predictions.group.to_numpy())
            or np.any(source[rows[predictions.split.eq("source_holdout").to_numpy()]] != "eval")):
        raise ValueError("Ledger includes incorrect roles, groups or source rows")
    unique, inverse = np.unique(rows, return_inverse=True)
    labels = read_npz_rows(snapshot / "labels.npz", "exact_dmax", unique)[inverse]
    if not np.array_equal(labels, predictions.y.to_numpy()):
        raise ValueError("Ledger labels do not match frozen exact Dmax labels")
    if not np.all((labels >= 0) & (labels <= 1)):
        raise ValueError("Exact Dmax labels are outside [0,1]")
    return manifest


def study(runs, output, *, draws=1000):
    if draws < 1 or output.exists() or not runs:
        raise ValueError("Need runs, positive draws and a fresh output path")
    report = []
    for run in runs:
        if not (run / "COMPLETE").is_file():
            raise ValueError(f"Incomplete run: {run}")
        config = json.loads((run / "config.json").read_text())
        ledger = pd.read_csv(run / "predictions.csv", float_precision="round_trip")
        snapshot = Path(config["snapshot"])
        check_ledger(snapshot, config, ledger)
        publication = pd.read_csv(snapshot / "rows.csv", usecols=["publication"])["publication"]
        for view in ("cv", "source_holdout"):
            pooled, _ = paired_view(ledger, ledger, view)
            mask = ledger.split.eq("source_holdout") if view == "source_holdout" else ledger.split.str.startswith("repeat")
            members = ledger.loc[mask, "prediction"]
            y = pooled.y.to_numpy()
            raw = pooled.prediction.to_numpy()
            clipped = np.clip(raw, 0., 1.)
            groups = pooled.group.to_numpy() if view == "cv" else publication.iloc[pooled.index].to_numpy()
            if pd.isna(groups).any():
                raise ValueError("Missing bootstrap group identity")
            raw_metrics, clipped_metrics = scores(y, raw, False), scores(y, clipped, False)
            report.append({"run": str(run), "view": view, "rows": len(y),
                           "groups": len(np.unique(groups)),
                           "fold_prediction_outside_bounds": int(((members < 0) | (members > 1)).sum()),
                           "pooled_prediction_outside_bounds": int(((raw < 0) | (raw > 1)).sum()),
                           "unclipped": raw_metrics, "clipped": clipped_metrics,
                           "rmse_improvement": raw_metrics["rmse"] - clipped_metrics["rmse"],
                           "paired_rmse_improvement_ci95": paired_interval(y, raw, clipped, groups, False, draws),
                           "config_sha256": sha256(run / "config.json"),
                           "predictions_sha256": sha256(run / "predictions.csv"),
                           "snapshot_manifest_sha256": sha256(snapshot / "manifest.json")})
    result = {"format": "tack-dmax-bounds-study/v1", "bootstrap_draws": draws,
              "script_sha256": sha256(__file__), "results": report,
              "note": ("CV pools one out-of-fold prediction per repeat; the source view uses a single source-holdout model. "
                       "Neither reproduces the full 25-member deployment ensemble. Clipping is applied after pooling. "
                       "Development-only; no recalibration or serving change.")}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=1000)
    args = parser.parse_args()
    result = study(args.runs, args.output, draws=args.draws)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
