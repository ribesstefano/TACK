"""Development-only activity ranking from paired out-of-fold endpoint predictions.

The fixed bottleneck score tests whether two supervised regression heads can
rank joint activity. It is an uncalibrated ranking score, not a probability and
does not assume independence between endpoint errors. Each activity evaluation
group is excluded from both endpoint model fits for that split.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from tackai.fusion.data import make_splits
from tackai.fusion.research import read_npz_rows, sha256, write_once_json
try:
    from .benchmark_surrogates import fit_learner
    from .fit_surrogate_candidate import _estimator
except ImportError:
    from benchmark_surrogates import fit_learner
    from fit_surrogate_candidate import _estimator


def bottleneck_score(dmax, pdc50):
    """Fixed-scale score; 0.5 corresponds to the nearer endpoint threshold."""
    return expit(np.minimum((np.asarray(dmax) - .8) / .2, np.asarray(pdc50) - 6.))


def predict_endpoint(snapshot, run, output):
    """Refit each selected fold on search groups and score activity-label rows."""
    if output.exists():
        raise FileExistsError(f"Use a fresh output directory: {output}")
    if not (run / "COMPLETE").is_file():
        raise ValueError("Endpoint benchmark must be complete")
    config = json.loads((run / "config.json").read_text())
    manifest = json.loads((snapshot / "manifest.json").read_text())
    if sha256(snapshot / "manifest.json") != config["snapshot_hash"]:
        raise ValueError("Endpoint benchmark was run against a different snapshot")
    if (config["task"] not in {"dmax", "pdc50"} or config["policy"] != "exact"
            or config["ablation"] != "all" or config["model"] not in {"xgb", "extratrees", "catboost"}):
        raise ValueError("Need a complete-feature exact-label regression benchmark")
    names = [f"{config['feature']}.npy", "labels.npz", "groups.npy", "partition.npy", "source_partition.npy"]
    for name in names:
        if sha256(snapshot / name) != manifest["artifact_hashes"][name]:
            raise ValueError(f"Snapshot artifact changed: {name}")
    groups = np.load(snapshot / "groups.npy", allow_pickle=False)
    roles = np.load(snapshot / "partition.npy", allow_pickle=False)
    source = np.load(snapshot / "source_partition.npy", allow_pickle=False)
    if np.any(np.isin(source, ["train", "eval"]) & (roles != "search")):
        raise ValueError("Source stress rows must be search rows")
    search = np.flatnonzero(roles == "search")
    labels = snapshot / "labels.npz"
    y_train = np.full(len(roles), np.nan)
    y_activity = np.full(len(roles), np.nan)
    y_train[search] = read_npz_rows(labels, f"exact_{config['task']}", search)
    y_activity[search] = read_npz_rows(labels, "exact_activity", search)
    X = np.load(snapshot / f"{config['feature']}.npy", mmap_mode="r", allow_pickle=False)
    folds = make_splits(np.zeros(len(search)), groups[search], config["repeats"], config["folds"])
    splits = [(f"repeat{r}_fold{f}", search[train], search[val])
              for r, repeat in enumerate(folds) for f, (train, val) in enumerate(repeat)]
    splits.append(("source_holdout", np.flatnonzero(source == "train"), np.flatnonzero(source == "eval")))
    records = json.loads((run / "metrics.json").read_text())
    by_split = {row["split"]: row["params"] for row in records}
    if len(by_split) != len(records) or set(by_split) != {name for name, _, _ in splits}:
        raise ValueError("Benchmark fold parameters are incomplete or duplicated")
    output.mkdir(parents=True)
    metadata = {"task": config["task"], "snapshot": str(snapshot), "snapshot_hash": config["snapshot_hash"],
                "source_run": str(run), "source_config_sha256": sha256(run / "config.json"),
                "source_metrics_sha256": sha256(run / "metrics.json"),
                "runner_sha256": sha256(__file__),
                "fit_tool_sha256": sha256("scripts/fit_surrogate_candidate.py"),
                "benchmark_sha256": sha256("scripts/benchmark_surrogates.py"),
                "feature": config["feature"],
                "repeats": config["repeats"], "folds": config["folds"]}
    write_once_json(output / "config.json", metadata)
    predictions = []
    for number, (name, train, val) in enumerate(splits):
        train = train[np.isfinite(y_train[train])]
        val = val[np.isfinite(y_activity[val])]
        if set(groups[train]) & set(groups[val]) or not len(train) or not len(val):
            raise ValueError(f"Invalid or overlapping endpoint groups in {name}")
        params = by_split[name]
        member = {"family": config["model"], "seed": 20261030 + number, "params": params}
        estimator = _estimator(member, config["task"], jobs=4)
        fit_learner(estimator, config["model"], X[train], y_train[train], params, False)
        values = np.asarray(estimator.predict(X[val]), dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite endpoint predictions")
        predictions.append(pd.DataFrame({"row": val, "group": groups[val], "split": name,
                                         "y": y_activity[val], "prediction": values}))
        pd.concat(predictions, ignore_index=True).to_csv(output / "predictions.csv", index=False)
        print(f"{config['task']} {name}: trained={len(train)} evaluated={len(val)}", flush=True)
    (output / "COMPLETE").write_text("All endpoint activity-evaluation splits complete.\n")


def combine(dmax, pdc50, output):
    """Join identical out-of-fold activity rows and apply the fixed margin score."""
    if output.exists():
        raise FileExistsError(f"Use a fresh output directory: {output}")
    inputs = []
    for directory, task in ((dmax, "dmax"), (pdc50, "pdc50")):
        if not (directory / "COMPLETE").is_file():
            raise ValueError(f"Incomplete endpoint run: {directory}")
        config = json.loads((directory / "config.json").read_text())
        if config["task"] != task:
            raise ValueError("Endpoint runs have incorrect tasks")
        frame = pd.read_csv(directory / "predictions.csv").sort_values(["split", "row"]).reset_index(drop=True)
        inputs.append((config, frame))
    left, right = inputs
    if any(left[0][key] != right[0][key] for key in ("snapshot_hash", "repeats", "folds")):
        raise ValueError("Endpoint runs use different snapshots or fold schedules")
    if not left[1][["row", "group", "split", "y"]].equals(right[1][["row", "group", "split", "y"]]):
        raise ValueError("Endpoint predictions do not align")
    result = left[1].copy()
    result["prediction"] = bottleneck_score(left[1].prediction, right[1].prediction)
    output.mkdir(parents=True)
    config = {"task": "activity", "model": "endpoint_margin", "feature": "mixed_endpoints",
              "policy": "exact", "eval_policy": "exact", "ablation": "all",
              "snapshot": left[0]["snapshot"],
              "snapshot_hash": left[0]["snapshot_hash"], "repeats": left[0]["repeats"],
              "folds": left[0]["folds"], "dmax_run": str(dmax), "pdc50_run": str(pdc50),
              "score_note": "Uncalibrated bottleneck ranking score, not an activity probability."}
    write_once_json(output / "config.json", config)
    result.to_csv(output / "predictions.csv", index=False)
    (output / "COMPLETE").write_text("All fixed endpoint ranking splits complete.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    first = sub.add_parser("predict")
    first.add_argument("--snapshot", type=Path, required=True)
    first.add_argument("--run", type=Path, required=True)
    first.add_argument("--output", type=Path, required=True)
    second = sub.add_parser("combine")
    second.add_argument("--dmax", type=Path, required=True)
    second.add_argument("--pdc50", type=Path, required=True)
    second.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "predict":
        predict_endpoint(args.snapshot, args.run, args.output)
    else:
        combine(args.dmax, args.pdc50, args.output)


if __name__ == "__main__":
    main()
