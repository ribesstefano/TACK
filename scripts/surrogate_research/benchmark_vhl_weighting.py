"""Development-only test of source-balanced activity training within VHL.

Source and E3 metadata affect training weights only; the model receives the same
frozen molecular and biological-context vectors. Each fold computes weights
using its own training rows and reuses that fold's exact-CatBoost settings.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tackai.fusion.data import make_splits
from tackai.fusion.research import read_npz_rows, sha256, write_once_json

try:
    from .benchmark_surrogates import estimator, scores
except ImportError:
    from benchmark_surrogates import estimator, scores


def vhl_source_weights(e3, source):
    """Equalize VHL source weights with bounded factors; other rows stay at one."""
    e3 = np.asarray(e3, dtype=str)
    source = np.asarray(source, dtype=str)
    if e3.shape != source.shape or e3.ndim != 1 or not len(e3):
        raise ValueError("Training E3 and source values must be nonempty and aligned")
    weights = np.ones(len(e3), dtype=float)
    mask = np.char.lower(np.char.strip(e3)) == "vhl"
    labels, counts = np.unique(source[mask], return_counts=True)
    if len(labels) > 1:
        for label, count in zip(labels, counts):
            factor = np.clip(mask.sum() / (len(labels) * count), .5, 2.)
            weights[mask & (source == label)] = factor
    return weights / weights.mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists() or not (args.run / "COMPLETE").is_file():
        raise ValueError("Need a completed source run and a fresh output directory")
    source_config = json.loads((args.run / "config.json").read_text())
    manifest = json.loads((args.snapshot / "manifest.json").read_text())
    if (source_config["task"] != "activity" or source_config["model"] != "catboost"
            or source_config["policy"] != "exact" or source_config["feature"] != "baseline"
            or source_config["snapshot_hash"] != sha256(args.snapshot / "manifest.json")):
        raise ValueError("Need the completed exact-label baseline-feature activity CatBoost run")
    for name in ("baseline.npy", "labels.npz", "groups.npy", "partition.npy", "source_partition.npy", "rows.csv"):
        if sha256(args.snapshot / name) != manifest["artifact_hashes"][name]:
            raise ValueError(f"Snapshot hash mismatch: {name}")
    metadata = pd.read_csv(args.snapshot / "rows.csv", usecols=["source", "e3_raw"]).fillna("")
    X = np.load(args.snapshot / "baseline.npy", mmap_mode="r", allow_pickle=False)
    groups = np.load(args.snapshot / "groups.npy", allow_pickle=False)
    roles = np.load(args.snapshot / "partition.npy", allow_pickle=False)
    source_roles = np.load(args.snapshot / "source_partition.npy", allow_pickle=False)
    if np.any(np.isin(source_roles, ["train", "eval"]) & (roles != "search")):
        raise ValueError("Source stress rows must belong to search")
    search = np.flatnonzero(roles == "search")
    y = np.full(len(roles), np.nan)
    y[search] = read_npz_rows(args.snapshot / "labels.npz", "exact_activity", search)
    repeats, folds = source_config["repeats"], source_config["folds"]
    shared = make_splits(np.zeros(len(search)), groups[search], repeats, folds)
    splits = [(f"repeat{r}_fold{f}", search[train], search[val])
              for r, repeat in enumerate(shared) for f, (train, val) in enumerate(repeat)]
    splits.append(("source_holdout", np.flatnonzero(source_roles == "train"),
                   np.flatnonzero(source_roles == "eval")))
    records = json.loads((args.run / "metrics.json").read_text())
    params_by_split = {row["split"]: row["params"] for row in records}
    if len(records) != len(params_by_split) or set(params_by_split) != {name for name, _, _ in splits}:
        raise ValueError("Source benchmark fold parameters are incomplete or duplicated")
    args.output.mkdir(parents=True)
    config = {"task": "activity", "model": "catboost_vhl_source_weighted", "feature": "baseline",
              "policy": "exact", "eval_policy": "exact", "ablation": "all",
              "snapshot": str(args.snapshot), "snapshot_hash": source_config["snapshot_hash"],
              "runner_hash": sha256(__file__), "source_run": str(args.run),
              "source_config_sha256": sha256(args.run / "config.json"),
              "source_metrics_sha256": sha256(args.run / "metrics.json"),
              "repeats": repeats, "folds": folds,
              "note": "Equal-source VHL sample weights from training rows only; factors clipped to [0.5, 2]."}
    write_once_json(args.output / "config.json", config)
    predictions, metrics = [], []
    for number, (name, train, val) in enumerate(splits):
        train, val = train[np.isfinite(y[train])], val[np.isfinite(y[val])]
        if not len(train) or not len(val) or set(groups[train]) & set(groups[val]):
            raise ValueError(f"Invalid grouped activity split: {name}")
        if len(np.unique(y[train])) != 2:
            raise ValueError(f"Single-class training split: {name}")
        weights = vhl_source_weights(metadata.e3_raw.to_numpy()[train], metadata.source.to_numpy()[train])
        model = estimator("catboost", True, params_by_split[name], args.jobs, 20261030 + number)
        model.fit(X[train], y[train], sample_weight=weights)
        pred = model.predict_proba(X[val])[:, 1]
        if not np.isfinite(pred).all():
            raise ValueError("Nonfinite activity predictions")
        predictions.append(pd.DataFrame({"row": val, "group": groups[val], "split": name,
                                         "y": y[val], "prediction": pred}))
        metrics.append({"split": name, "n_train": len(train), "n_eval": len(val),
                        "min_weight": float(weights.min()), "max_weight": float(weights.max()),
                        **scores(y[val], pred, True)})
        pd.concat(predictions, ignore_index=True).to_csv(args.output / "predictions.csv", index=False)
        (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
        print(json.dumps(metrics[-1]), flush=True)
    (args.output / "COMPLETE").write_text("All source-weighted development folds completed.\n")


if __name__ == "__main__":
    main()
