"""Compare fixed-parameter XGBoost using interval-censored Dmax supervision.

Only frozen search labels are read. The objective penalizes distance outside a
reported interval; exact observations are zero-width intervals. Evaluation uses
exact Dmax labels on the same grouped folds as other development pilots.
"""
import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from tackai.fusion.data import make_splits
from tackai.fusion.research import read_npz_rows, sha256, write_once_json

try:
    from .benchmark_surrogates import estimator, scores
except ImportError:
    from benchmark_surrogates import estimator, scores


def interval_gradient(prediction, bounds):
    """Squared-distance gradient, with a tiny Hessian floor for feasible rows."""
    prediction = np.asarray(prediction, dtype=float)
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (len(prediction), 2) or not np.isfinite(bounds).all():
        raise ValueError("Need finite aligned interval bounds")
    if np.any(bounds[:, 0] > bounds[:, 1]):
        raise ValueError("Interval bounds are reversed")
    lower, upper = bounds.T
    grad = np.where(prediction < lower, prediction - lower,
                    np.where(prediction > upper, prediction - upper, 0.))
    outside = (prediction < lower) | (prediction > upper) | (lower == upper)
    return grad, np.where(outside, 1., 1e-6)


def read_bound_rows(path, key, indices):
    """Read only requested rows of a numeric (N, 2) bound array from NPZ."""
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("Indices must be one-dimensional")
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(f"{key}.npy")
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError("Bound array must be uncompressed")
        with archive.open(info) as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError("Unsupported NPY version")
            if len(shape) != 2 or shape[1] != 2 or fortran or dtype.kind not in "fiu":
                raise ValueError("Expected a numeric (N, 2) bound array")
            if np.any(indices < 0) or np.any(indices >= shape[0]):
                raise ValueError("Bound indices are out of range")
            offset = stream.tell()
            values = np.empty((len(indices), 2), dtype=dtype)
            for i, row in enumerate(indices):
                stream.seek(offset + int(row) * 2 * dtype.itemsize)
                raw = stream.read(2 * dtype.itemsize)
                if len(raw) != 2 * dtype.itemsize:
                    raise ValueError("Truncated bound array")
                values[i] = np.frombuffer(raw, dtype=dtype, count=2)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Use a fresh output directory: {args.output}")
    manifest = json.loads((args.snapshot / "manifest.json").read_text())
    for name in ("baseline.npy", "labels.npz", "groups.npy", "partition.npy", "source_partition.npy"):
        if sha256(args.snapshot / name) != manifest["artifact_hashes"][name]:
            raise ValueError(f"Snapshot hash mismatch: {name}")
    X = np.load(args.snapshot / "baseline.npy", mmap_mode="r", allow_pickle=False)
    groups = np.load(args.snapshot / "groups.npy", allow_pickle=False)
    roles = np.load(args.snapshot / "partition.npy", allow_pickle=False)
    source = np.load(args.snapshot / "source_partition.npy", allow_pickle=False)
    if np.any(np.isin(source, ["train", "eval"]) & (roles != "search")):
        raise ValueError("Source stress rows must be search rows")
    search = np.flatnonzero(roles == "search")
    labels = args.snapshot / "labels.npz"
    bounds = np.full((len(roles), 2), np.nan)
    y_eval = np.full(len(roles), np.nan)
    bounds[search] = read_bound_rows(labels, "censored_dmax_bounds", search)
    y_eval[search] = read_npz_rows(labels, "exact_dmax", search)
    shared = make_splits(np.zeros(len(search)), groups[search], 1, 5)
    splits = [(f"repeat0_fold{f}", search[train], search[val])
              for f, (train, val) in enumerate(shared[0])]
    splits.append(("source_holdout", np.flatnonzero(source == "train"), np.flatnonzero(source == "eval")))
    args.output.mkdir(parents=True)
    config = {"task": "dmax", "model": "xgb_interval", "feature": "baseline",
              "policy": "censored", "eval_policy": "exact", "ablation": "all",
              "snapshot": str(args.snapshot), "snapshot_hash": sha256(args.snapshot / "manifest.json"),
              "runner_hash": sha256(__file__), "repeats": 1, "folds": 5,
              "note": "Fixed XGB parameters; squared distance outside censored bounds."}
    write_once_json(args.output / "config.json", config)
    predictions, metrics = [], []
    for number, (name, train, val) in enumerate(splits):
        train = train[np.isfinite(bounds[train]).all(axis=1)]
        val = val[np.isfinite(y_eval[val])]
        if not len(train) or not len(val) or set(groups[train]) & set(groups[val]):
            raise ValueError(f"Invalid group split: {name}")
        interval = bounds[train]
        model = estimator("xgb", False, {}, args.jobs, 20261030 + number)
        model.set_params(objective=lambda _y, pred: interval_gradient(pred, interval))
        model.fit(X[train], interval.mean(axis=1))
        pred = model.predict(X[val])
        if not np.isfinite(pred).all():
            raise ValueError("Nonfinite predictions")
        predictions.append(pd.DataFrame({"row": val, "group": groups[val], "split": name,
                                         "y": y_eval[val], "prediction": pred}))
        metrics.append({"split": name, "n_train": len(train), "n_eval": len(val),
                        **scores(y_eval[val], pred, False)})
        pd.concat(predictions, ignore_index=True).to_csv(args.output / "predictions.csv", index=False)
        (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
        print(json.dumps(metrics[-1]), flush=True)
    (args.output / "COMPLETE").write_text("All interval-label development folds completed.\n")


if __name__ == "__main__":
    main()
