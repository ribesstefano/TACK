"""Compare shared versus single-task interaction MLPs on frozen development folds."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

from tackai.fusion.data import make_splits
from tackai.fusion.neural import NeuralPredictor, TASK_ORDER
from tackai.fusion.research import sha256, write_once_json
from benchmark_surrogates import scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature", default="baseline")
    parser.add_argument("--mode", choices=["joint", "single"], default="joint")
    parser.add_argument("--policy", choices=["exact", "censored"], default="exact")
    parser.add_argument("--task", choices=TASK_ORDER)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=.1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.mode == "single" and args.task is None:
        parser.error("--task is required for single-task mode")
    if args.output.exists():
        raise FileExistsError(f"Use a fresh run directory: {args.output}")
    torch.set_num_threads(1)
    manifest = json.loads((args.snapshot / "manifest.json").read_text())
    for name in [f"{args.feature}.npy", "labels.npz", "groups.npy", "partition.npy", "source_partition.npy"]:
        if sha256(args.snapshot / name) != manifest["artifact_hashes"][name]:
            raise ValueError(f"Snapshot hash mismatch: {name}")
    X = np.load(args.snapshot / f"{args.feature}.npy", mmap_mode="r")
    with np.load(args.snapshot / "labels.npz") as labels:
        y = np.column_stack([labels[f"{args.policy}_{task}"] for task in TASK_ORDER])
        yeval = np.column_stack([labels[f"exact_{task}"] for task in TASK_ORDER])
        bounds = (np.stack([labels[f"censored_{task}_bounds"] for task in TASK_ORDER[:2]], axis=1)
                  if args.policy == "censored" else None)
    active_tasks = TASK_ORDER if args.mode == "joint" else (args.task,)
    for i, task in enumerate(TASK_ORDER):
        if task not in active_tasks:
            y[:, i] = np.nan
            if bounds is not None and i < 2:
                bounds[:, i] = np.nan
    groups = np.load(args.snapshot / "groups.npy")
    roles = np.load(args.snapshot / "partition.npy")
    source = np.load(args.snapshot / "source_partition.npy")
    search = np.flatnonzero(roles == "search")
    shared = make_splits(np.zeros(len(search)), groups[search], 1, 5)
    folds = [(f"repeat0_fold{f}", search[tr], search[va]) for f, (tr, va) in enumerate(shared[0])]
    folds.append(("source_holdout", np.flatnonzero(source == "train"), np.flatnonzero(source == "eval")))
    dims = manifest["features"][args.feature]["dims"]
    molecular_width = dims["fingerprint"] + dims["descriptors"]
    args.output.mkdir(parents=True)
    predictions, metrics = {}, {}
    for task in active_tasks:
        directory = args.output / task
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        config.update(task=task, model=f"{args.mode}_mlp", policy=args.policy, ablation="all",
                      snapshot_hash=sha256(args.snapshot / "manifest.json"),
                      runner_hash=sha256(__file__), model_hash=sha256("tackai/fusion/neural.py"),
                      versions={name: importlib.metadata.version(name) for name in
                                ["torch", "numpy", "scikit-learn"]})
        write_once_json(directory / "config.json", config)
        predictions[task], metrics[task] = [], []
    for fold_number, (name, train, validation) in enumerate(folds):
        eligible = np.isfinite(y[train]).any(axis=1)
        if bounds is not None:
            eligible |= np.isfinite(bounds[train]).any(axis=(1, 2))
        train = train[eligible]
        if set(groups[train]) & set(groups[validation]):
            raise ValueError("Outer/source group leakage")
        inner = GroupShuffleSplit(n_splits=1, test_size=.15, random_state=20261030 + fold_number)
        inner_fit, inner_val = next(inner.split(train, groups=groups[train]))
        start = time.monotonic()
        parameters = dict(molecular_width=molecular_width, width=args.width,
                          dropout=args.dropout, seed=args.seed + fold_number, device=args.device)
        pilot = NeuralPredictor(**parameters).fit(
            X[train[inner_fit]], y[train[inner_fit]], epochs=args.epochs,
            bounds=None if bounds is None else bounds[train[inner_fit]],
            validation=(X[train[inner_val]], y[train[inner_val]],
                        None if bounds is None else bounds[train[inner_val]]))
        epochs = pilot.epochs_
        del pilot
        model = NeuralPredictor(**parameters).fit(X[train], y[train], epochs=epochs,
                                                  bounds=None if bounds is None else bounds[train])
        model.save(args.output / "members" / name)
        prediction = model.predict(X[validation])
        seconds = time.monotonic() - start
        for i, task in enumerate(TASK_ORDER):
            if task not in active_tasks:
                continue
            observed = np.isfinite(yeval[validation, i])
            indices = validation[observed]
            values = prediction[task][observed]
            row = {"split": name, "epochs": epochs, "n_train_union": len(train),
                   "n_train": int(np.isfinite(y[train, i]).sum()), "n_eval": len(indices),
                   "seconds": seconds, **scores(yeval[indices, i], values, task == "activity")}
            metrics[task].append(row)
            predictions[task].append(pd.DataFrame({"row": indices, "group": groups[indices],
                                                  "split": name, "y": yeval[indices, i],
                                                  "prediction": values}))
            directory = args.output / task
            pd.concat(predictions[task], ignore_index=True).to_csv(directory / "predictions.csv", index=False)
            (directory / "metrics.json").write_text(json.dumps(metrics[task], indent=2, allow_nan=False))
            print(json.dumps({"task": task, **row}), flush=True)
    for task in active_tasks:
        (args.output / task / "COMPLETE").write_text("All development folds completed.\n")


if __name__ == "__main__":
    main()
