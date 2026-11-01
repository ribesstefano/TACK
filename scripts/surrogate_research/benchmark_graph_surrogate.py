"""Run one bounded Chemprop D-MPNN fold and/or source stress evaluation."""
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
from tackai.fusion.graph import GraphPredictor, TASK_ORDER
from tackai.fusion.research import read_npz_rows, sha256, write_once_json
try:
    from .benchmark_surrogates import scores
except ImportError:
    from benchmark_surrogates import scores


def _load_snapshot(snapshot, feature, policy):
    manifest = json.loads((snapshot / "manifest.json").read_text())
    for name in [f"{feature}.npy", "labels.npz", "groups.npy", "partition.npy",
                 "source_partition.npy", "rows.csv"]:
        if sha256(snapshot / name) != manifest["artifact_hashes"][name]:
            raise ValueError(f"Snapshot hash mismatch: {name}")
    matrix = np.load(snapshot / f"{feature}.npy", mmap_mode="r")
    dims = manifest["features"][feature]["dims"]
    fp_width = dims["fingerprint"]
    # Retain the already frozen descriptors and context vectors byte-for-byte.
    side = np.asarray(matrix[:, fp_width:], dtype=np.float32)
    expected = sum(dims[k] for k in ("descriptors", "e3", "cell", "poi", "assay", "assay_time"))
    if side.shape[1] != expected:
        raise ValueError("Feature manifest does not match descriptor/context matrix width")
    partition = np.load(snapshot / "partition.npy", allow_pickle=False)
    source = np.load(snapshot / "source_partition.npy", allow_pickle=False)
    if np.any(np.isin(source, ["train", "eval"]) & (partition != "search")):
        raise ValueError("Source stress rows must belong to search")
    search = np.flatnonzero(partition == "search")
    targets = np.full((len(partition), len(TASK_ORDER)), np.nan)
    for i, task in enumerate(TASK_ORDER):
        targets[search, i] = read_npz_rows(snapshot / "labels.npz", f"{policy}_{task}", search)
    rows = pd.read_csv(snapshot / "rows.csv", usecols=["canonical_smiles"])
    return manifest, side, targets, rows.canonical_smiles.to_numpy(dtype=str)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature", default="baseline")
    parser.add_argument("--policy", choices=["exact"], default="exact")
    parser.add_argument("--evaluated-splits", nargs="+", choices=["fold0", "source_holdout"],
                        default=["fold0"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if len(set(args.evaluated_splits)) != len(args.evaluated_splits):
        parser.error("--evaluated-splits cannot contain duplicates")
    if args.output.exists():
        raise FileExistsError(f"Use a fresh run directory: {args.output}")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    manifest, side, targets, smiles = _load_snapshot(args.snapshot, args.feature, args.policy)
    groups = np.load(args.snapshot / "groups.npy", allow_pickle=False)
    partition = np.load(args.snapshot / "partition.npy", allow_pickle=False)
    source = np.load(args.snapshot / "source_partition.npy", allow_pickle=False)
    search = np.flatnonzero(partition == "search")
    folds = make_splits(np.zeros(len(search)), groups[search], 1, 5)
    split_indices = {}
    train, validation = folds[0][0]
    split_indices["fold0"] = (search[train], search[validation])
    split_indices["source_holdout"] = (np.flatnonzero(source == "train"),
                                        np.flatnonzero(source == "eval"))
    dims = manifest["features"][args.feature]["dims"]
    side_dims = {name: dims[name] for name in
                 ("descriptors", "e3", "cell", "poi", "assay", "assay_time")}

    args.output.mkdir(parents=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(model="chemprop_dmpnn", snapshot_hash=sha256(args.snapshot / "manifest.json"),
                  runner_hash=sha256(__file__), model_hash=sha256("tackai/fusion/graph.py"),
                  evaluated_splits=args.evaluated_splits,
                  comparison_scope="partial pilot; not full-CV comparable",
                  molecular_input="Chemprop 2.3 directed-bond graph from isomeric canonical_smiles",
                  side_input="baseline descriptors plus frozen context columns; fingerprint removed",
                  versions={name: importlib.metadata.version(name)
                            for name in ["chemprop", "torch", "numpy", "scikit-learn"]})
    write_once_json(args.output / "config.json", config)
    predictions, metrics = {task: [] for task in TASK_ORDER}, {task: [] for task in TASK_ORDER}
    for fold_number, split in enumerate(args.evaluated_splits):
        train, evaluation = split_indices[split]
        eligible = np.isfinite(targets[train]).any(axis=1)
        train = train[eligible]
        if set(groups[train]) & set(groups[evaluation]):
            raise ValueError(f"Group leakage in {split}")
        inner = GroupShuffleSplit(n_splits=1, test_size=.15,
                                  random_state=20261030 + fold_number)
        fit_rows, early_stop_rows = next(inner.split(train, groups=groups[train]))
        inner_fit, inner_val = train[fit_rows], train[early_stop_rows]
        if set(groups[inner_fit]) & set(groups[inner_val]):
            raise ValueError("Early-stopping groups overlap training groups")
        parameters = dict(side_dim=side.shape[1], width=args.width, depth=args.depth,
                          dropout=args.dropout, seed=args.seed + fold_number,
                          batch_size=args.batch_size, device=args.device)
        start = time.monotonic()
        pilot = GraphPredictor(**parameters).fit(
            smiles[inner_fit], side[inner_fit], targets[inner_fit], epochs=args.epochs,
            validation=(smiles[inner_val], side[inner_val], targets[inner_val]))
        selected_epochs = pilot.epochs_
        del pilot
        model = GraphPredictor(**parameters).fit(smiles[train], side[train], targets[train],
                                                  epochs=selected_epochs)
        member = args.output / "members" / split
        member.parent.mkdir(parents=True, exist_ok=True)
        model.save(member, metadata={"split": split, "snapshot_hash": config["snapshot_hash"],
                                     "source_feature": args.feature,
                                     "frozen_context_fingerprint": manifest["context_fingerprint"],
                                     "side_feature_blocks": list(side_dims),
                                     "side_feature_dims": side_dims,
                                     "side_feature_width": side.shape[1]})
        prediction = model.predict(smiles[evaluation], side[evaluation])
        seconds = time.monotonic() - start
        for index, task in enumerate(TASK_ORDER):
            observed = np.isfinite(targets[evaluation, index])
            rows = evaluation[observed]
            values = prediction[task][observed]
            row = {"split": split, "epochs": selected_epochs,
                   "n_train_union": len(train),
                   "n_train": int(np.isfinite(targets[train, index]).sum()),
                   "n_eval": len(rows), "seconds": seconds,
                   **scores(targets[rows, index], values, task == "activity")}
            metrics[task].append(row)
            predictions[task].append(pd.DataFrame({"row": rows, "group": groups[rows],
                                                   "split": split, "y": targets[rows, index],
                                                   "prediction": values}))
            directory = args.output / task
            directory.mkdir(exist_ok=True)
            pd.concat(predictions[task], ignore_index=True).to_csv(
                directory / "predictions.csv", index=False)
            (directory / "metrics.json").write_text(
                json.dumps(metrics[task], indent=2, allow_nan=False))
            print(json.dumps({"task": task, **row}), flush=True)
    (args.output / "COMPLETE").write_text("Requested partial pilot splits completed.\n")


if __name__ == "__main__":
    main()
