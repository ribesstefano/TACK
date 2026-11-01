"""Report paired development errors by nearest training-molecule similarity.

Similarity uses chirality-aware radius-2 Morgan fingerprints only for the
diagnostic. For each fold, nearest neighbors come exclusively from that fold's
labeled training molecules. No model feature or fitted predictor is changed.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from tackai.fusion.data import make_splits
from tackai.fusion.research import read_npz_rows, sha256

try:
    from .benchmark_surrogates import scores
    from .summarize_surrogate_benchmarks import paired_interval, paired_view
except ImportError:
    from benchmark_surrogates import scores
    from summarize_surrogate_benchmarks import paired_interval, paired_view


def nearest_similarity(train_smiles, eval_smiles):
    """Return max Tanimoto to the explicitly supplied unique training molecules."""
    if not len(train_smiles):
        raise ValueError("Cannot measure similarity without training molecules")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048,
                                                          includeChirality=True)
    fingerprints = {}
    for smiles in set(train_smiles) | set(eval_smiles):
        molecule = Chem.MolFromSmiles(str(smiles))
        if molecule is None:
            raise ValueError("Snapshot contains an unparseable canonical molecule")
        fingerprints[smiles] = generator.GetFingerprint(molecule)
    training = [fingerprints[smiles] for smiles in set(train_smiles)]
    nearest = {smiles: max(DataStructs.BulkTanimotoSimilarity(fingerprints[smiles], training))
               for smiles in set(eval_smiles)}
    return np.asarray([nearest[smiles] for smiles in eval_smiles], dtype=float)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=1000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Use a fresh report path: {args.output}")
    configs, predictions = [], []
    for directory in (args.candidate, args.reference):
        if not (directory / "COMPLETE").is_file():
            raise ValueError(f"Incomplete benchmark: {directory}")
        configs.append(json.loads((directory / "config.json").read_text()))
        predictions.append(pd.read_csv(directory / "predictions.csv"))
    config = configs[0]
    for key in ("snapshot_hash", "task"):
        if config[key] != configs[1][key]:
            raise ValueError(f"Incompatible {key}")
    if config.get("eval_policy", "exact") != configs[1].get("eval_policy", "exact"):
        raise ValueError("Incompatible evaluation policies")
    snapshot = Path(config["snapshot"])
    if sha256(snapshot / "manifest.json") != config["snapshot_hash"]:
        raise ValueError("Snapshot manifest changed")
    manifest = json.loads((snapshot / "manifest.json").read_text())
    for name in ("rows.csv", "labels.npz", "groups.npy", "partition.npy", "source_partition.npy"):
        if sha256(snapshot / name) != manifest["artifact_hashes"][name]:
            raise ValueError(f"Snapshot artifact changed: {name}")
    rows = pd.read_csv(snapshot / "rows.csv", usecols=["canonical_smiles", "publication"])
    groups = np.load(snapshot / "groups.npy", allow_pickle=False)
    roles = np.load(snapshot / "partition.npy", allow_pickle=False)
    source = np.load(snapshot / "source_partition.npy", allow_pickle=False)
    if np.any(np.isin(source, ["train", "eval"]) & (roles != "search")):
        raise ValueError("Source split contains non-search rows")
    search = np.flatnonzero(roles == "search")
    labels = np.full(len(roles), np.nan)
    labels[search] = read_npz_rows(snapshot / "labels.npz", f"{config['policy']}_{config['task']}", search)
    splits = make_splits(np.zeros(len(search)), groups[search], config["repeats"], config["folds"])
    training = {f"repeat{r}_fold{f}": search[tr] for r, repeat in enumerate(splits)
                for f, (tr, _) in enumerate(repeat)}
    training["source_holdout"] = np.flatnonzero(source == "train")
    distances = []
    for split, frame in predictions[0].groupby("split", sort=True):
        if split not in training:
            raise ValueError(f"Unexpected split: {split}")
        train = training[split]
        train = train[np.isfinite(labels[train])]
        evaluation = frame.row.to_numpy(dtype=int)
        if set(groups[train]) & set(groups[evaluation]):
            raise ValueError(f"Training groups overlap evaluation groups: {split}")
        similarity = nearest_similarity(rows.canonical_smiles.iloc[train].to_numpy(),
                                        rows.canonical_smiles.iloc[evaluation].to_numpy())
        distances.append(pd.DataFrame({"row": evaluation, "split": split, "similarity": similarity}))
        print(f"Computed similarity for {split}: {len(evaluation)} rows", flush=True)
    distances = pd.concat(distances, ignore_index=True)
    reports = {}
    for view in ("cv", "source_holdout"):
        candidate, baseline = paired_view(*predictions, view)
        mask = distances.split.eq("source_holdout") if view == "source_holdout" else distances.split.str.startswith("repeat")
        similarity = distances[mask].groupby("row").similarity.mean().reindex(candidate.index)
        if similarity.isna().any():
            raise ValueError("Similarity values do not align with prediction rows")
        bins = pd.cut(similarity, [-.001, .3, .5, .7, .85, 1.],
                      labels=["0–0.3", ">0.3–0.5", ">0.5–0.7", ">0.7–0.85", ">0.85–1"])
        output = []
        for label in bins.cat.categories:
            index = bins.index[bins.eq(label)]
            if len(index) < 2:
                continue
            y = candidate.loc[index, "y"].to_numpy()
            candidate_pred = candidate.loc[index, "prediction"].to_numpy()
            baseline_pred = baseline.loc[index, "prediction"].to_numpy()
            units = (rows.publication.iloc[index].to_numpy() if view == "source_holdout"
                     else candidate.loc[index, "group"].to_numpy())
            if pd.isna(units).any():
                raise ValueError("Missing publication identity in source evaluation")
            output.append({"similarity": str(label), "n": len(index),
                           "groups": int(candidate.loc[index, "group"].nunique()),
                           "bootstrap_units": len(set(units)),
                           "small_slice": len(index) < 30 or len(set(units)) < 5,
                           "candidate": scores(y, candidate_pred, config["task"] == "activity"),
                           "reference": scores(y, baseline_pred, config["task"] == "activity"),
                           "improvement_ci": paired_interval(y, baseline_pred, candidate_pred,
                                                             units, config["task"] == "activity", args.draws)})
        reports[view] = output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"candidate": str(args.candidate), "reference": str(args.reference),
        "snapshot_hash": config["snapshot_hash"], "fold_results": reports,
        "note": "Nearest candidate-training molecule; fixed predictions; uncorrected descriptive slices."},
        indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
