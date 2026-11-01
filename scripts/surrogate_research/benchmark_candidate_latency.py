"""Measure local batched encoding and inference for a trusted candidate bundle.

Only SMILES and fixed context features from search rows are used. This is a CPU
microbenchmark; it is not a live-server or downstream RL latency measurement.
"""
import argparse
import csv
import json
import resource
import time
from pathlib import Path

import numpy as np

from tackai.fusion.candidate_artifact import CandidateArtifact
from tackai.fusion.context import ContextEncoder
from tackai.fusion.data import FusionData
from tackai.fusion.molecular_spec import MolecularFeatureEncoder
from tackai.fusion.research import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists() or args.batch < 1 or args.repeats < 1:
        raise ValueError("Need a fresh output path and positive batch and repeats")
    manifest = json.loads((args.snapshot / "manifest.json").read_text())
    start = time.perf_counter()
    artifact = CandidateArtifact.from_pretrained(args.bundle)
    load_seconds = time.perf_counter() - start
    if artifact.context_contract.get("snapshot_manifest_sha256") != sha256(args.snapshot / "manifest.json"):
        raise ValueError("Bundle and snapshot differ")
    feature = artifact.recipe["feature"]
    for filename in ("rows.csv", "partition.npy", f"{feature}.npy"):
        if sha256(args.snapshot / filename) != manifest["artifact_hashes"][filename]:
            raise ValueError(f"Snapshot hash mismatch: {filename}")
    roles = np.load(args.snapshot / "partition.npy", allow_pickle=False)
    X = np.load(args.snapshot / f"{feature}.npy", mmap_mode="r", allow_pickle=False)
    unique, first_row = [], None
    with (args.snapshot / "rows.csv").open(newline="") as stream:
        for i, row in enumerate(csv.DictReader(stream)):
            smiles = row["canonical_smiles"]
            if roles[i] == "search" and smiles not in unique:
                unique.append(smiles)
                if first_row is None:
                    first_row = i
                if len(unique) == args.batch:
                    break
    if len(unique) != args.batch:
        raise ValueError("Insufficient distinct search molecules for requested batch")
    encoder = ContextEncoder(cache_dir=args.snapshot / "context", protein_space="combined")
    if encoder.fingerprint() != artifact.context_contract.get("context_fingerprint"):
        raise ValueError("Context fingerprint mismatch")
    molecular = MolecularFeatureEncoder(artifact.molecular_spec, use_cache=True, n_workers=1)
    data = FusionData(encoder=encoder, featurizer=molecular)
    if data.dims != artifact.context_contract.get("feature_dims"):
        raise ValueError("Feature layout mismatch")
    molecular_width = data.dims["fingerprint"] + data.dims["descriptors"]
    context = np.asarray(X[first_row, molecular_width:]).reshape(1, -1)

    def score_batch():
        start = time.perf_counter()
        fp, desc, ok = molecular.featurize(unique)
        encoded = time.perf_counter()
        matrix = data.assemble_features(context, fp, desc)
        result = artifact.predict(matrix)
        end = time.perf_counter()
        if not ok.all() or any(not np.isfinite(value).all() for value in result.values()):
            raise ValueError("Invalid molecule or prediction in latency benchmark")
        return {"encode_ms": (encoded - start) * 1000,
                "predict_ms": (end - encoded) * 1000,
                "total_ms": (end - start) * 1000}

    cold = score_batch()
    warm = [score_batch() for _ in range(args.repeats)]
    molecular.close()
    output = {"bundle": str(args.bundle), "model_sha256": sha256(args.bundle / "candidate.joblib"),
              "snapshot_sha256": sha256(args.snapshot / "manifest.json"),
              "model_load_seconds": load_seconds, "batch": args.batch, "repeats": args.repeats,
              "cold": cold, "warm": {key: {"median": float(np.median([row[key] for row in warm])),
                  "p95": float(np.quantile([row[key] for row in warm], .95))} for key in cold},
              "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "note": "One search-context CPU process; excludes Python startup, HTTP and downstream RL."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
