"""Verify frozen data integrity, identity isolation and live feature parity."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd

from tackai.chem_utils import canonicalize_smiles
from tackai.fusion.context import ContextEncoder
from tackai.fusion.data import FusionData
from tackai.fusion.molecular_spec import MolecularFeatureEncoder, MolecularFeatureSpec
from tackai.fusion.research import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.snapshot / "manifest.json").read_text())
    for name, expected in manifest["artifact_hashes"].items():
        if sha256(args.snapshot / name) != expected:
            raise ValueError(f"Snapshot hash mismatch: {name}")
    context_manifest = json.loads((args.snapshot / "context/manifest.json").read_text())
    for name, meta in context_manifest["files"].items():
        if sha256(args.snapshot / "context" / name) != meta["sha256"]:
            raise ValueError(f"Frozen context hash mismatch: {name}")
    encoder = ContextEncoder(cache_dir=args.snapshot / "context", protein_space="combined")
    if encoder.fingerprint() != manifest["context_fingerprint"]:
        raise ValueError("Context fingerprint mismatch")
    rows = pd.read_csv(args.snapshot / "rows.csv", low_memory=False)
    groups = np.load(args.snapshot / "groups.npy")
    roles = np.load(args.snapshot / "partition.npy")
    source = np.load(args.snapshot / "source_partition.npy")
    if not rows.row_id.is_unique or len(rows) != manifest["summary"]["rows"]:
        raise ValueError("Snapshot row identity mismatch")
    if any(len(x) != len(rows) for x in (groups, roles, source)):
        raise ValueError("Partition arrays not aligned")
    if rows.groupby("group").partition.nunique().max() != 1:
        raise ValueError("Scaffold/identity component crosses partitions")
    if set(groups[source == "train"]) & set(groups[source == "eval"]):
        raise ValueError("Source holdout chemical overlap")
    if set(rows.publication[source == "train"].dropna()) & set(rows.publication[source == "eval"].dropna()):
        raise ValueError("Source holdout publication overlap")
    test_path = next(p for p in manifest["input_hashes"] if Path(p).name == "PROTAC-test.csv")
    if sha256(test_path) != manifest["input_hashes"][test_path]:
        raise ValueError("Historical test input changed")
    test = pd.read_csv(test_path, low_memory=False)
    test_molecules = {s for s in map(canonicalize_smiles, test.SMILES) if s}
    overlap = set(rows.canonical_smiles) & test_molecules
    if overlap:
        raise ValueError(f"Standardized historical-test molecule overlap: {len(overlap)}")
    ids = np.random.default_rng(20261030).choice(len(rows), min(32, len(rows)), replace=False)
    records = [{"smiles": r.smiles, "poi_seq": r.poi_seq, "e3_seq": r.e3_seq,
                "cell_id": r.cell_key, "assay": r.assay_raw, "assay_time": r.assay_time}
               for r in rows.iloc[ids].itertuples()]
    verified = {}
    for name, feature in manifest["features"].items():
        matrix = np.load(args.snapshot / f"{name}.npy", mmap_mode="r")
        if matrix.shape != (len(rows), sum(feature["dims"].values())):
            raise ValueError(f"Feature shape mismatch: {name}")
        if not np.isfinite(matrix).all():
            raise ValueError(f"Nonfinite features: {name}")
        spec = MolecularFeatureSpec(radius=feature["radius"], size=feature["fp_size"],
                                    count=feature["counts"],
                                    descriptor_names=tuple(feature["descriptors"]))
        data = FusionData(encoder=encoder, featurizer=MolecularFeatureEncoder(spec))
        actual = data.transform(records)
        np.testing.assert_allclose(actual, matrix[ids], rtol=1e-6, atol=1e-6,
                                   err_msg=f"Live feature parity failed: {name}")
        verified[name] = {"rows_checked": len(ids), "spec": spec.to_dict()}
    report = {"snapshot_hash": sha256(args.snapshot / "manifest.json"),
              "historical_test_standardized_molecule_overlap": 0,
              "source_publication_and_molecule_overlap": 0,
              "all_artifact_hashes_verified": True, "features": verified}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "verified", "features": list(verified)}))


if __name__ == "__main__":
    main()
