"""Freeze data, context, molecular alternatives and shared research partitions.

Run on a CPU allocation. The historical test is read solely for molecular
overlap exclusion; no test predictions or metrics are calculated here.
"""
import argparse
import importlib.metadata
import json
import os
import hashlib
from pathlib import Path
import subprocess

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from rdkit.Chem import rdFingerprintGenerator

from tackai.chem_utils import canonicalize_smiles
from tackai.fusion.context import ContextEncoder, stage_context_tables
from tackai.fusion.data import FusionData
from tackai.fusion.labels import research_targets
from tackai.fusion.research import assign_partitions, identity_groups, sha256, write_once_json
from train_fast_models import test_overlap_smiles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dev-files", nargs="+", default=[
        "data/AutoTPD-TPDdb.csv", "data/auto_tpd_plus/AutoTPDplus-PROTAC.csv"])
    parser.add_argument("--test-file", default="data/auto_tpd_plus/PROTAC-test.csv")
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen experiment {output}")
    output.mkdir(parents=True)
    encoder = ContextEncoder(cache_dir=args.cache_dir, protein_space="combined")
    stage_context_tables(args.cache_dir, output / "context", protein_space="combined")
    exclude = test_overlap_smiles(args.dev_files, args.test_file)
    data = FusionData.from_csv(args.dev_files, encoder=encoder, exclude_smiles=exclude,
                               verbose=True)
    tab = data.table.copy()
    canonical = [canonicalize_smiles(s) for s in tab.smiles]
    groups = identity_groups(data.groups, canonical, tab.Connectivity_Key)
    roles = assign_partitions(groups)
    tab["canonical_smiles"] = canonical
    tab["group"] = groups
    tab["partition"] = roles
    def publication(row):
        doi = str(row.DOI).strip().lower()
        if doi not in {"", "nan", "none"}:
            return "doi:" + doi.removeprefix("https://doi.org/")
        reference = str(row.Reference).strip()
        return "reference:" + reference if reference.lower() not in {"", "nan", "none"} else ""
    tab["publication"] = [publication(row) for row in tab.itertuples()]
    # Development-only source stress test: selected publications are held out;
    # chemical groups shared with them are purged from its training side.
    held_publications = {p for p in tab.publication if p and
                         int(hashlib.sha256(p.encode()).hexdigest()[:8], 16) % 5 == 0}
    source_eval = (roles == "search") & tab.publication.isin(held_publications).to_numpy()
    source_roles = np.full(len(tab), "not_search", dtype="U16")
    source_roles[roles == "search"] = "train"
    source_roles[(roles == "search") & np.isin(groups, groups[source_eval])] = "purged"
    source_roles[source_eval] = "eval"
    tab["source_partition"] = source_roles
    input_hashes = {str(Path(p).resolve()): sha256(p) for p in args.dev_files + [args.test_file]}
    tab["row_id"] = [f"{input_hashes[p][:16]}:{i}" for p, i in
                     zip(tab.source_path, tab.source_row)]
    if not tab.row_id.is_unique:
        raise ValueError("Nonunique raw row identities")
    tab.to_csv(output / "rows.csv", index=False)
    np.save(output / "groups.npy", groups)
    np.save(output / "partition.npy", roles)
    np.save(output / "source_partition.npy", source_roles)
    labels = {f"compat_{task}": data.target(task) for task in ("dmax", "pdc50", "activity")}
    for policy in ("exact", "censored"):
        targets, bounds = research_targets(tab, policy=policy)
        labels.update({f"{policy}_{task}": value for task, value in targets.items()})
        labels.update({f"{policy}_{task}_bounds": value for task, value in bounds.items()})
    np.savez(output / "labels.npz", **labels)
    np.save(output / "baseline.npy", data.X)
    # Descriptor and context columns are copied exactly, not recalculated.
    remainder = data.X[:, data.dims["fingerprint"]:]
    features = {"baseline": {"radius": 16, "fp_size": 1024, "counts": False,
                              "dims": data.dims, "descriptors": data.featurizer.descriptors}}
    molecules = {s: canonicalize_smiles(s, return_mol=True) for s in dict.fromkeys(tab.smiles)}
    for radius, width, counts in [(2, 2048, False), (3, 2048, False),
                                  (2, 4096, False), (3, 4096, False), (2, 2048, True)]:
        name = f"morgan_r{radius}_{width}_{'count' if counts else 'bit'}"
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=width, includeChirality=True)
        method = generator.GetCountFingerprintAsNumPy if counts else generator.GetFingerprintAsNumPy
        encoded = {s: method(mol).astype(np.float32) for s, mol in molecules.items()}
        fp = np.stack([encoded[s] for s in tab.smiles])
        np.save(output / f"{name}.npy", np.concatenate([fp, remainder], axis=1))
        features[name] = {"radius": radius, "fp_size": width, "counts": counts,
                          "dims": {**data.dims, "fingerprint": width},
                          "descriptors": data.featurizer.descriptors,
                          "mean_occupied_fraction": float((fp != 0).mean())}
        print(f"Saved {name}", flush=True)
    summary = {"rows": len(tab), "groups": len(set(groups)), "excluded": data.excluded,
               "dropped": data.dropped, "partitions": {},
               "source_stress": {r: int((source_roles == r).sum()) for r in np.unique(source_roles)}}
    for role in np.unique(roles):
        mask = roles == role
        summary["partitions"][role] = {
            "rows": int(mask.sum()), "groups": len(set(groups[mask])),
            "labels": {name: int(np.isfinite(y[mask]).sum()) for name, y in labels.items()
                       if y.ndim == 1}}
    manifest = {
        "format": "tack-surrogate-research/v1", "seed": 20261030,
        "input_hashes": input_hashes, "context_fingerprint": encoder.fingerprint(),
        "protein_space": "combined", "features": features,
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "code_hashes": {str(p): sha256(p) for p in [Path(__file__),
            Path("tackai/fusion/data.py"), Path("tackai/fusion/labels.py"),
            Path("tackai/fusion/research.py"), Path("tackai/fusion/mol_encoder.py"),
            Path("tackai/fusion/context.py")]},
        "versions": {name: importlib.metadata.version(name) for name in
                     ("numpy", "pandas", "scikit-learn", "xgboost", "rdkit")},
        "artifact_hashes": {p.name: sha256(p) for p in output.iterdir() if p.is_file()},
        "summary": summary,
        "confirmation_note": "Internal reserved groups for new recipes; not a prospective external test."
    }
    write_once_json(output / "manifest.json", manifest)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
