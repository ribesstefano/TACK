"""
Consolidate per-protein Boltz-2 structure embeddings, and their prediction
confidence, scattered across a Boltz output tree into a single npz file.

Boltz-2 (run with `--write_embeddings`, e.g. via submit-boltz.py's
`--per-file` mode on YAMLs from `generate_boltz_protein_yamls.py`) writes
one directory per job:

    <boltz_output_dir>/<record_id>/boltz_results_<record_id>/predictions/<record_id>/
        embeddings_<record_id>.npz              # {'s': (1,L,384), 'z': (1,L,L,128)}
        confidence_<record_id>_model_{i}.json    # one per model checkpoint

where `<record_id>` is the stem of the input YAML (e.g.
"CDK1_P06493_05a26796d4cbc181", i.e. "<name>_<uniprot>_<protein_id_hash>"
from `generate_boltz_protein_yamls.py`). This script walks that tree by
globbing for `embeddings_*.npz`, and for every record found:
  - Extracts chain A's trunk embedding: the per-residue 's' (shape (L, 384))
    or pairwise 'z' (shape (L, L, 128)), see `--component`. Raw by default
    (`--pool none`); optionally pooled to a fixed-size vector via
    `ProteinEmbedding.pool_boltz_embeddings` (same routine
    `tackai.data.embeddings.protein_embeddings.ProteinEmbedding` uses when
    reading these files on the fly).
  - Picks the highest-confidence `confidence_*_model_*.json` (Boltz predicts
    with several model checkpoints per job; model_0 is usually but not
    always best) and records its metrics.
  - Resolves the record's amino-acid sequence from its original input YAML
    (`--boltz-input-dir`, e.g. the directory `generate_boltz_protein_yamls.py`
    wrote to) so the consolidated file can be keyed by sequence -- directly
    loadable via `ProteinEmbedding(embeddings_type="precomputed", ...)`,
    which expects an npz whose keys ARE the raw sequences, matching how
    every other embedding cache in tackai is keyed (see
    `tackai.data.embeddings.utils.EmbeddingMixin.save`). Pass `--key hash`
    to skip this and key by the record's `protein_id` hash instead (no
    `--boltz-input-dir` needed).

Output:
  - `<output>`: one array per record, keyed per `--key`.
  - `<output stem>_manifest.csv`: record_id, key, sequence length, best
    model index, and its confidence metrics.

Usage:
    python scripts/collect_boltz_embeddings.py \\
        --boltz-output-dir boltz_output \\
        --boltz-input-dir boltz_yaml \\
        --output boltz_embeddings/protein_embeddings_boltz2_s.npz
"""
import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yaml

from tackai.data.embeddings.protein_embeddings import ProteinEmbedding

logger = logging.getLogger("collect_boltz_embeddings")

REPO_ROOT = Path(__file__).resolve().parent.parent

HASH_RE = re.compile(r"^[0-9a-f]{16}$")
CONFIDENCE_MODEL_RE = re.compile(r"_model_(\d+)\.json$")

CONFIDENCE_FIELDS = [
    "confidence_score", "ptm", "iptm", "ligand_iptm", "protein_iptm",
    "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde",
]


def protein_id_from_record(record_id: str) -> str:
    """ Extract the trailing `protein_id` hash from a `<name>_<uniprot>_<hash>` record id.

    Falls back to the full record id if it doesn't end in a 16-char hex hash
    (e.g. records not produced by `generate_boltz_protein_yamls.py`).
    """
    tail = record_id.rsplit("_", 1)[-1]
    return tail if HASH_RE.match(tail) else record_id


def load_sequence(boltz_input_dir: Path, record_id: str) -> Optional[str]:
    """ Read chain A's amino-acid sequence from the record's original input YAML. """
    yaml_path = boltz_input_dir / f"{record_id}.yaml"
    if not yaml_path.exists():
        return None
    with open(yaml_path) as f:
        doc = yaml.safe_load(f)
    return doc["sequences"][0]["protein"]["sequence"]


def load_best_confidence(pred_dir: Path, record_id: str) -> Optional[Dict]:
    """ Load the confidence JSON with the highest `confidence_score` among a record's models.

    Args:
        pred_dir: The record's `predictions/<record_id>` directory.
        record_id: The record id (matches the `confidence_<record_id>_model_*.json` glob).

    Returns:
        The parsed confidence dict (plus `_model_idx` and `_n_models`), or
        None if no confidence files were found.
    """
    candidates = sorted(pred_dir.glob(f"confidence_{record_id}_model_*.json"))
    best = None
    for path in candidates:
        with open(path) as f:
            data = json.load(f)
        match = CONFIDENCE_MODEL_RE.search(path.name)
        data["_model_idx"] = int(match.group(1)) if match else None
        if best is None or data["confidence_score"] > best["confidence_score"]:
            best = data
    if best is not None:
        best["_n_models"] = len(candidates)
    return best


def extract_component(
        emb_path: Path, component: str, pooling: str,
) -> np.ndarray:
    """ Load and optionally pool the requested embedding component for chain A.

    Args:
        emb_path: Path to `embeddings_<record_id>.npz`.
        component: "s" (per-residue, (L, 384)) or "z" (pairwise, (L, L, 128)).
        pooling: "none" to keep the raw array, else a pooling method passed
            to `ProteinEmbedding.pool_boltz_embeddings`.

    Returns:
        The (optionally pooled) embedding array.
    """
    data = np.load(emb_path)
    raw = data[component]  # (1, L, D) or (1, L, L, D)
    if pooling == "none":
        return raw[0]
    if component == "s":
        return ProteinEmbedding.pool_boltz_embeddings(s=raw, pooling=pooling)
    return ProteinEmbedding.pool_boltz_embeddings(z=raw, pooling=pooling)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--boltz-output-dir", required=True,
        help="Root directory of Boltz-2 prediction outputs to walk (searched recursively "
             "for embeddings_*.npz).",
    )
    parser.add_argument(
        "--boltz-input-dir",
        help="Directory of the original per-protein input YAMLs (e.g. generate_boltz_protein_"
             "yamls.py's --output-dir). Required when --key sequence (the default).",
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to write the consolidated embeddings npz to. A manifest CSV is written "
             "alongside it as '<stem>_manifest.csv'.",
    )
    parser.add_argument(
        "--key", choices=["sequence", "hash"], default="sequence",
        help="Key each array by the raw amino-acid sequence (default; requires "
             "--boltz-input-dir) or by the record's protein_id hash.",
    )
    parser.add_argument(
        "--component", choices=["s", "z"], default="s",
        help="Which Boltz-2 trunk embedding to collect: 's' (per-residue, default) or "
             "'z' (pairwise).",
    )
    parser.add_argument(
        "--pool", choices=["none", "mean", "sum", "max", "mean_sqrt_len"], default="none",
        help="Pool the raw embedding to a fixed-size vector via "
             "ProteinEmbedding.pool_boltz_embeddings (default: keep the raw per-residue/pairwise "
             "array, as expected by ProteinEmbedding(embeddings_type='precomputed')).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite --output / the manifest if they already exist (default: error out).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    boltz_output_dir = Path(args.boltz_output_dir)
    output_path = Path(args.output)
    manifest_path = output_path.with_name(f"{output_path.stem}_manifest.csv")

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it.")

    boltz_input_dir = Path(args.boltz_input_dir) if args.boltz_input_dir else None
    if args.key == "sequence" and boltz_input_dir is None:
        raise ValueError("--boltz-input-dir is required when --key sequence.")

    emb_paths = sorted(boltz_output_dir.glob("**/embeddings_*.npz"))
    logger.info(f"Found {len(emb_paths)} embeddings_*.npz file(s) under {boltz_output_dir}")

    embeddings: Dict[str, np.ndarray] = {}
    manifest_rows = []
    n_missing_sequence, n_missing_confidence, n_duplicate_key = 0, 0, 0

    for emb_path in emb_paths:
        pred_dir = emb_path.parent
        record_id = pred_dir.name
        protein_id_hash = protein_id_from_record(record_id)

        # Loaded whenever available (not just for --key sequence) to populate sequence_length.
        sequence = load_sequence(boltz_input_dir, record_id) if boltz_input_dir else None
        if args.key == "sequence" and sequence is None:
            n_missing_sequence += 1
            logger.warning(f"{record_id}: no matching input YAML in {boltz_input_dir}; skipping.")
            continue

        key = sequence if args.key == "sequence" else protein_id_hash
        if key in embeddings:
            n_duplicate_key += 1
            logger.warning(f"{record_id}: key already seen (duplicate {args.key}); keeping first, skipping.")
            continue

        embeddings[key] = extract_component(emb_path, args.component, args.pool)

        confidence = load_best_confidence(pred_dir, record_id)
        if confidence is None:
            n_missing_confidence += 1
            logger.warning(f"{record_id}: no confidence_*.json found.")

        row = {
            "record_id": record_id,
            "key": key if args.key == "hash" else protein_id_hash,
            "protein_id_hash": protein_id_hash,
            "sequence_length": len(sequence) if sequence else None,
            "best_model_idx": confidence.get("_model_idx") if confidence else None,
            "n_confidence_models": confidence.get("_n_models") if confidence else 0,
        }
        for field in CONFIDENCE_FIELDS:
            row[field] = confidence.get(field) if confidence else None
        manifest_rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **embeddings)

    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    logger.info(
        f"Collected {len(embeddings)} record(s) ({args.component}, pool={args.pool}, key={args.key})."
    )
    if n_missing_sequence:
        logger.info(f"Skipped {n_missing_sequence} record(s) with no matching input YAML.")
    if n_duplicate_key:
        logger.info(f"Skipped {n_duplicate_key} record(s) with a duplicate key.")
    if n_missing_confidence:
        logger.info(f"{n_missing_confidence} record(s) had no confidence_*.json.")
    logger.info(f"Embeddings: {output_path}")
    logger.info(f"Manifest:   {manifest_path}")


if __name__ == "__main__":
    main()
