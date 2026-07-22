"""
Generate Boltz-2 input YAML files for every unique protein (POI and E3-ligase
recruiter) referenced in a TACKv2 CSV, one YAML file per unique protein.

Each protein is identified by a SHA-256 hash of its (name, sequence) pair,
used as both the YAML file stem and the resulting Boltz-2 job name. Hashing
(rather than sanitizing the name, or using the raw sequence) is necessary
because: (1) sequences run past 7,000 residues in this dataset, well over
the ~255-byte path-component limit on most filesystems; (2) some protein
names map to more than one sequence (point mutants, see
`Degradation_Target_Uniprot_MutationID`), so the name alone is not a unique
key; and (3) hashing (name, sequence) together -- rather than the sequence
alone -- keeps distinct mutants of the same protein as distinct jobs.

Run Boltz-2 against the generated directory with `--write_embeddings` (e.g.
via submit-boltz.py's `--per-file` mode) to compute per-protein structure
embeddings consumed by
`tackai.data.embeddings.protein_embeddings.ProteinEmbedding`
(`embeddings_type="boltz2_s"` / `"boltz2_z"`, which expects predictions
under `<boltz_output_dir>/boltz_results_<protein_id>/predictions/<protein_id>/`).

See also `generate_boltz_complex_yamls.py` for full PROTAC-Target-Recruiter
ternary complex YAMLs (POI + E3 ligase + PROTAC ligand in one file).

Usage:
    python scripts/generate_boltz_protein_yamls.py \\
        --input-csv tack_v2_with_predictions_scores_splits.csv \\
        --output-dir ./boltz_inputs
"""
import argparse
import hashlib
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger("generate_boltz_protein_yamls")

REPO_ROOT = Path(__file__).resolve().parent.parent

# (name_column, sequence_column, uniprot_column, role_label)
PROTEIN_SOURCES = [
    ("Degradation_Target_Gene", "Degradation_Target_Sequence", "Degradation_Target_Uniprot", "POI"),
    ("Recruiter", "Recruiter_Sequence", "Recruiter_Uniprot", "E3_ligase"),
]

DIGEST_SIZE = 16  # hex chars kept from the SHA-256 digest (64 bits)


def protein_id(name: str, sequence: str, digest_size: int = DIGEST_SIZE) -> str:
    """ Deterministic, filesystem-safe identifier for a (name, sequence) pair.

    Args:
        name: Protein display name (e.g. gene symbol).
        sequence: Amino-acid sequence.
        digest_size: Number of hex characters to keep from the SHA-256 digest.

    Returns:
        A `digest_size`-character lowercase hex string.
    """
    # \x1f (unit separator) makes ("AB", "CD") and ("A", "BCD") hash differently.
    payload = f"{name}\x1f{sequence}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:digest_size]


def protein_yaml(sequence: str, use_msa_server: bool = False) -> str:
    """ Boltz-2 single-protein input YAML (mirrors submit-boltz.py's `_protein_yaml`).

    Args:
        sequence: Amino-acid sequence for chain A.
        use_msa_server: If False, sets `msa: empty` so Boltz skips MSA search.

    Returns:
        YAML document text.
    """
    msa_line = "" if use_msa_server else "\n      msa: empty"
    return (
        "version: 1\n"
        "sequences:\n"
        "  - protein:\n"
        "      id: A\n"
        f"      sequence: {sequence}{msa_line}\n"
    )


def collect_unique_proteins(df: pd.DataFrame) -> pd.DataFrame:
    """ Gather every unique (name, sequence) pair from the POI and recruiter columns.

    Args:
        df: TACKv2 dataframe.

    Returns:
        One row per unique (name, sequence) pair, with columns `name`,
        `sequence`, `uniprot`, and `roles` (roles it appears under, "|"-joined).
    """
    frames = []
    for name_col, seq_col, uniprot_col, role in PROTEIN_SOURCES:
        missing = [c for c in (name_col, seq_col) if c not in df.columns]
        if missing:
            raise KeyError(f"Expected column(s) {missing} not found in input CSV.")
        has_uniprot = uniprot_col in df.columns
        cols = [name_col, seq_col] + ([uniprot_col] if has_uniprot else [])
        sub = df[cols].copy()
        sub.columns = ["name", "sequence"] + (["uniprot"] if has_uniprot else [])
        if not has_uniprot:
            sub["uniprot"] = None
        sub["role"] = role
        frames.append(sub)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["name", "sequence"])
    combined["sequence"] = combined["sequence"].str.strip()
    combined = combined[combined["sequence"] != ""]

    grouped = combined.groupby(["name", "sequence"], as_index=False).agg(
        uniprot=("uniprot", "first"),
        roles=("role", lambda s: "|".join(sorted(set(s)))),
    )
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-csv", default=str(REPO_ROOT / "tack_v2_with_predictions_scores_splits.csv"),
        help="Path to the TACKv2 CSV.",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to write one Boltz-2 input YAML per unique protein into.",
    )
    parser.add_argument(
        "--no-msa-server", dest="use_msa_server", action="store_false",
        help="Set `msa: empty` so `boltz predict` skips MSA search instead of querying the MSA "
             "server (default: query the MSA server via `boltz predict --use_msa_server`).",
    )
    parser.set_defaults(use_msa_server=True)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite YAML files that already exist (default: skip them).",
    )
    parser.add_argument(
        "--write-manifest", action="store_true",
        help="Whether to write a summary manifest of the YAML file under output-dir",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading {input_csv}")
    df = pd.read_csv(input_csv)

    proteins = collect_unique_proteins(df)
    proteins["protein_id"] = [
        protein_id(name, sequence) for name, sequence in zip(proteins["name"], proteins["sequence"])
    ]

    dupes = proteins["protein_id"].duplicated(keep=False)
    if dupes.any():
        raise RuntimeError(
            f"Hash collision detected among {dupes.sum()} protein_id(s); "
            f"increase DIGEST_SIZE (currently {DIGEST_SIZE})."
        )

    written, skipped = 0, 0
    for row in proteins.itertuples(index=False):
        yaml_path = output_dir / f"{row.name}_{row.uniprot}_{row.protein_id}.yaml"
        if yaml_path.exists() and not args.overwrite:
            skipped += 1
            continue
        yaml_path.write_text(protein_yaml(row.sequence, use_msa_server=args.use_msa_server))
        written += 1

    if args.write_manifest:
        manifest_path = output_dir / "protein_manifest.csv"
        manifest = proteins[["protein_id", "name", "roles", "uniprot", "sequence"]].copy()
        manifest["sequence_length"] = proteins["sequence"].str.len()
        manifest.to_csv(manifest_path, index=False)

    n_poi = (proteins["roles"] == "POI").sum()
    n_ligase = (proteins["roles"] == "E3_ligase").sum()
    n_shared = (proteins["roles"] == "E3_ligase|POI").sum()
    logger.info(
        f"{len(proteins)} unique proteins ({n_poi} POI-only, {n_ligase} E3-ligase-only, "
        f"{n_shared} appearing as both)."
    )
    logger.info(f"Wrote {written} YAML file(s), skipped {skipped} already present, to {output_dir}")
    if args.write_manifest:
        logger.info(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
