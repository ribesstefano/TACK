"""
Generate Boltz-2 input YAML files for PROTAC-Target-Recruiter ternary
complexes referenced in a TACKv2 CSV: one YAML file per unique
(POI protein, E3-ligase protein, PROTAC ligand) combination, each with the
POI as chain A, the E3-ligase recruiter as chain B, and the PROTAC SMILES as
ligand C.

Companion to `generate_boltz_protein_yamls.py`, which emits single-protein
YAMLs for the same POI/recruiter pool; see that script's docstring for why
identifiers are hashed rather than derived from the raw name or sequence
(sequences run past 7,000 residues -- well over typical filesystem
path-component limits -- and some names map to more than one sequence, e.g.
point mutants).

Each complex is identified by a SHA-256 hash of its (poi_id, ligase_id,
smiles) triple, where poi_id/ligase_id are themselves SHA-256 hashes of
(name, sequence) -- computed with the same `protein_id()` as the sibling
script, so complexes and their constituent single-protein YAMLs share
identifiers. Filenames additionally embed the sanitized POI/recruiter names
and a short SMILES hash so a complex can be traced back at a glance without
opening the manifest; the full complex hash guarantees uniqueness even when
two rows share a POI/recruiter name pair (e.g. mutants) or a truncated
SMILES hash collides.

Run Boltz-2 against the generated directory with `--write_embeddings` (e.g.
via submit-boltz.py's `--per-file` mode) to compute per-complex structure
embeddings.

Usage:
    python scripts/generate_boltz_complex_yamls.py \\
        --input-csv tack_v2_with_predictions_scores_splits.csv \\
        --output-dir ./boltz_complex_inputs
"""
import argparse
import hashlib
import logging
import re
from pathlib import Path

import pandas as pd
import yaml

from scripts.boltz.generate_boltz_protein_yamls import DIGEST_SIZE, protein_id

logger = logging.getLogger("generate_boltz_complex_yamls")

REPO_ROOT = Path(__file__).resolve().parent.parent

POI_NAME_COL = "Degradation_Target_Gene"
POI_NAME_FALLBACK_COL = "Degradation_Target"
POI_SEQ_COL = "Degradation_Target_Sequence"
POI_UNIPROT_COL = "Degradation_Target_Uniprot"
LIGASE_NAME_COL = "Recruiter"
LIGASE_SEQ_COL = "Recruiter_Sequence"
LIGASE_UNIPROT_COL = "Recruiter_Uniprot"
SMILES_COL = "SMILES"
REFERENCE_COL = "Reference"

SMILES_DIGEST_SIZE = 8  # short hash fragment embedded in the filename
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9.-]+")


def sanitize_name(name: str) -> str:
    """ Make a display name safe to embed as a filename component. """
    return _UNSAFE_CHARS.sub("_", str(name).strip()).strip("_") or "unknown"


def content_id(text: str, digest_size: int = DIGEST_SIZE) -> str:
    """ SHA-256 hash of arbitrary text, truncated to `digest_size` hex chars. """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:digest_size]


def complex_id(poi_id: str, ligase_id: str, smiles: str, digest_size: int = DIGEST_SIZE) -> str:
    """ Deterministic identifier for a (POI, E3-ligase, PROTAC) complex.

    Args:
        poi_id: `protein_id()` of the POI (name, sequence) pair.
        ligase_id: `protein_id()` of the E3-ligase (name, sequence) pair.
        smiles: PROTAC SMILES string.
        digest_size: Number of hex characters to keep from the SHA-256 digest.

    Returns:
        A `digest_size`-character lowercase hex string.
    """
    payload = f"{poi_id}\x1f{ligase_id}\x1f{smiles}"
    return content_id(payload, digest_size=digest_size)


def complex_yaml(poi_sequence: str, ligase_sequence: str, smiles: str, use_msa_server: bool = True) -> str:
    """ Boltz-2 ternary-complex input YAML: POI (A) + E3-ligase recruiter (B) + PROTAC ligand (C).

    Args:
        poi_sequence: POI amino-acid sequence for chain A.
        ligase_sequence: E3-ligase amino-acid sequence for chain B.
        smiles: PROTAC SMILES string for ligand C.
        use_msa_server: If False, sets `msa: empty` on both protein chains so
            Boltz skips MSA search instead of querying the MSA server.

    Returns:
        YAML document text.
    """
    protein_a = {"id": "A", "sequence": poi_sequence}
    protein_b = {"id": "B", "sequence": ligase_sequence}
    if not use_msa_server:
        protein_a["msa"] = "empty"
        protein_b["msa"] = "empty"
    doc = {
        "version": 1,
        "sequences": [
            {"protein": protein_a},
            {"protein": protein_b},
            {"ligand": {"id": "C", "smiles": smiles}},
        ],
    }
    return yaml.safe_dump(doc, sort_keys=False, default_flow_style=False)


def collect_unique_complexes(df: pd.DataFrame) -> pd.DataFrame:
    """ Gather every unique (POI, E3-ligase, PROTAC SMILES) combination.

    Args:
        df: TACKv2 dataframe.

    Returns:
        One row per unique complex, with POI/ligase name/sequence/UniProt,
        SMILES, and the "|"-joined set of source `Reference` values.
    """
    required = [
        POI_NAME_COL, POI_NAME_FALLBACK_COL, POI_SEQ_COL, POI_UNIPROT_COL,
        LIGASE_NAME_COL, LIGASE_SEQ_COL, LIGASE_UNIPROT_COL, SMILES_COL, REFERENCE_COL,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Expected column(s) {missing} not found in input CSV.")

    sub = df[required].copy()
    sub["poi_name"] = sub[POI_NAME_COL].fillna(sub[POI_NAME_FALLBACK_COL])
    sub = sub.rename(columns={
        POI_SEQ_COL: "poi_sequence",
        POI_UNIPROT_COL: "poi_uniprot",
        LIGASE_NAME_COL: "ligase_name",
        LIGASE_SEQ_COL: "ligase_sequence",
        LIGASE_UNIPROT_COL: "ligase_uniprot",
        SMILES_COL: "smiles",
        REFERENCE_COL: "reference",
    })
    sub = sub.drop(columns=[POI_NAME_COL, POI_NAME_FALLBACK_COL])

    sub = sub.dropna(subset=["poi_name", "poi_sequence", "ligase_name", "ligase_sequence", "smiles"])
    for col in ("poi_sequence", "ligase_sequence", "smiles"):
        sub[col] = sub[col].str.strip()
    sub = sub[(sub["poi_sequence"] != "") & (sub["ligase_sequence"] != "") & (sub["smiles"] != "")]

    grouped = sub.groupby(
        ["poi_name", "poi_sequence", "ligase_name", "ligase_sequence", "smiles"], as_index=False
    ).agg(
        poi_uniprot=("poi_uniprot", "first"),
        ligase_uniprot=("ligase_uniprot", "first"),
        references=("reference", lambda s: "|".join(sorted(set(s)))),
        n_rows=("reference", "size"),
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
        help="Directory to write one Boltz-2 complex input YAML per unique "
             "(POI, E3-ligase, PROTAC) combination into.",
    )
    parser.add_argument(
        "--no-msa-server", dest="use_msa_server", action="store_false",
        help="Set `msa: empty` on both protein chains so `boltz predict` skips MSA search "
             "instead of querying the MSA server (default: query the MSA server via "
             "`boltz predict --use_msa_server`).",
    )
    parser.set_defaults(use_msa_server=True)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite YAML files that already exist (default: skip them).",
    )
    parser.add_argument(
        "--write-manifest", action="store_true",
        help="Whether to write a summary manifest of the YAML files under output-dir",
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

    complexes = collect_unique_complexes(df)
    complexes["poi_id"] = [
        protein_id(name, seq) for name, seq in zip(complexes["poi_name"], complexes["poi_sequence"])
    ]
    complexes["ligase_id"] = [
        protein_id(name, seq) for name, seq in zip(complexes["ligase_name"], complexes["ligase_sequence"])
    ]
    complexes["smiles_id"] = [content_id(s, digest_size=SMILES_DIGEST_SIZE) for s in complexes["smiles"]]
    complexes["complex_id"] = [
        complex_id(poi_id, ligase_id, smiles)
        for poi_id, ligase_id, smiles in zip(complexes["poi_id"], complexes["ligase_id"], complexes["smiles"])
    ]

    dupes = complexes["complex_id"].duplicated(keep=False)
    if dupes.any():
        raise RuntimeError(
            f"Hash collision detected among {dupes.sum()} complex_id(s); "
            f"increase DIGEST_SIZE (currently {DIGEST_SIZE})."
        )

    complexes["filename"] = [
        f"{sanitize_name(poi_name)}__{sanitize_name(ligase_name)}__{smiles_id}_{cid}.yaml"
        for poi_name, ligase_name, smiles_id, cid in zip(
            complexes["poi_name"], complexes["ligase_name"], complexes["smiles_id"], complexes["complex_id"]
        )
    ]

    written, skipped = 0, 0
    for row in complexes.itertuples(index=False):
        yaml_path = output_dir / row.filename
        if yaml_path.exists() and not args.overwrite:
            skipped += 1
            continue
        yaml_path.write_text(
            complex_yaml(row.poi_sequence, row.ligase_sequence, row.smiles, use_msa_server=args.use_msa_server)
        )
        written += 1

    if args.write_manifest:
        manifest_path = output_dir / "complex_manifest.csv"
        manifest = complexes[[
            "complex_id", "filename",
            "poi_id", "poi_name", "poi_uniprot",
            "ligase_id", "ligase_name", "ligase_uniprot",
            "smiles_id", "smiles",
            "n_rows", "references",
        ]].copy()
        manifest.to_csv(manifest_path, index=False)

    logger.info(
        f"{len(complexes)} unique complexes across "
        f"{complexes['poi_id'].nunique()} POI(s) and {complexes['ligase_id'].nunique()} E3-ligase(s)."
    )
    logger.info(f"Wrote {written} YAML file(s), skipped {skipped} already present, to {output_dir}")
    if args.write_manifest:
        logger.info(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
