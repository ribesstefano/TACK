"""Build ``demo/vocab.json``, the human-readable vocabulary the Gradio demo offers.

The published context tables (``ailab-bio/TACK-Context-Encoding``) are keyed by *machine*
identifiers: full amino-acid sequences for the POI and E3 blocks, Cellosaurus accessions for
the cell block. A user of the demo knows "BRD4" and "HeLa", not ``MSAESG...`` and
``CVCL_0030``, so this script joins the curated CSVs — the only place where gene symbol,
UniProt accession and sequence sit in one row — against the cached tables and writes the
intersection out as a small JSON file.

The intersection matters in both directions: a name whose sequence is not in the table would
raise at predict time (see the missing-key rationale in ``tackai/fusion/context.py``), and a
table key with no curated name cannot be offered as a label. Only pairs that satisfy both
survive, so every option the demo lists is guaranteed to encode.

Run it from the repo root after a context-table release, then commit the result:

    OMP_NUM_THREADS=1 python demo/build_vocab.py
"""
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")  # must precede the first xgboost/torch import

import pandas as pd

from tackai.fusion import FusionData
from tackai.fusion.context import normalize_assay

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = Path(__file__).resolve().parent / "vocab.json"

#: Curated tables to harvest names from. Every file carries the same POI/E3/cell columns.
SOURCES = (
    "data/yaochen/AutoTPDplus-PROTAC-training.csv",
    "data/yaochen/AutoTPD-PROTAC.csv",
    "data/yaochen/PROTACDB-training.csv",
    "data/yaochen/PROTACDB_basedata.csv",
    "data/yaochen/PROTACDB_pmc.csv",
    "data/yaochen/PROTAC-test.csv",
)

PROTEIN_COLUMNS = {
    "poi": ("Degradation_Target_Gene", "Degradation_Target_Uniprot", "Degradation_Target_Sequence"),
    "e3": ("Recruiter_Gene", "Recruiter_Uniprot", "Recruiter_Sequence"),
}
CELL_COLUMNS = ("Cell_Line_STD", "Cell_Line_STD_Accession")


def load_sources() -> pd.DataFrame:
    """Concatenate the curated CSVs, keeping only the columns this script reads."""
    wanted = [c for cols in PROTEIN_COLUMNS.values() for c in cols] + list(CELL_COLUMNS)
    frames = []
    for name in SOURCES:
        path = REPO_ROOT / name
        if not path.exists():
            print(f"  skipped (absent): {name}")
            continue
        frame = pd.read_csv(path, usecols=lambda c: c in wanted, low_memory=False)
        frames.append(frame)
        print(f"  read {len(frame):>6} rows from {name}")
    if not frames:
        raise FileNotFoundError(f"none of the curated CSVs exist under {REPO_ROOT}")
    return pd.concat(frames, ignore_index=True)


def protein_entries(table: pd.DataFrame, block: str, known: set) -> list:
    """`[{label, gene, uniprot, sequence}]` for every curated sequence the table holds.

    Args:
        table: the concatenated curated rows.
        block: ``"poi"`` or ``"e3"``.
        known: sequences the cached table for that block can encode.

    Returns:
        One entry per sequence, labelled ``GENE (ACCESSION)``, sorted by label. Where the
        same sequence appears under several gene symbols the most frequent one wins, so the
        label reflects how the literature usually names it.
    """
    gene_col, acc_col, seq_col = PROTEIN_COLUMNS[block]
    rows = table[[gene_col, acc_col, seq_col]].dropna()
    rows = rows[rows[seq_col].isin(known)]

    entries = []
    for sequence, group in rows.groupby(seq_col):
        gene = group[gene_col].value_counts().idxmax()
        uniprot = group[acc_col].value_counts().idxmax()
        entries.append({
            "label": f"{gene} ({uniprot})",
            "gene": gene,
            "uniprot": uniprot,
            "sequence": sequence,
        })
    unnamed = len(known) - len(entries)
    print(f"  {block}: {len(entries)} named of {len(known)} cached ({unnamed} unnamed)")
    return sorted(entries, key=lambda e: e["label"])


def cell_entries(table: pd.DataFrame, known: set) -> list:
    """`[{label, name, accession}]` for every curated cell line the cached table holds."""
    rows = table[list(CELL_COLUMNS)].dropna()
    rows = rows[rows[CELL_COLUMNS[1]].isin(known)]

    entries = []
    for accession, group in rows.groupby(CELL_COLUMNS[1]):
        name = group[CELL_COLUMNS[0]].value_counts().idxmax()
        entries.append({
            "label": f"{name} ({accession})",
            "name": name,
            "accession": accession,
        })
    print(f"  cell: {len(entries)} named of {len(known)} cached ({len(known) - len(entries)} unnamed)")
    return sorted(entries, key=lambda e: e["label"])


def main() -> None:
    print("curated sources:")
    table = load_sources()

    print("cached context tables:")
    encoder = FusionData.from_pretrained().encoder

    vocab = {
        "poi": protein_entries(table, "poi", set(encoder.table("poi"))),
        "e3": protein_entries(table, "e3", set(encoder.table("e3"))),
        "cell": cell_entries(table, set(encoder.table("cell"))),
        # The assay vocabulary is already human-readable, and closed: an unseen type falls
        # back to "unknown" rather than raising, so the table itself is the whole list.
        "assay": sorted(normalize_assay(k) for k in encoder.table("assay")),
    }

    OUT_PATH.write_text(json.dumps(vocab, indent=1, sort_keys=True) + "\n")
    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"wrote {OUT_PATH.relative_to(REPO_ROOT)} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
