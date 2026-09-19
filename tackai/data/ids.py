"""
Stable, order-independent row identifiers for the TACK dataset.

Two hashes are derived per row so that prediction outputs can reference a
compact, deterministic key instead of copying the full assay metadata:

- ``context_id`` — a hash over the biological/experimental *context* columns
  (compound, target, recruiter, cell line, assay, treatment time, reference).
  It groups every measurement taken in the same context.
- ``row_id`` — a hash over the context columns *plus* the raw DC50/Dmax
  readouts, uniquely pinning a single assay measurement. Prediction files
  reference this one.

Both are content hashes: they depend only on cell values, never on row order,
so they are stable across reruns and across reshuffles of the dataset. They are
deliberately independent of derived/computed feature columns (fingerprints,
synthesizability scores, LLM annotations, ...), so adding such columns never
changes an id.
"""
import hashlib
from typing import List, Sequence

import pandas as pd

# Columns defining the experimental context of a measurement. Missing columns
# are skipped, so this list spans BOTH dataset schemas that flow through the
# pipeline — the published ``ailab-bio/TACK`` (``POI_*`` / ``Ligase_*`` /
# ``Assay_Time``) and the raw TACK2.0 CSV (``Degradation_Target_*`` /
# ``Recruiter_*``) — as well as reduced custom CSVs. Whichever names a given
# dataset carries participate in the hash; the target/recruiter identity must be
# among them or two measurements against different proteins in the same cell /
# assay collide. Keep both schemas represented here.
CONTEXT_COLUMNS: List[str] = [
    "SMILES",
    # Degradation target (protein of interest) identity.
    "POI_Name",
    "POI_UniProt",
    "POI_UniProt_MutationID",
    "Degradation_Target_Gene",
    "Degradation_Target_Uniprot",
    "Degradation_Target_Uniprot_MutationID",
    # Recruiter (E3 ligase) identity.
    "Ligase_Name",
    "Ligase_UniProt",
    "Recruiter_Gene",
    "Recruiter_Uniprot",
    # Assay context.
    "Cell_Line_ID",
    "Assay",
    "Assay_Time",
    "Reference",
]

# Raw readout columns added on top of the context to identify a single row.
# Spans both schemas: the published dataset stores the readout in a generic
# ``Value``/``Value_Type`` pair (or per-task ``Value_Dmax``/``Value_DC50`` in
# the multitask config), whereas TACK2.0 carries native ``DC50``/``Dmax`` (and
# their operator/range/duration companions).
READOUT_COLUMNS: List[str] = [
    "Value",
    "Value_Type",
    "Value_Dmax",
    "Value_DC50",
    "DC50",
    "DC50_operator",
    "DC50_h",
    "DC50_Range_Min",
    "DC50_Range_Max",
    "Dmax",
    "Dmax_operator",
    "Dmax_h",
    "Dmax_Range_Min",
    "Dmax_Range_Max",
    "Dmax_conc",
    "Dmax_conc_operator",
    "Dmax_conc_Range_Min",
    "Dmax_conc_Range_Max",
]

CONTEXT_ID_COLUMN = "context_id"
ROW_ID_COLUMN = "row_id"

# Length (hex chars) of the emitted identifiers; blake2b digest_size is half.
_ID_HEX_LEN = 16


def _hash_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    """Return a per-row content hash over ``columns``.

    The canonical string is ``col=value`` pairs joined by ``|`` with columns
    sorted by name, so the hash is independent of column ordering. Missing
    values are normalized to the empty string.

    Args:
        df: Source dataframe.
        columns: Columns to include; those absent from ``df`` are ignored.

    Returns:
        A string Series (indexed like ``df``) of ``_ID_HEX_LEN``-char hashes.
    """
    present = sorted(c for c in columns if c in df.columns)

    # Build the canonical "col=value" string per row, vectorized column by
    # column. NaN/None become empty strings so an absent readout is stable.
    key = pd.Series("", index=df.index, dtype="object")
    for i, col in enumerate(present):
        values = df[col].where(df[col].notna(), "").astype(str)
        sep = "|" if i else ""
        key = key + sep + col + "=" + values

    return key.map(
        lambda s: hashlib.blake2b(
            s.encode("utf-8"), digest_size=_ID_HEX_LEN // 2
        ).hexdigest()
    )


def assign_dataset_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``context_id`` and ``row_id`` columns to a copy of ``df``.

    Args:
        df: Dataset rows (one assay measurement per row).

    Returns:
        A copy of ``df`` with the two identifier columns populated. Any
        pre-existing identifier columns are overwritten.
    """
    df = df.copy()
    df[CONTEXT_ID_COLUMN] = _hash_columns(df, CONTEXT_COLUMNS)
    df[ROW_ID_COLUMN] = _hash_columns(df, list(CONTEXT_COLUMNS) + list(READOUT_COLUMNS))
    return df
