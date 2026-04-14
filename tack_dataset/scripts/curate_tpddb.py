from __future__ import annotations
import argparse
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


# =============================================================================
# Constants
# =============================================================================

DATABASE_NAME = "TPD-DB"

PARSED_CSV_TYPES = (
    "general_info",
    "mode_of_action",
    "degradation_activities",
    "binding_affinities",
    "cytotoxic_activities",
)

ACTIVITY_SOURCES = (
    "degradation_activities",
    "binding_affinities",
    "cytotoxic_activities",
)

OUTPUT_COLUMNS = (
    "TPD_ID",
    "SMILES",
    "POI_Name",
    "POI_Sequence",
    "POI_UniProt",
    "Ligase_Name",
    "Ligase_Sequence",
    "Cell_Line",
    "Cell_Line_ID",
    "Reference",
    "Description",
    "Assay",
    "Modality",
    "Database",
    "DC50",
    "DC50_Unit",
    "DC50_Operator",
    "DC50_Category",
    "DC50_Range_Min",
    "DC50_Range_Max",
    "Dmax",
    "Dmax_Unit",
    "Dmax_Operator",
    "Dmax_Category",
    "Dmax_Range_Min",
    "Dmax_Range_Max",
    "Dmax_Concentration",
    "Dmax_Concentration_Unit",
)

UNIT_CONVERSION_TO_NM = {
    "nM": 1.0,
    "μM": 1e3,
    "M": 1e9,
    "pM": 1e-3,
    "mM": 1e6,
}

ASSAY_NAME_STANDARDIZATION = {
    "Htrf": "HTRF",
    "WB": "Western Blot",
    "WesternBlot": "Western Blot",
    "Western blot": "Western Blot",
    "In-cell Western": "In-Cell Western",
    "Flow cytometry": "Flow Cytometry",
    "High-Content Analysis(HCA)": "High-Content Analysis (HCA)",
    "High-Content Imaging, HCA": "High-Content Analysis (HCA)",
    "CKlα NanoBit Assay": "CK1α NanoBiT Assay",
    "Nano-Glo HiBiT Lytic Assay": "Nano-Glo HiBiT Lytic",
    "Enzyme Fragment Complementation, EFC(Prolabel Assay)": (
        "Enzyme Fragment Complementation, EFC (Prolabel Assay)"
    ),
}

COLUMN_RENAME_MAP = {
    "TPD ID": "TPD_ID",
    "TPD NAME": "TPD_Name",
    "Target ID": "POI_UniProt_Original",
    "Target Symbol": "POI_Symbol",
    "PubChem synonyms": "PubChem_Synonyms",
    "Fomula": "Formula",
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ActivityInfo:
    """Processed activity measurement data."""
    value: Any = None
    unit: Optional[str] = None
    operator: Optional[str] = None
    category: Optional[str] = None
    range_min: Optional[float] = None
    range_max: Optional[float] = None
    concentration: Optional[float] = None
    concentration_unit: Optional[str] = None
    cell_line: Optional[str] = None
    cell_line_id: Optional[str] = None
    description: Optional[str] = None
    assay: Optional[str] = None


@dataclass
class MoaIndex:
    """Indexed mode of action data for fast lookups."""
    by_tpd: dict = field(default_factory=dict)
    by_poi: dict = field(default_factory=dict)
    by_name: dict = field(default_factory=dict)
    df: pd.DataFrame = field(default_factory=pd.DataFrame)


# =============================================================================
# Utility Functions
# =============================================================================

def read_csv_safe(path: Path) -> pd.DataFrame:
    """Read CSV file, returning empty DataFrame if file doesn't exist or is empty."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def canonicalize_smiles(smiles: Optional[str]) -> Optional[str]:
    """Convert SMILES to canonical form using RDKit."""
    if pd.isnull(smiles) or not RDKIT_AVAILABLE:
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, canonical=True) if mol else smiles


def convert_to_nM(value: float, unit: str) -> float:
    """Convert concentration value to nanomolar."""
    if pd.isna(value) or pd.isna(unit):
        return np.nan
    multiplier = UNIT_CONVERSION_TO_NM.get(unit, 1.0)
    return float(value) * multiplier


def clean_operator(operator: Optional[str]) -> Optional[str]:
    """Standardize value operators (e.g., ≥ -> >=)."""
    if pd.isnull(operator):
        return None
    cleaned = str(operator).replace("≥", ">=").replace("≤", "<=")
    return cleaned if cleaned and cleaned != "*" else None


def clean_assay_name(assay: Optional[str]) -> Optional[str]:
    """Clean and standardize assay name."""
    if pd.isnull(assay):
        return None
    
    # Remove concentration info in parentheses
    cleaned = re.sub(r"\s*\(\d*\.?\d+\s*(nM|μM|M|pM|mM)\)\s*", "", str(assay)).strip()
    
    # Apply standardization mapping
    cleaned = ASSAY_NAME_STANDARDIZATION.get(cleaned, cleaned)
    
    # Capitalize first letter
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:] if len(cleaned) > 1 else cleaned.upper()
    
    return cleaned


def extract_concentration_from_assay(assay: Optional[str]) -> Optional[float]:
    """Extract concentration in nM from assay string like 'Assay (100 nM)'."""
    if pd.isnull(assay):
        return None
    match = re.search(r"\((\d*\.?\d+)\s*(nM|μM|M|pM|mM)\)", str(assay))
    if match:
        return convert_to_nM(float(match.group(1)), match.group(2))
    return None


# =============================================================================
# Data Loading
# =============================================================================

def load_tpd_ids(original_dir: Path, mol_types: Optional[list[str]] = None) -> dict[str, list[str]]:
    """
    Load TPD IDs from original main table files.
    
    Args:
        original_dir: Directory containing *_main_table.txt files.
        mol_types: Filter to specific molecule types (e.g., ['PROTAC', 'MG']).
    
    Returns:
        Dictionary mapping molecule type to list of TPD IDs.
    """
    txt_files = list(original_dir.glob("*_main_table.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No *_main_table.txt files found in {original_dir}")
    
    tpd_ids: dict[str, list[str]] = defaultdict(list)
    
    for txt_file in txt_files:
        mol_type = txt_file.stem.replace("_main_table", "")
        
        if mol_types and mol_type not in mol_types:
            continue
        
        with open(txt_file, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:  # Skip header
                parts = line.strip().split("\t")
                if parts and parts[0].startswith("TPD-"):
                    tpd_ids[mol_type].append(parts[0].strip())
    
    # Deduplicate
    return {k: list(set(v)) for k, v in tpd_ids.items()}


def load_original_data(original_dir: Path, mol_types: Optional[list[str]] = None) -> pd.DataFrame:
    """Load and combine original main table files."""
    dfs = []
    
    for txt_file in original_dir.glob("*_main_table.txt"):
        mol_type = txt_file.stem.replace("_main_table", "")
        
        if mol_types and mol_type not in mol_types:
            continue
        
        df = pd.read_csv(txt_file, sep="\t")
        df["Molecule_Type"] = mol_type
        dfs.append(df)
    
    if not dfs:
        return pd.DataFrame()
    
    combined = pd.concat(dfs, ignore_index=True)
    return combined.rename(columns=COLUMN_RENAME_MAP)


def preload_parsed_data(parsed_dir: Path, tpd_ids: list[str]) -> dict[str, dict[str, pd.DataFrame]]:
    """Preload all parsed CSV files for the given TPD IDs."""
    data = {csv_type: {} for csv_type in PARSED_CSV_TYPES}
    
    for tpd_id in tqdm(tpd_ids, desc="Loading parsed data"):
        for csv_type in PARSED_CSV_TYPES:
            csv_path = parsed_dir / csv_type / f"{tpd_id}_{csv_type}.csv"
            data[csv_type][tpd_id] = read_csv_safe(csv_path)
    
    return data


def build_moa_index(moa_data: dict[str, pd.DataFrame]) -> MoaIndex:
    """Build indexed lookups for mode of action data."""
    dfs = [df for df in moa_data.values() if not df.empty]
    
    if not dfs:
        return MoaIndex()
    
    moa_df = pd.concat(dfs, ignore_index=True).drop_duplicates().reset_index(drop=True)
    index = MoaIndex(df=moa_df)
    
    if "TPD_ID" in moa_df.columns:
        index.by_tpd = {tpd_id: group for tpd_id, group in moa_df.groupby("TPD_ID")}
    
    if "POI_ID" in moa_df.columns:
        index.by_poi = {poi_id: group for poi_id, group in moa_df.groupby("POI_ID")}
    
    if "Name" in moa_df.columns:
        index.by_name = {name: group for name, group in moa_df.groupby("Name")}
    
    return index


# =============================================================================
# Activity Processing
# =============================================================================

def _compute_numeric_value(row: pd.Series) -> Optional[float]:
    """Compute numeric value from activity row, handling ranges and multiples."""
    category = row.get("Value_Category")
    
    # Handle multiple values: pick smallest
    if category == "multiple":
        value_str = str(row.get("Value", ""))
        try:
            values = [
                float(re.sub(r"[^0-9.,-]", "", v))
                for v in value_str.split(",")
                if v.strip()
            ]
            if values:
                return min(values)
        except (ValueError, TypeError):
            pass
        return row.get("Value_Mean")
    
    # Handle range values: compute mean
    if category == "range":
        min_val = row.get("Value_Range_Min")
        max_val = row.get("Value_Range_Max")
        if pd.notna(min_val) and pd.notna(max_val):
            return (min_val + max_val) / 2
    
    return row.get("Value_Mean") or row.get("Value")


def _get_dmax_concentration(row: pd.Series) -> tuple[Optional[float], Optional[str]]:
    """Extract Dmax concentration from various possible sources."""
    # Try Type_Concentration first
    if pd.notna(row.get("Type_Concentration")) and pd.notna(row.get("Type_Concentration_Unit")):
        return row.get("Type_Concentration"), row.get("Type_Concentration_Unit")
    
    # Try Cell_Line_Concentration
    if pd.notna(row.get("Cell_Line_Concentration")) and pd.notna(row.get("Cell_Line_Concentration_Unit")):
        return row.get("Cell_Line_Concentration"), row.get("Cell_Line_Concentration_Unit")
    
    # Try extracting from Assay string
    if pd.notna(row.get("Assay")):
        conc_nM = extract_concentration_from_assay(row.get("Assay"))
        if pd.notna(conc_nM):
            return conc_nM, "nM"
    
    return None, None


def process_activity_row(row: pd.Series) -> ActivityInfo:
    """Process an activity row into structured ActivityInfo."""
    category = row.get("Value_Category")
    unit = row.get("Value_Unit")
    type_base = row.get("Type_Base")
    
    # Handle grade values (A, B, C, D) - preserve as string
    if category == "grade":
        raw_value = row.get("Value")
        value = str(raw_value).strip() if pd.notna(raw_value) else None
    else:
        # Process numeric values
        value = _compute_numeric_value(row)
        
        # Convert concentration units to nM
        if pd.notna(unit) and "M" in str(unit) and pd.notna(value):
            value = convert_to_nM(value, unit)
            unit = "nM"
    
    # Default unit for Dmax is %
    if pd.isnull(unit) and type_base == "Dmax":
        unit = "%"
    
    # Get Dmax concentration info
    concentration, concentration_unit = None, None
    if type_base == "Dmax":
        concentration, concentration_unit = _get_dmax_concentration(row)
    
    return ActivityInfo(
        value=value,
        unit=unit,
        operator=clean_operator(row.get("Value_Operator")),
        category=category,
        range_min=row.get("Value_Range_Min"),
        range_max=row.get("Value_Range_Max"),
        concentration=concentration,
        concentration_unit=concentration_unit,
        cell_line=row.get("Cell_Line"),
        cell_line_id=row.get("Cell_Line_ID"),
        description=row.get("Description"),
        assay=clean_assay_name(row.get("Assay")),
    )


# =============================================================================
# Dataset Creation
# =============================================================================

def _get_original_data(tpd_id: str, original_indexed: pd.DataFrame) -> dict[str, Any]:
    """Extract data from original main table for a TPD ID."""
    if tpd_id not in original_indexed.index:
        return {"smiles": None, "modality": None, "reference": None}
    
    orig = original_indexed.loc[tpd_id]
    if isinstance(orig, pd.DataFrame):
        orig = orig.iloc[0]
    
    return {
        "smiles": orig.get("SMILES"),
        "modality": orig.get("Molecule_Type"),
        "reference": orig.get("Source"),
    }


def _get_moa_info(tpd_id: str, moa_index: MoaIndex, entity_type: str) -> dict[str, Any]:
    """Get POI or Ligase info from mode of action data."""
    if tpd_id not in moa_index.by_tpd:
        return {}
    
    tpd_moa = moa_index.by_tpd[tpd_id]
    entity_rows = tpd_moa[tpd_moa["Type"] == entity_type]
    
    if entity_rows.empty:
        return {}
    
    return entity_rows.iloc[0].to_dict()


def _collect_activities(
    tpd_id: str,
    preloaded_data: dict[str, dict[str, pd.DataFrame]],
) -> tuple[list[pd.Series], list[pd.Series]]:
    """Collect DC50 and Dmax activities for a TPD ID."""
    dc50_activities = []
    dmax_activities = []
    
    for source in ACTIVITY_SOURCES:
        activity_df = preloaded_data[source].get(tpd_id, pd.DataFrame())
        
        if activity_df.empty or "Type_Base" not in activity_df.columns:
            continue
        
        for _, row in activity_df.iterrows():
            type_base = row.get("Type_Base")
            if type_base == "DC50":
                dc50_activities.append(row)
            elif type_base == "Dmax":
                dmax_activities.append(row)
    
    return dc50_activities, dmax_activities


def _build_record(
    tpd_id: str,
    original_data: dict[str, Any],
    poi_info: dict[str, Any],
    e3_info: dict[str, Any],
    dc50_info: Optional[ActivityInfo],
    dmax_info: Optional[ActivityInfo],
) -> dict[str, Any]:
    """Build a single record for the output dataset."""
    record = {
        "TPD_ID": tpd_id,
        "Database": DATABASE_NAME,
        "SMILES": canonicalize_smiles(original_data["smiles"]),
        "Modality": original_data["modality"],
        "Reference": original_data["reference"],
        "POI_Name": poi_info.get("Gene_Name"),
        "POI_Sequence": poi_info.get("Sequence"),
        "POI_UniProt": poi_info.get("POI_ID"),
        "Ligase_Name": e3_info.get("Gene_Name"),
        "Ligase_Sequence": e3_info.get("Sequence"),
        "Cell_Line": None,
        "Cell_Line_ID": None,
        "Description": None,
        "Assay": None,
        "DC50": None,
        "DC50_Unit": None,
        "DC50_Operator": None,
        "DC50_Category": None,
        "DC50_Range_Min": None,
        "DC50_Range_Max": None,
        "Dmax": None,
        "Dmax_Unit": None,
        "Dmax_Operator": None,
        "Dmax_Category": None,
        "Dmax_Range_Min": None,
        "Dmax_Range_Max": None,
        "Dmax_Concentration": None,
        "Dmax_Concentration_Unit": None,
    }
    
    # Populate DC50 data
    if dc50_info:
        record.update({
            "DC50": dc50_info.value,
            "DC50_Unit": dc50_info.unit,
            "DC50_Operator": dc50_info.operator,
            "DC50_Category": dc50_info.category,
            "DC50_Range_Min": dc50_info.range_min,
            "DC50_Range_Max": dc50_info.range_max,
        })
        # Use DC50 context if no context set yet
        if record["Cell_Line"] is None:
            record["Cell_Line"] = dc50_info.cell_line
            record["Cell_Line_ID"] = dc50_info.cell_line_id
            record["Description"] = dc50_info.description
            record["Assay"] = dc50_info.assay
    
    # Populate Dmax data
    if dmax_info:
        record.update({
            "Dmax": dmax_info.value,
            "Dmax_Unit": dmax_info.unit,
            "Dmax_Operator": dmax_info.operator,
            "Dmax_Category": dmax_info.category,
            "Dmax_Range_Min": dmax_info.range_min,
            "Dmax_Range_Max": dmax_info.range_max,
            "Dmax_Concentration": dmax_info.concentration,
            "Dmax_Concentration_Unit": dmax_info.concentration_unit,
        })
        # Use Dmax context if no context set yet
        if record["Cell_Line"] is None:
            record["Cell_Line"] = dmax_info.cell_line
            record["Cell_Line_ID"] = dmax_info.cell_line_id
            record["Description"] = dmax_info.description
            record["Assay"] = dmax_info.assay
    
    return record


def create_dataset(
    original_df: pd.DataFrame,
    preloaded_data: dict[str, dict[str, pd.DataFrame]],
    moa_index: MoaIndex,
    tpd_ids: list[str],
    info_fallback: bool = True,
) -> pd.DataFrame:
    """
    Create compound-centric dataset with all PROTACs.
    
    Args:
        original_df: Original main table data.
        preloaded_data: Preloaded parsed CSV data.
        moa_index: Indexed mode of action data.
        tpd_ids: List of TPD IDs to process.
        info_fallback: Whether to use general_info as fallback for SMILES/modality.
    
    Returns:
        DataFrame with one row per PROTAC.
    """
    records = []
    original_indexed = original_df.set_index("TPD_ID") if not original_df.empty else pd.DataFrame()
    
    for tpd_id in tqdm(tpd_ids, desc="Processing compounds"):
        # Get data from original table
        original_data = _get_original_data(tpd_id, original_indexed)
        
        # Fallback to general_info for SMILES/modality if needed
        if info_fallback:
            info_df = preloaded_data["general_info"].get(tpd_id, pd.DataFrame())
            if pd.isnull(original_data["smiles"]) and not info_df.empty and "SMILES" in info_df.columns:
                original_data["smiles"] = info_df["SMILES"].iloc[0]
            if pd.isnull(original_data["modality"]) and not info_df.empty and "Type" in info_df.columns:
                original_data["modality"] = info_df["Type"].iloc[0]
        
        # Get POI and ligase info
        poi_info = _get_moa_info(tpd_id, moa_index, "POI")
        e3_info = _get_moa_info(tpd_id, moa_index, "Ligase")
        
        # Collect activities
        dc50_activities, dmax_activities = _collect_activities(tpd_id, preloaded_data)
        
        # Process best activity values
        dc50_info = process_activity_row(dc50_activities[0]) if dc50_activities else None
        dmax_info = process_activity_row(dmax_activities[0]) if dmax_activities else None
        
        # Build and append record
        record = _build_record(tpd_id, original_data, poi_info, e3_info, dc50_info, dmax_info)
        records.append(record)
    
    if not records:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    
    df = pd.DataFrame(records)
    
    # Ensure all output columns exist and are in correct order
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = None
    
    return df[list(OUTPUT_COLUMNS)].drop_duplicates().reset_index(drop=True)


# =============================================================================
# Reporting
# =============================================================================

def print_summary(df: pd.DataFrame, output_path: Path) -> None:
    """Print dataset summary statistics."""
    print(f"\nOutput file: {output_path}")
    print(f"Total rows:  {len(df):,}")
    print(f"Columns:     {len(df.columns)}")
    
    if df.empty:
        return
    
    # Activity coverage
    has_dc50 = df["DC50"].notna().sum()
    has_dmax = df["Dmax"].notna().sum()
    has_both = (df["DC50"].notna() & df["Dmax"].notna()).sum()
    has_any = (df["DC50"].notna() | df["Dmax"].notna()).sum()
    has_none = len(df) - has_any
    
    print(f"\nActivity coverage:")
    print(f"  Rows with DC50:   {has_dc50:>6,} ({has_dc50/len(df)*100:5.1f}%)")
    print(f"  Rows with Dmax:   {has_dmax:>6,} ({has_dmax/len(df)*100:5.1f}%)")
    print(f"  Rows with both:   {has_both:>6,} ({has_both/len(df)*100:5.1f}%)")
    print(f"  Rows with any:    {has_any:>6,} ({has_any/len(df)*100:5.1f}%)")
    print(f"  Rows without any: {has_none:>6,} ({has_none/len(df)*100:5.1f}%)")
    
    # Column completeness
    print("\nColumn completeness:")
    for col in df.columns:
        non_null = df[col].notna().sum()
        pct = non_null / len(df) * 100
        print(f"  {col}: {non_null:,} ({pct:.1f}%)")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Curate TPDdb data into a compound-centric dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--parsed-dir",
        type=Path,
        default=Path("data/parsed"),
        help="Directory containing parsed CSV files (default: data/parsed)",
    )
    parser.add_argument(
        "--original-dir",
        type=Path,
        default=Path("data/original"),
        help="Directory containing original main table files (default: data/original)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/curated/tpddb_protac_all.csv"),
        help="Output CSV file path (default: data/curated/tpddb_protac_all.csv)",
    )
    parser.add_argument(
        "--mol-type",
        type=str,
        nargs="+",
        default=["PROTAC"],
        help="Molecule types to include, e.g., PROTAC MG (default: PROTAC)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Sample N TPD IDs for testing",
    )
    parser.add_argument(
        "--require-activity",
        action="store_true",
        help="Only include rows with DC50 or Dmax values",
    )
    parser.add_argument(
        "--require-both",
        action="store_true",
        help="Only include rows with both DC50 and Dmax values",
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    mol_types = None if "all" in args.mol_type else args.mol_type
    
    # Print configuration
    print("=" * 80)
    print("TPDdb Data Curation")
    print("=" * 80)
    print(f"Parsed directory:   {args.parsed_dir}")
    print(f"Original directory: {args.original_dir}")
    print(f"Output file:        {args.output}")
    print(f"Molecule types:     {mol_types or 'all'}")
    print(f"Require activity:   {args.require_activity}")
    print(f"Require both:       {args.require_both}")
    print("=" * 80)
    
    # Load TPD IDs
    print("\nLoading TPD IDs...")
    tpd_ids_by_type = load_tpd_ids(args.original_dir, mol_types)
    
    all_tpd_ids = []
    for mol_type, ids in tpd_ids_by_type.items():
        print(f"  {mol_type}: {len(ids):,} TPD IDs")
        all_tpd_ids.extend(ids)
    all_tpd_ids = list(set(all_tpd_ids))
    print(f"  Total unique: {len(all_tpd_ids):,}")
    
    # Sample if requested
    if args.sample:
        random.seed(42)
        all_tpd_ids = random.sample(all_tpd_ids, min(args.sample, len(all_tpd_ids)))
        print(f"  Sampled: {len(all_tpd_ids):,}")
    
    # Load original data
    print("\nLoading original data...")
    original_df = load_original_data(args.original_dir, mol_types)
    print(f"  Loaded {len(original_df):,} rows")
    
    # Preload parsed data
    print()
    preloaded_data = preload_parsed_data(args.parsed_dir, all_tpd_ids)
    
    # Build mode of action index
    print("\nIndexing mode of action data...")
    moa_index = build_moa_index(preloaded_data["mode_of_action"])
    print(f"  Indexed {len(moa_index.by_tpd):,} TPD IDs")
    
    # Create dataset
    print("\nCreating dataset...")
    df = create_dataset(
        original_df=original_df,
        preloaded_data=preloaded_data,
        moa_index=moa_index,
        tpd_ids=all_tpd_ids,
    )
    
    # Apply filters
    if args.require_both and not df.empty:
        before = len(df)
        df = df[df["DC50"].notna() & df["Dmax"].notna()]
        print(f"  Filtered to rows with both: {before:,} -> {len(df):,}")
    elif args.require_activity and not df.empty:
        before = len(df)
        df = df[df["DC50"].notna() | df["Dmax"].notna()]
        print(f"  Filtered to rows with activity: {before:,} -> {len(df):,}")
    
    # Save output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    
    # Print summary
    print("\n" + "=" * 80)
    print("Curation Complete!")
    print("=" * 80)
    print_summary(df, args.output)
    print("=" * 80)


if __name__ == "__main__":
    if not RDKIT_AVAILABLE:
        print("Warning: RDKit not available. SMILES canonicalization will be skipped.")
    main()
