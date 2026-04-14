"""
Data Curation Pipeline for TPD-DB Dataset.

Reads pre-parsed per-compound CSV files (produced by tpddb_parsing.py),
joins them with mode-of-action data, and writes a curated CSV in the
standardized TACK schema.

Output columns (same schema as curate_protacpedia.py and curate_protacdb.py):
    TPD_ID, SMILES,
    POI_Name, POI_UniProt, POI_Sequence,
    Ligase_Name, Ligase_UniProt, Ligase_Sequence,
    Cell_Line, Cell_Line_ID,
    Value, Value_Type, Value_Unit, Value_Operator, Value_Error,
    Value_Category, Value_Range_Min, Value_Range_Max,
    Value_Concentration, Value_Concentration_Unit,
    Assay, Assay_Time, Reference, Description, Modality, Database

Usage:
    python curate_tpddb.py --output-dir data/curation
    python curate_tpddb.py --output-dir data/curation --plot
"""

import argparse
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from tack_dataset.curation_utils import (
    canonicalize_smiles,
    convert_to_nM,
    normalize_operator,
    normalize_unit,
)
from tack_dataset.protein_utils import (
    fetch_uniprot_for_gene,
    fetch_uniprot_for_sequence,
    E3_TO_ORGANISM_TO_UNIPROT,
)
from tack_dataset.cell_utils import get_cell_species
from tack_dataset.logging_utils import setup_logging
from tack_dataset.tpddb_cleaning import clean_activity_data
from tack_dataset.tpddb_parsing import extract_target_degradation_activities
from tackai.data.embeddings.cell_embeddings import CellEmbedding

# Configure logging
log_file = setup_logging(
    # log_file=Path('logs') / 'tpddb_curation.log',
    log_dir=Path('logs'),
    log_base_name='tpddb_curation',
    verbose=1,
)
logger = logging.getLogger(__name__)
logger.info(f"Log file: {log_file}")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    """Configuration for the TPD-DB curation pipeline."""

    # Directories
    data_dir: str = 'data'
    output_dir: str = 'data/curation'

    # Output file
    output_file: str = 'tpddb_protac_glues_dc50_dmax.csv'

    # Activity types to keep in the final output
    value_types: List[str] = field(default_factory=lambda: ['DC50', 'Dmax'])

    # Known patents for which the value unit is missing in the raw data.
    # Maps reference string → correct unit.
    patent_units: Dict[str, str] = field(default_factory=lambda: {
        "COMPOUNDS AND THEIR USE IN TREATING CANCER (patent)": 'nM',
        "2,6-PIPERIDINEDIONE COMPOUND AND APPLICATION THEREOF (patent)": 'nM',
        "IRAK4 DEGRADER AND USE THEREOF (patent)": 'nM',
        "AROMATIC COMPOUND, PHARMACEUTICAL COMPOSITION CONTAINING SAME, AND USE THEREOF (patent)": 'nM',
        "NEW TYPE BRD4 BROMODOMAIN PROTAC PROTEIN DEGRADATION AGENT, PREPARATION METHOD THEREFOR AND MEDICAL USE THEREOF (patent)": 'nM',
        "DEGRADATION OF BRUTON'S TYROSINE KINASE (BTK) BY CONJUGATION OF BTK INHIBITORS WITH E3 LIGASE LIGAND AND METHODS OF USE (patent)": 'nM',
        "CHIMERIC COMPOUND FOR TARGETED DEGRADATION OF ANDROGEN RECEPTOR PROTEIN, PREPARATION METHOD THEREFOR, AND MEDICAL USE THEREOF (patent)": 'nM',
        "BCL-2/BCL-XL PROTEIN DEGRADER AND USE THEREOF (patent)": 'μM',
        "FLUOROIMIDAZOPYRIDINE COMPOUND AS IRAK4 DEGRADATION AGENT AND USE THEREOF (patent)": 'nM',
        "NITROGEN-CONTAINING TRICYCLIC BIFUNCTIONAL COMPOUND, PREPARATION METHOD THEREFOR, AND APPLICATION THEREOF (patent)": 'nM',
        "CYCLOBUTYL-CONTAINING COMPOUNDS (patent)": 'nM',
        "DEGRADATION OF BRUTON'S TYROSINE KINASE (BTK) BY CONJUGATION OF BTK INIDBITORS WITH E3 LIGASE LIGAND AND METHODS OF USE (patent)": 'μM',
        "PROTEIN DEGRADATION AGENT COMPOUND PREPARATION METHOD AND APPLICATION (patent)": 'nM',
        "GLUTARIMIDE-CONTAINING PAN-KRAS-MUTANT DEGRADER COMPOUNDS AND USES THEREOF (patent)": 'nM',
        "Improved small molecules (patent)": 'nM',
        "COMPOUND CONTAINING TRIFLUOROMETHYL GROUP (patent)": 'nM',
        "COMPOUND HAVING QUINAZOLINE STRUCTURE AND USE THEREOF (patent)": 'nM',
    })


# =============================================================================
# Helper Functions
# =============================================================================

def open_csv(csv_path: Path) -> pd.DataFrame:
    """
    Open a CSV file and return a DataFrame.

    Returns an empty DataFrame if the file is empty.
    Raises FileNotFoundError if the file does not exist.
    """
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    raise FileNotFoundError(f"File not found: {csv_path}")


def get_value_mean(row: pd.Series) -> Optional[float]:
    """
    Resolve the best single numeric value from a parsed activity row.

    For 'multiple' category rows, returns the smallest value.
    For 'range' category rows, returns the midpoint.
    Otherwise returns Value_Mean as-is.
    """
    if row['Value_Category'] == 'multiple':
        re_pattern = r'[^0-9.,-]'
        try:
            values = [float(re.sub(re_pattern, '', v))
                      for v in str(row['Value']).split(',') if v.strip()]
            return min(values) if values else None
        except (ValueError, AttributeError):
            return None

    if row['Value_Category'] == 'range':
        lo = row.get('Value_Range_Min')
        hi = row.get('Value_Range_Max')
        if pd.notnull(lo) and pd.notnull(hi):
            return (float(lo) + float(hi)) / 2
        return row.get('Value_Mean')

    return row.get('Value_Mean')


def extract_nM_concentration(assay: str) -> Optional[float]:
    """
    Extract a concentration value from an assay string and return it in nM.

    Matches patterns like '(100nM)', '(1μM)', '(0.5M)'.
    Returns None if no concentration is found.
    """
    match = re.search(r'\((\d*\.?\d+)(nM|μM|M|pM|mM)\)', str(assay))
    if match:
        return convert_to_nM(float(match.group(1)), match.group(2))
    return None


def clean_assay_name(assay: str) -> str:
    """
    Standardize an assay name string.

    - Removes parenthetical concentration info, e.g. '(100nM)'.
    - Applies a small set of known spelling fixes.
    - Capitalizes the first letter.
    """
    if pd.isnull(assay):
        return assay

    assay = re.sub(r'\s*\(.*?\)\s*', '', assay).strip()

    name_fixes = {
        'Htrf':                                                   'HTRF',
        'WB':                                                     'Western Blot',
        'WesternBlot':                                            'Western Blot',
        'Western blot':                                           'Western Blot',
        'In-cell Western':                                        'In-Cell Western',
        'Flow cytometry':                                         'Flow Cytometry',
        'High-Content Analysis(HCA)':                             'High-Content Analysis (HCA)',
        'High-Content Imaging, HCA':                              'High-Content Analysis (HCA)',
        'CKlα NanoBit Assay':                                     'CK1α NanoBiT Assay',
        'Nano-Glo HiBiT Lytic Assay':                             'Nano-Glo HiBiT Lytic',
        'Enzyme Fragment Complementation, EFC(Prolabel Assay)':   'Enzyme Fragment Complementation, EFC (Prolabel Assay)',
    }
    assay = name_fixes.get(assay, assay)

    if assay:
        assay = assay[0].upper() + assay[1:]

    return assay


def get_assay_time(assay: str) -> Optional[int]:
    """
    Extract the treatment time in hours from an assay string.

    Matches patterns like '(6 hours)', '(24 hours)'.
    Returns None if not found.
    """
    if pd.isnull(assay):
        return None
    match = re.search(r'(\d+)\s*hours?', assay)
    if match:
        return int(match.group(1))
    return None


# =============================================================================
# Main Curation Class
# =============================================================================

class TpddbCurator:
    """Curates the pre-parsed TPD-DB dataset into the standardized TACK schema."""

    def __init__(self, config: Config):
        self.config = config
        self.data_dir = Path(config.data_dir)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.parsed_dir = self.data_dir / 'parsed'
        self.cell_embedding = CellEmbedding()

    # ------------------------------------------------------------------
    # Step 1: Load TPD IDs
    # ------------------------------------------------------------------

    def load_tpd_ids(self) -> Dict[str, List[str]]:
        """
        Read TPD IDs from the *_main_table.txt files in data/original/.

        Returns a dict mapping molecule type (e.g. 'PROTAC', 'MG') to a list
        of unique TPD IDs.
        """
        original_dir = self.data_dir / 'original'
        tpd_ids: Dict[str, List[str]] = defaultdict(list)

        for txt_file in original_dir.glob('*.txt'):
            if 'main_table' not in txt_file.name:
                continue
            mol_type = txt_file.name.replace('_main_table.txt', '')
            with open(txt_file) as f:
                for line in f.readlines()[1:]:  # skip header
                    tpd_id = line.strip().split('\t')[0].strip()
                    if tpd_id.startswith('TPD-'):
                        tpd_ids[mol_type].append(tpd_id)

        for mol_type, ids in tpd_ids.items():
            logger.info(f"  {mol_type:10} {len(ids):7,} IDs")

        return dict(tpd_ids)

    # ------------------------------------------------------------------
    # Step 2: Load mode-of-action table
    # ------------------------------------------------------------------

    def load_moa(self, protac_ids: List[str]) -> pd.DataFrame:
        """
        Load (or build) the mode-of-action table for PROTACs and MG.

        If the cached CSV exists, loads it. Otherwise reads individual
        per-compound CSVs and concatenates them.
        """
        moa_csv = self.output_dir / 'protac_mode_of_action.csv'

        if moa_csv.exists():
            moa_df = pd.read_csv(moa_csv)
            logger.info(f"Loaded MOA table from {moa_csv} ({len(moa_df):,} rows)")
            return moa_df

        logger.info("Building MOA table from individual CSVs...")
        moa_dfs = []
        moa_dir = self.parsed_dir / 'mode_of_action'
        for tpd_id in tqdm(protac_ids, desc="Loading MOA CSVs"):
            moa_file = moa_dir / f"{tpd_id}_mode_of_action.csv"
            df = open_csv(moa_file)
            if not df.empty:
                moa_dfs.append(df)

        moa_df = pd.concat(moa_dfs, ignore_index=True)
        moa_df = moa_df.dropna(how='all').drop_duplicates().reset_index(drop=True)
        moa_df.to_csv(moa_csv, index=False)
        logger.info(f"Saved MOA table to {moa_csv} ({len(moa_df):,} rows)")
        return moa_df

    # ------------------------------------------------------------------
    # Step 3: Preload per-compound CSVs
    # ------------------------------------------------------------------

    def load_all_csvs(self, protac_ids: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Preload general_info, degradation_activities, binding_affinities,
        and cytotoxic_activities CSVs for all PROTAC IDs.

        Returns a dict: {'info': {tpd_id: df}, 'degr': {tpd_id: df}, ...}
        """
        logger.info(f"Preloading CSVs for {len(protac_ids):,} PROTAC IDs...")
        data: Dict[str, Dict[str, pd.DataFrame]] = {
            'info': {}, 'degr': {}, 'bind': {}, 'cyto': {},
        }
        activity_dirs = {
            'info': 'general_info',
            'degr': 'degradation_activities',
            'bind': 'binding_affinities',
            'cyto': 'cytotoxic_activities',
        }
        for tpd_id in tqdm(protac_ids, desc="Preloading CSVs"):
            for key, subdir in activity_dirs.items():
                csv_path = self.parsed_dir / subdir / f"{tpd_id}_{subdir}.csv"
                data[key][tpd_id] = open_csv(csv_path)

        return data

    # ------------------------------------------------------------------
    # Step 4: Build final records
    # ------------------------------------------------------------------

    def build_records(
        self,
        protac_ids: List[str],
        csv_data: Dict[str, Dict[str, pd.DataFrame]],
        moa_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Main loop: for each PROTAC, join activity rows with MOA data.

        Only DC50 and Dmax rows are kept. POI info is looked up via the
        mode-of-action table using the POI_ID column.

        Returns a DataFrame with the standardized TACK columns.
        """
        # Index MOA by TPD_ID, POI_ID, and Name for fast lookups.
        moa_by_tpd  = moa_df.groupby('TPD_ID')
        moa_by_poi  = moa_df.groupby('POI_ID') if 'POI_ID' in moa_df.columns else {}
        moa_by_name = moa_df.groupby('Name')   if 'Name'   in moa_df.columns else {}

        records = []

        for tpd_id in tqdm(protac_ids, desc="Building records"):
            info_df = csv_data['info'][tpd_id]

            smiles = None
            if not info_df.empty and 'SMILES' in info_df.columns:
                smiles = canonicalize_smiles(info_df['SMILES'].iloc[0])

            modality = None
            if not info_df.empty and 'Type' in info_df.columns:
                modality = info_df['Type'].iloc[0]

            # Look up the E3 ligase for this compound.
            e3_info = {}
            if tpd_id in moa_by_tpd.groups:
                moa_rows = moa_by_tpd.get_group(tpd_id)
                ligase_rows = moa_rows[moa_rows['Type'] == 'Ligase']
                if not ligase_rows.empty:
                    e3_info = ligase_rows.iloc[0].to_dict()

            # Process each activity type.
            for key in ('degr', 'bind', 'cyto'):
                activity_df = csv_data[key][tpd_id]

                required_cols = {'Value_Category', 'POI_ID'}
                if activity_df.empty or not required_cols.issubset(activity_df.columns):
                    continue

                target_rows = activity_df[
                    activity_df['Type_Base'].isin(self.config.value_types)
                ]

                for _, row in target_rows.iterrows():
                    row = row.copy()

                    # Resolve a single numeric value from the row.
                    value_mean = get_value_mean(row)
                    if pd.isnull(value_mean):
                        continue

                    # Convert molar concentrations to nM.
                    val_unit = row.get('Value_Unit')
                    if pd.notnull(val_unit) and 'M' in str(val_unit):
                        value_mean = convert_to_nM(value_mean, val_unit)
                        if pd.isnull(value_mean):
                            continue
                        val_unit = 'nM'

                    # Fall back to '%' for Dmax rows with a missing unit.
                    if pd.isnull(val_unit) and row['Type_Base'] == 'Dmax':
                        val_unit = '%'

                    val_unit = normalize_unit(val_unit)

                    # Look up the POI (target protein).
                    poi_info = self._find_poi_info(row, tpd_id, activity_df, moa_by_poi, moa_by_name)
                    if not poi_info:
                        continue
                    
                    # Clean cell line ID, by taking the first one if ',' is present
                    if 'Cell_Line_ID' in row and pd.notnull(row['Cell_Line_ID']):
                        row['Cell_Line_ID'] = str(row['Cell_Line_ID']).split(',')[0].strip()
                    
                    # If the cell line ID is not NaN, use the cell embedding to
                    # stardardize the cell line name
                    if 'Cell_Line_ID' in row and pd.notnull(row['Cell_Line_ID']):
                        standardized_name = self.cell_embedding.cell_id2data.get(row['Cell_Line_ID'], {}).get('ID')
                        if standardized_name:
                            row['Cell_Line'] = standardized_name

                    # Get the species of the cell line using the cell embedding
                    species = get_cell_species(row.get('Cell_Line'), cell_embedding=self.cell_embedding, cell_id=row.get('Cell_Line_ID'))
                    if pd.isnull(species):
                        species = 'Homo sapiens'
                    
                    # Based on the species, look up UniProt IDs for the POI and
                    # E3 ligase if not already present in the MOA table.
                    poi_infos = fetch_uniprot_for_gene(poi_info.get('Gene_Name'), species)
                    if poi_info is None:
                        poi_infos = fetch_uniprot_for_sequence(poi_info.get('Sequence'))
                    poi_uniprot_id = poi_infos['uniprot'] if poi_infos is not None else None
                    if poi_uniprot_id is None:
                        logger.warning(f"Failed to find UniProt for POI: {poi_info.get('Gene_Name')}")
                        manual_map = {
                            'BCR/ABL': 'Q16189',
                            'AR3': 'P10275',
                        }
                        poi_uniprot_id = manual_map.get(poi_info.get('Gene_Name'))
                    e3_uniprot_id = E3_TO_ORGANISM_TO_UNIPROT.get(species, {}).get(e3_info.get('Gene_Name')) if e3_info else None
                    # logger.info(f"Species for {row.get('Cell_Line')} (ID: {row.get('Cell_Line_ID')}): {species} - POI UniProt: {poi_uniprot_id}, E3 UniProt: {e3_uniprot_id}")

                    # For Dmax, try to extract the assay concentration.
                    concentration, concentration_unit = None, None
                    if row['Type_Base'] == 'Dmax':
                        concentration, concentration_unit = self._find_concentration(row)

                    records.append({
                        'TPD_ID':                    tpd_id,
                        'SMILES':                    smiles,
                        'POI_Name':                  poi_info.get('Gene_Name'),
                        'POI_UniProt':               poi_info.get('UniProt_ID', poi_uniprot_id),
                        'POI_Sequence':              poi_info.get('Sequence'),
                        'Ligase_Name':               e3_info.get('Gene_Name'),
                        'Ligase_UniProt':            e3_info.get('UniProt_ID', e3_uniprot_id),
                        'Ligase_Sequence':           e3_info.get('Sequence'),
                        'Cell_Line':                 row.get('Cell_Line'),
                        'Cell_Line_ID':              row.get('Cell_Line_ID'),
                        'Cell_Line_Species':         species,
                        'Value':                     value_mean,
                        'Value_Type':                row['Type_Base'],
                        'Value_Unit':                val_unit,
                        'Value_Operator':            normalize_operator(str(row.get('Value_Operator') or '')),
                        'Value_Error':               row.get('Value_Error'),
                        'Value_Category':            row.get('Value_Category'),
                        'Value_Range_Min':           row.get('Value_Range_Min'),
                        'Value_Range_Max':           row.get('Value_Range_Max'),
                        'Value_Concentration':       concentration,
                        'Value_Concentration_Unit':  concentration_unit,
                        'Type_Variant':              row.get('Type_Variant'),  # used in clean_assay_info
                        'Assay':                     clean_assay_name(row.get('Assay')),
                        'Reference':                 row.get('Reference'),
                        'Description':               row.get('Description'),
                        'Modality':                  modality,
                        'Database':                  'TPD-DB',
                    })

        df = pd.DataFrame(records)
        df = df.where(pd.notnull(df), None).drop_duplicates().reset_index(drop=True)
        logger.info(f"Built {len(df):,} records from {len(protac_ids):,} PROTAC IDs")
        return df

    def _find_poi_info(self, row, tpd_id, activity_df, moa_by_poi, moa_by_name) -> dict:
        """Look up POI (target protein) info from the MOA table."""
        poi_id   = row.get('POI_ID')
        poi_name = row.get('POI_Name')

        if pd.notnull(poi_id) and poi_id in moa_by_poi.groups:
            return moa_by_poi.get_group(poi_id).iloc[0].to_dict()

        if pd.notnull(poi_name) and poi_name in moa_by_name.groups:
            return moa_by_name.get_group(poi_name).iloc[0].to_dict()

        # Try other POI IDs from the same compound in case this row's POI_ID is missing.
        if 'POI_ID' in activity_df.columns:
            other_ids = activity_df[
                activity_df['TPD_ID'] == tpd_id
            ]['POI_ID'].dropna().unique()
            if len(other_ids) == 1 and other_ids[0] in moa_by_poi.groups:
                return moa_by_poi.get_group(other_ids[0]).iloc[0].to_dict()

        return {}

    def _find_concentration(self, row) -> tuple:
        """
        Return (concentration, unit) for Dmax rows.

        Tries Type_Concentration, then Cell_Line_Concentration,
        then extracts from the Assay string.
        """
        if pd.notnull(row.get('Type_Concentration')) and pd.notnull(row.get('Type_Concentration_Unit')):
            return row['Type_Concentration'], row['Type_Concentration_Unit']

        if pd.notnull(row.get('Cell_Line_Concentration')) and pd.notnull(row.get('Cell_Line_Concentration_Unit')):
            return row['Cell_Line_Concentration'], row['Cell_Line_Concentration_Unit']

        if pd.notnull(row.get('Assay')):
            conc_nM = extract_nM_concentration(row['Assay'])
            if pd.notnull(conc_nM):
                return conc_nM, 'nM'

        return None, None

    # ------------------------------------------------------------------
    # Step 5: Fix missing units for known patents
    # ------------------------------------------------------------------

    def fix_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing Value_Unit for rows whose Reference is a known patent.

        Then removes any rows that still have the anomalous unit 'C'
        (an artifact from parsing errors in a few TPD-DB entries).
        """
        def fill_unit(row):
            if pd.isnull(row['Value_Unit']):
                return self.config.patent_units.get(row['Reference'], row['Value_Unit'])
            return row['Value_Unit']

        df['Value_Unit'] = df.apply(fill_unit, axis=1)
        df = df[df['Value_Unit'] != 'C'].reset_index(drop=True)

        still_missing = df[df['Value_Unit'].isnull()]
        if not still_missing.empty:
            logger.info(f"{len(still_missing)} rows still have no unit.")
            for ref in still_missing['Reference'].unique():
                logger.info(f"  • {ref}")

        return df

    # ------------------------------------------------------------------
    # Step 6: Clean assay info and extract Assay_Time
    # ------------------------------------------------------------------

    def clean_assay_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Append mutation/time variant info to assay names for specific patents,
        then extract Assay_Time and drop the temporary Type_Variant column.
        """
        # Two patents encode assay variant info in the Type_Variant column.
        egfr_patent = 'BIFUNCTIONAL COMPOUNDS FOR DEGRADATION OF EGFR AND RELATED METHODS OF USE (patent)'
        benzo_patent = 'SUBSTITUTED 2,3-BENZODIAZEPINES DERIVATIVES (patent)'

        def patch_assay(row):
            assay = row.get('Assay')
            ref   = row.get('Reference', '')
            var   = row.get('Type_Variant')

            if pd.isnull(assay) or ref not in (egfr_patent, benzo_patent):
                return assay

            if ref == egfr_patent:
                if var == 3:
                    return f"{assay} DTC (Del19 / T790M / C797S triple mutation)"
                if var == 4:
                    return f"{assay} LTC (L858R / T790M / C797S triple mutation)"

            if ref == benzo_patent:
                if var == 1:
                    return f"{assay} (6 hours)"
                if var == 2:
                    return f"{assay} (24 hours)"

            return assay

        df['Assay'] = df.apply(patch_assay, axis=1)

        # Extract treatment time from assay name, then strip the time from the string.
        df['Assay_Time'] = df['Assay'].apply(get_assay_time)
        df['Assay'] = df['Assay'].str.replace(r'\s*\(.*?hours?\)\s*', '', regex=True).str.strip()

        df = df.drop(columns=['Type_Variant'], errors='ignore')
        return df

    # ------------------------------------------------------------------
    # Step 7: Deduplicate
    # ------------------------------------------------------------------

    # def deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     Remove compound-cell groups that have more than one Dmax value.

    #     These ambiguous groups arise when multiple Dmax measurements at
    #     different concentrations are not tagged with a concentration.
    #     """
    #     key_cols = ['TPD_ID', 'SMILES', 'POI_Name', 'Ligase_Name', 'Cell_Line_ID']
    #     grouped = df.groupby(key_cols)
    #     multi_dmax = grouped.filter(lambda x: (x['Value_Type'] == 'Dmax').sum() > 1)

    #     logger.info(f"Entries with multiple Dmax values (to remove): {len(multi_dmax):,}")

    #     df_clean = df.merge(
    #         multi_dmax[key_cols + ['Value_Type', 'Value']],
    #         on=key_cols + ['Value_Type', 'Value'],
    #         how='left',
    #         indicator=True,
    #     )
    #     df_clean = df_clean[df_clean['_merge'] == 'left_only'].drop(columns=['_merge'])
    #     df_clean = df_clean.reset_index(drop=True)

    #     logger.info(f"After deduplication: {len(df_clean):,} rows")
    #     return df_clean

    def deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Deduplicate compound-cell groups by taking the median of Dmax values.
        For all other metadata, keep the most frequent non-NaN value.
        """
        key_cols = ['TPD_ID', 'SMILES', 'POI_Name', 'Ligase_Name', 'Cell_Line_ID']
        
        # Separate Dmax rows from the rest of the dataframe
        dmax_mask = df['Value_Type'] == 'Dmax'
        df_dmax = df[dmax_mask]
        df_other = df[~dmax_mask]
        
        # Safeguard if there are no Dmax values at all
        if df_dmax.empty:
            return df
            
        # Count groups with multiple Dmax values for logging
        dmax_counts = df_dmax.groupby(key_cols).size()
        multi_dmax_groups = (dmax_counts > 1).sum()
        logger.info(f"Groups with multiple Dmax values (to merge via median): {multi_dmax_groups:,}")

        # Helper function to find the most frequent non-NaN value
        def most_frequent(x):
            # value_counts automatically drops NaNs by default
            counts = x.value_counts()
            # Return the most frequent value, or pd.NA if the series is entirely empty/NaNs
            return counts.index[0] if not counts.empty else pd.NA

        # Create an aggregation dictionary: 
        # Calculate 'median' for Value, and use the custom mode function for metadata
        agg_dict = {col: most_frequent for col in df.columns if col not in key_cols}
        agg_dict['Value'] = 'median'

        # Group and aggregate Dmax rows
        df_dmax_clean = df_dmax.groupby(key_cols, as_index=False).agg(agg_dict)

        # Recombine the cleanly aggregated Dmax rows with the non-Dmax rows
        df_clean = pd.concat([df_other, df_dmax_clean], ignore_index=True)

        logger.info(f"After deduplication: {len(df_clean):,} rows")
        return df_clean

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def curate(self) -> pd.DataFrame:
        """Run the full curation pipeline and save the output CSV."""
        logger.info("Starting TPD-DB curation pipeline...")

        # Load TPD IDs
        tpd_ids = self.load_tpd_ids()
        protac_ids = list(set(tpd_ids.get('PROTAC', []) + tpd_ids.get('MG', [])))
        logger.info(f"Unique PROTAC+MG IDs: {len(protac_ids):,}")

        # Load MOA and per-compound CSVs
        moa_df = self.load_moa(protac_ids)
        csv_data = self.load_all_csvs(protac_ids)

        # Build, clean, deduplicate
        df = self.build_records(protac_ids, csv_data, moa_df)
        df = self.fix_units(df)
        df = self.clean_assay_info(df)
        df = self.deduplicate(df)

        # Save
        output_path = self.output_dir / self.config.output_file
        df.to_csv(output_path, index=False)
        logger.info(f"Saved curated data to {output_path}")

        self._print_summary(df)
        logging.info(f"Log file: {log_file}")
        return df

    def _print_summary(self, df: pd.DataFrame):
        logger.info("=" * 60)
        logger.info("Curation Summary")
        logger.info("=" * 60)
        logger.info(f"Total rows:        {len(df)}")
        logger.info(f"Unique SMILES:     {df['SMILES'].nunique()}")
        logger.info(f"Unique POIs:       {df['POI_Name'].nunique()}")
        logger.info(f"Unique ligases:    {df['Ligase_Name'].nunique()}")
        logger.info(f"Unique POI UniProt: {df['POI_UniProt'].nunique()}")
        logger.info(f"Unique ligase UniProt: {df['Ligase_UniProt'].nunique()}")
        for vt, count in df['Value_Type'].value_counts().items():
            logger.info(f"  {vt}: {count}")
        logger.info(
            f"POI sequences:    "
            f"{df['POI_Sequence'].notna().sum()}/{len(df)} "
            f"({100 * df['POI_Sequence'].notna().mean():.1f}%)"
        )
        logger.info(
            f"POI UniProt IDs:  "
            f"{df['POI_UniProt'].notna().sum()}/{len(df)} "
            f"({100 * df['POI_UniProt'].notna().mean():.1f}%)"
        )
        logger.info(
            f"Ligase UniProt IDs: "
            f"{df['Ligase_UniProt'].notna().sum()}/{len(df)} "
            f"({100 * df['Ligase_UniProt'].notna().mean():.1f}%)"
        )
        logger.info("=" * 60)


# =============================================================================
# Optional Plots
# =============================================================================

def plot_distributions(df: pd.DataFrame):
    """
    Plot Dmax and DC50 value distributions.

    Called only when --plot is passed on the command line.
    Requires matplotlib and seaborn.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    dmax_vals = df[df['Value_Type'] == 'Dmax']['Value'].dropna()
    dc50_vals = df[df['Value_Type'] == 'DC50']['Value'].dropna()

    sns.histplot(dmax_vals, ax=axes[0], kde=True, bins=30, color='skyblue')
    axes[0].set_title('Dmax Value Distribution')
    axes[0].set_xlabel('Dmax (%)')
    axes[0].grid(axis='both', alpha=0.5)

    sns.histplot(dc50_vals, ax=axes[1], kde=True, bins=30, color='salmon', log_scale=(True, False))
    axes[1].set_title('DC50 Value Distribution')
    axes[1].set_xlabel('DC50 (nM)')
    axes[1].grid(axis='both', alpha=0.5)

    plt.tight_layout()
    plt.savefig('tpddb_value_distributions.png', dpi=150)
    logger.info("Saved plot to tpddb_value_distributions.png")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Curate TPD-DB dataset from pre-parsed CSV files"
    )
    parser.add_argument('--data-dir',    default='data',
                        help='Root data directory (default: data)')
    parser.add_argument('--output-dir',  default='data/curation',
                        help='Output directory (default: data/curation)')
    parser.add_argument('--output-file', default='tpddb_protac_glues_dc50_dmax.csv',
                        help='Output CSV filename')
    parser.add_argument('--log-dir',     default='logs',
                        help='Directory for log files')
    parser.add_argument('--plot',        action='store_true',
                        help='Generate distribution plots after curation')
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        output_file=args.output_file,
    )
    curator = TpddbCurator(config)
    df = curator.curate()

    if args.plot:
        plot_distributions(df)


if __name__ == '__main__':
    main()
