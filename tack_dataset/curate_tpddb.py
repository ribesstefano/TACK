# %% [markdown]
# # Data Curation
# 
# This notebook outlines the steps taken to curate and preprocess the parsed TPD-DB dataset for further analysis and modeling.

# %% [markdown]
# ## Setup

# %%
import os
import re
import textwrap
import random
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict

import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import rdkit
from rdkit import Chem
from scipy.stats import skew, kurtosis
from scipy.optimize import curve_fit

from tack_dataset.logging_utils import setup_logging
from tack_dataset.tpddb_parsing import extract_target_degradation_activities
from tack_dataset.tpddb_cleaning import clean_activity_data


# Setup logging
log_file = setup_logging(
    log_dir=Path('logs'),
    log_base_name='tpddb_curation',
    verbose=1, # Enable INFO level logging
)
logger = logging.getLogger(__name__)

logger.info(f"Log file: {log_file}")


# %%
# Get all txt files in the data/original directory
original_data_dir = Path(os.path.join(os.getcwd(), 'data', 'original'))
txt_files = list(original_data_dir.glob("*.txt"))
logger.info(f"Looking for text files in {original_data_dir} ...")
logger.info(f"Found {len(txt_files)} text files to process.")

tpd_ids = defaultdict(list)

for txt_file in txt_files:
    if "main_table" not in txt_file.name:
        continue
    mol_type = txt_file.name.split("_main_table.txt")[0]
    with open(txt_file, "r") as f:
        lines = f.readlines()[1:]

        for line in lines:
            tpd_id = line.strip().split("\t")[0].strip()
            if not tpd_id.startswith("TPD-"):
                continue
            # Skip empty lines
            if tpd_id:
                tpd_ids[mol_type].append(tpd_id)

# Print the collected TPD IDs
for mol_type, ids in tpd_ids.items():
    logger.info(f"{mol_type:10}{len(ids):7,}")

# %%
def open_csv(csv_filename: Path) -> pd.DataFrame:
    """ Open a CSV file and return a DataFrame. If the file is empty, return an
        empty DataFrame. If the file does not exist, raise a FileNotFoundError.
        
    Args:
        csv_filename (Path): Path to the CSV file.
        
    Returns:
        pd.DataFrame: DataFrame containing the CSV data or empty DataFrame.
    """
    if csv_filename.exists():
        try:
            return pd.read_csv(csv_filename)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    else:
        raise FileNotFoundError(f"{csv_filename} does not exist.")

# %% [markdown]
# ## Plotting % and Mol Values

# %%
df_names = [
    'cytotoxic_activities',
    'degradation_activities',
    "binding_preference",
    "binding_affinities",
]

output_file = Path("data/curation") / "activities.csv"
if output_file.parent.exists() and output_file.exists():
    degr_df = pd.read_csv(output_file)
    logger.info(f"Loaded existing DataFrame from {output_file} with shape: {degr_df.shape}")
else:
    degr_df = []

    data_dir = Path("data/parsed/")
    for name in df_names:
        for mol_type, ids in tpd_ids.items():
            for tpd_id in tqdm(ids, desc=f"Processing {mol_type}"):
                    csv_file = data_dir / name / f"{tpd_id}_{name}.csv"
                    if csv_file.exists():
                        degr_df.append(open_csv(csv_file))
                    else:
                        logger.info(f"Warning: {csv_file} does not exist. Skipping.")

    degr_df = pd.concat(degr_df, ignore_index=True)
    logger.info(f"Combined DataFrame shape: {degr_df.shape}")
    
    if not output_file.parent.exists():
        output_file.parent.mkdir(parents=True, exist_ok=True)
    degr_df.to_csv(output_file, index=False)
    logger.info(f"Saved combined DataFrame to {output_file}")

# %%
for col in degr_df.columns:
    num_unique = degr_df[col].nunique()
    logger.info(f"Column '{col}' has {num_unique:,} unique values.")

# %%
for k, c in degr_df['Value_Unit'].value_counts().to_dict().items():
    logger.info(f"{k:4} {c:6,}")
d = degr_df['Type_Base'].value_counts().to_dict()
# Sort dictionary by keys
d = dict(sorted(d.items()))
for k, c in d.items():
    logger.info(f"{k:23}{c:6,}")

conc_df = degr_df[degr_df['Value_Unit'].str.contains('M', na=False)]

val_cols = [c for c in conc_df.columns if 'Value' in c]
logger.info(val_cols)

# # Clean the 'range' values by calculating their mean and assigning it to 'Value_Mean'
def get_mean_range(row):
    if row['Value_Category'] == 'range':
        return (row['Value_Range_Min'] + row['Value_Range_Max']) / 2
    return row['Value_Mean']

# conc_df['Value_Mean'] = conc_df.apply(get_mean_range, axis=1)

for val in conc_df[conc_df['Value_Category'] == 'multiple']['Value'].unique():
    logger.info(val)

def pick_smallest_in_range(row):
    # Regex to remove all non-numeric characters and split by comma
    re_pattern = r'[^0-9.,-]'
    if row['Value_Category'] == 'multiple':
        values = [float(re.sub(re_pattern, '', v)) for v in row['Value'].split(',') if v]
        return min(values)
    return row['Value_Mean']

conc_df['Value_Mean'] = conc_df.apply(pick_smallest_in_range, axis=1)
conc_df['Value_Mean'].unique()

# %%
def convert_to_nM(row):
    """
    Convert all concentrations to nM for uniformity. Possible units: nM, μM, M, pM
    """
    if pd.isna(row['Value_Unit']) or pd.isna(row['Value_Mean']):
        return np.nan
    unit = row['Value_Unit']
    value = float(row['Value_Mean'])
    if unit == 'nM':
        return value
    elif unit == 'μM':
        return value * 1e3
    elif unit == 'M':
        return value * 1e9
    elif unit == 'pM':
        return value / 1e3
    elif unit == 'mM':
        return value * 1e6
    else:
        return np.nan

def get_mean_range(row):
    if row['Value_Category'] == 'range':
        return (row['Value_Range_Min'] + row['Value_Range_Max']) / 2
    return row['Value_Mean']

def pick_smallest_in_range(row):
    # Regex to remove all non-numeric characters and split by comma
    re_pattern = r'[^0-9.,-]'
    if row['Value_Category'] == 'multiple':
        values = [float(re.sub(re_pattern, '', v)) for v in row['Value'].split(',') if v]
        return min(values)
    return row['Value_Mean']

# Select rows with units containing 'M'
conc_df = degr_df[degr_df['Value_Unit'].str.contains('M', na=False)]

# # Filter only numeric value categories
# conc_df = conc_df[conc_df['Value_Category'] == 'numeric']
conc_df.loc[:, 'Value_Mean'] = conc_df.apply(pick_smallest_in_range, axis=1)
conc_df.loc[:, 'Value_Mean'] = conc_df.apply(get_mean_range, axis=1)

# Apply conversion to nM
conc_df.loc[:, 'Value_nM'] = conc_df.apply(convert_to_nM, axis=1)

# Drop rows where conversion failed
conc_df = conc_df.dropna(subset='Value_nM')

logger.info(f"Converted {conc_df['Value_nM'].notna().sum():,} values to nM (total: {len(conc_df):,}).")

# %%
# Sort the unique types according to their number of entries
type_order = conc_df['Type_Base'].value_counts().index.tolist()

plt.figure(figsize=(8, 5))

# Plot the 'Value_nM' distribution with log scale on x-axis
for type_base in type_order:
    logger.info(f"Plotting type: {type_base} with {len(conc_df[conc_df['Type_Base'] == type_base]):,} entries.")
    
    subset = conc_df[conc_df['Type_Base'] == type_base]
    
    # Group by 'Molecule_Type' and calculate the number of entries
    logger.info(subset['Molecule_Type'].value_counts())

    
    label = f"{type_base} (support: {len(subset):,})"
    sns.histplot(subset['Value_nM'], log_scale=(True, False), kde=True, label=label, alpha=0.5, bins=30, line_kws={'linewidth':2})

plt.xlabel('Activity Value (nM)')
plt.ylabel('')
plt.grid(axis='both', alpha=0.5)
plt.legend(title='Activity Type')
plt.tight_layout()


# %%
odd_types = [
    'LNCaP',
    'VCaP',
    'MEC',
]

tmp = conc_df[conc_df['Type_Base'].isin(odd_types)]
for desc in tmp['Description'].unique():
    # Print the descriptions wrapped to 80 characters
    wrapped_desc = textwrap.fill(desc, width=80)
    logger.info(wrapped_desc)
    logger.info("-" * 80)

for ref in tmp['Reference'].unique():
    logger.info(ref)
logger.info("-" * 80)

for v in tmp['Value'].unique():
    logger.info(v)
logger.info("-" * 80)

for cell in tmp['Cell_Line'].unique():
    logger.info(cell)

# %%
dmax_df = degr_df[degr_df['Value_Unit'].str.contains('%', na=False)]
logger.info(f"Dmax DataFrame shape: {dmax_df.shape}")

# dmax_df = dmax_df[dmax_df['Value_Category'] == 'numeric']
# dmax_df = dmax_df[dmax_df['Value_Mean'].notna()]

dmax_df.loc[:, 'Value_Mean'] = dmax_df.apply(pick_smallest_in_range, axis=1)
dmax_df.loc[:, 'Value_Mean'] = dmax_df.apply(get_mean_range, axis=1)

# Sort the unique types according to their number of entries
type_order = dmax_df['Type_Base'].value_counts().index.tolist()

plt.figure(figsize=(8, 5))

for type_base in type_order:
    subset = dmax_df[dmax_df['Type_Base'] == type_base]
    
    # Group by 'Molecule_Type' and calculate the number of entries
    logger.info(f"Plotting type: {type_base} with {len(subset):,} entries.")
    logger.info(subset['Molecule_Type'].value_counts())

    
    label = f"{type_base} (support: {len(subset):,})"
    sns.histplot(subset['Value_Mean'], kde=True, label=label, alpha=0.5, bins=30, line_kws={'linewidth':2})

# sns.histplot(data=dmax_df, x='Value_Mean', bins=30, kde=True)
plt.xlabel('Value (%)')
plt.ylabel('')
plt.grid(axis='both', alpha=0.5)
plt.legend(title='Activity Type')
plt.tight_layout()


# %%
# List 10 TPD-IDs related to 'Degradation'
for tpd_id in dmax_df[dmax_df['Type_Base'] == 'Ymin']['TPD_ID'].unique()[:10]:
    logger.info(f"• {tpd_id}")

# %% [markdown]
# ## Join Tables into a Single DataFrame

# %% [markdown]
# Get all mode of actions in order to get the Ligase and POI information.

# %%
protac_ids = tpd_ids['PROTAC'] + tpd_ids['MG']
logger.info(f"Total PROTAC IDs: {len(protac_ids):,}")
# Remove duplicates and reprint
protac_ids = list(set(protac_ids))
logger.info(f"Unique PROTAC IDs: {len(protac_ids):,}")

moa_csv = Path("data/curation/protac_mode_of_action.csv")

if moa_csv.exists():
    moa_df = pd.read_csv(moa_csv)
    logger.info(f"Loaded existing MOA DataFrame from {moa_csv} with shape: {moa_df.shape}")
else:
    moa_dir = Path("data/parsed/mode_of_action")
    # Get all mode_of_action files for PROTACs
    # moa_files = [moa_dir / f"{tpd_id}_mode_of_action.csv" for tpd_id in protac_ids]
    # moa_dfs = [open_csv(moa_file) for moa_file in moa_files]
    # moa_df = pd.concat([df for df in moa_dfs if not df.empty], ignore_index=True)
    moa_df = []
    for tpd_id in tqdm(protac_ids, desc="Processing MOA files"):
        moa_file = moa_dir / f"{tpd_id}_mode_of_action.csv"
        df = open_csv(moa_file)
        if not df.empty:
            moa_df.append(df)
    moa_df = pd.concat(moa_df, ignore_index=True)
    moa_df = moa_df.dropna(how='all').drop_duplicates().reset_index(drop=True)
    logger.info(f"Combined MOA DataFrame shape: {moa_df.shape}")

    moa_df.to_csv(moa_csv, index=False)
    logger.info(f"Saved MOA DataFrame to {moa_csv}")

tpd_id = 'TPD-I5IVLV'
tpd_id = 'TPD-VGQ6IR'
tpd_id = 'TPD-ZUNETX'
tpd_id = 'TPD-VYYLKJ'
tpd_id = 'TPD-VGQ6IR'
tpd_id = 'TPD-SQ4BT1'

html_path = Path(f"data/html/{tpd_id}.html")
logger.debug(clean_activity_data(extract_target_degradation_activities(html_path)))
logger.debug(open_csv(Path(f"data/parsed/degradation_activities/{tpd_id}_degradation_activities.csv")))

parsed_dir = Path("data/parsed/")

# Sub-sample the protac_ids for quicker testing
random.seed(42)
subset_protac_ids = protac_ids
# subset_protac_ids = random.sample(protac_ids, 1000)

# Preload all required dataframes into dictionaries for faster access
info_dfs = {}
degr_dfs = {}
bind_dfs = {}
cyto_dfs = {}

logger.info(f"Preloading CSVs for {len(subset_protac_ids):,} PROTAC IDs...")
logger.info("This will take approximately 5~6 minutes for processing all IDs...")


for tpd_id in tqdm(subset_protac_ids, desc="Preloading CSVs"):
    info_dfs[tpd_id] = open_csv(parsed_dir / 'general_info' / f"{tpd_id}_general_info.csv")
    degr_dfs[tpd_id] = open_csv(parsed_dir / 'degradation_activities' / f"{tpd_id}_degradation_activities.csv")
    bind_dfs[tpd_id] = open_csv(parsed_dir / 'binding_affinities' / f"{tpd_id}_binding_affinities.csv")
    cyto_dfs[tpd_id] = open_csv(parsed_dir / 'cytotoxic_activities' / f"{tpd_id}_cytotoxic_activities.csv")

# Pre-index moa_df by TPD_ID and POI_ID for fast lookup
moa_by_tpd = moa_df.groupby('TPD_ID')
moa_by_poi = moa_df.groupby('POI_ID')
moa_by_name = moa_df.groupby('Name')

# %%
def canonicalize_smiles(smiles):
    """ Convert a SMILES string to its canonical form using RDKit. """
    if pd.isnull(smiles):
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol, canonical=True)

def convert_to_nM(row):
    """
    Convert all concentrations to nM for uniformity. Possible units: nM, μM, M, pM
    """
    if pd.isna(row['Value_Unit']) or pd.isna(row['Value_Mean']):
        return np.nan
    unit = row['Value_Unit']
    value = float(row['Value_Mean'])
    if unit == 'nM':
        return value
    elif unit == 'μM':
        return value * 1e3
    elif unit == 'M':
        return value * 1e9
    elif unit == 'pM':
        return value / 1e3
    elif unit == 'mM':
        return value * 1e6
    else:
        return np.nan

def get_mean_range(row):
    if row['Value_Category'] == 'range':
        return (row['Value_Range_Min'] + row['Value_Range_Max']) / 2
    return row['Value_Mean']

def pick_smallest_in_range(row):
    # Regex to remove all non-numeric characters and split by comma
    if row['Value_Category'] == 'multiple':
        re_pattern = r'[^0-9.,-]'
        values = [float(re.sub(re_pattern, '', v)) for v in row['Value'].split(',') if v]
        return min(values)
    return row['Value_Mean']

def extract_nM_concentration(assay):
    """ Extract concentration value from assay string and convert to nM."""
    match = re.search(r'\((\d*\.?\d+)(nM|μM|M|pM|mM)\)', str(assay))
    if match:
        conc_value = float(match.group(1))
        conc_unit = match.group(2)
        # Convert to nM
        if conc_unit == 'nM':
            return conc_value
        elif conc_unit == 'μM':
            return conc_value * 1e3
        elif conc_unit == 'M':
            return conc_value * 1e9
        elif conc_unit == 'pM':
            return conc_value / 1e3
        elif conc_unit == 'mM':
            return conc_value * 1e6
    else:
        return np.nan

def clean_assay(assay):
    """ Clean the assay string by removing concentration info in parentheses."""
    if pd.isnull(assay):
        return assay
    assay = re.sub(r'\s*\(.*?\)\s*', '', assay).strip()
    dirty2clean ={
        'Htrf': 'HTRF',
        'WB': 'Western Blot',
        'WesternBlot': 'Western Blot',
        'Western blot': 'Western Blot',
        'In-cell Western': 'In-Cell Western',
        'Flow cytometry': 'Flow Cytometry',
        'High-Content Analysis(HCA)': 'High-Content Analysis (HCA)',
        'High-Content Imaging, HCA': 'High-Content Analysis (HCA)',
        'CKlα NanoBit Assay': 'CK1α NanoBiT Assay',
        'Nano-Glo HiBiT Lytic Assay': 'Nano-Glo HiBiT Lytic',
        'Enzyme Fragment Complementation, EFC(Prolabel Assay)': 'Enzyme Fragment Complementation, EFC (Prolabel Assay)'
    }
    assay = dirty2clean.get(assay, assay)
    # Capitalize first letter only
    assay = assay[0].upper() + assay[1:] if len(assay) > 1 else assay.upper()
    return assay

final_records = []

for tpd_id in tqdm(subset_protac_ids, desc="Curating final PROTAC DataFrame"):
    info_df = info_dfs[tpd_id]
    degr_df = degr_dfs[tpd_id]
    bind_df = bind_dfs[tpd_id]
    cyto_df = cyto_dfs[tpd_id]

    # Ligase info
    e3_info_rows = moa_by_tpd.get_group(tpd_id) if tpd_id in moa_by_tpd.groups else pd.DataFrame()
    e3_info = e3_info_rows[e3_info_rows['Type'] == 'Ligase'].iloc[0].to_dict() if not e3_info_rows.empty and any(e3_info_rows['Type'] == 'Ligase') else {}

    for df_name, activity_df in zip(['degr', 'bind', 'cyto'], [degr_df, bind_df, cyto_df]):
        if activity_df.empty or 'Value_Category' not in activity_df.columns or 'POI_ID' not in activity_df.columns:
            continue

        # Filter only rows with Type_Base in ['DC50', 'Dmax']
        filtered_rows = activity_df[activity_df['Type_Base'].isin(['DC50', 'Dmax'])]

        for _, row in filtered_rows.iterrows():
            row = row.copy()  # Avoid SettingWithCopyWarning

            row['Value_Mean'] = pick_smallest_in_range(row)
            row['Value_Mean'] = get_mean_range(row)
            if pd.notnull(row['Value_Unit']) and 'M' in row['Value_Unit']:
                row['Value_Mean'] = convert_to_nM(row)
            if pd.isnull(row['Value_Mean']):
                continue # Skip rows where Value_Mean could not be determined

            # POI info
            poi_info = {}
            if pd.notnull(row['POI_ID']) and row['POI_ID'] in moa_by_poi.groups:
                poi_info = moa_by_poi.get_group(row['POI_ID']).iloc[0].to_dict()
            elif pd.notnull(row.get('POI_Name')) and row['POI_Name'] in moa_by_name.groups:
                poi_info = moa_by_name.get_group(row['POI_Name']).iloc[0].to_dict()
            elif 'POI_ID' in activity_df.columns:
                other_poi_ids = activity_df[activity_df['TPD_ID'] == tpd_id]['POI_ID'].dropna().unique()
                if len(other_poi_ids) == 1 and other_poi_ids[0] in moa_by_poi.groups:
                    poi_info = moa_by_poi.get_group(other_poi_ids[0]).iloc[0].to_dict()
            if not poi_info:
                continue

            # Concentration/unit logic
            concentration = None
            concentration_unit = None
            if row['Type_Base'] == 'Dmax':
                if pd.notnull(row.get('Type_Concentration')) and pd.notnull(row.get('Type_Concentration_Unit')):
                    concentration = row.get('Type_Concentration')
                    concentration_unit = row.get('Type_Concentration_Unit')
                elif pd.notnull(row.get('Cell_Line_Concentration')) and pd.notnull(row.get('Cell_Line_Concentration_Unit')):
                    concentration = row.get('Cell_Line_Concentration')
                    concentration_unit = row.get('Cell_Line_Concentration_Unit')
                elif pd.notnull(row.get('Assay')):
                    conc_nM = extract_nM_concentration(row.get('Assay'))
                    if pd.notnull(conc_nM):
                        concentration = conc_nM
                        concentration_unit = 'nM'

            assay = clean_assay(row.get('Assay'))
            val_unit = row.get('Value_Unit')
            if pd.isnull(val_unit) and row['Type_Base'] == 'Dmax':
                val_unit = '%'
                
            operator = row.get('Value_Operator')
            if pd.notnull(operator):
                # Convert '≥' and '≤' to '>=' and '<='
                operator = operator.replace('≥', '>=').replace('≤', '<=')
                if operator == '*':
                    operator = pd.NA

            final_records.append({
                'TPD_ID': tpd_id,
                'SMILES': canonicalize_smiles(info_df['SMILES'].iloc[0]) if not info_df.empty else None,
                'Value': row['Value_Mean'],
                'Value_Type': row['Type_Base'],
                'Value_Unit': val_unit,
                'Value_Operator': operator,
                'Value_Category': row.get('Value_Category'),
                'Value_Range_Min': row.get('Value_Range_Min'),
                'Value_Range_Max': row.get('Value_Range_Max'),
                'Value_Error': row.get('Value_Error'),
                'Value_Concentration': concentration,
                'Value_Concentration_Unit': concentration_unit,
                'Type_Variant': row.get('Type_Variant'), # Used later to clean assay column
                'Ligase_Name': e3_info.get('Gene_Name'),
                'Ligase_Sequence': e3_info.get('Sequence'),
                'POI_Name': poi_info.get('Gene_Name'),
                'POI_Sequence': poi_info.get('Sequence'),
                'Cell_Line': row.get('Cell_Line'),
                'Cell_Line_ID': row.get('Cell_Line_ID'),
                'Modality': info_df['Type'].iloc[0] if not info_df.empty else None,
                'Assay': assay,
                'Reference': row.get('Reference'),
                'Description': row.get('Description'),
            })

final_df = pd.DataFrame(final_records)
# Convert all None to NaN
final_df = final_df.where(pd.notnull(final_df), None).drop_duplicates().reset_index(drop=True)

logger.info(f"Final curated DataFrame shape: {final_df.shape}")
logger.debug(final_df)

# Check non-null counts for each column
for col in final_df.columns:
    non_null_count = final_df[col].notna().sum()
    null_count = len(final_df) - non_null_count
    logger.info(f"Column '{col}' has {null_count:,} null values ({null_count / len(final_df):.2%}%)")

# %%
units = {
    "COMPOUNDS AND THEIR USE IN TREATING CANCER (patent)": 'nM',
    "2,6-PIPERIDINEDIONE COMPOUND AND APPLICATION THEREOF (patent)": 'nM',
    "IRAK4 DEGRADER AND USE THEREOF (patent)": 'nM',
    "AROMATIC COMPOUND, PHARMACEUTICAL COMPOSITION CONTAINING SAME, AND USE THEREOF (patent)": 'nM',
    "NEW TYPE BRD4 BROMODOMAIN PROTAC PROTEIN DEGRADATION AGENT, PREPARATION METHOD THEREFOR AND MEDICAL USE THEREOF (patent)": 'nM',
    "DEGRADATION OF BRUTON'S TYROSINE KINASE (BTK) BY CONJUGATION OF BTK INHIBITORS WITH E3 LIGASE LIGAND AND METHODS OF USE (patent)": 'nM',
    "CHIMERIC COMPOUND FOR TARGETED DEGRADATION OF ANDROGEN RECEPTOR PROTEIN, PREPARATION METHOD THEREFOR, AND MEDICAL USE THEREOF (patent)": 'nM',
    "BCL-2/BCL-XL PROTEIN DEGRADER AND USE THEREOF (patent)": 'μM',
    "FLUOROIMIDAZOPYRIDINE COMPOUND AS IRAK4 DEGRADATION AGENT AND USE THEREOF (patent)": 'nM',
    "NITROGEN-CONTAINING TRICYCLIC BIFUNCTIONAL COMPOUND, PREPARATION METHOD THEREFOR, AND APPLICATION THEREOF (patent)":'nM',
    "CYCLOBUTYL-CONTAINING COMPOUNDS (patent)": 'nM',
    "DEGRADATION OF BRUTON'S TYROSINE KINASE (BTK) BY CONJUGATION OF BTK INIDBITORS WITH E3 LIGASE LIGAND AND METHODS OF USE (patent)": 'μM',
    "PROTEIN DEGRADATION AGENT COMPOUND PREPARATION METHOD AND APPLICATION (patent)": 'nM',
    "GLUTARIMIDE-CONTAINING PAN-KRAS-MUTANT DEGRADER COMPOUNDS AND USES THEREOF (patent)": 'nM',
    "Improved small molecules (patent)": 'nM',
    "COMPOUND CONTAINING TRIFLUOROMETHYL GROUP (patent)": 'nM',
    "COMPOUND HAVING QUINAZOLINE STRUCTURE AND USE THEREOF (patent)": 'nM',
}
# Map the Reference to its corresponding unit in Value_Unit column if it is NaN
final_df['Value_Unit'] = final_df.apply(
    lambda row: units[row['Reference']] if pd.isnull(row['Value_Unit']) else row['Value_Unit'],
    axis=1
)

# Remove entries with unit equal to 'C'
final_df = final_df[final_df['Value_Unit'] != 'C'].reset_index(drop=True)

logger.info("Patents of DC50 without unit:")
for ref in final_df[final_df['Value_Unit'].isnull()]['Reference'].unique():
    logger.info(f"• {ref}")
logger.info("TPD IDs of DC50 without unit:")
for tpd_id in final_df[final_df['Value_Unit'].isnull()]['TPD_ID'].unique()[:20]:
    logger.info(f"• https://tpddb.idrblab.net/data/tpd/details/{tpd_id}")

# %%
def clean_assay(row: pd.Series) -> pd.Series:
    """ Clean the assay according to specific patent-related information. 
    
    For patent 'BIFUNCTIONAL COMPOUNDS FOR DEGRADATION OF EGFR AND RELATED METHODS OF USE (patent)',
    append mutation information to the assay name based on 'Type_Variant' column.
    """
    assay = row.get('Assay')
    if pd.isnull(assay):
        return row

    refs = [
        'BIFUNCTIONAL COMPOUNDS FOR DEGRADATION OF EGFR AND RELATED METHODS OF USE (patent)',
        'SUBSTITUTED 2,3-BENZODIAZEPINES DERIVATIVES (patent)',
    ]
    if row.get('Reference', '') not in refs:
        return row

    if row.get('Reference', '') == refs[0]:
        if row['Type_Variant'] == 3:
            assay_cleaned = f'{assay} DTC (Del19 / T790M / C797S triple mutation)'
        elif row['Type_Variant'] == 4:
            assay_cleaned = f'{assay} LTC (L858R / T790M / C797S triple mutation)'
        else:
            assay_cleaned = assay
    else:
        if row['Type_Variant'] == 1:
            assay_cleaned = f'{assay} (6 hours)'
        elif row['Type_Variant'] == 2:
            assay_cleaned = f'{assay} (24 hours)'
        else:
            assay_cleaned = assay
        
    row['Assay'] = assay_cleaned
    return row

final_df = final_df.apply(clean_assay, axis=1)
    
def get_assay_time(assay: str) -> int:
    if pd.isnull(assay):
        return None
    match = re.search(r'(\d+)\s*hours?', assay)
    if match:
        return int(match.group(1))
    else:
        return None

final_df['Assay_Time'] = final_df['Assay'].apply(get_assay_time)
# Remove the (6 hours) or (24 hours) from the assay names
final_df['Assay'] = final_df['Assay'].str.replace(r'\s*\(.*?hours?\)\s*', '', regex=True).str.strip()
# Drop the Type_Variant column as it is no longer needed
final_df = final_df.drop(columns=['Type_Variant'])

for assay in final_df['Assay'].dropna().unique():
    logger.info(assay)

# %%
# For each column in final_df, print the number of unique values and their counts
for col in final_df.columns:
    unique_values = final_df[col].nunique(dropna=True)
    logger.info(f"• Column '{col}' has {unique_values:,} unique values.")

# %%
# Group by key columns, then get the values with multiple Dmax values in 'Value_Type'
key_cols = ['TPD_ID', 'SMILES', 'POI_Name', 'Ligase_Name', 'Cell_Line_ID']
grouped = final_df.groupby(key_cols)
multi_dmax = grouped.filter(lambda x: (x['Value_Type'] == 'Dmax').sum() > 1)
logger.info(f"Entries with multiple Dmax values: {multi_dmax.shape[0]}")

# Remove entries from final_df that are in multi_dmax
final_df_cleaned = final_df.merge(multi_dmax[key_cols + ['Value_Type', 'Value']], on=key_cols + ['Value_Type', 'Value'], how='left', indicator=True)
final_df_cleaned = final_df_cleaned[final_df_cleaned['_merge'] == 'left_only'].drop(columns=['_merge'])
logger.info(f"Cleaned final DataFrame shape: {final_df_cleaned.shape}")

logger.debug(final_df_cleaned[final_df_cleaned['Modality'] != 'MG']['Value_Type'].value_counts())
logger.debug(final_df_cleaned[final_df_cleaned['Modality'].str.contains('MG')]['Value_Type'].value_counts())

grouped = final_df_cleaned.groupby(key_cols)
assert len(grouped.filter(lambda x: (x['Value_Type'] == 'Dmax').sum() > 1)) == 0

# %% [markdown]
# ## Curve-Fit Degradation Values

# %%
# Group by 'SMILES', 'POI_Name', 'Ligase_Name', 'Cell_Line_ID, then get the values with multiple Dmax values in 'Value_Type'
grouped = final_df.groupby(['TPD_ID', 'SMILES', 'POI_Name', 'Ligase_Name', 'Cell_Line_ID'])
single_dmax = grouped.filter(lambda x: (x['Value_Type'] == 'Dmax').sum() == 1).drop_duplicates().reset_index(drop=True)
single_dmax[single_dmax['Modality'] != 'MG']

# Group by 'SMILES', 'POI_Name', 'Ligase_Name', 'Cell_Line_ID, then get the values with multiple Dmax values in 'Value_Type'
grouped = final_df.groupby(['TPD_ID', 'SMILES', 'POI_Name', 'Ligase_Name', 'Cell_Line_ID'])
multi_dmax = grouped.filter(lambda x: (x['Value_Type'] == 'Dmax').sum() > 1)
multi_dc50 = grouped.filter(lambda x: (x['Value_Type'] == 'DC50').sum() > 1)

# Assign an ID string tuple in a different column for easier plotting
multi_dmax.loc[:, 'ID_Tuple'] = multi_dmax.apply(lambda row: f"{row['TPD_ID']},{row['SMILES']},{row['POI_Name']},{row['Ligase_Name']},{row['Cell_Line_ID']}", axis=1)
multi_dc50.loc[:, 'ID_Tuple'] = multi_dc50.apply(lambda row: f"{row['TPD_ID']},{row['SMILES']},{row['POI_Name']},{row['Ligase_Name']},{row['Cell_Line_ID']}", axis=1)

# Assign an integer ID for each unique ID_Tuple for easier plotting
unique_ids_dmax = multi_dmax['ID_Tuple'].unique()
id_map_dmax = {id_tuple: idx for idx, id_tuple in enumerate(unique_ids_dmax)}
multi_dmax.loc[:, 'ID_Tuple'] = multi_dmax['ID_Tuple'].map(id_map_dmax)
unique_ids_dc50 = multi_dc50['ID_Tuple'].unique()
id_map_dc50 = {id_tuple: idx for idx, id_tuple in enumerate(unique_ids_dc50)}
multi_dc50.loc[:, 'ID_Tuple'] = multi_dc50['ID_Tuple'].map(id_map_dc50)

# Boxplot of the Dmax values for the multiple entries
plt.figure(figsize=(20, 5))
sns.boxplot(data=multi_dmax, x='ID_Tuple', y='Value')
# plt.yscale('log')
plt.xlabel('Grouped by: TPD_ID, SMILES, POI_Name, Ligase_Name, Cell_Line_ID')
plt.ylabel('Value')
# Rotate x-ticks for better visibility, and reduce font size
plt.xticks(rotation=90)
plt.title('Boxplot of Dmax Values for Multiple Entries')
plt.grid(axis='y', alpha=0.5)
# Plot an horizontal line at y=0 and y=100
plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
plt.axhline(y=100, color='red', linestyle='--', linewidth=1)
plt.tight_layout()


plt.figure(figsize=(20, 5))
sns.boxplot(data=multi_dc50, x='ID_Tuple', y='Value')
plt.yscale('log')
plt.xlabel('Grouped by: TPD_ID, SMILES, POI_Name, Ligase_Name, Cell_Line_ID')
plt.ylabel('Value')
plt.xticks(rotation=90)
plt.title('Boxplot of DC50 Values for Multiple Entries')
plt.grid(axis='y', alpha=0.5)
# plt.legend(title='Type Base')
plt.tight_layout()


# Quantify the spread and skedasticity of the two dataframes
def quantify_spread_skewness(df: pd.DataFrame, value_col: str = 'Value') -> Dict[str, float]:
    """ Quantify the spread and skewness of the given DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        value_col (str): Column name containing the values to analyze.
        
    Returns:
        Dict[str, float]: Dictionary with spread and skewness metrics.
    """
    values = df[value_col].dropna().values
    spread = np.std(values)
    skewness = skew(values)
    kurt = kurtosis(values)
    
    return {
        'spread': spread,
        'skewness': skewness,
        'kurtosis': kurt,
    }
    
spread_skew_dmax = quantify_spread_skewness(multi_dmax[multi_dmax['Value_Type'] == 'Dmax'])
spread_skew_dc50 = quantify_spread_skewness(multi_dc50[multi_dc50['Value_Type'] == 'DC50'])
logger.info("Dmax Spread and Skewness:", spread_skew_dmax)
logger.info("DC50 Spread and Skewness:", spread_skew_dc50)

# 1. Define the 4PL (Four Parameter Logistic) function
def four_param_logistic(x, a, d, c, b):
    """
    a = min (bottom)
    d = max (top)
    c = EC50/IC50
    b = Hill slope
    """
    return d + (a - d) / (1.0 + (x / c)**b)

group_cols = ['TPD_ID', 'SMILES', 'POI_Name', 'Ligase_Name', 'Cell_Line_ID']
grouped = final_df.groupby(group_cols)
multi_dmax = grouped.filter(lambda x: (x['Value_Type'] == 'Dmax').sum() > 1)

unknown_conc_tpd_ids = []
failed_tpd_ids = []

# Loop over each group and fit the 4PL curve if there are enough data points
fit_results = []
for group_keys, dmax_df in multi_dmax.groupby(group_cols):
    
    # Get all Dmax values as y_data
    y_data = dmax_df['Value'].values
    # Cap the Dmax values at 100 if above it
    y_data = np.clip(y_data, None, 100)

    x_data = dmax_df['Value_Concentration'].values
    x_unit = dmax_df['Value_Concentration_Unit'].values[0]

    x_data = np.array(x_data)
    # Remove entries with NaN concentrations
    valid_indices = ~np.isnan(x_data)
    x_data = x_data[valid_indices]
        
    if len(x_data) == 0:
        unknown_conc_tpd_ids.append(group_keys[0])
        continue
    
    y_data = y_data[valid_indices]

    # Sort the values in ascending order of x_data
    sorted_indices = np.argsort(x_data)
    x_data = x_data[sorted_indices]
    y_data = y_data[sorted_indices]

    # Initial guesses and bounds
    # Parameters (ordered):
    # - a: min (bottom)
    # - d: max (top)
    # - c: EC50/IC50
    # - b: Hill slope
    p0 = [min(y_data), max(y_data), np.median(x_data), 1.0]
    # Bounds: ([low_a, low_d, low_c, low_b], [high_a, high_d, high_c, high_b])
    # bounds = [(min(y_data) - 10, max(y_data) - 10, 0, -5), (min(y_data) + 10, max(y_data) + 10, np.inf, 5)] # <- This worked
    bounds = [(-np.inf, -np.inf, 0, -5), (100, 100, np.inf, 5)]
    
    try:
        popt, _ = curve_fit(
            four_param_logistic,
            x_data,
            y_data,
            p0=p0,
            bounds=bounds
        )
        fit_results.append({
            'group_keys': group_keys,
            'fit_params': popt,
            'x_data': x_data,
            'y_data': y_data,
        })
    except Exception as e:
        logger.info(f"Failed to fit group {group_keys}: {e}")
        logger.info(f"• {dmax_df['Value_Concentration_Unit'].values[0]}:   {x_data}")
        logger.info(f"• Dmax: {y_data}")
        failed_tpd_ids.append(group_keys[0])
    
    if group_keys[0] in ['TPD-1EM6OB', 'TPD-3870VB', 'TPD-8D2P1I', 'TPD-G9A2M8']:
        # Plot the data and the fit
        plt.figure(figsize=(8, 5))
        plt.scatter(x_data, y_data, label='Data', color='blue')
        if 'popt' in locals():
            x_fit = np.logspace(np.log10(min(x_data)*0.1), np.log10(max(x_data)*10), 100)
            y_fit = four_param_logistic(x_fit, *popt)
            plt.plot(x_fit, y_fit, label='4PL Fit', color='red')
        plt.xscale('log')
        plt.ylim(-10, 110)
        plt.xlabel(f'Concentration ({x_unit})')
        plt.ylabel('Dmax (%)')
        plt.title(f"4PL Fit for TPD-ID: {group_keys[0]}")
        plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        plt.axhline(y=100, color='gray', linestyle='--', linewidth=1)
        # Plot the c_fit as a vertical line
        if 'popt' in locals():
            c_fit = popt[2]
            plt.axvline(x=c_fit, color='green', linestyle='-', linewidth=1, label=f'DC50 = {c_fit:.2f} {x_unit}')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)


# Print fit results
logger.info('-' * 80)
for result in fit_results:
    group_keys = result['group_keys']
    a_fit, d_fit, c_fit, b_fit = result['fit_params']
    logger.info(f"TPD-ID: https://tpddb.idrblab.net/data/tpd/details/{group_keys[0]} - Fit Params: Min={a_fit:.2f}, Max={d_fit:.2f}, DC50={c_fit:.2f}, Slope={b_fit:.2f}, X={result['x_data']}, Y={result['y_data']}")
    # logger.info(f"Group {group_keys} - Fit Params: Min={a_fit:.2f}, Max={d_fit:.2f}, DC50={c_fit:.2f}, Slope={b_fit:.2f}, X={result['x_data']}, Y={result['y_data']}")

# Print the TPD IDs with unknown concentration
logger.info('-' * 80)
logger.info(f"N. {len(unknown_conc_tpd_ids)} TPD IDs with unknown concentration:")
for tpd_id in set(unknown_conc_tpd_ids):
    logger.info(f"• https://tpddb.idrblab.net/data/tpd/details/{tpd_id}")

logger.info('-' * 80)
logger.info(f"N. {len(failed_tpd_ids)} TPD IDs with failed curve fitting:")
for tpd_id in set(failed_tpd_ids):
    logger.info(f"• https://tpddb.idrblab.net/data/tpd/details/{tpd_id}")

# %% [markdown]
# ## Save Curated DataFrame to CSV

# %%
# Save the final curated DataFrame
final_output_file = Path("data/curation") / "tpddb_protac_glues_dc50_dmax.csv"
final_df_cleaned.to_csv(final_output_file, index=False)
logger.info(f"Saved final curated DataFrame to: {final_output_file}")
final_df = pd.read_csv(final_output_file).reset_index(drop=True)

# %% [markdown]
# ## Plotting
# 
# Get a sense of the sizes of the train/vallidation/test splits:

# %%
logger.info(f"Number of entries with Dmax values: {final_df[final_df['Value_Type'] == 'Dmax'].shape[0]:,}")
logger.info(f"Number of entries with DC50 values: {final_df[final_df['Value_Type'] == 'DC50'].shape[0]:,}")
# Print the sizes of train/val/test splits
train_frac = 0.7
val_frac = 0.15
test_frac = 0.15
# For Dmax values
dmax_df = final_df[final_df['Value_Type'] == 'Dmax']
n_dmax = len(dmax_df)
n_dmax_train = int(n_dmax * train_frac)
n_dmax_val = int(n_dmax * val_frac)
n_dmax_test = n_dmax - n_dmax_train - n_dmax_val
logger.info(f"Dmax split sizes - Train: {n_dmax_train:,}, Val: {n_dmax_val:,}, Test: {n_dmax_test:,}")
# For DC50 values
dc50_df = final_df[final_df['Value_Type'] == 'DC50']
n_dc50 = len(dc50_df)
n_dc50_train = int(n_dc50 * train_frac)
n_dc50_val = int(n_dc50 * val_frac)
n_dc50_test = n_dc50 - n_dc50_train - n_dc50_val
logger.info(f"DC50 split sizes - Train: {n_dc50_train:,}, Val: {n_dc50_val:,}, Test: {n_dc50_test:,}")

# %%
# Plot the Dmax and DC50 value distributions in two separate plots side by side
plt.figure(figsize=(8, 5))
# Dmax plot
plt.subplot(1, 2, 1)
sns.histplot(final_df_cleaned[final_df_cleaned['Value_Type'] == 'Dmax']['Value'], kde=True, bins=30, color='skyblue', line_kws={'linewidth':2})
plt.title('Dmax Value Distribution')
plt.xlabel('Dmax (%)')
plt.ylabel('')
plt.grid(axis='both', alpha=0.5)
# DC50 plot
plt.subplot(1, 2, 2)
sns.histplot(final_df_cleaned[final_df_cleaned['Value_Type'] == 'DC50']['Value'], log_scale=(True, False), kde=True, bins=30, color='salmon', line_kws={'linewidth':2})
plt.title('DC50 Value Distribution')
plt.xlabel('DC50 (nM)')
plt.ylabel('')
plt.grid(axis='both', alpha=0.5)
plt.tight_layout()


# %%
# Print the number of unique SMILES in the final DataFrame
num_unique_smiles = final_df['SMILES'].nunique()
num_unique_smiles_dmax = final_df[final_df['Value_Type'] == 'Dmax']['SMILES'].nunique()
num_unique_smiles_dc50 = final_df[final_df['Value_Type'] == 'DC50']['SMILES'].nunique()
logger.info(f"Number of unique SMILES in the final DataFrame: {num_unique_smiles:,} ({num_unique_smiles / len(final_df):.2%}%)")
logger.info(f"Number of unique SMILES with Dmax values: {num_unique_smiles_dmax:,} ({num_unique_smiles_dmax / len(final_df[final_df['Value_Type'] == 'Dmax']):.2%}%)")
logger.info(f"Number of unique SMILES with DC50 values: {num_unique_smiles_dc50:,} ({num_unique_smiles_dc50 / len(final_df[final_df['Value_Type'] == 'DC50']):.2%}%)")

# %%
# Plot the categorical distribution of the following columns:
# 'Ligase_Name', 'POI_Name', 'Cell_Line'
categorical_columns = ['Ligase_Name', 'POI_Name', 'Cell_Line']
for col in categorical_columns:
    plt.figure(figsize=(10, 10))
    sns.countplot(data=final_df, y=col, order=final_df[col].value_counts().index)
    # Show the count on the bars
    for p in plt.gca().patches:
        width = p.get_width()
        plt.gca().text(width + 5, p.get_y() + p.get_height() / 2, f'{int(width):,}', va='center')
    plt.title(f'Distribution of {col}')
    plt.xlabel('Count')
    plt.ylabel(col)
    plt.grid(axis='x', alpha=0.5)
    plt.tight_layout()


# %% [markdown]
# ---

# %%
df_names = [
    'general_info',
    'mode_of_action',
    'cytotoxic_activities',
    'degradation_activities',
    'disease_info',
    "binding_preference",
    "binding_affinities",
]

tpd_id = "TPD-ZXVFY7"
parsed_dir = Path("data/parsed/")

info_df = open_csv(parsed_dir / 'general_info' / f"{tpd_id}_general_info.csv")
logger.debug(info_df)
joined_df = info_df.copy()

for name in df_names[1:]:
    csv_file = parsed_dir / name / f"{tpd_id}_{name}.csv"
    df = open_csv(csv_file)
    if df.empty:
        continue
    logger.info(name)
    logger.debug(df)

    # Determine the columns to join on
    on_cols = ['TPD_ID', 'Molecule_Type']
    if 'Reference' in df.columns and 'Reference' in joined_df.columns:
        on_cols.append('Reference')
    if 'POI_ID' in df.columns and 'POI_ID' in joined_df.columns:
        on_cols.append('POI_ID')

    joined_df = pd.merge(joined_df, df, how='left', on=on_cols)

logger.info('-' * 80)
logger.debug(joined_df)

for c in sorted(joined_df.columns):
    logger.info(c)