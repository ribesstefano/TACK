# %% [markdown]
# # Curate PROTAC-DB, TPDdb, and PROTAC-Pedia Data

import requests
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler

from tack_dataset.logging_utils import setup_logging


# Configure logging
log_file = setup_logging(
    log_dir=Path('logs'),
    log_base_name='protacdb_tpddb_protacpedia_curation',
    verbose=1, # Enable INFO level logging
)
logger = logging.getLogger(__name__)

logger.info(f"Log file: {log_file}")

data_dir = Path('data/curation/')
protacdb_df = pd.concat([
    pd.read_csv(data_dir / 'protacdb_protac_dc50_dmax.csv'),
    # pd.read_csv(data_dir / 'protacdb_protac_percent_degradation.csv')
], ignore_index=True)
protacdb_df['Database'] = 'PROTAC-DB'

tpddb_df = pd.read_csv(data_dir / 'tpddb_protac_glues_dc50_dmax.csv')
# Remove glues from TPD-DB dataset
tpddb_df = tpddb_df[~tpddb_df['Modality'].str.contains('MG')]
tpddb_df['Database'] = 'TPD-DB'

df = pd.concat([protacdb_df, tpddb_df], ignore_index=True)

max_len = max(len(col) for col in df.columns)
for col in df.columns:
    logger.info(f"• {col:{max_len}} | {df[col].nunique():5,} unique values | {df[col].isnull().sum():5,} ({df[col].isnull().sum()/len(df)*100:.2f}%) null values")

# %% [markdown]
# ## Remove Duplicates

# %%
key_cols = ['SMILES', 'POI_Name', 'POI_Sequence', 'Ligase_Name', 'Cell_Line', 'Value_Type', 'Value_Unit', 'Assay_Time', 'Assay']

# Find and print the duplicated rows based on key columns
duplicated_rows = df[df.duplicated(subset=key_cols, keep=False)]
logger.info(f'Number of duplicated rows based on {key_cols}: {len(duplicated_rows)}')

# Print all the References for the duplicated rows
for ref in duplicated_rows['Reference'].unique():
    logger.info(f'• {ref}')
logger.info('-' * 80)

# For each of the duplicated rows, group by the key columns and list the different Values
# grouped = duplicated_rows.groupby(key_cols, dropna=False)
grouped = df.groupby(key_cols, dropna=False)
num_conflicts = 0
i = 0
for name, group in grouped:
    values = group['Value'].unique()
    if len(values) > 1:
        i += 1
        logger.info(f'Key N.{i}')
        for k in key_cols:
            logger.info(f'{k}: {len(group[k].unique())}x {group[k].unique()}')
        logger.info(f'Values: {len(values)}x {values}')
        logger.info(f'Assays: {len(group["Assay"].unique())}x {group["Assay"].unique()}')
        logger.info(f'Databases: {len(group["Database"].unique())}x {group["Database"].unique()}')
        logger.info(f'References: {len(group["Reference"].unique())}x {group["Reference"].unique()}')
        logger.info('-' * 80)
        num_conflicts += len(values)

logger.info(f'Number of conflicts found: {num_conflicts}')

# Remove duplicated all rows that have a duplicate, use duplicated_df to find them
df = df[~df.duplicated(subset=key_cols, keep=False)].copy()
logger.info(f'Original dataset size: {len(df):,}')
logger.info(f'Cleaned dataset size: {len(df):,}')

# %% [markdown]
# ## Load PROTAC-Pedia

# %%
# Load the cleaned PROTAC-Pedia dataset
protacpedia_df = pd.read_csv(data_dir / 'protacpedia_protac_dc50_dmax.csv')
protacpedia_df['Database'] = 'PROTAC-Pedia'

logger.info(f"Number of rows in cleaned PROTAC-Pedia dataset: {len(protacpedia_df):,}")

# Convert 'Value' for 'Value_Type' == 'pDC50' back to DC50
def convert_pdc50_to_dc50(row):
    if row['Value_Type'] == 'pDC50':
        pdc50_value = row['Value']
        dc50_value = 10 ** (-pdc50_value) * 1e9  # Convert from M to nM
        return dc50_value
    else:
        return row['Value']

protacpedia_df['Value'] = protacpedia_df.apply(convert_pdc50_to_dc50, axis=1)
# Change the 'Value_Unit' of 'pDC50' rows to 'nM'
protacpedia_df.loc[protacpedia_df['Value_Type'] == 'pDC50', 'Value_Unit'] = 'nM'
# Change the 'Value_Type' of 'pDC50' rows to 'DC50'
protacpedia_df.loc[protacpedia_df['Value_Type'] == 'pDC50', 'Value_Type'] = 'DC50'

# Only consider rows with 'Value_Type' in ['DC50', 'Dmax']
protacpedia_df = protacpedia_df[protacpedia_df['Value_Type'].isin(['DC50', 'Dmax'])]

# Convert all POI_Names to uppercase
protacpedia_df['POI_Name'] = protacpedia_df['POI_Name'].str.upper()
protacpedia_df['POI_UniProt'] = protacpedia_df['POI_UniProt'].str.upper()

# For rows missing POI_Sequence, try to fill it based on POI_Name and POI_UniProt using df as mapping
def get_poi_sequence(row):
    if pd.notna(row['POI_Sequence']):
        return row['POI_Sequence']
    poi_name = row['POI_Name']
    poi_uniprot = row['POI_UniProt']
    match = df[df['POI_Name'] == poi_name.upper()]
    if not match.empty:
        return match.iloc[0]['POI_Sequence']
    match = df[df['POI_UniProt'] == poi_uniprot.upper()]
    if not match.empty:
        return match.iloc[0]['POI_Sequence']
    return pd.NA

logger.info(f'Filling missing {protacpedia_df["POI_Sequence"].isna().sum()} missing POI_Sequence values...')
protacpedia_df['POI_Sequence'] = protacpedia_df.apply(get_poi_sequence, axis=1)
logger.info(f'Now {protacpedia_df["POI_Sequence"].isna().sum()} missing POI_Sequence values remain.')

# Remove rows with ',' in either: 'POI_Name', 'POI_UniProt', 'Cell_Line'
protacpedia_df = protacpedia_df[
    ~protacpedia_df['POI_Name'].str.contains(',', na=False) &
    ~protacpedia_df['POI_UniProt'].str.contains(',', na=False) &
    ~protacpedia_df['Cell_Line'].str.contains(',', na=False)
]

# Remove rows with any of the key columns missing
key_cols = ['SMILES', 'POI_Name', 'POI_Sequence', 'Ligase_Name', 'Cell_Line', 'Value_Type'] #, 'Assay', 'Assay_Time']
protacpedia_df = protacpedia_df.dropna(subset=key_cols, how='any')

# Remove any row with duplicate values in the key columns, do not keep any duplicates
duplicated_rows = protacpedia_df[protacpedia_df.duplicated(subset=key_cols, keep=False)]
protacpedia_df = protacpedia_df[~protacpedia_df.index.isin(duplicated_rows.index)]

logger.info(f"Number of rows in cleaned PROTAC-Pedia dataset: {len(protacpedia_df):,}")

# %%
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

protacpedia_df['Value'] = protacpedia_df.apply(pick_smallest_in_range, axis=1)

# %%
# If 'Assay_Time' is missing, set it to 'Treatment_Time'
protacpedia_df['Assay_Time'] = protacpedia_df.apply(lambda row: row['Treatment_Time'] if pd.isna(row['Assay_Time']) else row['Assay_Time'], axis=1)

# If the 'Value_Concentration' is the same as 'Value', something went wrong
# during PROTAC-Pedia data extraction, so we set the concentration and its unit
# to NaN
protacpedia_df.loc[protacpedia_df['Value_Concentration'] == protacpedia_df['Value'], ['Value_Concentration', 'Value_Concentration_Unit']] = pd.NA

protacpedia_df = protacpedia_df.drop(columns=['Value_Symbol', 'Ligand_IC50',
        'Ligand_IC50_Unit', 'PROTAC_EC50', 'PROTAC_EC50_Unit', 'Ligand_EC50',
        'Ligand_EC50_Unit', 'Treatment_Time', 'Has_Structure', 'PROTAC_IC50',
        'PROTAC_IC50_Unit', 'Covalent', 'Selectivity_Info'])

# %% [markdown]
# Convert PudMed IDs stored in `Reference` column to DOIs using PubMed API:
# 
# **TODO**: Currently not working, skipping this step for now.

# %%
# from tqdm import tqdm
# from Bio import Entrez
# import time

# YOUR_EMAIL = "steribes92@gmail.com"  # Always provide your email

# def get_doi_from_pubmed(pmid: int) -> str:
#     """ Get DOI from PubMed using the Entrez API.
    
#     Args:
#         pmid (int): PubMed ID to fetch the DOI for.
        
#     Returns:
#         str: DOI if available, otherwise a link to the PubMed article.
#     """
#     Entrez.email = YOUR_EMAIL
#     handle = Entrez.efetch(db="pubmed", id=pmid, retmode="xml")
#     records = Entrez.read(handle)
#     handle.close()
    
#     time.sleep(0.1)  # To avoid hitting the API rate limit

#     # Extracting DOI
#     try:
#         article = records['PubmedArticle'][0]
#         for el in article['MedlineCitation']['Article']['ELocationID']:
#             if el.attributes['EIdType'] == 'doi':
#                 return el
#     except IndexError:
#         return f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/'

# # Convert PubMed IDs to string and get unique values (otherwise they'll be treated as floats)
# pubmed_ids = protacpedia_cleaned_df['Reference'].dropna().unique().astype(str).tolist()
# pubmed2doi = {}

# # Get a DOI for each PubMed ID
# for pubmed_id in tqdm(pubmed_ids, desc='Getting DOIs from PubMed', total=len(pubmed_ids)):
#     # logger.info(f'Processing PubMed ID: {pubmed_id}')
#     if pd.isnull(pubmed_id) or pubmed_id in {'nan', 'n/a', 'NaN', ''}:
#         continue
#     # Convert to int if it's a string representation of an integer
#     if pubmed_id not in pubmed2doi:
#         # logger.info(f'Fetching DOI for PubMed ID: {pubmed_id}')
#         pubmed2doi[pubmed_id] = get_doi_from_pubmed(int(float(pubmed_id)))

# # Map the DOIs back to the DataFrame
# protacpedia_cleaned_df['Reference'] = protacpedia_cleaned_df['Reference'].astype(str).map(pubmed2doi)

# %% [markdown]
# Concatenate all DataFrames and remove duplicates based on key columns with the following priority: TPDdb > PROTAC-DB > PROTAC-Pedia.
# This ensures that for duplicate entries, the data from the more reliable source is retained.

# %%
# Concatenate with df
merged_df = pd.concat([df, protacpedia_df], ignore_index=True)
logger.info(f'Original dataset size: {len(merged_df):,}')

# Remove duplicates based on key_cols, but keep based on 'Database' priority: TPD-DB > PROTAC-DB > PROTAC-Pedia
# Sort by Database priority
database_priority = {'TPD-DB': 0, 'PROTAC-DB': 1, 'PROTAC-Pedia': 2}
merged_df['Database_Priority'] = merged_df['Database'].map(database_priority)
merged_df = merged_df.sort_values(by='Database_Priority')

# Remove duplicates based on key_cols, keep the first (highest priority)
key_cols = ['SMILES', 'POI_Name', 'POI_Sequence', 'Ligase_Name', 'Cell_Line', 'Value_Type']
merged_df = merged_df[~merged_df.duplicated(subset=key_cols, keep='first')].copy()

# Drop the Database_Priority column
merged_df = merged_df.drop(columns=['Database_Priority'])
logger.info(f'Cleaned dataset size: {len(merged_df):,}')

# Remove columns for which all values are NaN
merged_df = merged_df.dropna(axis=1, how='all').drop_duplicates()

# %%
df = merged_df.copy()

# %%
df.columns

# %% [markdown]
# ## Convert Concentration to Standard Units

# %%
tmp = df[df['Value_Type'] == 'DC50']
tmp['Value_Unit'].value_counts()

# %%
def convert_to_nM(value, unit):
    value = float(value)
    if unit == 'M':
        value = value * 1e9
    elif unit == 'pM':
        value = value / 1e3
    elif unit == 'nM':
        value = value
    elif unit == 'μM':
        value = value * 1e3
    else:
        return value
    if value <= 0:
        logger.info(f"Warning: Non-positive DC50 value encountered: {value} {unit}")
    return value

df.loc[df['Value_Type'] == 'DC50', 'Value'] = df[df['Value_Type'] == 'DC50'].apply(lambda row: convert_to_nM(row['Value'], row['Value_Unit']), axis=1)
df.loc[df['Value_Type'] == 'DC50', 'Value_Unit'] = 'nM'

df[df['Value_Type'] == 'DC50']['Value_Unit'].value_counts()

# %% [markdown]
# ## Identify Cell Line Species

# %%
# Use the CelloSaurus API to identify species for each cell line has missing species

# Get all cell lines with missing species
cell_line2species = {}
missing_species_cell_lines = df[df['Cell_Line_Species'].isna()]['Cell_Line'].unique().tolist()

for cell_line in missing_species_cell_lines:
    url = f"https://web.expasy.org/cellosaurus/api/cell-line/{cell_line}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        species = data.get('species', 'Unknown')
        cell_line2species[cell_line] = species
    else:
        cell_line2species[cell_line] = 'Unknown'
    time.sleep(0.1)  # To avoid hitting the API rate limit

cell_line2species

# # Map the species back to the DataFrame
# df.loc[df['Cell_Line'].isin(cell_line2species.keys()), 'Cell_Line_Species'] = df['Cell_Line'].map(cell_line2species)

# %% [markdown]
# ## Save to CSV

# %%
df.to_csv(data_dir / 'protacdb_tpddb_protacpedia_protac_dc50_dmax_activities.csv', index=False)

# %% [markdown]
# ## Plotting and Statistics

# %%
logger.info(df['Value_Type'].value_counts())
# logger.info()
# Count DC50 and Dmax entries per Database
logger.info(df.groupby('Database')['Value_Type'].value_counts())

# %%
# Show percentage of Value_Operator per Database
df.groupby('Database')['Value_Operator'].value_counts()

# %%
logger.info('Counting database entries that have both Dmax and DC50 values:')

cols = ['SMILES', 'POI_Name', 'POI_Sequence', 'Ligase_Name', 'Cell_Line', 'Cell_Line_ID', 'Assay', 'Assay_Time']
total_multitask_entries = 0
for db in df['Database'].unique():
    db_df = df[df['Database'] == db]
    dmax_df = db_df[db_df['Value_Type'] == 'Dmax']
    dc50_df = db_df[db_df['Value_Type'] == 'DC50']
    merged = pd.merge(
        dmax_df,
        dc50_df,
        on=cols,
        suffixes=('_Dmax', '_DC50')
    ).dropna(subset=['Value_Dmax', 'Value_DC50'], how='any')
    logger.info(f'• Database: {db}: {len(merged):,}')
    total_multitask_entries += len(merged)
logger.info(f'Total multitask entries across all databases: {total_multitask_entries:,}')

df.groupby(['Database', 'Value_Type'], dropna=True)['Value'].count()

# %%
# Print number of NaN values and their percentage for each column in key_cols
key_cols = ['SMILES', 'POI_Name', 'POI_Sequence', 'Ligase_Name', 'Cell_Line', 'Cell_Line_ID', 'Value_Type', 'Assay', 'Assay_Time']
for col in key_cols:
    num_missing = df[col].isna().sum()
    percent_missing = (num_missing / len(df)) * 100
    logger.info(f'Num. missing in "{col}": {num_missing} ({percent_missing:.2f}%)')

# logger.info()
for col in key_cols:
    if df[col].isna().sum() > 0:
        # Group by Database and count missing values
        missing_counts = df[df[col].isna()].groupby('Database').size()
        logger.info(f'Missing values in column "{col}" by Database:')
        for db, count in missing_counts.items():
            logger.info(f'• {db}: {count:,} ({(count / len(df[df["Database"] == db]) * 100):.2f}%) missing')

# %%
key_cols = ['SMILES', 'POI_Name', 'POI_Sequence', 'Ligase_Name', 'Cell_Line', 'Cell_Line_ID', 'Value_Type', 'Assay', 'Assay_Time']
for col in key_cols:
    logger.info(df.groupby('Database')[col].nunique())
    # logger.info()

# %%
# Plot the distribution of Dmax values according to Database
tmp = df[df['Value_Type'] =='Dmax'].copy()
tmp['Database'] = tmp['Database'].astype('category')

# Fix the warning: 'UserWarning: No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.'
plt.figure(figsize=(8, 6))
ax = sns.histplot(data=tmp, x='Value', hue='Database', bins=30, kde=True)
plt.title('Distribution of Dmax (%) by Database')
plt.xlabel('Dmax (%)')
plt.ylabel('Density')
plt.grid(axis='both', alpha=0.5)
handles, labels = ax.get_legend_handles_labels()
if handles and labels:
    plt.legend(handles=handles, labels=labels, title='Database')


def convert_to_pdc50(value, unit):
    value = float(value)
    if unit == 'M':
        value = value * 1e9
    elif unit == 'pM':
        value = value / 1e3
    elif unit == 'nM':
        value = value
    elif unit == 'μM':
        value = value * 1e3
    else:
        return value
    if value <= 0:
        logger.info(f"Warning: Non-positive DC50 value encountered: {value} {unit}")
    # Turn into pDC50
    return -np.log10(value * 1e-9 + 1e-30)

# Plot the distribution of Dmax values according to Database
# tmp = df[(df['Value_Type'] == 'DC50') & (df['Database'] == 'PROTAC-DB')].copy()
# tmp = df[(df['Value_Type'] == 'DC50') & (df['Database'] == 'TPD-DB')].copy()
tmp = df[df['Value_Type'] == 'DC50'].copy()
tmp['Database'] = tmp['Database'].astype('category')

# Convert all DC50 values to nM
tmp['Value'] = tmp.apply(lambda row: convert_to_pdc50(row['Value'], row['Value_Unit']), axis=1)

# Fix the warning: 'UserWarning: No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.'
plt.figure(figsize=(8, 6))
ax = sns.histplot(data=tmp, x='Value', hue='Database', bins=30, kde=True)
plt.title('Distribution of pDC50 by Database')
plt.xlabel('pDC50')
plt.ylabel('Density')
# Make the x-axis logarithmic
# plt.xscale('log')
plt.grid(axis='both', alpha=0.5)
handles, labels = ax.get_legend_handles_labels()
if handles and labels:
    plt.legend(handles=handles, labels=labels, title='Database')


# Quantile transform for Dmax
dmax_df = df[df['Value_Type'] == 'Dmax'].copy()
qt_dmax = QuantileTransformer(output_distribution='normal', random_state=42)
dmax_df['Value_QT'] = qt_dmax.fit_transform(dmax_df[['Value']].to_numpy())
scaler_dmax = MinMaxScaler()
dmax_df['Value_QT_Scaled'] = scaler_dmax.fit_transform(dmax_df[['Value_QT']].to_numpy())

plt.figure(figsize=(8, 6))
ax = sns.histplot(data=dmax_df, x='Value_QT_Scaled', hue='Database', bins=30, kde=True)
plt.title('Standard Scaled Quantile Transformed Dmax (normal) by Database')
plt.xlabel('Dmax (QT + MinMaxScaler)')
plt.ylabel('Density')
handles, labels = ax.get_legend_handles_labels()
if handles and labels:
    plt.legend(handles=handles, labels=labels, title='Database')
plt.grid(axis='both', alpha=0.5)


# Quantile transform for DC50
dc50_df = df[df['Value_Type'] == 'DC50'].copy()
qt_dc50 = QuantileTransformer(output_distribution='normal', random_state=42)
dc50_df['Value_QT'] = qt_dc50.fit_transform(dc50_df[['Value']].to_numpy())
scaler_dc50 = MinMaxScaler()
dc50_df['Value_QT_Scaled'] = scaler_dc50.fit_transform(dc50_df[['Value_QT']].to_numpy())

plt.figure(figsize=(8, 6))
ax = sns.histplot(data=dc50_df, x='Value_QT_Scaled', hue='Database', bins=30, kde=True)
plt.title('Standard Scaled Quantile Transformed DC50 (normal) by Database')
plt.xlabel('DC50 (QT + MinMaxScaler)')
plt.ylabel('Density')
handles, labels = ax.get_legend_handles_labels()
if handles and labels:
    plt.legend(handles=handles, labels=labels, title='Database')
plt.grid(axis='both', alpha=0.5)


dmax_df['Value_Recovered'] = qt_dmax.inverse_transform(scaler_dmax.inverse_transform(dmax_df[['Value_QT_Scaled']]))
plt.figure(figsize=(8, 6))
ax = sns.histplot(data=dmax_df, x='Value_Recovered', hue='Database', bins=30, kde=True)
plt.title('Recovered Dmax values after QT and MinMaxScaler')
plt.xlabel('Original Dmax')
plt.ylabel('Recovered Dmax')
handles, labels = ax.get_legend_handles_labels()
if handles and labels:
    plt.legend(handles=handles, labels=labels, title='Database')
plt.grid(axis='both', alpha=0.5)


dc50_df['Value_Recovered'] = qt_dc50.inverse_transform(scaler_dc50.inverse_transform(dc50_df[['Value_QT_Scaled']]))
dc50_df['Value_Recovered'] = dc50_df.apply(lambda row: convert_to_pdc50(row['Value_Recovered'], row['Value_Unit']), axis=1)

plt.figure(figsize=(8, 6))
ax = sns.histplot(data=dc50_df, x='Value_Recovered', hue='Database', bins=30, kde=True)
plt.title('Recovered DC50 values after QT and MinMaxScaler')
plt.xlabel('Original DC50')
plt.ylabel('Recovered DC50')
handles, labels = ax.get_legend_handles_labels()
if handles and labels:
    plt.legend(handles=handles, labels=labels, title='Database')
plt.grid(axis='both', alpha=0.5)


# %%
# Count the unique POI_Names per Database
num_poi = df.groupby('Database')['POI_Name'].nunique()
logger.info(f"Number of unique POI_Names per Database: {num_poi.to_dict()}")
# Count the number of overlapping POI_Names between the two databases
protacdb_pois = set(df[df['Database'] == 'PROTAC-DB']['POI_Name'].unique())
tpddb_pois = set(df[df['Database'] == 'TPD-DB']['POI_Name'].unique())
protacpedia_pois = set(df[df['Database'] == 'PROTAC-Pedia']['POI_Name'].unique())
logger.info(f"Number of unique POI_Names in PROTAC-DB: {len(protacdb_pois)}")
logger.info(f"Number of unique POI_Names in TPD-DB: {len(tpddb_pois)}")
logger.info(f"Number of unique POI_Names in PROTAC-Pedia: {len(protacpedia_pois)}")
logger.info('-' * 80)

overlapping_pois = protacdb_pois.intersection(tpddb_pois)
logger.info(f"Number of overlapping POI_Names between PROTAC-DB and TPD-DB: {len(overlapping_pois)}")
logger.info(f"Overlapping POI_Names: {sorted(overlapping_pois)}")
logger.info(f"Non-overlapping POI_Names in PROTAC-DB: {protacdb_pois - overlapping_pois}")
logger.info('-' * 80)

overlapping_pois_pp = protacdb_pois.intersection(protacpedia_pois)
logger.info(f"Number of overlapping POI_Names between PROTAC-DB and PROTAC-Pedia: {len(overlapping_pois_pp)}")
logger.info(f"Overlapping POI_Names: {sorted(overlapping_pois_pp)}")
logger.info(f"Non-overlapping POI_Names in PROTAC-DB: {protacdb_pois - overlapping_pois_pp}")
logger.info('-' * 80)

overlapping_pois_tp = tpddb_pois.intersection(protacpedia_pois)
logger.info(f"Number of overlapping POI_Names between TPD-DB and PROTAC-Pedia: {len(overlapping_pois_tp)}")
logger.info(f"Overlapping POI_Names: {sorted(overlapping_pois_tp)}")
logger.info(f"Non-overlapping POI_Names in TPD-DB: {tpddb_pois - overlapping_pois_tp}")
logger.info('-' * 80)

overlapping_pois_all = protacdb_pois.intersection(tpddb_pois).intersection(protacpedia_pois)
logger.info(f"Number of overlapping POI_Names between all three databases: {len(overlapping_pois_all)}")
logger.info(f"Overlapping POI_Names: {sorted(overlapping_pois_all)}")
logger.info('-' * 80)
logger.info(f"Unique POI_Names in PROTAC-DB: {sorted(protacdb_pois - tpddb_pois - protacpedia_pois)}")
logger.info(f"Unique POI_Names in TPD-DB: {sorted(tpddb_pois - protacdb_pois - protacpedia_pois)}")
logger.info(f"Unique POI_Names in PROTAC-Pedia: {sorted(protacpedia_pois - protacdb_pois - tpddb_pois)}")

# %%
# Count the unique Cell_Lines per Database
num_cell = df.groupby('Database')['Cell_Line'].nunique()
logger.info(f"Number of unique Cell_Lines per Database: {num_cell.to_dict()}")
# Count the number of overlapping Cell_Lines between the two databases
protacdb_cells = set(df[df['Database'] == 'PROTAC-DB']['Cell_Line'].dropna().unique())
tpddb_cells = set(df[df['Database'] == 'TPD-DB']['Cell_Line'].dropna().unique())
protacpedia_cells = set(df[df['Database'] == 'PROTAC-Pedia']['Cell_Line'].dropna().unique())
logger.info(f"Number of unique Cell_Lines in PROTAC-DB: {len(protacdb_cells)}")
logger.info(f"Number of unique Cell_Lines in TPD-DB: {len(tpddb_cells)}")
logger.info(f"Number of unique Cell_Lines in PROTAC-Pedia: {len(protacpedia_cells)}")
logger.info('-' * 80)

overlapping_cells = protacdb_cells.intersection(tpddb_cells)
logger.info(f"Number of overlapping Cell_Lines between PROTAC-DB and TPD-DB: {len(overlapping_cells)}")
logger.info(f"Overlapping Cell_Lines: {sorted(overlapping_cells)}")
logger.info(f"Non-overlapping Cell_Lines in PROTAC-DB: {protacdb_cells - overlapping_cells}")
logger.info('-' * 80)

overlapping_cells_pp = protacdb_cells.intersection(protacpedia_cells)
logger.info(f"Number of overlapping Cell_Lines between PROTAC-DB and PROTAC-Pedia: {len(overlapping_cells_pp)}")
logger.info(f"Overlapping Cell_Lines: {sorted(overlapping_cells_pp)}")
logger.info(f"Non-overlapping Cell_Lines in PROTAC-DB: {protacdb_cells - overlapping_cells_pp}")
logger.info('-' * 80)

overlapping_cells_tp = tpddb_cells.intersection(protacpedia_cells)
logger.info(f"Number of overlapping Cell_Lines between TPD-DB and PROTAC-Pedia: {len(overlapping_cells_tp)}")
logger.info(f"Overlapping Cell_Lines: {sorted(overlapping_cells_tp)}")
logger.info(f"Non-overlapping Cell_Lines in TPD-DB: {tpddb_cells - overlapping_cells_tp}")
logger.info('-' * 80)

overlapping_cells_all = protacdb_cells.intersection(tpddb_cells).intersection(protacpedia_cells)
logger.info(f"Number of overlapping Cell_Lines between all three databases: {len(overlapping_cells_all)}")
logger.info(f"Overlapping Cell_Lines: {sorted(overlapping_cells_all)}")
logger.info('-' * 80)
logger.info(f"Unique Cell_Lines in PROTAC-DB: {sorted(protacdb_cells - tpddb_cells - protacpedia_cells)}")
logger.info(f"Unique Cell_Lines in TPD-DB: {sorted(tpddb_cells - protacdb_cells - protacpedia_cells)}")
logger.info(f"Unique Cell_Lines in PROTAC-Pedia: {sorted(protacpedia_cells - protacdb_cells - tpddb_cells)}")

# %%
# Count the unique Cell_Line_IDs per Database
num_cell = df.groupby('Database')['Cell_Line_ID'].nunique()
logger.info(f"Number of unique Cell_Line_IDs per Database: {num_cell.to_dict()}")
# Count the number of overlapping Cell_Line_IDs between the two databases
protacdb_cells = set(df[df['Database'] == 'PROTAC-DB']['Cell_Line_ID'].dropna().unique())
tpddb_cells = set(df[df['Database'] == 'TPD-DB']['Cell_Line_ID'].dropna().unique())
protacpedia_cells = set(df[df['Database'] == 'PROTAC-Pedia']['Cell_Line_ID'].dropna().unique())
logger.info(f"Number of unique Cell_Line_IDs in PROTAC-DB: {len(protacdb_cells)}")
logger.info(f"Number of unique Cell_Line_IDs in TPD-DB: {len(tpddb_cells)}")
logger.info(f"Number of unique Cell_Line_IDs in PROTAC-Pedia: {len(protacpedia_cells)}")
logger.info('-' * 80)

overlapping_cells = protacdb_cells.intersection(tpddb_cells)
logger.info(f"Number of overlapping Cell_Line_IDs between PROTAC-DB and TPD-DB: {len(overlapping_cells)}")
logger.info(f"Overlapping Cell_Line_IDs: {sorted(overlapping_cells)}")
logger.info(f"Non-overlapping Cell_Line_IDs in PROTAC-DB: {protacdb_cells - overlapping_cells}")
logger.info('-' * 80)

overlapping_cells_pp = protacdb_cells.intersection(protacpedia_cells)
logger.info(f"Number of overlapping Cell_Line_IDs between PROTAC-DB and PROTAC-Pedia: {len(overlapping_cells_pp)}")
logger.info(f"Overlapping Cell_Line_IDs: {sorted(overlapping_cells_pp)}")
logger.info(f"Non-overlapping Cell_Line_IDs in PROTAC-DB: {protacdb_cells - overlapping_cells_pp}")
logger.info('-' * 80)

overlapping_cells_tp = tpddb_cells.intersection(protacpedia_cells)
logger.info(f"Number of overlapping Cell_Line_IDs between TPD-DB and PROTAC-Pedia: {len(overlapping_cells_tp)}")
logger.info(f"Overlapping Cell_Line_IDs: {sorted(overlapping_cells_tp)}")
logger.info(f"Non-overlapping Cell_Line_IDs in TPD-DB: {tpddb_cells - overlapping_cells_tp}")
logger.info('-' * 80)

overlapping_cells_all = protacdb_cells.intersection(tpddb_cells).intersection(protacpedia_cells)
logger.info(f"Number of overlapping Cell_Line_IDs between all three databases: {len(overlapping_cells_all)}")
logger.info(f"Overlapping Cell_Line_IDs: {sorted(overlapping_cells_all)}")
logger.info('-' * 80)
logger.info(f"Unique Cell_Line_IDs in PROTAC-DB: {sorted(protacdb_cells - tpddb_cells - protacpedia_cells)}")
logger.info(f"Unique Cell_Line_IDs in TPD-DB: {sorted(tpddb_cells - protacdb_cells - protacpedia_cells)}")
logger.info(f"Unique Cell_Line_IDs in PROTAC-Pedia: {sorted(protacpedia_cells - protacdb_cells - tpddb_cells)}")