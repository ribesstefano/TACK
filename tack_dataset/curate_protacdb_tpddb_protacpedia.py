""" Curate PROTAC-DB, TPDdb, and PROTAC-Pedia Data
"""
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
from tack_dataset.curation_utils import convert_to_nM


def get_report(df_in: pd.DataFrame) -> pd.DataFrame:
    """ Generate a report of unique values and missing values for each column in
        the DataFrame. """
    df = df_in.copy()
    report_df = []
    for col in sorted(df.columns):
        num_unique = df[col].nunique()
        num_missing = df[col].isnull().sum()
        percent_missing = (num_missing / len(df)) * 100
        report_df.append({
            'Column': col,
            'Unique Values': num_unique,
            'Missing Values': num_missing,
            'Percent Missing': percent_missing
        })
    return pd.DataFrame(report_df)


# Configure logging
log_file = setup_logging(
    log_dir=Path('logs'),
    log_base_name='protacdb_tpddb_protacpedia_curation',
    verbose=1, # Enable INFO level logging
    log_file=Path('logs/protacdb_tpddb_protacpedia_curation.log')
)
logger = logging.getLogger(__name__)


data_dir = Path('data/curation/')
# ==============================================================================
# Concatenate the datasets, keeping track of the source database for each entry.
# ==============================================================================
# ----------------
# PROTAC-DB
# ----------------
protacdb_df = pd.concat([
    pd.read_csv(data_dir / 'protacdb_protac_dc50_dmax.csv'),
    # pd.read_csv(data_dir / 'protacdb_protac_percent_degradation.csv')
], ignore_index=True)
protacdb_df['Database'] = 'PROTAC-DB'
logger.info('PROTAC-DB:\n' + str(get_report(protacdb_df).round(2).to_markdown(index=False)))

# ----------------
# TPDdb
# ----------------
tpddb_df = pd.read_csv(data_dir / 'tpddb_protac_glues_dc50_dmax.csv')
# Remove glues from TPD-DB dataset
tpddb_df = tpddb_df[~tpddb_df['Modality'].str.contains('MG')]
tpddb_df['Database'] = 'TPDdb'
logger.info('TPDdb:\n' + str(get_report(tpddb_df).round(2).to_markdown(index=False)))

# ----------------
# PROTACpedia
# ----------------
protacpedia_df = pd.read_csv(data_dir / 'protacpedia_protac_dc50_dmax.csv')
protacpedia_df['Database'] = 'PROTACpedia'
logger.info('PROTACpedia:\n' + str(get_report(protacpedia_df).round(2).to_markdown(index=False)))

# ----------------
# Merged
# ----------------
merged_df = pd.concat([protacdb_df, tpddb_df, protacpedia_df], ignore_index=True)
logger.info('Merged:\n' + str(get_report(merged_df).round(2).to_markdown(index=False)))

# ## Remove Duplicates

key_cols = ['SMILES', 'POI_Name', 'POI_Sequence', 'Ligase_Name', 'Cell_Line', 'Value_Type', 'Value_Unit', 'Assay_Time', 'Assay']

# Find and print the duplicated rows based on key columns
duplicated_rows = merged_df[merged_df.duplicated(subset=key_cols, keep=False)]
logger.info(f'Number of duplicated rows based on {key_cols}: {len(duplicated_rows)}')

# Print all the References for the duplicated rows
for ref in duplicated_rows['Reference'].unique():
    logger.info(f'• {ref}')
logger.info('-' * 80)

# For each of the duplicated rows, group by the key columns and list the different Values
# grouped = duplicated_rows.groupby(key_cols, dropna=False)
grouped = merged_df.groupby(key_cols, dropna=False)
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
merged_df = merged_df[~merged_df.duplicated(subset=key_cols, keep=False)].copy()
logger.info(f'Original dataset size: {len(merged_df):,}')
logger.info(f'Cleaned dataset size: {len(merged_df):,}')

# Concatenate all DataFrames and remove duplicates based on key columns with the following priority: TPDdb > PROTAC-DB > PROTAC-Pedia.
# This ensures that for duplicate entries, the data from the more reliable source is retained.

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

df = merged_df.copy()

# ## Convert Concentration to Standard Units

tmp = df[df['Value_Type'] == 'DC50']
tmp['Value_Unit'].value_counts()

df.loc[df['Value_Type'] == 'DC50', 'Value'] = df[df['Value_Type'] == 'DC50'].apply(lambda row: convert_to_nM(row['Value'], row['Value_Unit']), axis=1)
df.loc[df['Value_Type'] == 'DC50', 'Value_Unit'] = 'nM'

df[df['Value_Type'] == 'DC50']['Value_Unit'].value_counts()

# ## Identify Cell Line Species

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

# # Map the species back to the DataFrame
# df.loc[df['Cell_Line'].isin(cell_line2species.keys()), 'Cell_Line_Species'] = df['Cell_Line'].map(cell_line2species)

# ## Save to CSV

logger.info('\n' + str(get_report(df).round(2).to_markdown(index=False)))

df.to_csv(data_dir / 'protacdb_tpddb_protacpedia_protac_dc50_dmax_activities.csv', index=False)

# ## Plotting and Statistics

logger.info(df['Value_Type'].value_counts())
# logger.info()
# Count DC50 and Dmax entries per Database
logger.info(df.groupby('Database')['Value_Type'].value_counts())

# Show percentage of Value_Operator per Database
df.groupby('Database')['Value_Operator'].value_counts()

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

key_cols = ['SMILES', 'POI_Name', 'POI_Sequence', 'Ligase_Name', 'Cell_Line', 'Cell_Line_ID', 'Value_Type', 'Assay', 'Assay_Time']
for col in key_cols:
    logger.info(df.groupby('Database')[col].nunique())
    # logger.info()

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

print('-' * 80)
print(f"Log file saved at: {log_file}")
print('-' * 80)