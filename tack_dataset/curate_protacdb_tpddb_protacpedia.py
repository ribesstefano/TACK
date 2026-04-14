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

# Check if protac_splitter is installed
try:
    from protac_splitter import split_protac
    PROTAC_SPLITTER_AVAILABLE = True
except ImportError:
    PROTAC_SPLITTER_AVAILABLE = False


from tackai.data.embeddings.cell_embeddings import CellEmbedding
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

# ## Normalize DC50 units to nM before aggregation
merged_df.loc[merged_df['Value_Type'] == 'DC50', 'Value'] = (
    merged_df[merged_df['Value_Type'] == 'DC50']
    .apply(lambda row: convert_to_nM(row['Value'], row['Value_Unit']), axis=1)
)
merged_df.loc[merged_df['Value_Type'] == 'DC50', 'Value_Unit'] = 'nM'

if PROTAC_SPLITTER_AVAILABLE:
    # TODO: Add ternary split of PROTACs into warhead, linker, and E3 ligand
    pass

# ## Aggregate Duplicates
# Group by identity columns and aggregate: median for numerics, most common for
# categoricals, join unique values for provenance columns (Reference, Database, etc.)

key_cols = ['SMILES', 'POI_Name', 'POI_Sequence', 'Ligase_Name',
            'Cell_Line', 'Value_Type', 'Value_Unit']


def first_non_null(s):
    non_null = s.dropna()
    return non_null.iloc[0] if len(non_null) > 0 else np.nan


def most_common(s):
    non_null = s.dropna()
    return non_null.mode().iloc[0] if len(non_null) > 0 else np.nan


def join_unique(s):
    vals = s.dropna().unique()
    return '; '.join(str(v) for v in vals) if len(vals) > 0 else np.nan


numeric_cols = ['Value', 'Value_Error', 'Value_Range_Min', 'Value_Range_Max', 'Value_Concentration', 'Assay_Time']
mode_cols = ['Value_Operator', 'Value_Category', 'Value_Concentration_Unit', 'Modality']
identity_cols = ['POI_UniProt', 'Ligase_UniProt', 'Ligase_Sequence', 'Cell_Line_ID', 'Cell_Line_Species']
provenance_cols = ['Reference', 'Description', 'Database', 'Assay']
if 'TPD_ID' in merged_df.columns:
    provenance_cols.append('TPD_ID')

agg_dict = {}
for col in numeric_cols:
    if col in merged_df.columns:
        agg_dict[col] = 'median'
for col in mode_cols:
    if col in merged_df.columns:
        agg_dict[col] = most_common
for col in identity_cols:
    if col in merged_df.columns:
        agg_dict[col] = first_non_null
for col in provenance_cols:
    if col in merged_df.columns:
        agg_dict[col] = join_unique

logger.info(f'Dataset size before aggregation: {len(merged_df):,}')
merged_df = (
    merged_df
    .groupby(key_cols, dropna=False)
    .agg(agg_dict)
    .reset_index()
)
logger.info(f'Dataset size after aggregation: {len(merged_df):,}')

# Remove columns for which all values are NaN
merged_df = merged_df.dropna(axis=1, how='all')

df = merged_df.copy()

## Add cell descriptions to the DataFrame

cell_embedding = CellEmbedding()
descriptions = {}
for cell_line in df['Cell_Line_ID'].dropna().unique():
    descriptions[cell_line] = cell_embedding.cell2description.get(
        cell_line,
        cell_embedding.not_found_description,
    )
df['Cell_Line_Description'] = df['Cell_Line_ID'].map(descriptions)

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