# %% [markdown]
# # Data Curation
# 
# After running this notebook top to bottom, the curated data will be saved in the `data/curation` directory in the CSV file: `data/curation/PROTAC-Degradation-DB.csv`.

# %% [markdown]
# ## Setup

# %% [markdown]
# ### Imports

# %%
import logging
import warnings
import re
import os
import pickle
import pickle
import requests
import json
import random
import time
import sys
import yaml
import shutil
import hashlib
from pathlib import Path
from typing import Literal, Union, List, Optional
from itertools import zip_longest
from functools import lru_cache, reduce

from thefuzz import process
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import Entrez

from tack_dataset.logging_utils import setup_logging
from tackai.data.embeddings.cell_embeddings import (
    CellEmbedding,
)

# Setup logging
log_file = setup_logging(
    log_dir='logs',
    log_base_name='protacdb_curation',
    verbose=1, # Enable INFO level logging
)
logger = logging.getLogger(__name__)

logger.info(f"Log file: {log_file}")

RDLogger.DisableLog('rdApp.*')

# %% [markdown]
# Filter out some warnings...

# %%
def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)


# Filter out annoying Pytorch Lightning printouts
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', '.*Covariance of the parameters could not be estimated.*')
warnings.filterwarnings('ignore', '.*You seem to be using the pipelines sequentially on GPU.*')

# %% [markdown]
# ### Download Raw Data

# %% [markdown]
# Setup working directories:

# %%
data_dir = os.path.join(os.getcwd(), 'data')
data_raw_dir = os.path.join(data_dir, 'original')
data_curation_dir = os.path.join(data_dir, 'curation')

for d in [data_dir, data_raw_dir, data_curation_dir]:
    if not os.path.exists(d):
        os.makedirs(d)

print(f"Data directories set up at: {data_dir}")

# %% [markdown]
# Download or load the raw PROTAC-DB dataset:

# %%
protacdb_file = os.path.join(data_raw_dir, 'PROTAC-DB.csv')
protacdb_url = 'http://cadd.zju.edu.cn/protacdb/statics/binaryDownload/csv/protac/protac.csv'

if os.path.exists(protacdb_file):
    protacdb_df = pd.read_csv(protacdb_file).reset_index(drop=True)
else:
    print(f'Downloading {protacdb_url}')
    response = requests.get(protacdb_url)
    with open(protacdb_file, 'wb') as f:
        f.write(response.content)
    protacdb_df = pd.read_csv(protacdb_file).reset_index(drop=True)
print('PROTAC-DB loaded.')

old2new = {
    'E3 ligase': 'E3 Ligase',
}
protacdb_df = protacdb_df.rename(columns=old2new)

# %% [markdown]
# ## Utilities

# %%
def save_dict(
    d: dict,
    filepath: str,
    indent: int = 4,
):
    """
    Save a dictionary to a file in JSON format.
    
    Args:
        d (dict): Dictionary to save.
        filepath (str): Path to the file where the dictionary will be saved.
        indent (int): Indentation level for JSON formatting. Default is 4.
        mode (str): File mode, either 'w' for text or 'wb' for binary. Default is 'w'.
    """
    if filepath.endswith('.json'):
        with open(filepath, 'w') as f:
            json.dump(d, f, indent=indent)
    elif filepath.endswith('.pkl'):
        with open(filepath, 'wb') as f:
            pickle.dump(d, f)
    else:
        raise ValueError(f'Unsupported file extension: {filepath}. Use .json or .pkl.')

def load_dict(filepath: str) -> Union[dict, List[dict]]:
    """
    Load a dictionary from a file in JSON or pickle format.
    
    Args:
        filepath (str): Path to the file from which the dictionary will be loaded.
        
    Returns:
        dict: The loaded dictionary. If the file does not exist, returns an empty dictionary.
    """
    if not os.path.exists(filepath):
        return {}
    if filepath.endswith('.json'):
        with open(filepath, 'r') as f:
            return json.load(f)
    elif filepath.endswith('.pkl'):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f'Unsupported file extension: {filepath}. Use .json or .pkl.')

# %%
def parse_single_value(value_str):
    """Parse a single numeric value with optional operator, error bar, and unit"""
    
    # Clean the value string first
    value_str = value_str.strip()
    
    # Pattern 1: Error bar with optional tilde prefix (0.785±0.03μM, 67±1.4%, ~58.7±0.03%)
    match = re.match(r'^~?\s*(-?\d+\.?\d*|\d*\.\d+)\s*[±]\s*(\d+\.?\d*|\d*\.\d+)\s*([a-zA-Zμµ%]+)?$', value_str)
    if match:
        return {
            'mean': float(match.group(1)),
            'error': float(match.group(2)),
            'unit': match.group(3),
            'operator': None
        }
    
    # Pattern 2: Operator with optional spaces (>10μM, <100nM, ≥150nM, <=100nM, > 3.16E-07M)
    match = re.match(r'^([>≥<≤]+|<=|>=)\s*(\d+\.?\d*(?:[eE][+-]?\d+)?|\d*\.\d+)\s*([a-zA-Zμµ%/]+)?$', value_str)
    if match:
        return {
            'mean': float(match.group(2)),
            'error': None,
            'unit': match.group(3),
            'operator': match.group(1)
        }
    
    # Pattern 3: Tilde prefix (~58.7%, ~3nM, ~30.1 %)
    match = re.match(r'^~\s*(-?\d+\.?\d*|\d*\.\d+)\s*([a-zA-Zμµ%]+)?$', value_str)
    if match:
        return {
            'mean': float(match.group(1)),
            'error': None,
            'unit': match.group(2),
            'operator': '~'
        }

    # Pattern 4: Numeric with exponential notation (1.2e3 nM, 3.5E-2 μM, 3.16E-07M)
    match = re.match(r'^(-?\d+\.?\d*|\d*\.\d+)[eE][+-]?\d+\s*([a-zA-Zμµ%/]+)?$', value_str)
    if match:
        # Extract the full number including exponent by finding where the unit starts
        numeric_part = value_str
        if match.group(2):  # If there's a unit
            numeric_part = value_str[:value_str.rfind(match.group(2))].strip()
        return {
            'mean': float(numeric_part),
            'error': None,
            'unit': match.group(2),
            'operator': None
        }
    
    # Pattern 5: Negative numbers (-36%, -15%, -5.3%)
    match = re.match(r'^(-\d+\.?\d*|\d*\.\d+)\s*([a-zA-Zμµ%]+)?$', value_str)
    if match:
        return {
            'mean': float(match.group(1)),
            'error': None,
            'unit': match.group(2),
            'operator': None
        }
    
    # Pattern 6: Asterisk suffix (88.9*, 0.08nM*, 7nM*)
    match = re.match(r'^(-?\d+\.?\d*|\d*\.\d+)\s*([a-zA-Zμµ%]+)?\s*\*$', value_str)
    if match:
        return {
            'mean': float(match.group(1)),
            'error': None,
            'unit': match.group(2),
            'operator': '*'
        }
    
    # Pattern 7: Value with parenthetical annotation (0.022 (1%), 0.052 (42%))
    # Extract first number before parentheses
    match = re.match(r'^(-?\d+\.?\d*|\d*\.\d+)\s*([a-zA-Zμµ%]+)?\s*\([^)]+\)$', value_str)
    if match:
        return {
            'mean': float(match.group(1)),
            'error': None,
            'unit': match.group(2),
            'operator': None
        }
    
    # Pattern 8: Standard numeric (2.63nM, 0.701μM, 67%, 0.001µM)
    match = re.match(r'^(-?\d+\.?\d*|\d*\.\d+)\s*([a-zA-Zμµ%/]+)?$', value_str)
    if match:
        return {
            'mean': float(match.group(1)),
            'error': None,
            'unit': match.group(2),
            'operator': None
        }
    
    return None

def clean_string(s: str) -> str:
    """ Clean a string by removing <, >, =, NaN, and ranges like 100-200.
    Args:
        s(str): string to clean
    Returns:
        str: cleaned string
    """
    if pd.isnull(s) or s in {'nan', 'n/a', 'NaN', ''}:
        return None
    if 'N.D.' in s:
        return '0'
    if ':' in s:
        return None
    s = s.strip('(WB)').strip()
    # # Combine regex operations for efficiency
    # s = re.sub(r'[<=>]|NaN|[\d]+[-~]', '', s)  # Remove <, >, =, NaN, and ranges like 100-200
    # Remove <, >, =, NaN
    s = re.sub(r'[<=>]|NaN', '', s)
    # Replace ranges like 100-200 or 1~3 with the left-most value in the range
    s = re.sub(r'\b(\d+)[-~]\d+\b', r'\1', s)
    # Replace (n/a) with nan
    s = s.replace('(n/a)', 'nan')
    s = re.sub(r'[~<=>% ]', '', s)  # Remove ~, <, >, =, % and spaces
    # Remove'±'
    s = s.split('±')[0]
    return s


def split_clean_str(s: str, return_floats: bool = False) -> Union[List[str], List[float]]:
    """ Split a string by '/' and clean each part.
    Args:
        s(str): string to split
        return_floats(bool): whether to return floats or strings
    Returns:
        list: list of cleaned strings or floats
    """
    if pd.isnull(s) or s in {'nan', 'n/a', 'NaN', ''}:
        return None
    s = s.replace('(n/a)', 'nan')
    values = [parse_single_value(part.strip()) for part in s.split('/')]
    if return_floats:
        return [float(value['mean']) if value is not None else None for value in values]
    else:
        return [value for value in values]


print(split_clean_str('-100-200/-5/(n/a)/<=90.317/>1000/NaN', return_floats=True))
print(split_clean_str('N.D.', return_floats=True))
print(split_clean_str('96/73 (WB)', return_floats=True))
print(split_clean_str('1.0~3/3.14', return_floats=True))
print(split_clean_str('2000-1800:00:00', return_floats=True))
print(split_clean_str('0.081/0.14/0.53', return_floats=True))
print()
print(parse_single_value('0.785±0.03μM'))

# %% [markdown]
# ## Add Row Identifiers

# %%
protacdb_df['Row ID'] = protacdb_df.index.astype(int).to_list()
protacdb_df['Row ID']

# %% [markdown]
# ## Canonize SMILES

# %%
def canonize_smiles(smi):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi))

protacdb_df['Smiles'] = protacdb_df['Smiles'].map(canonize_smiles)

# %% [markdown]
# ## Define Assay-Related Columns

# %%
def get_assay_texts(df: pd.DataFrame, assay_column: str) -> List[str]:
    """   Extracts unique assay texts from a specified column in a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing assay data.
        assay_column (str): Name of the column containing assay texts.
    Returns:
        List[str]: A list of unique assay texts, cleaned and formatted.
    """
    tmp = df[assay_column].dropna()
    if tmp.empty:
        return []
    return tmp.unique().tolist()

def clean_assay_text(assay: str) -> str:
    """ Clean the assay text by replacing certain substrings and formatting.

    Args:
        assay (str): The assay text to clean.

    Returns:
        str: The cleaned assay text.
    """
    tmp = assay.replace('/', ' and ')
    tmp = tmp.replace('BRD4 BD1 and 2', 'BRD4 BD1 and BRD4 BD2')
    tmp = tmp.replace('(Ba and F3 WT)', '(Ba/F3 WT)')
    tmp = tmp.replace('(EGFR L858R and T790M)', '(EGFR L858R/T790M)')
    return tmp

# assays = {}
# for c in protacdb_df.columns:
#     if 'Assay' in c:
#         assays[c] = get_assay_texts(protacdb_df, c)

# # Remove duplicates and clean assay texts
# texts = list(set([x for y in assays.values() for x in y]))
# print(len(texts))
# print(sum([len(x) for x in assays.values()]))

# %%
assay_to_val_cols = {
    "Assay (DC50/Dmax)": ["DC50 (nM)", "Dmax (%)"],
    "Assay (Percent degradation)": ["Percent degradation (%)"],
    "Assay (Protac to Target, IC50)": ["IC50 (nM, Protac to Target)"],
    "Assay (Protac to Target, EC50)": ["EC50 (nM, Protac to Target)"],
    "Assay (Protac to Target, Kd)": ["Kd (nM, Protac to Target)"],
    "Assay (Protac to Target, Ki)": ["Ki (nM, Protac to Target)"],
    "Assay (Protac to Target, G/H/-TS)": ["delta G (kcal/mol, Protac to Target)", "delta H (kcal/mol, Protac to Target)", "-T*delta S (kcal/mol, Protac to Target)"],
    "Assay (Protac to Target, kon/koff/t1/2)": ["kon (1/Ms, Protac to Target)", "koff (1/s, Protac to Target)", "t1/2 (s, Protac to Target)"],
    "Assay (Protac to E3, IC50)": ["IC50 (nM, Protac to E3)"],
    "Assay (Protac to E3, EC50)": ["EC50 (nM, Protac to E3)"],
    "Assay (Protac to E3, Kd)": ["Kd (nM, Protac to E3)"],
    "Assay (Protac to E3, Ki)": ["Ki (nM, Protac to E3)"],
    "Assay (Protac to E3, G/H/-TS)": ["delta G (kcal/mol, Protac to E3)", "delta H (kcal/mol, Protac to E3)", "-T*delta S (kcal/mol, Protac to E3)"],
    "Assay (Protac to E3, kon/koff/t1/2)": ["kon (1/Ms, Protac to E3)", "koff (1/s, Protac to E3)", "t1/2 (s, Protac to E3)"],
    "Assay (Ternary complex, IC50)": ["IC50 (nM, Ternary complex)"],
    "Assay (Ternary complex, EC50)": ["EC50 (nM, Ternary complex)"],
    "Assay (Ternary complex, Kd)": ["Kd (nM, Ternary complex)"],
    "Assay (Ternary complex, Ki)": ["Ki (nM, Ternary complex)"],
    "Assay (Ternary complex, G/H/-TS)": ["delta G (kcal/mol, Ternary complex)", "delta H (kcal/mol, Ternary complex)", "-T*delta S (kcal/mol, Ternary complex)"],
    "Assay (Ternary complex, kon/koff/t1/2)": ["kon (1/Ms, Ternary complex)", "koff (1/s, Ternary complex)", "t1/2 (s, Ternary complex)"],
    "Assay (Cellular activities, IC50)": ["IC50 (nM, Cellular activities)"],
    "Assay (Cellular activities, EC50)": ["EC50 (nM, Cellular activities)"],
    "Assay (Cellular activities, GI50)": ["GI50 (nM, Cellular activities)"],
    "Assay (Cellular activities, ED50)": ["ED50 (nM, Cellular activities)"],
    "Assay (Cellular activities, GR50)": ["GR50 (nM, Cellular activities)"],
    "Assay (Permeability, PAMPA Papp)": ["PAMPA Papp (nm/s, Permeability)"],
    "Assay (Permeability, Caco-2 A2B Papp)": ["Caco-2 A2B Papp (nm/s, Permeability)"],
    "Assay (Permeability, Caco-2 B2A Papp)": ["Caco-2 B2A Papp (nm/s, Permeability)"]
}

assay_cols = []
for assay_col, val_cols in assay_to_val_cols.items():
    assay_cols += val_cols + [assay_col]

key_cols = non_assay_cols = list(set(protacdb_df) - set(assay_cols))
print(f'Assay related columns: {assay_cols}')
print(f'Non-assay related columns:')
for c in non_assay_cols:
    print(f'  - {c}')

# %%
# Sort the assay_to_val_cols dictionary by the length of the dataframe of unique values in the keys and parameters columns
assay_to_val_cols = dict(sorted(
    assay_to_val_cols.items(),
    key=lambda item: len(protacdb_df[[item[0]] + item[1]].dropna(how='all').drop_duplicates()),
    reverse=True
))

for assay, cols in assay_to_val_cols.items():
    tmp = protacdb_df[[assay] + cols].dropna(how='all').drop_duplicates()

    print('-' * 100)
    print(f'Assay: {assay}')
    print(f'Number of (unique) rows: {len(tmp)}')
    print('-' * 100)
    print(tmp.sample(n=min(10, len(tmp))).to_markdown(index=False))

# %% [markdown]
# ## Extract Information from Assay-Related Columns
# 
# We define a specific parsing function for each assay-related column.

# %%
parsing_functions = {}

# %% [markdown]
# #### Assay (DC50/Dmax)

# %%
def parse_assay_dmax_dc50(text: str) -> dict:
    """
    Extracts protein targets, cell lines, and treatment times from a text string
    using regular expressions, preserving the original order of items.

    Args:
        text: The input string to parse.

    Returns:
        A dictionary with keys "targets", "cells", and "times".
        Values are lists of found items in their original order or None if not found.
    """
    # Default dictionary for results
    results = {"targets": None, "cells": None, "times": None}

    # Ignore empty or invalid lines
    if not isinstance(text, str) or not text.strip() or text.strip().isdigit() or "Degradation of" not in text:
        return results
    
    if text.strip() == 'Degradation of GPX4 in HT1080 cells after 6 h treatment/H1650 cells after 24 h treatment/H1650R cells after 24 h treatment':
        results["targets"] = ['GPX4'] * 3
        results["cells"] = ['HT1080', 'H1650', 'H1650R']
        results["times"] = [6, 24, 24]
        return results

    if text.strip() == 'Degradation of GPX4 in HT1080 cells after 6h treatment/in Calu-1 cells after 24h treatment/in H1650 cells after 24h treatment':
        results["targets"] = ['GPX4'] * 3
        results["cells"] = ['HT1080', 'Calu-1', 'H1650']
        results["times"] = [6, 24, 24]
        return results

    if text.strip() == 'Degradation of SMARCA2 proteins by the HiBit degradation assay/in SK-Mel-28 cells for 24h/in SK-Mel-5 cells for 24h/in H838 cells for 24h':
        results["targets"] = ['SMARCA2'] * 3
        results["cells"] = ['SK-Mel-28', 'SK-Mel-5', 'H838']
        results["times"] = [24, 24, 24]
        return results

    if text.strip() == 'Degradation of SMARCA4 proteins by the HiBit degradation assay/in SK-Mel-28 cells for 24h':
        results["targets"] = ['SMARCA4']
        results["cells"] = ['SK-Mel-28']
        results["times"] = [24]
        return results

    if text.strip() == 'Degradation of MEK1 in A549/ MEK1/2 in A375 cells after 24h treatment':
        results["targets"] = ['MEK1', 'MEK1', 'MEK2']
        results["cells"] = ['A549', 'A375', 'A375']
        results["times"] = [24, 24, 24]
        return results

    if text.strip() == 'Degradation of MPro in MPro-eGFP stable cell lines':
        # From thi publication: https://www.biorxiv.org/content/10.1101/2023.09.29.560163v1
        results["targets"] = ['MPro'] * 2
        results["cells"] = ['293T', 'A549']
        return results

    if text.strip() == 'Degradation of Exon 19 del/L858R EGFR in HCC827/H3255 cells after 24 h treatment':
        results["targets"] = ['EGFR Exon 19 DEL/L858R'] * 2
        results["cells"] = ['HCC827', 'H3255']
        results["times"] = [24, 24]
        return results

    if text.strip() == 'Degradation of WT/Exon 20 Ins EGFR in OVCAR8/HeLa cells after 24 h treatment':
        results["targets"] = ['EGFR WT/Exon 20 INS'] * 2
        results["cells"] = ['OVCAR8', 'HeLa']
        results["times"] = [24, 24]
        return results

    if text.strip() == 'Degradation of BRD4 BD1 assessed by EGFP/mCherry reporter assay':
        results["targets"] = ['BRD4 BD1']
        return results

    if text.strip() == 'Degradation of total tau/P-tau in A152T neurons after 24 h treatment':
        results["targets"] = ['total tau', 'P-tau']
        results["cells"] = ['A152T neurons'] * 2
        results["times"] = [24, 24]
        return results

    if text.strip() == 'Degradation of L858R, T790M EGFR in H1975 cells after 24 h treatment':
        results["targets"] = ['EGFR L858R/T790M']
        results["cells"] = ['H1975']
        results["times"] = [24]
        return results

    if text.strip() == 'Degradation of BRD9 in HEK293/BRD9-HiBiT cells after 2h treatment':
        results["targets"] = ['BRD9']
        results["cells"] = ['HEK293']
        results["times"] = [2]
        return results

    if text.strip() == 'Degradation of BRD7 in HEK293/BRD7-HiBiT cells after 2h treatment':
        results["targets"] = ['BRD7']
        results["cells"] = ['HEK293']
        results["times"] = [2]
        return results

    if text.strip() == 'Degradation of NPM-ALK/EML4-ALK in SU-DHL-1/NCI-H2228 cells after 16 h treatment':
        results["targets"] = ['NPM-ALK', 'EML4-ALK']
        results["cells"] = ['SU-DHL-1', 'NCI-H2228']
        results["times"] = [16, 16]
        return results

    if text.strip() == 'Degradation of BRD4 BD1/2 assessed by EGFP/mCherry reporter assay':
        results["targets"] = ['BRD4 BD1', 'BRD4 BD2']
        return results

    if text.strip() == 'Degradation of TPM3-TRKA/TRKA in KM12/HEL cells after 6 h treatment':
        results["targets"] = ['TPM3-TRKA', 'TRKA']
        results["cells"] = ['KM12', 'HEL']
        results["times"] = [6, 6]
        return results

    if text.strip() == 'Degradation of BRD4 short/long in HeLa cells after 24 h treatment':
        results["targets"] = ['BRD4 short', 'BRD4 long']
        results["cells"] = ['HeLa'] * 2
        results["times"] = [24, 24]
        return results
    
    if text.strip() == 'Degradation of BCL-xL in MOLT-4/platelets cells after 16 h treatment':
        results["targets"] = ['BCL-xL'] * 2
        results["cells"] = ['MOLT-4', 'AP-3']
        results["times"] = [16, 16]
        return results
    
    if text.strip() == 'Degradation of HDAC6 in MM1S/Mouse 4935 cells after 4/6 h treatment':
        results["targets"] = ['HDAC6'] * 2
        results["cells"] = ['MM1S', 'Mouse 4935']
        results["times"] = [4, 6]
        return results

    if text.strip() == 'Degradation of CDK12 in HeLa cells after 24h treatment (cytoblot/westernblot)':
        pass

    # Keep an original copy for time parsing
    original_text = text

    # --- 1. Time Extraction ---
    times = []
    # Simplified pattern to find only hour values. It finds all occurrences.
    # e.g., "4h", "6 h", "4/14/14 hrs", "2-4h"
    time_matches = re.findall(r'(\d+(?:[./-]\d+)*)\s*h(?:r|rs)?\b', original_text, re.I)
    
    for num_str in time_matches:
        # Standardize range separators ('-') to list separators ('/')
        num_str = num_str.replace('-', '/')
        for n in num_str.split('/'):
            try:
                # Convert to float for consistency, will be converted to int later if possible
                val = float(n)
                times.append(val)
            except ValueError:
                continue

    # Handle the "overnight" keyword
    if 'overnight' in original_text.lower():
        times.append(16)

    # Handle complex time formats like (DC50: 15min, Dmax: 4h)
    complex_time_matches = re.findall(r'(DC50|Dmax):\s*(\d+)(min|h)', original_text, re.I)
    for _, num_val, unit_val in complex_time_matches:
        val = float(num_val)
        if unit_val.lower() == 'min':
            val /= 60
        times.append(round(val, 2))

    if times:
        # **FIX**: Remove duplicates while preserving order, then format numbers
        results["times"] = [int(t) if isinstance(t, float) and t.is_integer() else t for t in times]

    # --- 2. Target and Cell Extraction ---
    parsing_text = text
    parsing_text = re.sub(r'\s*/\s*', '/', parsing_text)
    parsing_text = re.sub(r'\s*(?:after|for)\s+.*', '', parsing_text, flags=re.I)
    parsing_text = re.sub(r'\s+by .*$', '', parsing_text, flags=re.I)
    parsing_text = re.sub(r'\s+using .*$', '', parsing_text, flags=re.I)
    parsing_text = re.sub(r'\s+on a WES .*$', '', parsing_text, flags=re.I)
    parsing_text = re.sub(r'\s+assessed by.*$', '', parsing_text, flags=re.I)
    parsing_text = re.sub(r'\s+\(.*\)$', '', parsing_text)
    
    match = re.search(r'Degradation of\s+(.+?)(?:\s+in\s+(.+))?$', parsing_text.strip(), re.I)

    if match:
        # --- Process Targets ---
        target_str = match.group(1).strip()
        target_str = re.sub(r'\s+proteins?$', '', target_str, flags=re.I).strip()
        
        target_parts = []
        for part in re.split(r'\s+and\s+', target_str):
            # target_parts.extend(part.split('/'))
            target_parts.append(part)
        
        targets_list = [t.strip().replace('total ', '') for t in target_parts if t.strip()]
        if targets_list:
            # **FIX**: Remove duplicates while preserving order
            results["targets"] = list(dict.fromkeys(targets_list))

        # --- Process Cells ---
        cell_str = match.group(2)
        if cell_str:
            cell_str = cell_str.strip()
            cell_str = re.sub(r'\s+cells?$', '', cell_str, flags=re.I).strip()
            
            # Rename Ba/F3 to BaF3 to avoid splitting it
            cell_str = cell_str.replace('(Ba/F3)', 'BaF3')
            cell_str = cell_str.replace('Ba/F3', 'BaF3')
            
            cells_list = [c.strip() for c in cell_str.split('/') if c.strip()]
            if cells_list:
                # **FIX**: Remove duplicates while preserving order
                results["cells"] = list(dict.fromkeys(cells_list))

    # --- Post-process times ---
    # Remove times that are unreasonably high (>96 hours)
    if results["times"]:
        results["times"] = [t for t in results["times"] if t <= 96]
        if not results["times"]:
            results["times"] = None

    # If the length of the time list is one, duplicate it to be the same length
    # as the maximum of targets and cells lists
    max_len = max(len(results["targets"] or []), len(results["cells"] or []))
    if results["times"] and len(results["times"]) == 1 and max_len > 1:
        results["times"] = results["times"] * max_len

    # If times == [4, 0.25, 4], then change it to [4]
    if results["times"] == [4, 0.25, 4]:
        results["times"] = [4]

    # --- Post-process targets ---
    # If "long" is in targets, then rename it to "BRD4 long"
    if results["targets"]:
        results["targets"] = ["BRD4 long" if t.lower() == "long" else t for t in results["targets"]]
    
    # If '2 assessed' in targets, rename it to 'BRD4 BD2'
    if results["targets"] and '2 assessed' in results["targets"]:
        results["targets"] = ['BRD4 BD2' if t == '2 assessed' else t for t in results["targets"]]

    # If 'ERK1', '2' in targets, rename them to 'ERK1' and 'ERK2'
    if results["targets"] and 'ERK1' in results["targets"] and '2' in results["targets"]:
        results["targets"] = ['ERK2' if t == '2' else t for t in results["targets"]]

    # Merge mutations in targets if they are adjacent (e.g., A123B and DEL19)
    # For example: "Degradation of EGFR L858R/T790M in H1975 cells after 16h treatment",
    # shall result in targets = ['EGFR L858R/T790M'], instead of ['EGFR L858R', 'T790M']
    if results["targets"]:
        merged_targets = []
        skip_next = False
        for i in range(len(results["targets"])):
            if skip_next:
                skip_next = False
                continue
            if i < len(results["targets"]) - 1:
                # Check if the next target is a mutation (e.g., DEL19, L858R)
                if re.match(r'^(DEL|INS|DUP|FS|[A-Z]\d+[A-Z])$', results["targets"][i + 1], re.I):
                    merged_targets.append(f"{results['targets'][i]} {results['targets'][i + 1]}")
                    skip_next = True
                else:
                    merged_targets.append(results["targets"][i])
            else:
                merged_targets.append(results["targets"][i])
        results["targets"] = merged_targets

    # If the length of targets is one one, duplicate it to be the same length
    # as the maximum of cells and times lists
    max_len = max(len(results["cells"] or []), len(results["times"] or []))
    if results["targets"] and len(results["targets"]) == 1 and max_len > 1:
        results["targets"] = results["targets"] * max_len

    # --- Post-process cells ---
    # If 'Snca OE-PFF seeding HEK293T' in cells, rename it to 'HEK293T'
    if results["cells"] and 'Snca OE-PFF seeding HEK293T' in results["cells"]:
        results["cells"] = ['HEK293T' if t == 'Snca OE-PFF seeding HEK293T' else t for t in results["cells"]]

    # If 'LNCaP (AR T878A)' in cells, rename it to 'LNCaP'
    if results["cells"] and 'LNCaP (AR T878A)' in results["cells"]:
        results["cells"] = ['LNCaP' if t == 'LNCaP (AR T878A)' else t for t in results["cells"]]

    # Rename 'BaF3' back to 'Ba/F3'
    if results["cells"] and 'BaF3' in results["cells"]:
        results["cells"] = ['Ba/F3' if t == 'BaF3' else t for t in results["cells"]]

    # If the length of cells is one one, duplicate it to be the same length
    # as the maximum of targets and times lists
    max_len = max(len(results["targets"] or []), len(results["times"] or []))
    if results["cells"] and len(results["cells"]) == 1 and max_len > 1:
        results["cells"] = results["cells"] * max_len

    return results

assay_cols = [
    'Assay (DC50/Dmax)',
    'Assay (Cellular activities, IC50)',
    'Assay (Protac to Target, IC50)',
    'Assay (Percent degradation)',
]
assay_col = 'Assay (DC50/Dmax)'
assay_comments = protacdb_df[assay_col].dropna().unique().tolist()
# Sample 10 random assay comments from the assay_comments list
assay_comments = random.sample(assay_comments, min(10, len(assay_comments)))

for c in assay_comments:
    print(f"TEXT: {c}")
    print(parse_assay_dmax_dc50(c))
    print()
    print("-" * 50)

parsing_functions[assay_col] = parse_assay_dmax_dc50

# %% [markdown]
# #### Assay (Cellular activities, IC50)

# %%
def extract_inhibition_info(text: str) -> dict:
    """
    Extracts protein targets, cell lines, and treatment times from texts
    describing inhibition or proliferation assays, preserving item order.

    Args:
        text: The input string to parse.

    Returns:
        A dictionary with "targets", "cells", and "times" as keys.
    """
    results = {"targets": None, "cells": None, "times": None}

    # Ignore empty or irrelevant lines
    if not isinstance(text, str) or not text.strip():
        return results
    if any(s in text.lower() for s in ["assay kit", "determined by the mtt", "antiviral activity"]):
        return results

    # --- Handle specific hardcoded cases ---
    if text.strip() == 'Inhibit proliferation of H1975/(Ba/F3) cells expressing EGFR L858R/T790M':
        results["targets"] = ['EGFR L858R/T790M'] * 2
        results["cells"] = ['H1975', 'Ba/F3']
        return results

    if text.strip() == 'Inhibition of MDP-stimulated TNFalpha release in human whole blood':
        results["cells"] = ['TNFalpha']
        return results

    if text.strip() == 'Inhibition of the Hh pathway provoked by SAG':
        # SAG, a Smoothened agonist, activates the Hedgehog (Hh) signaling
        # pathway by directly binding to and activating the Smoothened (Smo)
        # protein, which facilitates its translocation to the primary cilium and
        # stabilization in an active form.
        return results

    if text.strip() == 'Suppression of cellular c-MYC levels measured by ELISA':
        # An "ELISA assay cell line" is not a specific type of cell line, but
        # rather a cell line that is used as the biological sample in a
        # cell-based or in-cell ELISA assay to quantify intracellular proteins,
        # signaling pathways, or other cellular functions in their native
        # environment.
        return results

    if text.strip() == 'Inhibit the cell activity of K562/the imatinib resistance KA cells':
        results["cells"] = ['K562', 'KA']
        return results

    if text.strip() == 'Inhibit the cell growth of HL-60/K562/A498':
        results["cells"] = ['HL-60', 'K562', 'A498']
        return results

    if text.strip() == 'Inhibition of cell proliferation in MCF-7/LCC2/T47D':
        results["cells"] = ['MCF-7', 'LCC2', 'T47D']
        return results

    if text.strip() == 'Inhibit viability of LNCaP/VCaP/NCI-H929':
        results["cells"] = ['LNCaP', 'VCaP', 'NCI-H929']
        return results

    if text.strip() == 'Inhibition of cell proliferation in MCF-10A (N)/MCF-7 (N)/MCF-7 (H)':
        results["cells"] = ['MCF-10A', 'MCF-7', 'MCF-7 TH']
        return results

    if text.strip() == 'Inhibit AR-independent PC-3 cell growth in LNCaP/VCaP/22Rv1 cell':
        results["cells"] = ['LNCaP', 'VCaP', '22Rv1']
        return results

    if text.strip() == 'Inhibition of viability of patient-derived CRC organoid models (MCC19990-006/MCC19990-010/MCC19990-013)':
        return results

    if text.strip() == 'Dephosphorylation of Rab10 in WT LRRK2 MEFs':
        results["targets"] = ['Rab10']
        results["cells"] = ['MEF Lrrk2 KO']
        return results

    if text.strip() == 'Dephosphorylation of Rab10 in G2019S LRRK2 MEFs':
        results["cells"] = ['MEF Lrrk2-p.G2019S KI']
        return results

    if text.strip() == 'Inhibit tumor HCT116/A549/MCF-7 cell':
        results["cells"] = ['HCT116', 'A549', 'MCF-7']
        return results

    if text.strip() == 'Inhibit proliferation of L858R/T790M EGFR-Ba/F3 cells':
        results["targets"] = ['EGFR L858R/T790M']
        results["cells"] = ['Ba/F3']
        return results

    if text.strip() == 'Inhibit growth of Ba/F3 EGFR del19/T790M/C797S cells':
        results["targets"] = ['EGFR del19/T790M/C797S']
        results["cells"] = ['Ba/F3']
        return results

    if text.strip() == 'Inhibit growth of HCC827 EGFR del19 cells':
        results["targets"] = ['EGFR DEL19']
        results["cells"] = ['HCC827']
        return results

    if text.strip() == 'Inhibit growth of H1975 EGFR L858R/T790M cells':
        results["targets"] = ['EGFR L858R/T790M']
        results["cells"] = ['H1975']
        return results

    if text.strip() == 'Inhibit proliferation of BaF3 FLT3-ITD-D835V cells':
        results["targets"] = ['FLT3-ITD D835V']
        results["cells"] = ['Ba/F3']
        return results

    if text.strip() == 'Inhibit proliferation of BaF3 FLT3-ITD-F691L cells':
        results["targets"] = ['FLT3-ITD F691L']
        results["cells"] = ['Ba/F3']
        return results

    if text.strip() == 'Inhibit cell proliferation of L858R/T790M/C797S Ba/F3 cells':
        results["targets"] = ['EGFR L858R/T790M/C797S']
        results["cells"] = ['Ba/F3']
        return results

    if text.strip() == 'Inhibition of cell proliferation in JeKo-1/BTK C481S Ba/F3 cells':
        results["targets"] = ['BTK C481S']
        results["cells"] = ['JeKo-1', 'Ba/F3']
        return results

    if text.strip() == 'Inhibit proliferation of all CD138+ patient cells':
        results["cells"] = ['293T human CD138']
        return results

    if text.strip() == 'Inhibit proliferation of K562/BCR-ABL1 transformed Ba/F3 cells':
        results["targets"] = ['EGFR L858R/T790M']
        results["cells"] = ['Ba/F3']
        return results

    if text.strip() == 'Inhibit the proliferation of Ba/F3-TEL-FGFR2/KATO III/SNU16 cells':
        results["cells"] = ['Ba/F3', 'KATO III', 'SNU16']
        return results

    if text.strip() == 'Inhibit the proliferation of Ba/F3-TEL-FGFR1 cells':
        results["targets"] = ['TEL-FGFR1']
        results["cells"] = ['Ba/F3']
        return results

    if text.strip() == 'Inhibit cell proliferation of Del19/T790M/C797S Ba/F3 cells':
        results["targets"] = ['EGFR DEL19/T790M/C797S']
        results["cells"] = ['Ba/F3']
        return results

    if text.strip() == 'Inhibit cell proliferation of Jeko-1 (7 days)/Mino (7 days)/MM.1S (7 days) cells':
        results["cells"] = ['Jeko-1', 'Mino', 'MM.1S']
        results["times"] = [7 * 24] * 3
        return results

    # Boh?
    if text.strip() == 'Cell cytotoxicity (IC50) was determined in MV4-11 cells after 48 h treatment/Post-WO IC50 was determined after 6 h treatment and a total incubation time of 48 h/No-WO IC50 was determined after 6 h treatment and a total incubation time of 48 h':
        results["cells"] = ['MV4-11', 'Post-WO', 'No-WO']
        results["times"] = [48, 6, 6]
        return results

    # --- Pre-processing ---
    original_text = text
    # Standardize separators for easier parsing
    text = re.sub(r'\s*/\s*', '/', text)
    if 'cells' in text.lower():
        # Get the text between "of" and "cells"
        match = re.search(r'(?:of|in)\s+(.+?)\s+cells', text, re.I)
        if match:
            text = match.group(1).strip()
        else:
            return results

        text = text.split('of ')[-1].strip()
        text = text.split('in ')[-1].strip()

        # Rename Ba/F3 to BaF3 to avoid splitting it (it will be renamed back later)
        text = text.replace('(Ba/F3)', 'BaF3')
        text = text.replace('Ba/F3', 'BaF3')

        # Split text by "/" only if "/" is NOT inside parentheses
        parts = re.split(r'/\s*(?![^(]*\))', text)
        parsed_cells = [part.strip() for part in parts]

        # For each cell, if it contains parentheses, extract the content inside parentheses as target
        targets = []
        cells = []
        for cell in parsed_cells:
            cell = cell.strip()
            paren_match = re.search(r'\((.*?)\)', cell)
            if paren_match:
                target = paren_match.group(1).strip()
                # Remove the parentheses part from the cell name
                cell_name = re.sub(r'\(.*?\)', '', cell).strip()
                if cell_name:
                    cells.append(cell_name)
                if target:
                    targets.append(target)
            else:
                cells.append(cell)

        if targets:
            results["targets"] = targets
        if cells:
            results["cells"] = cells

        if '5 d treatment' in original_text:
            results["times"] = [24 * 5] * max(len(results["targets"] or []), len(results["cells"] or []))

        # --- Post-process targets and cells ---
        # If 'EA. Hy926' in cells, rename it to 'EA.hy926'
        if results["cells"] and 'EA. Hy926' in results["cells"]:
            results["cells"] = ['EA.hy926' if t == 'EA. Hy926' else t for t in results["cells"]]

    # If 'the ' in cells, remove it
    if results["cells"]:
        results["cells"] = [t.replace('the ', '') for t in results["cells"]]

    # If BaF3 in cells, rename it back to Ba/F3
    if results["cells"] and 'BaF3' in results["cells"]:
        results["cells"] = ['Ba/F3' if t == 'BaF3' else t for t in results["cells"]]

    return results


assay_cols = [
    'Assay (DC50/Dmax)',
    'Assay (Cellular activities, IC50)',
    'Assay (Protac to Target, IC50)',
    'Assay (Percent degradation)',
]
assay_col = 'Assay (Cellular activities, IC50)'
assay_comments = protacdb_df[assay_col].dropna().unique().tolist()
# # Sample 10 random assay comments from the assay_comments list
assay_comments = random.sample(assay_comments, min(10, len(assay_comments)))

for c in assay_comments:
    print(f"TEXT: {c}")
    print(extract_inhibition_info(c))
    print()
    print("-" * 50)

parsing_functions[assay_col] = extract_inhibition_info

# %% [markdown]
# #### Assay (Protac to Target, IC50)

# %%
def extract_cdk_targets(text: str) -> list[str]:
    """Return a list of normalised CDK / CDK-partner targets from a sentence.

    Normalisation rules distilled from the example dictionary:
    ▸ If the CDK is joined to its partner with a *slash*,
      - return BOTH the naked CDK and the CDK-partner form.
      - Inside the partner replace every space with '-'.
      - The bare code `D1`, `D2` ... becomes `cyclin-D1`, `cyclin-D2` ...
    ▸ If the CDK is joined with a *hyphen*,
      - return ONLY the CDK-partner form.
      - Leave spaces inside the partner untouched.
      - `D1`, `D2` ... becomes `cyclin D1`, `cyclin D2` ...
    ▸ Stand-alone CDKs (no slash / hyphen behind the token) are returned as-is.
    ▸ Common typo “Cyciln” is corrected to “cyclin”.
    """

    text = re.sub(r'\b[Cc]yciln\b', 'cyclin', text)        # typo fix

    targets: list[str] = []

    pair_pat = re.compile(
        r'\b(CDK\d+)\s*([-/])\s*'                           # CDK + separator
        r'(cyclin\s?[A-Za-z0-9]+|CycT1|cyclinK|p\d+|D\d+)', # partner
        flags=re.I,
    )

    # 1) handle slash / hyphen pairs
    for cdkn, sep, raw_partner in pair_pat.findall(text):
        cdkn = cdkn.upper()
        partner = raw_partner.strip()

        # normalise bare D-numbers → cyclin X
        if re.fullmatch(r'D\d+', partner, re.I):
            if sep == '/':
                partner = f'cyclin-{partner.upper()}'
            else:                                            # sep == '-'
                partner = f'cyclin {partner.upper()}'
        else:
            if sep == '/':
                partner = partner.replace(' ', '-')          # spaces ➜ hyphens

        # assemble full CDK-partner token
        full = f'{cdkn}-{partner}'
        if sep == '/':                                       # slash ⇒ also base CDK
            if cdkn not in targets:
                targets.append(cdkn)
        if full not in targets:
            targets.append(full)

    # 2) add solitary CDKs (those **not** immediately followed by / or -)
    solo_pat = re.compile(r'\b(CDK\d+)\b(?!\s*[-/])', flags=re.I)
    for solo in solo_pat.findall(text):
        solo = solo.upper()
        if solo not in targets:
            targets.append(solo)

    return targets

def extract_protac2target_ic50(text: str) -> dict:
    """
    Extracts protein targets, cell lines, and treatment times from texts
    describing Protac to Target IC50 assays, preserving item order.

    Args:
        text: The input string to parse.

    Returns:
        A dictionary with "targets", "cells", and "times" as keys.
    """
    results = {"targets": None, "cells": None, "times": None}

    # Ignore empty or irrelevant lines
    if not isinstance(text, str) or not text.strip():
        return results

    # --- Handle specific hardcoded cases ---
    if text.strip() == 'Inhibit SIRT2 deacetylase/defatty-acylase':
        results["targets"] = ['SIRT2']
        return results

    if text.strip() == 'Inhibition activity of\xa0PARP1\xa0assay was carried out by Bioduro-Sundia':
        results["targets"] = ['PARP1']
        return results
    
    if text.strip() == 'Inhibit FEM1B-FNIP1 degron fluorescence polarization':
        results["targets"] = ['FEM1B-FNIP1']
        return results

    if text.strip() == 'Inhibition of human/murine sEH':
        results["targets"] = ['sEH human', 'sEH murine']
        return results

    if text.strip() == 'IC50 of BRD4 BD1/BD2 was tested by TR-FRET':
        results["targets"] = ['BRD4 BD1', 'BRD4 BD2']
        return results

    if text.strip() == 'IC50 of BRD4 BD1/2 was assessed by HTRF':
        results["targets"] = ['BRD4 BD1', 'BRD4 BD2']
        return results

    if text.strip() == 'IC50 of L858R/T790M EGFR was assessed using HTRF assay':
        results['targets'] = ['EGFR L858R/T790M']
        return results

    if text.strip() == 'Inhibition of EGFR L858R/T790M by ELISA':
        results['targets'] = ['EGFR L858R/T790M']
        return results

    if text.strip() == 'Inhibit of enzymatic NAMPT activities':
        results['targets'] = ['NAMPT']
        return results

    # Return empty for the following specific cases
    for s in [
        'LanthaScreen Eu IRAK3 binding displacement assay using an Alexa Fluor conjugate',
        'FP',
        'NanoBRET assay',
        'The biochemical enzymatic assay',
        'FRET',
        'ABPP',
        'TR-FRET',
    ]:
        if text.strip() == s:
            return results

    cdk_targets = extract_cdk_targets(text)
    if cdk_targets:
        results["targets"] = cdk_targets
        return results

    # --- Pre-processing ---
    original_text = text
    
    # Get everything after "of " as the relevant text
    if 'of ' in original_text:
        text = original_text.split('of ', 1)[1].strip()
        
        for after_word in [' was', ' by', ' in']:
            if after_word in text:
                text = text.split(after_word, 1)[0].strip()

    if ' to ' in original_text:
        text = original_text.split(' to ', 1)[1].strip()

    # Split text by "/"
    if '/' in text:
        results["targets"] = [t.strip() for t in text.split('/') if t.strip()]
    else:
        results["targets"] = [text.strip()]

    # --- Post-process targets ---
    # Remove ' (nM)' suffix from targets
    if results["targets"]:
        results["targets"] = [t.replace(' (nM)', '').strip() for t in results["targets"]]

    # If 'IC50' is contained in any target, remove the target from the list
    if results["targets"]:
        results["targets"] = [t for t in results["targets"] if 'IC50' not in t]
        results["targets"] = [t for t in results["targets"] if 'assay' not in t.lower()]
        if not results["targets"]:
            results["targets"] = None

    # Remove ' complex' suffix from targets
    if results["targets"]:
        results["targets"] = [t.replace(' complex', '') for t in results["targets"]]

    # Move 'human ' or 'murine' to the end of the target
    if results["targets"]:
        def move_species(t: str) -> str:
            for species in ['human', 'murine']:
                if t.lower().startswith(species + ' '):
                    return t[len(species) + 1:] + ' ' + species
            return t
        results["targets"] = [move_species(t) for t in results["targets"]]

    return results
    
assay_cols = [
    'Assay (DC50/Dmax)',
    'Assay (Cellular activities, IC50)',
    'Assay (Protac to Target, IC50)',
    'Assay (Percent degradation)',
]
assay_col = 'Assay (Protac to Target, IC50)'
assay_comments = protacdb_df[assay_col].dropna().unique().tolist()
# Sample 10 random assay comments from the assay_comments list
# assay_comments = random.sample(assay_comments, min(10, len(assay_comments)))

for c in assay_comments:
    print(f"TEXT: {c}")
    print(extract_protac2target_ic50(c)['targets'])
    print()
    print("-" * 50)

parsing_functions[assay_col] = extract_protac2target_ic50

# %% [markdown]
# #### Assay (Percent degradation)

# %%
def extract_degradation_information(text: str) -> dict:
    """
    Extracts protein targets, cell lines, and treatment times from texts
    describing Protac to Target IC50 assays, preserving item order.

    Args:
        text: The input string to parse.

    Returns:
        A dictionary with "targets", "cells", and "times" as keys.
    """
    results = {"targets": None, "cells": None, "times": None, 'dc50': None}

    # Ignore empty or irrelevant lines
    if not isinstance(text, str) or not text.strip():
        return results

    # --- Handle specific hardcoded cases ---
    if text.strip() == '% KEAP1 degradation in wild-type (WT) MM.1S cells after 4h treatment at 100/1000/10000 nM':
        results['targets'] = ['KEAP1'] * 3
        results['cells'] = ['MM.1S'] * 3
        results['times'] = [4] * 3
        results['dc50'] = [100, 1000, 10000]
        return results

    if text.strip() == '% AR-FL degradation in LNCaP cells after treatment at 1000/5000 nM and in VCaP cells after treatment at 1000/5000 nM':
        results['targets'] = ['AR-FL'] * 4
        results['cells'] = ['LNCaP', 'LNCaP', 'VCaP', 'VCaP']
        results['times'] = None
        results['dc50'] = [1000, 5000, 1000, 5000]
        return results

    if text.strip() == '% AR-FL degradation in LNCaP cells after treatment at 1000 nM and in VCaP cells after treatment at 1000/5000 nM':
        results['targets'] = ['AR-FL'] * 3
        results['cells'] = ['LNCaP', 'VCaP', 'VCaP']
        results['times'] = None
        results['dc50'] = [1000, 1000, 5000]
        return results

    if text.strip() == '% MERTK degradation in EGFR/TAM chimeric cells after treatment at 100 nM/on peritoneal macrophages from C57BL/6 mice at 10 nM':
        results['targets'] = ['MERTK'] * 2
        results['cells'] = ['TAM chimeric', 'TAM chimeric']
        results['times'] = None
        results['dc50'] = [100, 10]
        return results

    if text.strip() == '% AXL degradation in EGFR/TAM chimeric cells after treatment at 100 nM/in EO771 cells at 10 nM':
        results['targets'] = ['AXL'] * 2
        results['cells'] = ['TAM chimeric', 'EO771']
        results['times'] = None
        results['dc50'] = [100, 10]
        return results

    if text.strip() == '% pVHL30 degradation in HeLa cells after 12 h treatment at 10000 nM/1000 nM':
        results['targets'] = ['pVHL30'] * 2
        results['cells'] = ['HeLa'] * 2
        results['times'] = [12] * 2
        results['dc50'] = [10000, 1000]
        return results

    if text.strip() == '% MEK1/2 protein degradation in A549 cells at 1000 nM for 24h':
        results['targets'] = ['MEK1/2']
        results['cells'] = ['A549']
        results['times'] = [24]
        results['dc50'] = [1000]
        return results

    if text.strip() == '% MEK1/2 protein degradation in A549 cells at 10000 nM for 24h':
        results['targets'] = ['MEK1/2']
        results['cells'] = ['A549']
        results['times'] = [24]
        results['dc50'] = [10000]
        return results

    # --- Target pre-processing ---
    original_text = text
    
    # Extract the text between "%" and "degradation" or "protein degradation"
    match = re.search(r'%\s*([\w\-/\.\s]+?)\s*(?:degradation|protein degradation)', original_text, re.I)
    if match:
        text = match.group(1).strip()
        results["targets"] = [text.strip() for text in re.split(r'\s+and\s+|/', text) if text.strip()]

    # --- Cell pre-processing ---
    # Extract the text between "in" and "cells"
    match = re.search(r'\s+in\s+([\w\-/\.\s]+?)\s*cells', original_text, re.I)
    if match:
        cell_str = match.group(1).strip()

        # Rename Ba/F3 to BaF3 to avoid splitting it (it will be renamed back later)
        cell_str = cell_str.replace('(Ba/F3)', 'BaF3')
        cell_str = cell_str.replace('Ba/F3', 'BaF3')

        cells_list = [c.strip() for c in cell_str.split('/') if c.strip()]
        if cells_list:
            results["cells"] = cells_list

    # --- 1. Time Extraction ---
    times = []
    # Simplified pattern to find only hour values. It finds all occurrences.
    # e.g., "4h", "6 h", "4/14/14 hours", "2-4h", and similar formats
    time_matches = re.findall(r'(\d+(?:[./-]\d+)*)\s*h(?:r|rs|ours)?\b', original_text, re.I)
    
    for num_str in time_matches:
        # Standardize range separators ('-') to list separators ('/')
        num_str = num_str.replace('-', '/')
        for n in num_str.split('/'):
            try:
                # Convert to float for consistency, will be converted to int later if possible
                val = float(n)
                times.append(val)
            except ValueError:
                continue
    
    results["times"] = [int(t) if isinstance(t, float) and t.is_integer() else t for t in times] if times else None

    #  --- Extract DC50 values ---
    dc50s = []
    # Improved regex: matches "at 100 nM", "after 100 nM treatment", "with 100 nM", etc.
    dc50_matches = re.findall(
        r'(?:at|after|with)?\s*(\d+(?:[./-]\d+)*)\s*(nM|μM|uM)\s*(?:treatment)?',
        original_text, re.I
    )
    for num_str, unit in dc50_matches:
        num_str = num_str.replace('-', '/')
        for n in num_str.split('/'):
            try:
                val = float(n)
                if unit.lower() in ['μm', 'um']:
                    val *= 1000  # Convert μM to nM
                dc50s.append(int(val) if float(val).is_integer() else float(val))
            except ValueError:
                continue
    results["dc50"] = dc50s if dc50s else None

    # --- Post-process cells ---
    # Rename 'MM.1 S' to 'MM.1S'
    if results["cells"] and 'MM.1 S' in results["cells"]:
        results["cells"] = ['MM.1S' if t == 'MM.1 S' else t for t in results["cells"]]

    # Rename 'BaF3' back to 'Ba/F3'
    if results["cells"] and 'BaF3' in results["cells"]:
        results["cells"] = ['Ba/F3' if t == 'BaF3' else t for t in results["cells"]]

    # ['EGFR L858R', 'T790M'] ➜ ['EGFR L858R/T790M']
    if results["targets"] and 'EGFR L858R' in results["targets"] and 'T790M' in results["targets"]:
        results["targets"] = ['EGFR L858R/T790M' if t in ['EGFR L858R', 'T790M'] else t for t in results["targets"]]
        # Remove the duplicates while preserving order
        results["targets"] = list(dict.fromkeys(results["targets"]))

    # If the length of the targets list is one, duplicate it to be the same
    # length as the maximum of cells, times, and dc50 lists
    max_len = max(len(results["cells"] or []), len(results["times"] or []), len(results["dc50"] or []))

    if results["targets"] and len(results["targets"]) == 1 and max_len > 1:
        results["targets"] = results["targets"] * max_len

    if results["cells"] and len(results["cells"]) == 1 and max_len > 1:
        results["cells"] = results["cells"] * max_len

    if results["times"] and len(results["times"]) == 1 and max_len > 1:
        results["times"] = results["times"] * max_len

    return results

assay_cols = [
    'Assay (DC50/Dmax)',
    'Assay (Cellular activities, IC50)',
    'Assay (Protac to Target, IC50)',
    'Assay (Percent degradation)',
]
assay_col = 'Assay (Percent degradation)'
assay_comments = protacdb_df[assay_col].dropna().unique().tolist()
# Sample 10 random assay comments from the assay_comments list
assay_comments = random.sample(assay_comments, min(10, len(assay_comments)))

for c in assay_comments:
    print(f"TEXT: {c}")
    print(extract_degradation_information(c))
    print()
    print("-" * 50)

parsing_functions[assay_col] = extract_degradation_information

# %% [markdown]
# ### Parse Assay (DC50/Dmax)

def iterate_dict_lists(d):
    """
    Iterate over a dict of lists in parallel.

    Args:
        d (dict): mapping keys -> list or None

    Yields:
        tuple: (index, {key: value_or_None, ...})
    """
    # Prepare iterable lists: replace None with empty list
    lists = {k: v if v is not None else [] for k, v in d.items()}
    # Use zip_longest to pad shorter lists with None
    for idx, values in enumerate(zip_longest(*lists.values(), fillvalue=None)):
        yield idx, dict(zip(lists.keys(), values))

# Example usage
example_dict = {
    'A': [1, 2, 3],
    'B': ['x', 'y'],
    'C': None,
    'D': [True, False, True, False],
}
for index, combo in iterate_dict_lists(example_dict):
    print(f"Index: {index}, Combo: {combo}")

# %% [markdown]
# Let's first remove all entries with _all_ missing values in the `DC50 (nM)`, `Dmax (%)`, and `Assay (DC50/Dmax)` columns:

# %%
# Remove rows with NaN in "Assay (DC50/Dmax)" and all value columns
val_cols = assay_to_val_cols["Assay (DC50/Dmax)"]
dc50_dmax_df = protacdb_df.dropna(subset=["Assay (DC50/Dmax)"] + val_cols, how='all')

parsed_table = []

for i, row in tqdm(dc50_dmax_df.iterrows(), total=len(dc50_dmax_df), desc='Extracting DC50/Dmax info'):
    assay = row['Assay (DC50/Dmax)']
    dc50_val = row['DC50 (nM)']
    dmax_val = row['Dmax (%)']

    extracted_info = {
        'Target (DC50/Dmax)': None,
        'Cell Type (DC50/Dmax)': None,
        'Treatment Time (h) (DC50/Dmax)': None,
    }
    if pd.notnull(assay):
        extracted_info = parsing_functions['Assay (DC50/Dmax)'](assay)
        extracted_info = {
            'Target (DC50/Dmax)': extracted_info.get('targets'),
            'Cell Type (DC50/Dmax)': extracted_info.get('cells'),
            'Treatment Time (h) (DC50/Dmax)': extracted_info.get('times'),    
        }

    targets = extracted_info['Target (DC50/Dmax)']
    cells = extracted_info['Cell Type (DC50/Dmax)']
    treatment_times = extracted_info['Treatment Time (h) (DC50/Dmax)']
    dc50_vals = split_clean_str(row['DC50 (nM)']) #, return_floats=True)
    dmax_vals = split_clean_str(row['Dmax (%)']) #, return_floats=True)
    
    # If any of the DC50 is zero, print the whole row for manual checking
    if dc50_vals is not None and any(val['mean'] == 0 for val in dc50_vals if val is not None):
        print(f"Zero DC50 value found in row:\n• DC50 row: {row['DC50 (nM)']}\n• Dmax row: {row['Dmax (%)']}\n• DC50 values: {dc50_vals}\n• Dmax values: {dmax_vals}")
        print(f'https://doi.org/{row["Article DOI"]}')

    # if 'N.D.' in str(row['DC50 (nM)']):
    #     print(f'https://doi.org/{doi} has N.D. in DC50 (nM) column.\n - DC50: {row["DC50 (nM)"]}\n - Dmax: {row["Dmax (%)"]}')
    # if 'N.D.' in str(row['Dmax (%)']):
    #     print(f'https://doi.org/{doi} has N.D. in Dmax (%) column.\n - DC50: {row["DC50 (nM)"]}\n - Dmax: {row["Dmax (%)"]}')
    
    for target, cell, treat_time, dc50_val, dmax_val in zip_longest(
        targets if targets is not None else [],
        cells if cells is not None else [],
        treatment_times if treatment_times is not None else [],
        dc50_vals if dc50_vals is not None else [],
        dmax_vals if dmax_vals is not None else [],
    ):
        # Create a new row as a copy of the current row, stripped of assay
        # columns
        new_row = row[non_assay_cols + ['Dmax (%)', 'DC50 (nM)']].copy().to_dict()

        # Add the parsed information to the new row
        new_row['Assay (DC50/Dmax)'] = assay
        new_row['Target (DC50/Dmax)'] = target
        new_row['Cell Type (DC50/Dmax)'] = cell
        new_row['Treatment Time (h) (DC50/Dmax)'] = treat_time

        # return {
        #     'mean': float(match.group(1)),
        #     'error': None,
        #     'unit': match.group(2),
        #     'operator': None
        # }

        if dc50_val is not None:
            new_row['DC50 (nM) Value (DC50/Dmax)'] = dc50_val['mean']
            new_row['DC50 (nM) Error (DC50/Dmax)'] = dc50_val['error']
            new_row['DC50 (nM) Unit (DC50/Dmax)'] = dc50_val['unit']
            new_row['DC50 (nM) Operator (DC50/Dmax)'] = dc50_val['operator']

            # Replace zero values in dc50_vals with NaN
            if dc50_val['mean'] == 0:
                print(f'Warning: DC50 value is zero for assay: "{assay}"')
                new_row['DC50 (nM) Value (DC50/Dmax)'] = pd.NA

        # If the Dmax value is above 100%, we cap it at 100.0 and issue a
        # warning.
        if dmax_val is not None:
            if dmax_val["mean"] > 100:
                print(f'Warning: Dmax value {dmax_val["mean"]} for assay: "{assay}" is greater than 100, setting to 100.0 instead.')
                # dmax_val = 100.0
            new_row['Dmax (%) Value (DC50/Dmax)'] = dmax_val['mean']
            new_row['Dmax (%) Error (DC50/Dmax)'] = dmax_val['error']
            new_row['Dmax (%) Unit (DC50/Dmax)'] = dmax_val['unit']
            new_row['Dmax (%) Operator (DC50/Dmax)'] = dmax_val['operator']

        # If there is only one target, cell type, or treatment time, we assume
        # that they apply it to all experiments in the assay description.
        if treatment_times is not None and len(treatment_times) == 1:
            new_row['Treatment Time (h) (DC50/Dmax)'] = treatment_times[0]
        if cells is not None and len(cells) == 1:
            new_row['Cell Type (DC50/Dmax)'] = cells[0]
        if targets is not None and len(targets) == 1:
            new_row['Target (DC50/Dmax)'] = targets[0]

        parsed_table.append(new_row)

protacdb_dc50dmax_df = pd.DataFrame(parsed_table)
# protacdb_dc50dmax_df = pd.DataFrame(parsed_table).dropna(subset=[
#     'Target (DC50/Dmax)', 'Cell Type (DC50/Dmax)', 'Treatment Time (h) (DC50/Dmax)', 'DC50 (nM)', 'Dmax (%)'
# ], how='all')
protacdb_dc50dmax_df['Database'] = 'PROTAC-DB'

# Sort by 'Assay (DC50/Dmax)'
# protacdb_dc50dmax_df = protacdb_dc50dmax_df.sort_values(by=['Assay (DC50/Dmax)']).reset_index(drop=True)

print(protacdb_dc50dmax_df.head(50)[['Assay (DC50/Dmax)', 'DC50 (nM)', 'DC50 (nM) Value (DC50/Dmax)', 'Dmax (%)', 'Dmax (%) Value (DC50/Dmax)', 'Target (DC50/Dmax)', 'Cell Type (DC50/Dmax)', 'Treatment Time (h) (DC50/Dmax)']])
print(f'Parsed table len: {len(protacdb_dc50dmax_df)}')

# Count all None and NaN values in the parsed table "Cell Type" column
none_count = protacdb_dc50dmax_df['Cell Type (DC50/Dmax)'].isna().sum()
nan_count = protacdb_dc50dmax_df['Cell Type (DC50/Dmax)'].isnull().sum()
print(f'None count in "Cell Type": {none_count}')
print(f'NaN count in "Cell Type": {nan_count}')

# %%
protacdb_dc50dmax_df['Database'] = 'PROTAC-DB'

# %% [markdown]
# ### Parse Assay (Cellular activities, IC50)

# %%
assay_col = "Assay (Cellular activities, IC50)"
assay_name = assay_col.split('(')[-1].split(')')[0].strip()
val_cols = assay_to_val_cols[assay_col]

cell_ic50_df = protacdb_df.dropna(subset=[assay_col] + val_cols, how='all')

parsed_table = []

for i, row in tqdm(cell_ic50_df.iterrows(), total=len(cell_ic50_df), desc='Extracting info'):
    assay = row[assay_col]

    extracted_info = {
        f'Target ({assay_name})': None,
        f'Cell Type ({assay_name})': None,
        f'Treatment Time (h) ({assay_name})': None,
    }
    if pd.notnull(assay):
        # extracted_info = extrac_assay_info(assay)
        # extracted_info = {f'{k} ({assay_name})': v for k, v in extracted_info.items()}
        extracted_info = parsing_functions[assay_col](assay)
        extracted_info = {
            f'Target ({assay_name})': extracted_info.get('targets'),
            f'Cell Type ({assay_name})': extracted_info.get('cells'),
            f'Treatment Time (h) ({assay_name})': extracted_info.get('times'),
        }

    for col in val_cols:
        extracted_info[col] = split_clean_str(row[col], return_floats=True)

    for _, parsed_info in iterate_dict_lists(extracted_info):

        new_row = row[non_assay_cols].copy().to_dict()
        new_row[assay_col] = assay

        # Add the parsed information to the new row
        for col, value in parsed_info.items():
            new_row[col] = value

        # # If there is only one target, cell type, or treatment time, we assume
        # # that they apply it to all experiments in the assay description.
        # if extracted_info[f'Treatment Time (h) ({assay_name})'] is not None and len(extracted_info[f'Treatment Time (h) ({assay_name})']) == 1:
        #     new_row[f'Treatment Time (h) ({assay_name})'] = extracted_info[f'Treatment Time (h) ({assay_name})'][0]
        # if extracted_info[f'Cell Type ({assay_name})'] is not None and len(extracted_info[f'Cell Type ({assay_name})']) == 1:
        #     new_row[f'Cell Type ({assay_name})'] = extracted_info[f'Cell Type ({assay_name})'][0]
        # if extracted_info[f'Target ({assay_name})'] is not None and len(extracted_info[f'Target ({assay_name})']) == 1:
        #     new_row[f'Target ({assay_name})'] = extracted_info[f'Target ({assay_name})'][0]
        
        parsed_table.append(new_row)

protacdb_cell_ic50_df = pd.DataFrame(parsed_table).dropna(subset=[f'Target ({assay_name})', f'Cell Type ({assay_name})', f'Treatment Time (h) ({assay_name})'] + val_cols, how='all')
protacdb_cell_ic50_df['Database'] = 'PROTAC-DB'

print(f'Parsed table len: {len(protacdb_cell_ic50_df):,}')
print(protacdb_cell_ic50_df.sample(n=10))

protacdb_cell_ic50_df[f'Treatment Time (h) ({assay_name})'].unique().tolist()

# %% [markdown]
# ### Parse Assay (Protac to Target, IC50)

# %%
assay_col = "Assay (Protac to Target, IC50)"
assay_name = assay_col.split('(')[-1].split(')')[0].strip()
val_cols = assay_to_val_cols[assay_col]

protac_ic50_df = protacdb_df.dropna(subset=[assay_col] + val_cols, how='all')

parsed_table = []

for i, row in tqdm(protac_ic50_df.iterrows(), total=len(protac_ic50_df), desc='Extracting info'):
    assay = row[assay_col]
    extracted_info = {
        f'Target ({assay_name})': None,
        f'Cell Type ({assay_name})': None,
        f'Treatment Time (h) ({assay_name})': None,
    }
    if pd.notnull(assay):
        # extracted_info = extrac_assay_info(assay)
        # extracted_info = {f'{k} ({assay_name})': v for k, v in extracted_info.items()}
        extracted_info = parsing_functions[assay_col](assay)
        extracted_info = {
            f'Target ({assay_name})': extracted_info.get('targets'),
            f'Cell Type ({assay_name})': extracted_info.get('cells'),
            f'Treatment Time (h) ({assay_name})': extracted_info.get('times'),
        }

    for col in val_cols:
        extracted_info[col] = split_clean_str(row[col], return_floats=True)

    for _, parsed_info in iterate_dict_lists(extracted_info):

        new_row = row[non_assay_cols].copy().to_dict()
        new_row[assay_col] = assay

        # Add the parsed information to the new row
        for col, value in parsed_info.items():
            new_row[col] = value

        # # If there is only one target, cell type, or treatment time, we assume
        # # that they apply it to all experiments in the assay description.
        # if extracted_info[f'Treatment Time (h) ({assay_name})'] is not None and len(extracted_info[f'Treatment Time (h) ({assay_name})']) == 1:
        #     new_row[f'Treatment Time (h) ({assay_name})'] = extracted_info[f'Treatment Time (h) ({assay_name})'][0]
        # if extracted_info[f'Cell Type ({assay_name})'] is not None and len(extracted_info[f'Cell Type ({assay_name})']) == 1:
        #     new_row[f'Cell Type ({assay_name})'] = extracted_info[f'Cell Type ({assay_name})'][0]
        # if extracted_info[f'Target ({assay_name})'] is not None and len(extracted_info[f'Target ({assay_name})']) == 1:
        #     new_row[f'Target ({assay_name})'] = extracted_info[f'Target ({assay_name})'][0]
        
        parsed_table.append(new_row)

protacdb_protac_ic50_df = pd.DataFrame(parsed_table).dropna(subset=[f'Target ({assay_name})', f'Cell Type ({assay_name})', f'Treatment Time (h) ({assay_name})'] + val_cols, how='all')
protacdb_protac_ic50_df['Database'] = 'PROTAC-DB'

print(f'Parsed table len: {len(protacdb_protac_ic50_df):,}')
print(protacdb_protac_ic50_df.sample(n=10))

# %% [markdown]
# ### Parse Assay (Percent degradation)

# %%
# Remove rows with NaN in "Assay (Percent degradation)" and all value columns
assay_col = "Assay (Percent degradation)"
assay_name = 'Percent degradation'
val_cols = assay_to_val_cols[assay_col]

degr_df = protacdb_df.dropna(subset=[assay_col] + val_cols, how='all')

parsed_table = []

for i, row in tqdm(degr_df.iterrows(), total=len(degr_df), desc='Extracting info'):
    assay = row[assay_col]
    extracted_info = {
        f'Target ({assay_name})': None,
        f'Cell Type ({assay_name})': None,
        f'Treatment Time (h) ({assay_name})': None,
    }
    if pd.notnull(assay):
        extracted_info = parsing_functions[assay_col](assay)
        extracted_info = {
            f'Target ({assay_name})': extracted_info.get('targets'),
            f'Cell Type ({assay_name})': extracted_info.get('cells'),
            f'Treatment Time (h) ({assay_name})': extracted_info.get('times'),
            f'DC (nM) ({assay_name})': extracted_info.get('dc50'),
        }

    for col in val_cols:
        extracted_info[col] = split_clean_str(row[col], return_floats=True)

    for _, parsed_info in iterate_dict_lists(extracted_info):

        new_row = row[non_assay_cols].copy().to_dict()
        new_row[assay_col] = assay

        # Add the parsed information to the new row
        for col, value in parsed_info.items():
            new_row[col] = value
        
        parsed_table.append(new_row)

protacdb_degr_df = pd.DataFrame(parsed_table).dropna(subset=[f'Target ({assay_name})', f'Cell Type ({assay_name})', f'Treatment Time (h) ({assay_name})'] + val_cols, how='all')
protacdb_degr_df['Database'] = 'PROTAC-DB'

print(f'Parsed table len: {len(protacdb_degr_df):,}')
print(protacdb_degr_df.sample(n=10))

print(sorted(protacdb_degr_df[f'Percent degradation (%)'].unique().tolist()))
print(sorted(protacdb_degr_df[f'Treatment Time (h) ({assay_name})'].unique().tolist()))
print(sorted(protacdb_degr_df[f'DC (nM) ({assay_name})'].unique().tolist()))
print(sorted(protacdb_degr_df[f'Target ({assay_name})'].unique().tolist()))
# protacdb_df[f'Percent degradation (%)'].unique().tolist()

# %% [markdown]
# As far as I understood it, according to the work reported at this [DOI](https://pubs.acs.org/doi/full/10.1021/acsmedchemlett.2c00446), entries with target equal to `MEK1/2` should be split into two separate entries, one for `MEK1` and one for `MEK2`. The `DC (nM)` and `Percent degradation (%)` values are therefore duplicated and are set the same for both entries.

# %%
# Duplicate all the rows with f'Target ({assay_name})'] == 'MEK1/2', the duplicate rows should have the f'Target ({assay_name})'] set to 'MEK1' and 'MEK2' respectively
mek1_df = protacdb_degr_df[protacdb_degr_df[f'Target ({assay_name})'] == 'MEK1/2'].copy()
mek2_df = mek1_df.copy()
mek1_df[f'Target ({assay_name})'] = 'MEK1'
mek2_df[f'Target ({assay_name})'] = 'MEK2'

mek1_2_df = pd.concat([mek1_df, mek2_df], ignore_index=True).drop_duplicates()

print(len(mek1_df), mek1_df[f'Target ({assay_name})'].unique().tolist())
print(len(mek2_df), mek2_df[f'Target ({assay_name})'].unique().tolist())
print(len(mek1_2_df), mek1_2_df[f'Target ({assay_name})'].unique().tolist())

# Remove the original rows with f'Target ({assay_name})'] == 'MEK1/2'
protacdb_degr_df = protacdb_degr_df[protacdb_degr_df[f'Target ({assay_name})'] != 'MEK1/2']
print(f'Parsed table len after removing MEK1/2: {len(protacdb_degr_df):,}')
print(sorted(protacdb_degr_df[f'Target ({assay_name})'].unique().tolist()))

protacdb_degr_df = pd.concat([protacdb_degr_df, mek1_2_df], ignore_index=True).drop_duplicates()

print(f'Parsed table len after concatenating: {len(protacdb_degr_df):,}')
print(sorted(protacdb_degr_df[f'Target ({assay_name})'].unique().tolist()))

# %% [markdown]
# ## Put Parsed Tables into a Dictionary

# %%
df_dict = {
    '(DC50/Dmax)': protacdb_dc50dmax_df.copy(),
    '(Cellular activities, IC50)': protacdb_cell_ic50_df.copy(),
    '(Protac to Target, IC50)': protacdb_protac_ic50_df.copy(),
    '(Percent degradation)': protacdb_degr_df.copy(),
}

# In protacdb_protac_ic50_df, rename 'IC50 (nM, Protac to Target)' to 'IC50 (nM) (Protac to Target, IC50)'
df_dict['(Protac to Target, IC50)'] = df_dict['(Protac to Target, IC50)'].rename(
    columns={'IC50 (nM, Protac to Target)': 'IC50 (nM) (Protac to Target, IC50)'}
)
df_dict['(Cellular activities, IC50)'] = df_dict['(Cellular activities, IC50)'].rename(
    columns={'IC50 (nM, Cellular activities)': 'IC50 (nM) (Cellular activities, IC50)'}
)

# Drop unnecessary columns from each dataframe
cols_to_drop = [
    'Name',
    'Compound ID',
    'Molecular Formula',
    'InChI',
    'Hydrogen Bond Donor Count',
    'Exact Mass',
    'Topological Polar Surface Area',
    'Heavy Atom Count',
    'Molecular Weight',
    'Rotatable Bond Count',
    'Hydrogen Bond Acceptor Count',
    'PDB',
    'XLogP3',
    'Ring Count',
    'InChI Key',
    'Database',
]
df_dict = {
    df_name: df.drop(columns=[col for col in df.columns if col in cols_to_drop], errors='ignore')
    for df_name, df in df_dict.items() if len(df) > 0 and not df.columns.isin(cols_to_drop).all()
}

# For each column in the dataframes, if the column does not contain the df name,
# and it is not in the list of columns to ignore, rename the column to include
# the name of the dataframe at the end of the column name.
# For example, 'Cell Type' in the dataframe '(DC50/Dmax)' becomes 'Cell Type (DC50/Dmax)'
cols_to_ignore = [
    'Smiles',
    'E3 Ligase',
    'Target',
    'Uniprot',
    'Article DOI',
]
df_dict = {
    df_name: df.rename(
        columns={col: f"{col} {df_name}" for col in df.columns if df_name not in col and col not in cols_to_ignore}
    )
    for df_name, df in df_dict.items() if len(df) > 0 and not df.columns.isin(cols_to_ignore).all()
}

for df_name, df in df_dict.items():
    print(f"Dataframe name: {df_name}")
    for col in df.columns:
        print(f"- {col}")
    print()

# %% [markdown]
# ## Clean E3 Ligases Names
# 
# We manually mapped E3 ligases names to their Uniprot IDs. We assume human proteins only, for now.
# 
# Later further down, once we have the cell lines curated, we will assign different species to the E3 ligases, if needed.

# %%
e3ligase2uniprot = {
    'VHL': 'P40337',
    'CRBN': 'Q96SW2',
    'DCAF1': 'Q9Y4B6',
    'DCAF11': 'Q8TEB1',
    'DCAF15': 'Q66K64',
    'DCAF16': 'Q9NXF7',
    'MDM2': 'Q00987',
    'XIAP': 'P98170',
    'IAP': 'P98170', # IAP is too generic, so we set it to XIAP instead
    'cIAP1': 'Q13490',
    'AhR': 'P35869',
    'RNF4': 'P78317',
    'RNF114': 'Q9Y508',
    'FEM1B': 'Q9UK73',
    'UBR1': 'Q8IWV7',
    'UBR box': 'G3V2G3', # We associate the UBR box with the UBR7 gene
    'KLHL20': 'Q9Y2M5',
    'KLHDC2': 'Q9Y2U9',
    'FBXO22': 'Q8NEZ5',
    'KEAP1': 'Q14145',
}

# Do the same to df_dict
for df_name, df in df_dict.items():
    df['E3 Ligase'] = df['E3 Ligase'].replace({'Keap1': 'KEAP1', 'BRD4': 'VHL'})
    df['E3 Ligase Uniprot'] = df['E3 Ligase'].map(e3ligase2uniprot)

    # Count the nan values in the 'E3 Ligase' and 'E3 Ligase Uniprot' columns
    e3_ligase_nan_count = df['E3 Ligase'].isna().sum()
    e3_ligase_uniprot_nan_count = df['E3 Ligase Uniprot'].isna().sum()
    print(f"{df_name} - E3 Ligase NaN count: {e3_ligase_nan_count}, E3 Ligase Uniprot NaN count: {e3_ligase_uniprot_nan_count}")

# %% [markdown]
# ## Clean Targets Names

# %%
def clean_target(t: str) -> str:
    """ Clean the target string by removing special characters and extra spaces. """
    if pd.isnull(t):
        return t
    t = t.replace(' and ', '/').upper().strip()
    letter2unicode = {
        'alpha': 'α',
        'beta': 'β',
        'gamma': 'γ',
        'delta': 'δ',
        'epsilon': 'ε',
    }
    # Replace all greek letters
    for letter in ['alpha', 'beta', 'gamma', 'delta', 'epsilon']:
        t = t.replace(letter.upper(), letter)
        if letter in t:
            # If the greek letter is not preceded by a -, add it in front
            # If the greek letter is preceded by a space, substitute it with a hyphen
            i = t.find(letter)
            if i > 0 and t[i - 1] != '-':
                if t[i - 1] == ' ':
                    t = t[:i - 1] + '-' + t[i:]
                else:
                    t = t[:i] + '-' + t[i:]
            # If the greek letter is not at the end, and its next letter is a character and not a hyphen, add a hyphen between them
            if i + len(letter) + 1 < len(t) and t[i + len(letter)] not in ['-', ' ']:
                t = t[:i + len(letter)] + '-' + t[i + len(letter):]
            # # Finally, replace the letter with its unicode equivalent
            # t = t.replace('-' + letter, letter2unicode[letter])
            # t = t.replace(letter + '-', letter2unicode[letter])

    # If a mutation, with pattern like CharacterNumbersCharacter, is preceded by a -, substitute the - with a space
    pattern = r'-(\w+\d+\w+)'
    # Replace the pattern with a space before the mutation
    t = re.sub(pattern, r' \1', t)
    # Remove any trailing slashes or hyphens
    t = t.replace('- ', ' ')
    t = t.replace('  ', ' ')
    # Replace any mention to "fusion" or "fusion protein" with "(fusion)" at the end of the string
    if '(FUSION)' in t:
        t = t.replace('(FUSION)', '(fusion)')
    elif 'FUSION' in t or 'FUSION PROTEIN' in t:
        t = t.replace('FUSION PROTEIN', '(fusion)').replace('FUSION', '(fusion)')
    # Remove any trailing spaces
    t = t.strip()

    # Remove any mention in any case of "[-]hibit[-]"
    t = re.sub(r'[- ]?HIBIT[- ]?', '', t).strip()
    t = re.sub(r'[- ]?HiBit[- ]?', '', t).strip()
    t = re.sub(r'[- ]?Hibit[- ]?', '', t).strip()
    t = re.sub(r'[- ]?hibit[- ]?', '', t).strip()

    # Replace 'HDAC ' followed by a number with 'HDAC' followed by the number
    t = re.sub(r'HDAC (\d+)', r'HDAC\1', t).strip()

    # Remove 'HUMAN', as we already assume all targets are human by default
    t = t.replace('HUMAN', '').strip()
    t = t.replace('BROMODOMAIN', 'BD').strip()

    # # Remove ' BD' and ' BD[digit]' from the target
    # t = re.sub(r' BD\d*', '', t).strip()

    # Corner cases
    corner_cases = {
        'PDGFR-beta': 'PDGFRB',
        'CDK12-CYCLINK': 'CDK12-CYCLIN-K',
        'CDK6-CYCLIN D1': 'CDK6-CYCLIN-D1',
        'CDK6-CYCLIN D3': 'CDK6-CYCLIN-D3',        
        'CDK9-CYCT1': 'CDK9-CYCLIN-T1',
        'CDK4-CYCLIN D1': 'CDK4-CYCLIN-D1',
        'CDK4-CYCLIN D3': 'CDK4-CYCLIN-D3',
        'BTK OF HEK293 CELLS': 'BTK',
        'BPTF BROMODOMAIN': 'BPTF BD',
        'CECR2 BROMODOMAIN': 'CECR2 BD',
        'BRD9 BROMODOMAIN': 'BRD9 BD',
        'CA2': 'HCA2',
        'AR V7': 'AR-V7',
        'EGFRDEL19': 'EGFR DEL19',
        'GSK-3BATA': 'GSK3B', # 'GSK-3-beta',
        'GSK-3-beta': 'GSK3B',
        'alpha-SYN': 'alpha-SYNUCLEIN',
        'P300': 'EP300',
        'H-PGDS': 'HPGDS',
        'B-RAF': 'BRAF',
        'TCPTP': 'TC-PTP',
        'P STAT3Y705': 'p-STAT3-Y705',
        'P P38': 'p-P38',
        'HADC6': 'HDAC6',
        'BRD4-L': 'BRD4 LONG',
        'FLT-3': 'FLT3',
        'G1202R ALK': 'ALK G1202R',
        'G2019S LRRK2': 'LRRK2 G2019S',
        'HCAII': 'HCA2',
    }

    if t in corner_cases:
        return corner_cases[t]

    return t

# %% [markdown]
# Clean target names in all dataframes:

# %%
for df_name, df in df_dict.items():
    target_cols = [col for col in df.columns if col.startswith('Target')]
    for col in target_cols:
        tqdm.pandas(desc=f'Cleaning targets in {df_name}.{col}')
        df[col] = df[col].progress_apply(clean_target)

# %% [markdown]
# ## Get AA Sequences

# %% [markdown]
# ### Fetch Information from Uniprot API

# %% [markdown]
# Now that we have associated targets and Uniprot IDs, we can fetch the sequences and other relevant information from the Uniprot API.
# 
# We use the UniprotKT rest API to fetch information about the proteins. The following function fetches a JSON object for a given Uniprot ID that contains a vast amount of information about the protein, including its sequence, names, and other relevant data.

@lru_cache()
def fetch_uniprot_entry(uniprot_id: str) -> Optional[dict]:
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        time.sleep(0.5)  # To avoid hitting the API too hard
        return r.json()
    except Exception as e:
        print(f"[UniProt] fetch failed for {uniprot_id}: {e}")
        return None

# %% [markdown]
# The following is a function to fetch and filter the Uniprot JSON entries.

# %%
def extract_protein_info(uniprot_id: str, skip_isoforms: bool = False) -> Optional[dict]:
    """ Extracts detailed information about a protein from UniProt.
    
    Args:
        uniprot_id (str): The UniProt ID of the protein.
        skip_isoforms (bool): If True, skips fetching isoform information.
        
    Returns:
        dict: A dictionary containing the protein information, or None if the fetch fails. List of keys:
            - 'accession': Primary accession number.
            - 'secondary_accessions': List of secondary accession numbers.
            - 'sequence': Canonical sequence of the protein.
            - 'full_names': List of full names of the protein.
            - 'short_names': List of short names of the protein.
            - 'isoforms': List of isoform information (if not skipped).
            - 'locations': List of subcellular locations.
            - 'natural_variants': List of natural variant sequences.
            - 'natural_variants_ids': List of IDs for the natural variants.
    """
    # Fetch the UniProt entry
    entry = fetch_uniprot_entry(uniprot_id)
    if not entry:
        print(f"[UniProt] {uniprot_id} fetch failed.")
        return None

    # Setup the information dictionary to return
    info = {
        'accession': entry.get('primaryAccession'),
        'secondary_accessions': entry.get('secondaryAccessions'),
        'sequence': entry.get('sequence', {}).get('value'),
        'organism': entry.get('organism', {}).get('scientificName'),
    }

    # Obtain full names and short names
    alternative_names = entry.get('proteinDescription', {}).get('alternativeNames', [])
    info['full_names'] = [n.get('fullName', {}).get('value', 'N/A') for n in alternative_names]
    info['short_names'] = [n.get('value', 'N/A') for an in alternative_names for n in an.get('shortNames', [])]

    # Parse comments for isoforms and locations in cell
    info['isoforms'] = []
    info['locations'] = []
    comments = entry.get('comments', [])
    for comment in comments:
        # Get isoforms IDs if present, they will be recursively fetched later
        if comment.get('commentType', '') == 'ALTERNATIVE PRODUCTS':
            if not skip_isoforms:
                for isoform in comment.get('isoforms', []):
                    if isoform.get('isoformIds'):
                        info['isoforms'] += isoform['isoformIds']

        # Get subcellular locations
        elif comment.get('commentType', '') == 'SUBCELLULAR LOCATION':
            for location in comment.get('subcellularLocations', []):
                location = location.get('location', {})
                if location.get('value'):
                    info['locations'].append(location['value'])
    
    # Ensure locations are lowercase and unique
    locations = []
    for loc in info['locations']:
        for l in loc.split(', '):
            l = l.strip().lower()
            if l not in locations:
                locations.append(l)
    info['locations'] = locations

    info['natural_variants'] = []
    info['natural_variants_ids'] = []
    features = entry.get('features', [])
    for feature in features:
        if feature.get('type', '') == 'Natural variant':
            loc = feature.get('location', {})
            start = loc.get('start', {}).get('value')
            end = loc.get('end', {}).get('value')
            if start is not None and end is not None:
                alt_seq_info = feature.get('alternativeSequence', {})
                original_seq = alt_seq_info.get('originalSequence', '')
                alt_seq = ''.join(alt_seq_info.get('alternativeSequences', ['']))
                if info['sequence'][start-1:end] != original_seq:
                    print(f"[WARNING] Sequence mismatch for {uniprot_id} at {start}-{end}: {info['sequence'][start-1:end]} != {original_seq}")
                natural_variant = (
                    info['sequence'][:start-1] + alt_seq + info['sequence'][end:]
                )
                info['natural_variants'].append(natural_variant)
                info['natural_variants_ids'].append(feature.get('featureId'))

    # If isoforms are not skipped, fetch their details recursively
    # NOTE: Recursion is disabled within an isoform extraction.
    info['isoforms'] = [extract_protein_info(iso_id, skip_isoforms=True) for iso_id in info['isoforms']]
    info['isoforms'] = [iso for iso in info['isoforms'] if iso is not None]
    
    return info

# Example usage
for k, v in extract_protein_info("O60885").items():
    if k == 'isoforms':
        for i, iso in enumerate(v):
            print(f"- {k}[{i}]:")
            for sub_k, sub_v in iso.items():
                if isinstance(sub_v, list):
                    print(f"  {sub_k}: {', '.join(sub_v)}")
                else:
                    print(f"  {sub_k}: {sub_v}")
    else:
        print(f"{k}: {v}")

# %%
uniprots = []
e3_uniprots = []
for df_name, df in df_dict.items():
    uniprots += df['Uniprot'].dropna().unique().tolist()
    e3_uniprots += df['E3 Ligase Uniprot'].dropna().unique().tolist()

uniprots = list(set(uniprots))
e3_uniprots = list(set(e3_uniprots))

# %%
uniprot_dir = os.path.join(data_curation_dir, 'uniprot_infos')
os.makedirs(uniprot_dir, exist_ok=True)

uniprot2infos = {}
for uniprot_id in tqdm(uniprots, desc='Fetching UniProt entries'):
    json_info = load_dict(os.path.join(uniprot_dir, f'{uniprot_id}.json'))
    if json_info:
        uniprot2infos[uniprot_id] = json_info
    else:
        info = extract_protein_info(uniprot_id)
        if info:
            uniprot2infos[uniprot_id] = info
            # Save each entry to a separate JSON file
            save_dict(info, os.path.join(uniprot_dir, f'{uniprot_id}.json'))

for e3_uniprot_id in tqdm(e3_uniprots, desc='Fetching E3 ligase UniProt entries'):
    json_info = load_dict(os.path.join(uniprot_dir, f'{e3_uniprot_id}.json'))
    if json_info:
        uniprot2infos[e3_uniprot_id] = json_info
    else:
        info = extract_protein_info(e3_uniprot_id)
        if info:
            uniprot2infos[e3_uniprot_id] = info
            # Save each entry to a separate JSON file
            save_dict(info, os.path.join(uniprot_dir, f'{e3_uniprot_id}.json'))

# %%
# Collect all the full names and short names from the UniProt entries and
# map them to their Uniprot IDs
uniprot2names = {}
names2uniprot = {}
for uniprot_id, info in uniprot2infos.items():
    full_names = info.get('full_names', [])
    short_names = info.get('short_names', [])
    names = full_names + short_names
    if names:
        uniprot2names[uniprot_id] = names
        for name in names:
            names2uniprot[name] = uniprot_id

list(names2uniprot.keys())[:10]

# %% [markdown]
# ### Assign Sequences to Targets
# 
# We start by getting all Uniprot IDs and their associated targets. We do the same for the targets and their associated Uniprot IDs. This is done to ensure that we have all the necessary information to fetch the sequences.

# %%
# Group by Uniprot and collect targets associated with each Uniprot ID
uniprot2targets = {}
for df_name, df in df_dict.items():
    for _, row in df.iterrows():
        uniprot_id = row['Uniprot']
        if pd.notnull(uniprot_id):
            targets = uniprot2targets.get(uniprot_id, set())
            targets.add(row[f'Target {df_name}'])
            uniprot2targets[uniprot_id] = targets

# Update uniprot2targets with the entries from protacdb_df
for _, row in protacdb_df.iterrows():
    uniprot_id = row['Uniprot']
    if pd.notnull(uniprot_id):
        targets = uniprot2targets.get(uniprot_id, set())
        targets.add(clean_target(row['Target']))
        uniprot2targets[uniprot_id] = targets

# Group by Target and collect Uniprot IDs associated with each target
target2uniprots = {}
for df_name, df in df_dict.items():
    for _, row in df.iterrows():
        uniprot_id = row['Uniprot']
        target = row[f'Target {df_name}']
        if pd.notnull(target) and pd.notnull(uniprot_id):
            uniprots = target2uniprots.get(target, set())
            uniprots.add(uniprot_id)
            target2uniprots[target] = uniprots

# Update target2uniprots with the entries from protacdb_df
for _, row in protacdb_df.iterrows():
    uniprot_id = row['Uniprot']
    target = clean_target(row['Target'])
    if pd.notnull(target) and pd.notnull(uniprot_id):
        uniprots = target2uniprots.get(target, set())
        uniprots.add(uniprot_id)
        target2uniprots[target] = uniprots

# Convert sets to lists for easier handling later
for target, uniprots in target2uniprots.items():
    target2uniprots[target] = list(uniprots)

for uniprot, targets in uniprot2targets.items():
    uniprot2targets[uniprot] = list(targets)

print(f'Unique targets: {len(target2uniprots):,}')
print(f'Unique Uniprots: {len(uniprot2targets):,}')

# %%
# Print the maximum number of targets associated with a single Uniprot ID
max_targets = max(len(targets) for targets in uniprot2targets.values())
print(f'Maximum number of targets associated with a single Uniprot ID: {max_targets}')
# Print the maximum number of Uniprot IDs associated with a single target
max_uniprots = max(len(uniprots) for uniprots in target2uniprots.values())
print(f'Maximum number of Uniprot IDs associated with a single target: {max_uniprots}')

# Sort uniprot2targets and target2uniprots alphabetically on the keys
uniprot2targets = dict(sorted(uniprot2targets.items()))
target2uniprots = dict(sorted(target2uniprots.items()))

print(f'Number of unique targets: {len(target2uniprots):,}')
# print("Unique targets:")
# for target, uniprots in target2uniprots.items():
#     print(f'  - {target:25}: {", ".join(uniprots)}')

save_dict(uniprot2targets, os.path.join(data_curation_dir, 'uniprot2targets.json'))
save_dict(target2uniprots, os.path.join(data_curation_dir, 'target2uniprots.json'))

# %%
# Assign the respecitve 'Article DOI' to each target in the merged dataframe
target2dois = {}
for df_name, df in list(df_dict.items()) + [('', protacdb_df)]:
    doi_columns = [col for col in df.columns if col.startswith('Article DOI')]
    
    for _, row in df.iterrows():
        for doi_col in doi_columns:
            if pd.isnull(row[doi_col]):
                continue
            doi = row[doi_col]
            target = row['Target']
            if pd.notnull(target):
                if target not in target2dois:
                    target2dois[target] = set()
                target2dois[target].add(f'https://doi.org/{doi}')

print(f'Number of unique targets with DOIs: {len(target2dois):,}')

# %% [markdown]
# ### Divide Targets into Unique and Multiple Uniprot IDs
# 
# There are some targets that have missing or multiple Uniprot IDs associated with them. The following cells try to identify these targets and print them out for further inspection.

# %%
# Get all pair of entries in uniprot2targets and target2uniprots for which both values have length == 1
targets_w_unique_uniprot = []
for uniprot_id, targets in uniprot2targets.items():
    if len(targets) == 1:
        target = targets[0]
        if target in target2uniprots and len(target2uniprots[target]) == 1:
            if not pd.isnull(uniprot_id) and not pd.isnull(target):
                targets_w_unique_uniprot.append(target)
                # print(f"{uniprot_id} -> {target}")
print()
print(f"Number of 1-to-1 pairs: {len(targets_w_unique_uniprot)} ({len(targets_w_unique_uniprot) / len(target2uniprots):.2%} of targets)")

# %%
targets_w_many_uniprots = []
for target, uniprots in target2uniprots.items():
    if len(uniprots) > 1:
        targets_w_many_uniprots.append(target)
        print(f"{target} -> {', '.join(uniprots)} [{', '.join(target2dois.get(target, []))}]")
print()
print(f"Number of targets with multiple Uniprot IDs: {len(targets_w_many_uniprots)} ({len(targets_w_many_uniprots) / len(target2uniprots):.2%} of targets)")

# %%
targets_w_mutations = []
for target, uniprots in target2uniprots.items():
    if target in targets_w_many_uniprots or target in targets_w_unique_uniprot:
        continue
    # If a target finished with a pattern mutation, we assume it is a unique target
    if re.search(r'\b[A-Z]\d+[A-Z]\b', target) or re.search(r'\bDEL', target):
        targets_w_mutations.append(target)
        print(f"{target} -> {', '.join(uniprots)} (mutation)")
        continue
print()
print(f"Number of targets with mutations: {len(targets_w_mutations)} ({len(targets_w_mutations) / len(target2uniprots):.2%} of targets)")

targets_w_no_uniprot = []
for target, uniprots in target2uniprots.items():
    # Skip targets that have already been processed
    if target in targets_w_many_uniprots or target in targets_w_unique_uniprot or target in targets_w_mutations:
        continue
    if not uniprots:
        targets_w_no_uniprot.append(target)
        print(f"{target} -> No Uniprot ID, DOIs: {', '.join(['https://doi.org/' + doi for doi in target2dois.get(target, [])])}")
print()
print(f"Number of targets with no Uniprot ID: {len(targets_w_no_uniprot)} ({len(targets_w_no_uniprot) / len(target2uniprots):.2%} of targets)")
print()

# For targets with no Uniprot ID, we try to find a match in the names2uniprot dictionary
targets_w_no_uniprot_matches = []
for target in targets_w_no_uniprot:
    # Use fuzzy matching to find the best match in names2uniprot
    match, score = process.extractOne(target, names2uniprot.keys())
    if score >= 50:  # Set a threshold for the match quality
        uniprot_id = names2uniprot[match]
        targets_w_no_uniprot_matches.append((target, uniprot_id, match, score))
        print(f"{target} -> {uniprot_id} (matched with '{match}' with score {score})")

# %%
force_refetch = False

targets_w_no_uniprot_mapping = {
    'NS3': 'O39228',
    'alpha-TUBULIN': 'Q9UQM3',
    'beta-3-TUBULIN': 'Q13509',
    'HSP90': 'P07900',
    'C-SRC': 'P12931',
    'BCR-ABL': 'A9UF07',
    'BCR-ABL (fusion)': 'A9UF07',
    'p-P38': 'A0A510GDE6', # The sequence associated to this Mitogen-activated protein kinase p38 is not very accurate according to Uniprot. From: 10.1021/acscentsci.2c01369
    'EML4-ALK': 'A9YLN7',
    'EML4-ALK C1156Y': 'Q9UM73', # A9YLN7 sequence is too short for applying mutations on it, so we use Q9UM73 instead
    'EML4-ALK L1196M': 'Q9UM73', # A9YLN7 sequence is too short for applying mutations on it, so we use Q9UM73 instead
    'EML4-ALK L1196M/G1202R': 'Q9UM73', # A9YLN7 sequence is too short for applying mutations on it, so we use Q9UM73 instead
    'EML4-ALK G1202R': 'Q9UM73', # A9YLN7 sequence is too short for applying mutations on it, so we use Q9UM73 instead
    'BCR-ABL T315I': 'A9UF07',
    'MEK2': 'Q02750', # TODO: Possible other to decide: P36507 [https://doi.org/10.1021/acsmedchemlett.2c00446, https://doi.org/10.1021/acs.jmedchem.0c01609, https://doi.org/10.1021/acs.jmedchem.9b00810, https://doi.org/10.1021/acs.jmedchem.9b01528]
    'SMARCA2': 'P51531', # TODO: Possible other to decide: P51532 [https://doi.org/10.1021/acsmedchemlett.1c00657, https://doi.org/10.1038/s41467-023-39904-5, https://doi.org/10.1038/s41467-022-34562-5, https://doi.org/10.1021/acsmedchemlett.0c00347, https://doi.org/10.1038/s41589-019-0294-6, https://doi.org/10.1038/s41586-021-04246-z, https://doi.org/10.1038/s41467-022-33430-6, https://doi.org/10.1021/acs.jmedchem.3c01781, https://doi.org/10.1021/acs.jmedchem.3c00953]
    'STAT5': 'P51692', # TODO: Possible other to decide: P42229 [https://doi.org/10.1038/s41589-022-01248-4, https://doi.org/10.1021/acs.jmedchem.2c01665]
}

uniprot_to_fetch = set(targets_w_no_uniprot_mapping.values())

for uniprot_id in tqdm(uniprot_to_fetch, desc='Fetching missing UniProt entries'):
    json_info = load_dict(os.path.join(uniprot_dir, f'{uniprot_id}.json'))
    if json_info and not force_refetch:
        uniprot2infos[uniprot_id] = json_info
    else:
        info = extract_protein_info(uniprot_id)
        if info:
            uniprot2infos[uniprot_id] = info
            # Save each entry to a separate JSON file
            save_dict(info, os.path.join(uniprot_dir, f'{uniprot_id}.json'))

# %%
prev_targets = targets_w_many_uniprots + targets_w_unique_uniprot + targets_w_mutations + targets_w_no_uniprot
targets_leftover = []
for target, uniprots in target2uniprots.items():
    # Skip targets that have already been processed
    if target in prev_targets:
        continue
    # Print all other targets that have the same Uniprot IDs
    other_targets = [t for t, us in target2uniprots.items() if us == uniprots and t != target and t not in prev_targets]
    if other_targets:
        # print(f"{target} -> {', '.join(uniprots)} (also associated with N.{len(other_targets)}: {', '.join(other_targets)}) [{', '.join(target2dois.get(target, []))}]")
        print(f"{target} -> {', '.join(uniprots)} (also associated with N.{len(other_targets)}: {', '.join(other_targets)})")
    targets_leftover.append(target)
print()
print(f"Number of leftover targets: {len(targets_leftover)} ({len(targets_leftover) / len(target2uniprots):.2%} of targets)")

# %%
# TODO: Manually map leftover targets to Uniprot IDs after inspecting their
# sequences and the associated DOIs...
# manual_target_mapping = {}

# %% [markdown]
# ### Apply Mutations

# %%
def apply_mutation(
        seq: str,
        gene: str,
        on_error: Union[bool, Literal['raise', 'ignore']] = 'raise',
        verbose: int = 0,
) -> str:
    """ Apply the mutation to the sequence, if possible.
    
    Args:
        uniprot (str): The UniProt ID of the protein.
        gene (str): The gene name or mutation description.
        seq (str): The original protein sequence.
        on_error (str): What to do on error ('raise' or 'ignore').
        
    Returns:
        str: The mutated sequence if the mutation is valid, otherwise the original sequence.
        
    Raises:
        ValueError: If the mutation cannot be applied and `on_error` is 'raise'.
    """
    # Check if both gene and sequence are not nan
    if pd.isna(gene) or pd.isna(seq):
        return seq

    # # TODO: Just use a dictionary and replace these sequences straightaway...
    # uniprot_exceptions = {
    #     ('O60885', 'BRD4 BD1'): uniprot2sequence['O60885'],
    #     ('P25440', 'BRD2 BD2'): uniprot2sequence['P25440'],
    #     ('P10275', 'AR-V7'): uniprot2sequence['P10275'],
    #     # TODO: Not working... why???
    #     ('P00533', 'EGFR e19d'): uniprot2sequence['P10275'],
    # }
    # # Handle exceptions
    # if (uniprot, gene) in uniprot_exceptions:
    #     return uniprot_exceptions[(uniprot, gene)]

    # Use regex to get all mutations in the gene string
    if re.search(r'\b[A-Z]\d+[A-Z]\b', gene) or re.search(r'\bDEL', gene):
        mutations = re.findall(r'\b[A-Z]\d+[A-Z]\b|\bDEL\d+\b', gene.upper())
    else:
        return seq

    if verbose > 0:
        print(f'Applying mutations: {mutations} to sequence: {seq} (length: {len(seq)})')

    original_seq = seq
    del_ops = 0
    for op in mutations:
        if 'del' in op.lower():
            idx = int(op.lower().split('del')[1]) - 1
            seq = seq[:idx] + seq[idx + 1:]
            del_ops += 1
        else:
            # Replace aminoacid at a specific index
            # NOTE: The indexing starts from one, not zero.
            curr, idx, mutation = op[0].upper(), int(op[1:-1])-1, op[-1].upper()
            # NOTE: If a deletion has happened before, the index is still
            # relative to the whole sequence lenght (weird...)
            idx -= del_ops
            if verbose > 1:
                print(f'Operation: {op} on ...{seq[idx-8:idx]}[{seq[idx]} -> {mutation}]{seq[idx+1:idx+8]}...')
            if idx < 0 or idx >= len(seq):
                msg = f'Index {idx} out of bounds for sequence of length {len(seq)}.'
                if on_error == 'raise' or on_error is True:
                    raise ValueError('ERROR. ' + msg)
                else:
                    if verbose > 0:
                        print('WARNING. ' + msg + ' No mutation is applied.')
                    return original_seq

            if curr != seq[idx]:
                msg = f'Replacement at position {idx} failed. Expected "{curr}", found: "{seq[idx]}".'
                if on_error == 'raise' or on_error is True:
                    raise ValueError('ERROR. ' + msg)
                else:
                    if verbose > 0:
                        print('WARNING. ' + msg + ' No mutation is applied.')
                    return original_seq
            seq = seq[:idx] + mutation + seq[idx + 1:]

    return seq

uniprot_id = 'P00533'
target = 'EGFR DEL19/T790M/C797S'

infos = uniprot2infos[uniprot_id]
seq = infos['sequence']
mutated_seq = apply_mutation(seq, target, on_error='ignore', verbose=1)

print(f'Original sequence for {uniprot_id}: {seq}')
print(f'Mutated sequence for {uniprot_id}:  {mutated_seq}')
assert seq != mutated_seq, f'Sequence for {uniprot_id} was not mutated as expected: {seq} == {mutated_seq}'

# %% [markdown]
# ### Assign Missing Uniprot IDs to Fuzzy Targets

# %%
def assign_missing_uniprot(row, df_name, force_replace: bool = False):
    if pd.notna(row['Uniprot']):
        return row['Uniprot']

    target = row[f'Target']
    target_parsed = row[f'Target {df_name}']

    # NOTE: The specific target column will have precedence over the generic
    # 'Target' column from the original PROTAC-DB dataframe.
    if target_parsed in targets_w_no_uniprot_mapping:
        return targets_w_no_uniprot_mapping[target_parsed]

    if target in targets_w_no_uniprot_mapping:
        return targets_w_no_uniprot_mapping[target]

    uniprots = target2uniprots.get(target_parsed, [])
    if len(uniprots) == 1:
        return uniprots[0]

    uniprots = target2uniprots.get(target, [])
    if len(uniprots) == 1:
        return uniprots[0]

    return pd.NA

missing_target2uniprots = []

for df_name, df in df_dict.items():
    print(f"Number of missing Targets in {df_name}: {df['Target'].isna().sum():,}")
    print(f"Number of missing Targets {df_name} in {df_name}: {df[f'Target {df_name}'].isna().sum():,}")
    print(f"Number of missing Uniprot IDs in {df_name}: {df['Uniprot'].isna().sum():,}")
    df['Uniprot'] = df.apply(lambda row: assign_missing_uniprot(row, df_name, force_replace=True), axis=1)
    print(f"Number of missing Uniprot IDs in {df_name} after assignment: {df['Uniprot'].isna().sum():,}")
    print()
    
    for _, row in df.iterrows():
        if pd.isna(row['Uniprot']):
            target_cols = [col for col in df.columns if col.startswith('Target')]
            targets = [row[col] for col in target_cols if pd.notna(row[col])]
            missing_target2uniprots += targets

missing_target2uniprots = list(set(missing_target2uniprots))
print(f"Number of targets with missing Uniprot IDs: {len(missing_target2uniprots):,}")
print("Missing targets:")
for target in missing_target2uniprots:
    print(f'  - {target}')

assert len(missing_target2uniprots) == 0, "There are still targets with missing Uniprot IDs."

# %% [markdown]
# ### Add a 'POI Sequence' and a 'E3 Ligase Sequence' Columns 

# %%
def get_sequence_from_uniprot(uniprot_id):
    if pd.isnull(uniprot_id):
        return None
    info = uniprot2infos.get(uniprot_id)
    if info is not None and 'sequence' in info:
        return info['sequence']
    return None

for df_name, df in df_dict.items():
    # Add 'POI Sequence' column using the 'Uniprot' column
    df['POI Sequence'] = df['Uniprot'].apply(get_sequence_from_uniprot)
    
    # Add 'E3 Ligase Sequence' column using the 'E3 Ligase Uniprot' column
    df['E3 Ligase Sequence'] = df['E3 Ligase Uniprot'].apply(get_sequence_from_uniprot)

    # Count the number of NaN values in the 'POI Sequence' and 'E3 Ligase Sequence' columns
    poi_seq_nan_count = df['POI Sequence'].isna().sum()
    e3_ligase_seq_nan_count = df['E3 Ligase Sequence'].isna().sum()
    print(f"{df_name} - POI Sequence NaN count: {poi_seq_nan_count}")
    print(f"E3 Ligase Sequence NaN count: {e3_ligase_seq_nan_count}")
    print()

# %%
# Apply mutation to the 'POI Sequence' column based on the 'Target' columns
def apply_mutation_to_sequence(row):
    uniprot = row['Uniprot']
    seq = row['POI Sequence']
    target = row['Target']
    
    if pd.isnull(seq) or pd.isnull(target):
        return seq
    try:
        mutated_seq = apply_mutation(seq, target, on_error='ignore', verbose=0)
        return mutated_seq
    except ValueError as e:
        print(f"Applying mutation for {row['Uniprot']} with target '{target}'")
        print(f"Error applying mutation for {row['Uniprot']} with target '{target}': {e}")
        return seq

for df_name, df in df_dict.items():
    df['POI Sequence'] = df.apply(apply_mutation_to_sequence, axis=1)

# %% [markdown]
# ## Standardize Cell Names
# %%
# Build a dictionary mapping each cell type to a set of DOIs
celltype2dois = {}
for df_name, df in df_dict.items():
    doi_cols = [col for col in df.columns if 'Article DOI' in col]
    cell_cols = [col for col in df.columns if 'Cell Type' in col]
    for _, row in df.iterrows():
        for doi_col in doi_cols:
            for cell_col in cell_cols:
                doi = row[doi_col]
                cell = row[cell_col]
                if pd.isnull(doi) or pd.isnull(cell):
                    continue
                if cell not in celltype2dois:
                    celltype2dois[cell] = set()
                celltype2dois[cell].add(f'https://doi.org/{doi}')

# Convert sets to sorted lists for easier handling later
for cell, dois in celltype2dois.items():
    celltype2dois[cell] = sorted(dois)

print(f'Number of unique cell types with DOIs: {len(celltype2dois):,}')


# Skip defining the cell_embedding if already instantiated
if 'cell_embedding' not in globals():
    pass
cell_embedding = CellEmbedding(load_from_cache=False)

# %% [markdown]
# Based on fuzzy matches, and after checking the referenced DOIs, we manually standardized the cell names by mapping them to entries in the Cellosaurus database.
# 
# Some notes:
# 
# - "Primary" refers to cells taken from patients, so they are not a single cell line, but rather a mix of cells (think about a tissue)

# %%
manual_cell_lines = {
    '22Rv1 prostate cancer': '22Rv1', # 22Rv1 prostate cancer cell line
    '231MFP': 'MDA-MB-231',
    'A152T neurons': 'Sporadic FTD iPSC #9', # Mutations in the gene encoding tau (MAPT) in neurons
    'A549 lung cancer': 'A-549',
    'BMDM': 'Bone marrow macrophage immortalized BALB/c', # Bone Marrow Derived Macrophages
    'BaF3 FLT3-ITD': 'Ba/F3 FLT3-ITD [KYinno]',
    'DA-MB-231': 'MDA-MB-231',
    'EGFR': 'Bone marrow macrophage immortalized BALB/c', # Modified BMDM cell line with EGFR/TAM overexpression: 10.3389/fimmu.2023.1135373
    'ER-positive breast cancer cell lines': 'MCF-7', # And T-47D
    'H1650R': 'H1650', # H1650 with multiple radiation applied, not available in Cellosaurus, reported in: 10.1016/j.bmc.2022.117115
    'H293T': 'HL-60', # HL-60 leukemia cells, as reported in 10.1021/acs.jmedchem.2c01659
    'HAP 1': 'HAP1',
    'HBL-1': 'HBL1',
    'HEK293-hTau': 'HEK293-H',
    'HT-1080 fibrosarcoma': 'HT-1080',
    'Hep3B2 1-7': 'Hep 3B2.1-7',
    'IL2 PBMC': 'PBMC iPSC #1', # interleukin 2 (IL2) peripheral blood mononuclear cell (PBMC)
    'IgE MM': 'U266B1', # IgE multiple myeloma cell line
    'KU812 CML': 'Ku812', # chronic myelogenous leukemia (CML)
    'L-O2': 'LO2', # Human normal liver cell line
    'LnCaP95': 'LNCaP95',
    'MB-MDA-231': 'MDA-MB-231',
    'MDA-Pca-2b': 'MDA-PCa-2b',
    'MEK1': 'A-549', # MEK1/2 is the target, according to: 10.1021/acsmedchemlett.2c00446
    'MM.1 S': 'MM.1S',
    'MM.1S wild-type (WT)': 'MM.1S',
    'MPro-eGFP stable': '293T', # 293T cells that stably express M^Pro-eGFP, from: 10.1101/2023.09.29.560163
    'MV4; 11': 'MV4-11',
    'MV4;11': 'MV4-11',
    'Molm-16': 'MOLM-16',
    'Molm-13': 'MOLM-13',
    'Mouse 4935': 'FO [Mouse myeloma]', # Found in supplementary information of: 10.1021/acsmedchemlett.0c00046, "Mouse 4935 cells was generated by Dr. Jing Zhang’s lab (University of Wisconsin-Madison). 4935 cell line was established from a VkMYC; NrasQ61R/+ mouse which developed an aggressive multiple myeloma."
    'OCI-ly10': 'OCI-Ly10',
    'PBMC': 'PBMC iPSC #1',
    'hPBMC': 'PBMC iPSC #1', # Human
    'PBMC cells': 'PBMC iPSC #1',
    'PC3-S1': 'PC3-STEAP-1',
    'PDX SJBALL020589': 'ALL-1', # PDX (Patient-derived xenograft), but they specify ALL cell in: 10.1021/acsmedchemlett.1c00650
    'Primary Cardiomyocytes': 'C2C12', # They also report using HeLa cells in: 10.1038/s41589-019-0379-2
    'RPMI-8826': 'RPMI-8226', # It's a typo, they use RPMI-8226 in: 10.1021/acs.jmedchem.2c01817
    'RS4; 11': 'RS4;11',
    'SRD15': 'SRD-15',
    'Sk-Mel-28': 'SK-MEL-28',
    'T-cell': 'Jurkat',
    'TAM': 'EO771', # The authors in 10.3389/fimmu.2023.1135373 also employ engineered EGFR-TAM chimeric reporter cell lines and primary bone-marrow-derived macrophages for selectivity and mechanistic assays, but EO771 is the only bona-fide established cell line used to assess PROTAC efficacy.
    'TAM chimeric': 'EO771', # The authors in 10.3389/fimmu.2023.1135373 also employ engineered EGFR-TAM chimeric reporter cell lines and primary bone-marrow-derived macrophages for selectivity and mechanistic assays, but EO771 is the only bona-fide established cell line used to assess PROTAC efficacy.
    'Taxol': 'A549-Taxol',
    'VCaP AR+': 'VCaP', # VCaP prostate cancer cell line with AR overexpression
    'XLA': 'Ba/F3 BTK C481S', # XLA (X-linked agammaglobulinemia) is a condition caused by a mutation in the BTK gene: 10.1021/acs.biochem.8b00391
    'germ': 'SCIT-C8', # Primary germ cells, also known as primordial germ cells (PGCs), are the precursor cells to sperm and eggs (gametes)
    'human dermal papilla': 'HaCaT', # They both tested the PROTAC on 'HaCaT' and 'HSA-S4' in: 10.1002/smtd.202201293
    'platelets': 'MOLT-4',
    'primary Sertoli': 'human Sertoli 1',
    'Jeko-1 (7 days)': 'Jeko-1',
    'Mino (7 days)': 'Mino',
    'MM.1S (7 days)': 'MM.1S',
    'melanoma A375': 'A-375', # A375 melanoma cell line
    'LPS-Induced RAW264.7': 'RAW264.7', # Lipopolysaccharide (LPS)-induced RAW264.7 macrophages
    'class IIa Jurkat': 'Jurkat',
    'T-cell leukemia Jurkat': 'Jurkat',
    'MV4-11 (WDR5-HiBiT)': 'MV4-11',
    'Kelly cells 16 h treatment': 'Kelly',
    'EOL-1 cells 4 h treatment': 'EOL-1',
    'MEFs': 'MEF',
    'Hep3B2.1-7': 'Hep 3B2.1-7',
    'KU182': 'Ku812',
    'HepG-2': 'Hep-G2',
    'CCK-8': 'CCK-81',
    'TRIM37-amplified MCF-7 breast cancer': 'MCF-7',
    'Kasumi': 'Kasumi-1',
    'TNFalpha': 'THP-1', # Checked the publication: https://pubs.acs.org/doi/10.1021/acs.jmedchem.1c01118
    'ALL': 'ALL-1', # Replace ALL (a fish cell line) with ALL-1 (a human cell line), see: https://pubs.acs.org/doi/10.1021/acsmedchemlett.3c00082
    'C481S BTK': 'BTK C481S',
}

# %% [markdown]
# If we wanna see them in a more readable format, together with their publications, we can do:

# %%
def get_df(row):
    article_doi = 'https://doi.org/' + row['Article DOI'].split(';')[0].strip() if pd.notna(row['Article DOI']) else None
    cell_cols = [col for col in row.index if 'Cell Type' in col]
    cell_type = row[cell_cols[0]] if cell_cols else pd.NA
     # If there are multiple cell type columns, prefer the one that is not NaN
    if pd.isna(cell_type):
        return pd.Series({
            'Cell Type': cell_type,
            'Manually Curated Cell Line (in CelloSaurus)': None,
            'Article DOI': article_doi,
        })
    if cell_type not in manual_cell_lines:
        return pd.Series({
            'Cell Type': cell_type,
            'Manually Curated Cell Line (in CelloSaurus)': None,
            'Article DOI': article_doi,
        })
    standardized = manual_cell_lines[cell_type]
    return pd.Series({
        'Cell Type': cell_type,
        'Manually Curated Cell Line (in CelloSaurus)': standardized,
        'Article DOI': article_doi,
    })

tmp = []
for df_name, df in df_dict.items():
    tmp.append(df.apply(get_df, axis=1).dropna())

tmp = pd.concat(tmp, axis=0).drop_duplicates().reset_index(drop=True)
print(tmp.to_markdown())

# %%
def standardize_cell_line(row):
    cell_cols = [col for col in row.index if col.startswith('Cell Type')]
    for cell_col in cell_cols:
        cell_line = row[cell_col]
        row[cell_col.replace('Cell Type', 'Cell ID')] = pd.NA  # Initialize Cell ID column

        if pd.isna(cell_line):
            continue

        # First, apply manual mapping if available
        cell_line = manual_cell_lines.get(cell_line, cell_line)

        # Check if the cell line is already standardized
        if cell_line in cell_embedding.synonym2cell_line:
            row[cell_col] = cell_embedding.synonym2cell_line[cell_line]

        elif cell_line in cell_embedding.cell2data:
            row[cell_col] = cell_line
        else:
            # Print a warning if the cell line is not found
            assay_col = [c for c in row.index if 'Assay' in c][0]
            assay = row[assay_col]
            article_doi = 'https://doi.org/' + row['Article DOI']
            if 'DC50 (nM)' in row:
                tmp = protacdb_df[protacdb_df['Assay (Dmax/DC50)'] == assay]
            print('WARNING: Cell line not found in Cellosaurus embeddings:')
            print('•', assay, article_doi, cell_line)
            continue

        cell_id = cell_embedding.cell2cell_id.get(cell_line)
        row[cell_col.replace('Cell Type', 'Cell ID')] = cell_id
    return row

# Apply the standardization function to each row
for df_name, df in df_dict.items():
    print('--' * 40)
    print(f"Standardizing cell lines in {df_name}...")
    print('--' * 40)
    df_dict[df_name] = df.apply(standardize_cell_line, axis=1)
    print(df_dict[df_name].head(3))

# %% [markdown]
# ## Infer Missing Data based on DOIs

# %% [markdown]
# ~~Many rows lack cell type information, even in the assay columns. We infer the missing cell lines by cross-referencing the cell lines in all the parsed assay-specific dataframes. In particular, we look for the combination of SMILES, E3 ligase, Uniprot ID, and publication DOI. If we find a match, we use the cell line from that row to fill in the missing cell line in the current row.~~
# 
# We skip the above step, as there exist cell-free assays, so we cannot assume that the cell line is always present.

# %%
# from collections import defaultdict

# smiles_e3_uniprot_doi_2_missing = defaultdict(list)
# for df_name, df in df_dict.items():
#     print(f"Processing {df_name}...")

#     missing_col = f'Cell Type {df_name}'
#     if missing_col not in df.columns:
#         print(f"Column '{missing_col}' not found in {df_name}. Skipping...")
#         continue

#     for _, row in df.iterrows():
#         smiles = row['Smiles']
#         e3 = row['E3 Ligase']
#         uniprot = row['Uniprot']
#         doi = '' # row['Article DOI']

#         missing_cell = row[missing_col]
#         # Continue if any of the values are NaN
#         # if pd.isna(smiles) or pd.isna(e3) or pd.isna(uniprot) or pd.isna(doi) or pd.isna(missing_cell):
#         if pd.isna(smiles) or pd.isna(e3) or pd.isna(uniprot) or pd.isna(doi) or pd.isna(missing_cell):
#             continue
#         smiles_e3_uniprot_doi_2_missing[(smiles, e3, uniprot, doi)].append(missing_cell)

# # Remove all entries that have more than one cell type
# smiles_e3_uniprot_doi_2_missing = {
#     k: v[0] for k, v in smiles_e3_uniprot_doi_2_missing.items() if len(v) == 1
# }

# # Loop over all dataframes and if there is a match in the
# # smiles_e3_uniprot_doi_2_missing dictionary, and the cell type is Nan, add
# # the cell type to the row
# print('-' * 80)
# for df_name, df in df_dict.items():
#     print(f"Processing {df_name}...")
#     num_empty_val = 0
#     num_edits = 0
    
#     missing_col = f'Cell Type {df_name}'
#     if missing_col not in df.columns:
#         print(f"Column '{missing_col}' not found in {df_name}. Skipping...")
#         continue

#     for idx, row in df.iterrows():
#         smiles = row['Smiles']
#         e3 = row['E3 Ligase']
#         uniprot = row['Uniprot']
#         doi = '' # row['Article DOI']

#         missing_cell = row[missing_col]
#         if pd.isna(missing_cell):
#             num_empty_val += 1
#             key = (smiles, e3, uniprot, doi)
#             if key in smiles_e3_uniprot_doi_2_missing:
#                 num_edits += 1
#                 # print(f"Adding cell type {smiles_e3_uniprot_doi_2_missing[key]} for {key} in {df_name}")
#                 # df.at[idx, missing_col] = smiles_e3_uniprot_doi_2_missing[key]
#                 df.at[idx, missing_col] = smiles_e3_uniprot_doi_2_missing[key]

#     print(f"Number of empty {missing_col}: {num_empty_val:,} ({num_empty_val / len(df) * 100:.2f}%)")
#     print(f"Number of edits made to add {missing_col}: {num_edits:,}")
#     print(f"Coverage of {missing_col}: {num_edits / (num_empty_val + 1e-8):.2%}")
#     print('-' * 80)

# %%
# Remove columns with all NaN values
for df_name, df in df_dict.items():
    print(f"Number of columns in {df_name} before removing all-NaN columns: {len(df.columns.tolist())}")
    df_dict[df_name] = df.dropna(axis=1, how='all')
    print(f"Number of columns in {df_name} after removing all-NaN columns: {len(df_dict[df_name].columns.tolist())}")
    print()

# %% [markdown]
# ## Add Species Based on Cell Lines

# %%
def get_cell_species(cell_type):
    if pd.isna(cell_type):
        return np.nan
    data = cell_embedding.cell2data.get(cell_type)
    if data and 'OX' in data:
        organism = data.get('OX')
        if organism is not None:
            # Example: from "NCBI_TaxID=9606; ! Homo sapiens (Human)"
            #          return "Homo sapiens"
            return organism.split(' ! ')[-1].split(' (')[0]
    return np.nan

for df_name, df in df_dict.items():
    cell_col = [c for c in df.columns if c.startswith('Cell Type')]
    if cell_col:
        df['Cell Species'] = df[cell_col[0]].apply(get_cell_species)
        print(df['Cell Species'].unique())

# %% [markdown]
# ### Modify E3 Ligase Based on Species

# %%
e3ligase2uniprot = {
    'Homo sapiens': {
        'VHL': 'P40337',
        'CRBN': 'Q96SW2',
        'DCAF1': 'Q9Y4B6',
        'DCAF11': 'Q8TEB1',
        'DCAF15': 'Q66K64',
        'DCAF16': 'Q9NXF7',
        'MDM2': 'Q00987',
        'XIAP': 'P98170',
        'IAP': 'P98170', # IAP is too generic, so we set it to XIAP instead
        'cIAP1': 'Q13490', # BIRC2_HUMAN
        'AhR': 'P35869',
        'RNF4': 'P78317',
        'RNF114': 'Q9Y508',
        'FEM1B': 'Q9UK73',
        'UBR1': 'Q8IWV7',
        'UBR box': 'G3V2G3', # We associate the UBR box with the UBR7 gene
        'KLHL20': 'Q9Y2M5',
        'KLHDC2': 'Q9Y2U9',
        'FBXO22': 'Q8NEZ5',
        'KEAP1': 'Q14145',
    },
    'Mus musculus': {
        'CRBN': 'Q8C7D2',
        'VHL': 'P40338',
        'cIAP1': 'Q62210', # BIRC2_MOUSE
        'MDM2': 'P23804',
        'FEM1B': 'Q9Z2G0',
    },
    'Cricetulus griseus': {
        'CRBN': 'Q96SW2', # <- It's safer to use the human one # 'G3ICB0', # G3ICB0_CRIGR
    },
    'Rattus norvegicus': {
        'CRBN': 'Q56AP7',
    },
}

uniprots = set()
for _, old2new in e3ligase2uniprot.items():
    for _, new in old2new.items():
        uniprots.add(new)

# -- Fetch and cache UniProt entries defined above --
for uniprot_id in tqdm(uniprots, desc='Fetching UniProt entries'):
    json_info = load_dict(os.path.join(uniprot_dir, f'{uniprot_id}.json'))
    if json_info:
        uniprot2infos[uniprot_id] = json_info
    else:
        info = extract_protein_info(uniprot_id)
        if info:
            uniprot2infos[uniprot_id] = info
            # Save each entry to a separate JSON file
            save_dict(info, os.path.join(uniprot_dir, f'{uniprot_id}.json'))


# -- Update the 'E3 Ligase Uniprot' column based on the 'E3 Ligase' and 'Cell Species' columns --
def update_e3ligase_uniprot(row):
    e3 = row['E3 Ligase']
    cell_species = row['Cell Species']
    current_uniprot = row['E3 Ligase Uniprot']

    if pd.isna(e3) or pd.isna(cell_species):
        return current_uniprot
    
    uniprot = e3ligase2uniprot.get(cell_species, {}).get(e3)
    if uniprot:
        return uniprot
    return current_uniprot

# -- Update the 'E3 Ligase Sequence' based on the 'E3 Ligase Uniprot' --
def update_e3ligase_sequence(row):
    uniprot_id = row['E3 Ligase Uniprot']
    current_sequence = row['E3 Ligase Sequence']

    if pd.isna(uniprot_id):
        return current_sequence
    
    info = uniprot2infos.get(uniprot_id)
    if info and 'sequence' in info:
        return info['sequence']
    return current_sequence


for df_name, df in df_dict.items():
    print('--' * 40)
    if 'Cell Species' not in df.columns:
        print(f"'Cell Species' column not found in {df_name}. Skipping...")
        continue
    print(f"Dataframe: {df_name}")
    tqdm.pandas(desc='Updating E3 Ligase Uniprot')
    df['E3 Ligase Uniprot'] = df.progress_apply(update_e3ligase_uniprot, axis=1)

    tqdm.pandas(desc='Updating E3 Ligase Sequence')
    df['E3 Ligase Sequence'] = df.progress_apply(update_e3ligase_sequence, axis=1)

# %% [markdown]
# ### Modify POI Based on Species

# %% [markdown]
# Check whether the reported Uniprot IDs match the species of the cell line used in the assay. If not, try to find a matching Uniprot ID for the reported target name and the species of the cell line.

# %%
def get_poi_species(uniprot):
    if pd.isna(uniprot):
        return None
    infos = uniprot2infos.get(uniprot)
    if infos and 'organism' in infos:
        return infos['organism']
    return None

for df_name, df in df_dict.items():
    df['POI Species'] = df['Uniprot'].apply(get_poi_species)

# %%
print('List of non-human POI entries:')
for df_name, df in df_dict.items():
    print()
    print(f"Dataframe: {df_name}")
    print('-' * 40)
    cell_col = [c for c in df.columns if c.startswith('Cell Type')]
    doi_cols = [c for c in df.columns if c.startswith('Article')]
    target_cols = [c for c in df.columns if c.startswith('Target')]
    if not cell_col:
        print(f"'Cell Type' column not found in {df_name}. Skipping...")
        continue
    cell_col = cell_col[0]

    tmp = df[(df['Cell Species'] != 'Homo sapiens') & df['Cell Species'].notnull()]
    # Add "https://doi.org/" in front of the doi_cols
    for col in doi_cols:
        tmp[col] = tmp[col].apply(lambda x: f"https://doi.org/{x}" if pd.notna(x) and not str(x).startswith("https://doi.org/") else x)

    print(tmp.drop_duplicates(subset=['Uniprot', 'Cell Species'])[['Uniprot', *target_cols, cell_col, 'Cell Species'] + doi_cols].fillna('-').reset_index(drop=True).to_markdown(index=True))

# %% [markdown]
# The following is a manual mapping of Uniprot IDs that were found to be incorrect based on the species of the cell line used in the assay. The new Uniprot IDs were found by searching the Uniprot database for the target name and the species of the cell line.
# 
# |    | Uniprot   | New Uniprot   | Cell Species       | Comment                                |
# |---:|:----------|:--------------|:-------------------|:---------------------------------------|
# |  0 | Q9ULX9    | Q01279        | Mus musculus       |                                        |
# |  1 | P00533    | O54791        | Mus musculus       |                                        |
# |  2 | P11802    | P30285        | Mus musculus       |                                        |
# |  3 | Q00534    | Q64261        | Mus musculus       |                                        |
# |  4 | P36888    | Q00342        | Mus musculus       |                                        |
# |  5 | P04035    | P00347        | Cricetulus griseus |                                        |
# |  6 | Q07817    | Q64373        | Mus musculus       |                                        |
# |  7 | Q14145    | Q9Z2X8        | Mus musculus       |                                        |
# |  8 | O60885    | B2RSE4        | Mus musculus       |                                        |
# |  9 | P24941    | P97377        | Mus musculus       |                                        |
# | 10 | A9UF07    | P00520        | Mus musculus       |                                        |
# | 11 | Q8TBX8    | Q91XU3        | Mus musculus       |                                        |
# | 12 | P06493    | P11440        | Mus musculus       |                                        |
# | 13 | P14625    | P08113        | Mus musculus       |                                        |
# | 14 | A9YLN7    | Q3UMY5        | Mus musculus       |                                        |
# | 15 | Q9UM73    | Q3UMY5        | Mus musculus       |                                        |
# | 16 | P53350    | Q07832        | Mus musculus       |                                        |
# | 17 | Q05397    | O35346        | Rattus norvegicus  |                                        |
# | 18 | Q05397    | P34152        | Mus musculus       |                                        |
# | 19 | P09874    | P11103        | Mus musculus       |                                        |
# | 20 | Q5S007    | Q5S006        | Mus musculus       |                                        |
# | 21 | Q13489    | Q13489        | Mus musculus       | Mapped to human, no reliable mouse Uniprot found |
# | 22 | P21802    | E9Q7C7        | Mus musculus       |                                        |
# | 23 | P11362    | P16092        | Mus musculus       |                                        |
# | 24 | Q9NZQ7    | Q9EP73        | Mus musculus       |                                        |
# | 25 | O15379    | O88895        | Mus musculus       |                                        |
# | 26 | P30530    | Q00993        | Mus musculus       |                                        |
# | 27 | Q06187    | P35991        | Mus musculus       |                                        |
# | 28 | P04629    | Q3UFB7        | Mus musculus       |                                        |
# | 29 | P52789    | O08528        | Mus musculus       |                                        |

# %%
species2uniprot = {
    'Cricetulus griseus': {'P04035': 'P00347'},
    'Rattus norvegicus': {'Q05397': 'O35346'},
    'Mus musculus': {
        'Q9ULX9': 'Q01279',
        'P00533': 'O54791',
        'P11802': 'P30285',
        'Q00534': 'Q64261',
        'P36888': 'Q00342',
        'Q07817': 'Q64373',
        'Q14145': 'Q9Z2X8',
        'O60885': 'B2RSE4',
        'P24941': 'P97377',
        'A9UF07': 'P00520',
        'Q8TBX8': 'Q91XU3',
        'P06493': 'P11440',
        'P14625': 'P08113',
        'A9YLN7': 'Q3UMY5',
        'Q9UM73': 'Q3UMY5',
        'P53350': 'Q07832',
        'Q05397': 'P34152',
        'P09874': 'P11103',
        'Q5S007': 'Q5S006',
        'Q13489': 'Q13489', # Human, no reliable mouse Uniprot found
        'P21802': 'E9Q7C7',
        'P11362': 'P16092',
        'Q9NZQ7': 'Q9EP73',
        'O15379': 'O88895',
        'P30530': 'Q00993',
        'Q06187': 'P35991',
        'P04629': 'Q3UFB7',
        'P52789': 'O08528',
    }
}

uniprots = set()
for _, old2new in species2uniprot.items():
    for _, new in old2new.items():
        uniprots.add(new)

# -- Fetch and cache UniProt entries defined above --
for uniprot_id in tqdm(uniprots, desc='Fetching UniProt entries'):
    json_info = load_dict(os.path.join(uniprot_dir, f'{uniprot_id}.json'))
    if json_info:
        uniprot2infos[uniprot_id] = json_info
    else:
        info = extract_protein_info(uniprot_id)
        if info:
            uniprot2infos[uniprot_id] = info
            # Save each entry to a separate JSON file
            save_dict(info, os.path.join(uniprot_dir, f'{uniprot_id}.json'))

# -- Map the POI Uniprot ID based on the Cell Species --
def map_poi_uniprot_from_species(row):
    uniprot = row['Uniprot']
    poi_species = row['Cell Species']
    if pd.isna(uniprot) or pd.isna(poi_species):
        return uniprot
    if poi_species in species2uniprot and uniprot in species2uniprot[poi_species]:
        return species2uniprot[poi_species][uniprot]
    return uniprot

# -- Map the POI Sequence based on the Uniprot ID --
def map_poi_sequence_from_uniprot(row):
    uniprot = row['Uniprot']
    seq = row['POI Sequence']
    if pd.isna(uniprot):
        return seq
    if uniprot in uniprot2infos:
        return uniprot2infos[uniprot]['sequence']
    return seq

for df_name, df in df_dict.items():
    print('-' * 40)
    print(f"Dataframe: {df_name}")
    print('-' * 40)
    tqdm.pandas(desc='Mapping POI Uniprot ID from species')
    if 'Cell Species' not in df.columns:
        print(f"'Cell Species' column not found in {df_name}. Skipping...")
        continue
    df['Uniprot'] = df.progress_apply(map_poi_uniprot_from_species, axis=1)

    tqdm.pandas(desc='Mapping POI Sequence from Uniprot ID')
    df['POI Sequence'] = df.progress_apply(map_poi_sequence_from_uniprot, axis=1)

# %% [markdown]
# ## Get Assay Type

# %%
def get_assay_type(row, df_name):
    assay_col = f'Assay {df_name}'
    if pd.isnull(row[assay_col]):
        return None

    if 'HiBit' in row[assay_col]:
        return 'HiBit'
    elif 'ELISA' in row[assay_col]:
        return 'ELISA'
    elif 'Western Blot' in row[assay_col] or 'WB' in row[assay_col] or 'WB' in row['Target']:
        return 'Western Blot'
    
    if row[assay_col] == 'Degradation of CDK12 in HeLa cells after 24h treatment (cytoblot/westernblot)':
        if row[f'DC50 (nM) {df_name}'] == 57 or row[f'DC50 (nM) {df_name}'] == 74:
            return 'Cytoblot'
        else:
            return 'Western Blot'
    return None

for df_name, df in df_dict.items():
    # Create a new column named 'Assay Type' based on the 'Assay {df_name}' column
    df['Assay Type'] = df.apply(lambda row: get_assay_type(row, df_name), axis=1)
    print(f"Unique assay types in {df_name}: {df['Assay Type'].value_counts()}")
    print()

# %% [markdown]
# ## Save Dataframes

for df_name, df in df_dict.items():
    print(f"Processing dataframe: {df_name}")
    print('-' * 40)
    
    # Rename columns for TPD-DB compatibility
    old2new_cols = {
        'Smiles': 'SMILES',
        'E3 Ligase': 'Ligase_Name',
        'Uniprot': 'POI_UniProt',
        'POI Sequence': 'POI_Sequence',
        'E3 Ligase Uniprot': 'Ligase_UniProt',
        'E3 Ligase Sequence': 'Ligase_Sequence',
        f'Cell Type {df_name}': 'Cell_Line',
        f'Cell ID {df_name}': 'Cell_Line_ID',
        'Cell Species': 'Cell_Line_Species',
        'Article DOI': 'Reference',
        f'Assay {df_name}': 'Description',
        'Assay Type': 'Assay',
        f'Treatment Time (h) {df_name}': 'Assay_Time'
    }
    tmp = df.rename(columns=old2new_cols)

    # Assign a 'POI_Name' column based on the 'Target {df_name}' column if not NaN, else use 'Target'
    tmp['POI_Name'] = df[f'Target {df_name}'].combine_first(df['Target'])
    tmp['Modality'] = 'PROteolysis-TArgeting Chimera (PROTAC)'

    # Keep track of columns to save
    cols_to_save = list(old2new_cols.values())
    cols_to_save += ['POI_Name', 'Modality', 'Value', 'Value_Type', 'Value_Unit']
    
    if df_name == '(DC50/Dmax)':
        # Split the dataframe so that 'DC50 (nM) (DC50/Dmax)', 'Dmax (%) (DC50/Dmax)' are separate rows
        tmp = tmp.melt(
            id_vars=[col for col in tmp.columns if col not in ['DC50 (nM) Value (DC50/Dmax)', 'Dmax (%) Value (DC50/Dmax)']],
            value_vars=['DC50 (nM) Value (DC50/Dmax)', 'Dmax (%) Value (DC50/Dmax)'],
            var_name='Value_Type',
            value_name='Value',
        )
        tmp = tmp.dropna(subset=['Value'])

        # Rename ['DC50 (nM) Value (DC50/Dmax)', 'Dmax (%) (DC50/Dmax)'] in 'Value_Type' column to ['DC50', 'Dmax']
        tmp['Value_Type'] = tmp['Value_Type'].replace({
            'DC50 (nM) Value (DC50/Dmax)': 'DC50',
            'Dmax (%) Value (DC50/Dmax)': 'Dmax',
        })

        # Add a column 'Value_Unit' with values 'nM' for 'DC50' and '%' for 'Dmax'
        tmp['Value_Unit'] = tmp['Value_Type'].replace({
            'DC50': 'nM',
            'Dmax': '%',
        })
        
        # "Merge" the columns 'DC50 (nM) Operator (DC50/Dmax)', 'Dmax (%) Operator (DC50/Dmax)' into a single 'Value_Operator' column
        def get_value_operator(row):
            if row['Value_Type'] == 'DC50':
                return row['DC50 (nM) Operator (DC50/Dmax)']
            elif row['Value_Type'] == 'Dmax':
                return row['Dmax (%) Operator (DC50/Dmax)']
            return np.nan
        tmp['Value_Operator'] = tmp.apply(get_value_operator, axis=1)
        tmp = tmp.drop(columns=['DC50 (nM) Operator (DC50/Dmax)', 'Dmax (%) Operator (DC50/Dmax)'])
        cols_to_save += ['Value_Operator']

        print(f"Shape of the temporary dataframe for {df_name}: {tmp.shape}")
        print(tmp['Value_Operator'].dropna().value_counts())
        print(tmp[cols_to_save])
        
        for assay in tmp['Assay'].unique():
            print(assay)

        # Save DataFrame to data_curation_dir as CSV
        csv_path = Path(data_curation_dir) / f'protacdb_protac_dc50_dmax.csv'
        tmp[cols_to_save].drop_duplicates().to_csv(csv_path, index=False)
        
        break
        
    elif df_name == '(Percent degradation)':
        # Rename a few columns to match TPD-DB format
        tmp = tmp.rename(columns={
            'DC (nM) (Percent degradation)': 'Value_Concentration',
            'Percent degradation (%) (Percent degradation)': 'Value',
        })
        tmp['Value_Type'] = 'Degradation'
        tmp['Value_Unit'] = '%'
        tmp['Value_Concentration_Unit'] = 'nM'
        tmp['Assay_Time_Unit'] = 'h'
        cols_to_save += ['Value_Concentration', 'Value_Concentration_Unit', 'Assay_Time', 'Assay_Time_Unit']

        print(f"Shape of the temporary dataframe for {df_name}: {tmp.shape}")
        print(tmp[cols_to_save])
        # Save DataFrame to data_curation_dir as CSV
        csv_path = Path(data_curation_dir) / f'protacdb_protac_percent_degradation.csv'
        tmp[cols_to_save].drop_duplicates().to_csv(csv_path, index=False)

    elif df_name == '(Cellular activities, IC50)':
        print(f'Skipping dataframe {df_name} for now...')
    elif df_name == '(Protac to Target, IC50)':
        print(f'Skipping dataframe {df_name} for now...')
