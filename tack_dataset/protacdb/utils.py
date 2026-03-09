import re
import os
import json
import pickle
import logging
from pathlib import Path
from itertools import zip_longest
from typing import Union, List, Literal
import time
import requests

import pandas as pd


def save_dict(
    d: dict,
    filepath: Union[str, Path],
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
    filepath = Path(filepath)
    if filepath.suffix == '.json':
        with open(filepath, 'w') as f:
            json.dump(d, f, indent=indent)
    elif filepath.suffix == '.pkl':
        with open(filepath, 'wb') as f:
            pickle.dump(d, f)
    else:
        raise ValueError(f'Unsupported file extension: {filepath}. Use .json or .pkl.')

def load_dict(filepath: Union[Path, str]) -> Union[dict, List[dict]]:
    """
    Load a dictionary from a file in JSON or pickle format.
    
    Args:
        filepath (str): Path to the file from which the dictionary will be loaded.
        
    Returns:
        dict: The loaded dictionary. If the file does not exist, returns an empty dictionary.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return {}
    if filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            return json.load(f)
    elif filepath.suffix == '.pkl':
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f'Unsupported file extension: {filepath}. Use .json or .pkl.')

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