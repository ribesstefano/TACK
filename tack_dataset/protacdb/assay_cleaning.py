import re
from typing import List, Union, Dict
import pandas as pd


def parse_single_value(value_str: str) -> Union[Dict[str, Union[float, str, None]], None]:
    """ Parse a single numeric value with optional operator, error bar, and unit.
    
    Args:
        value_str (str): The input string to parse, e.g., "0.785±0.03μM", ">10μM", "~58.7%", "1.2e3 nM", "-36%", "88.9*", "0.022 (1%)".
        
    Returns:
        dict: A dictionary with keys 'mean', 'error', 'unit', and 'operator'. 
              'mean' is the main numeric value (float), 'error' is the error bar if present (float or None), 
              'unit' is the unit string if present (str or None), and 'operator' is the operator if present (str or None).
              Returns None if the input string cannot be parsed.
    """
    
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
