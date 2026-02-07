"""
Data Curation Pipeline for PROTAC-Pedia Dataset.

This script curates the PROTAC-Pedia dataset to match the standardized format
used by TPD-DB and PROTAC-DB. The output includes cleaned SMILES, validated
cell lines with Cellosaurus IDs, POI sequences from UniProt, parsed assay
values, and standardized column names.

Output format includes key columns:
    - SMILES: Canonical SMILES string
    - POI_Name: Protein of Interest name
    - POI_UniProt: POI UniProt ID
    - POI_Sequence: POI amino acid sequence
    - Ligase_Name: E3 ligase name
    - Ligase_UniProt: E3 ligase UniProt ID
    - Ligase_Sequence: E3 ligase amino acid sequence
    - Cell_Line: Standardized cell line name
    - Cell_Line_ID: Cellosaurus accession ID (CVCL_####)
    - Cell_Line_Species: Species of the cell line
    - Value: Measured value (DC50 or Dmax)
    - Value_Type: Type of measurement
    - Value_Unit: Unit of measurement
    - Value_Symbol: Symbol for approximate values (<, >, ~)
    - Assay: Assay description
    - Assay_Time: Treatment time in hours
    - Reference: PubMed ID or source

Usage:
    python curate_protacpedia.py --output_dir ../data/curation
"""

import argparse
import json
import logging
import os
import pickle
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from rdkit import Chem
from rdkit import RDLogger
from tqdm.auto import tqdm
from Bio import Entrez

from tack_dataset.curate_protacpedia_manual_curation_utils import get_manual_curation_overrides
from tack_dataset.logging_utils import setup_logging

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Configure logging
log_file = setup_logging(
    log_dir=Path('logs'),
    log_base_name='protacpedia_curation',
    verbose=1, # Enable INFO level logging
)
logger = logging.getLogger(__name__)

logger.info(f"Log file: {log_file}")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    """Configuration for the PROTAC-Pedia curation pipeline."""
    
    # Directories
    output_dir: str = "data/curation"
    cache_dir: str = "data/curation"
    log_dir: str = "logs"
    
    # API settings
    api_delay: float = 0.5
    force_refetch: bool = False
    
    # Input file (original PROTACpedia CSV)
    input_file: str = 'protacpedia_protac_dc50_dmax.csv'
    
    # Output file
    output_file: str = 'protacpedia_protac_dc50_dmax_cleaned.csv'
    
    # E3 Ligase to UniProt mapping
    e3_ligase_to_uniprot: Dict[str, str] = None
    
    def __post_init__(self):
        if self.e3_ligase_to_uniprot is None:
            self.e3_ligase_to_uniprot = {
                'VHL': 'P40337',
                'CRBN': 'Q96SW2',
                'DCAF1': 'Q9Y4B6',
                'DCAF11': 'Q8TEB1',
                'DCAF15': 'Q66K64',
                'DCAF16': 'Q9NXF7',
                'MDM2': 'Q00987',
                'XIAP': 'P98170',
                'IAP': 'P98170',
                'cIAP1': 'Q13490',
                'AhR': 'P35869',
                'RNF4': 'P78317',
                'RNF114': 'Q9Y508',
                'FEM1B': 'Q9UK73',
                'UBR1': 'Q8IWV7',
                'UBR box': 'G3V2G3',
                'KLHL20': 'Q9Y2M5',
                'KLHDC2': 'Q9Y2U9',
                'FBXO22': 'Q8NEZ5',
                'KEAP1': 'Q14145',
            }


# =============================================================================
# Utility Functions
# =============================================================================

def get_doi_from_pubmed(pmid: int, email: str) -> str:
    """ Get DOI from PubMed using the Entrez API.
    
    Args:
        pmid (int): PubMed ID to fetch the DOI for.
        email (str): Email address to use for Entrez API requests.
        
    Returns:
        str: DOI if available, otherwise a link to the PubMed article.
    """
    Entrez.email = email
    handle = Entrez.efetch(db="pubmed", id=pmid, retmode="xml")
    records = Entrez.read(handle)
    handle.close()

    # Extracting DOI
    try:
        article = records['PubmedArticle'][0]
        for el in article['MedlineCitation']['Article']['ELocationID']:
            if el.attributes['EIdType'] == 'doi':
                return el
    except IndexError:
        return f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/'


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """
    Canonicalize a SMILES string using RDKit.
    
    Args:
        smiles: Input SMILES string
        
    Returns:
        Canonical SMILES or None if invalid
    """
    if pd.isna(smiles) or not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        pass
    return None


def parse_value_symbol_unit(s):
    """
    Parses a string like '53 nM', '< 1 uM', '~ 3 nM', '> 10 uM' and returns (value, symbol, unit).
    - value: float
    - symbol: '', '<', '>', '~'
    - unit: 'nM' or 'μM' (unicode mu)
    """
    s = s.strip()
    # Regex: optional symbol, value (float), unit (nM/uM)
    match = re.match(r'^(?P<symbol>[<>=~]*)\s*(?P<value>[0-9.]+)\s*(?P<unit>nM|uM)$', s)
    if not match:
        raise ValueError(f"Could not parse: {s}")
    symbol = match.group('symbol')
    value = float(match.group('value'))
    unit = match.group('unit')
    if unit == 'uM':
        unit = 'μM'  # Unicode mu
    return value, symbol, unit


def parse_percent_value_symbol(s):
    """
    Parses strings like '100 %', '~ 90 %', '> 85 %', '< 70 %', '0.03 %', '0' and returns (value, symbol, unit).
    - value: float
    - symbol: '', '<', '>', '~'
    - unit: '%' (always percent)
    """
    s = s.strip()
    # Handle missing percent sign (e.g., '0', '55')
    if not s.endswith('%'):
        s = s + ' %'
    match = re.match(r'^(?P<symbol>[<>=~]*)\s*(?P<value>[0-9.]+)\s*%$', s)
    if not match:
        raise ValueError(f"Could not parse: {s}")
    symbol = match.group('symbol')
    value = float(match.group('value'))
    unit = '%'
    return value, symbol, unit


# =============================================================================
# Cellosaurus API Functions
# =============================================================================

CELLOSAURUS_API = "https://api.cellosaurus.org/cell-line/{}"
CELLOSAURUS_SEARCH_API = "https://api.cellosaurus.org/search/cell-line"


@lru_cache(maxsize=2048)
def fetch_cellosaurus_by_name(cell_line_name: str) -> Optional[dict]:
    """
    Fetch cell line metadata from Cellosaurus API by name.
    
    Args:
        cell_line_name: Cell line name to search for
        
    Returns:
        Dictionary with cell line metadata or None
    """
    if not cell_line_name:
        return None
    
    try:
        # Use search endpoint (the direct endpoint requires accession IDs, not names)
        params = {'q': cell_line_name}
        resp = requests.get(CELLOSAURUS_SEARCH_API, params=params, timeout=10)
        resp.raise_for_status()
        
        data = resp.json()
        cell_lines = data.get('Cellosaurus', {}).get('cell-line-list', [])
        
        if cell_lines:
            # Return first match (usually the best match)
            return cell_lines[0]
    except Exception as e:
        logger.debug(f"Failed to fetch Cellosaurus data for {cell_line_name}: {e}")
    
    return None


def clean_cell_line_name(cell_line: Optional[str]) -> Optional[str]:
    """
    Standardize cell line name for Cellosaurus lookup.
    
    Args:
        cell_line: Raw cell line string
        
    Returns:
        Cleaned cell line name
    """
    if pd.isna(cell_line) or not cell_line:
        return None
    
    s = str(cell_line).strip()
    
    # If multiple cell lines are listed (comma or semicolon separated), take only the first one
    if ',' in s or ';' in s:
        s = re.split(r'[,;]', s)[0].strip()
    
    # Remove parenthetical annotations
    s = re.sub(r'\s*\(.*?\)', '', s).strip()
    
    # Remove common suffixes
    s = re.sub(r'\s+(cells?|cell\s+line|monocytes?|cancer|breast|leukemia|carcinoma)$', '', s, flags=re.IGNORECASE).strip()
    
    # Remove common prefixes
    s = re.sub(r'^(human|mouse|rat)\s+', '', s, flags=re.IGNORECASE).strip()
    
    # Common normalizations
    replacements = {
        'MOLT4': 'MOLT-4',
        'Hela': 'HeLa',
        'hela': 'HeLa',
        'HT1080': 'HT-1080',
        'HT 1080': 'HT-1080',
        'THP': 'THP-1',  # THP is usually THP-1
        'Hs578t': 'Hs 578T',  # Standard Cellosaurus format
        'Panc0213': 'Panc 02.13',
        'Panc02.13': 'Panc 02.13',
        'NAMALWA': 'Namalwa',  # Case normalization
    }
    
    for old, new in replacements.items():
        if s.lower() == old.lower():
            return new
    
    return s if s else None


def get_cellosaurus_accession(cell_line: str) -> Optional[str]:
    """
    Get Cellosaurus accession ID (CVCL_####) for a cell line.
    
    Args:
        cell_line: Cell line name
        
    Returns:
        Cellosaurus accession ID or None
    """
    if not cell_line:
        return None
    
    metadata = fetch_cellosaurus_by_name(cell_line)
    if metadata:
        # New API format: accession-list with type "primary"
        accession_list = metadata.get('accession-list', [])
        for acc_item in accession_list:
            if isinstance(acc_item, dict) and acc_item.get('type') == 'primary':
                value = acc_item.get('value', '')
                if value.startswith('CVCL_'):
                    return value
        
        # Fallback to direct accession field
        for key in ['accession', 'accession-id', 'id', 'ac']:
            if key in metadata:
                acc = metadata[key]
                if isinstance(acc, str) and acc.startswith('CVCL_'):
                    return acc
                if isinstance(acc, list) and acc and acc[0].startswith('CVCL_'):
                    return acc[0]
    
    return None


def get_cellosaurus_species(cell_line: str) -> Optional[str]:
    """
    Get species information from Cellosaurus.
    
    Args:
        cell_line: Cell line name
        
    Returns:
        Species name or None
    """
    if not cell_line:
        return None
    
    metadata = fetch_cellosaurus_by_name(cell_line)
    if metadata:
        # New API format: species-list with label field
        species_list = metadata.get('species-list', [])
        if species_list:
            if isinstance(species_list[0], dict):
                # Extract from label (e.g., "Homo sapiens (Human)")
                label = species_list[0].get('label', '')
                if label:
                    # Extract just species name before parentheses
                    species = label.split('(')[0].strip()
                    return species
                return species_list[0].get('value') or species_list[0].get('name')
            return str(species_list[0])
        
        # Fallback to other possible keys
        for key in ['species', 'organism']:
            if key in metadata:
                species_data = metadata[key]
                if isinstance(species_data, str):
                    return species_data
                if isinstance(species_data, list) and species_data:
                    if isinstance(species_data[0], dict):
                        return species_data[0].get('value') or species_data[0].get('name')
                    return str(species_data[0])
    
    # Fallback heuristics based on cell line name
    s = cell_line.lower()
    if any(k in s for k in ['mouse', 'murine', '3t3', 'baf3', 'ba/f3']):
        return 'Mus musculus'
    if 'rat' in s:
        return 'Rattus norvegicus'
    
    # Common human cell lines
    human_hints = ['hela', 'a549', 'hct', 'ht', 'k562', 'molt', 'calu', 
                   'h358', 'h1975', 'mia', 'panc', 'pc3', 'lncap']
    if any(k in s for k in human_hints):
        return 'Homo sapiens'
    
    return 'Unknown'


# =============================================================================
# UniProt API Functions
# =============================================================================

class UniProtFetcher:
    """Handles UniProt API requests with caching."""
    
    def __init__(self, cache_dir: str, delay: float = 0.5):
        self.cache_dir = Path(cache_dir)
        self.delay = delay
        self.cache_file = self.cache_dir / 'uniprot_cache.json'
        self.cache = self._load_cache()
    
    def _load_cache(self) -> dict:
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}
    
    def save_cache(self):
        """Save cache to disk."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
    
    def fetch_entry(self, uniprot_id: str) -> Optional[dict]:
        """
        Fetch a UniProt entry by ID.
        
        Args:
            uniprot_id: UniProt accession ID
            
        Returns:
            UniProt entry as dictionary or None
        """
        if uniprot_id in self.cache:
            return self.cache[uniprot_id]
        
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            time.sleep(self.delay)
            entry = response.json()
            self.cache[uniprot_id] = entry
            return entry
        except Exception as e:
            logger.warning(f"Failed to fetch UniProt entry {uniprot_id}: {e}")
            return None
    
    def get_sequence(self, uniprot_id: str) -> Optional[str]:
        """
        Get the canonical sequence for a UniProt ID.
        
        Args:
            uniprot_id: UniProt accession ID
            
        Returns:
            Amino acid sequence or None
        """
        entry = self.fetch_entry(uniprot_id)
        if entry and 'sequence' in entry:
            return entry['sequence'].get('value')
        return None
    
    def get_gene_names(self, uniprot_id: str) -> List[str]:
        """
        Get gene names for a UniProt ID.
        
        Args:
            uniprot_id: UniProt accession ID
            
        Returns:
            List of gene names
        """
        entry = self.fetch_entry(uniprot_id)
        if not entry:
            return []
        
        names = []
        for gene in entry.get('genes', []):
            if 'geneName' in gene:
                names.append(gene['geneName'].get('value', ''))
            for syn in gene.get('synonyms', []):
                names.append(syn.get('value', ''))
        
        return [n for n in names if n]


# =============================================================================
# Assay Time Extraction
# =============================================================================

def extract_assay_time(row: pd.Series) -> Optional[int]:
    """
    Extract assay time in hours from Comments or Time column.
    
    Args:
        row: DataFrame row
        
    Returns:
        Assay time in hours or None
    """
    # Try Time column first
    if 'Time' in row and pd.notna(row['Time']):
        time_str = str(row['Time']).strip()
        # Match patterns like "48", "24 h", "16 hours", etc.
        match = re.search(r'(\d+)\s*(?:h|hr|hrs|hour|hours)?', time_str, re.I)
        if match:
            return int(match.group(1))
    
    # Try Comments column
    if 'Comments' in row and pd.notna(row['Comments']):
        comments = str(row['Comments'])
        # Look for time mentions in comments
        match = re.search(r'(\d+)\s*(?:h|hr|hrs|hour|hours)', comments, re.I)
        if match:
            return int(match.group(1))
    
    return None


def parse_value_with_symbol(value_str: str) -> Tuple[Optional[float], str]:
    """
    Parse value with optional symbol.
    
    Args:
        value_str: String like "100", ">50", "~90", "< 10"
        
    Returns:
        (value, symbol) tuple
    """
    value_str = value_str.strip()
    symbol = ''
    
    # Extract symbol
    match = re.match(r'^([<>~≤≥]+)\s*([\d.]+)', value_str)
    if match:
        symbol = match.group(1)
        value_str = match.group(2)
    
    # Normalize symbols
    symbol = symbol.replace('≤', '<=').replace('≥', '>=')
    if symbol in ['<=', '<']:
        symbol = '<'
    elif symbol in ['>=', '>']:
        symbol = '>'
    elif symbol in ['~', '≈']:
        symbol = '~'
    
    try:
        value = float(value_str)
    except ValueError:
        value = None
    
    return value, symbol


def parse_single_value(value_str: str) -> Optional[Dict[str, Any]]:
    """
    Parse a single numeric value with optional operator, error bar, and unit.
    
    Examples:
        "0.785±0.03μM" -> {'mean': 0.785, 'error': 0.03, 'unit': 'μM', 'operator': None}
        ">10μM" -> {'mean': 10.0, 'error': None, 'unit': 'μM', 'operator': '>'}
        "~58.7%" -> {'mean': 58.7, 'error': None, 'unit': '%', 'operator': '~'}
    
    Args:
        value_str: String containing a value
        
    Returns:
        Dict with mean, error, unit, operator or None
    """
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
        operator = match.group(1)
        # Normalize operators
        operator = operator.replace('≤', '<=').replace('≥', '>=')
        if operator in ['<=', '<']:
            operator = '<'
        elif operator in ['>=', '>']:
            operator = '>'
        return {
            'mean': float(match.group(2)),
            'error': None,
            'unit': match.group(3),
            'operator': operator
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
        numeric_part = value_str
        if match.group(2):
            numeric_part = value_str[:value_str.rfind(match.group(2))].strip()
        return {
            'mean': float(numeric_part),
            'error': None,
            'unit': match.group(2),
            'operator': None
        }
    
    # Pattern 5: Standard numeric (2.63nM, 0.701μM, 67%, 0.001µM)
    match = re.match(r'^(-?\d+\.?\d*|\d*\.\d+)\s*([a-zA-Zμµ%/]+)?$', value_str)
    if match:
        return {
            'mean': float(match.group(1)),
            'error': None,
            'unit': match.group(2),
            'operator': None
        }
    
    return None


def parse_range_value(value_str: str) -> Optional[Dict[str, Any]]:
    """
    Parse range values like '10nM≤x<100nM', '0.01-0.1μM', or '150-200'.
    
    Args:
        value_str: String containing a range
        
    Returns:
        Dict with min, max, unit or None
    """
    value_str = value_str.strip()
    
    # Pattern 1: Hyphen ranges with unit (0.01-0.1μM, 100nM-300nM, 1-5μM)
    match = re.match(r'^(\d+\.?\d*|\d*\.\d+)\s*([a-zA-Zμµ%]+)?\s*-\s*(\d+\.?\d*|\d*\.\d+)\s*([a-zA-Zμµ%]+)?$', value_str)
    if match:
        return {
            'min': float(match.group(1)),
            'max': float(match.group(3)),
            'unit': match.group(2) or match.group(4)
        }
    
    # Pattern 2: Inequality ranges with x (10nM≤x<100nM, 1.0μM≤x<3.0μM)
    match = re.search(
        r'(\d+\.?\d*|\d*\.\d+)\s*([a-zA-Zμµ%]+)?\s*[<≤]\s*x\s*[<≤]\s*(\d+\.?\d*|\d*\.\d+)\s*([a-zA-Zμµ%]+)?',
        value_str
    )
    if match:
        return {
            'min': float(match.group(1)),
            'max': float(match.group(3)),
            'unit': match.group(2) or match.group(4)
        }
    
    # Pattern 3: Operator with inequality (≥150nM, <100, >10μM, <=100nM, >=10)
    match = re.match(r'^([>≥<≤]+|<=|>=)\s*(\d+\.?\d*|\d*\.\d+)\s*([a-zA-Zμµ%]+)?$', value_str)
    if match:
        operator = match.group(1)
        val = float(match.group(2))
        unit = match.group(3)
        
        operator = operator.replace('≤', '<=').replace('≥', '>=')
        if operator in ['>', '≥', '>=']:
            return {'min': val, 'max': None, 'unit': unit}
        elif operator in ['<', '≤', '<=']:
            return {'min': None, 'max': val, 'unit': unit}
    
    return None


def normalize_units(unit: Optional[str]) -> Optional[str]:
    """Normalize units to standard forms."""
    if not unit:
        return None
    
    unit_mapping = {
        'nm': 'nM',
        'nM': 'nM',
        'nmol/L': 'nM',
        'uM': 'μM',
        'µM': 'μM',
        'um': 'μM',
        'μM': 'μM',
        'M': 'M',
        '%': '%',
    }
    
    return unit_mapping.get(unit, unit)


def parse_cell_line_specific_values(comments: str) -> List[Dict[str, Any]]:
    """
    Parse comments to extract cell-line-specific DC50/Dmax values.
    
    Examples:
        "DC50 is 0.86nM in LNCaP, 0.76 in VCaP and 10.4 nM at 1uM in 22Rv1"
        "Dmax for KYSE520 cell: >95%; Dmax for MV4;11 cell: >90%"
        "DC50: 0.25~0.76uM; Dmax: ~75%-90%"
    
    Args:
        comments: Comment text
        
    Returns:
        List of dicts with cell_line, value_type, value, unit, symbol
    """
    if pd.isna(comments):
        return []
    
    results = []
    text = str(comments)
    
    # Pattern 1: "DC50 is X nM in CellLine, Y nM in CellLine2"
    pattern1 = r'DC50\s+is\s+([<>~≤≥]?\s*[\d.]+)\s*(nM|μM|uM|pM|M)\s+in\s+([A-Za-z0-9\-]+)'
    for match in re.finditer(pattern1, text, re.IGNORECASE):
        value_str, unit, cell_line = match.groups()
        value, symbol = parse_value_with_symbol(value_str.strip())
        if unit.lower() == 'um':
            unit = 'μM'
        results.append({
            'cell_line': cell_line.strip(),
            'value_type': 'DC50',
            'value': value,
            'unit': unit,
            'symbol': symbol
        })
    
    # Pattern 2: "DC50 for CellLine: X nM" or "DC50 for CellLine cell: X nM"
    pattern2 = r'DC50\s+(?:for|in)\s+([A-Za-z0-9\-\s]+?)(?:\s+cells?)?\s*:\s*([<>~≤≥]?\s*[\d.]+)\s*(nM|μM|uM|pM|M)'
    for match in re.finditer(pattern2, text, re.IGNORECASE):
        cell_line, value_str, unit = match.groups()
        # Clean up cell line name
        cell_line = cell_line.strip()
        # Skip if it looks like a protein name (contains lowercase or slash)
        if '/' in cell_line or any(c.islower() for c in cell_line.replace(' ', '')):
            continue
        value, symbol = parse_value_with_symbol(value_str.strip())
        if unit.lower() == 'um':
            unit = 'μM'
        results.append({
            'cell_line': cell_line,
            'value_type': 'DC50',
            'value': value,
            'unit': unit,
            'symbol': symbol
        })
    
    # Pattern 3: "Dmax for CellLine: X%" or "Dmax for CellLine cells: X%"
    pattern3 = r'Dmax\s+(?:for|in)\s+([A-Za-z0-9\-\s]+?)(?:\s+cells?)?\s*:\s*([<>~≤≥]?\s*[\d.]+)\s*%'
    for match in re.finditer(pattern3, text, re.IGNORECASE):
        cell_line, value_str = match.groups()
        cell_line = cell_line.strip()
        # Skip protein names
        if '/' in cell_line or any(c.islower() for c in cell_line.replace(' ', '')):
            continue
        value, symbol = parse_value_with_symbol(value_str.strip())
        results.append({
            'cell_line': cell_line,
            'value_type': 'Dmax',
            'value': value,
            'unit': '%',
            'symbol': symbol
        })
    
    # Pattern 4: "Dmax in CellLine/CellLine2: X%±Y% and Z%±W%, respectively"
    pattern4 = r'Dmax\s+in\s+([A-Za-z0-9\-]+)/([A-Za-z0-9\-]+)\s*:\s*([<>~≤≥]?\s*[\d.]+)%?[^,]*and\s+([<>~≤≥]?\s*[\d.]+)%'
    for match in re.finditer(pattern4, text, re.IGNORECASE):
        cell1, cell2, val1_str, val2_str = match.groups()
        value1, symbol1 = parse_value_with_symbol(val1_str.strip())
        value2, symbol2 = parse_value_with_symbol(val2_str.strip())
        results.append({
            'cell_line': cell1.strip(),
            'value_type': 'Dmax',
            'value': value1,
            'unit': '%',
            'symbol': symbol1
        })
        results.append({
            'cell_line': cell2.strip(),
            'value_type': 'Dmax',
            'value': value2,
            'unit': '%',
            'symbol': symbol2
        })
    
    return results


def extract_additional_metrics_from_comments(comments: str) -> Dict[str, Any]:
    """
    Extract additional metrics from comments like IC50, EC50, pDC50, pEC50, etc.
    
    Examples:
        "IC50 of ligand is 51.0 nM (WT BTK), 30.7 (C481S)"
        "EC50 of PROTAC is 28nM in MV-4-11, 68nM in NCI-H1568"
        "pDC50 for Brd4 short/Brd4 long/Brd3/Brd2: 7.0/7.0/6.5/6.2"
    
    Args:
        comments: Comment text
        
    Returns:
        Dict with extracted metrics
    """
    if pd.isna(comments):
        return {}
    
    metrics = {}
    text = str(comments)
    
    # Extract Ligand IC50
    match = re.search(r'IC50\s+of\s+(?:the\s+)?ligand\s+is\s+(?:between\s+)?([<>~]?\s*[\d.]+)\s*-?\s*[\d.]*\s*(nM|μM|uM|pM)', text, re.IGNORECASE)
    if match:
        metrics['Ligand_IC50'] = match.group(1).strip()
        metrics['Ligand_IC50_Unit'] = match.group(2)
    
    # Extract Ligand EC50
    match = re.search(r'EC50\s+of\s+(?:the\s+)?ligand\s+is\s+(?:between\s+)?([<>~]?\s*[\d.]+)\s*-?\s*[\d.]*\s*(nM|μM|uM)', text, re.IGNORECASE)
    if match:
        metrics['Ligand_EC50'] = match.group(1).strip()
        metrics['Ligand_EC50_Unit'] = match.group(2)
    
    # Extract PROTAC IC50
    match = re.search(r'IC50\s+of\s+(?:the\s+)?PROTAC\s+is\s+(?:between\s+)?([<>~]?\s*[\d.]+)\s*-?\s*[\d.]*\s*(nM|μM|uM)', text, re.IGNORECASE)
    if match:
        metrics['PROTAC_IC50'] = match.group(1).strip()
        metrics['PROTAC_IC50_Unit'] = match.group(2)
    
    # Extract PROTAC EC50
    match = re.search(r'EC50\s+of\s+(?:the\s+)?PROTAC\s+is\s+(?:between\s+)?([<>~]?\s*[\d.]+)\s*-?\s*[\d.]*\s*(nM|μM|uM)', text, re.IGNORECASE)
    if match:
        metrics['PROTAC_EC50'] = match.group(1).strip()
        metrics['PROTAC_EC50_Unit'] = match.group(2)
    
    # Extract assay time from specific patterns
    match = re.search(r'(\d+)\s*h(?:our)?(?:s)?\s+(?:post-treatment|of\s+treatment)', text, re.IGNORECASE)
    if match:
        metrics['Treatment_Time'] = int(match.group(1))
    
    # Check for structural information
    if re.search(r'(?:crystal|ternary)\s+(?:complex\s+)?structure', text, re.IGNORECASE):
        metrics['Has_Structure'] = True
        # Try to extract PDB ID
        pdb_match = re.search(r'\b([0-9][A-Z0-9]{3})\b', text)
        if pdb_match:
            metrics['PDB_ID'] = pdb_match.group(1)
    
    # Check for selectivity information
    if re.search(r'selectiv(?:e|ity)', text, re.IGNORECASE):
        metrics['Selectivity_Info'] = True
    
    # Check for covalent binding
    if re.search(r'covalent', text, re.IGNORECASE):
        metrics['Covalent'] = True
    
    return metrics


def parse_multi_protein_degradation(comment: str, row: pd.Series) -> List[Dict[str, Any]]:
    """
    Parse multi-protein degradation data.
    
    Example: "pDC50 for Brd4 short/Brd4 long/Brd3/Brd2: 7.0/7.0/6.5/6.2"
    
    Args:
        comment: Comment text
        row: Original row data
        
    Returns:
        List of row dicts for each protein
    """
    results = []
    
    # Pattern for pDC50 with multiple proteins
    match = re.search(
        r'pDC50\s+for\s+([^:]+):\s+([\d./]+)',
        comment,
        re.IGNORECASE
    )
    
    if not match:
        return results
    
    proteins_str = match.group(1)
    values_str = match.group(2)
    
    # Split proteins and values
    proteins = [p.strip() for p in proteins_str.split('/')]
    values = [v.strip() for v in values_str.split('/')]
    
    if len(proteins) != len(values):
        logger.warning(f"Mismatch in proteins ({len(proteins)}) and values ({len(values)})")
        return results
    
    # Also check for Dmax values
    dmax_match = re.search(
        r'Dmax\s+for\s+([^:]+):\s+([\d./%]+)',
        comment,
        re.IGNORECASE
    )
    
    dmax_values = []
    if dmax_match:
        dmax_str = dmax_match.group(2)
        dmax_values = [v.strip().replace('%', '') for v in dmax_str.split('/')]
    
    # Create a row for each protein
    for i, (protein, pdc50_val) in enumerate(zip(proteins, values)):
        # Skip if value is NA or invalid
        if pdc50_val.upper() in ['NA', 'N.A.', 'ND']:
            continue
        
        try:
            pdc50_float = float(pdc50_val)
        except ValueError:
            continue
        
        result_row = {
            'POI_Name': protein,
            'Value': pdc50_float,
            'Value_Type': 'pDC50',
            'Value_Unit': 'log(M)',
            'Value_Symbol': '',
        }
        
        # Add Dmax if available
        if i < len(dmax_values):
            try:
                dmax_float = float(dmax_values[i])
                # Create a separate Dmax row
                dmax_row = result_row.copy()
                dmax_row.update({
                    'Value': dmax_float,
                    'Value_Type': 'Dmax',
                    'Value_Unit': '%',
                })
                results.append(dmax_row)
            except ValueError:
                pass
        
        results.append(result_row)
    
    return results


def parse_dual_cell_line_dmax(comment: str, row: pd.Series) -> List[Dict[str, Any]]:
    """
    Parse Dmax values for two cell lines with error bars.
    
    Example: "Dmax in BBL358/T47D: 93%±5% and 87%±3%, respectively"
    
    Args:
        comment: Comment text
        row: Original row data
        
    Returns:
        List of row dicts for each cell line
    """
    results = []
    
    match = re.search(
        r'Dmax\s+in\s+([A-Za-z0-9\-]+)/([A-Za-z0-9\-]+):\s*([\d.]+)%±([\d.]+)%\s+and\s+([\d.]+)%±([\d.]+)%',
        comment,
        re.IGNORECASE
    )
    
    if not match:
        return results
    
    cell1, cell2, val1, err1, val2, err2 = match.groups()
    
    results.append({
        'Cell_Line': cell1,
        'Value': float(val1),
        'Value_Type': 'Dmax',
        'Value_Unit': '%',
        'Value_Error': float(err1),
        'Value_Symbol': '',
    })
    
    results.append({
        'Cell_Line': cell2,
        'Value': float(val2),
        'Value_Type': 'Dmax',
        'Value_Unit': '%',
        'Value_Error': float(err2),
        'Value_Symbol': '',
    })
    
    return results


def parse_concentration_dependent_dmax(comment: str, row: pd.Series) -> List[Dict[str, Any]]:
    """
    Parse concentration-dependent Dmax measurements.
    
    Example: "DCmax was measured in 100nM (at 10 nM 99.6 %)"
    
    Args:
        comment: Comment text
        row: Original row data
        
    Returns:
        List of row dicts with concentration information
    """
    results = []
    
    # Pattern: "at X nM Y %"
    matches = re.finditer(
        r'at\s+([\d.]+)\s*([nμ]M)\s+([\d.]+)\s*%',
        comment,
        re.IGNORECASE
    )
    
    for match in matches:
        conc, unit, dmax_val = match.groups()
        results.append({
            'Value': float(dmax_val),
            'Value_Type': 'Dmax',
            'Value_Unit': '%',
            'Value_Concentration': float(conc),
            'Value_Concentration_Unit': normalize_units(unit),
            'Value_Symbol': '',
        })
    
    return results

# =============================================================================
# Reference Processing
# =============================================================================

def process_reference(row: pd.Series) -> str:
    """
    Process reference information to create a standardized reference string.
    
    Args:
        row: DataFrame row
        
    Returns:
        Reference string (PubMed ID or other identifier)
    """
    if 'Pubmed' in row and pd.notna(row['Pubmed']):
        return str(row['Pubmed'])
    
    if 'PATENT' in row and pd.notna(row['PATENT']):
        return f"{row['PATENT']} (patent)"
    
    if 'PROTACDB ID' in row and pd.notna(row['PROTACDB ID']):
        return f"PROTACDB-{row['PROTACDB ID']}"
    
    return "Unknown"


# =============================================================================
# Main Curation Pipeline
# =============================================================================

class ProtacPediaCurator:
    """Main class for curating PROTAC-Pedia dataset."""
    
    def __init__(self, config: Config):
        self.config = config
        self.uniprot_fetcher = UniProtFetcher(config.cache_dir, config.api_delay)
        
        # Create output directories
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.cache_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    
    def load_data(self) -> pd.DataFrame:
        """Load the partially processed PROTAC-Pedia data."""
        filepath = Path(self.config.input_file)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Input file not found: {filepath}")
        
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} rows from {filepath}")
        return df
    
    def preprocess_and_rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess column names to standardize them."""
        df = df.dropna(subset=['Dc50', 'Dmax'], how='all')

        # Rename columns to match other datasets
        df = df.rename(columns={
            'PROTAC SMILES': 'SMILES',
            'Target': 'POI_UniProt',
            'E3 Ligase': 'Ligase_Name',
            'Cells': 'Cell_Line',
            'Dc50': 'DC50',
        })

        df['SMILES'] = df['SMILES'].apply(canonicalize_smiles)

        # In 'Ligase_Name', rename 'Cereblon' to 'CRBN', 'Mdm2' to 'MDM2', 'Iap' to 'IAP', 'Ubr1' to 'UBR box'
        df['Ligase_Name'] = df['Ligase_Name'].replace({
            'Cereblon': 'CRBN',
            'Mdm2': 'MDM2',
            'Iap': 'IAP',
            'Ubr1': 'UBR1'
        })

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
        df['Ligase_UniProt'] = df['Ligase_Name'].map(e3ligase2uniprot)

        # Parse DC50 values
        dc50_values = []
        dc50_symbols = []
        dc50_units = []
        for s in df['DC50']:
            if pd.isna(s):
                dc50_values.append(pd.NA)
                dc50_symbols.append(pd.NA)
                dc50_units.append(pd.NA)
            else:
                value, symbol, unit = parse_value_symbol_unit(s)
                dc50_values.append(value)
                dc50_symbols.append(symbol)
                dc50_units.append(unit)
        df['DC50_Value'] = dc50_values
        df['DC50_Value_Symbol'] = dc50_symbols
        df['DC50_Value_Unit'] = dc50_units

        # Parse Dmax values
        dmax_values = []
        dmax_symbols = []
        dmax_units = []
        for s in df['Dmax']:
            if pd.isna(s):
                dmax_values.append(pd.NA)
                dmax_symbols.append(pd.NA)
                dmax_units.append(pd.NA)
            else:
                value, symbol, unit = parse_percent_value_symbol(s)
                dmax_values.append(value)
                dmax_symbols.append(symbol)
                dmax_units.append(unit)
        df['Dmax_Value'] = dmax_values
        df['Dmax_Value_Symbol'] = dmax_symbols
        df['Dmax_Value_Unit'] = dmax_units
        
        return df
    
    def enrich_with_poi_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract POI names from UniProt IDs using UniProt API.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with POI_Name column
        """
        logger.info("Extracting POI names from UniProt...")
        
        poi_names = []
        unique_uniprots = df['POI_UniProt'].dropna().unique()
        
        uniprot_to_name = {}
        for uniprot_id in tqdm(unique_uniprots, desc="Fetching POI names"):
            gene_names = self.uniprot_fetcher.get_gene_names(uniprot_id)
            if gene_names:
                uniprot_to_name[uniprot_id] = gene_names[0]
            else:
                uniprot_to_name[uniprot_id] = None
        
        df['POI_Name'] = df['POI_UniProt'].map(uniprot_to_name)
        
        # Fill missing POI names with UniProt ID
        df['POI_Name'] = df['POI_Name'].fillna(df['POI_UniProt'])
        
        return df
    
    def enrich_with_poi_sequences(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fetch POI sequences from UniProt.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with POI_Sequence column
        """
        logger.info("Fetching POI sequences from UniProt...")
        
        unique_uniprots = df['POI_UniProt'].dropna().unique()
        sequences = {}
        
        for uniprot_id in tqdm(unique_uniprots, desc="Fetching POI sequences"):
            seq = self.uniprot_fetcher.get_sequence(uniprot_id)
            if seq:
                sequences[uniprot_id] = seq
        
        df['POI_Sequence'] = df['POI_UniProt'].map(sequences)
        
        missing = df['POI_Sequence'].isna().sum()
        if missing > 0:
            logger.warning(f"{missing} rows missing POI sequences")
        
        return df
    
    def enrich_with_ligase_sequences(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fetch E3 ligase sequences from UniProt.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with Ligase_Sequence column
        """
        logger.info("Fetching E3 ligase sequences from UniProt...")
        
        unique_uniprots = df['Ligase_UniProt'].dropna().unique()
        sequences = {}
        
        for uniprot_id in tqdm(unique_uniprots, desc="Fetching ligase sequences"):
            seq = self.uniprot_fetcher.get_sequence(uniprot_id)
            if seq:
                sequences[uniprot_id] = seq
        
        df['Ligase_Sequence'] = df['Ligase_UniProt'].map(sequences)
        
        return df
    
    def validate_and_enrich_cell_lines(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate cell lines against Cellosaurus and add IDs and species.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with Cell_Line_ID and Cell_Line_Species columns
        """
        logger.info("Validating cell lines with Cellosaurus...")
        
        # Clean cell line names
        df['Cell_Line_Clean'] = df['Cell_Line'].apply(clean_cell_line_name)
        
        # Get unique cell lines
        unique_cell_lines = df['Cell_Line_Clean'].dropna().unique()
        
        # Fetch Cellosaurus data
        cell_line_map = {}
        for cell_line in tqdm(unique_cell_lines, desc="Fetching Cellosaurus data"):
            accession = get_cellosaurus_accession(cell_line)
            species = get_cellosaurus_species(cell_line)
            cell_line_map[cell_line] = {
                'accession': accession,
                'species': species
            }
            # Add small delay to avoid overwhelming the API
            time.sleep(0.1)
        
        # Map results back to dataframe
        df['Cell_Line'] = df['Cell_Line_Clean']
        df['Cell_Line_ID'] = df['Cell_Line'].map(lambda x: cell_line_map.get(x, {}).get('accession'))
        df['Cell_Line_Species'] = df['Cell_Line'].map(lambda x: cell_line_map.get(x, {}).get('species'))
        
        # Drop temporary column
        df = df.drop(columns=['Cell_Line_Clean'])
        
        # Report validation results
        total = len(df)
        with_id = df['Cell_Line_ID'].notna().sum()
        logger.info(f"Cell line validation: {with_id}/{total} ({100*with_id/total:.1f}%) matched to Cellosaurus")
        
        return df
    
    def extract_assay_information(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract assay time and create assay description.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with Assay_Time and Assay columns
        """
        logger.info("Extracting assay information...")
        
        # Extract assay time
        df['Assay_Time'] = df.apply(extract_assay_time, axis=1)
        
        # Create assay description from Comments
        def create_assay_description(row):
            if pd.notna(row.get('Comments')):
                return str(row['Comments'])
            if pd.notna(row.get('Time')):
                return f"Degradation assay ({row['Time']})"
            return "Degradation assay"
        
        df['Assay'] = df.apply(create_assay_description, axis=1)
        
        return df
    
    def transform_to_long_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform wide format to long format with separate rows for DC50 and Dmax.
        Also parses Comments to extract cell-line-specific values and additional metrics.
        
        New columns added:
            - Value_Mean: Numeric mean value
            - Value_Error: Error bar (±)
            - Value_Operator: Comparison operator (<, >, ~)
            - Value_Range_Min: Range minimum
            - Value_Range_Max: Range maximum
            - Value_Concentration: Concentration at which value was measured
            - Value_Concentration_Unit: Unit of concentration
            - Value_Category: Type of value (numeric, range, text, etc.)
            - Additional metrics: Ligand_IC50, Ligand_EC50, PROTAC_IC50, PROTAC_EC50, etc.
        
        Args:
            df: Input DataFrame in wide format
            
        Returns:
            DataFrame in long format
        """
        logger.info("Transforming to long format with enhanced parsing...")
        
        rows = []
        manual_overrides = get_manual_curation_overrides()
        
        for idx, row in df.iterrows():
            base_row = {
                'SMILES': row.get('SMILES'),
                'POI_Name': row.get('POI_Name'),
                'POI_UniProt': row.get('POI_UniProt'),
                'POI_Sequence': row.get('POI_Sequence'),
                'Ligase_Name': row.get('Ligase_Name'),
                'Ligase_UniProt': row.get('Ligase_UniProt'),
                'Ligase_Sequence': row.get('Ligase_Sequence'),
                'Cell_Line': row.get('Cell_Line'),
                'Cell_Line_ID': row.get('Cell_Line_ID'),
                'Cell_Line_Species': row.get('Cell_Line_Species'),
                'Assay': row.get('Assay'),
                'Assay_Time': row.get('Assay_Time'),
                'Reference': process_reference(row),
                'Modality': 'PROteolysis-TArgeting Chimera (PROTAC)',
            }
            
            # Extract additional metrics from comments
            additional_metrics = extract_additional_metrics_from_comments(row.get('Comments'))
            base_row.update(additional_metrics)
            
            # Check for manual overrides
            comments = str(row.get('Comments', ''))
            handled_by_override = False
            
            for override_id, override_config in manual_overrides.items():
                pattern = override_config.get('pattern')
                if pattern and re.search(pattern, comments, re.IGNORECASE):
                    handler_name = override_config.get('handler')
                    if handler_name:
                        handler = globals().get(handler_name)
                        if handler:
                            try:
                                override_rows = handler(comments, row)
                                for override_row in override_rows:
                                    merged_row = base_row.copy()
                                    merged_row.update(override_row)
                                    rows.append(merged_row)
                                handled_by_override = True
                                logger.debug(f"Row {idx}: Handled by {handler_name}, created {len(override_rows)} rows")
                            except Exception as e:
                                logger.warning(f"Row {idx}: Override handler {handler_name} failed: {e}")
            
            if handled_by_override:
                continue
            
            # Try to parse cell-line-specific values from Comments
            cell_line_values = parse_cell_line_specific_values(row.get('Comments'))
            
            # If we found cell-line-specific values, create rows for each
            if cell_line_values:
                for cl_value in cell_line_values:
                    # Skip if no valid value
                    if cl_value['value'] is None:
                        continue
                    
                    # Create a modified base row with the specific cell line
                    specific_row = base_row.copy()
                    if cl_value.get('cell_line'):
                        # Clean and validate the cell line name
                        cleaned_cl = clean_cell_line_name(cl_value['cell_line'])
                        if cleaned_cl:
                            specific_row['Cell_Line'] = cleaned_cl
                            # Try to get Cellosaurus info for this cell line
                            accession = get_cellosaurus_accession(cleaned_cl)
                            species = get_cellosaurus_species(cleaned_cl)
                            if accession:
                                specific_row['Cell_Line_ID'] = accession
                            if species:
                                specific_row['Cell_Line_Species'] = species
                    
                    specific_row.update({
                        'Value': cl_value['value'],
                        'Value_Mean': cl_value['value'],
                        'Value_Type': cl_value['value_type'],
                        'Value_Unit': normalize_units(cl_value['unit']),
                        'Value_Symbol': cl_value['symbol'],
                        'Value_Operator': cl_value['symbol'] if cl_value['symbol'] else None,
                        'Value_Category': 'numeric',
                    })
                    rows.append(specific_row)
            else:
                # No cell-line-specific values found in comments, use regular approach
                # Add DC50 row if present
                if pd.notna(row.get('DC50_Value')):
                    dc50_row = base_row.copy()
                    parsed_value = parse_single_value(str(row.get('DC50', '')))
                    
                    dc50_row.update({
                        'Value': row['DC50_Value'],
                        'Value_Mean': row['DC50_Value'],
                        'Value_Type': 'DC50',
                        'Value_Unit': normalize_units(row.get('DC50_Value_Unit', 'nM')),
                        'Value_Symbol': row.get('DC50_Value_Symbol', ''),
                        'Value_Operator': row.get('DC50_Value_Symbol', '') if row.get('DC50_Value_Symbol') else None,
                        'Value_Category': 'numeric',
                    })
                    
                    if parsed_value:
                        dc50_row['Value_Error'] = parsed_value.get('error')
                    
                    rows.append(dc50_row)
                
                # Add Dmax row if present
                if pd.notna(row.get('Dmax_Value')):
                    dmax_row = base_row.copy()
                    parsed_value = parse_single_value(str(row.get('Dmax', '')))
                    
                    dmax_row.update({
                        'Value': row['Dmax_Value'],
                        'Value_Mean': row['Dmax_Value'],
                        'Value_Type': 'Dmax',
                        'Value_Unit': normalize_units(row.get('Dmax_Value_Unit', '%')),
                        'Value_Symbol': row.get('Dmax_Value_Symbol', ''),
                        'Value_Operator': row.get('Dmax_Value_Symbol', '') if row.get('Dmax_Value_Symbol') else None,
                        'Value_Category': 'numeric',
                    })
                    
                    if parsed_value:
                        dmax_row['Value_Error'] = parsed_value.get('error')
                    
                    rows.append(dmax_row)
        
        result = pd.DataFrame(rows)
        
        # Ensure all value columns exist
        value_columns = [
            'Value_Mean', 'Value_Error', 'Value_Operator', 
            'Value_Range_Min', 'Value_Range_Max',
            'Value_Concentration', 'Value_Concentration_Unit',
            'Value_Category'
        ]
        for col in value_columns:
            if col not in result.columns:
                result[col] = None
        
        logger.info(f"Created {len(result)} rows in long format from {len(df)} input rows")
        
        return result
    
    def curate(self) -> pd.DataFrame:
        """
        Run the full curation pipeline.
        
        Returns:
            Curated DataFrame
        """
        logger.info("Starting PROTAC-Pedia curation pipeline...")
        
        # Load data
        df = self.load_data()
        
        # Preprocess and rename columns
        df = self.preprocess_and_rename_columns(df)
        
        # Enrich with POI information
        df = self.enrich_with_poi_names(df)
        df = self.enrich_with_poi_sequences(df)
        
        # Enrich with ligase sequences
        df = self.enrich_with_ligase_sequences(df)
        
        # Validate and enrich cell lines
        df = self.validate_and_enrich_cell_lines(df)
        
        # Extract assay information
        df = self.extract_assay_information(df)
        
        # Transform to long format
        df_long = self.transform_to_long_format(df)
        
        # Save cache
        self.uniprot_fetcher.save_cache()
        
        # Save output
        output_path = Path(self.config.output_dir) / self.config.output_file
        df_long.to_csv(output_path, index=False)
        logger.info(f"Saved curated data to {output_path}")
        
        # Print summary statistics
        self._print_summary(df_long)
        
        return df_long
    
    def _print_summary(self, df: pd.DataFrame):
        """Print summary statistics."""
        logger.info("=" * 80)
        logger.info("Curation Summary")
        logger.info("=" * 80)
        logger.info(f"Total rows: {len(df)}")
        logger.info(f"Unique SMILES: {df['SMILES'].nunique()}")
        logger.info(f"Unique POIs: {df['POI_Name'].nunique()}")
        logger.info(f"Unique E3 ligases: {df['Ligase_Name'].nunique()}")
        logger.info(f"Unique cell lines: {df['Cell_Line'].nunique()}")
        
        logger.info("Value types:")
        for vt, count in df['Value_Type'].value_counts().items():
            logger.info(f"  {vt}: {count}")
        
        logger.info("Data completeness:")
        logger.info(f"  POI sequences: {df['POI_Sequence'].notna().sum()}/{len(df)} ({100*df['POI_Sequence'].notna().sum()/len(df):.1f}%)")
        logger.info(f"  Ligase sequences: {df['Ligase_Sequence'].notna().sum()}/{len(df)} ({100*df['Ligase_Sequence'].notna().sum()/len(df):.1f}%)")
        logger.info(f"  Cell line IDs: {df['Cell_Line_ID'].notna().sum()}/{len(df)} ({100*df['Cell_Line_ID'].notna().sum()/len(df):.1f}%)")
        logger.info(f"  Assay times: {df['Assay_Time'].notna().sum()}/{len(df)} ({100*df['Assay_Time'].notna().sum()/len(df):.1f}%)")

        logger.info("=" * 80)
        logger.info("Final files:")
        logger.info("=" * 80)
        logger.info(f"Curated dataset: {self.config.output_dir}/{self.config.output_file}")
        logger.info(f"UniProt cache: {self.config.cache_dir}/uniprot_cache.json")
        logger.info(f"Log file: {log_file}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Curate PROTAC-Pedia dataset with Cellosaurus validation"
    )

    parser.add_argument(
        "--input-file",
        type=str,
        default="data/original/PROTAC-Pedia.csv",
        help="Input CSV filename",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../data/curation",
        help="Output directory for curated data",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="protacpedia_protac_dc50_dmax_cleaned.csv",
        help="Output CSV filename",
    )
    parser.add_argument(
        "--force-refetch",
        action="store_true",
        help="Force re-fetching of UniProt data",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory to store log files",
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create configuration
    config = Config(
        output_dir=args.output_dir,
        cache_dir=args.output_dir,
        log_dir=args.log_dir,
        input_file=args.input_file,
        output_file=args.output_file,
        force_refetch=args.force_refetch,
    )
    
    # Run curation
    curator = ProtacPediaCurator(config)
    curator.curate()


if __name__ == "__main__":
    main()
