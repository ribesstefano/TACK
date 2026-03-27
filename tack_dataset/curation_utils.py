"""
Shared utilities for the TACK dataset curation pipeline.

This module consolidates functions that were previously duplicated across
curate_protacdb.py, curate_tpddb.py, and curate_protacpedia.py.

Exports
-------
Chemistry:
    canonicalize_smiles

Units:
    normalize_unit
    convert_to_nM

Operators:
    normalize_operator

UniProt:
    UniProtFetcher

Cellosaurus:
    clean_cell_line_name
    CellosaurusClient

Logging:
    set_global_logging_level
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import List, Optional, Dict

import pandas as pd
import requests
from rdkit import Chem, RDLogger

from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog('rdApp.*')

# =============================================================================
# Chemistry
# =============================================================================

def canonicalize_smiles(smiles: str, unique_inchikeys: Optional[Dict[str, Chem.Mol]] = None) -> Optional[str]:
    """ Convert a SMILES string to its canonical form using RDKit.
    
    Args:
        smiles (str): The input SMILES string to canonicalize.
        unique_inchikeys (dict, optional): A dictionary to track unique InChI.
            If provided, the function will return None for any molecule whose
            InChIKey is already in the dictionary, effectively filtering out
            duplicates. The dictionary is updated with new InChIKeys for unique
            molecules.
            
    Returns:
        str or None: The canonical SMILES string, or None if invalid or duplicate.
    """
    if pd.isna(smiles) or not smiles:
        return None

    mol = Chem.MolFromSmiles(str(smiles), sanitize=False)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    try:
        # Keep only the largest fragment (i.e., remove salts)
        mol = rdMolStandardize.FragmentParent(mol)
        
        # Uncharge the molecule (i.e., remove formal charges)
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)

        if unique_inchikeys is not None:
            inchikey = Chem.MolToInchiKey(mol)
            if inchikey in unique_inchikeys:
                return None
            unique_inchikeys[inchikey] = mol

        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


# =============================================================================
# Units
# =============================================================================

# Maps raw unit strings to their standard forms.
UNIT_NORMALIZATION = {
    'nm':     'nM',
    'nM':     'nM',
    'nmol/L': 'nM',
    'uM':     'μM',
    'µM':     'μM',
    'um':     'μM',
    'μM':     'μM',
    'M':      'M',
    'pM':     'pM',
    'mM':     'mM',
    '%':      '%',
}


def normalize_unit(unit: Optional[str]) -> Optional[str]:
    """
    Normalize a single unit string to its canonical form.

    Examples:
        'nM'     → 'nM'
        'uM'     → 'μM'
        'µM'     → 'μM'
        'nmol/L' → 'nM'

    Returns None if the input is None or empty.
    """
    if not unit:
        return None
    return UNIT_NORMALIZATION.get(unit, unit)


def convert_to_nM(value: float, unit: str) -> Optional[float]:
    """
    Convert a concentration value to nanomolar (nM).

    Supported units: nM, μM, M, pM, mM.
    Returns None if the unit is unrecognised or the value is missing.
    """
    if pd.isna(value) or pd.isna(unit):
        return None

    unit = normalize_unit(unit) or unit

    conversion = {
        'nM': 1.0,
        'μM': 1e3,
        'M':  1e9,
        'pM': 1e-3,
        'mM': 1e6,
    }

    factor = conversion.get(unit)
    if factor is None:
        return None

    return float(value) * factor


# =============================================================================
# Operators
# =============================================================================

def normalize_operator(op: Optional[str]) -> Optional[str]:
    """
    Normalize a comparison operator to one of: '<', '>', '~', or None.

    Input variants:
        '<=', '≤'  → '<'
        '>=', '≥'  → '>'
        '~', '≈'   → '~'
        '*'         → None  (used in some sources to mean "approximate")
        None / ''   → None
    """
    if not op:
        return None
    op = op.strip()
    if op in ('<', '<=', '≤'):
        return '<'
    if op in ('>', '>=', '≥'):
        return '>'
    if op in ('~', '≈'):
        return '~'
    if op == '*':
        return None
    return op or None


# =============================================================================
# UniProt
# =============================================================================

class UniProtFetcher:
    """
    Fetch UniProt entries with disk + in-memory caching.

    Each entry is stored as a JSON file named <uniprot_id>.json inside
    cache_dir, so results persist across runs. An in-memory dict avoids
    repeated disk reads within the same run.

    Usage
    -----
    fetcher = UniProtFetcher(cache_dir=Path("data/curation/uniprot_cache"))
    seq = fetcher.get_sequence("P40337")
    names = fetcher.get_gene_names("P40337")
    """

    BASE_URL = "https://rest.uniprot.org/uniprotkb/{}.json"

    def __init__(self, cache_dir: Path, delay: float = 0.5):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self._memory: dict = {}
        self.logger = logging.getLogger(__name__)

    def _cache_path(self, uniprot_id: str) -> Path:
        return self.cache_dir / f"{uniprot_id}.json"

    def fetch_entry(self, uniprot_id: str) -> Optional[dict]:
        """
        Fetch a UniProt entry by accession ID.

        Checks the in-memory cache first, then the disk cache, then queries
        the UniProt REST API. Successful API responses are saved to disk.
        """
        if not uniprot_id:
            return None

        if uniprot_id in self._memory:
            return self._memory[uniprot_id]

        cache_path = self._cache_path(uniprot_id)
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    entry = json.load(f)
                self._memory[uniprot_id] = entry
                return entry
            except Exception as e:
                self.logger.debug(f"Failed to read disk cache for {uniprot_id}: {e}")

        url = self.BASE_URL.format(uniprot_id)
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            time.sleep(self.delay)
            entry = response.json()
            with open(cache_path, 'w') as f:
                json.dump(entry, f)
            self._memory[uniprot_id] = entry
            return entry
        except Exception as e:
            self.logger.warning(f"Failed to fetch UniProt entry {uniprot_id}: {e}")
            return None

    def get_sequence(self, uniprot_id: str) -> Optional[str]:
        """Return the canonical amino acid sequence for a UniProt accession."""
        entry = self.fetch_entry(uniprot_id)
        if entry and 'sequence' in entry:
            return entry['sequence'].get('value')
        return None

    def get_gene_names(self, uniprot_id: str) -> List[str]:
        """
        Return all gene names (primary + synonyms) for a UniProt accession.

        The first element of the returned list is the primary gene name.
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
# Cellosaurus
# =============================================================================

CELLOSAURUS_SEARCH_URL = "https://api.cellosaurus.org/search/cell-line"

# Known spelling variants mapped to the canonical Cellosaurus name.
CELL_LINE_NAME_FIXES = {
    'MOLT4':     'MOLT-4',
    'Hela':      'HeLa',
    'hela':      'HeLa',
    'HT1080':    'HT-1080',
    'HT 1080':   'HT-1080',
    'THP':       'THP-1',
    'Hs578t':    'Hs 578T',
    'Panc0213':  'Panc 02.13',
    'Panc02.13': 'Panc 02.13',
    'NAMALWA':   'Namalwa',
}


def clean_cell_line_name(cell_line: Optional[str]) -> Optional[str]:
    """
    Standardize a raw cell line string before a Cellosaurus lookup.

    Steps applied:
    1. If multiple cell lines are listed (comma/semicolon), keep only the first.
    2. Remove parenthetical annotations, e.g. "HeLa (human)" → "HeLa".
    3. Remove trailing biology words such as "cells", "cell line", "cancer".
    4. Remove leading species words such as "human", "mouse", "rat".
    5. Apply a small set of known spelling corrections.

    Returns None if the result is empty or the input is missing.
    """
    if pd.isna(cell_line) or not cell_line:
        return None

    s = str(cell_line).strip()

    # Keep only the first cell line if multiple are listed.
    if ',' in s or ';' in s:
        s = re.split(r'[,;]', s)[0].strip()

    # Remove parenthetical annotations.
    s = re.sub(r'\s*\(.*?\)', '', s).strip()

    # Remove trailing biology descriptors.
    s = re.sub(
        r'\s+(cells?|cell\s+line|monocytes?|cancer|breast|leukemia|carcinoma)$',
        '', s, flags=re.IGNORECASE,
    ).strip()

    # Remove leading species words.
    s = re.sub(r'^(human|mouse|rat)\s+', '', s, flags=re.IGNORECASE).strip()

    # Apply known spelling fixes (case-insensitive key match).
    for old, new in CELL_LINE_NAME_FIXES.items():
        if s.lower() == old.lower():
            return new

    return s or None


class CellosaurusClient:
    """
    Query the Cellosaurus API with disk + in-memory caching.

    Results are stored in a single JSON file (cellosaurus_cache.json) inside
    cache_dir, keyed by the cell line name string. This avoids re-querying
    the API for the same name across runs.

    Usage
    -----
    client = CellosaurusClient(cache_dir=Path("data/curation"))
    accession = client.get_accession("HeLa")   # → "CVCL_0030"
    species   = client.get_species("HeLa")     # → "Homo sapiens"
    client.save_cache()
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = self.cache_dir / 'cellosaurus_cache.json'
        self._cache: dict = self._load_cache()
        self.logger = logging.getLogger(__name__)

    def _load_cache(self) -> dict:
        if self._cache_file.exists():
            try:
                with open(self._cache_file) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def save_cache(self):
        """Write the in-memory cache to disk."""
        with open(self._cache_file, 'w') as f:
            json.dump(self._cache, f, indent=2)

    def _fetch(self, cell_line_name: str) -> Optional[dict]:
        """
        Query Cellosaurus for cell_line_name and return the first result.

        The result (or None if not found) is stored in the cache.
        """
        # Use a sentinel value to distinguish "not cached" from "cached as None".
        sentinel = '__NOT_CACHED__'
        cached = self._cache.get(cell_line_name, sentinel)
        if cached != sentinel:
            return cached

        result = None
        try:
            resp = requests.get(
                CELLOSAURUS_SEARCH_URL,
                params={'q': cell_line_name},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            cell_lines = data.get('Cellosaurus', {}).get('cell-line-list', [])
            if cell_lines:
                result = cell_lines[0]
        except Exception as e:
            self.logger.debug(f"Cellosaurus lookup failed for '{cell_line_name}': {e}")

        self._cache[cell_line_name] = result
        return result

    def get_accession(self, cell_line_name: str) -> Optional[str]:
        """
        Return the primary Cellosaurus accession (e.g. 'CVCL_0030') for a
        cell line name, or None if not found.
        """
        if not cell_line_name:
            return None
        entry = self._fetch(cell_line_name)
        if not entry:
            return None

        for acc in entry.get('accession-list', []):
            if isinstance(acc, dict) and acc.get('type') == 'primary':
                val = acc.get('value', '')
                if val.startswith('CVCL_'):
                    return val

        # Fallback: try direct keys that some API versions use.
        for key in ('accession', 'accession-id', 'id', 'ac'):
            acc = entry.get(key)
            if isinstance(acc, str) and acc.startswith('CVCL_'):
                return acc
            if isinstance(acc, list) and acc and acc[0].startswith('CVCL_'):
                return acc[0]

        return None

    def get_species(self, cell_line_name: str) -> Optional[str]:
        """
        Return the organism name (e.g. 'Homo sapiens') for a cell line.

        Falls back to heuristics based on the name if the API does not
        return a species.
        """
        if not cell_line_name:
            return None

        entry = self._fetch(cell_line_name)
        if entry:
            for sp in entry.get('species-list', []):
                if isinstance(sp, dict):
                    label = sp.get('label', '')
                    if label:
                        # Labels look like "Homo sapiens (Human)"; strip the parenthetical.
                        return label.split('(')[0].strip()
                    return sp.get('value') or sp.get('name')
                return str(sp)

            for key in ('species', 'organism'):
                sp = entry.get(key)
                if isinstance(sp, str):
                    return sp
                if isinstance(sp, list) and sp:
                    first = sp[0]
                    if isinstance(first, dict):
                        return first.get('value') or first.get('name')
                    return str(first)

        # Heuristic fallback based on the cell line name.
        name_lower = cell_line_name.lower()
        if any(k in name_lower for k in ('mouse', 'murine', '3t3', 'baf3', 'ba/f3')):
            return 'Mus musculus'
        if 'rat' in name_lower:
            return 'Rattus norvegicus'

        return None


# =============================================================================
# Logging
# =============================================================================

def set_global_logging_level(level=logging.ERROR, prefices=("",)):
    """
    Set the logging level for all loggers whose name starts with any of the
    given prefixes.

    Useful to silence noisy third-party libraries (e.g. rdkit, matplotlib).

    Example
    -------
        set_global_logging_level(logging.WARNING, prefices=["rdkit"])
    """
    for name, logger_obj in logging.root.manager.loggerDict.items():
        for prefix in prefices:
            if name.startswith(prefix):
                if isinstance(logger_obj, logging.Logger):
                    logger_obj.setLevel(level)
