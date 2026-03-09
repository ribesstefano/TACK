
from typing import Dict
import logging

import pandas as pd

from tackai.data.embeddings.cell_embeddings import (
    CellEmbedding,
)


def get_manual_cell_mapping(
    row: pd.Series,
    manual_cell_lines: Dict[str, str],
) -> pd.Series:
    """ Get the manually curated cell line mapping for a given row.
    If the cell type is not in the manual mapping, return None for the
    standardized cell line. Used for logging and analysis purposes.
    
    Args:
        row: A row of the dataframe containing cell type information.
        manual_cell_lines: A dictionary mapping raw cell types to standardized cell lines.
        
    Returns:
        A pandas Series with the original cell type, the manually curated cell line (or None), and the article DOI for reference.
    """
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

def standardize_cell_line(
    row: pd.Series,
    cell_embedding: CellEmbedding,
    manual_cell_lines: Dict[str, str],
    logger: logging.Logger = None,
) -> pd.Series:
    """ Standardize the cell line information in the given row using the
        Cellosaurus embeddings and manual mapping.
        
    Args:
        row (pd.Series): A row of the dataframe containing cell type information.
        cell_embedding (CellEmbedding): An instance of CellEmbedding containing the cell line data.
        manual_cell_lines (Dict[str, str]): A dictionary mapping raw cell types to standardized cell lines.
        logger (logging.Logger, optional): A logger for logging warnings. If None, no warnings will be logged.

    Returns:
        pd.Series: The input row with standardized cell line information and added Cell ID.
    """
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
            if logger:
                logger.warning('WARNING: Cell line not found in Cellosaurus embeddings:')
                logger.warning(f'- {assay}\t{article_doi}\t{cell_line}')
            continue

        cell_id = cell_embedding.cell2cell_id.get(cell_line)
        row[cell_col.replace('Cell Type', 'Cell ID')] = cell_id
    return row

def get_cell_species(cell_type: str, cell_embedding: CellEmbedding) -> str:
    """Get the species of a cell line based on its type using Cellosaurus embeddings.
    
    Args:
        cell_type (str): The cell type for which to determine the species.
        cell_embedding (CellEmbedding): An instance of CellEmbedding containing the cell line data.
        
    Returns:
        str: The species of the cell line, or NaN if it cannot be determined.
    """
    if pd.isna(cell_type):
        return pd.NA
    data = cell_embedding.cell2data.get(cell_type)
    if data and 'OX' in data:
        organism = data.get('OX')
        if organism is not None:
            # Example: from "NCBI_TaxID=9606; ! Homo sapiens (Human)"
            #          return "Homo sapiens"
            return organism.split(' ! ')[-1].split(' (')[0]
    return pd.NA