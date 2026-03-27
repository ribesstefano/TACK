""" """
import os
import re
import logging
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict

import pandas as pd
import numpy as np
from tqdm import tqdm

from tack_dataset.logging_utils import setup_logging, set_global_logging_level
from tack_dataset.curation_utils import canonicalize_smiles
from tack_dataset.protacdb.assay_cleaning import parse_single_value
from tackai.data.embeddings.cell_embeddings import (
    CellEmbedding,
)
from tack_dataset.cell_utils import (
    standardize_cell_line,
    get_cell_species,
)
from tack_dataset.protein_utils import (
    E3_TO_ORGANISM_TO_UNIPROT,
    fetch_protein_info,
    fetch_uniprot_for_gene,
    apply_mutation,
)
from tack_dataset.protacdb.utils import load_dict, save_dict


def clean_cell_name(
        cell_name: str,
        cell_embedding: Optional[CellEmbedding],
        manual_cell_mapping: Optional[Dict[str, str]] = None,
        logger: Optional[logging.Logger] = None,
        fuzzy_matching_threshold: float = 0.6,
) -> Tuple[Optional[str], Optional[str]]:
    """" Wraps standardize_cell_line with manual overrides for known problematic cases in the PROTACpedia dataset.
    
    Args:
        cell_name (str): The raw cell line name to clean.
        manual_cell_mapping (Optional[Dict[str, str]]): A dictionary of manual mappings from raw cell line names to standardized names. This will be merged with the default mappings for PROTACpedia.
        logger (Optional[logging.Logger]): An optional logger for logging warnings or info during the cleaning process.
        fuzzy_matching_threshold (float): The threshold for fuzzy matching when standardizing cell line names. Default is 0.6.
        
    Returns:
        Tuple[Optional[str], Optional[str]]: A tuple containing the cleaned cell line name and its corresponding Cellosaurus ID. Both values will be None if the cell line name cannot be cleaned
    """
    if pd.isna(cell_name):
        return None, None
    cell_name = cell_name.replace('cell line', '').strip()
    if not cell_name:
        return None, None
    if manual_cell_mapping is None:
        manual_cell_mapping = {}
    return standardize_cell_line(
        cell_name,
        cell_embedding,
        manual_cell_mapping=dict(**{
            'K562 CML': 'K-562',
            '293FTCRBN−/−': 'HEK293FT',
            'CRBN-/-': 'HEK293FT',
            'PBMCs': 'PBMC iPSC #1',
            'Hela': 'HeLa',
            'Hella': 'HeLa',
            'hela': 'HeLa',
            'HeLa (EGFR Exon 20 Ins)': 'HeLa',
            'GFP-KRASG12C reporter  in Flp-In 293': 'Flp-In 293',
            'OVCAR8 (WT EGFR)': 'OVCAR8',
            'MCF-7 breast cancer cells': 'MCF-7',
            'KYSE520 esophageal cancer': 'KYSE520',
            'WI38 platelets': 'MOLT-4',
            'DU145/Cy': 'DU-145',
            '5W1573': 'SW1573',
            'BBL358': 'BL-358',
            'IgEMM': 'U266B1',
            'Human THP-1 monocytes': 'THP-1',
            'DLBCL': 'HBL-1 [Human diffuse large B-cell lymphoma]',
            'MM1.SCRBN−/−': 'MM.1S',
            'INC-H23': 'NCI-H23', # Typo in original data
            'PC-3': 'PC-3',
            'Cy': 'CY-6',
        }, **manual_cell_mapping),
        fuzzy_matching_threshold=fuzzy_matching_threshold,
        logger=logger,
    )


def main():
    parser = argparse.ArgumentParser(description='Curate PROTAC-Pedia dataset.')
    parser.add_argument('--input_path', type=str, default=Path('data/original/PROTAC-Pedia.csv'), help='Path to the raw PROTAC-Pedia CSV file.')
    parser.add_argument('--input_dir', type=str, default=Path('data/original'), help='Directory containing the raw PROTAC-Pedia CSV file (alternative to --input_path).')
    parser.add_argument('--output_dir', type=str, default=Path('data/curation'), help='Directory to save the curated dataset and intermediate files.')
    parser.add_argument('--force_refetch', action='store_true', help='Force refetching of UniProt entries even if cached files exist.')
    parser.add_argument('--log_dir', type=str, default=Path('logs'), help='Directory to save log files.')
    parser.add_argument('--verbose', '-v', action='count', default=0, help='Increase output verbosity (e.g. -v for INFO, -vv for DEBUG, -vvv for more detailed DEBUG).')

    args = parser.parse_args()

    # Setup logging
    log_file = setup_logging(args.log_dir, log_base_name='protacpedia_curation', verbose=args.verbose)
    set_global_logging_level(logging.DEBUG if args.verbose >= 3 else logging.INFO if args.verbose == 2 else logging.WARNING)
    logger = logging.getLogger(__name__)

    # Setup working directories
    data_curation_dir = Path(args.output_dir)
    os.makedirs(data_curation_dir, exist_ok=True)

    # Make the data_curation_dir / 'uniprot_infos' directory if it doesn't exist
    uniprot_infos_dir = data_curation_dir / 'uniprot_infos'
    os.makedirs(uniprot_infos_dir, exist_ok=True)

    protacpedia_file = Path(args.input_path)
    if os.path.exists(protacpedia_file):
        protacpedia_df = pd.read_csv(protacpedia_file).reset_index(drop=True)
    else:
        raise FileNotFoundError(f"PROTAC-Pedia file not found at: {protacpedia_file}")

    logging.info(f"Number of rows in PROTAC-Pedia before curation: {len(protacpedia_df)}")
    cols = ['Comments', 'Dc50', 'Dmax']
    protacpedia_df = protacpedia_df.dropna(subset=cols, how='all')
    logging.info(f"Number of rows in PROTAC-Pedia after dropping rows with all-NaN in {cols}: {len(protacpedia_df)}")

    cols_to_keep = [
        'PROTAC SMILES',
        'E3 Ligase',
        'Target',
        'Cells',
        'Dc50',
        'Dmax',
        'Time',
        'Comments',
        'Curator',
        'PATENT',
        'Ligand PDB',
        'Pubmed',
        'Ligand ID',
        'Secondary Pubmed',
    ]
    protacpedia_df = protacpedia_df[cols_to_keep]

    # Rename a few columns for consistency
    protacpedia_df = protacpedia_df.rename(columns={
        'PROTAC SMILES': 'SMILES',
        'Dc50': 'DC50',
        'Cells': 'Cell_Line',
        'E3 Ligase': 'Ligase_Name',
        'Target': 'POI_UniProt',
        'Time': 'Assay_Time',
        'Comments': 'Description',
    })

    def get_reference(row):
        """ Combine 'Secondary Pubmed', 'Pubmed', and 'PATENT' columns into a single
        'Reference' column with priority: Secondary Pubmed > Pubmed > PATENT. """
        if pd.notna(row['Secondary Pubmed']):
            return f"https://pubmed.ncbi.nlm.nih.gov/{int(row['Secondary Pubmed'])}/"
        elif pd.notna(row['Pubmed']):
            return f"https://pubmed.ncbi.nlm.nih.gov/{int(row['Pubmed'])}/"
        elif pd.notna(row['PATENT']):
            return f"Patent: {row['PATENT']}"

    protacpedia_df['Reference'] = protacpedia_df.apply(get_reference, axis=1)

    # ## Standardize SMILES

    logging.info(f"Number of unique SMILES before canonicalization: {protacpedia_df['SMILES'].nunique()}")
    protacpedia_df['SMILES'] = protacpedia_df['SMILES'].apply(canonicalize_smiles)
    protacpedia_df = protacpedia_df.dropna(subset=['SMILES']).reset_index(drop=True)
    logging.info(f"Number of unique SMILES after canonicalization:  {protacpedia_df['SMILES'].nunique()}")

    # ## POI Resolution

    all_e3_uniprots = []
    for species, ligase2uniprot in E3_TO_ORGANISM_TO_UNIPROT.items():
        for ligase, uniprot in ligase2uniprot.items():
            all_e3_uniprots.append(uniprot)
    logging.info(f"All E3 ligase Uniprot IDs from E3_LIGASE_2_UNIPROT: {all_e3_uniprots}")

    # Add manual mappings for corner cases in the dataset
    GENE_TO_UNIPROT = {
        'PBRM1': 'Q86U86',
        'BTK WT': 'Q06187',
        'BTK C481S': 'Q06187',
        'BRD4 LONG': 'O60885-1',
        'BRD4 SHORT': 'O60885-2',
        'EGFR WT': 'P00533',
        'EGFR Exon 20 Ins': 'P00533',
        'EGFR Exon 19 del': 'P00533',
        'EGFR L858R': 'P00533',
        'BCL-XL': 'Q64373',
    }

    # Replace ',' with '' in the POI_UniProt column, to split on spaces later
    protacpedia_df['POI_UniProt'] = protacpedia_df['POI_UniProt'].str.replace(', ', ' ', regex=False)

    # Print all POI_UniProt for which there are characters other than letters, numbers and spaces
    for uniprot_id in protacpedia_df['POI_UniProt'].unique():
        if re.search(r'[^a-zA-Z0-9\- ]', uniprot_id):
            logging.warning(f"WARNING: Uniprot ID '{uniprot_id}' contains non-alphanumeric characters.")

    uniprots = set(GENE_TO_UNIPROT.values())
    uniprots.update(all_e3_uniprots)
    for uniprot_id in protacpedia_df['POI_UniProt'].dropna().unique():
        if len(uniprot_id.split(' ')) > 1:
            for part in uniprot_id.split(' '):
                uniprots.add(part)
        else:
            uniprots.add(uniprot_id)
    logging.info(f"Unique Uniprot IDs (after splitting on spaces): {uniprots}")

    # Map uniprots to their gene names using the Uniprot API
    uniprot2info = {}
    uniprot2gene = {}
    uniprot2seq = {}
            
    # -- Fetch and cache UniProt entries defined above --
    for uniprot_id in tqdm(uniprots, desc='Fetching UniProt entries'):
        json_info = load_dict(data_curation_dir / 'uniprot_infos' / f'{uniprot_id}.json')
        if json_info and not args.force_refetch:
            uniprot2info[uniprot_id] = json_info
            uniprot2gene[uniprot_id] = json_info['gene_primary']
            uniprot2seq[uniprot_id] = json_info['sequence']
            for isoform in json_info.get('isoforms', []):
                uniprot_id = isoform['accession']
                # NOTE: We do not add isoforms to uniprot2gene since they share the
                # same gene name
                uniprot2info[uniprot_id] = isoform
                uniprot2seq[uniprot_id] = isoform['sequence']
        else:
            infos = fetch_protein_info(uniprot_id, skip_isoforms=False)
            if infos:
                # Save each entry to a separate JSON file
                save_dict(infos, data_curation_dir / 'uniprot_infos' / f'{uniprot_id}.json')
                uniprot2gene[uniprot_id] = infos['gene_primary']
                uniprot2seq[uniprot_id] = infos['sequence']
                uniprot2info[uniprot_id] = infos
                for isoform in infos.get('isoforms', []):
                    uniprot_id = isoform['accession']
                    save_dict(isoform, data_curation_dir / 'uniprot_infos' / f'{uniprot_id}.json')
                    uniprot2gene[uniprot_id] = isoform['gene_primary']
                    uniprot2seq[uniprot_id] = isoform['sequence']
            else:
                uniprot2gene[uniprot_id] = None
            
    gene2uniprot = {gene: uniprot for uniprot, gene in uniprot2gene.items() if gene is not None}
    gene2uniprot = {**gene2uniprot, **GENE_TO_UNIPROT}

    logging.info(f"Gene Name to Uniprot mapping: {gene2uniprot}")
    logging.info(f"Uniprot to Gene Name mapping: {uniprot2gene}")
    logging.info(f"Mapped Uniprots: {list(uniprot2info.keys())}")

    # Mapping (dict of dict): ('Curator', 'POI_UniProt') -> ('Description', 'Cell_Line', 'DC50', 'Dmax') - > POI Uniprot
    MANUAL_POI_MAP = {
        ("Ronen Gabizon", "O14976 O75385 P06239 P07332 P11802 P16591 P24941 P30291 P35991 P36888 P42680 P50613 P50750 P51451 P53671 Q00534 Q00537 Q05397 Q08881 Q13131 Q14004 Q14289 Q2M2I8 Q7KZI7 Q91820 Q96GD4 Q96SZ6 Q9NYV4"): {
            # Link: https://pubmed.ncbi.nlm.nih.gov/29129717/
            ("General kinase PROTAC, DCmax is for the most degraded kinase. IC50 of the ligand is for 193 kinases in the panel. IC50 of the PROTAC is by FLT3 kinase activity.", "MOLT-4, MOLM14", "< 100 nM", "> 85 %"): "P36888",
        },
        ("Ella Livnah", "O15264 Q16539"): {
            # Link: https://pubmed.ncbi.nlm.nih.gov/30631068/
            ("DC50 error ± 1.0 nM. DMAX error ± 1.1 %.", "MDA-MB-231, HeLa", "9.5 nM", "99.6 %"): "Q16539",
            # Link: https://pubmed.ncbi.nlm.nih.gov/30631068/
            ("DC50 error ± 81.3 nM. DMAX error ± 10.1 %.", "MDA-MB-231, HeLa", "45.9 nM", "34.5 %"): "Q16539",
        },
        ("Yangwode Jing", "O15379 Q13547 Q92769"): {
            # Link: https://pubmed.ncbi.nlm.nih.gov/32201871/
            (None, "E14 mouse embryonic stem cells; Human colon cancer cell line HCT116", "~ 10 uM", None): "Q13547",
            # Link: https://pubmed.ncbi.nlm.nih.gov/32201871/
            ("For HDAC1/HDAC2/HDAC3, the Dmax of this PROTAC is: >85%, >76%, >63%, respectively.", "E14 mouse embryonic stem cells; Human colon cancer cell line HCT116", "~ 1 uM", None): "Q13547",
        },
        ("Ronen Gabizon", "O60674 P23458"): {
            # Link: https://pubmed.ncbi.nlm.nih.gov/32001089/
            ("Structutre with ligand is modeled in the paper.", "THP", "2.5 uM", "60 %"): "P23458",
            # Link: https://pubmed.ncbi.nlm.nih.gov/32001089/
            ("Structutre with ligand is modeled in the paper.", "THP", "> 5 uM", "30 %"): "P23458",
            # Link: https://pubmed.ncbi.nlm.nih.gov/32001089/
            ("Structutre with ligand is modeled in the paper.", "THP", "> 5 uM", "50 %"): "P23458",
        },
        ("Ronen Gabizon", "O60674 P23458 P52333"): {
            # Link: https://pubmed.ncbi.nlm.nih.gov/32001089/
            ("Structutre with ligand is modeled in the paper.", "THP", None, None): "P23458",
            # Link: https://pubmed.ncbi.nlm.nih.gov/32001089/
            ("Structutre with ligand is modeled in the paper.", "THP", None, None): "P23458",
            # Link: https://pubmed.ncbi.nlm.nih.gov/32001089/
            ("Structutre with ligand is modeled in the paper.", "THP", None, None): "P23458",
            # Link: https://pubmed.ncbi.nlm.nih.gov/32001089/
            ("Structutre with ligand is modeled in the paper.", "THP", None, None): "P23458",
            # Link: https://pubmed.ncbi.nlm.nih.gov/32001089/
            ("Structutre with ligand is modeled in the paper.", "THP", None, None): "P23458",
            # Link: https://pubmed.ncbi.nlm.nih.gov/32001089/
            ("Structutre with ligand is modeled in the paper.", "THP", None, None): "P23458",
            # Link: https://pubmed.ncbi.nlm.nih.gov/32001089/
            ("Structutre with ligand is modeled in the paper.", "THP", "~ 5 uM", "60 %"): "P23458",
            # Link: https://pubmed.ncbi.nlm.nih.gov/32001089/
            ("Structutre with ligand is modeled in the paper.", "THP", None, None): "P23458",
            # Link: https://pubmed.ncbi.nlm.nih.gov/32001089/
            ("Structutre with ligand is modeled in the paper.", "THP", None, None): "P23458",
            # Link: https://pubmed.ncbi.nlm.nih.gov/32001089/
            ("Structutre with ligand is modeled in the paper. IC50 of PROTAC is between 10nM-50nM.", "THP", None, None): "P23458",
            # Link: https://pubmed.ncbi.nlm.nih.gov/32001089/
            ("Structutre with ligand is modeled in the paper. IC50 of PROTAC is between 10nM-50nM.", "THP", None, None): "P23458",
            # Link: https://pubmed.ncbi.nlm.nih.gov/32001089/
            ("Structutre with ligand is modeled in the paper.", "THP", None, None): "P23458",
            # Link: https://pubmed.ncbi.nlm.nih.gov/32001089/
            ("DC50 is between 1uM-2uM. Dmax is for JAK1.", "THP", "< 2 uM", "50 %"): "P23458",
            # Link: https://pubmed.ncbi.nlm.nih.gov/32001089/
            ("IC50 of PROTAC is 10-100 nM", "THP", None, None): "P23458",
            # Link: https://pubmed.ncbi.nlm.nih.gov/32001089/
            ("Dmax is for JAK1", "THP", "~ 2.5 uM", "60 %"): "P23458",
            # Link: https://pubmed.ncbi.nlm.nih.gov/32001089/
            ("IC50 of PROTAC is 10-100 nM", "THP", None, None): "P23458",
            # Link: https://pubmed.ncbi.nlm.nih.gov/32001089/
            ("IC50 of PROTAC is 10-100 nM", "THP", None, None): "P23458",
        },
        ("Daniel Zaidman", "O60885 P25440"): {
            # Link: https://pubmed.ncbi.nlm.nih.gov/31767403/
            ("EC50 of ligand is 0.370 uM for MV4- 11, 3.369 uM for Molm-13. EC50 of PROTAC is 1.648uM for MV4-11, >10uM for Molm-13", "MV4-11, Molm-13", None, None): "O60885",
            # Link: https://pubmed.ncbi.nlm.nih.gov/31767403/
            ("EC50 of ligand is 0.370 uM for MV4- 11, 3.369 uM for Molm-13. EC50 of PROTAC is 0.025uM for MV4-11, 0.18uM for Molm-13", "MV4-11, Molm-13", None, None): "O60885",
            # Link: https://pubmed.ncbi.nlm.nih.gov/31767403/
            ("EC50 of ligand is 0.370 uM for MV4- 11, 3.369 uM for Molm-13. EC50 of PROTAC is 0.012uM for MV4-11, 0.052uM for Molm-13.", "Big sellection of cacer cell lines", None, None): "O60885",
            # Link: https://pubmed.ncbi.nlm.nih.gov/31767403/
            ("EC50 of ligand is 0.370 uM for MV4- 11, 3.369 uM for Molm-13. EC50 of PROTAC is 0.032uM for MV4-11, 0.177uM for Molm-13.", "MV4-11, Molm-13", None, None): "O60885",
            # Link: https://pubmed.ncbi.nlm.nih.gov/31767403/
            ("EC50 of ligand is 0.370 uM for MV4- 11, 3.369 uM for Molm-13. EC50 of PROTAC is 3.429uM for MV4-11, >10uM for Molm-13.", "MV4-11, Molm-13", None, None): "O60885",
        },
        ("Ronen Gabizon", "O60885 P25440 Q15059"): {
            # Link: https://pubmed.ncbi.nlm.nih.gov/28339196/
            ("Direct degration studies in this paper only conducted for first and final compounds; during the med chem campaign cell viability was used as the read out. Complex structure with the ligand was modeled based on 4Z93. IC50 of the ligand is between 2nM and 7nM, depending on which BRD. DC50 is between 3nM-10nM.", "RS4;11, MOLM-13", "< 10 nM", "100 %"): "O60885",
            # Link: https://pubmed.ncbi.nlm.nih.gov/28339196/
            ("Complex structure with the ligand was modeled based on 4Z93. IC50 of the ligand is between 2nM and 7nM, depending on which BRD", "RS4;11, MOLM-13", "< 1 nM", "100 %"): "O60885",
            # Link: https://pubmed.ncbi.nlm.nih.gov/28339196/
            ("Complex structure with the ligand was modeled based on 4Z93. IC50 of the ligand is between 2nM and 7nM, depending on which BRD", "RS4;11, MOLM-13", "< 1 nM", "100 %"): "O60885",
            # Link: https://pubmed.ncbi.nlm.nih.gov/28339196/
            ("Complex structure with the ligand was modeled based on 4Z93. IC50 of the ligand is between 2nM and 7nM, depending on which BRD", "RS4;11, MOLM-13", "~ 3 nM", None): "O60885",
            # Link: https://pubmed.ncbi.nlm.nih.gov/28339196/
            ("Complex structure with the ligand was modeled based on 4Z93. IC50 of the ligand is between 2nM and 7nM, depending on which BRD", "RS4;11, MOLM-13", "~ 10 nM", None): "O60885",
        },
        ("Yangwode Jing", "O60885 P25440 Q15059"): {
            # Link: https://pubmed.ncbi.nlm.nih.gov/28595007/
            ("""pEC50 for MV4;11 and HL60 cells: 6.75±0.03 and 5.84±0.06, respectively.
pDC50 for Brd4 short/Brd4 long/Brd3/Brd2: 7.0/7.0/6.5/6.2, respectively (24h, HeLa cells).
Dmax for Brd4 short/Brd4 long/Brd3/Brd2: 96%/97%/97%/93%, respectively (HeLa cells).""", "HeLa, HL60, MV4;11", "< 0.1 uM", "> 93 %"): "O60885",
            # Link: https://pubmed.ncbi.nlm.nih.gov/28595007/
            ("pEC50 for MV4;11 and HL60 cells: 7.57±0.03 and 6.66±0.05, respectively. pDC50 for Brd4 short/Brd4 long/Brd3/Brd2: 8.1/8.6/7.0/7.4, respectively (24h, HeLa cells). Dmax for Brd4 short/Brd4 long/Brd3/Brd2: 98%/100%/100%/98%, respectively (HeLa cells).", "HeLa, HL60, MV4;11", "< 2.5 nM", "> 98 %"): "O60885",
            # Link: https://pubmed.ncbi.nlm.nih.gov/28595007/
            ("pEC50 for MV4;11 and HL60 cells: 6.91±0.04 and 5.90±0.05, respectively. pDC50 for Brd4 short/Brd4 long/Brd3/Brd2: 8.4/8.0/6.5/6.7, respectively (24h, HeLa cells). Dmax for Brd4 short/Brd4 long/Brd3/Brd2: 99%/100%/99%/97%, respectively (HeLa cells).", "HeLa, HL60, MV4;11", "< 4 nM", "> 97 %"): "O60885",
            # Link: https://pubmed.ncbi.nlm.nih.gov/28595007/
            ("pEC50 for MV4;11 and HL60 cells: 7.77±0.06 and 7.46±0.03, respectively. pDC50 for Brd4 short/Brd4 long/Brd3/Brd2: 9.2/9.0/9.1/8.2, respectively (24h, HeLa cells). Dmax for Brd4 short/Brd4 long/Brd3/Brd2: 97%/100%/98%/83%, respectively (HeLa cells).", "HeLa, HL60, MV4;11", "< 1 nM", "> 83 %"): "O60885",
        },
        ("Ronen Gabizon", "O60885 P53350"): {
            # Link: https://pubmed.ncbi.nlm.nih.gov/31708096/
            ("They show \"dual\" degradation but don't do any test that the dual degradation actually made any difference. They make it sound novel but of course we can call our PROTACs \"dual\" degraders if you can the off-targets \"targets\". EC50 of ligand is in measured in MV4-11 cell line. EC50 of PROTAC is between 4.5nM-6.94nM (depended on the cell line).", "MV4-11, MOLM-13, KG1", "< 5 nM", "~ 100 %"): "O60885",
        },
        ("Yangwode Jing", "O60885 Q15059"): {
            # Link: https://pubmed.ncbi.nlm.nih.gov/28595007/
            ("pEC50 for MV4;11 and HL60 cells: 6.24±0.05 and 6.17±0.03, respectively. pDC50 for Brd4 short/Brd4 long/Brd3/Brd2: 6.9/6.7/6.8/NA, respectively (24h, HeLa cells). Dmax for Brd4 short/Brd4 long/Brd3/Brd2: 94%/78%/74%/37%, respectively (HeLa cells).", "HeLa, HL60, MV4;11", "~ 0.1 uM", None): "O60885",
            # Link: https://pubmed.ncbi.nlm.nih.gov/28595007/
            ("pEC50 for MV4;11 and HL60 cells: 7.31±0.03 and 6.57±0.02, respectively. pDC50 for Brd4 short/Brd4 long/Brd3/Brd2: 8.1/7.6/7.3/NA, respectively (24h, HeLa cells). Dmax for Brd4 short/Brd4 long/Brd3/Brd2: 98%/95%/91%/43%, respectively (HeLa cells).", "HeLa, HL60, MV4;11", "~ 7.9 nM", "> 90 %"): "O60885",
            # Link: https://pubmed.ncbi.nlm.nih.gov/28595007/
            ("pEC50 for MV4;11 and HL60 cells: 7.08±0.05 and 6.37±0.03, respectively. pDC50 for Brd4 short/Brd4 long/Brd3/Brd2: 8.1/7.5/7.7/NA, respectively (24h, HeLa cells). Dmax for Brd4 short/Brd4 long/Brd3/Brd2: 95%/93%/92%/26%, respectively (HeLa cells).", "HeLa, HL60, MV4;11", "~ 7.9 nM", "> 90 %"): "O60885",
        },
        ("Shimrit Azulay", "O75530 Q15022 Q15910"): {
            # Link: https://pubmed.ncbi.nlm.nih.gov/31831267/
            (None, "HeLa", None, "47 %"): "O75530",
            # Link: https://pubmed.ncbi.nlm.nih.gov/31831267/
            ("DC50 value is for HeLa cells. DC50 is 0.61 uM for DLBCL cells. DMAX value is for HeLa cells. DMAX is 96 % for DLBCL cells.", "HeLa, DLBCL", "0.79 uM", "92 %"): "O75530",
            # Link: https://pubmed.ncbi.nlm.nih.gov/31831267/
            (None, "HeLa", None, "24 %"): "O75530",
            # Link: https://pubmed.ncbi.nlm.nih.gov/31831267/
            (None, "HeLa", None, "0.03 %"): "O75530",
            # Link: https://pubmed.ncbi.nlm.nih.gov/31831267/
            (None, "HeLa", None, "0.04 %"): "O75530",
            # Link: https://pubmed.ncbi.nlm.nih.gov/31831267/
            (None, "HeLa", None, "29 %"): "O75530",
        },
        ("Barr Tivon", "P00533 P04626"): {
            # Link: https://pubmed.ncbi.nlm.nih.gov/29129716/
            ("DC50 is for WT EGFR. DC50 for EGFR Exon 20 Ins is 736.2 nM. DMAX is for WT EGFR. DMAX for EGFR Exon 20 Ins is 68.8 %.", "OVCAR8 (WT EGFR), HeLa (EGFR Exon 20 Ins), SKBr3 (HER2)", "39.2 nM", "97.6 %"): "P00533",
        },
        ("Daniel Zaidman", "P11802 Q00534"): {
            # Link: https://pubmed.ncbi.nlm.nih.gov/30595531/
            (None, "AML cells", None, "~ 100 %"): "Q00534",
            # Link: https://pubmed.ncbi.nlm.nih.gov/32184044/
            ("inactive for CDK4, active for CDK6", "Jurkat", None, None): "Q00534",
        },
        ("Yangwode Jing", "P11802 Q00534"): {
            # Link: https://pubmed.ncbi.nlm.nih.gov/30802347/
            ("IC50: 77nM/27.6nM for CDK4/CDK6, respectively.", "Jurkat", None, None): "Q00534",
            # Link: https://pubmed.ncbi.nlm.nih.gov/30802347/
            ("IC50: 25.7nM/7.57nM for CDK4/CDK6, respectively. BSJ-02-162 is capable of degrading both CDK4 and CDK6.", "Jurkat; Molt4; Granta-519; Mino; Jeko; Rec1; Maver", None, None): "Q00534",
            # Link: https://pubmed.ncbi.nlm.nih.gov/30802347/
            ("IC50: 69.6nM/34.6nM for CDK4/CDK6, respectively.", "Jurkat", None, None): "Q00534",
            # Link: https://pubmed.ncbi.nlm.nih.gov/30802347/
            ("IC50: 130nM/45.1nM for CDK4/CDK6 respectively.", "Jurkat", None, None): "Q00534",
            # Link: https://pubmed.ncbi.nlm.nih.gov/30802347/
            ("IC50: 175nM/142nM for CDK4/CDK6, respectively.", "Jurkat", None, None): "Q00534",
            # Link: https://pubmed.ncbi.nlm.nih.gov/30802347/
            ("IC50: 79.7nM/65.5nM for CDK4/CDK6, respectively.", "Jurkat", None, None): "Q00534",
        },
        ("Ella Livnah", "P24941 P50750"): {
            # Link: https://pubmed.ncbi.nlm.nih.gov/31846828/
            ("Controls were done against CDK2. tested competition with ligand and with pomalidomide", "PC-3", None, None): "P50750",
            # Link: https://pubmed.ncbi.nlm.nih.gov/31846828/
            ("preferentially degrades CDK9 over CDK2. Controls were done against CDK2 and CDK9. tested competition with ligand and with pomalidomide. DC50 is CDK2: 62 nM, CDK9: 33 nM.", "PC-3", "< 62 nM", None): "P50750",
        },
        ("Yangwode Jing, Ronen Gabizon", "P29373 Q13490"): {
            # Link: https://pubmed.ncbi.nlm.nih.gov/22658364/
            (None, "HT1080, IMR-32", "~ 1 uM", None): "Q13490",
        },
        ("Shimrit Azulay, Daniel Zaidman", "P36507 Q02750"): {
            # Link: https://pubmed.ncbi.nlm.nih.gov/31804822/
            ("IC50 value of PROTAC tested with MEK1", "A375", None, None): "Q02750",
            # Link: https://pubmed.ncbi.nlm.nih.gov/31804822/
            ("IC50 value of PROTAC tested with MEK1", "A375", None, None): "Q02750",
            # Link: https://pubmed.ncbi.nlm.nih.gov/31804822/
            ("IC50 value of PROTAC tested with MEK1", "A375", None, None): "Q02750",
        },
        ("Efrat Resnick", "P51531 P51532"): {
            # Link: https://pubmed.ncbi.nlm.nih.gov/31178587/
            ("poor cellular permeability, ternary complex crystal structure: 6HAY (VCB:PROTAC 1:SMARCA2). DC50 is SMARCA2 300nM, SMARCA4 250nM. DCmax isSMARCA2 65%, SMARCA4 70%.", "MV-4-11", "< 300 nM", "< 70 %"): "P51531",
            # Link: https://pubmed.ncbi.nlm.nih.gov/31178587/
            ("ternary complex crystal structure: 6HAX (VCB:PROTAC 2:SMARCA2BD), 6HAR2 (VCB:PROTAC 2:SMARCA4BD)", None, None, None): "P51531",
            # Link: https://pubmed.ncbi.nlm.nih.gov/31178587/
            ("EC50 of PROTAC is 28nM in MV-4-11, 68nM in NCI-H1568. DC50 is (SMARCA2 6nM, SMARCA4 11nM, PBRM1 32nM in MV-4-11; SMARCA2 3.3nM, PBRM1 15.6nM in NCI-H1568)", "MV-4-11 SK-MEL-5, NCI-H1568", "< 32 nM", None): "P51531",
        },
        ("Yangwode Jing", "Q05397 Q14289"): {
            # Link: https://pubmed.ncbi.nlm.nih.gov/33062164/
            (None, "PA1", None, "> 70 %"): "Q05397",
            # Link: https://pubmed.ncbi.nlm.nih.gov/33062164/
            (None, "PA1", None, "> 60 %"): "Q05397",
            # Link: https://pubmed.ncbi.nlm.nih.gov/33062164/
            (None, "PA1", None, "> 90 %"): "Q05397",
            # Link: https://pubmed.ncbi.nlm.nih.gov/33062164/
            (None, "PA1", None, "> 95 %"): "Q05397",
            # Link: https://pubmed.ncbi.nlm.nih.gov/33062164/
            (None, "PA1", None, "> 90 %"): "Q05397",
            # Link: https://pubmed.ncbi.nlm.nih.gov/33062164/
            (None, "PA1", None, "> 80 %"): "Q05397",
            # Link: https://pubmed.ncbi.nlm.nih.gov/33062164/
            (None, "PA1", None, "> 65 %"): "Q05397",
            # Link: https://pubmed.ncbi.nlm.nih.gov/33062164/
            (None, "PA1", None, "> 90 %"): "Q05397",
            # Link: https://pubmed.ncbi.nlm.nih.gov/33062164/
            (None, "PA1", None, "> 94 %"): "Q05397",
            # Link: https://pubmed.ncbi.nlm.nih.gov/33062164/
            (None, "PA1", None, "> 80 %"): "Q05397",
            # Link: https://pubmed.ncbi.nlm.nih.gov/33062164/
            (None, "PA1", None, "> 90 %"): "Q05397",
            # Link: https://pubmed.ncbi.nlm.nih.gov/33062164/
            (None, "PA1", None, "> 90 %"): "Q05397",
            # Link: https://pubmed.ncbi.nlm.nih.gov/33062164/
            (None, "PA1", None, "> 90 %"): "Q05397",
            # Link: https://pubmed.ncbi.nlm.nih.gov/33062164/
            (None, "PA1", None, "> 90 %"): "Q05397",
            # Link: https://pubmed.ncbi.nlm.nih.gov/33062164/
            (None, "PA1", None, "> 95 %"): "Q05397",
            # Link: https://pubmed.ncbi.nlm.nih.gov/33062164/
            (None, "PA1", None, "> 90 %"): "Q05397",
            # Link: https://pubmed.ncbi.nlm.nih.gov/33062164/
            (None, "PA1", None, "> 85 %"): "Q05397",
        },
        ("Yangwode Jing, Ronen Gabizon", "Q05397 Q14289"): {
            # Link: https://pubmed.ncbi.nlm.nih.gov/32451721/
            ("This is an article that use a previously reported PROTAC (PMID: 33062164) as chemical biology tool to investigate the non-enzymatic FAK function in mice. Therefore, the relevant data were not given in this article.", "Mice primary Sertoli cells and primary Germ cells", "~ 1 nM", "~ 100 %"): "Q05397",
        },
        ("Efrat Resnick", "Q07820 Q92934"): {
            # Link: https://pubmed.ncbi.nlm.nih.gov/31389699/
            ("IC50 error: for ligand ± 0.87 uM, for PROTAC ± 3.66 uM", "hela", None, None): "Q07820",
            # Link: https://pubmed.ncbi.nlm.nih.gov/31389699/
            ("IC50 error: for ligand ± 0.87 uM, for PROTAC ± 2.13 uM", "hela", None, None): "Q07820",
            # Link: https://pubmed.ncbi.nlm.nih.gov/31389699/
            ("IC50 error: for ligand ± 0.87 uM, for PROTAC ± 0.58 uM", "hela", None, None): "Q07820",
            # Link: https://pubmed.ncbi.nlm.nih.gov/31389699/
            ("IC50 error: for ligand ± 0.87 uM, for PROTAC ± 4.44 uM", "hela", None, None): "Q07820",
            # Link: https://pubmed.ncbi.nlm.nih.gov/31389699/
            ("IC50 error: for ligand ± 0.87 uM, for PROTAC ± 0.89 uM", "hela", None, None): "Q07820",
            # Link: https://pubmed.ncbi.nlm.nih.gov/31389699/
            ("IC50 error: for ligand ± 1.39 uM, for PROTAC ± 4.33 uM", "hela", None, None): "Q07820",
            # Link: https://pubmed.ncbi.nlm.nih.gov/31389699/
            ("IC50 error: for ligand ± 1.39 uM, for PROTAC ± 2.90 uM", "hela", None, None): "Q07820",
            # Link: https://pubmed.ncbi.nlm.nih.gov/31389699/
            ("IC50 error: for ligand ± 1.39 uM, for PROTAC ± 0.27 uM", "hela", None, None): "Q07820",
            # Link: https://pubmed.ncbi.nlm.nih.gov/31389699/
            ("IC50 error: for ligand ± 1.39 uM, for PROTAC ± 4.57 uM", "hela", None, None): "Q07820",
            # Link: https://pubmed.ncbi.nlm.nih.gov/31389699/
            ("IC50 error: for ligand ± 1.39 uM, for PROTAC ± 1.36 uM", "hela", "3 uM", None): "Q07820",
            # Link: https://pubmed.ncbi.nlm.nih.gov/31389699/
            ("IC50 error: for ligand ± 1.39 uM, for PROTAC ± 2.28 uM", "hela", None, None): "Q07820",
        },
        ("Daniel Zaidman", "Q92830 Q92831"): {
            # Link: https://pubmed.ncbi.nlm.nih.gov/30200762/
            ("DC50 is 1.5nM/3nM. DCmax is 97%/91%.", "THP1", "< 3 nM", "> 91 %"): "Q92830",
            # Link: https://pubmed.ncbi.nlm.nih.gov/30200762/
            (None, "THP1", None, "97 %"): "Q92830",
            # Link: https://pubmed.ncbi.nlm.nih.gov/30200762/
            ("only slightly active", "THP1", None, "38 %"): "Q92830",
        },
        ("Daniel Zaidman, Yangwode Jing", "Q9H8M2 Q9NPI1"): {
            # Link: https://pubmed.ncbi.nlm.nih.gov/30540463/
            ("Engagment was tested in-vitro", "Hella", None, "32 %"): "Q9H8M2",
            # Link: https://pubmed.ncbi.nlm.nih.gov/30540463/
            (None, "Hella", None, "46 %"): "Q9H8M2",
            # Link: https://pubmed.ncbi.nlm.nih.gov/30540463/
            (None, "Hella", None, "20 %"): "Q9H8M2",
            # Link: https://pubmed.ncbi.nlm.nih.gov/30540463/
            (None, "Hella", None, "5 %"): "Q9H8M2",
            # Link: https://pubmed.ncbi.nlm.nih.gov/30540463/
            ("Engagment was tested in-vitro", "Hella", "560 nM", "10 %"): "Q9H8M2",
            # Link: https://pubmed.ncbi.nlm.nih.gov/30540463/
            (None, "Hella", None, "12 %"): "Q9H8M2",
            # Link: https://pubmed.ncbi.nlm.nih.gov/30540463/
            (None, "Hella", None, "5 %"): "Q9H8M2",
            # Link: https://pubmed.ncbi.nlm.nih.gov/30540463/
            (None, "Hella", None, "11 %"): "Q9H8M2",
            # Link: https://pubmed.ncbi.nlm.nih.gov/30540463/
            (None, "Hella", None, "2 %"): "Q9H8M2",
            # Link: https://pubmed.ncbi.nlm.nih.gov/30540463/
            (None, "Hella, RI-1", None, "92 %"): "Q9H8M2",
            # Link: https://pubmed.ncbi.nlm.nih.gov/30540463/
            (None, "Hella, RI-1", None, "97 %"): "Q9H8M2",
            # Link: https://pubmed.ncbi.nlm.nih.gov/30540463/
            ("DC50 is 1.76 nM and 4.5 nM", "Hella, RI-1, EOL-1, A-204", "< 4.5 nM", "90 %"): "Q9H8M2",
            # Link: https://pubmed.ncbi.nlm.nih.gov/30540463/
            (None, "Hella, RI-1", None, "35 %"): "Q9H8M2",
            # Link: https://pubmed.ncbi.nlm.nih.gov/30540463/
            (None, "Hella, RI-1", None, "17 %"): "Q9H8M2",
            # Link: https://pubmed.ncbi.nlm.nih.gov/30540463/
            (None, "Hella, RI-1", None, "71 %"): "Q9H8M2",
            # Link: https://pubmed.ncbi.nlm.nih.gov/30540463/
            (None, "Hella, RI-1", None, "47 %"): "Q9H8M2",
            # Link: https://pubmed.ncbi.nlm.nih.gov/30540463/
            (None, "Hella, RI-1", None, "2 %"): "Q9H8M2",
            # Link: https://pubmed.ncbi.nlm.nih.gov/30540463/
            (None, "Hella, RI-1", None, "15 %"): "Q9H8M2",
            # Link: https://pubmed.ncbi.nlm.nih.gov/30540463/
            (None, "Hella, RI-1", None, "75 %"): "Q9H8M2",
            # Link: https://pubmed.ncbi.nlm.nih.gov/30540463/
            (None, "Hella, RI-1", None, "46 %"): "Q9H8M2",
            # Link: https://pubmed.ncbi.nlm.nih.gov/30540463/
            (None, "Hella, RI-1", None, "69 %"): "Q9H8M2",
            # Link: https://pubmed.ncbi.nlm.nih.gov/30540463/
            (None, "Hella, RI-1", None, "3 %"): "Q9H8M2",
        },
    }

    # Assign the 'Unclear_POI' column based on whether 'POI_UniProt' is NaN or contains multiple entries
    protacpedia_df['Unclear_POI'] = protacpedia_df['POI_UniProt'].apply(lambda x: pd.isna(x) or (pd.notna(x) and len(x.split()) > 1))

    # ## Manual curation of Comments

    # Group by 'Curator' and get their respective comments
    curator_comments = protacpedia_df.groupby('Curator')['Description'].apply(list).to_dict()

    # Remove NaN values from comments
    for curator, comments in curator_comments.items():
        curator_comments[curator] = [comment for comment in comments if pd.notnull(comment)]
        curator_comments[curator] = list(set(curator_comments[curator]))  # Keep only unique comments

    # Remove curators with no comments
    curator_comments = {curator: comments for curator, comments in curator_comments.items() if len(comments) > 0}

    # Sort comments by number of comments per curator
    curator_comments = dict(sorted(curator_comments.items(), key=lambda item: len(item[1]), reverse=True))

    logger.debug(f'Number of curators: {len(curator_comments)}')
    logger.debug(f'Number of unique comments: {len(protacpedia_df["Description"].dropna().unique())}')
    logger.debug('-' * 80)

    num_comments = 0

    for curator, comments in curator_comments.items():
        # Filter comments that contain the words: "degradation", "dmax", "dc"
        in_words = ['degradation', 'dmax', 'dc']
        out_words = ['dcmax']
        comments = [comment for comment in comments if any(word in comment.lower() for word in in_words) and not any(word in comment.lower() for word in out_words)]
        
        if len(comments) == 0:
            continue
        
        num_comments += len(comments)

        # print(f"Curator: {curator}")
        # print(f"Number of comments: {len(comments)}")
        # print("Sample comments:")
        
        for i, comment in enumerate(comments):
            # print(f"\"\"\"{comment}\"\"\"")
            # print(f"{i+1}. {comment}")
            # print("-" * 40)
            pass

    logger.debug(f'Total number of comments containing "degradation", "dmax", or "dc" (but not "dcmax"): {num_comments}')

    # Helper: default dict template
    def _d(Value_Type, Value, Value_Unit='%', Cell_Line=None, Assay=None,
        Assay_Time=None, POI_Name=None, Value_Operator=None,
        Value_Category='numeric', Value_Range_Min=None, Value_Range_Max=None,
        Value_Error=None, Value_Concentration=None,
        Value_Concentration_Unit=None, Value_Mean=None):
        return {
            'Cell_Line': Cell_Line,
            'POI_Name': POI_Name,
            'Assay': Assay,
            'Assay_Time': Assay_Time,
            'Value': Value,
            'Value_Type': Value_Type,
            'Value_Unit': Value_Unit,
            'Value_Operator': Value_Operator,
            'Value_Category': Value_Category,
            'Value_Range_Min': Value_Range_Min,
            'Value_Range_Max': Value_Range_Max,
            'Value_Error': Value_Error,
            'Value_Concentration': Value_Concentration,
            'Value_Concentration_Unit': Value_Concentration_Unit,
            'Value_Mean': Value_Mean,
        }

    MANUAL_PARSED_COMMENTS = {
        "Dmax in BBL358/T47D: 74%±3% and 16%±13%, respectively.": [
            _d('Dmax', 74, '%', Cell_Line='BBL358', Value_Error=3),
            _d('Dmax', 16, '%', Cell_Line='T47D', Value_Error=13),
        ],
        "Dmax in BBL358/T47D: 67%±17% and 47%±32%, respectively.": [
            _d('Dmax', 67, '%', Cell_Line='BBL358', Value_Error=17),
            _d('Dmax', 47, '%', Cell_Line='T47D', Value_Error=32),
        ],
        "Dmax for KYSE520 cell: >95%; Degradation rate in MV4;11 cell: 6% (0.1uM compound)": [
            _d('Dmax', 95, '%', Cell_Line='KYSE520', Value_Operator='>'),
            _d('Dmax', 6, '%', Cell_Line='MV4;11',
            Value_Concentration=0.1, Value_Concentration_Unit='uM'),
        ],
        "IC50: 70.8nM/52.2nM for CDK4/CDK6, respectively; Show selective degradation of CDK4 (therefore active for CDK4 and inactive for CDK6).": [
            _d('IC50', 70.8, 'nM', POI_Name='CDK4'),
            _d('IC50', 52.2, 'nM', POI_Name='CDK6'),
        ],
        "EC50/DC50/Dmax reported above were obtained using NCI-H2030 cells. DC50: 0.25~0.76uM; Dmax: ~75%-90%, specific value depends on the cell line.": [
            _d('DC50', 'original', 'uM', Cell_Line='NCI-H2030'),
            _d('Dmax', 'original', '%', Cell_Line='NCI-H2030'),
            _d('DC50', None, 'uM', Value_Category='range', Value_Range_Min=0.25, Value_Range_Max=0.76),
            _d('Dmax', None, '%', Value_Category='range', Value_Range_Min=75, Value_Range_Max=90, Value_Operator='~'),
        ],
        "IC50: 50.6nM/30nM for CDK4/CDK6, respectively; Show selective degradation of CDK4 (therefore active for CDK4 and inactive for CDK6).": [
            _d('IC50', 50.6, 'nM', POI_Name='CDK4'),
            _d('IC50', 30, 'nM', POI_Name='CDK6'),
        ],
        "pEC50 for MV4;11 and HL60 cells: 6.91±0.04 and 5.90±0.05, respectively. pDC50 for Brd4 short/Brd4 long/Brd3/Brd2: 8.4/8.0/6.5/6.7, respectively (24h, HeLa cells). Dmax for Brd4 short/Brd4 long/Brd3/Brd2: 99%/100%/99%/97%, respectively (HeLa cells).": [
            _d('pEC50', 6.91, '', Cell_Line='MV4;11', Value_Error=0.04),
            _d('pEC50', 5.90, '', Cell_Line='HL60', Value_Error=0.05),
            _d('pDC50', 8.4, '', Cell_Line='HeLa', POI_Name='BRD4 SHORT', Assay_Time=24),
            _d('pDC50', 8.0, '', Cell_Line='HeLa', POI_Name='BRD4 LONG', Assay_Time=24),
            _d('pDC50', 6.5, '', Cell_Line='HeLa', POI_Name='BRD3', Assay_Time=24),
            _d('pDC50', 6.7, '', Cell_Line='HeLa', POI_Name='BRD2', Assay_Time=24),
            _d('Dmax', 99, '%', Cell_Line='HeLa', POI_Name='BRD4 SHORT'),
            _d('Dmax', 100, '%', Cell_Line='HeLa', POI_Name='BRD4 LONG'),
            _d('Dmax', 99, '%', Cell_Line='HeLa', POI_Name='BRD3'),
            _d('Dmax', 97, '%', Cell_Line='HeLa', POI_Name='BRD2'),
        ],
        "Dmax in BBL358/T47D: 3%±3% and 20%±20%, respectively.": [
            _d('Dmax', 3, '%', Cell_Line='BBL358', Value_Error=3),
            _d('Dmax', 20, '%', Cell_Line='T47D', Value_Error=20),
        ],
        "Degradation rate for KYSE520 cell: 87% (1uM compound); Degradation rate for MV4;11 cell: 85% (0.1uM compound)": [
            _d('Dmax', 87, '%', Cell_Line='KYSE520', Value_Concentration=1, Value_Concentration_Unit='uM'),
            _d('Dmax', 85, '%', Cell_Line='MV4;11', Value_Concentration=0.1, Value_Concentration_Unit='uM'),
        ],
        "Dmax for KYSE520 cell: >80%; Degradation rate for MV4;11 cell: 9% (0.1uM compound)": [
            _d('Dmax', 80, '%', Cell_Line='KYSE520', Value_Operator='>'),
            _d('Dmax', 9, '%', Cell_Line='MV4;11', Value_Concentration=0.1, Value_Concentration_Unit='uM'),
        ],
        "Reported DC50 and Dmax above are in HeLa cells. DC50 for HEK293 cells: 230nM; Dmax for HEK293 cells: 98%. 14a can degrade VHL at higher concentration. See Target UniprotID P40337 for details.": [
            _d('DC50', 'original', 'nM', Cell_Line='HeLa'),
            _d('Dmax', 'original', '%', Cell_Line='HeLa'),
            _d('DC50', 230, 'nM', Cell_Line='HEK293'),
            _d('Dmax', 98, '%', Cell_Line='HEK293'),
        ],
        """pEC50 for MV4;11 and HL60 cells: 6.75±0.03 and 5.84±0.06, respectively.
    pDC50 for Brd4 short/Brd4 long/Brd3/Brd2: 7.0/7.0/6.5/6.2, respectively (24h, HeLa cells).
    Dmax for Brd4 short/Brd4 long/Brd3/Brd2: 96%/97%/97%/93%, respectively (HeLa cells).""": [
            _d('pEC50', 6.75, '', Cell_Line='MV4;11', Value_Error=0.03),
            _d('pEC50', 5.84, '', Cell_Line='HL60', Value_Error=0.06),
            _d('pDC50', 7.0, '', Cell_Line='HeLa', POI_Name='BRD4 SHORT', Assay_Time=24),
            _d('pDC50', 7.0, '', Cell_Line='HeLa', POI_Name='BRD4 LONG', Assay_Time=24),
            _d('pDC50', 6.5, '', Cell_Line='HeLa', POI_Name='BRD3', Assay_Time=24),
            _d('pDC50', 6.2, '', Cell_Line='HeLa', POI_Name='BRD2', Assay_Time=24),
            _d('Dmax', 96, '%', Cell_Line='HeLa', POI_Name='BRD4 SHORT'),
            _d('Dmax', 97, '%', Cell_Line='HeLa', POI_Name='BRD4 LONG'),
            _d('Dmax', 97, '%', Cell_Line='HeLa', POI_Name='BRD3'),
            _d('Dmax', 93, '%', Cell_Line='HeLa', POI_Name='BRD2'),
        ],
        "Dmax in BBL358/T47D: 40%±27% and 0, respectively.": [
            _d('Dmax', 40, '%', Cell_Line='BBL358', Value_Error=27),
            _d('Dmax', 0, '%', Cell_Line='T47D'),
        ],
        "pEC50 for MV4;11 and HL60 cells: 7.57±0.03 and 6.66±0.05, respectively. pDC50 for Brd4 short/Brd4 long/Brd3/Brd2: 8.1/8.6/7.0/7.4, respectively (24h, HeLa cells). Dmax for Brd4 short/Brd4 long/Brd3/Brd2: 98%/100%/100%/98%, respectively (HeLa cells).": [
            _d('pEC50', 7.57, '', Cell_Line='MV4;11', Value_Error=0.03),
            _d('pEC50', 6.66, '', Cell_Line='HL60', Value_Error=0.05),
            _d('pDC50', 8.1, '', Cell_Line='HeLa', POI_Name='BRD4 SHORT', Assay_Time=24),
            _d('pDC50', 8.6, '', Cell_Line='HeLa', POI_Name='BRD4 LONG', Assay_Time=24),
            _d('pDC50', 7.0, '', Cell_Line='HeLa', POI_Name='BRD3', Assay_Time=24),
            _d('pDC50', 7.4, '', Cell_Line='HeLa', POI_Name='BRD2', Assay_Time=24),
            _d('Dmax', 98, '%', Cell_Line='HeLa', POI_Name='BRD4 SHORT'),
            _d('Dmax', 100, '%', Cell_Line='HeLa', POI_Name='BRD4 LONG'),
            _d('Dmax', 100, '%', Cell_Line='HeLa', POI_Name='BRD3'),
            _d('Dmax', 98, '%', Cell_Line='HeLa', POI_Name='BRD2'),
        ],
        "pEC50 for MV4;11 and HL60 cells: 7.77±0.06 and 7.46±0.03, respectively. pDC50 for Brd4 short/Brd4 long/Brd3/Brd2: 9.2/9.0/9.1/8.2, respectively (24h, HeLa cells). Dmax for Brd4 short/Brd4 long/Brd3/Brd2: 97%/100%/98%/83%, respectively (HeLa cells).": [
            _d('pEC50', 7.77, '', Cell_Line='MV4;11', Value_Error=0.06),
            _d('pEC50', 7.46, '', Cell_Line='HL60', Value_Error=0.03),
            _d('pDC50', 9.2, '', Cell_Line='HeLa', POI_Name='BRD4 SHORT', Assay_Time=24),
            _d('pDC50', 9.0, '', Cell_Line='HeLa', POI_Name='BRD4 LONG', Assay_Time=24),
            _d('pDC50', 9.1, '', Cell_Line='HeLa', POI_Name='BRD3', Assay_Time=24),
            _d('pDC50', 8.2, '', Cell_Line='HeLa', POI_Name='BRD2', Assay_Time=24),
            _d('Dmax', 97, '%', Cell_Line='HeLa', POI_Name='BRD4 SHORT'),
            _d('Dmax', 100, '%', Cell_Line='HeLa', POI_Name='BRD4 LONG'),
            _d('Dmax', 98, '%', Cell_Line='HeLa', POI_Name='BRD3'),
            _d('Dmax', 83, '%', Cell_Line='HeLa', POI_Name='BRD2'),
        ],
        "pEC50 for MV4;11 and HL60 cells: 7.08±0.05 and 6.37±0.03, respectively. pDC50 for Brd4 short/Brd4 long/Brd3/Brd2: 8.1/7.5/7.7/NA, respectively (24h, HeLa cells). Dmax for Brd4 short/Brd4 long/Brd3/Brd2: 95%/93%/92%/26%, respectively (HeLa cells).": [
            _d('pEC50', 7.08, '', Cell_Line='MV4;11', Value_Error=0.05),
            _d('pEC50', 6.37, '', Cell_Line='HL60', Value_Error=0.03),
            _d('pDC50', 8.1, '', Cell_Line='HeLa', POI_Name='BRD4 SHORT', Assay_Time=24),
            _d('pDC50', 7.5, '', Cell_Line='HeLa', POI_Name='BRD4 LONG', Assay_Time=24),
            _d('pDC50', 7.7, '', Cell_Line='HeLa', POI_Name='BRD3', Assay_Time=24),
            _d('pDC50', None, '', Cell_Line='HeLa', POI_Name='BRD2', Assay_Time=24),
            _d('Dmax', 95, '%', Cell_Line='HeLa', POI_Name='BRD4 SHORT'),
            _d('Dmax', 93, '%', Cell_Line='HeLa', POI_Name='BRD4 LONG'),
            _d('Dmax', 92, '%', Cell_Line='HeLa', POI_Name='BRD3'),
            _d('Dmax', 26, '%', Cell_Line='HeLa', POI_Name='BRD2'),
        ],
        "XD2-149 was initially designed to degrade STAT3. However, experiments showed that XD2-149 down-regulate STAT3 level in a proteasome-independent manner. Proteomics data revealed that an E3 ligase, ZFP91, was the true substrate for this PROTAC. This paper reported a total of 22 PROTACs, which differed from each other in linker design and E3 binder choices (pomalidomide/thalidomide/lenalidomide). However, the authors didn't mention whether the remaining 21 molecules could degrade ZFP91 or not. It is also interesting that pomalidomide itself can induce the degradation of CRBN neo-substrates like ZFP91 (DC50: 0.42uM, 5-fold less potent than XD2-149), since pomalidomide can remodel CRBN surface for binding proteins like ZFP91 (Nat Med., 2019, doi: 10.1038/s41591-019-0668-z).": [
            _d('DC50', 0.42, 'uM', POI_Name='STAT3'),
        ],
        "pEC50 for MV4;11 and HL60 cells: 6.24±0.05 and 6.17±0.03, respectively. pDC50 for Brd4 short/Brd4 long/Brd3/Brd2: 6.9/6.7/6.8/NA, respectively (24h, HeLa cells). Dmax for Brd4 short/Brd4 long/Brd3/Brd2: 94%/78%/74%/37%, respectively (HeLa cells).": [
            _d('pEC50', 6.24, '', Cell_Line='MV4;11', Value_Error=0.05),
            _d('pEC50', 6.17, '', Cell_Line='HL60', Value_Error=0.03),
            _d('pDC50', 6.9, '', Cell_Line='HeLa', POI_Name='BRD4 SHORT', Assay_Time=24),
            _d('pDC50', 6.7, '', Cell_Line='HeLa', POI_Name='BRD4 LONG', Assay_Time=24),
            _d('pDC50', 6.8, '', Cell_Line='HeLa', POI_Name='BRD3', Assay_Time=24),
            _d('pDC50', None, '', Cell_Line='HeLa', POI_Name='BRD2', Assay_Time=24),
            _d('Dmax', 94, '%', Cell_Line='HeLa', POI_Name='BRD4 SHORT'),
            _d('Dmax', 78, '%', Cell_Line='HeLa', POI_Name='BRD4 LONG'),
            _d('Dmax', 74, '%', Cell_Line='HeLa', POI_Name='BRD3'),
            _d('Dmax', 37, '%', Cell_Line='HeLa', POI_Name='BRD2'),
        ],
        "Dmax for KYSE520 cell: >95%; Dmax for MV4;11 cell: >90%": [
            _d('Dmax', 95, '%', Cell_Line='KYSE520', Value_Operator='>'),
            _d('Dmax', 90, '%', Cell_Line='MV4;11', Value_Operator='>'),
        ],
        "IC50: 137nM/39nM for CDK4/CDK6, respectively; Show selective degradation of CDK6 (therefore active for CDK6 and inactive for CDK4).": [
            _d('IC50', 137, 'nM', POI_Name='CDK4'),
            _d('IC50', 39, 'nM', POI_Name='CDK6'),
        ],
        "Degradation rate: 20% (0.1uM compound) or 14% (1uM compound)": [
            _d('Dmax', 20, '%', Value_Concentration=0.1, Value_Concentration_Unit='uM'),
            _d('Dmax', 14, '%', Value_Concentration=1, Value_Concentration_Unit='uM'),
        ],
        "pEC50 for MV4;11 and HL60 cells: 7.31±0.03 and 6.57±0.02, respectively. pDC50 for Brd4 short/Brd4 long/Brd3/Brd2: 8.1/7.6/7.3/NA, respectively (24h, HeLa cells). Dmax for Brd4 short/Brd4 long/Brd3/Brd2: 98%/95%/91%/43%, respectively (HeLa cells).": [
            _d('pEC50', 7.31, '', Cell_Line='MV4;11', Value_Error=0.03),
            _d('pEC50', 6.57, '', Cell_Line='HL60', Value_Error=0.02),
            _d('pDC50', 8.1, '', Cell_Line='HeLa', POI_Name='BRD4 SHORT', Assay_Time=24),
            _d('pDC50', 7.6, '', Cell_Line='HeLa', POI_Name='BRD4 LONG', Assay_Time=24),
            _d('pDC50', 7.3, '', Cell_Line='HeLa', POI_Name='BRD3', Assay_Time=24),
            _d('pDC50', None, '', Cell_Line='HeLa', POI_Name='BRD2', Assay_Time=24),
            _d('Dmax', 98, '%', Cell_Line='HeLa', POI_Name='BRD4 SHORT'),
            _d('Dmax', 95, '%', Cell_Line='HeLa', POI_Name='BRD4 LONG'),
            _d('Dmax', 91, '%', Cell_Line='HeLa', POI_Name='BRD3'),
            _d('Dmax', 43, '%', Cell_Line='HeLa', POI_Name='BRD2'),
        ],
        "Degradation rate in KYSE520 cell: 90% (1uM compound); Degradation rate in MV4;11 cell: 8% (0.1uM compound)": [
            _d('Dmax', 90, '%', Cell_Line='KYSE520', Value_Concentration=1, Value_Concentration_Unit='uM'),
            _d('Dmax', 8, '%', Cell_Line='MV4;11', Value_Concentration=0.1, Value_Concentration_Unit='uM'),
        ],
        "Dmax in BBL358/T47D: 0.5% and 76%, respectively.": [
            _d('Dmax', 0.5, '%', Cell_Line='BBL358'),
            _d('Dmax', 76, '%', Cell_Line='T47D'),
        ],
        "Dmax in BBL358/T47D: 77%±3% and 82%±10%, respectively.": [
            _d('Dmax', 77, '%', Cell_Line='BBL358', Value_Error=3),
            _d('Dmax', 82, '%', Cell_Line='T47D', Value_Error=10),
        ],
        "For HDAC1/HDAC2/HDAC3, the Dmax of this PROTAC is: >85%, >76%, >63%, respectively.": [
            _d('Dmax', 85, '%', POI_Name='HDAC1', Value_Operator='>'),
            _d('Dmax', 76, '%', POI_Name='HDAC2', Value_Operator='>'),
            _d('Dmax', 63, '%', POI_Name='HDAC3', Value_Operator='>'),
        ],
        "Dmax in BBL358/T47D: 93%±5% and 87%±3%, respectively.": [
            _d('Dmax', 93, '%', Cell_Line='BBL358', Value_Error=5),
            _d('Dmax', 87, '%', Cell_Line='T47D', Value_Error=3),
        ],
        "Dmax for KYSE520 cell: >95%; Degradation rate in MV4;11 cell: 47% (0.1uM compound)": [
            _d('Dmax', 95, '%', Cell_Line='KYSE520', Value_Operator='>'),
            _d('Dmax', 47, '%', Cell_Line='MV4;11', Value_Concentration=0.1, Value_Concentration_Unit='uM'),
        ],
        "Dmax in BBL358/T47D: 84%±2% and 89%±2%, respectively.": [
            _d('Dmax', 84, '%', Cell_Line='BBL358', Value_Error=2),
            _d('Dmax', 89, '%', Cell_Line='T47D', Value_Error=2),
        ],
        "DC50 for KYSE520 cell: 6.0nM;DC50 for MV4;11 cell: 2.6nM; EC50 for KYSE520 cell: 0.66uM; EC50 for MV4;11 cell: 9.9nM;": [
            _d('DC50', 6.0, 'nM', Cell_Line='KYSE520'),
            _d('DC50', 2.6, 'nM', Cell_Line='MV4;11'),
            _d('EC50', 0.66, 'uM', Cell_Line='KYSE520'),
            _d('EC50', 9.9, 'nM', Cell_Line='MV4;11'),
        ],
        "Dmax in BBL358/T47D: 8% and 0, respectively.": [
            _d('Dmax', 8, '%', Cell_Line='BBL358'),
            _d('Dmax', 0, '%', Cell_Line='T47D'),
        ],
        "Dmax is measured in 10uM. EC50 of PROTAC is MCF-7: 2.70 ± 0.19 MDA-MB-231: 21.21 ± 1.95 HepG2: 18.70 ± 1.65 LO2: 41.11 ± 3.70 B16: 22.68 ± 2.03 uM. EC50 of ligand is MCF-7: 4.17 ± 0.31 MDA-MB-231: 21.33 ± 1.96 HepG2: 10.59 ± 0.94 LO2: 35.57 ± 2.81 B16: 14.49 ± 1.28 uM.": [
            _d('Dmax', 'original', '%', Value_Concentration=10, Value_Concentration_Unit='uM'),
            _d('EC50', 2.70, 'uM', Cell_Line='MCF-7', Value_Error=0.19),
            _d('EC50', 21.21, 'uM', Cell_Line='MDA-MB-231', Value_Error=1.95),
            _d('EC50', 18.70, 'uM', Cell_Line='HepG2', Value_Error=1.65),
            _d('EC50', 41.11, 'uM', Cell_Line='LO2', Value_Error=3.70),
            _d('EC50', 22.68, 'uM', Cell_Line='B16', Value_Error=2.03),
            _d('EC50', 4.17, 'uM', Cell_Line='MCF-7', Value_Error=0.31),
            _d('EC50', 21.33, 'uM', Cell_Line='MDA-MB-231', Value_Error=1.96),
            _d('EC50', 10.59, 'uM', Cell_Line='HepG2', Value_Error=0.94),
            _d('EC50', 35.57, 'uM', Cell_Line='LO2', Value_Error=2.81),
            _d('EC50', 14.49, 'uM', Cell_Line='B16', Value_Error=1.28),
        ],
        "protein reduction by nearly 50% was observed as early as 1 h post\u2010treatment, and almost complete degradation was achieved after 4 h of treatment": [
            _d('Dmax', 50, '%', Value_Operator='~', Assay_Time=1),
            _d('Dmax', 100, '%', Value_Operator='~', Assay_Time=4),
        ],
        "half the paper develops the ligand and checks its selectivity. Dmax is 76% Karpas 422, 59% ULA, 84% SUDHL4, 82% OCI-Ly1, 82% Ramos.": [
            _d('Dmax', 76, '%', Cell_Line='Karpas 422'),
            _d('Dmax', 59, '%', Cell_Line='ULA'),
            _d('Dmax', 84, '%', Cell_Line='SUDHL4'),
            _d('Dmax', 82, '%', Cell_Line='OCI-Ly1'),
            _d('Dmax', 82, '%', Cell_Line='Ramos'),
        ],
        "DC50 for CDK4 > 100 nM": [
            _d('DC50', 100, 'nM', POI_Name='CDK4', Value_Operator='>'),
        ],
        "they heva a ternary complex crystal structure but dont give PDB. DC50 is between 25-125nM": [
            _d('DC50', None, 'nM', Value_Category='range', Value_Range_Min=25, Value_Range_Max=125),
        ],
        "DC50 for CDK4 > 500 nM": [
            _d('DC50', 500, 'nM', POI_Name='CDK4', Value_Operator='>'),
        ],
        "DC50 for CDK4 > 50 nM": [
            _d('DC50', 50, 'nM', POI_Name='CDK4', Value_Operator='>'),
        ],
        "EC50 of PROTAC is 28nM in MV-4-11, 68nM in NCI-H1568. DC50 is (SMARCA2 6nM, SMARCA4 11nM, PBRM1 32nM in MV-4-11; SMARCA2 3.3nM, PBRM1 15.6nM in NCI-H1568)": [
            _d('EC50', 28, 'nM', Cell_Line='MV-4-11'),
            _d('EC50', 68, 'nM', Cell_Line='NCI-H1568'),
            _d('DC50', 6, 'nM', Cell_Line='MV-4-11', POI_Name='SMARCA2'),
            _d('DC50', 11, 'nM', Cell_Line='MV-4-11', POI_Name='SMARCA4'),
            _d('DC50', 32, 'nM', Cell_Line='MV-4-11', POI_Name='PBRM1'),
            _d('DC50', 3.3, 'nM', Cell_Line='NCI-H1568', POI_Name='SMARCA2'),
            _d('DC50', 15.6, 'nM', Cell_Line='NCI-H1568', POI_Name='PBRM1'),
        ],
        "DC50 for CDK4 > 100 nM. EC50 of PROTAC is 10nM in MM.lS 8nM in Mino.": [
            _d('DC50', 100, 'nM', POI_Name='CDK4', Value_Operator='>'),
            _d('EC50', 10, 'nM', Cell_Line='MM.1S'),
            _d('EC50', 8, 'nM', Cell_Line='Mino'),
        ],
        "DC50 is 1-3nM": [
            # NOTE: The corresponding Cell_Line column contains: "HeLa, MV4;11, A549"
            _d('DC50', None, 'nM', Value_Category='range', Value_Range_Min=1, Value_Range_Max=3, Cell_Line='HeLa'),
            _d('DC50', None, 'nM', Value_Category='range', Value_Range_Min=1, Value_Range_Max=3, Cell_Line='MV4;11'),
            _d('DC50', None, 'nM', Value_Category='range', Value_Range_Min=1, Value_Range_Max=3, Cell_Line='A549'),
        ],
        "DC50 is 0.86nM in LNCaP, 0.76 in VCaP and 10.4 nM at 1uM in 22Rv1. EC50 in cells is 0.25nM for LNCaP, 0.34 nM for VCaP, 183nM for 22Rv1.": [
            _d('DC50', 0.86, 'nM', Cell_Line='LNCaP'),
            _d('DC50', 0.76, 'nM', Cell_Line='VCaP'),
            _d('DC50', 10.4, 'nM', Cell_Line='22Rv1', Value_Concentration=1, Value_Concentration_Unit='uM'),
            _d('EC50', 0.25, 'nM', Cell_Line='LNCaP'),
            _d('EC50', 0.34, 'nM', Cell_Line='VCaP'),
            _d('EC50', 183, 'nM', Cell_Line='22Rv1'),
        ],
        "DC50 is 10-30nM": [
            # NOTE: The corresponding Cell_Line column contains: "HeLa, MV4;11, A549"
            _d('DC50', None, 'nM', Value_Category='range', Value_Range_Min=10, Value_Range_Max=30, Cell_Line='HeLa'),
            _d('DC50', None, 'nM', Value_Category='range', Value_Range_Min=10, Value_Range_Max=30, Cell_Line='MV4;11'),
            _d('DC50', None, 'nM', Value_Category='range', Value_Range_Min=10, Value_Range_Max=30, Cell_Line='A549'),
        ],
        "DC50 is between 0.1uM-1uM": [
            _d('DC50', None, 'uM', Value_Category='range', Value_Range_Min=0.1, Value_Range_Max=1),
        ],
        "Dmax is difficult to quantify because AC220 raises FLT3 expression levels. IC50 of ligand is 1.6 nM (binding); 1.1-4 (activity). EC50 of ligand is measured on MOLM-14 cell lines. EC50 of PROTAC was measured on MOLM-14 cell lines.": [
            _d('IC50', 1.6, 'nM', Assay='ligand binding'),
            _d('IC50', None, 'nM', Assay='ligand activity', Value_Category='range', Value_Range_Min=1.1, Value_Range_Max=4),
            _d('EC50', 'original', 'nM', Cell_Line='MOLM-14'),
            _d('EC50', 'original', 'nM', Cell_Line='MOLM-14'),
        ],
        "DC50 is between 1uM-2uM. Dmax is for JAK1.": [
            _d('DC50', None, 'uM', Value_Category='range', Value_Range_Min=1, Value_Range_Max=2),
            _d('Dmax', 'original', '%', POI_Name='JAK1'),
        ],
        "Dmax is for JAK1": [
            _d('Dmax', 'original', '%', POI_Name='JAK1'),
        ],
        "Direct degration studies in this paper only conducted for first and final compounds; during the med chem campaign cell viability was used as the read out. Complex structure with the ligand was modeled based on 4Z93. IC50 of the ligand is between 2nM and 7nM, depending on which BRD. DC50 is between 3nM-10nM.": [
            _d('IC50', None, 'nM', Assay='ligand vs BRD', Value_Category='range', Value_Range_Min=2, Value_Range_Max=7),
            _d('DC50', None, 'nM', Value_Category='range', Value_Range_Min=3, Value_Range_Max=10),
        ],
        'They show "dual" degradation but don\'t do any test that the dual degradation actually made any difference. They make it sound novel but of course we can call our PROTACs "dual" degraders if you can the off-targets "targets". EC50 of ligand is in measured in MV4-11 cell line. EC50 of PROTAC is between 4.5nM-6.94nM (depended on the cell line).': [
            _d('EC50', 'original', 'nM', Cell_Line='MV4-11'),
            _d('EC50', None, 'nM', Value_Category='range', Value_Range_Min=4.5, Value_Range_Max=6.94),
        ],
        "DC50 value is for HeLa cells. DC50 is 0.61 uM for DLBCL cells. DMAX value is for HeLa cells. DMAX is 96 % for DLBCL cells.": [
            _d('DC50', 'original', 'uM', Cell_Line='HeLa'),
            _d('DC50', 0.61, 'uM', Cell_Line='DLBCL'),
            _d('Dmax', 'original', '%', Cell_Line='HeLa'),
            _d('Dmax', 96, '%', Cell_Line='DLBCL'),
        ],
        "also degrades ibrutinib-resistant C481S BTK. DC50 is 6.3 nM (HBL1), 8.5 nM (Ramos), 9.2 nM (Mino), 11.4 nM (IgE MM)": [
            _d('DC50', 6.3, 'nM', Cell_Line='HBL1'),
            _d('DC50', 8.5, 'nM', Cell_Line='Ramos'),
            _d('DC50', 9.2, 'nM', Cell_Line='Mino'),
            _d('DC50', 11.4, 'nM', Cell_Line='IgE MM'),
        ],
        "also tested degradation for two common mutants of ERa. EC50 of PROTAC was measured in MCF-7 cell line.": [
            _d('EC50', 'original', 'nM', Cell_Line='MCF-7'),
        ],
        "achieved full degradation after 24hrs with addition of off-target CDK10 degradation. EC50 of ligand is measured on MOLT-4 cell line": [
            _d('Dmax', 100, '%', Value_Operator='~', Assay_Time=24),
            _d('EC50', 'original', 'nM', Cell_Line='MOLT-4'),
        ],
        "preferentially degrades CDK9 over CDK2. Controls were done against CDK2 and CDK9. tested competition with ligand and with pomalidomide. DC50 is CDK2: 62 nM, CDK9: 33 nM.": [
            _d('DC50', 62, 'nM', POI_Name='CDK2'),
            _d('DC50', 33, 'nM', POI_Name='CDK9'),
        ],
        "DC50 error ± 81.3 nM. DMAX error ± 10.1 %.": [
            # NOTE: The corresponding Cell_Line column contains: "MDA-MB-231, HeLa"
            _d('DC50', 'original', 'nM', Value_Error=81.3, Cell_Line='MDA-MB-231'),
            _d('Dmax', 'original', '%', Value_Error=10.1, Cell_Line='HeLa'),
        ],
        "DC50 error ± 1.0 nM. DMAX error ± 1.1 %.": [
            # NOTE: The corresponding Cell_Line column contains: "MDA-MB-231, HeLa"
            _d('DC50', 'original', 'nM', Value_Error=1.0, Cell_Line='MDA-MB-231'),
            _d('Dmax', 'original', '%', Value_Error=1.1, Cell_Line='HeLa'),
        ],
        "DC50 is for EGFR (Exon 19 del). DC50 for EGFR (L858R) 22.3 nM. DMAX is for EGFR (Exon 19 del). DMAX for EGFR (L858R) 96.6 %.": [
            # NOTE: The corresponding Cell_Line column contains: "HCC827 (Exon 19 del), H3255 (L858R)"
            _d('DC50', 'original', 'nM', POI_Name='EGFR Exon 19 del', Cell_Line='HCC827'),
            _d('DC50', 22.3, 'nM', POI_Name='EGFR L858R', Cell_Line='H3255'),
            _d('Dmax', 'original', '%', POI_Name='EGFR Exon 19 del', Cell_Line='HCC827'),
            _d('Dmax', 96.6, '%', POI_Name='EGFR L858R', Cell_Line='H3255'),
        ],
        "DC50 is 9.1 nM (WT BTK, NAMALWA cells), 14.6 nM (WT BTK, XLA cells), 14.9 nM (C481S, XLA cells). IC50 of PROTAC is 46.9 (WT BTK), 20.9 (C481S). IC50 of ligand is 51.0 nM (WT BTK), 30.7 (C481S)": [
            _d('DC50', 9.1, 'nM', Cell_Line='NAMALWA', POI_Name='BTK WT'),
            _d('DC50', 14.6, 'nM', Cell_Line='XLA', POI_Name='BTK WT'),
            _d('DC50', 14.9, 'nM', Cell_Line='XLA', POI_Name='BTK C481S'),
            _d('IC50', 46.9, 'nM', POI_Name='BTK WT', Assay='PROTAC'),
            _d('IC50', 20.9, 'nM', POI_Name='BTK C481S', Assay='PROTAC'),
            _d('IC50', 51.0, 'nM', POI_Name='BTK WT', Assay='ligand'),
            _d('IC50', 30.7, 'nM', POI_Name='BTK C481S', Assay='ligand'),
        ],
        "DC50 is for WT EGFR. DC50 for EGFR Exon 20 Ins is 736.2 nM. DMAX is for WT EGFR. DMAX for EGFR Exon 20 Ins is 68.8 %.": [
            # NOTE: The corresponding Cell_Line column contains: "OVCAR8 (WT EGFR), HeLa (EGFR Exon 20 Ins), SKBr3 (HER2)"
            _d('DC50', 'original', 'nM', POI_Name='EGFR WT', Cell_Line='OVCAR8'),
            _d('DC50', 736.2, 'nM', POI_Name='EGFR Exon 20 Ins', Cell_Line='HeLa'),
            _d('Dmax', 'original', '%', POI_Name='EGFR WT', Cell_Line='OVCAR8'),
            _d('Dmax', 68.8, '%', POI_Name='EGFR Exon 20 Ins', Cell_Line='HeLa'),
        ],
        "DC50 and Dmax are for MCF-7 cells": [
            _d('DC50', 'original', 'nM', Cell_Line='MCF-7'),
            _d('Dmax', 'original', '%', Cell_Line='MCF-7'),
        ],
        "proteomics of PDEdelta degradation show upregulation of enzymes involved in lipid metabolism - deltasonamide 1 also causes this. Dmax was measured in Panc-Tu-1 cell line. DC50 is 83.4% (24 h, Panc-Tu-1), 85% (24 h, 1 uM, Jurkat). IC50 Deltasonamide 1 is 203 pM, and of the Bn derivative is 8 nM.": [
            # NOTE: I think there is a typp and the annotator meant to say "Dmax" instead of "DC50".
            _d('Dmax', 83.4, '%', Cell_Line='Panc-Tu-1', Assay_Time=24),
            _d('Dmax', 85, '%', Cell_Line='Jurkat', Assay_Time=24, Value_Concentration=1, Value_Concentration_Unit='uM'),
            _d('IC50', 203, 'pM', Assay='Deltasonamide 1'),
            _d('IC50', 8, 'nM', Assay='Bn derivative'),
        ],
        "Selective BCLXL degradation: only in MOLT4 cells, not platelets": [
            _d('Dmax', 'original', '%', Cell_Line='MOLT4', POI_Name='BCL-XL'),
            # NOTE: Instead of platelets, we point to human acute megakaryoblastic
            # leukemia cell line used to study platelet production.
            _d('Dmax', 0, '%', Cell_Line='LK-4', POI_Name='BCL-XL'),
        ],
        "proteomics of PDEdelta degradation show upregulation of enzymes involved in lipid metabolism - deltasonamide 1 also causes this. IC50 Deltasonamide 1 is 203 pM, and of the Bn derivative is 8 nM.": [
            _d('IC50', 203, 'pM', Assay='Deltasonamide 1'),
            _d('IC50', 8, 'nM', Assay='Bn derivative'),
        ],
        "Tested reversal of docetaxel resistance by degradation of CYP, followed by lysis and WB, model with the ligand is available in previous publication": [
            # No quantitative DC50/Dmax/pDC50 data to extract
            _d('Dmax', 'original', '%', Assay='Western Blot'),
            _d('DC50', 'original', 'nM', Assay='Western Blot'),
        ],
        "DC50 is 1.76 nM and 4.5 nM": [
            _d('DC50', 1.76, 'nM'),
            _d('DC50', 4.5, 'nM'),
        ],
        "competition with ligand only slightly rescued protein degradation. EC50 of ligand and PROTAC is measured on SUDHL-1 cell line. DC50 is SU-DHL-1: 3 ± 1 nM, NCI-H2228: 34 ± 9 nM.": [
            _d('EC50', 'original', 'nM', Cell_Line='SUDHL-1'),
            _d('EC50', 'original', 'nM', Cell_Line='SUDHL-1'),
            _d('DC50', 3, 'nM', Cell_Line='SU-DHL-1', Value_Error=1),
            _d('DC50', 34, 'nM', Cell_Line='NCI-H2228', Value_Error=9),
        ],
        "competition with ligand only slightly rescued protein degradation. EC50 of ligand and PROTAC is measured on SUDHL-1 cell line. DC50 is SU-DHL-1: 11 ± 2 nM, NCI-H2228: 59 ± 16 nM.": [
            _d('EC50', 'original', 'nM', Cell_Line='SUDHL-1'),
            _d('EC50', 'original', 'nM', Cell_Line='SUDHL-1'),
            _d('DC50', 11, 'nM', Cell_Line='SU-DHL-1', Value_Error=2),
            _d('DC50', 59, 'nM', Cell_Line='NCI-H2228', Value_Error=16),
        ],
        "Exhibits potent anti-HCV activity. The first demonstration of a PROTAC antiviral effect by degradation of a host protein (cyclophilin A).": [
            # No quantitative DC50/Dmax/pDC50 data to extract
        ],
        # NOTE: I checked and the annotator meant DMax, not DCmax. Check: https://pmc.ncbi.nlm.nih.gov/articles/PMC6077745/
        "Ligand name is from PMID 24068666. DCmax is 36-64%. DC50 is 1487 - 1994.5 nM.": [
            # NOTE: The corresponding Cell_Line column contains: 'Ramos, THP-1'
            _d('Dmax', None, '%', Value_Category='range', Value_Range_Min=36, Value_Range_Max=64, Cell_Line='Ramos'),
            _d('DC50', None, 'nM', Value_Category='range', Value_Range_Min=1487, Value_Range_Max=1994.5, Cell_Line='THP-1'),
        ],
        "Ligand name is from PMID 24068666. DCmax is 68-85%. DC50 is 36.9 - 398.5 nM.": [
            # NOTE: The corresponding Cell_Line column contains: 'Ramos, THP-1'
            _d('Dmax', None, '%', Value_Category='range', Value_Range_Min=68, Value_Range_Max=85, Cell_Line='Ramos'),
            _d('DC50', None, 'nM', Value_Category='range', Value_Range_Min=36.9, Value_Range_Max=398.5, Cell_Line='THP-1'),
        ],
        "Ligand name is from PMID 24068666. DCmax is 63-84%. DC50 is 21.8 - 469.9 nM.": [
            # NOTE: The corresponding Cell_Line column contains: 'Ramos, THP-1'
            _d('Dmax', None, '%', Value_Category='range', Value_Range_Min=63, Value_Range_Max=84, Cell_Line='Ramos'),
            _d('DC50', None, 'nM', Value_Category='range', Value_Range_Min=21.8, Value_Range_Max=469.9, Cell_Line='THP-1'),
        ],
        "Ligand name is from PMID 24068666. DCmax is 75-87%. DC50 is 4.5 - 90.5 nM.": [
            # NOTE: The corresponding Cell_Line column contains: 'Ramos, THP-1'
            _d('Dmax', None, '%', Value_Category='range', Value_Range_Min=75, Value_Range_Max=87, Cell_Line='Ramos'),
            _d('DC50', None, 'nM', Value_Category='range', Value_Range_Min=4.5, Value_Range_Max=90.5, Cell_Line='THP-1'),
        ],
        "Ligand name is from PMID 24068666. DCmax is 71-85%. DC50 is 5.9 - 217.7 nM.": [
            # NOTE: The corresponding Cell_Line column contains: 'Ramos, THP-1'
            _d('Dmax', None, '%', Value_Category='range', Value_Range_Min=71, Value_Range_Max=85, Cell_Line='Ramos'),
            _d('DC50', None, 'nM', Value_Category='range', Value_Range_Min=5.9, Value_Range_Max=217.7, Cell_Line='THP-1'),
        ],
        "Ligand name is from PMID 24068666. DCmax is 80-87%. DC50 is 1.1 - 37.4 nM.": [
            # NOTE: The corresponding Cell_Line column contains: 'Ramos, THP-1'
            _d('Dmax', None, '%', Value_Category='range', Value_Range_Min=80, Value_Range_Max=87, Cell_Line='Ramos'),
            _d('DC50', None, 'nM', Value_Category='range', Value_Range_Min=1.1, Value_Range_Max=37.4, Cell_Line='THP-1'),
        ],
        "Ligand name is from PMID 24068666. DCmax is 70-85%. DC50 is 9.7 - 184.1 nM.": [
            # NOTE: The corresponding Cell_Line column contains: 'Ramos, THP-1'
            _d('Dmax', None, '%', Value_Category='range', Value_Range_Min=70, Value_Range_Max=85, Cell_Line='Ramos'),
            _d('DC50', None, 'nM', Value_Category='range', Value_Range_Min=9.7, Value_Range_Max=184.1, Cell_Line='THP-1'),
        ],
    }

    total_entries = sum(len(v) for v in MANUAL_PARSED_COMMENTS.values())
    logger.info(f"Total comments: {len(MANUAL_PARSED_COMMENTS)}")
    logger.info(f"Total extracted entries: {total_entries}")
    # Quick validation
    required_keys = {
        'Cell_Line', 'POI_Name', 'Assay', 'Assay_Time', 'Value', 'Value_Type',
        'Value_Unit', 'Value_Operator', 'Value_Category',
        'Value_Range_Min', 'Value_Range_Max', 'Value_Error',
        'Value_Concentration', 'Value_Concentration_Unit', 'Value_Mean',
    }
    num_dc50 = sum(1 for entries in MANUAL_PARSED_COMMENTS.values() for entry in entries if entry['Value_Type'] == 'DC50')
    num_dmax = sum(1 for entries in MANUAL_PARSED_COMMENTS.values() for entry in entries if entry['Value_Type'] == 'Dmax')
    logger.info(f"Total DC50 entries: {num_dc50}")
    logger.info(f"Total Dmax entries: {num_dmax}")

    for i, (comment, entries) in enumerate(MANUAL_PARSED_COMMENTS.items()):
        if comment not in protacpedia_df['Description'].values:
            logger.warning(f"Comment n.{i+1} not in DataFrame:\n```\n{comment}\n```")
            # raise ValueError("Comment not found in DataFrame")
        for i, entry in enumerate(entries):
            missing = required_keys - set(entry.keys())
            extra = set(entry.keys()) - required_keys
            if missing or extra:
                logger.info(f"  Key issue in: {comment[:60]}... entry {i}")
                if missing:
                    logger.info(f"    Missing: {missing}")
                if extra:
                    logger.info(f"    Extra: {extra}")
    logger.info("Validation complete.")

    # ## Cells Standardization

    def is_unclear_cell(reported_cell_line):
        if pd.isna(reported_cell_line):
            return False
        if reported_cell_line.strip() == 'MV-4-11 SK-MEL-5':
            return True
        elif '; ' in reported_cell_line:  # Notice the space after the semicolon!!
            return True
        elif reported_cell_line.endswith(','):
            return True
        elif ',' in reported_cell_line:
            return True
        elif ' and ' in reported_cell_line:
            return True
        elif '/' in reported_cell_line:
            return True
        return False

    protacpedia_df['Unclear_Cell_Line'] = protacpedia_df['Cell_Line'].apply(is_unclear_cell)
    num_unclear = protacpedia_df['Unclear_Cell_Line'].sum()
    logger.info(f"Number of entries with unclear cell line: {num_unclear}")

    cell_embedding = CellEmbedding(verbose=1)

    MANUAL_CELL_MAP = {
        # B16 is the classic mouse melanoma line. The parental B16 doesn't have
        # its own Cellosaurus entry separate from subclones; B16-F0 is the
        # closest "parental" entry.
        # TODO: If the data doesn't specify the subclone, B16-F0 (CVCL_F602) is the
        # safest default. Alternatively we can use B16-F10 (CVCL_0159) if the paper
        # used the metastatic variant.
        'B16': 'B16-F0',  # CVCL_F602

        # 'H3255 (L858R)' -> 'H32' (CVCL_E3Z2) — WRONG
        # This is NCI-H3255, an NSCLC line with EGFR L858R mutation.
        'H3255 (L858R)': 'NCI-H3255',  # CVCL_5194

        # 'HBL1' -> 'HBL-1 [Human AIDS-related non-Hodgkin lymphoma]' (CVCL_M572) — WRONG
        # In PROTAC/BTK literature, HBL-1 is the DLBCL (ABC subtype) line.
        # CVCL_4213 = HBL-1 [Human diffuse large B-cell lymphoma]
        'HBL1': 'HBL-1 [Human diffuse large B-cell lymphoma]',  # CVCL_4213

        # HCC827 harbors EGFR exon 19 deletion; the annotation is just metadata.
        'HCC827 (Exon 19 del)': 'HCC827',  # CVCL_2063

        # KG-1 is a human AML cell line.
        'KG1': 'KG-1',  # CVCL_0374

        # Same as above, just without the hyphen.
        'PC3': 'PC-3',  # CVCL_0035

        # SK-BR-3 is a HER2+ breast cancer line.
        'SKBr3 (HER2)': 'SK-BR-3',  # CVCL_0033

        # Parsing artifact; this is 22Rv1.
        'and 22Rv': '22Rv1',  # CVCL_1045

        # This is two cell lines jammed together. Map to MV4-11 as primary;
        # the splitting should happen upstream in your cell line parser.
        'MV-4-11 SK-MEL-5': 'MV4-11',  # CVCL_0064

        # 'HOP62/INC-H23' -> 'Ho' (CVCL_M698) — WRONG
        # Likely a typo for "HOP-62/NCI-H23". These are two NCI-60 lines.
        'HOP62/INC-H23': 'HOP-62',  # CVCL_1290

        'AML cells': 'AML-1',  # TODO: Needs paper-level resolution

        # This is a free-text description, not a cell line name.
        'Big sellection of cacer cell lines': None,

        # 'E14 mouse embryonic stem cells' -> 'CRL-6440' (CVCL_ZE35) — WRONG
        # E14 mESCs = E14Tg2a (CVCL_9108), a widely used 129-derived mESC line.
        'E14 mouse embryonic stem cells': 'ES-E14TG2a',  # CVCL_9108

        # Flag-Cdc20 is a tagged protein construct, not a cell line.
        # The host cell line depends on the paper. Map to None.
        'Flag-Cdc20': None,  # Tagged construct, not a cell line

        # Primary cells, not an established line.
        'Mice primary Sertoli cells': None,  # Primary cells

        # 'primary Germ cells' -> 'Ger' (CVCL_8353) — WRONG
        # Primary cells, not an established line. We might map 'germ' -> 'SCIT-C8'
        # which is questionable, though.
        'primary Germ cells': 'SCIT-C8',  # Primary cells
        '231MFP breast cancer cells': 'MDA-MB-231',  # CVCL_0062

        # 'MM1.SW' -> 'MM1.S' (CVCL_8792) — POSSIBLY WRONG
        # MM1.S-W is a dexamethasone-sensitive subline. Cellosaurus doesn't
        # have a separate entry; MM1.S (CVCL_8792) is acceptable.

        # It seems correct given the PROTAC literature context (BTK degraders tested
        # on DLBCL cell lines; HBL-1 is the canonical ABC-DLBCL line).

        # 'MM.lS' (typo in original data for MM.1S line)
        'MM.lS': 'MM1.S',  # CVCL_8792  (lowercase L mistaken for 1)

        # 'Panc-Tu-1' variant if it appears
        'Panc-Tu-1': 'PancTu-I',  # CVCL_4012
        'KYSE520': 'KYSE-520',  # CVCL_1355
        'Karpas 422': 'Karpas-422',  # CVCL_1325

        # 'SU-DHL-1' and 'SUDHL4' — already correct in output, but adding
        # common variants for robustness
        'SUDHL-1': 'SU-DHL-1',  # CVCL_0538
        'SUDHL4': 'SU-DHL-4',  # CVCL_0539
        'SU-DHL-1': 'SU-DHL-1',  # CVCL_0538  (identity, ensures no fuzzy mismatch)
    }

    unique_cell_lines = set()
    cell_line2comments = {}

    for reported_cell_line in protacpedia_df['Cell_Line'].dropna().unique():
        if reported_cell_line.strip() == 'MV-4-11 SK-MEL-5':
            print('-- Special case: splitting "MV-4-11 SK-MEL-5" into two cell lines ---')
            cell_lines = ['MV-4-11', 'SK-MEL-5']
        elif '; ' in reported_cell_line:  # Notice the space after the semicolon!!
            cell_lines = [cl.strip() for cl in reported_cell_line.split('; ')]
        elif reported_cell_line.endswith(','):
            cell_lines = [cl.strip() for cl in reported_cell_line[:-1].split(',')]
        elif ',' in reported_cell_line:
            cell_lines = [cl.strip() for cl in reported_cell_line.split(',')]
        elif ' and ' in reported_cell_line:
            cell_lines = [cl.strip() for cl in reported_cell_line.split(' and ')]
        elif '/' in reported_cell_line:
            cell_lines = [cl.strip() for cl in reported_cell_line.split('/')]
        else:
            cell_lines = [reported_cell_line.strip()]
        for cl in cell_lines:
            unique_cell_lines.add(cl)
            cell_line2comments.setdefault(cl, []).append(reported_cell_line)
        
    cell_line2cleaned = {}
    for cl in sorted(unique_cell_lines):
        clean_cl, clean_id = clean_cell_name(cl, cell_embedding, MANUAL_CELL_MAP, logger)
        cell_line2cleaned[cl] = (clean_cl, clean_id)
        if clean_cl is None:
            # logger.info(f"  WARNING: Could not clean cell line name '{cl}' from reported '{reported_cell_line}'")
            logger.warning(f" * WARNING: Failed '{cl}' - (From: '{cell_line2comments[cl]}')")
        else:
            # logger.info(f"'{reported_cell_line}' -> '{cl}' -> '{clean_cl}' ({clean_id})")
            # logger.info(f"'{cl}' -> '{clean_cl}' ({clean_id}) -> (From: '{cell_line2comments[cl]}')")
            logger.info(f"'{cl}' -> '{clean_cl}' ({clean_id})")

    # ## Duplicate Rows

    def pdc50_to_dc50(pdc50):
        if pd.isna(pdc50):
            return None
        try:
            return (10 ** (-pdc50)) * 1e9  # Convert M to nM
        except Exception as e:
            print(f"Error converting pDC50 to DC50 for value: {pdc50}")
            raise e


    def resolve_assay_time(
            time_entries, index, num_parsed, num_original_vals,
            default_standard_assay_time: float = 24,
    ):
        if not time_entries:
            return None
        if len(time_entries) == 1:
            return time_entries[0]
        total_entries = num_parsed + num_original_vals
        if len(time_entries) == total_entries:
            return time_entries[index % len(time_entries)]
        if default_standard_assay_time in time_entries:
            return default_standard_assay_time
        return max(time_entries)


    def parse_cell_lines(cell_line_str):
        if pd.isna(cell_line_str):
            return []
        cl = str(cell_line_str).strip()
        if not cl:
            return []
        if cl == 'MV-4-11 SK-MEL-5':
            return ['MV-4-11', 'SK-MEL-5']
        if '; ' in cl:
            return [c.strip() for c in cl.split('; ') if c.strip()]
        if cl.endswith(','):
            return [c.strip() for c in cl[:-1].split(',') if c.strip()]
        if ',' in cl:
            return [c.strip() for c in cl.split(',') if c.strip()]
        if ' and ' in cl:
            return [c.strip() for c in cl.split(' and ') if c.strip()]
        return [cl]


    curated_rows = []

    for _, row in protacpedia_df.iterrows():
        curator = row.get('Curator', 'Unknown')
        comment = row['Description']

        # ── 1. Parse assay times ──────────────────────────────────────
        time_entries = []
        if pd.notna(row.get('Assay_Time', np.nan)):
            time_str = str(row['Assay_Time']).replace(' ', '').replace('h', '')
            for t in time_str.split(','):
                try:
                    time_entries.append(float(t))
                except ValueError:
                    pass

        # ── 2. Parse cell lines ───────────────────────────────────────
        raw_cell_lines = parse_cell_lines(row.get('Cell_Line'))
        cleaned_cell_lines = []
        for cl in raw_cell_lines:
            if cl in cell_line2cleaned:
                cleaned_cell_lines.append(cell_line2cleaned[cl])
            else:
                cleaned_cell_lines.append(clean_cell_name(cl, cell_embedding, MANUAL_CELL_MAP, logger))
        multiple_cell_lines = len(cleaned_cell_lines) > 1

        # ── 3. Parse UniProts ─────────────────────────────────────────
        poi_str = row.get('POI_UniProt', '')
        if pd.isna(poi_str):
            poi_str = ''

        # Assign conflicting Uniprot IDs based on curator+POI_Name and other
        # contextual info, if available
        if (curator, poi_str) in MANUAL_POI_MAP:
            desc = comment if pd.notna(comment) else None
            cell = row['Cell_Line'] if pd.notna(row['Cell_Line']) else None
            dc50 = row['DC50'] if pd.notna(row['DC50']) else None
            dmax = row['Dmax'] if pd.notna(row['Dmax']) else None
            poi_mapped = MANUAL_POI_MAP.get((curator, poi_str), {}).get((desc, cell, dc50, dmax))
            if poi_mapped is not None:
                logger.info(f"Mapping POI_UniProt '{poi_str}' to '{poi_mapped}' for curator '{curator}' based on comment/cell/DC50/Dmax context")
            else:
                logger.warning(f"Could not map POI_UniProt '{poi_str}' for curator '{curator}' with comment/cell/DC50/Dmax context: {desc}")
            row['POI_UniProt'] = poi_mapped

        # ── 4. Parse original DC50/Dmax ───────────────────────────────
        parsed_values = {}
        num_original_vals = 0
        for value_col in ['DC50', 'Dmax']:
            if pd.notna(row.get(value_col)):
                parsed_values[value_col] = parse_single_value(row[value_col])
                num_original_vals += 1
            else:
                parsed_values[value_col] = None

        num_parsed_degradation = sum(
            1 for e in MANUAL_PARSED_COMMENTS.get(comment, [])
            if e.get('Value_Type') in {'DC50', 'Dmax', 'pDC50'}
        )

        # ── 5. Emit rows from original DC50/Dmax columns ─────────────
        original_entry_idx = 0
        for value_col in ['DC50', 'Dmax']:
            if parsed_values[value_col] is None:
                continue

            # Skip emitting the row if the comment includes references to
            # 'original' values: in this case, the comment will include the
            # columns
            entries = MANUAL_PARSED_COMMENTS.get(comment, [])
            entries = [e for e in entries if e.get('Value_Type') == value_col and e.get('Value') == 'original']
            if len(entries) > 0:
                continue

            curated_row = row.copy().to_dict()
            curated_row['Value_Type'] = value_col
            curated_row['Value'] = parsed_values[value_col]['mean']
            curated_row['Value_Unit'] = parsed_values[value_col]['unit']
            curated_row['Value_Operator'] = parsed_values[value_col]['operator']
            curated_row['Value_Category'] = 'numeric'
            curated_row['Description'] = comment

            # Cell line: both DC50 and Dmax come from same experiment
            if cleaned_cell_lines:
                curated_row['Cell_Line'] = cleaned_cell_lines[0][0]
                curated_row['Cell_Line_ID'] = cleaned_cell_lines[0][1]
            else:
                curated_row['Cell_Line'] = None
                curated_row['Cell_Line_ID'] = None
            curated_row['Unclear_Cell_Line'] = multiple_cell_lines

            curated_row['Assay_Time'] = resolve_assay_time(
                time_entries, original_entry_idx,
                num_parsed_degradation, num_original_vals,
            )
            
            curated_row['Manually_Curated'] = False

            curated_rows.append(curated_row)
            original_entry_idx += 1

        # ── 6. Emit rows from parsed comments ────────────────────────
        if comment in MANUAL_PARSED_COMMENTS:
            for i, entry in enumerate(MANUAL_PARSED_COMMENTS[comment]):
                if entry['Value_Type'] not in {'DC50', 'Dmax', 'pDC50'}:
                    continue

                curated_row = row.to_dict()
                
                # Overwrite with parsed values from comment (e.g. Cell_Line,
                # Assay, POI_Name) which may be more specific than the row-level
                # values. The Value/Value_Type columns will be overwritten below
                curated_row.update(entry)

                # Time
                if time_entries and pd.isna(curated_row.get('Assay_Time')):
                    curated_row['Assay_Time'] = resolve_assay_time(
                        time_entries, num_original_vals + i,
                        num_parsed_degradation, num_original_vals,
                    )

                # pDC50 → DC50
                if entry['Value_Type'] == 'pDC50':
                    curated_row['Value_Type'] = 'DC50'
                    if entry['Value'] == 'original':
                        if pd.notna(row.get('pDC50')):
                            pdc50_val = row['pDC50']
                            if isinstance(pdc50_val, str):
                                pdc50_val = float(pdc50_val)
                            curated_row['Value'] = pdc50_to_dc50(pdc50_val)
                            curated_row['Value_Unit'] = 'nM'
                            if parsed_values.get('DC50'):
                                curated_row['Value_Operator'] = parsed_values['DC50']['operator']
                        else:
                            curated_row['Value'] = None
                    else:
                        curated_row['Value'] = pdc50_to_dc50(entry['Value'])
                        curated_row['Value_Unit'] = 'nM'
                        curated_row['Value_Operator'] = entry.get('Value_Operator', '')

                # 'original' value
                elif entry['Value'] == 'original':
                    vtype = entry['Value_Type']
                    if parsed_values.get(vtype) is not None:
                        curated_row['Value'] = parsed_values[vtype]['mean']
                        curated_row['Value_Unit'] = parsed_values[vtype]['unit']
                        curated_row['Value_Operator'] = parsed_values[vtype]['operator']
                    else:
                        curated_row['Value'] = None

                # Range → mean
                if entry.get('Value_Category') == 'range' and entry['Value'] is None:
                    range_min = entry['Value_Range_Min']
                    range_max = entry['Value_Range_Max']
                    curated_row['Value'] = (range_min + range_max) / 2
                    curated_row['Value_Mean'] = (range_min + range_max) / 2

                if curated_row['Value'] is None:
                    print(f"WARNING: Could not assign value for entry from comment: '{comment}...' entry: {entry}")

                # Cell line
                if entry.get('Cell_Line') is not None:
                    cl_name, cl_id = clean_cell_name(entry['Cell_Line'], cell_embedding, MANUAL_CELL_MAP, logger)
                    curated_row['Cell_Line'] = cl_name
                    curated_row['Cell_Line_ID'] = cl_id
                    curated_row['Unclear_Cell_Line'] = False
                elif cleaned_cell_lines:
                    curated_row['Cell_Line'] = cleaned_cell_lines[0][0]
                    curated_row['Cell_Line_ID'] = cleaned_cell_lines[0][1]
                    curated_row['Unclear_Cell_Line'] = (len(cleaned_cell_lines) > 1)
                else:
                    curated_row['Cell_Line'] = None
                    curated_row['Cell_Line_ID'] = None
                    curated_row['Unclear_Cell_Line'] = False

                # POI_Name → POI_UniProt resolution: if a comment specifies a
                # POI_Name, this is likely more accurate than the row-level
                # POI_UniProt annotation which may be missing or incorrect.
                # So we attempt to resolve the POI_Name to a UniProt ID and
                # overwrite the row-level POI_UniProt with the resolved value.
                if entry.get('POI_Name') is not None:
                    poi_uniProt = gene2uniprot.get(entry['POI_Name'])
                    if poi_uniProt is None:
                        logger.warning(f"WARNING: Could not resolve POI_Name '{entry['POI_Name']}' to UniProt ID for comment: '{comment[:60]}...'")
                    curated_row['POI_UniProt'] = poi_uniProt
                
                curated_row['Manually_Curated'] = True
                curated_rows.append(curated_row)

    curated_df = pd.DataFrame(curated_rows)
    curated_df['Modality'] = 'PROTAC'

    # Assign all Dmax type entries the Value_Unit to '%'
    curated_df.loc[curated_df['Value_Type'] == 'Dmax', 'Value_Unit'] = '%'

    # Isolate relevant columns only
    curated_df = curated_df[[
        'SMILES',
        'Ligase_Name',
        'POI_Name',
        'POI_UniProt',
        'Unclear_POI',
        'Value',
        'Value_Type',
        'Value_Unit',
        'Value_Operator',
        'Cell_Line',
        'Cell_Line_ID',
        'Unclear_Cell_Line',
        'Assay',
        'Assay_Time',
        'Value_Category',
        'Value_Range_Min',
        'Value_Range_Max',
        'Value_Error',
        'Value_Concentration',
        'Value_Concentration_Unit',
        'Value_Mean',
        'Reference',
        'Description',
        'Modality',
        'Manually_Curated',
    ]]

    # Convert all molar values to nM for consistency
    def convert_to_nM(row):
        if row['Value_Type'] in {'DC50', 'EC50', 'IC50'} and pd.notna(row['Value']) and pd.notna(row['Value_Unit']):
            try:
                if row['Value_Unit'] == 'M':
                    return row['Value'] * 1e9
                elif row['Value_Unit'] == 'uM':
                    return row['Value'] * 1e3
                elif row['Value_Unit'] == 'nM':
                    return row['Value']
                elif row['Value_Unit'] == 'pM':
                    return row['Value'] * 1e-3
                else:
                    print(f"WARNING: Unrecognized {row['Value_Type']} unit '{row['Value_Unit']}' for value: {row['Value']} in comment: '{row['Description']}'")
                    return row['Value']
            except Exception as e:
                print(f"Error converting value to nM for value: {row['Value']} with unit: {row['Value_Unit']} in comment: '{row['Description']}'")
                raise e
        else:
            return row['Value']

    curated_df['Value'] = curated_df.apply(convert_to_nM, axis=1)
    # Convert all DC50 units to nM
    curated_df.loc[curated_df['Value_Type'] == 'DC50', 'Value_Unit'] = 'nM'

    # Rename POI_Name values for consistency
    curated_df['POI_Name'] = curated_df['POI_Name'].replace({
        'BCLXL': 'BCL-XL',
        'BTK WT': 'BTK',
        'Brd2': 'BRD2',
        'Brd3': 'BRD3',
        'Brd4': 'BRD4',
        'Brd4 long': 'BRD4 LONG',
        'Brd4 short': 'BRD4 SHORT',
    })
    logger.info(curated_df['POI_Name'].value_counts())

    # ## Assign Species

    # Get all Cell_Line_Species
    curated_df['Cell_Line_Species'] = curated_df['Cell_Line'].apply(lambda cl: get_cell_species(cl, cell_embedding))
    logger.info(curated_df['Cell_Line_Species'].value_counts())

    # Replace Ligase_Name with standardized ones
    curated_df['Ligase_Name'] = curated_df['Ligase_Name'].replace({
        'Cereblon': 'CRBN',
        'Mdm2': 'MDM2',
        'Iap': 'IAP',
        'Ubr1': 'UBR1', # Typo in original data
    })

    # Map E3 ligases to UniProt IDs
    def get_e3_uniprot(row: pd.Series) -> Optional[str]:
        e3 = row.get('Ligase_Name')
        species = row.get('Cell_Line_Species')
        if pd.isna(e3):
            raise ValueError(f"Missing Ligase_Name in row with comment: '{row['Description'][:60]}...'")
        
        e3_mapping = E3_TO_ORGANISM_TO_UNIPROT.get(species)
        if e3_mapping is None:
            # Get the default human mapping
            e3_mapping = E3_TO_ORGANISM_TO_UNIPROT.get('Homo sapiens', {})

        e3_uniprot = e3_mapping.get(e3)
        if e3_uniprot is None:
            raise ValueError(f"E3 Ligase '{e3}' is not in the known ones!")
        
        return e3_uniprot

    curated_df['Ligase_UniProt'] = curated_df.apply(get_e3_uniprot, axis=1)
    logger.info(curated_df['Ligase_UniProt'].value_counts())

    def update_uniprot(row, logger):
        infos = fetch_uniprot_for_gene(row['POI_Name'], row['Cell_Line_Species'])
        new_uniprot = infos['uniprot'] if infos is not None else row['POI_UniProt']
        if new_uniprot is not None and new_uniprot != row['POI_UniProt']:
            logger.warning(f"Updating Uniprot {row['POI_UniProt']} with {new_uniprot}, for organism: {row['Cell_Line_Species']}")
        return new_uniprot

    curated_df['POI_UniProt'] = curated_df.apply(lambda x: update_uniprot(x, logger), axis=1)
    uniprots_to_fetch = list(curated_df['POI_UniProt'].unique()) + list(curated_df['Ligase_UniProt'].unique())

    # Update dictionaries with newly found Uniprots
    for uniprot_id in tqdm(uniprots_to_fetch, desc='Fetching UniProt entries'):
        json_info = load_dict(data_curation_dir / 'uniprot_infos' / f'{uniprot_id}.json')
        if json_info:
            uniprot2info[uniprot_id] = json_info
            uniprot2gene[uniprot_id] = json_info['gene_primary']
            uniprot2seq[uniprot_id] = json_info['sequence']
            gene2uniprot[json_info['gene_primary']] = uniprot_id
            for isoform in json_info.get('isoforms', []):
                uniprot_id = isoform['accession']
                # NOTE: We do not add isoforms to uniprot2gene since they share the
                # same gene name
                uniprot2info[uniprot_id] = isoform
                uniprot2seq[uniprot_id] = isoform['sequence']
        else:
            infos = fetch_protein_info(uniprot_id, skip_isoforms=False)
            if infos:
                # Save each entry to a separate JSON file
                save_dict(infos, data_curation_dir / 'uniprot_infos' / f'{uniprot_id}.json')
                uniprot2gene[uniprot_id] = infos['gene_primary']
                uniprot2seq[uniprot_id] = infos['sequence']
                uniprot2info[uniprot_id] = infos
                gene2uniprot[infos['gene_primary']] = uniprot_id
                for isoform in infos.get('isoforms', []):
                    uniprot_id = isoform['accession']
                    save_dict(isoform, data_curation_dir / 'uniprot_infos' / f'{uniprot_id}.json')
                    uniprot2gene[uniprot_id] = isoform['gene_primary']
                    uniprot2seq[uniprot_id] = isoform['sequence']
            else:
                uniprot2gene[uniprot_id] = None

    # Assign missing POI_Name values based on UniProt ID
    def assign_poi_name(row):
        if pd.isna(row['POI_Name']) and pd.notna(row['POI_UniProt']):
            uniprot_id = row['POI_UniProt']
            gene_name = uniprot2gene.get(uniprot_id)
            if gene_name:
                return gene_name
        return row['POI_Name']

    curated_df['POI_Name'] = curated_df.apply(assign_poi_name, axis=1)

    # Clean 'BRD4 LONG' and 'BRD4 SHORT' entries with their isoforms (based on POI_UniProt)
    def assign_brd4_isoform(row):
        if pd.notnull(row['POI_Name']) and 'BRD4' in row['POI_Name']:
            if 'LONG' in row['POI_Name']:
                return row['POI_UniProt'] + '-1'
            elif 'SHORT' in row['POI_Name']:
                return row['POI_UniProt'] + '-2'
        return row['POI_UniProt']

    curated_df['POI_UniProt'] = curated_df.apply(assign_brd4_isoform, axis=1)

    # ## Get Sequences and Apply Mutations

    curated_df['POI_Sequence'] = curated_df['POI_UniProt'].apply(lambda uid: uniprot2seq.get(uid))
    curated_df['Ligase_Sequence'] = curated_df['Ligase_UniProt'].apply(lambda uid: uniprot2seq.get(uid))

    curated_df['POI_Sequence'] = curated_df.apply(lambda row: apply_mutation(row['POI_Sequence'], row['POI_Name']), axis=1)

    curated_df = curated_df.dropna(subset=['Value'])

    # ## Resolve Duplicates

    def resolve_duplicates(df, duplicate_subset):
        """ Resolve duplicates by keeping all manually curated entries and dropping
            automatically parsed ones only if there is a manually curated entry in
            the same group. """
        df = df.copy()

        df['_has_manual'] = df.groupby(duplicate_subset, dropna=False)['Manually_Curated'].transform('any')
        df['_keep'] = ~df['_has_manual'] | df['Manually_Curated']

        resolved = df[df['_keep']].drop(columns=['_has_manual', '_keep'])
        return resolved

    duplicate_subset = ['SMILES', 'Cell_Line', 'Ligase_Name', 'POI_Name', 'POI_UniProt', 'Value_Type', 'Assay', 'Assay_Time']
    curated_df = resolve_duplicates(curated_df, duplicate_subset)

    # Check the amount of missing data in each column
    missing_data = curated_df.isna().sum()
    logger.info("Missing data counts per column:")
    logger.info('\n' + str(missing_data))

    # Convert all None values to NaN for consistency
    curated_df = curated_df.where(pd.notnull(curated_df), None)

    # ## Save to CSV
    curated_df.to_csv(data_curation_dir / 'protacpedia_protac_dc50_dmax.csv', index=False)

    print('-' * 80)
    print(f"Logs saved to: {log_file}")
    print('-' * 80)


    # import numpy as np
    # import matplotlib.pyplot as plt
    # import seaborn as sns

    # # Plot the distribution of DC50 and Dmax values

    # def dc50_to_pdc50(dc50):
    #     if pd.isna(dc50):
    #         return None
    #     try:
    #         return -np.log10(dc50 * 1e-9)  # Convert nM to M and then to pDC50
    #     except Exception as e:
    #         print(f"Error converting DC50 to pDC50 for value: {dc50}")
    #         raise e

    # plt.figure(figsize=(12, 6))
    # sns.histplot([dc50_to_pdc50(x) for x in curated_df[curated_df['Value_Type'] == 'DC50']['Value'].dropna()], bins=30, kde=True)
    # plt.title('Distribution of pDC50 Values')
    # plt.xlabel('pDC50 (-Log10(nM))')
    # plt.ylabel('Frequency')
    # plt.show()

    # # Plot the distribution of Dmax values
    # plt.figure(figsize=(12, 6))
    # sns.histplot(curated_df[curated_df['Value_Type'] == 'Dmax']['Value'].dropna(), bins=30, kde=True)
    # plt.title('Distribution of Dmax Values')
    # plt.xlabel('Dmax (%)')
    # plt.ylabel('Frequency')
    # plt.show()


if __name__ == "__main__":
    main()