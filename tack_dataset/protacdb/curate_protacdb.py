import re
import os
import logging
import warnings
import requests
import random
import argparse
from pathlib import Path
from itertools import zip_longest

import pandas as pd
from rdkit import RDLogger
from tqdm.auto import tqdm
from thefuzz import process

from tack_dataset.logging_utils import setup_logging, set_global_logging_level
from tack_dataset.curation_utils import canonicalize_smiles
from tackai.data.embeddings.cell_embeddings import (
    CellEmbedding,
)
from tack_dataset.protacdb.assay_cleaning import (
    split_clean_str,
    parse_assay_dmax_dc50,
    extract_inhibition_info,
    extract_protac2target_ic50,
    extract_degradation_information,
)
from tack_dataset.protacdb.protein_utils import (
    map_poi_sequence_from_uniprot,
    fetch_protein_info,
    clean_target,
    assign_missing_uniprot,
    update_e3ligase_uniprot,
    update_e3ligase_sequence,
    get_poi_species,
    get_sequence_from_uniprot,
    apply_mutation_to_sequence,
    map_poi_uniprot_from_species,
)
from tack_dataset.cell_utils import (
    standardize_cell_line_protacdb,
    get_cell_species,
    get_manual_cell_mapping,
)
from tack_dataset.protacdb.utils import (
    save_dict,
    load_dict,
    iterate_dict_lists,
    get_assay_type,
)


def main():
    parser = argparse.ArgumentParser(description='Curate PROTAC-DB dataset.')
    parser.add_argument('--input_path', type=str, default=Path('data/original/PROTAC-DB.csv'), help='Path to the raw PROTAC-DB CSV file.')
    parser.add_argument('--input_dir', type=str, default=Path('data/original'), help='Directory containing the raw PROTAC-DB CSV file (alternative to --input_path).')
    parser.add_argument('--output_dir', type=str, default=Path('data/curation'), help='Directory to save the curated dataset and intermediate files.')
    parser.add_argument('--force_refetch', action='store_true', help='Force refetching of UniProt entries even if cached files exist.')
    parser.add_argument('--log_dir', type=str, default=Path('logs'), help='Directory to save log files.')
    parser.add_argument('--verbose', '-v', action='count', default=0, help='Increase output verbosity (e.g. -v for INFO, -vv for DEBUG, -vvv for more detailed DEBUG).')

    args = parser.parse_args()

    # Setup logging
    log_file = setup_logging(args.log_dir, log_base_name='protacdb_curation', verbose=args.verbose)
    set_global_logging_level(logging.DEBUG if args.verbose >= 3 else logging.INFO if args.verbose == 2 else logging.WARNING)
    logger = logging.getLogger(__name__)

    targets_w_no_uniprot_mapping = {}
    target2uniprots = {}
    e3ligase2uniprot = {}
    uniprot2infos = {}
    species2uniprot = {}
    uniprot2locations = {}

    # Filter out some warnings...
    RDLogger.DisableLog('rdApp.*')
    warnings.filterwarnings('ignore')

    # ### Download Raw Data
    # --------------------------------------------------------------------------

    # Setup working directories
    data_raw_dir = Path(args.input_dir)
    data_curation_dir = Path(args.output_dir)
    uniprot_dir = data_curation_dir / 'uniprot_infos'
    os.makedirs(data_raw_dir, exist_ok=True)
    os.makedirs(data_curation_dir, exist_ok=True)
    os.makedirs(uniprot_dir, exist_ok=True)

    # Download or load the raw PROTAC-DB dataset:
    if args.input_path is not None:
        protacdb_file = args.input_path
    else:
        protacdb_file = data_raw_dir / 'PROTAC-DB.csv'
    protacdb_url = 'http://cadd.zju.edu.cn/protacdb/statics/binaryDownload/csv/protac/protac.csv'

    if os.path.exists(protacdb_file):
        protacdb_df = pd.read_csv(protacdb_file).reset_index(drop=True)
    else:
        logger.info(f'Downloading {protacdb_url}')
        response = requests.get(protacdb_url)
        with open(protacdb_file, 'wb') as f:
            f.write(response.content)
        protacdb_df = pd.read_csv(protacdb_file).reset_index(drop=True)
    logger.info('PROTAC-DB loaded.')

    # ==========================================================================
    # ## Canonize SMILES
    # ==========================================================================
    protacdb_df = protacdb_df.rename(columns={'E3 ligase': 'E3 Ligase'})
    protacdb_df['Smiles'] = protacdb_df['Smiles'].map(canonicalize_smiles)

    # ==========================================================================
    # ## Define Assay-Related Columns
    # ==========================================================================
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

    non_assay_cols = list(set(protacdb_df) - set(assay_cols))
    logger.debug(f'Assay related columns: {assay_cols}')
    logger.debug(f'Non-assay related columns:')
    for c in non_assay_cols:
        logger.debug(f'  - {c}')

    # Sort the assay_to_val_cols dictionary by the length of the dataframe of
    # unique values in the keys and parameters columns
    assay_to_val_cols = dict(sorted(
        assay_to_val_cols.items(),
        key=lambda item: len(protacdb_df[[item[0]] + item[1]].dropna(how='all').drop_duplicates()),
        reverse=True
    ))

    for assay, cols in assay_to_val_cols.items():
        tmp = protacdb_df[[assay] + cols].dropna(how='all').drop_duplicates()
        logger.debug('-' * 100)
        logger.debug(f'Assay: {assay}')
        logger.debug(f'Number of (unique) rows: {len(tmp)}')
        logger.debug('-' * 100)
        logger.debug('\n' + str(tmp.sample(n=min(10, len(tmp))).to_markdown(index=False)))

    # ==========================================================================
    # We define a specific parsing function for each assay-related column.
    # ==========================================================================
    parsing_functions = {
        'Assay (DC50/Dmax)': parse_assay_dmax_dc50,
        'Assay (Cellular activities, IC50)': extract_inhibition_info,
        'Assay (Protac to Target, IC50)': extract_protac2target_ic50,
        'Assay (Percent degradation)': extract_degradation_information,
    }

    # Debug level: print sample assay comments and their parsed results
    if args.verbose >= 2:
        for assay_col in parsing_functions.keys():
            if assay_col not in protacdb_df.columns:
                logger.warning(f'Warning: {assay_col} column not found in the dataset. Skipping parsing for this assay.')
                continue
            logger.debug(f'Sample assay comments for column "{assay_col}":')
            assay_comments = protacdb_df[assay_col].dropna().unique().tolist()
            sample_comments = random.sample(assay_comments, min(10, len(assay_comments)))
            for c in sample_comments:
                logger.debug(f"TEXT: {c}")
                logger.debug(parsing_functions[assay_col](c))
                logger.debug('')
                logger.debug("-" * 50)

    df_dict = {}

    # ==========================================================================
    # Parse Assay (DC50/Dmax)
    # ==========================================================================
    # Remove rows with NaN in "Assay (DC50/Dmax)" and all value columns
    assay_col = "Assay (DC50/Dmax)"
    val_cols = assay_to_val_cols[assay_col]
    dc50_dmax_df = protacdb_df.dropna(subset=[assay_col] + val_cols, how='all')

    parsed_table = []
    for _, row in tqdm(dc50_dmax_df.iterrows(), total=len(dc50_dmax_df), desc='Extracting DC50/Dmax info'):
        assay = row[assay_col]
        dc50_val = row['DC50 (nM)']
        dmax_val = row['Dmax (%)']

        extracted_info = parsing_functions[assay_col](assay)
        extracted_info = {
            'Target (DC50/Dmax)': extracted_info.get('targets'),
            'Cell Type (DC50/Dmax)': extracted_info.get('cells'),
            'Treatment Time (h) (DC50/Dmax)': extracted_info.get('times'),    
        }

        targets = extracted_info['Target (DC50/Dmax)']
        cells = extracted_info['Cell Type (DC50/Dmax)']
        treatment_times = extracted_info['Treatment Time (h) (DC50/Dmax)']
        dc50_vals = split_clean_str(row['DC50 (nM)'])
        dmax_vals = split_clean_str(row['Dmax (%)'])
        
        # If any of the DC50 is zero, print the whole row for manual checking
        if dc50_vals is not None and any(val['mean'] == 0 for val in dc50_vals if val is not None):
            logger.info(f'Zero DC50 value found in row:\n• DC50 row: {row["DC50 (nM)"]}\n• Dmax row: {row["Dmax (%)"]}\n• DC50 values: {dc50_vals}\n• Dmax values: {dmax_vals}')
            logger.info(f'https://doi.org/{row["Article DOI"]}')

        # if 'N.D.' in str(row['DC50 (nM)']):
        #     logger.info(f'https://doi.org/{doi} has N.D. in DC50 (nM) column.\n - DC50: {row["DC50 (nM)"]}\n - Dmax: {row["Dmax (%)"]}')
        # if 'N.D.' in str(row['Dmax (%)']):
        #     logger.info(f'https://doi.org/{doi} has N.D. in Dmax (%) column.\n - DC50: {row["DC50 (nM)"]}\n - Dmax: {row["Dmax (%)"]}')
        
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

            if dc50_val is not None:
                new_row['DC50 (nM) Value (DC50/Dmax)'] = dc50_val['mean']
                new_row['DC50 (nM) Error (DC50/Dmax)'] = dc50_val['error']
                new_row['DC50 (nM) Unit (DC50/Dmax)'] = dc50_val['unit']
                new_row['DC50 (nM) Operator (DC50/Dmax)'] = dc50_val['operator']

                # Replace zero values in dc50_vals with NaN
                if dc50_val['mean'] == 0:
                    logger.warning(f'Warning: DC50 value is zero for assay: "{assay}"')
                    new_row['DC50 (nM) Value (DC50/Dmax)'] = pd.NA

            # If the Dmax value is above 100%, we cap it at 100.0 and issue a
            # warning.
            if dmax_val is not None:
                if dmax_val["mean"] > 100:
                    logger.warning(f'Warning: Dmax value {dmax_val["mean"]} for assay: "{assay}" is greater than 100, setting to 100.0 instead.')
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
    protacdb_dc50dmax_df['Database'] = 'PROTAC-DB'
    df_dict['(DC50/Dmax)'] = protacdb_dc50dmax_df.copy()

    logger.debug('\n' + str(protacdb_dc50dmax_df.head(50)[['Assay (DC50/Dmax)', 'DC50 (nM)', 'DC50 (nM) Value (DC50/Dmax)', 'Dmax (%)', 'Dmax (%) Value (DC50/Dmax)', 'Target (DC50/Dmax)', 'Cell Type (DC50/Dmax)', 'Treatment Time (h) (DC50/Dmax)']]))
    logger.info(f'Parsed table len: {len(protacdb_dc50dmax_df)}')

    # ==========================================================================
    # Parse Assay (Cellular activities, IC50)
    # ==========================================================================
    assay_col = "Assay (Cellular activities, IC50)"
    assay_name = assay_col.split('(')[-1].split(')')[0].strip()
    val_cols = assay_to_val_cols[assay_col]
    cell_ic50_df = protacdb_df.dropna(subset=[assay_col] + val_cols, how='all')

    parsed_table = []
    for _, row in tqdm(cell_ic50_df.iterrows(), total=len(cell_ic50_df), desc='Extracting info'):
        assay = row[assay_col]

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
    # Rename 'IC50 (nM, Cellular activities)' to 'IC50 (nM) (Cellular activities, IC50)'
    protacdb_cell_ic50_df = protacdb_cell_ic50_df.rename(
        columns={'IC50 (nM, Cellular activities)': 'IC50 (nM) (Cellular activities, IC50)'}
    )
    df_dict['(Cellular activities, IC50)'] = protacdb_cell_ic50_df.copy()

    logger.info(f'Parsed table len: {len(protacdb_cell_ic50_df):,}')
    logger.info(protacdb_cell_ic50_df.sample(n=10))

    # ==========================================================================
    # Parse Assay (Protac to Target, IC50)
    # ==========================================================================
    assay_col = "Assay (Protac to Target, IC50)"
    assay_name = assay_col.split('(')[-1].split(')')[0].strip()
    val_cols = assay_to_val_cols[assay_col]
    protac_ic50_df = protacdb_df.dropna(subset=[assay_col] + val_cols, how='all')

    parsed_table = []
    for _, row in tqdm(protac_ic50_df.iterrows(), total=len(protac_ic50_df), desc='Extracting info'):
        assay = row[assay_col]

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
    
    # Rename 'IC50 (nM, Protac to Target)' to 'IC50 (nM) (Protac to Target, IC50)'
    protacdb_protac_ic50_df = protacdb_protac_ic50_df.rename(
        columns={'IC50 (nM, Protac to Target)': 'IC50 (nM) (Protac to Target, IC50)'}
    )

    df_dict['(Protac to Target, IC50)'] = protacdb_protac_ic50_df.copy()

    logger.info(f'Parsed table len: {len(protacdb_protac_ic50_df):,}')
    logger.info(protacdb_protac_ic50_df.sample(n=10))

    # ==========================================================================
    # Parse Assay (Percent degradation)
    # ==========================================================================
    # Remove rows with NaN in "Assay (Percent degradation)" and all value columns
    assay_col = "Assay (Percent degradation)"
    assay_name = 'Percent degradation'
    val_cols = assay_to_val_cols[assay_col]
    degr_df = protacdb_df.dropna(subset=[assay_col] + val_cols, how='all')

    parsed_table = []
    for _, row in tqdm(degr_df.iterrows(), total=len(degr_df), desc='Extracting info'):
        assay = row[assay_col]
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
    df_dict['(Percent degradation)'] = protacdb_degr_df.copy()

    logger.info(f'Parsed table len: {len(protacdb_degr_df):,}')
    logger.info(protacdb_degr_df.sample(n=10))

    # --------------------------------------------------------------------------
    # Special handling of MEK1/2 entries in the (Percent degradation) assay
    # --------------------------------------------------------------------------
    # As far as I understood it, according to the work reported at this
    # [DOI](https://pubs.acs.org/doi/full/10.1021/acsmedchemlett.2c00446),
    # entries with target equal to `MEK1/2` should be split into two separate
    # entries, one for `MEK1` and one for `MEK2`. The `DC (nM)` and
    # `Percent degradation (%)` values are therefore duplicated and are set the
    # same for both entries.

    # Duplicate all the rows with f'Target ({assay_name})'] == 'MEK1/2', the duplicate rows should have the f'Target ({assay_name})'] set to 'MEK1' and 'MEK2' respectively
    mek1_df = protacdb_degr_df[protacdb_degr_df[f'Target ({assay_name})'] == 'MEK1/2'].copy()
    mek2_df = mek1_df.copy()
    mek1_df[f'Target ({assay_name})'] = 'MEK1'
    mek2_df[f'Target ({assay_name})'] = 'MEK2'

    mek1_2_df = pd.concat([mek1_df, mek2_df], ignore_index=True).drop_duplicates()

    logger.info(f'Parsed table len before removing MEK1/2: {len(protacdb_degr_df):,}')
    logger.info(f"MEK1: {len(mek1_df)}\tTargets: {mek1_df[f'Target ({assay_name})'].unique().tolist()}")
    logger.info(f"MEK2: {len(mek2_df)}\tTargets: {mek2_df[f'Target ({assay_name})'].unique().tolist()}")
    logger.info(f"Concatenated MEK1/2: {len(mek1_2_df)}\tTargets: {mek1_2_df[f'Target ({assay_name})'].unique().tolist()}")

    # Remove the original rows with f'Target ({assay_name})'] == 'MEK1/2'
    protacdb_degr_df = protacdb_degr_df[protacdb_degr_df[f'Target ({assay_name})'] != 'MEK1/2']
    logger.info(f'Parsed table len after removing MEK1/2: {len(protacdb_degr_df):,}')
    logger.info(sorted(protacdb_degr_df[f'Target ({assay_name})'].unique().tolist()))

    protacdb_degr_df = pd.concat([protacdb_degr_df, mek1_2_df], ignore_index=True).drop_duplicates()

    logger.info(f'Parsed table len after concatenating: {len(protacdb_degr_df):,}')
    logger.info(sorted(protacdb_degr_df[f'Target ({assay_name})'].unique().tolist()))

    # ==========================================================================
    # Common cleaning and formatting for all parsed dataframes
    # ==========================================================================
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
        logger.debug(f"Dataframe name: {df_name}")
        for col in df.columns:
            logger.debug(f"- {col}")
        logger.debug('')

    # ==========================================================================
    # ## Clean E3 Ligases Names
    # ==========================================================================
    # We manually mapped E3 ligases names to their Uniprot IDs.
    # NOTE: We assume human proteins only, for now. Later further down, once we
    # have the cell lines curated, we will assign different species to the E3
    # ligases, if needed.
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

    for df_name, df in df_dict.items():
        df['E3 Ligase'] = df['E3 Ligase'].replace({'Keap1': 'KEAP1', 'BRD4': 'VHL'})
        df['E3 Ligase Uniprot'] = df['E3 Ligase'].map(e3ligase2uniprot)
        # Count the nan values in the 'E3 Ligase' and 'E3 Ligase Uniprot' columns
        e3_ligase_nan_count = df['E3 Ligase'].isna().sum()
        e3_ligase_uniprot_nan_count = df['E3 Ligase Uniprot'].isna().sum()
        logger.info(f"{df_name} - E3 Ligase NaN count: {e3_ligase_nan_count}, E3 Ligase Uniprot NaN count: {e3_ligase_uniprot_nan_count}")

    # ==========================================================================
    # ## Standardize Targets Names
    # ==========================================================================
    for df_name, df in df_dict.items():
        target_cols = [col for col in df.columns if col.startswith('Target')]
        for col in target_cols:
            tqdm.pandas(desc=f'Cleaning targets in {df_name}.{col}')
            df[col] = df[col].progress_apply(clean_target)

    # ==========================================================================
    # ## Get AA Sequences
    # ==========================================================================
    # We use the UniprotKT rest API to fetch information about the proteins. The
    # following function fetches a JSON object for a given Uniprot ID that
    # contains a vast amount of information about the protein, including its
    # sequence, names, and other relevant data.

    uniprots = []
    e3_uniprots = []
    for df_name, df in df_dict.items():
        uniprots += df['Uniprot'].dropna().unique().tolist()
        e3_uniprots += df['E3 Ligase Uniprot'].dropna().unique().tolist()

    uniprots = list(set(uniprots))
    e3_uniprots = list(set(e3_uniprots))

    uniprot2infos = {}
    for uniprot_id in tqdm(uniprots, desc='Fetching UniProt entries'):
        json_info = load_dict(os.path.join(uniprot_dir, f'{uniprot_id}.json'))
        if json_info:
            uniprot2infos[uniprot_id] = json_info
        else:
            info = fetch_protein_info(uniprot_id)
            if info:
                uniprot2infos[uniprot_id] = info
                # Save each entry to a separate JSON file
                save_dict(info, os.path.join(uniprot_dir, f'{uniprot_id}.json'))

    for e3_uniprot_id in tqdm(e3_uniprots, desc='Fetching E3 ligase UniProt entries'):
        json_info = load_dict(os.path.join(uniprot_dir, f'{e3_uniprot_id}.json'))
        if json_info:
            uniprot2infos[e3_uniprot_id] = json_info
        else:
            info = fetch_protein_info(e3_uniprot_id)
            if info:
                uniprot2infos[e3_uniprot_id] = info
                # Save each entry to a separate JSON file
                save_dict(info, os.path.join(uniprot_dir, f'{e3_uniprot_id}.json'))

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

    # ==========================================================================
    # ### Assign Sequences to Targets
    # ==========================================================================
    # We start by getting all Uniprot IDs and their associated targets. We do
    # the same for the targets and their associated Uniprot IDs. This is done to
    # ensure that we have all the necessary information to fetch the sequences.

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

    logger.info(f'Unique targets: {len(target2uniprots):,}')
    logger.info(f'Unique Uniprots: {len(uniprot2targets):,}')

    # Print the maximum number of targets associated with a single Uniprot ID
    max_targets = max(len(targets) for targets in uniprot2targets.values())
    logger.info(f'Maximum number of targets associated with a single Uniprot ID: {max_targets}')
    # Print the maximum number of Uniprot IDs associated with a single target
    max_uniprots = max(len(uniprots) for uniprots in target2uniprots.values())
    logger.info(f'Maximum number of Uniprot IDs associated with a single target: {max_uniprots}')

    # Sort uniprot2targets and target2uniprots alphabetically on the keys
    uniprot2targets = dict(sorted(uniprot2targets.items()))
    target2uniprots = dict(sorted(target2uniprots.items()))

    logger.info(f'Number of unique targets: {len(target2uniprots):,}')

    save_dict(uniprot2targets, data_curation_dir / 'uniprot2targets.json')
    save_dict(target2uniprots, data_curation_dir / 'target2uniprots.json')

    # Assign the respective 'Article DOI' to each target
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

    logger.info(f'Number of unique targets with DOIs: {len(target2dois):,}')

    # ==========================================================================
    # ### Divide Targets into Unique and Multiple Uniprot IDs
    # ==========================================================================
    # There are some targets that have missing or multiple Uniprot IDs
    # associated with them. The following code tries to identify these targets
    # and print them out for further inspection.

    # Get all pair of entries in uniprot2targets and target2uniprots for which both values have length == 1
    targets_w_unique_uniprot = []
    for uniprot_id, targets in uniprot2targets.items():
        if len(targets) == 1:
            target = targets[0]
            if target in target2uniprots and len(target2uniprots[target]) == 1:
                if not pd.isnull(uniprot_id) and not pd.isnull(target):
                    targets_w_unique_uniprot.append(target)
                    logger.debug(f"{uniprot_id} -> {target}")
    logger.info(f"Number of 1-to-1 pairs: {len(targets_w_unique_uniprot)} ({len(targets_w_unique_uniprot) / len(target2uniprots):.2%} of targets)")

    targets_w_many_uniprots = []
    for target, uniprots in target2uniprots.items():
        if len(uniprots) > 1:
            targets_w_many_uniprots.append(target)
            logger.info(f"{target} -> {', '.join(uniprots)} [{', '.join(target2dois.get(target, []))}]")
    logger.info(f"Number of targets with multiple Uniprot IDs: {len(targets_w_many_uniprots)} ({len(targets_w_many_uniprots) / len(target2uniprots):.2%} of targets)")

    targets_w_mutations = []
    for target, uniprots in target2uniprots.items():
        if target in targets_w_many_uniprots or target in targets_w_unique_uniprot:
            continue
        # If a target finished with a pattern mutation, we assume it is a unique target
        if re.search(r'\b[A-Z]\d+[A-Z]\b', target) or re.search(r'\bDEL', target):
            targets_w_mutations.append(target)
            logger.info(f"{target} -> {', '.join(uniprots)} (mutation)")
            continue
    logger.info(f"Number of targets with mutations: {len(targets_w_mutations)} ({len(targets_w_mutations) / len(target2uniprots):.2%} of targets)")

    targets_w_no_uniprot = []
    for target, uniprots in target2uniprots.items():
        # Skip targets that have already been processed
        if target in targets_w_many_uniprots or target in targets_w_unique_uniprot or target in targets_w_mutations:
            continue
        if not uniprots:
            targets_w_no_uniprot.append(target)
            logger.info(f"{target} -> No Uniprot ID, DOIs: {', '.join(['https://doi.org/' + doi for doi in target2dois.get(target, [])])}")
    logger.info(f"Number of targets with no Uniprot ID: {len(targets_w_no_uniprot)} ({len(targets_w_no_uniprot) / len(target2uniprots):.2%} of targets)")

    # For targets with no Uniprot ID, we try to find a match in the names2uniprot dictionary
    targets_w_no_uniprot_matches = []
    for target in targets_w_no_uniprot:
        # Use fuzzy matching to find the best match in names2uniprot
        match, score = process.extractOne(target, names2uniprot.keys())
        if score >= 50:  # Set a threshold for the match quality
            uniprot_id = names2uniprot[match]
            targets_w_no_uniprot_matches.append((target, uniprot_id, match, score))
            logger.info(f"{target} -> {uniprot_id} (matched with '{match}' with score {score})")

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
        if json_info and not args.force_refetch:
            uniprot2infos[uniprot_id] = json_info
        else:
            info = fetch_protein_info(uniprot_id)
            if info:
                uniprot2infos[uniprot_id] = info
                # Save each entry to a separate JSON file
                save_dict(info, os.path.join(uniprot_dir, f'{uniprot_id}.json'))

    prev_targets = targets_w_many_uniprots + targets_w_unique_uniprot + targets_w_mutations + targets_w_no_uniprot
    targets_leftover = []
    for target, uniprots in target2uniprots.items():
        # Skip targets that have already been processed
        if target in prev_targets:
            continue
        # Print all other targets that have the same Uniprot IDs
        other_targets = [t for t, us in target2uniprots.items() if us == uniprots and t != target and t not in prev_targets]
        if other_targets:
            logger.debug(f"{target} -> {', '.join(uniprots)} (also associated with N.{len(other_targets)}: {', '.join(other_targets)})")
        targets_leftover.append(target)
    logger.info(f"Number of leftover targets: {len(targets_leftover)} ({len(targets_leftover) / len(target2uniprots):.2%} of targets)")

    # TODO: Manually map leftover targets to Uniprot IDs after inspecting their
    # sequences and the associated DOIs...

    # ==========================================================================
    # ### Assign Missing Uniprot IDs to Fuzzy Targets
    # ==========================================================================
    missing_target2uniprots = []
    for df_name, df in df_dict.items():
        logger.info(f"Number of missing Targets in {df_name}: {df['Target'].isna().sum():,}")
        logger.info(f"Number of missing Targets {df_name} in {df_name}: {df[f'Target {df_name}'].isna().sum():,}")
        logger.info(f"Number of missing Uniprot IDs in {df_name}: {df['Uniprot'].isna().sum():,}")
        df['Uniprot'] = df.apply(
            lambda row: assign_missing_uniprot(
                row,
                df_name,
                force_replace=True,
                targets_w_no_uniprot_mapping=targets_w_no_uniprot_mapping,
                target2uniprots=target2uniprots,
            ), axis=1)
        logger.info(f"Number of missing Uniprot IDs in {df_name} after assignment: {df['Uniprot'].isna().sum():,}")
        
        for _, row in df.iterrows():
            if pd.isna(row['Uniprot']):
                target_cols = [col for col in df.columns if col.startswith('Target')]
                targets = [row[col] for col in target_cols if pd.notna(row[col])]
                missing_target2uniprots += targets

    missing_target2uniprots = list(set(missing_target2uniprots))
    logger.info(f"Number of targets with missing Uniprot IDs: {len(missing_target2uniprots):,}")
    logger.info("Missing targets:")
    for target in missing_target2uniprots:
        logger.info(f'  - {target}')

    assert len(missing_target2uniprots) == 0, "There are still targets with missing Uniprot IDs."

    # ### Add a 'POI Sequence' and a 'E3 Ligase Sequence' Columns 
    for df_name, df in df_dict.items():
        # Add 'POI Sequence' column using the 'Uniprot' column
        df['POI Sequence'] = df['Uniprot'].apply(lambda row: get_sequence_from_uniprot(row, uniprot2infos))
        
        # Add 'E3 Ligase Sequence' column using the 'E3 Ligase Uniprot' column
        df['E3 Ligase Sequence'] = df['E3 Ligase Uniprot'].apply(lambda row: get_sequence_from_uniprot(row, uniprot2infos))

        # Count the number of NaN values in the 'POI Sequence' and 'E3 Ligase Sequence' columns
        poi_seq_nan_count = df['POI Sequence'].isna().sum()
        e3_ligase_seq_nan_count = df['E3 Ligase Sequence'].isna().sum()
        logger.info(f"{df_name} - POI Sequence NaN count: {poi_seq_nan_count}")
        logger.info(f"E3 Ligase Sequence NaN count: {e3_ligase_seq_nan_count}")

    # ==========================================================================
    # ### Apply Mutations
    # ==========================================================================
    for df_name, df in df_dict.items():
        df['POI Sequence'] = df.apply(apply_mutation_to_sequence, axis=1)

    # ==========================================================================
    # ## Standardize Cell Names
    # ==========================================================================
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
    logger.debug(f'Number of unique cell types with DOIs: {len(celltype2dois):,}')

    # --------------------------------------------------------------------------
    # Based on fuzzy matches, and after checking the referenced DOIs, we
    # manually standardized the cell names by mapping them to entries in the
    # Cellosaurus database.
    # --------------------------------------------------------------------------
    # NOTE: "Primary" refers to cells taken from patients, so they are not a
    # single cell line, but rather a mix of cells (think about a tissue)
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

    # If we wanna see them in a more readable format, together with their publications, we can do:
    tmp = []
    for df_name, df in df_dict.items():
        tmp.append(df.apply(lambda row: get_manual_cell_mapping(row, manual_cell_lines), axis=1).dropna())

    tmp = pd.concat(tmp, axis=0).drop_duplicates().reset_index(drop=True)
    logger.info('Manually curated cell lines:\n' + str(tmp.to_markdown()))

    # Define the CellEmbedding class, which downloads and processes the
    # CelloSaurus database to get cell lines and their synonyms.
    cell_embedding = CellEmbedding(load_from_cache=False)

    # Apply the standardization function to the dataframes
    for df_name, df in df_dict.items():
        logger.debug('--' * 40)
        logger.debug(f"Standardizing cell lines in {df_name}...")
        logger.debug('--' * 40)
        df_dict[df_name] = df.apply(lambda row: standardize_cell_line_protacdb(row, cell_embedding, manual_cell_lines, logger), axis=1)
        logger.debug(df_dict[df_name].head(3))

    # ==========================================================================
    # ## Add Species Based on Cell Lines
    # ==========================================================================
    for df_name, df in df_dict.items():
        cell_col = [c for c in df.columns if c.startswith('Cell Type')]
        if cell_col:
            df['Cell Species'] = df[cell_col[0]].apply(lambda row: get_cell_species(row, cell_embedding))
            logger.debug(df['Cell Species'].unique())

    # ### Modify E3 Ligase Based on Species
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

    # Fetch and cache UniProt entries defined above
    for uniprot_id in tqdm(uniprots, desc='Fetching UniProt entries'):
        json_info = load_dict(uniprot_dir / f'{uniprot_id}.json')
        if json_info:
            uniprot2infos[uniprot_id] = json_info
        else:
            info = fetch_protein_info(uniprot_id)
            if info:
                uniprot2infos[uniprot_id] = info
                # Save each entry to a separate JSON file
                save_dict(info, uniprot_dir / f'{uniprot_id}.json')

    for df_name, df in df_dict.items():
        logger.info('--' * 40)
        if 'Cell Species' not in df.columns:
            logger.warning(f"'Cell Species' column not found in {df_name}. Skipping...")
            continue
        logger.info(f"Dataframe: {df_name}")
        tqdm.pandas(desc='Updating E3 Ligase Uniprot')
        df['E3 Ligase Uniprot'] = df.progress_apply(lambda row: update_e3ligase_uniprot(row, e3ligase2uniprot), axis=1)

        tqdm.pandas(desc='Updating E3 Ligase Sequence')
        df['E3 Ligase Sequence'] = df.progress_apply(lambda row: update_e3ligase_sequence(row, uniprot2infos), axis=1)

    # ### Modify POI Based on Species

    # Check whether the reported Uniprot IDs match the species of the cell line
    # used in the assay. If not, try to find a matching Uniprot ID for the
    # reported target name and the species of the cell line.
    for df_name, df in df_dict.items():
        df['POI Species'] = df['Uniprot'].apply(lambda row: get_poi_species(row, uniprot2infos))

    # Report all entries for which the POI species does not match the cell
    # species
    logger.debug('List of non-human POI entries:')
    for df_name, df in df_dict.items():
        logger.debug(f"Dataframe: {df_name}")
        logger.debug('-' * 40)
        cell_col = [c for c in df.columns if c.startswith('Cell Type')]
        doi_cols = [c for c in df.columns if c.startswith('Article')]
        target_cols = [c for c in df.columns if c.startswith('Target')]
        if not cell_col:
            logger.warning(f"'Cell Type' column not found in {df_name}. Skipping...")
            continue
        cell_col = cell_col[0]
        tmp = df[(df['Cell Species'] != 'Homo sapiens') & df['Cell Species'].notnull()]
        # Add "https://doi.org/" in front of the doi_cols
        for col in doi_cols:
            tmp[col] = tmp[col].apply(lambda x: f"https://doi.org/{x}" if pd.notna(x) and not str(x).startswith("https://doi.org/") else x)
        logger.debug('\n' + str(tmp.drop_duplicates(subset=['Uniprot', 'Cell Species'])[['Uniprot', *target_cols, cell_col, 'Cell Species'] + doi_cols].fillna('-').reset_index(drop=True).to_markdown(index=True)))

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

    # Fetch and cache UniProt entries defined above
    for uniprot_id in tqdm(uniprots, desc='Fetching UniProt entries'):
        json_info = load_dict(uniprot_dir / f'{uniprot_id}.json')
        if json_info:
            uniprot2infos[uniprot_id] = json_info
        else:
            info = fetch_protein_info(uniprot_id)
            if info:
                uniprot2infos[uniprot_id] = info
                # Save each entry to a separate JSON file
                save_dict(info, uniprot_dir / f'{uniprot_id}.json')

    # Map the POI Sequence based on the Uniprot ID
    for df_name, df in df_dict.items():
        logger.info('-' * 40)
        logger.info(f"Dataframe: {df_name}")
        logger.info('-' * 40)
        tqdm.pandas(desc='Mapping POI Uniprot ID from species')
        if 'Cell Species' not in df.columns:
            logger.info(f"'Cell Species' column not found in {df_name}. Skipping...")
            continue
        df['Uniprot'] = df.progress_apply(lambda row: map_poi_uniprot_from_species(row, species2uniprot), axis=1)

        tqdm.pandas(desc='Mapping POI Sequence from Uniprot ID')
        df['POI Sequence'] = df.progress_apply(lambda row: map_poi_sequence_from_uniprot(row, uniprot2infos), axis=1)

    # ==========================================================================
    # ## Get Assay Type
    # ==========================================================================
    for df_name, df in df_dict.items():
        # Create a new column named 'Assay Type' based on the 'Assay {df_name}' column
        df['Assay Type'] = df.apply(lambda row: get_assay_type(row, df_name), axis=1)
        logger.info(f"Unique assay types in {df_name}:\n{df['Assay Type'].value_counts()}")

    # ==========================================================================
    # ## Save Dataframes
    # ==========================================================================
    for df_name, df in df_dict.items():
        logger.info(f"Processing dataframe: {df_name}")
        logger.info('-' * 40)
        
        # Remove columns with all NaN values
        logger.debug(f"Number of columns in {df_name} before removing all-NaN columns: {len(df.columns.tolist())}")
        df_dict[df_name] = df.dropna(axis=1, how='all')
        logger.debug(f"Number of columns in {df_name} after removing all-NaN columns: {len(df_dict[df_name].columns.tolist())}")
        
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
                return pd.NA

            tmp['Value_Operator'] = tmp.apply(get_value_operator, axis=1)
            tmp = tmp.drop(columns=['DC50 (nM) Operator (DC50/Dmax)', 'Dmax (%) Operator (DC50/Dmax)'])
            cols_to_save += ['Value_Operator']

            logger.debug(f"Shape of the temporary dataframe for {df_name}: {tmp.shape}")
            logger.debug(tmp['Value_Operator'].dropna().value_counts())
            logger.debug(tmp[cols_to_save])
            
            for assay in tmp['Assay'].unique():
                logger.debug(assay)

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

            logger.debug(f"Shape of the temporary dataframe for {df_name}: {tmp.shape}")
            logger.debug(tmp[cols_to_save])
            # Save DataFrame to data_curation_dir as CSV
            csv_path = Path(data_curation_dir) / f'protacdb_protac_percent_degradation.csv'
            tmp[cols_to_save].drop_duplicates().to_csv(csv_path, index=False)

        elif df_name == '(Cellular activities, IC50)':
            logger.info(f'Skipping dataframe {df_name} for now...')
        elif df_name == '(Protac to Target, IC50)':
            logger.info(f'Skipping dataframe {df_name} for now...')

    print('-' * 80)
    print(f"Logs saved to: {log_file}")
    print('-' * 80)

    # ==========================================================================
    # TODO: Add Protein Location

    # all_locations = set()
    # uniprot2locations = {}

    # for uniprot in merged_df['Uniprot'].dropna().unique():
    #     locations = uniprot2infos.get(uniprot, {}).get('locations')
    #     if locations:
    #         all_locations.update(set(locations))

    # for uniprot in merged_df['Uniprot'].dropna().unique():
    #     locations = list(uniprot2infos.get(uniprot, {}).get('locations'))
    #     uniprot2locations[uniprot] = {k: 1 if k in locations else 0 for k in all_locations}

    # logger.info(f"Number of unique locations across all Uniprot IDs: {len(all_locations):,}")
    # for location in sorted(all_locations):
    #     logger.info(f"- {location}")
    # tqdm.pandas(desc=f"Adding location information...")
    # merged_df = merged_df.progress_apply(add_location, axis=1)
    # logger.info(f"Number of columns after adding location information: {len(merged_df.columns.tolist())}")

if __name__ == "__main__":
    main()