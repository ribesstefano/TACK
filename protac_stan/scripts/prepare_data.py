import os
import sys
from typing import Union, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import KFold, GroupKFold
from datasets import load_dataset

from data import PROTACData

DATASET_NAME = "ailab-bio/TACK"
TASK_TO_HF_CONFIG = {'bin': 'multitask', 'dc50': 'DC50', 'dmax': 'Dmax'}
DC50_THRESH = 100.0
DMAX_THRESH = 80.0


def sanitize_sequence(seq: str) -> str:
    if not isinstance(seq, str):
        return ""
    return seq.strip().replace('U', 'C').replace('O', 'K').replace('B', 'D').replace('Z', 'E').replace('J', 'L')


def get_protein_key(uniprot_id, sequence: str) -> str:
    if pd.notna(uniprot_id) and str(uniprot_id).strip() not in ['None', '', 'nan']:
        return str(uniprot_id).strip()
    return hashlib.md5(sanitize_sequence(sequence).encode('utf-8')).hexdigest()


def compute_label(row, task: str):
    dc50_val, dmax_val = None, None
    
    if 'Value_DC50' in row:
        dc50_val = row['Value_DC50']
    elif 'DC50' in row:
        dc50_val = row['DC50']
    elif task == 'dc50' and 'Value' in row:
        dc50_val = row['Value']
    
    if 'Value_Dmax' in row:
        dmax_val = row['Value_Dmax']
    elif 'Dmax' in row:
        dmax_val = row['Dmax']
    elif task == 'dmax' and 'Value' in row:
        dmax_val = row['Value']
    
    dc50_val = float(dc50_val) if pd.notna(dc50_val) else None
    dmax_val = float(dmax_val) if pd.notna(dmax_val) else None
    
    if task == 'dc50':
        return 1 if dc50_val is not None and dc50_val <= DC50_THRESH else (0 if dc50_val else np.nan)
    elif task == 'dmax':
        return 1 if dmax_val is not None and dmax_val >= DMAX_THRESH else (0 if dmax_val else np.nan)
    elif task == 'bin':
        if dc50_val is None or dmax_val is None:
            return np.nan
        if dc50_val > DC50_THRESH:
            return 0
        if dmax_val < DMAX_THRESH:
            return 0
        if dc50_val <= DC50_THRESH and dmax_val >= DMAX_THRESH:
            return 1
        return np.nan
    return np.nan


def compute_descriptors(smiles_list: list) -> tuple:
    desc_cols = [
        'Molecular Weight', 'Exact Mass', 'XLogP3', 'Heavy Atom Count',
        'Ring Count', 'Hydrogen Bond Acceptor Count', 'Hydrogen Bond Donor Count',
        'Rotatable Bond Count', 'Topological Polar Surface Area'
    ]
    descriptors, valid_indices = [], []
    
    for i, smiles in tqdm(enumerate(smiles_list), total=len(smiles_list), desc="Computing descriptors"):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                desc = [
                    Descriptors.MolWt(mol), Descriptors.ExactMolWt(mol), Descriptors.MolLogP(mol),
                    Descriptors.HeavyAtomCount(mol), Descriptors.RingCount(mol), Descriptors.NumHAcceptors(mol),
                    Descriptors.NumHDonors(mol), Descriptors.NumRotatableBonds(mol), Descriptors.TPSA(mol)
                ]
                descriptors.append(desc)
                valid_indices.append(i)
        except Exception:
            pass
    
    return descriptors, valid_indices, desc_cols


def create_cv_splits(
    ds: pd.DataFrame,
    n_splits: int = 5,
    n_repeats: int = 5,
    group_col: Optional[str] = None,
    base_seed: int = 42,
):
    """Create repeated k-fold cross-validation splits.
    
    Args:
        ds: Dataset or DataFrame to split.
        n_splits: Number of folds.
        n_repeats: Number of repeats.
        group_col: Column name for grouping (if None, uses standard K-Fold).
        base_seed: Base random seed.
        
    Yields:
        Dictionary with repeat, cv_fold, fold, train_idx, test_idx.
    """
    if group_col is None:
        groups = np.zeros(len(ds))
    else:
        groups = np.array(ds[group_col])
    
    for repeat in range(n_repeats):
        seed = base_seed + repeat
        if group_col is None:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        else:
            kf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X=np.zeros(len(groups)), groups=groups)):
            yield {
                "repeat": repeat,
                "fold": fold,
                "global_fold_id": repeat * n_splits + fold,
                "train_idx": train_idx,
                "test_idx": test_idx,
            }


def load_data(args) -> pd.DataFrame:
    if args.custom_dataset_csv:
        print(f"Loading custom CSV: {args.custom_dataset_csv}")
        return pd.read_csv(args.custom_dataset_csv)
    
    hf_config = TASK_TO_HF_CONFIG[args.task]
    print(f"Loading from HuggingFace: {DATASET_NAME} ({hf_config})")
    ds = load_dataset(DATASET_NAME, hf_config, split="train", token=True)
    return ds.to_pandas()


def process_dataframe(df: pd.DataFrame, task: str, held_out: bool) -> pd.DataFrame:
    if 'SMILES_Held_Out' in df.columns:
        is_held_out = df['SMILES_Held_Out'].astype(str).str.lower() == 'true'
        df = df[is_held_out].copy() if held_out else df[~is_held_out].copy()
        print(f"Filtered to {len(df)} {'held-out' if held_out else 'training'} rows")
    elif held_out:
        print("Warning: SMILES_Held_Out column not found, using all data")
    
    df['Uniprot'] = df.apply(lambda x: get_protein_key(x.get('POI_UniProt'), x.get('POI_Sequence', '')), axis=1)
    df['E3 ligase Uniprot'] = df.apply(lambda x: get_protein_key(x.get('Ligase_UniProt'), x.get('Ligase_Sequence', '')), axis=1)
    
    df['label'] = df.apply(lambda row: compute_label(row, task), axis=1)
    df = df.dropna(subset=['label']).copy()
    df['label'] = df['label'].astype(int)
    
    if 'SMILES' in df.columns:
        df = df.rename(columns={'SMILES': 'Smiles'})
    
    return df


def process_subset(args, held_out: bool):
    """Process either training or held-out subset."""
    output_name = f"held_out_{args.task}" if held_out else f"custom_{args.task}"
    subset_label = "held-out" if held_out else "training"
    
    print(f"\n{'='*60}")
    print(f"Processing {subset_label.upper()} data")
    print(f"{'='*60}")
    
    # Load and process data
    df = load_data(args)
    df = process_dataframe(df, args.task, held_out)
    
    if len(df) == 0:
        print(f"No {subset_label} samples found. Skipping.")
        return None
    
    print(f"Labeled samples: {len(df)}")
    
    # Compute molecular descriptors
    descriptors, valid_indices, desc_cols = compute_descriptors(df['Smiles'].tolist())
    df = df.iloc[valid_indices].copy()
    desc_df = pd.DataFrame(descriptors, columns=desc_cols, index=df.index)
    df = pd.concat([df, desc_df], axis=1)
    
    # Fill cluster columns
    for col in ['SMILES_Scaffold_Cluster', 'SMILES_Butina_Cluster']:
        if col in df.columns:
            df[col] = df[col].fillna(-1).astype(int)
    
    # Sort and save
    df = df.sort_values(by=['Smiles', 'Uniprot', 'E3 ligase Uniprot']).reset_index(drop=True)
    csv_path = os.path.join(args.output_dir, f"{output_name}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} rows to {csv_path}")
    
    # Generate CV splits (training only)
    if not held_out:
        group_col = 'SMILES_Scaffold_Cluster' if 'SMILES_Scaffold_Cluster' in df.columns else None
        if group_col is None:
            print(f"Warning: SMILES_Scaffold_Cluster not found, using standard K-Fold")
        
        # Call your custom function and format for JSON serialization
        splits = []
        for split_dict in create_cv_splits(df, group_col=group_col):
            # json.dump requires standard lists, so we apply .tolist() to the numpy arrays
            split_dict["train_idx"] = split_dict["train_idx"].tolist()
            split_dict["test_idx"] = split_dict["test_idx"].tolist()
            splits.append(split_dict)
            
        splits_path = f"cv_splits_{args.task}.json"
        with open(splits_path, 'w') as f:
            json.dump(splits, f)
        print(f"Saved {len(splits)} splits to {splits_path}")
    
    # Build PyG graph data
    esm_path = os.path.join(args.output_dir, 'esm_s_map.pkl')
    if not os.path.exists(esm_path):
        print(f"WARNING: {esm_path} not found. Skipping graph building.")
        return output_name
    
    processed_dir = os.path.join(args.output_dir, 'processed', output_name)
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)
    
    PROTACData(root=args.output_dir, name=output_name)
    print(f"Built graph data in {processed_dir}")
    
    return output_name


def main():
    parser = argparse.ArgumentParser(description="Prepare PROTAC data for training")
    parser.add_argument('--task', type=str, choices=['bin', 'dc50', 'dmax'], required=True)
    parser.add_argument('--custom_dataset_csv', type=str, default=None,
                        help='Path to custom CSV (overrides HuggingFace)')
    parser.add_argument('--output_dir', type=str, default='data/custom')
    parser.add_argument('--skip_held_out', action='store_true', help='Skip held-out subset processing')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process training data
    print(f"\n{'='*60}\nPreparing data for task: {args.task.upper()}\n{'='*60}")
    train_name = process_subset(args, held_out=False)
    
    # Process held-out data (unless skipped)
    heldout_name = None
    if not args.skip_held_out:
        heldout_name = process_subset(args, held_out=True)
    
    print(f"\n{'='*60}")
    print(f"Done! Outputs: {train_name}" + (f", {heldout_name}" if heldout_name else ""))
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

