"""
Script to train PROTAC degradation prediction models using different architectures
and configurations. Supports MLP, BERT-based, and XGBoost models with options for
running 5x5 CV on different data splitting strategies and with hyperparameter
tuning (currently only for XGBoost).

Author: Stefano Ribes
"""
import argparse
import logging
from pathlib import Path
from typing import Union, Dict, Any

import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from datasets import load_dataset, Dataset

from tackai.config import load_config_from_yaml  # noqa: E402
from tackai.training import run_cv_experiment
from staeda import DegradationComplexDataModule

def get_bin_label(
        row: Union[pd.Series, Dict[str, Any]],
        dmax_threshold: float = 80.0,
        dc50_threshold: float = 100.0,
) -> Union[int, float]:
    """ Get binary activity label based on Dmax and DC50 thresholds.
    
    Args:
        row (pd.Series | Dict[str, Any]): A row from the dataframe containing 'Dmax' and 'DC50' columns.
        dmax_threshold (float): Threshold for Dmax to consider a compound active.
        dc50_threshold (float): Threshold for DC50 to consider a compound active.
        
    Returns:
        int | float: 1 for active, 0 for inactive, np.nan for undefined.
    """
    dc50 = row.get('DC50', np.nan)
    dmax = row.get('Dmax', np.nan)

    # 1. Definite INACTIVE (0)
    if pd.notna(dc50) and float(dc50) >= dc50_threshold:
        return 0
    if pd.notna(dmax) and float(dmax) < dmax_threshold:
        return 0

    # 2. Definite ACTIVE (1)
    if pd.notna(dc50) and float(dc50) < dc50_threshold:
        return 1
    if pd.notna(dmax) and float(dmax) >= dmax_threshold:
        return 1

    return np.nan

def map_bin_labels(example: Dict[str, Any]) -> Dict[str, Any]:
    """ Map binary activity labels to the example based on Dmax and DC50 values. """
    row = {
        'Dmax': example.get('Value_Dmax', np.nan),
        'DC50': example.get('Value_DC50', np.nan),
    }
    example['Activity'] = get_bin_label(row)
    return example

def map_bin_dmax_labels(example: Dict[str, Any]) -> Dict[str, Any]:
    """ Map binary activity labels based on Dmax values only. """
    dmax = example.get('Dmax', np.nan)
    if pd.notna(dmax):
        example['Activity'] = 1 if float(dmax) >= 80.0 else 0
    else:
        example['Activity'] = np.nan
    return example

def map_bin_dc50_labels(example: Dict[str, Any]) -> Dict[str, Any]:
    """ Map binary activity labels based on DC50 values only. """
    dc50 = example.get('DC50', np.nan)
    if pd.notna(dc50):
        example['Activity'] = 1 if float(dc50) < 100.0 else 0
    else:
        example['Activity'] = np.nan
    return example

def convert_dc50_to_pdc50(example, type_col='Value_Type', value_col='Value'):
    """ Convert DC50 in nano Molar to pDC50 (-log10(M)). """
    if type_col in example and example[type_col] == 'Dmax':
        return example
    return DegradationComplexDataModule.convert_dc50_to_pdc50(example[value_col])

def split_held_out(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ Split the dataframe into held-out and non-held-out sets based on the
    'SMILES_Held_Out' column.
    
    Args:
        df (pd.DataFrame): The input dataframe containing a 'SMILES_Held_Out' column.
        
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two dataframes:
            - The first dataframe contains non-held-out samples.
            - The second dataframe contains held-out samples.
    """
    held_out_df = df[df['SMILES_Held_Out']].copy().reset_index(drop=True)
    df = df[~df['SMILES_Held_Out']].copy().reset_index(drop=True)
    return df, held_out_df

def main():
    parser = argparse.ArgumentParser(description="Train PROTAC degradation prediction models.")
    parser.add_argument('--model_type', type=str, choices=['mlp', 'bert', 'xgboost'], default='bert',
                        help='Type of model to train.')
    parser.add_argument('--task', type=str, choices=['dmax', 'dc50', 'multitask', 'bin', 'dmax_bin', 'dc50_bin'], default='dmax',
                        help='Task to train the model on.')
    parser.add_argument('--group', type=str, choices=['random', 'scaffold', 'butina'], default='random',
                        help='Data splitting strategy.')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save model checkpoints.')
    parser.add_argument('--predictions_dir', type=str, default='./predictions',
                        help='Directory to save model predictions.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training.')
    parser.add_argument('--num_proc', type=int, default=1,
                        help='Number of processes for data loading.')
    parser.add_argument('--data_config', type=str, default=None,
                        help='Path to YAML file with data configuration.')
    parser.add_argument('--model_config', type=str, default=None,
                        help='Path to YAML file with model configuration.')
    pasrser.add_argument('--tune_hyperparameters', action='store_true',
                        help='Whether to perform hyperparameter tuning (unused, in development).')
    parser.add_argument('--n_tuning_trials', type=int, default=20,
                        help='Number of hyperparameter tuning trials (unused, in development).')
    parser.add_argument('--custom_dataset_csv', type=str, default=None,
                        help='Path to a custom dataset CSV file that overrides the default TACK dataset (must contain the same columns).')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    args = parser.parse_args()
    
    # Set float32 matmul precision to high for better performance
    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision('high')
    
    # Enable DEBUG logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # Setup checkpoint and predictions directories if they don't exist
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.predictions_dir).mkdir(parents=True, exist_ok=True)

    # Load custom dataset if provided, otherwise load default TACK dataset
    if args.custom_dataset_csv is not None:
        logger.debug(f"Loading custom dataset from CSV file: {args.custom_dataset_csv}")
        df = pd.read_csv(args.custom_dataset_csv)
        ds = Dataset.from_pandas(df, preserve_index=False)
    else:
        # Download and prepare dataset
        ds_config = 'default'
        if 'dmax' in args.task:
            ds_config = 'Dmax'
        elif 'dc50' in args.task:
            ds_config = 'DC50'
        elif args.task == 'bin' or (args.task == 'multitask' and args.model_type != 'bert'):
            ds_config = 'multitask'
        ds = load_dataset(
            "ailab-bio/TACK",
            ds_config,
            split="train",
        )

    # Process dataset labels based on task
    labels = ['Value']
    if args.task == 'dmax':
        ds = ds.rename_column('Value', 'Dmax')
        labels = ['Dmax']
    elif args.task == 'dc50':
        ds = ds.rename_column('Value', 'DC50')
        labels = ['DC50']
    elif args.task == 'bin':
        ds = ds.map(map_bin_labels)
        ds = ds.filter(lambda x: pd.notna(x['Activity']))
        labels = ['Activity']
    elif args.task == 'dmax_bin':
        ds = ds.rename_column('Value', 'Dmax')
        ds = ds.map(map_bin_dmax_labels)
        ds = ds.filter(lambda x: pd.notna(x['Activity']))
        labels = ['Activity']
    elif args.task == 'dc50_bin':
        ds = ds.rename_column('Value', 'DC50')
        ds = ds.map(map_bin_dc50_labels)
        ds = ds.filter(lambda x: pd.notna(x['Activity']))
        labels = ['Activity']
    elif args.task == 'multitask' and args.model_type != 'bert':
        # ds = ds.map(lambda x: convert_dc50_to_pdc50(x, type_col='Value_Type', value_col='Value_DC50'))
        # Rename 'Value_Dmax' and 'Value_DC50' to 'Dmax' and 'DC50'
        # NOTE: This is needed for better reporting when collecting metrics
        ds = ds.rename_column('Value_Dmax', 'Dmax')
        ds = ds.rename_column('Value_DC50', 'DC50')
        labels = ['Dmax', 'DC50']
    elif args.task == 'multitask' and args.model_type == 'bert':
        raise ValueError("Multitask training with BERT model is not supported anymore. One needs to update the handling the label normalization of the 'Value' column.")        

    # Clip 'Dmax' values to [0, 100]
    if 'Dmax' in labels:
        def clip_dmax(example):
            example['Dmax'] = np.clip(example['Dmax'], 0, 100)
            return example
        ds = ds.map(clip_dmax)

    # Split held-out set
    df = ds.to_pandas()
    df, held_out_df = split_held_out(df)
    ds = Dataset.from_pandas(df, preserve_index=False)
    held_out_ds = Dataset.from_pandas(held_out_df, preserve_index=False)
        
    # Print the distribution of labels
    logger.debug(f"Label distribution for task {args.task}:")
    if args.task != 'multitask':
        for label in labels:
            logger.debug(f"[TRAIN/VAL] {label} - mean: {df[label].mean()}, std: {df[label].std()}, min: {df[label].min()}, max: {df[label].max()}")
            logger.debug(f"[HELD-OUT]  {label} - mean: {held_out_df[label].mean()}, std: {held_out_df[label].std()}, min: {held_out_df[label].min()}, max: {held_out_df[label].max()}")

    # Setup data config
    if args.data_config is not None:
        print(f"Loading data config from YAML file: {args.data_config}")
        data_config = load_config_from_yaml(Path(args.data_config))
    else:
        print(f"Using default data config for model type: {args.model_type}")
        data_config = _get_default_data_config(args.model_type)

    # Override batch size and num_proc from command line args
    data_config['num_proc'] = args.num_proc
    data_config['batch_size'] = args.batch_size

    # Setup model configs
    if args.model_config is not None:
        print(f"Loading model config from YAML file: {args.model_config}")
        model_config = load_config_from_yaml(Path(args.model_config))
    else:
        model_config = _get_default_model_config(args.model_type)

    run_cv_experiment(
        dataset=ds,
        held_out_dataset=held_out_ds,
        model_type=args.model_type,
        data_config=data_config,
        model_config=model_config,
        task_name=args.task,
        labels=labels,
        group=args.group,
        checkpoints_dir=Path(args.checkpoint_dir),
        results_dir=Path(args.predictions_dir),
        tune_hyperparameters=True,
        n_tuning_trials=20 if args.model_type == 'xgboost' else 100,
    )

def _get_default_data_config(model_type: str) -> dict:
    """ Get default data configuration based on model type. """
    if model_type == 'bert':
        return {
            'use_tokenizer': True,
            'tokenizer_name': 'google-bert/bert-base-cased',
            'cell_line_col': 'Cell_Line',
            'normalize_labels': True,
            'use_fingerprints': False,
            'use_descriptors': False,
            'use_relevant_descriptors': True,
            'fp_size': 512,
            'radius': 16,
            'max_length': 512,
            'batch_size': 16,
            'num_workers': 0,
            'num_proc': 1,
            'use_assay_type_encoding': False,
            'use_treatment_time': False,
            'use_cell_name_embedding': False,
            'use_poi_name_embedding': False,
            'use_poi_sequence_embedding': False,
            'use_ligase_name_embedding': False,
            'use_cell_description_embedding': False,
            'impute_labels': False,
            'include_prompt': False,
            'is_bert_multitask': False,
            'use_poi_pca': False,
            'poi_pca_n_components': 0.95,
            'use_ligase_pca': False,
            'ligase_pca_n_components': 0.95,
        }
    else:
        return {
            'use_cell_name_embedding': True,
            'use_poi_name_embedding': True,
            'use_ligase_name_embedding': True,
            'use_descriptors': True,
            'cell_line_col': "Cell_Line_ID",
            'normalize_labels': True,
            'use_fingerprints': False,
            'use_relevant_descriptors': True,
            'use_tokenizer': False,
            'fp_size': 512,
            'radius': 16,
            'tokenizer_name': 'google-bert/bert-base-cased',
            'max_length': 512,
            'batch_size': 16,
            'num_workers': 0,
            'num_proc': 1,
            'use_assay_type_encoding': False,
            'use_treatment_time': False,
            'use_poi_sequence_embedding': False,
            'use_cell_description_embedding': False,
            'impute_labels': False,
            'include_prompt': False,
            'is_bert_multitask': False,
            'use_poi_pca': False,
            'poi_pca_n_components': 0.95,
            'use_ligase_pca': False,
            'ligase_pca_n_components': 0.95,
        }

def _get_default_model_config(model_type: str) -> dict:
    """ Get default model configuration based on model type. """
    if model_type == 'bert':
        return {
            'model_config': {
                'learning_rate': 1e-3,
                'lr_scheduler_type': 'reduce_on_plateau',
                'warmup_ratio': 0.2,
                'num_cycles': 5,
                'model_name': 'google-bert/bert-base-cased',
                'task_type': 'point',
                'head_dropout': 0.0,
                'head_depth': 3,
                'freeze_bert': True,
            },
            'training_config': {
                'max_epochs': 100,
                'patience': 20,
            },
        }
    elif model_type == 'mlp':
        return {
            'model_config': {
                'learning_rate': 1e-3,
                'hidden_dim': 512,
                'mlp_depth': 4,
                'head_depth': 1,
                'dropout': 0.0,
                'head_dropout': 0.0,
                'task_type': 'point',
            },
            'training_config': {
                'max_epochs': 200,
                'patience': 5,
            },
        }
    elif model_type == 'xgboost':
        return {
            'model_config': {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'learning_rate': 0.1,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'tree_method': 'hist',
                'multi_strategy': 'multi_output_tree',
            },
            'training_config': {
                'num_boost_round': 1000,
                'early_stopping_rounds': 50,
            }
        }
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

if __name__ == "__main__":
    main()