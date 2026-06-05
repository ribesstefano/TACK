"""
Task-specific dataset loading and label preparation for PROTAC degradation
prediction. Centralizes the logic that turns the raw TACK dataset (or a custom
CSV) into the train/validation and held-out splits expected by the training
pipeline.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset

from tackai import DegradationComplexDataModule

logger = logging.getLogger(__name__)


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

    # Return inactive (0) if either DC50 is above threshold or Dmax is below threshold
    if pd.notna(dc50) and float(dc50) >= dc50_threshold:
        return 0
    if pd.notna(dmax) and float(dmax) < dmax_threshold:
        return 0

    # If either DC50 or Dmax is missing, we cannot determine activity, return np.nan
    if pd.isna(dc50) or pd.isna(dmax):
        return np.nan

    # Return active (1) if DC50 is below threshold and Dmax is above or equal to threshold
    if float(dc50) < dc50_threshold and float(dmax) >= dmax_threshold:
        return 1
    return 0


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


def convert_dc50_to_pdc50(
    example: Dict[str, Any],
    type_col: str = 'Value_Type',
    value_col: str = 'Value',
) -> Dict[str, Any]:
    """ Convert DC50 in nano Molar to pDC50 (-log10(M)). """
    if type_col in example and example[type_col] == 'Dmax':
        return example
    example['Value'] = DegradationComplexDataModule.convert_dc50_to_pdc50(example[value_col])
    return example


def split_held_out(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
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


def load_task_dataset(
    task: str,
    model_type: str,
    custom_dataset_csv: Optional[str] = None,
) -> Tuple[Dataset, Dataset, List[str]]:
    """ Load and prepare the dataset for a given task.

    Loads either a custom CSV or the published TACK dataset, applies the
    task-specific label processing (column renames, binary labelling, Dmax
    clipping), and splits off the held-out set.

    Args:
        task: Task to train on ('dmax', 'dc50', 'bin', 'dmax_bin', 'dc50_bin',
            'multitask').
        model_type: Model type ('mlp', 'bert', 'xgboost'); affects dataset
            configuration selection and multitask handling.
        custom_dataset_csv: Optional path to a CSV that overrides the default
            TACK dataset (must contain the same columns).

    Returns:
        Tuple of (train/validation dataset, held-out dataset, label names).
    """
    # Load custom dataset if provided, otherwise load the default TACK dataset
    if custom_dataset_csv is not None:
        logger.debug(f"Loading custom dataset from CSV file: {custom_dataset_csv}")
        df = pd.read_csv(custom_dataset_csv)
        ds = Dataset.from_pandas(df, preserve_index=False)
    else:
        ds_config = 'default'
        if 'dmax' in task:
            ds_config = 'Dmax'
        elif 'dc50' in task:
            ds_config = 'DC50'
        elif task == 'bin' or (task == 'multitask' and model_type != 'bert'):
            ds_config = 'multitask'
        ds = load_dataset("ailab-bio/TACK", ds_config, split="train")

    # Process dataset labels based on task
    labels = ['Value']
    if task == 'dmax':
        ds = ds.rename_column('Value', 'Dmax')
        labels = ['Dmax']
    elif task == 'dc50':
        ds = ds.rename_column('Value', 'DC50')
        labels = ['DC50']
    elif task == 'bin':
        ds = ds.map(map_bin_labels)
        ds = ds.filter(lambda x: pd.notna(x['Activity']))
        labels = ['Activity']
    elif task == 'dmax_bin':
        ds = ds.rename_column('Value', 'Dmax')
        ds = ds.map(map_bin_dmax_labels)
        ds = ds.filter(lambda x: pd.notna(x['Activity']))
        labels = ['Activity']
    elif task == 'dc50_bin':
        ds = ds.rename_column('Value', 'DC50')
        ds = ds.map(map_bin_dc50_labels)
        ds = ds.filter(lambda x: pd.notna(x['Activity']))
        labels = ['Activity']
    elif task == 'multitask' and model_type != 'bert':
        # Rename 'Value_Dmax' and 'Value_DC50' to 'Dmax' and 'DC50'
        # NOTE: This is needed for better reporting when collecting metrics
        ds = ds.rename_column('Value_Dmax', 'Dmax')
        ds = ds.rename_column('Value_DC50', 'DC50')
        labels = ['Dmax', 'DC50']
    elif task == 'multitask' and model_type == 'bert':
        raise ValueError(
            "Multitask training with BERT model is not supported anymore. One "
            "needs to update the handling the label normalization of the "
            "'Value' column."
        )

    # Clip 'Dmax' values to [0, 100]
    # NOTE: There is no need to clip before calculating binary activity, since
    # an entry will be labeled in the same way regardless, as Dmax < 0 is always
    # below the 80% threshold, and Dmax > 100 is always above it. Clipping is
    # only needed for regression tasks to avoid outliers dominating the training.
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
    logger.debug(f"Label distribution for task {task}:")
    if task != 'multitask':
        for label in labels:
            logger.debug(f"[TRAIN/VAL] {label} - mean: {df[label].mean()}, std: {df[label].std()}, min: {df[label].min()}, max: {df[label].max()}")
            logger.debug(f"[HELD-OUT]  {label} - mean: {held_out_df[label].mean()}, std: {held_out_df[label].std()}, min: {held_out_df[label].min()}, max: {held_out_df[label].max()}")

    return ds, held_out_ds, labels
