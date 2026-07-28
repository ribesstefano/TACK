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
from tackai.data.ids import assign_dataset_ids
from tackai.training.splitting import assign_group_column, assign_held_out_column

logger = logging.getLogger(__name__)


def _ensure_label_column(ds: Dataset, source: str, target: str) -> Dataset:
    """Ensure ``ds`` exposes a ``target`` label column.

    The published TACK dataset stores regression labels under a generic
    ``Value`` column that is renamed per task, whereas a raw dataset such as
    TACK2.0 already carries the task-named columns (``Dmax`` / ``DC50``). This
    renames ``source`` to ``target`` only when needed, so both layouts work.

    Args:
        ds: Dataset to inspect.
        source: Generic column name to rename from (e.g. ``Value``).
        target: Task-specific label column expected downstream (e.g. ``Dmax``).

    Returns:
        The dataset with a ``target`` column.

    Raises:
        ValueError: If neither ``source`` nor ``target`` is present.
    """
    if target in ds.column_names:
        return ds
    if source in ds.column_names:
        return ds.rename_column(source, target)
    raise ValueError(
        f"Dataset must contain a '{target}' or '{source}' column; "
        f"found columns: {ds.column_names}"
    )


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
    # Prefer the multitask 'Value_*' columns (published TACK) but fall back to
    # the native 'Dmax'/'DC50' columns of a raw dataset such as TACK2.0.
    row = {
        'Dmax': example.get('Value_Dmax', example.get('Dmax', np.nan)),
        'DC50': example.get('Value_DC50', example.get('DC50', np.nan)),
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
    group: str = 'random',
    smiles_col: str = 'SMILES',
    held_out_frac: float = 0.10,
    held_out_seed: int = 42,
) -> Tuple[Dataset, Dataset, List[str]]:
    """ Load and prepare the dataset for a given task.

    Loads either a custom CSV or the published TACK dataset, applies the
    task-specific label processing (column renames, binary labelling, Dmax
    clipping), isolates the held-out set, and derives the CV grouping column.

    When the dataset lacks the split columns of the published TACK dataset
    (``SMILES_Held_Out`` and the per-``group`` cluster column) — as a raw
    TACK2.0 CSV does — they are computed on the fly from the SMILES structures
    (see :mod:`tackai.training.splitting`): the held-out set is isolated first
    with a MaxMin diverse pick, then the remaining train/validation rows are
    clustered for the requested ``group``. Datasets that already carry these
    columns are left untouched.

    Args:
        task: Task to train on ('dmax', 'dc50', 'bin', 'dmax_bin', 'dc50_bin',
            'multitask').
        model_type: Model type ('mlp', 'bert', 'xgboost'); affects dataset
            configuration selection and multitask handling.
        custom_dataset_csv: Optional path to a CSV that overrides the default
            TACK dataset. Regression labels may be provided either as a generic
            ``Value`` column or as native ``Dmax`` / ``DC50`` columns.
        group: CV grouping strategy ('random', 'scaffold', 'butina'); selects
            which cluster column to derive when it is absent.
        smiles_col: Name of the SMILES column used for on-the-fly splitting.
        held_out_frac: Fraction of unique compounds held out by the MaxMin pick
            (only used when ``SMILES_Held_Out`` is absent).
        held_out_seed: Fixed seed for the MaxMin held-out pick, kept independent
            of the model seed so the held-out set is identical across runs.

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
        ds = _ensure_label_column(ds, 'Value', 'Dmax')
        # Keep only rows that actually carry the target. The published 'Dmax'
        # config is already pre-filtered, but a raw TACK2.0 CSV mixes rows that
        # report only DC50 (null Dmax), which have no regression label.
        ds = ds.filter(lambda x: pd.notna(x['Dmax']))
        labels = ['Dmax']
    elif task == 'dc50':
        ds = _ensure_label_column(ds, 'Value', 'DC50')
        ds = ds.filter(lambda x: pd.notna(x['DC50']))
        labels = ['DC50']
    elif task == 'bin':
        ds = ds.map(map_bin_labels)
        ds = ds.filter(lambda x: pd.notna(x['Activity']))
        labels = ['Activity']
    elif task == 'dmax_bin':
        ds = _ensure_label_column(ds, 'Value', 'Dmax')
        ds = ds.map(map_bin_dmax_labels)
        ds = ds.filter(lambda x: pd.notna(x['Activity']))
        labels = ['Activity']
    elif task == 'dc50_bin':
        ds = _ensure_label_column(ds, 'Value', 'DC50')
        ds = ds.map(map_bin_dc50_labels)
        ds = ds.filter(lambda x: pd.notna(x['Activity']))
        labels = ['Activity']
    elif task == 'multitask' and model_type != 'bert':
        # Rename 'Value_Dmax'/'Value_DC50' to 'Dmax'/'DC50' for metric reporting;
        # a native TACK2.0 CSV already exposes these columns directly.
        ds = _ensure_label_column(ds, 'Value_Dmax', 'Dmax')
        ds = _ensure_label_column(ds, 'Value_DC50', 'DC50')
        labels = ['Dmax', 'DC50']
    elif task == 'multitask' and model_type == 'bert':
        raise ValueError(
            "Multitask training with BERT model is not supported anymore. One "
            "needs to update the handling the label normalization of the "
            "'Value' column."
        )

    df = ds.to_pandas()

    # Attach the stable content-hash identifiers (context_id / row_id) before
    # any splitting or label derivation so every fold and the held-out set carry
    # the same ids. The hash is taken over the RAW readouts (see ids.py), so it
    # must precede the Dmax clipping below — otherwise row_id would depend on the
    # derived [0, 100] value and stop being reproducible from the source dataset.
    df = assign_dataset_ids(df)

    # Clip 'Dmax' values to [0, 100].
    # NOTE: There is no need to clip before calculating binary activity, since
    # an entry will be labeled in the same way regardless, as Dmax < 0 is always
    # below the 80% threshold, and Dmax > 100 is always above it. Clipping is
    # only needed for regression tasks to avoid outliers dominating the training.
    # Null Dmax is valid in multitask (a row may report only DC50); clip() leaves
    # NaN untouched, so clipping stays a regression-only concern.
    if 'Dmax' in labels:
        df['Dmax'] = df['Dmax'].clip(lower=0, upper=100)

    # Isolate the held-out set first, then derive the CV grouping column. When
    # the dataset already carries these columns (published TACK) the helpers are
    # no-ops; for a raw TACK2.0 CSV they are computed on the fly from the SMILES
    # structures. The grouping column is derived for BOTH splits so the fold
    # DatasetDict has a consistent schema across train/validation and test.
    df = assign_held_out_column(
        df, smiles_col=smiles_col, frac=held_out_frac, seed=held_out_seed,
    )
    df, held_out_df = split_held_out(df)
    df = assign_group_column(df, group=group, smiles_col=smiles_col)
    held_out_df = assign_group_column(held_out_df, group=group, smiles_col=smiles_col)

    ds = Dataset.from_pandas(df, preserve_index=False)
    held_out_ds = Dataset.from_pandas(held_out_df, preserve_index=False)

    # Print the distribution of labels
    logger.debug(f"Label distribution for task {task}:")
    if task != 'multitask':
        for label in labels:
            logger.debug(f"[TRAIN/VAL] {label} - mean: {df[label].mean()}, std: {df[label].std()}, min: {df[label].min()}, max: {df[label].max()}")
            logger.debug(f"[HELD-OUT]  {label} - mean: {held_out_df[label].mean()}, std: {held_out_df[label].std()}, min: {held_out_df[label].min()}, max: {held_out_df[label].max()}")

    return ds, held_out_ds, labels
