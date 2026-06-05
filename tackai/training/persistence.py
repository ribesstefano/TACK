"""
Persistence helpers for data modules and cross-validation prediction outputs.
"""
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from datasets import DatasetDict

from tackai import DegradationComplexDataModule
from tackai.config import save_config_to_yaml


def save_data_module(
    data_module: DegradationComplexDataModule,
    save_dir: Path,
    name: str,
) -> None:
    """Save data module hyperparameters and state to files.

    Args:
        data_module: The data module to save.
        save_dir: Directory to save the files.
        name: Base name for the saved files.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save hyperparameters as YAML
    hparams_path = save_dir / f"{name}_hparams.yaml"
    hparams = data_module.get_hyperparameters()
    save_config_to_yaml(hparams, hparams_path)

    # Save state dictionary (label transformers, etc.)
    state_path = save_dir / f"{name}_state.pt"
    state_dict = data_module.state_dict()
    torch.save(state_dict, state_path)


def load_data_module_state(
    data_module: DegradationComplexDataModule,
    save_dir: Path,
    name: str,
) -> DegradationComplexDataModule:
    """Load data module state from saved files.

    Args:
        data_module: The data module to load state into.
        save_dir: Directory containing the saved files.
        name: Base name of the saved files.

    Returns:
        Data module with loaded state.
    """
    state_path = save_dir / f"{name}_state.npz"
    if state_path.exists():
        state_dict = dict(np.load(state_path, allow_pickle=True))
        data_module.load_state_dict(state_dict)
    return data_module


def save_predictions(
    predictions: Dict[str, np.ndarray],
    data_module: DegradationComplexDataModule,
    model_type: str,
    model_name: str,
    task_name: str,
    group: str,
    fold_id: int,
    results_dir: Path,
    fold_dataset: DatasetDict,
) -> None:
    """Save prediction results to CSV files.

    Args:
        predictions: Dictionary with predictions and targets.
        data_module: Data module with task information.
        model_type: Type of model.
        model_name: Name of the model.
        task_name: Name of the task.
        group: Grouping strategy.
        fold_id: Fold identifier.
        results_dir: Directory to save results.
    """
    if task_name == 'multitask' and len(data_module.labels) > 1:
        # Multi-task MLP: save separate files for each label
        for label in data_module.labels:
            for split_name in ['val', 'test']:
                preds_key = f'{split_name}_preds_{label}'
                targets_key = f'{split_name}_targets_{label}'

                if preds_key not in predictions:
                    print(f"Warning: Predictions for key '{preds_key}' not found. Skipping...")
                    continue

                preds_df = pd.DataFrame({
                    'target': predictions[targets_key],
                    'pred': predictions[preds_key],
                    'fold': [fold_id] * len(predictions[preds_key]),
                    'group': [group] * len(predictions[preds_key]),
                    'method': [model_name] * len(predictions[preds_key]),
                    'task': [label.lower()] * len(predictions[preds_key]),
                })

                preds_path = results_dir / f"preds-model={model_name}-task={label.lower()}-group={group}-fold={fold_id}-split={split_name}.csv"
                preds_df.to_csv(preds_path, index=False)
                print(f"Saved {label} predictions to: {preds_path}")

    elif model_type == 'bert' and data_module.val_tasks is not None:
        # BERT with Value_Type: split by Value_Type
        val_value_types = data_module.val_tasks
        test_value_types = data_module.test_tasks

        for split_name, value_types in [('val', val_value_types), ('test', test_value_types)]:
            preds_df = pd.DataFrame({
                'value_type': value_types,
                'target': predictions[f'{split_name}_targets'],
                'pred': predictions[f'{split_name}_preds'],
                'fold': [fold_id] * len(predictions[f'{split_name}_preds']),
                'group': [group] * len(predictions[f'{split_name}_preds']),
                'method': [model_name] * len(predictions[f'{split_name}_preds']),
            })

            for value_type in ['Dmax', 'DC50']:
                type_df = preds_df[preds_df['value_type'] == value_type].copy()
                if len(type_df) == 0:
                    continue

                type_df['task'] = value_type.lower()
                type_df = type_df.drop(columns=['value_type'])

                preds_path = results_dir / f"preds-model={model_name}-task={value_type.lower()}-group={group}-fold={fold_id}-split={split_name}.csv"
                type_df.to_csv(preds_path, index=False)
                print(f"Saved {value_type} predictions to: {preds_path}")
    else:
        # Single-task: save single file
        for split_name in ['val', 'test']:
            preds_df = pd.DataFrame({
                'target': predictions[f'{split_name}_targets'],
                'pred': predictions[f'{split_name}_preds'],
                'fold': [fold_id] * len(predictions[f'{split_name}_preds']),
                'group': [group] * len(predictions[f'{split_name}_preds']),
                'method': [model_name] * len(predictions[f'{split_name}_preds']),
                'task': [task_name] * len(predictions[f'{split_name}_preds']),
            })

            # Add uncertainty estimates if available: lower and upper quantiles
            if 'val_preds_quantiles_lower' in predictions and split_name == 'val':
                preds_df['pred_lower'] = predictions['val_preds_quantiles_lower']
                preds_df['pred_upper'] = predictions['val_preds_quantiles_upper']
            if 'test_preds_quantiles_lower' in predictions and split_name == 'test':
                preds_df['pred_lower'] = predictions['test_preds_quantiles_lower']
                preds_df['pred_upper'] = predictions['test_preds_quantiles_upper']

            # Add uncertainty estimates if available: predicted variance (for MVE)
            if 'val_uct' in predictions and split_name == 'val':
                preds_df['pred_variance'] = predictions['val_uct']
            if 'test_uct' in predictions and split_name == 'test':
                preds_df['pred_variance'] = predictions['test_uct']

            ds_split = 'validation' if split_name == 'val' else 'test'
            df = fold_dataset[ds_split].to_pandas()

            # Drop the column 'Description' for better readability
            df = df.drop(columns=['Description'], errors='ignore')

            if len(df) == len(preds_df):
                # Concatenate the two dataframes "side-by-side", to extend preds_df with original columns
                preds_df = pd.concat([
                    df.reset_index(drop=True),
                    preds_df.reset_index(drop=True)], axis=1)

            preds_path = results_dir / f"preds-model={model_name}-task={task_name}-group={group}-fold={fold_id}-split={split_name}.csv"
            preds_df.to_csv(preds_path, index=False)
            print(f"Saved predictions to: {preds_path}")
