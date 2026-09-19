"""
Nested cross-validation orchestration for PROTAC degradation prediction models.
"""
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
import xgboost as xgb
import pytorch_lightning as pl
from datasets import Dataset, DatasetDict
from sklearn.model_selection import GroupKFold, KFold

from tackai import DegradationComplexDataModule, TACKModel
from tackai.config import save_config_to_yaml, load_config_from_yaml
from tackai.inference import (
    get_xgboost_predictions,
    get_lightning_model_predictions,
)
from tackai.training.xgboost_models import (
    train_xgboost,
    tune_xgboost_hyperparameters,
)
from tackai.training.lightning_models import (
    train_lightning_model,
    tune_lightning_hyperparameters,
)
from tackai.training.persistence import (
    save_data_module,
    save_predictions,
    collect_predictions,
    fold_predictions_exist,
)
from tackai.training.splitting import GROUP_TO_COLUMN


def create_cv_splits(
    ds: Union[Dataset, pd.DataFrame],
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
                "cv_fold": fold,
                "fold": repeat * n_splits + fold,
                "train_idx": train_idx,
                "test_idx": test_idx,
            }


def run_cv_experiment(
    dataset: Dataset,
    held_out_dataset: Dataset,
    data_config: Dict[str, Any],
    model_type: Literal['xgboost', 'mlp', 'bert'],
    model_config: Dict[str, Any],
    task_name: Literal['dmax', 'dc50', 'multitask'],
    group: Literal['random', 'scaffold', 'butina'] = 'random',
    n_splits: int = 5,
    n_repeats: int = 5,
    labels: List[str] = ['Value'],
    checkpoints_dir: Path = Path('checkpoints'),
    results_dir: Path = Path('results'),
    tune_hyperparameters: bool = False,
    n_tuning_trials: int = 20,
    tune_first_fold_only: bool = True,
):
    """ Run a complete 5x5 cross-validation experiment.

    Args:
        dataset: Dataset for training/validation.
        held_out_dataset: Held-out test dataset.
        data_config: Data configuration dictionary.
        model_type: Type of model to train.
        model_config: Model configuration dictionary.
        task_name: Task name ('dmax', 'dc50', 'multitask').
        group: Grouping strategy for CV splits.
        n_splits: Number of folds.
        n_repeats: Number of repeats.
        labels: List of label names.
        checkpoints_dir: Directory to save model checkpoints.
        results_dir: Directory to save prediction results.
        tune_hyperparameters: Whether to tune hyperparameters.
        n_tuning_trials: Number of tuning trials.
        tune_first_fold_only: Whether to tune only on the first fold.

    Returns:
        List of results dictionaries for each fold.
    """
    # Setup directories
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Determine group column (canonical mapping lives in splitting.py).
    group_col = GROUP_TO_COLUMN.get(group)

    # Initialize results storage
    all_results = []

    # Track optimized config for reuse across folds
    optimized_config = None

    # Iterate over CV splits
    for split in create_cv_splits(dataset, n_splits, n_repeats, group_col):
        fold_id = split['fold']

        pl.seed_everything(42 + fold_id)

        # Create data module
        fold_dataset = DatasetDict({
            'train': dataset.select(split['train_idx']),
            'validation': dataset.select(split['test_idx']),
            'test': held_out_dataset,
        })

        # Manually force 'is_bert_multitask' if the task is multitask and model is BERT
        if task_name == 'multitask' and model_type == 'bert':
            data_config['is_bert_multitask'] = True
        else:
            data_config['is_bert_multitask'] = False

        # Create the data module
        data_module = DegradationComplexDataModule(
            dataset=fold_dataset,
            labels=labels,
            verbose=0,
            **data_config,
        )

        # Get data config name from data module's __str__ method
        data_name = str(data_module)

        # Update model name based on model type and special objectives/losses
        xgb_objective = model_config['model_config'].get('objective')
        mlp_loss = model_config['model_config'].get('task_type')
        if xgb_objective == 'reg:quantileerror':
            model_name = f"{model_type}_qr_{task_name}_protac-data={data_name}"
        elif (xgb_objective == 'reg:mve') or (mlp_loss == 'mve'):
            model_name = f"{model_type}_mve_{task_name}_protac-data={data_name}"
        else:
            model_name = f"{model_type}_{task_name}_protac-data={data_name}"

        print(f"\n{'='*70}")
        print(f"Task: {task_name} | Group: {group} | Fold: {fold_id} | Data: {data_name}")
        print(f"{'='*70}")

        # Check if the predictions for this fold already exist, if so, skip it.
        # Looks at both the per-fold files (present mid-run) and the collapsed
        # combined file (present after a completed run, once per-fold files
        # were removed by collect_predictions).
        tasks = ['dmax', 'dc50'] if task_name == 'multitask' else [task_name]
        if fold_predictions_exist(results_dir, model_name, tasks, group, fold_id):
            print(f"Predictions for fold {fold_id} already exist. Skipping...")
            continue

        # Setup data module if we need to train/evaluate the model for this fold
        data_module.setup()

        # Save data module hyperparameters and state after setup
        dm_save_name = f"datamodule-data={data_name}-group={group}-fold={fold_id}"
        save_data_module(data_module, checkpoints_dir, dm_save_name)
        print(f"Saved data module config to: {checkpoints_dir / dm_save_name}_hparams.yaml")

        # Load existing optimized config if available (first fold only)
        optimized_config_path = checkpoints_dir / f"config-model={model_name}-task={task_name}-group={group}.yaml"
        if optimized_config is None:
            if optimized_config_path.exists():
                optimized_config = load_config_from_yaml(optimized_config_path)
                print(f"Loaded existing optimized XGBoost config from: {optimized_config_path}")

        # Setup the checkpoint path template for the models
        checkpoint_path = checkpoints_dir / f"model={model_name}-group={group}-fold={fold_id}"

        if model_type == 'xgboost':
            # Get DMatrix data
            if model_config['model_config']['objective'] == 'reg:quantileerror':
                dtrain = data_module.get_xgboost_dataset('train', quantile_matrix=True)
                dval = data_module.get_xgboost_dataset('validation', quantile_matrix=True, dtrain=dtrain)
                dtest = data_module.get_xgboost_dataset('test', quantile_matrix=True, dtrain=dtrain)
            else:
                dtrain = data_module.get_xgboost_dataset('train')
                dval = data_module.get_xgboost_dataset('validation')
                dtest = data_module.get_xgboost_dataset('test')

            checkpoint_path = checkpoint_path.with_suffix('.json')

            if checkpoint_path.exists():
                print(f"Loading existing model from: {checkpoint_path}")
                model = xgb.Booster()
                model.load_model(checkpoint_path)
            else:
                current_config = model_config.copy()

                if tune_hyperparameters:
                    if tune_first_fold_only:
                        if optimized_config is None:
                            print(f"Tuning hyperparameters on first fold ({n_tuning_trials} trials)...")
                            optimized_config = tune_xgboost_hyperparameters(
                                dtrain, dval, n_trials=n_tuning_trials, base_config=model_config, seed=42 + fold_id
                            )
                        current_config = optimized_config
                    else:
                        current_config = tune_xgboost_hyperparameters(
                            dtrain, dval, n_trials=n_tuning_trials, base_config=model_config, seed=42 + fold_id
                        )

                    # Save optimized config to YAML
                    save_config_to_yaml(current_config, optimized_config_path)
                    print(f"Saved optimized config to: {optimized_config_path}")

                # Change 'seed' to ensure different seeds per fold
                current_config['model_config']['seed'] = 42 + fold_id

                print("Training model...")
                model = train_xgboost(dtrain, dval, current_config)
                model.save_model(checkpoint_path)
                print(f"Saved model to: {checkpoint_path}")

            predictions = get_xgboost_predictions(model, dval, dtest, data_module, fold_dataset)

        elif model_type in ['bert', 'mlp']:
            checkpoint_path = checkpoint_path.with_suffix('.ckpt')

            if checkpoint_path.exists():
                print(f"Loading existing model from: {checkpoint_path}")
                model = TACKModel.load_from_checkpoint(checkpoint_path)
            else:
                current_config = model_config.copy()

                # Hyperparameter tuning for MLP/BERT
                if tune_hyperparameters:
                    if tune_first_fold_only:
                        if optimized_config is None:
                            print(f"\n{'='*70}")
                            print(f"Tuning {model_type.upper()} hyperparameters on first fold")
                            print(f"Number of trials: {n_tuning_trials}")
                            print(f"{'='*70}\n")

                            optimized_config = tune_lightning_hyperparameters(
                                data_module=data_module,
                                checkpoint_dir=checkpoints_dir,
                                experiment_name=f"model={model_name}-task={task_name}-group={group}",
                                label_names=labels,
                                n_trials=n_tuning_trials,
                                base_config=model_config,
                                model_type=model_type,
                                seed=42 + fold_id,
                            )
                        current_config = optimized_config
                    else:
                        # Tune on every fold
                        print(f"\nTuning {model_type.upper()} hyperparameters on fold {fold_id}...")
                        current_config = tune_lightning_hyperparameters(
                            data_module=data_module,
                            checkpoint_dir=checkpoints_dir,
                            experiment_name=f"model={model_name}-task={task_name}-group={group}-fold={fold_id}",
                            label_names=labels,
                            n_trials=n_tuning_trials,
                            base_config=model_config,
                            model_type=model_type,
                            seed=42 + fold_id,
                        )

                    # Save optimized config to YAML
                    save_config_to_yaml(current_config, optimized_config_path)
                    print(f"Saved optimized config to: {optimized_config_path}")

                # Set input_dim for MLP
                if model_type == 'mlp':
                    feature_dim = data_module.get_total_feature_dim()
                    vocab_sizes = data_module.get_categorical_vocab_sizes()
                    if vocab_sizes:
                        current_config['model_config']['categorical_vocab_sizes'] = vocab_sizes
                        current_config['model_config']['input_dim'] = feature_dim - len(vocab_sizes)
                    else:
                        current_config['model_config']['input_dim'] = feature_dim

                print(f"Training {model_type.upper()} model...")
                model = train_lightning_model(
                    data_module=data_module,
                    model_config=current_config,
                    experiment_name=f"model={model_name}-group={group}-fold={fold_id}",
                    checkpoint_dir=checkpoints_dir,
                    model_type=model_type,
                    label_names=labels,
                )

            print("Getting predictions...")
            predictions = get_lightning_model_predictions(model, data_module)
        else:
            raise ValueError(f"Unknown model type: {model_type}, must be one of ['xgboost', 'mlp', 'bert']")

        # Save predictions
        save_predictions(
            predictions=predictions,
            data_module=data_module,
            model_type=model_type,
            model_name=model_name,
            task_name=task_name,
            group=group,
            fold_id=fold_id,
            results_dir=results_dir,
            fold_dataset=fold_dataset,
        )

        # Store results
        all_results.append({
            'task': task_name,
            'group': group,
            'fold': fold_id,
            'data_config': data_name,
            'model_type': model_type,
            'predictions': predictions,
        })

    # Consolidate every per-fold prediction CSV of this run into a single
    # combined file for convenience/resume, but KEEP the per-fold '-split=' files
    # in place: ensemble selection (scripts/ensemble_comparison.py) parses the
    # split label from those filenames, and removing them would leave it with
    # nothing to load. Deleting them is left to an explicit `tack collect`.
    # Non-fatal: a run that trained models successfully should not fail just
    # because the roll-up step did (e.g. all folds were skipped).
    try:
        collect_predictions(results_dir, remove_sources=False)
    except FileNotFoundError:
        print(f"No per-fold prediction files found in {results_dir}; "
              "skipping collection.")
    except Exception as exc:  # noqa: BLE001 - roll-up must not abort a run
        print(f"Warning: failed to collect predictions in {results_dir}: {exc}")

    return all_results
