"""
PyTorch Lightning (MLP / BERT) training and Optuna hyperparameter tuning for
PROTAC degradation prediction models.
"""
import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import optuna
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from tackai import DegradationComplexDataModule, TACKModel
from tackai.config import save_config_to_yaml


def train_lightning_model(
    data_module: DegradationComplexDataModule,
    model_config: Dict[str, Any],
    experiment_name: str,
    checkpoint_dir: Optional[Path] = None,
    model_type: Literal['mlp', 'bert'] = 'bert',
    label_names: List[str] = ['Value'],
) -> pl.LightningModule:
    """ Train a BERT-based or MLP regression model.

    Args:
        data_module: Data module with tokenized inputs.
        config: BERT configuration (BERTConfig or dict).
        checkpoint_dir: Directory to save checkpoints.
        experiment_name: Name for this experiment.
        model_type: Type of model ('mlp' or 'bert').
        label_names: List of label names.

    Returns:
        Trained PyTorch Lightning module.
    """
    # Create model
    model = TACKModel(
        model_type=model_type,
        label_names=label_names,
        **model_config['model_config'],
    )

    # REFERENCE: https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/03-initialization-and-optimization.html
    def kaiming_init(model):
        for name, param in model.named_parameters():
            if name.endswith(".bias"):
                param.data.fill_(0)
            elif name.startswith("model.mlp.0"):  # The first layer does not have ReLU applied on its input
                if len(param.shape) >= 2:
                    param.data.normal_(0, 1 / np.sqrt(param.shape[1]))
            else:
                if len(param.shape) >= 2:
                    param.data.normal_(0, np.sqrt(2) / np.sqrt(param.shape[1]))

    kaiming_init(model)

    # Setup default training config
    training_config = model_config.get('training_config', {
        'max_epochs': 10,
        'patience': 10,
    })

    # Setup callbacks and logger
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=training_config.pop('patience', 5),
            mode='min',
        ),
    ]
    logger = None

    if checkpoint_dir is not None:
        callbacks.append(ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f'{experiment_name}',
            monitor='val_loss',
            mode='min',
            save_top_k=1,
        ))

        logger = TensorBoardLogger(
            save_dir=checkpoint_dir / 'logs',
            name=experiment_name,
        )

    # Setup default training config values, then pass it to Trainer
    # NOTE: If the key does not exist, insert the key with default value
    training_config.setdefault('enable_progress_bar', False)
    training_config.setdefault('gradient_clip_val', 1.0)
    training_config.setdefault('accelerator', 'auto')
    training_config.setdefault('devices', 1)
    training_config.setdefault('precision', '16-mixed')
    training_config.setdefault('max_epochs', 5)

    # Create trainer
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        **training_config,
    )

    # Train
    trainer.fit(model, data_module)

    return model


def tune_lightning_hyperparameters(
    data_module: DegradationComplexDataModule,
    checkpoint_dir: Path,
    experiment_name: str,
    label_names: List[str],
    n_trials: int = 50,
    base_config: Optional[Dict[str, Any]] = None,
    model_type: Literal['mlp', 'bert'] = 'mlp',
    seed: int = 42,
) -> Dict[str, Any]:
    """Tune MLP/BERT hyperparameters using Optuna.

    Args:
        data_module: Configured data module.
        checkpoint_dir: Directory to save final best config (not trial checkpoints).
        experiment_name: Base name for experiment.
        label_names: List of label names.
        n_trials: Number of optimization trials.
        base_config: Base configuration to start from.
        model_type: Type of model ('mlp' or 'bert').

    Returns:
        Optimized configuration dictionary.
    """
    base_config = base_config.copy() if base_config is not None else {}

    print(f'\n{"="*70}')
    print(f"Starting hyperparameter tuning for {model_type.upper()}")
    print(f"Number of trials: {n_trials}")
    print(f"Base config: {base_config}")
    print(f'{"="*70}\n')

    # Track best validation loss across all trials
    best_val_loss = float('inf')
    best_trial_config = None

    def objective(trial):
        """Optuna objective function."""
        nonlocal best_val_loss, best_trial_config

        # Create a deep copy of base config for this trial
        trial_config = copy.deepcopy(base_config)

        # Ensure nested dictionaries exist
        if 'model_config' not in trial_config:
            trial_config['model_config'] = {}
        if 'training_config' not in trial_config:
            trial_config['training_config'] = {}

        model_config = trial_config['model_config']
        train_config = trial_config['training_config']

        # Get task type from base config (don't override it)
        task_type = model_config.get('task_type', 'point')

        # Sample hyperparameters based on model type
        if model_type == 'mlp':
            # Architecture hyperparameters
            hidden_dim_options = [
                [256],
                [512],
                [256, 128],
                [512, 256],
                [512, 256, 128],
                [1024, 512],
                [1024, 512, 256],
            ]
            model_config['hidden_dim'] = trial.suggest_categorical('hidden_dim', hidden_dim_options)
            model_config['dropout'] = trial.suggest_float('dropout', 0.0, 0.5, step=0.1)
            model_config['norm_type'] = trial.suggest_categorical('norm_type', [None, 'batch', 'layer'])
            model_config['activation'] = trial.suggest_categorical('activation', ['relu', 'gelu', 'silu'])
            # mlp_depth is determined by len(hidden_dim), so remove it if present
            model_config.pop('mlp_depth', None)
            if task_type != 'bin':
                model_config['head_dropout'] = trial.suggest_float('head_dropout', 0.0, 0.3, step=0.1)
                model_config['head_depth'] = trial.suggest_int('head_depth', 1, 3)

        elif model_type == 'bert':
            # BERT has fewer tunable architecture parameters
            model_config['dropout'] = trial.suggest_float('dropout', 0.0, 0.5, step=0.1)
            model_config['head_dropout'] = trial.suggest_float('head_dropout', 0.0, 0.3, step=0.1)
            model_config['head_depth'] = trial.suggest_int('head_depth', 1, 3)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # "Training" hyperparameters (common to both MLP and BERT)
        model_config['learning_rate'] = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        model_config['lr_scheduler_type'] = trial.suggest_categorical('lr_scheduler_type', ['cosine', 'reduce_on_plateau'])
        model_config['warmup_ratio'] = trial.suggest_float('warmup_ratio', 0.0, 0.2, step=0.05)

        # For cosine scheduler, suggest num_cycles
        if model_config['lr_scheduler_type'] == 'cosine':
            model_config['num_cycles'] = trial.suggest_int('num_cycles', 1, 10)
        else:
            model_config.pop('num_cycles', None)

        # Override training config for trials (disable checkpointing, logging, etc.)
        train_config['gradient_clip_val'] = trial.suggest_float('gradient_clip_val', 0.5, 2.0, step=0.5)
        train_config['patience'] = 5  # Early stopping patience
        train_config['enable_checkpointing'] = False
        train_config['enable_progress_bar'] = False
        train_config['enable_model_summary'] = False

        # Preserve other training config values from base config
        train_config.setdefault('max_epochs', base_config.get('training_config', {}).get('max_epochs', 200))
        train_config.setdefault('accelerator', 'auto')
        train_config.setdefault('devices', 1)
        train_config.setdefault('precision', base_config.get('training_config', {}).get('precision', '16-mixed'))

        # Add input_dim for MLP (this is essential and should come from data_module)
        if model_type == 'mlp':
            feature_dim = data_module.get_total_feature_dim()
            vocab_sizes = data_module.get_categorical_vocab_sizes()
            if vocab_sizes:
                model_config['categorical_embedding_dim'] = trial.suggest_categorical(
                    'categorical_embedding_dim', [4, 8, 16, 32])
                model_config['categorical_vocab_sizes'] = vocab_sizes
                model_config['input_dim'] = feature_dim - len(vocab_sizes)
            else:
                model_config['input_dim'] = feature_dim

        try:
            # Train model using the existing train_lightning_model function
            # Pass None for checkpoint_dir to disable checkpointing
            model = train_lightning_model(
                data_module=data_module,
                model_config=trial_config,
                checkpoint_dir=None,  # No checkpoints for trials
                experiment_name=f"{experiment_name}_trial_{trial.number}",
                model_type=model_type,
                label_names=label_names,
            )

            # Get best validation loss from trainer
            if hasattr(model.trainer, 'callback_metrics'):
                val_loss = model.trainer.callback_metrics.get('val_loss', float('inf'))
                if isinstance(val_loss, torch.Tensor):
                    val_loss = val_loss.item()
            else:
                val_loss = float('inf')

            # Update best config if this trial is better
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_trial_config = trial_config.copy()
                print(f"\n New best trial {trial.number}: val_loss={val_loss:.4f}")

            # Clean up
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            return val_loss

        except Exception as e:
            logging.error(f"Trial {trial.number} failed with error: {e}")
            import traceback
            traceback.print_exc()
            return float('inf')

    sampler = optuna.samplers.TPESampler(
        seed=seed,
        n_startup_trials=min(max(int(n_trials * 0.2), 10), 30),  # 20% or 10-30
        multivariate=True,  # Model parameter interactions
        group=True,
        warn_independent_sampling=False,  # Suppress warnings
    )

    # Run optimization
    study = optuna.create_study(direction="minimize", sampler=sampler)

    # Use tqdm progress bar if available
    try:
        from tqdm.auto import tqdm
        with tqdm(total=n_trials, desc="Hyperparameter Tuning") as pbar:
            def callback(study, trial):
                pbar.update(1)
                pbar.set_postfix({"best_loss": f"{study.best_value:.4f}"})
            study.optimize(objective, n_trials=n_trials, callbacks=[callback])
    except ImportError:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Log best trial info
    print(f'\n{"="*70}')
    print(f"Hyperparameter tuning completed!")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best validation loss: {study.best_trial.value:.4f}")
    print(f"Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f'{"="*70}\n')

    # Restore training config for actual training (enable checkpointing, etc.)
    if best_trial_config is not None:
        best_trial_config['training_config']['enable_checkpointing'] = True
        best_trial_config['training_config']['enable_progress_bar'] = False
        best_trial_config['training_config']['enable_model_summary'] = True

        # Save best config to checkpoint directory
        best_config_path = checkpoint_dir / f"best_config-{experiment_name}.yaml"
        save_config_to_yaml(best_trial_config, best_config_path)
        print(f"Saved best config to: {best_config_path}")

        return best_trial_config
    else:
        # Fallback to base config if all trials failed
        print("Warning: All trials failed. Returning base config.")
        return base_config
