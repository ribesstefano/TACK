"""
Training utilities for PROTAC degradation prediction models.
"""
import copy
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union, Literal, Any

import torch
import optuna
import numpy as np
import pandas as pd
import xgboost as xgb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from datasets import Dataset, DatasetDict
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import mean_squared_error

from tackai import DegradationComplexDataModule
from tackai import TACKModel
from tackai.config import (
    save_config_to_yaml,
    load_config_from_yaml,
)
from tackai.inference import (
    get_xgboost_predictions,
    get_lightning_model_predictions,
)

# Suppress Optuna warnings
warnings.filterwarnings("ignore")
optuna.logging.disable_default_handler()
logging.getLogger("optuna").setLevel(logging.CRITICAL)


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

def gradient_gaussian_nll(preds, dtrain):
    targets = dtrain.get_label()
    
    # Reshape preds to (N, 2) to handle mu and log_var separate columns
    preds = preds.reshape(-1, 2)
    mu = preds[:, 0]
    log_var = preds[:, 1]
    
    # --- Clip log_var ---
    # Keeps variance between e^-5 (~0.006) and e^5 (~148)
    # Since we are scaling data to Mean=0, Std=1, this is plenty of room.
    log_var = np.clip(log_var, -5.0, 5.0)
    var = np.exp(log_var)
    
    # Gradients
    # d(Loss)/d(mu)
    grad_mu = -(targets - mu) / var
    # d(Loss)/d(log_var)
    grad_var = 0.5 - 0.5 * (targets - mu)**2 / var
    
    # Hessians
    hess_mu = 1 / var
    # --- Floor Hessian ---
    hess_var = np.maximum(0.5 * (targets - mu)**2 / var, 1e-6)

    if any(dtrain.get_weight()):
        weights = dtrain.get_weight()
        grad_mu *= weights
        grad_var *= weights
        hess_mu *= weights
        hess_var *= weights
    
    # Reshape to (N, 2) to satisfy XGBoost warnings
    grad = np.column_stack([grad_mu, grad_var])
    hess = np.column_stack([hess_mu, hess_var])
    
    # Flatten if specific XGBoost version still strictly demands 1D
    # But usually (N, 2) is preferred now. If you get a shape error,
    # switch to: return grad.flatten(), hess.flatten()
    return grad, hess


def train_xgboost(
    dtrain: xgb.DMatrix,
    dval: xgb.DMatrix,
    config: Dict[str, Any],
) -> xgb.Booster:
    """ Train an XGBoost model.
    
    Args:
        dtrain: Training data as DMatrix.hg
        dval: Validation data as DMatrix.
        config: XGBoost configuration dictionary. Can contain 'model_config' (XGBoost parameters) and 'training_config'.
        
    Returns:
        Trained XGBoost Booster model.
    """
    if 'model_config' in config:
        params = config['model_config']
    else:
        params = config

    if 'training_config' in config:
        train_config = config['training_config']
    else:
        train_config = {}

    # Train with early stopping
    if params.get('objective') == 'reg:mve':
        # Use custom objective for Gaussian NLL
        params['objective'] = None  # Remove custom objective name
        params['disable_default_eval_metric'] = 1
        params['num_class'] = 2
        train_config.pop('early_stopping_rounds', None)  # No early stopping for custom objective
        model = xgb.train(
            params,
            dtrain,
            obj=gradient_gaussian_nll,
            num_boost_round=train_config.get('num_boost_round', 500),
            verbose_eval=False,
        )
        return model
    else:
        model = xgb.train(
            params,
            dtrain,
            evals=[(dval, 'validation')],
            num_boost_round=train_config.get('num_boost_round', 500),
            early_stopping_rounds=train_config.get('early_stopping_rounds', 20),
            verbose_eval=False,
        )
        return model


def tune_xgboost_hyperparameters(
    dtrain: xgb.DMatrix,
    dval: xgb.DMatrix,
    n_trials: int = 100,
    base_config: Optional[Dict[str, Any]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Tune XGBoost hyperparameters using Optuna.
    
    Args:
        dtrain: Training data as DMatrix.
        dval: Validation data as DMatrix.
        n_trials: Number of optimization trials.
        base_config: Base configuration to start from.
        
    Returns:
        Optimized XGBoostConfig.
    """
    base_config = {} if base_config is None else base_config
    
    if base_config and 'model_config' in base_config:
        model_config = base_config['model_config']
        # For quantile regression, set alpha if not already set
        if model_config['objective'] == 'reg:quantileerror':
            model_config['quantile_alpha'] = model_config.get('quantile_alpha', [0.05, 0.5, 0.95])
            model_config['tree_method'] = 'hist'
            model_config['eval_metric'] = None
        base_config['model_config'] = model_config
    
    print(f'Base config for tuning: {base_config}')
    
    def objective(trial):
        """Optuna objective function."""
        model_config = base_config.get('model_config', {})
        train_config = base_config.get('training_config', {})
        config = {
            'model_config': {
                'objective': model_config.get('objective', 'reg:squarederror'),
                'eval_metric': model_config.get('eval_metric', 'rmse'),
                'tree_method': model_config.get('tree_method', 'hist'),
                "n_estimators": 2000, # Needs more rounds because LR is low
                'seed': model_config.get('seed', 42) + trial.number,
                # Hyperparameters to tune
                'learning_rate': trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
                'max_depth': trial.suggest_int("max_depth", 3, 9),
                'min_child_weight': trial.suggest_int("min_child_weight", 1, 25),
                'subsample': trial.suggest_float("subsample", 0.4, 1.0),
                'colsample_bytree': trial.suggest_float("colsample_bytree", 0.5, 1.0),
                'reg_alpha': trial.suggest_float("reg_alpha", 0.001, 10.0, log=True),
                'reg_lambda': trial.suggest_float("reg_lambda", 0.001, 10.0, log=True),
                'gamma': trial.suggest_float("gamma", 0.001, 10.0, log=True),
            },
            'training_config': {
                'num_boost_round': train_config.get('num_boost_round', 100),
                'early_stopping_rounds': train_config.get('early_stopping_rounds', 5),
                'n_jobs': train_config.get('n_jobs', 1),
            }
        }
        # Remove suggested params from base config, then update it with
        # suggested config
        for param in ['learning_rate', 'max_depth', 'min_child_weight', 'subsample',
                      'colsample_bytree', 'reg_alpha', 'reg_lambda', 'gamma']:
            if param in model_config:
                del model_config[param]
        config['model_config'].update(model_config)
        
        model = train_xgboost(dtrain, dval, config)
        val_preds = model.predict(dval)

        if config['model_config']['objective'] == 'binary:logistic':
            # Binary classification
            val_preds = (val_preds >= 0.5).astype(int)
            accuracy = (val_preds == dval.get_label()).mean()
            return 1.0 - accuracy  # Minimize 1 - accuracy
        elif config['model_config']['objective'] == 'reg:quantileerror':
            # Predict on validation set
            # Get median predictions (assuming 0.5 quantile is in the middle)
            median_idx = len(config['model_config']['quantile_alpha']) // 2
            median_preds = val_preds[:, median_idx]
            rmse = np.sqrt(mean_squared_error(dval.get_label(), median_preds))
            return rmse
        elif config['model_config']['objective'] == 'reg:mve':
            val_preds = val_preds.reshape(-1, 2)
            mean_preds = val_preds[:, 0].flatten()
            rmse = np.sqrt(mean_squared_error(dval.get_label(), mean_preds))
        else:
            # Predict on validation set
            # NOTE: By default, the labels in a DMatrix are 1D arrays, so we need
            # to flatten any predictions from XGBoost, even in a multitask setting.
            rmse = np.sqrt(mean_squared_error(dval.get_label(), val_preds.flatten()))
            return rmse
    
    sampler = optuna.samplers.TPESampler(
        seed=seed,
        n_startup_trials=min(max(int(n_trials * 0.2), 10), 30),  # 20% or 10-30
        multivariate=True,  # Model parameter interactions
        group=True,
        warn_independent_sampling=False,  # Suppress warnings
    )

    # Run optimization
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Create optimized config
    optimized_config = base_config.copy()
    optimized_config['model_config'].update(study.best_params)
    
    return optimized_config


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
        tune_hyperparameters: Whether to tune hyperparameters (XGBoost only).
        n_tuning_trials: Number of tuning trials.
        tune_first_fold_only: Whether to tune only on the first fold.
        
    Returns:
        List of results dictionaries for each fold.
    """
    # Setup directories
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine group column
    group_col = None
    if group == 'scaffold':
        group_col = 'SMILES_Scaffold_Cluster'
    elif group == 'butina':
        group_col = 'SMILES_Butina_Cluster'
    
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
        
        # Check if the predictions for this fold already exist, if so, skip it
        skip_fold = True
        tasks = ['dmax', 'dc50'] if task_name == 'multitask' else [task_name]
        for task in tasks:
            for split_name in ['val', 'test']:
                preds_path = results_dir / f"preds-model={model_name}-task={task}-group={group}-fold={fold_id}-split={split_name}.csv"
                if not preds_path.exists():
                    skip_fold = False
                    break
                print(f"Found existing predictions at: {preds_path}")
            if not skip_fold:
                break
        if skip_fold:
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
            # ------------------------------------------------------------------
            # Old code block for training/loading Lightning models
            # ------------------------------------------------------------------
            # checkpoint_path = checkpoint_path.with_suffix('.ckpt')
            
            # if checkpoint_path.exists():
            #     print(f"Loading existing model from: {checkpoint_path}")
            #     model = TACKModel.load_from_checkpoint(checkpoint_path)
            # else:                
            #     if model_type == 'mlp':
            #         feature_dim = data_module.get_total_feature_dim()
            #         model_config['model_config']['input_dim'] = feature_dim

            #     print(f"Training {model_type.upper()} model...")
            #     model = train_lightning_model(
            #         data_module=data_module,
            #         model_config=model_config,
            #         checkpoints_dir=checkpoints_dir,
            #         experiment_name=f"model={model_name}-group={group}-fold={fold_id}",
            #         model_type=model_type,
            #         label_names=labels,
            #     )
            
            # print("Getting predictions...")
            # predictions = get_lightning_model_predictions(model, data_module)
            # ------------------------------------------------------------------
        else:
            raise ValueError(f"Unknown model type: {model_type}, must be one of ['xgboost', 'mlp', 'bert']")
        
        # Save predictions
        _save_predictions(
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
    
    return all_results


def _save_predictions(
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