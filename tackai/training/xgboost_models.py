"""
XGBoost training and Optuna hyperparameter tuning for PROTAC degradation
prediction models.
"""
from typing import Any, Dict, Optional

import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import mean_squared_error


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
                'n_jobs': model_config.get('n_jobs', 1),
            },
            'training_config': {
                'num_boost_round': train_config.get('num_boost_round', 100),
                'early_stopping_rounds': train_config.get('early_stopping_rounds', 5),
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
