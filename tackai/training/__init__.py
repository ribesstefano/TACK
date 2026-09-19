"""
Training utilities for PROTAC degradation prediction models.

The implementation is split across focused modules:

- ``xgboost_models``: XGBoost training and Optuna tuning.
- ``lightning_models``: MLP / BERT (PyTorch Lightning) training and tuning.
- ``cross_validation``: repeated k-fold splitting and the nested CV orchestrator.
- ``persistence``: data-module and prediction-output serialization.

The public API is re-exported here so that ``from tackai.training import ...``
keeps working as before.
"""
import logging
import warnings

import optuna

from tackai.training.xgboost_models import (
    gradient_gaussian_nll,
    train_xgboost,
    tune_xgboost_hyperparameters,
)
from tackai.training.lightning_models import (
    train_lightning_model,
    tune_lightning_hyperparameters,
)
from tackai.training.cross_validation import (
    create_cv_splits,
    run_cv_experiment,
)
from tackai.training.persistence import (
    save_data_module,
    load_data_module_state,
    save_predictions,
    collect_predictions,
    combined_predictions_path,
    fold_predictions_exist,
)
from tackai.training.splitting import (
    assign_group_column,
    assign_held_out_column,
    bemis_murcko_clusters,
    butina_clusters,
    maxmin_held_out_mask,
)

# Suppress Optuna warnings
warnings.filterwarnings("ignore")
optuna.logging.disable_default_handler()
logging.getLogger("optuna").setLevel(logging.CRITICAL)

__all__ = [
    "gradient_gaussian_nll",
    "train_xgboost",
    "tune_xgboost_hyperparameters",
    "train_lightning_model",
    "tune_lightning_hyperparameters",
    "create_cv_splits",
    "run_cv_experiment",
    "save_data_module",
    "load_data_module_state",
    "save_predictions",
    "collect_predictions",
    "combined_predictions_path",
    "fold_predictions_exist",
    "assign_group_column",
    "assign_held_out_column",
    "bemis_murcko_clusters",
    "butina_clusters",
    "maxmin_held_out_mask",
]
