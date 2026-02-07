"""
Inference utilities for PROTAC degradation predictor models.

Authors: Stefano Ribes
"""
from typing import Dict, Optional, Union, Any

import torch
import numpy as np
import xgboost as xgb
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets import DatasetDict, Dataset

from staeda import DegradationComplexDataModule

def get_xgboost_predictions(
    model: xgb.Booster,
    dval: xgb.DMatrix,
    dtest: xgb.DMatrix,
    data_module: Optional[DegradationComplexDataModule] = None,
    dataset: Optional[Union[DatasetDict, Dict[str, Dataset]]] = None,
) -> Dict[str, np.ndarray]:
    """Get predictions from XGBoost model.
    
    Args:
        model: Trained XGBoost Booster.
        dval: Validation data.
        dtest: Test data.
        data_module: Data module for inverse label transformation.
        dataset: Original dataset for retrieving true, un-normalized labels.
        
    Returns:
        Dictionary with val_preds, val_targets, test_preds, test_targets.
    """
    # Check if this is a quantile regression model
    is_quantile_model = False
    is_mve_model = False
    
    # Also check by looking at prediction shape
    val_preds_raw = model.predict(dval)
    test_preds_raw = model.predict(dtest)
    
    # If predictions have multiple columns, it's likely quantile regression
    if val_preds_raw.ndim > 1 and val_preds_raw.shape[1] > 2:
        is_quantile_model = True
        # Use median (middle quantile, typically index 1 for [0.05, 0.5, 0.95])
        median_idx = val_preds_raw.shape[1] // 2
        val_preds = val_preds_raw[:, median_idx]
        test_preds = test_preds_raw[:, median_idx]
    elif val_preds_raw.ndim > 1 and val_preds_raw.shape[1] > 1:
        is_mve_model = True
        # For MVE, use the mean predictions (first column)
        val_preds = val_preds_raw[:, 0]
        test_preds = test_preds_raw[:, 0]
    else:
        val_preds = val_preds_raw.flatten() if val_preds_raw.ndim > 1 else val_preds_raw
        test_preds = test_preds_raw.flatten() if test_preds_raw.ndim > 1 else test_preds_raw
    
    val_targets = dval.get_label()
    test_targets = dtest.get_label()
        
    if dataset is not None:
        val_dataset = dataset['validation']
        test_dataset = dataset['test']
    else:
        val_dataset = None
        test_dataset = None

    ret = {
        'val_preds': val_preds,
        'val_targets': val_targets,
        'test_preds': test_preds,
        'test_targets': test_targets,
        'is_quantile_model': is_quantile_model,
        'is_mve_model': is_mve_model,
    }
    
    # Store raw quantile predictions if available
    if is_quantile_model and val_preds_raw.ndim > 1:
        ret['val_preds_quantiles'] = val_preds_raw
        ret['test_preds_quantiles'] = test_preds_raw
        if data_module is not None:
            ret['val_preds_quantiles_lower'] = val_preds_raw[:, 0]
            ret['val_preds_quantiles_upper'] = val_preds_raw[:, -1]
            ret['test_preds_quantiles_lower'] = test_preds_raw[:, 0]
            ret['test_preds_quantiles_upper'] = test_preds_raw[:, -1]
            # Normalize quantiles (there must be only one label, since XGBoost
            # quantile regression only supports single-output)
            label = data_module.labels[0]
            ret['val_preds_quantiles_lower'] = data_module.inverse_transform_labels(
                ret['val_preds_quantiles_lower'], label
            ).flatten()
            ret['val_preds_quantiles_upper'] = data_module.inverse_transform_labels(
                ret['val_preds_quantiles_upper'], label
            ).flatten()
            ret['test_preds_quantiles_lower'] = data_module.inverse_transform_labels(
                ret['test_preds_quantiles_lower'], label
            ).flatten()
            ret['test_preds_quantiles_upper'] = data_module.inverse_transform_labels(
                ret['test_preds_quantiles_upper'], label
            ).flatten()
    
    # Denormalize using data module, if applicable
    if data_module is not None:
        # Reshape predictions and targets for inverse transformation
        num_labels = len(data_module.labels)
        val_preds = val_preds.reshape(-1, num_labels)
        test_preds = test_preds.reshape(-1, num_labels)
        val_targets = val_targets.reshape(-1, num_labels)
        test_targets = test_targets.reshape(-1, num_labels)
        
        for i, label in enumerate(data_module.labels):
            val_preds[:, i] = data_module.inverse_transform_labels(val_preds[:, i], label).flatten()
            test_preds[:, i] = data_module.inverse_transform_labels(test_preds[:, i], label).flatten()

            # Use the actual labels from data module datasets, if provided
            if val_dataset is not None and test_dataset is not None:
                val_targets[:, i] = np.array(val_dataset[label]).flatten()
                test_targets[:, i] = np.array(test_dataset[label]).flatten()
            else:
                val_targets[:, i] = data_module.inverse_transform_labels(val_targets[:, i], label).flatten()
                test_targets[:, i] = data_module.inverse_transform_labels(test_targets[:, i], label).flatten()
            
            ret[f'val_preds_{label}'] = val_preds[:, i].flatten()
            ret[f'val_targets_{label}'] = val_targets[:, i].flatten()
            ret[f'test_preds_{label}'] = test_preds[:, i].flatten()
            ret[f'test_targets_{label}'] = test_targets[:, i].flatten()

        if num_labels == 1:
            # Flatten back for single-label case for allowing reporting to CSV
            val_preds = val_preds.flatten()
            test_preds = test_preds.flatten()
            val_targets = val_targets.flatten()
            test_targets = test_targets.flatten()

        ret['val_preds'] = val_preds
        ret['val_targets'] = val_targets
        ret['test_preds'] = test_preds
        ret['test_targets'] = test_targets

    return ret

def get_lightning_model_predictions(
    model: pl.LightningModule,
    data_module: DegradationComplexDataModule,
    data_loader: Optional[DataLoader] = None,
) -> Dict[str, np.ndarray]:
    """Get predictions from BERT/MLP model.
    
    Args:
        model: Trained PyTorch Lightning module.
        data_module: Data module.
        data_loader: Optional custom data loader.
        
    Returns:
        Dictionary with predictions and targets.
    """
    # Create trainer for prediction
    trainer = pl.Trainer(accelerator='auto', devices=1, enable_progress_bar=False)

    if data_loader is None:
        # Get predictions
        val_preds = trainer.predict(model, data_module.val_dataloader())
        test_preds = trainer.predict(model, data_module.test_dataloader())
        
        # Flatten predictions
        val_preds_tensor = torch.cat([x['pred'] for x in val_preds])
        test_preds_tensor = torch.cat([x['pred'] for x in test_preds])
        
        # Get uncertainty if available
        if model.task_type == 'mve':
            val_uct_tensor = torch.cat([x['pred_uct'] for x in val_preds])
            test_uct_tensor = torch.cat([x['pred_uct'] for x in test_preds])
        
        # Get targets from dataloader
        val_targets = torch.cat([batch[data_module.labels[0]] for batch in data_module.val_dataloader()]).cpu().numpy()
        test_targets = torch.cat([batch[data_module.labels[0]] for batch in data_module.test_dataloader()]).cpu().numpy()
        
        ret = {}
        
        if len(data_module.labels) > 1:
            # Multi-task MLP: multiple labels predicted simultaneously
            for idx, label in enumerate(data_module.labels):
                val_preds_np = val_preds_tensor[:, idx].cpu().numpy().flatten()
                test_preds_np = test_preds_tensor[:, idx].cpu().numpy().flatten()
                val_targets_np = torch.cat([batch[label] for batch in data_module.val_dataloader()]).cpu().numpy()
                test_targets_np = torch.cat([batch[label] for batch in data_module.test_dataloader()]).cpu().numpy()
                
                if model.task_type == 'mve':
                    val_uct_np = val_uct_tensor[:, idx].cpu().numpy().flatten()
                    test_uct_np = test_uct_tensor[:, idx].cpu().numpy().flatten()
                
                # Denormalize if needed
                if data_module.normalize_labels:
                    val_preds_np = data_module.inverse_transform_labels(val_preds_np, label).flatten()
                    test_preds_np = data_module.inverse_transform_labels(test_preds_np, label).flatten()
                    val_targets_np = data_module.inverse_transform_labels(val_targets_np, label).flatten()
                    test_targets_np = data_module.inverse_transform_labels(test_targets_np, label).flatten()
                    
                    if model.task_type == 'mve':
                        val_preds_np, val_uct_np = recover_moments(
                            val_preds_np, val_uct_np,
                            data_module.label_transformers[label],
                        )
                        test_preds_np, test_uct_np = recover_moments(
                            test_preds_np, test_uct_np,
                            data_module.label_transformers[label],
                        )
                    
                ret[f'val_preds_{label}'] = val_preds_np
                ret[f'test_preds_{label}'] = test_preds_np
                ret[f'val_targets_{label}'] = val_targets_np
                ret[f'test_targets_{label}'] = test_targets_np
                
                if model.task_type == 'mve':
                    ret[f'val_uct_{label}'] = val_uct_np
                    ret[f'test_uct_{label}'] = test_uct_np
        else:
            # Single-label model
            val_preds_np = val_preds_tensor.cpu().numpy().flatten()
            test_preds_np = test_preds_tensor.cpu().numpy().flatten()
            
            if model.task_type == 'mve':
                val_uct_np = val_uct_tensor.cpu().numpy().flatten()
                test_uct_np = test_uct_tensor.cpu().numpy().flatten()
            
            # Check if this is BERT multi-task (single label but multiple Value_Types)
            if data_module.normalize_labels and data_module.is_bert_multitask:
                # BERT multi-task: denormalize based on Value_Type
                val_value_types = data_module.val_tasks
                test_value_types = data_module.test_tasks
                
                # Initialize output arrays
                val_preds_denorm = np.zeros_like(val_preds_np)
                test_preds_denorm = np.zeros_like(test_preds_np)
                val_targets_denorm = np.zeros_like(val_targets)
                test_targets_denorm = np.zeros_like(test_targets)
                
                for value_type in ['Dmax', 'DC50']:
                    if value_type not in data_module.label_transformers:
                        continue
                        
                    # Validation set
                    val_mask = np.array([vt == value_type for vt in val_value_types])
                    if val_mask.any():
                        val_preds_denorm[val_mask] = data_module.inverse_transform_labels(
                            val_preds_np[val_mask], value_type
                        ).flatten()
                        val_targets_denorm[val_mask] = data_module.inverse_transform_labels(
                            val_targets[val_mask], value_type
                        ).flatten()
                    
                    # Test set
                    test_mask = np.array([vt == value_type for vt in test_value_types])
                    if test_mask.any():
                        test_preds_denorm[test_mask] = data_module.inverse_transform_labels(
                            test_preds_np[test_mask], value_type
                        ).flatten()
                        test_targets_denorm[test_mask] = data_module.inverse_transform_labels(
                            test_targets[test_mask], value_type
                        ).flatten()
                
                val_preds_np = val_preds_denorm
                test_preds_np = test_preds_denorm
                val_targets = val_targets_denorm
                test_targets = test_targets_denorm
                
            elif data_module.normalize_labels:
                # Single-task: use the label name for denormalization
                label = data_module.labels[0]
                val_preds_np = data_module.inverse_transform_labels(val_preds_np, label).flatten()
                test_preds_np = data_module.inverse_transform_labels(test_preds_np, label).flatten()
                val_targets = data_module.inverse_transform_labels(val_targets, label).flatten()
                test_targets = data_module.inverse_transform_labels(test_targets, label).flatten()
                
                if model.task_type == 'mve':
                    val_preds_np, val_uct_np = recover_moments(
                        val_preds_np, val_uct_np,
                        data_module.label_transformers[label],
                    )
                    test_preds_np, test_uct_np = recover_moments(
                        test_preds_np, test_uct_np,
                        data_module.label_transformers[label],
                    )
            
            ret['val_preds'] = val_preds_np
            ret['test_preds'] = test_preds_np
            ret['val_targets'] = val_targets
            ret['test_targets'] = test_targets
            
            if model.task_type == 'mve':
                ret['val_uct'] = val_uct_np
                ret['test_uct'] = test_uct_np

        return ret
    else:
        # Get predictions from provided data_loader
        preds = trainer.predict(model, data_loader)
        preds_tensor = torch.cat([x['pred'] for x in preds])
        
        ret = {}

        if len(data_module.labels) > 1:
            for idx, label in enumerate(data_module.labels):
                preds_np = preds_tensor[:, idx].cpu().numpy().flatten()
                targets_np = torch.cat([batch[label] for batch in data_loader]).cpu().numpy()
                
                if data_module.normalize_labels:
                    preds_np = data_module.inverse_transform_labels(preds_np, label).flatten()
                    targets_np = data_module.inverse_transform_labels(targets_np, label).flatten()
                
                ret[f'preds_{label}'] = preds_np
                ret[f'targets_{label}'] = targets_np
        else:
            preds_np = preds_tensor.cpu().numpy().flatten()
            targets_np = torch.cat([batch[data_module.labels[0]] for batch in data_loader]).cpu().numpy()
            
            if data_module.normalize_labels:
                label = data_module.labels[0]
                preds_np = data_module.inverse_transform_labels(preds_np, label).flatten()
                targets_np = data_module.inverse_transform_labels(targets_np, label).flatten()
            
            ret['preds'] = preds_np
            ret['targets'] = targets_np

        return ret

def recover_moments(
    pred_means: np.array,
    pred_vars: np.array,
    transformer: Any,
    n_samples: int = 1000,
) -> tuple:
    """
    Efficiently recovers mean and variance in original space for N predictions.
    
    Args:
        pred_means (np.array): Shape (N,), predicted means in transformed space.
        pred_vars (np.array): Shape (N,), predicted variances in transformed space.
        transformer: Fitted sklearn QuantileTransformer.
        n_samples (int): Number of Monte Carlo samples per prediction.
        
    Returns:
        means_orig (np.array): Shape (N,), estimated means in original space.
        vars_orig (np.array): Shape (N,), estimated variances in original space.
    """
    # Ensure inputs are numpy arrays
    pred_means = np.asanyarray(pred_means)
    pred_vars = np.asanyarray(pred_vars)
    N = len(pred_means)
    
    # Generate standard normal noise for all N predictions at once
    # Shape: (N, n_samples)
    rng = np.random.default_rng()
    noise = rng.standard_normal((N, n_samples))
    
    # Scale and shift to get samples in transformed space
    # Use broadcasting: (N, 1) + (N, n_samples) * (N, 1)
    samples_trans = pred_means[:, None] + noise * np.sqrt(pred_vars[:, None])
    
    # Inverse transform
    # The transformer expects shape (total_samples, n_features)
    # Flatten the matrix to fit this, assuming n_features=1 (target variable).
    samples_trans_flat = samples_trans.reshape(-1, 1)
    
    # Inverse transform all samples at once
    samples_orig_flat = transformer.inverse_transform(samples_trans_flat)
    
    # Reshape back to (N, n_samples) to separate the groups again
    samples_orig = samples_orig_flat.reshape(N, n_samples)
    
    # Calculate metrics along axis 1 (across the samples for each prediction)
    means_orig = np.mean(samples_orig, axis=1)
    vars_orig = np.var(samples_orig, axis=1)
    
    return means_orig, vars_orig

def predict_with_uncertainty(X: np.ndarray, models: list) -> tuple:
    """ Predict with an ensemble of single-output XGBoost models and estimate uncertainty.
    
    Args:
        X: Array-like of shape (n_samples, n_features) or (n_features,) for a single sample.
        models: List of trained XGBoost models (Booster or sklearn wrapper) that output a single value.
        
    Returns:
        (mean_predictions, std_predictions) where both are numpy arrays of shape (n_samples,).
        std_predictions is the standard deviation of model predictions (ensemble uncertainty).
    """
    if not models:
        raise ValueError("models list must not be empty")

    X_arr = np.asarray(X)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(1, -1)

    dmat = xgb.DMatrix(X_arr)
    preds_list = []

    for m in models:
        try:
            preds = m.predict(dmat)
        except Exception:
            # fallback for wrappers that expect ndarray input
            preds = m.predict(X_arr)
        preds_list.append(np.asarray(preds).flatten())

    # Ensure all model predictions have the same length
    lengths = [p.shape[0] for p in preds_list]
    if len(set(lengths)) != 1:
        raise ValueError(f"Inconsistent prediction lengths from models: {lengths}")

    preds_stack = np.vstack(preds_list)  # shape: (n_models, n_samples)
    mean_preds = preds_stack.mean(axis=0)
    std_preds = preds_stack.std(axis=0)

    return mean_preds, std_preds