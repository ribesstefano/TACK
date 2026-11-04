"""
Ensemble Selection for Cross-Validated Models
"""
import warnings
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, log_loss


warnings.filterwarnings('ignore')


class EnsembleSelector:
    """Base class for ensemble selection methods."""

    def __init__(self, metric: str = 'rmse', verbose: bool = True) -> None:
        """
        Initialize the EnsembleSelector.

        Args:
            metric (str): Metric to use for evaluation ('rmse', 'mse', 'r2', 'log_loss', 'brier').
            verbose (bool): Whether to print verbose output.
        """
        self.metric = metric
        self.verbose = verbose
        self.ensemble_weights = {}

    def compute_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the performance metric between true and predicted values.

        Args:
            y_true (np.ndarray): Ground truth target values.
            y_pred (np.ndarray): Predicted target values.

        Returns:
            float: Computed metric value (lower is better for all supported metrics).

        Raises:
            ValueError: If an unknown metric is specified.
        """
        if self.metric == 'rmse':
            return np.sqrt(mean_squared_error(y_true, y_pred))
        elif self.metric == 'mse':
            return mean_squared_error(y_true, y_pred)
        elif self.metric == 'r2':
            return -r2_score(y_true, y_pred)  # Negative for minimization
        elif self.metric == 'log_loss':
            y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
            return log_loss(y_true, y_pred_clipped)
        elif self.metric == 'brier':
            y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
            return np.mean((y_true - y_pred_clipped) ** 2)
        else:
            raise ValueError(f"Unknown metric: {self.metric}. "
                           f"Supported: 'rmse', 'mse', 'r2', 'log_loss', 'brier'.")

    def greedy_selection(
        self,
        predictions_dict: Dict[str, np.ndarray],
        y_true: np.ndarray,
        n_iterations: int = 100,
        with_replacement: bool = True,
        sorted_init: int = 5,
        n_bags: int = 10,
        bag_fraction: float = 0.5,
    ) -> Dict[str, float]:
        """
        Perform Caruana-style greedy forward ensemble selection with enhancements.

        Args:
            predictions_dict (Dict[str, np.ndarray]): 
                Dictionary mapping model names to their prediction arrays (shape: [n_samples]).
            y_true (np.ndarray): 
                Ground truth target values (shape: [n_samples]).
            n_iterations (int, optional): 
                Maximum number of greedy selection iterations per bag. Default is 100.
            with_replacement (bool, optional): 
                If True, allows models to be selected multiple times (with replacement). Default is True.
            sorted_init (int, optional): 
                Number of top-performing models to use for sorted initialization (0 disables). Default is 5.
            n_bags (int, optional): 
                Number of bagging iterations (ensembles). If >1, bagged ensemble selection is used. Default is 10.
            bag_fraction (float, optional): 
                Fraction of models to sample in each bag (if n_bags > 1). Default is 0.5.

        Returns:
            Dict[str, float]: 
                Dictionary mapping model names to their normalized ensemble weights.
        """
        model_names = list(predictions_dict.keys())
        n_models = len(model_names)

        if n_bags > 1:
            # Bagged ensemble selection
            bag_ensembles = []
            for bag_idx in range(n_bags):
                # Random sample of models
                np.random.seed(bag_idx)
                bag_models = np.random.choice(
                    model_names,
                    size=max(1, int(n_models * bag_fraction)),
                    replace=False
                )
                bag_preds = {k: v for k, v in predictions_dict.items() if k in bag_models}

                # Run selection on this bag
                bag_ensemble = self._single_selection(
                    bag_preds, y_true, n_iterations, with_replacement, sorted_init
                )
                bag_ensembles.append(bag_ensemble)

            # Merge bag ensembles
            merged_weights = defaultdict(float)
            for ensemble in bag_ensembles:
                for model, weight in ensemble.items():
                    merged_weights[model] += weight

            # Normalize
            total = sum(merged_weights.values())
            return {k: v / total for k, v in merged_weights.items()} if total > 0 else {}
        else:
            return self._single_selection(
                predictions_dict, y_true, n_iterations, with_replacement, sorted_init
            )

    def _single_selection(
        self,
        predictions_dict: Dict[str, np.ndarray],
        y_true: np.ndarray,
        n_iterations: int,
        with_replacement: bool,
        sorted_init: int,
    ) -> Dict[str, float]:
        """
        Perform a single greedy forward ensemble selection run.

        Args:
            predictions_dict (Dict[str, np.ndarray]): 
                Dictionary mapping model names to their prediction arrays (shape: [n_samples]).
            y_true (np.ndarray): 
                Ground truth target values (shape: [n_samples]).
            n_iterations (int): 
                Maximum number of greedy selection iterations.
            with_replacement (bool): 
                If True, allows models to be selected multiple times (with replacement).
            sorted_init (int): 
                Number of top-performing models to use for sorted initialization (0 disables).

        Returns:
            Dict[str, float]: 
                Dictionary mapping model names to their normalized ensemble weights.
        """
        # A few important notes:
        # - the predictions are assumed to be numpy arrays of shape (n_samples,)
        # - the ensemble is represented as a dict mapping model names to counts (number of times selected)
        
        model_names = list(predictions_dict.keys())

        # Sorted initialization: start with top N models
        if sorted_init > 0:
            # Calculate initial scores per model
            initial_scores = {}
            for name in model_names:
                score = self.compute_metric(y_true, predictions_dict[name])
                initial_scores[name] = score

            # Select best models
            sorted_models = sorted(initial_scores.items(), key=lambda x: x[1])
            init_models = [name for name, _ in sorted_models[:sorted_init]]

            # Initialize ensemble with the top-sorted_init models
            ensemble = {m: 1 for m in init_models}
            current_pred = np.mean([predictions_dict[m] for m in init_models], axis=0)
            current_score = self.compute_metric(y_true, current_pred)
        else:
            ensemble = {}
            current_pred = np.zeros_like(y_true, dtype=float)
            current_score = float('inf')

        # Greedy selection
        best_overall_score = current_score
        best_overall_ensemble = ensemble.copy()

        for _ in range(n_iterations):
            best_model = None
            best_score = current_score

            # Try adding each model
            for model_name in model_names:
                ensemble_size = sum(ensemble.values())

                # New prediction if we add this model
                if ensemble_size == 0:
                    new_pred = predictions_dict[model_name]
                else:
                    new_pred = (current_pred * ensemble_size + predictions_dict[model_name]) / (ensemble_size + 1)

                score = self.compute_metric(y_true, new_pred)

                if score < best_score:
                    best_score = score
                    best_model = model_name

            # Add best model
            if best_model is not None:
                ensemble[best_model] = ensemble.get(best_model, 0) + 1
                ensemble_size = sum(ensemble.values())
                current_pred = (
                    current_pred * (ensemble_size - 1) + predictions_dict[best_model]
                ) / ensemble_size
                current_score = best_score

                # Track best ensemble
                if current_score < best_overall_score:
                    best_overall_score = current_score
                    best_overall_ensemble = ensemble.copy()
            else:
                # No improvement possible
                if not with_replacement:
                    break
                # With replacement, we keep going but performance plateaus

        # Normalize weights between 0 and 1
        total = sum(best_overall_ensemble.values())
        return {k: v / total for k, v in best_overall_ensemble.items()} if total > 0 else {}

    def predict(self, predictions_dict: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
        """
        Make prediction using ensemble weights.

        Args:
            predictions_dict (Dict[str, np.ndarray]): Dictionary mapping model names to their prediction arrays.
            weights (Dict[str, float]): Dictionary mapping model names to their ensemble weights.

        Returns:
            np.ndarray: Weighted ensemble prediction.
        """
        # pred = np.zeros_like(next(iter(predictions_dict.values())), dtype=float)
        pred = np.zeros_like(list(predictions_dict.values())[0], dtype=float)
        for model_name, weight in weights.items():
            pred += predictions_dict[model_name] * weight
        return pred


class NestedCVSelector(EnsembleSelector):
    """ Method 1: Nested Cross-Validation Ensemble Selection """
    
    def fit(self, data_dict, n_iterations=100):
        """
        For each outer fold, create ensemble using only models that didn't see that fold
        """
        self.fold_ensembles = {}
        n_folds = len(set(data_dict['val'].keys()))
        
        if self.verbose:
            print(f"\nMethod 1: Nested CV Ensemble Selection")
            print(f"Creating {n_folds} fold-specific ensembles...")
        
        for outer_fold in range(n_folds):
            if self.verbose:
                print(f"  Processing outer fold {outer_fold}...")
            
            # Get validation data for this fold
            val_data = data_dict['val'][outer_fold]
            
            # Split val data into selection (70%) and validation (30%)
            n_samples = len(val_data)
            indices = np.random.RandomState(outer_fold).permutation(n_samples)
            split_idx = int(0.7 * n_samples)
            sel_indices = indices[:split_idx]
            
            # Get eligible models (those that didn't train on this fold)
            eligible_predictions = {}
            for model_name, fold_preds in data_dict['val'].items():
                # Extract fold number from model_name (assumed format)
                model_fold = int(model_name)
                if model_fold != outer_fold:
                    eligible_predictions[model_name] = fold_preds[sel_indices]
            
            y_true = val_data[sel_indices]
            
            # Run greedy selection
            weights = self.greedy_selection(eligible_predictions, y_true, 
                                          n_iterations=n_iterations,
                                          n_bags=10)
            
            self.fold_ensembles[outer_fold] = weights
            
            if self.verbose:
                n_models = len(weights)
                print(f"    Selected {n_models} models")
        
        return self
    
    def predict(self, predictions_dict):
        """Average predictions from all fold ensembles"""
        # Collect predictions from each fold ensemble
        fold_predictions = []
        for outer_fold, weights in self.fold_ensembles.items():
            fold_pred = super().predict(predictions_dict, weights)
            fold_predictions.append(fold_pred)
        
        return np.mean(fold_predictions, axis=0)


class HeldOutTestSelector(EnsembleSelector):
    """ Method 2: Held-Out Test Set Ensemble Selection """
    
    def fit(self, test_predictions, test_targets, n_iterations=100):
        """
        Use held-out test set, split 50-50 for selection and validation
        """
        if self.verbose:
            print(f"\nMethod 2: Held-Out Test Set Ensemble Selection")
        
        # Split test set 50-50
        n_samples = len(test_targets)
        indices = np.random.RandomState(42).permutation(n_samples)
        split_idx = n_samples // 2
        sel_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        # Prepare selection data
        sel_predictions = {k: v[sel_indices] for k, v in test_predictions.items()}
        y_sel = test_targets[sel_indices]
        
        # Run selection
        self.ensemble_weights = self.greedy_selection(sel_predictions, y_sel,
                                                     n_iterations=n_iterations,
                                                     n_bags=10)
        
        # Evaluate on validation half
        val_predictions = {k: v[val_indices] for k, v in test_predictions.items()}
        y_val = test_targets[val_indices]
        val_pred = super().predict(val_predictions, self.ensemble_weights)
        val_score = self.compute_metric(y_val, val_pred)
        
        if self.verbose:
            print(f"  Selected {len(self.ensemble_weights)} models")
            print(f"  Validation score: {val_score:.4f}")
        
        return self
    
    def predict(self, predictions_dict):
        return super().predict(predictions_dict, self.ensemble_weights)


class OutOfFoldSelector(EnsembleSelector):
    """Method 3: Out-of-Fold Predictions for Selection"""
    
    def fit(self, all_predictions, all_targets, fold_map, n_iterations=100):
        """
        Use out-of-fold predictions for selection
        fold_map: dict mapping sample index to fold number for each model
        """
        if self.verbose:
            print(f"\nMethod 3: Out-of-Fold Predictions Ensemble Selection")
        
        # This is complex - need to track which predictions are OOF for each sample
        # For simplicity, we'll use average OOF performance for selection
        # In practice, you'd want more sophisticated bookkeeping
        
        model_names = list(all_predictions.keys())
        n_models = len(model_names)
        
        # For each model, compute OOF score
        oof_scores = {}
        for model_name in model_names:
            # Model trained on folds != model_fold
            model_fold = int(model_name)
            mask = fold_map != model_fold
            if np.sum(mask) > 0:
                oof_pred = all_predictions[model_name][mask]
                oof_true = all_targets[mask]
                oof_scores[model_name] = self.compute_metric(oof_true, oof_pred)
            else:
                oof_scores[model_name] = float('inf')
        
        # Simple approach: weight by inverse OOF score
        inverse_scores = {k: 1.0 / (v + 1e-10) for k, v in oof_scores.items()}
        total = sum(inverse_scores.values())
        self.ensemble_weights = {k: v/total for k, v in inverse_scores.items()}
        
        if self.verbose:
            print(f"  Using {len(self.ensemble_weights)} models")
        
        return self
    
    def predict(self, predictions_dict):
        return super().predict(predictions_dict, self.ensemble_weights)


class ArchitectureLevelSelector(EnsembleSelector):
    """Method 4: Architecture-Level Selection (Simplest)"""
    
    def fit(self, architecture_predictions, test_targets, n_iterations=50):
        """
        Select at architecture level (each architecture's CV models are pre-averaged)
        """
        if self.verbose:
            print(f"\nMethod 4: Architecture-Level Ensemble Selection")
        
        # Split test set for selection/validation
        n_samples = len(test_targets)
        indices = np.random.RandomState(42).permutation(n_samples)
        split_idx = n_samples // 2
        sel_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        sel_predictions = {k: v[sel_indices] for k, v in architecture_predictions.items()}
        y_sel = test_targets[sel_indices]
        
        # Run selection on architecture meta-models
        self.ensemble_weights = self.greedy_selection(sel_predictions, y_sel,
                                                     n_iterations=n_iterations,
                                                     n_bags=10)
        
        # Evaluate
        val_predictions = {k: v[val_indices] for k, v in architecture_predictions.items()}
        y_val = test_targets[val_indices]
        val_pred = super().predict(val_predictions, self.ensemble_weights)
        val_score = self.compute_metric(y_val, val_pred)
        
        if self.verbose:
            print(f"  Selected {len(self.ensemble_weights)} architectures")
            print(f"  Validation score: {val_score:.4f}")
        
        return self
    
    def predict(self, predictions_dict):
        return super().predict(predictions_dict, self.ensemble_weights)


# Main execution will be in a separate script
if __name__ == "__main__":
    print("Ensemble Selection Module Loaded")
    print("This module provides 4 ensemble selection methods:")
    print("  1. NestedCVSelector - Nested cross-validation approach")
    print("  2. HeldOutTestSelector - Held-out test set approach")  
    print("  3. OutOfFoldSelector - Out-of-fold predictions approach")
    print("  4. ArchitectureLevelSelector - Architecture-level selection")
