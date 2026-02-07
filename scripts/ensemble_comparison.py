"""
Ensemble Selection Analysis
Ensures all methods evaluated on same unseen data
Supports regression (Dmax, DC50) and binary classification (bin) tasks
"""
from pathlib import Path
import argparse
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, 
    log_loss, 
    roc_auc_score,
    r2_score,
    brier_score_loss,
    accuracy_score,
)

# =============================================================================
# Caruana-style greedy forward ensemble selection method
# =============================================================================

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

# =============================================================================
# ENSEMBLE WEIGHTS I/O
# =============================================================================

def save_ensemble_weights(
    weights: Dict[str, float],
    output_path: Path,
    task: str,
    method_name: str,
    metric_value: float,
    metadata: Optional[Dict] = None,
) -> str:
    """
    Save ensemble weights to a JSON file.
    
    Args:
        weights: Dict of model_name -> weight
        output_path: Directory to save the weights file
        task: Task name (dmax, dc50, bin)
        method_name: Name of the selection method
        metric_value: Final metric value achieved
        metadata: Optional additional metadata
        
    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Clean method name for filename
    method_clean = method_name.lower().replace(' ', '_').replace('-', '_')
    filename = f"ensemble_weights_{task}_{method_clean}.json"
    filepath = output_path / filename
    
    # Build the weights document
    weights_doc = {
        "version": "1.0",
        "created_at": datetime.now().isoformat(),
        "task": task,
        "method": method_name,
        "metric_name": "log_loss" if task == "bin" else "rmse",
        "metric_value": float(metric_value),
        "n_models": len(weights),
        "weights": {k: float(v) for k, v in sorted(weights.items(), key=lambda x: -x[1])},
    }
    
    if metadata:
        weights_doc["metadata"] = metadata
    
    with open(filepath, 'w') as f:
        json.dump(weights_doc, f, indent=2)
    
    print(f"\n✅ Saved ensemble weights to: {filepath}")
    print(f"   Task: {task}, Method: {method_name}")
    print(f"   Models: {len(weights)}, Metric: {metric_value:.4f}")
    
    return str(filepath)


def load_ensemble_weights(weights_path: str) -> Tuple[Dict[str, float], Dict]:
    """
    Load ensemble weights from a JSON file.
    
    Args:
        weights_path: Path to the weights JSON file
        
    Returns:
        Tuple of (weights dict, metadata dict)
    """
    with open(weights_path, 'r') as f:
        doc = json.load(f)
    
    weights = doc.get("weights", {})
    metadata = {k: v for k, v in doc.items() if k != "weights"}
    
    return weights, metadata

matplotlib.use('Agg')
sns.set_style('whitegrid')


# =============================================================================
# METRICS AND UTILITIES
# =============================================================================

def dc50_to_pdc50(x: np.ndarray) -> np.ndarray:
    """Convert DC50 in nM to pDC50."""
    return -np.log10(x * 1e-9 + 1e-12)


def compute_metric(y_true: np.ndarray, y_pred: np.ndarray, task: str) -> float:
    """
    Compute task-appropriate metric (lower is better for all).
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values (probabilities for binary, values for regression)
        task: 'bin', 'dmax', or 'dc50'
    
    Returns:
        Metric value (lower is better)
    """
    if task == 'bin':
        # Clip probabilities to avoid log(0)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        return log_loss(y_true, y_pred_clipped)
    else:
        return np.sqrt(mean_squared_error(y_true, y_pred))


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, task: str) -> Dict[str, float]:
    """
    Compute all relevant metrics for the task.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        task: Task type
        
    Returns:
        Dictionary of metric name -> value
    """
    metrics = {}
    
    if task == 'bin':
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        y_pred_binary = (y_pred >= 0.5).astype(int)
        
        metrics['log_loss'] = log_loss(y_true, y_pred_clipped)
        metrics['brier_score'] = brier_score_loss(y_true, y_pred_clipped)
        metrics['accuracy'] = accuracy_score(y_true, y_pred_binary)
        
        # AUC-ROC (only if both classes present)
        if len(np.unique(y_true)) > 1:
            metrics['auc_roc'] = roc_auc_score(y_true, y_pred_clipped)
        else:
            metrics['auc_roc'] = np.nan
    else:
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        
        # R² score
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return metrics


# =============================================================================
# UNCERTAINTY QUANTIFICATION
# =============================================================================

def compute_ensemble_uncertainty(
    predictions_dict: Dict[str, np.ndarray],
    weights: Optional[Dict[str, float]] = None,
    task: str = 'dmax',
) -> Dict[str, np.ndarray]:
    """
    Compute uncertainty metrics for ensemble predictions.
    
    Args:
        predictions_dict: Model predictions
        weights: Optional ensemble weights (None = uniform)
        task: Task type for appropriate metrics
        
    Returns:
        Dictionary with uncertainty metrics per sample
    """
    all_preds = np.array(list(predictions_dict.values()))  # Shape: (n_models, n_samples)
    n_models, n_samples = all_preds.shape
    
    uncertainty = {}
    
    # 1. Prediction variance (disagreement between models)
    uncertainty['variance'] = np.var(all_preds, axis=0)
    uncertainty['std'] = np.std(all_preds, axis=0)
    
    # 2. Interquartile range (robust measure)
    q75 = np.percentile(all_preds, 75, axis=0)
    q25 = np.percentile(all_preds, 25, axis=0)
    uncertainty['iqr'] = q75 - q25
    
    # 3. Range (max - min prediction)
    uncertainty['range'] = np.max(all_preds, axis=0) - np.min(all_preds, axis=0)
    
    if task == 'bin':
        # 4. Entropy of average prediction (for classification)
        avg_pred = np.mean(all_preds, axis=0)
        avg_pred_clipped = np.clip(avg_pred, 1e-7, 1 - 1e-7)
        uncertainty['predictive_entropy'] = -(
            avg_pred_clipped * np.log(avg_pred_clipped) + 
            (1 - avg_pred_clipped) * np.log(1 - avg_pred_clipped)
        )
        
        # 5. Mutual information (epistemic uncertainty)
        # MI = H[y|x] - E[H[y|x, w]] where w are model weights
        individual_entropies = -(
            np.clip(all_preds, 1e-7, 1-1e-7) * np.log(np.clip(all_preds, 1e-7, 1-1e-7)) +
            np.clip(1-all_preds, 1e-7, 1-1e-7) * np.log(np.clip(1-all_preds, 1e-7, 1-1e-7))
        )
        expected_entropy = np.mean(individual_entropies, axis=0)
        uncertainty['mutual_information'] = uncertainty['predictive_entropy'] - expected_entropy
    
    return uncertainty


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainty: np.ndarray,
    task: str = 'dmax',
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Compute calibration metrics to assess uncertainty quality.
    
    Args:
        y_true: Ground truth
        y_pred: Ensemble predictions
        uncertainty: Uncertainty estimates (e.g., std)
        task: Task type
        n_bins: Number of bins for calibration
        
    Returns:
        Calibration metrics
    """
    metrics = {}
    
    if task == 'bin':
        # Expected Calibration Error (ECE)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            mask = (y_pred_clipped >= bin_boundaries[i]) & (y_pred_clipped < bin_boundaries[i+1])
            if np.sum(mask) > 0:
                bin_accuracy = np.mean(y_true[mask])
                bin_confidence = np.mean(y_pred_clipped[mask])
                bin_size = np.sum(mask) / len(y_true)
                ece += bin_size * np.abs(bin_accuracy - bin_confidence)
        
        metrics['ece'] = ece
        
        # Maximum Calibration Error
        mce = 0.0
        for i in range(n_bins):
            mask = (y_pred_clipped >= bin_boundaries[i]) & (y_pred_clipped < bin_boundaries[i+1])
            if np.sum(mask) > 0:
                bin_accuracy = np.mean(y_true[mask])
                bin_confidence = np.mean(y_pred_clipped[mask])
                mce = max(mce, np.abs(bin_accuracy - bin_confidence))
        metrics['mce'] = mce
        
    else:
        # For regression: check if uncertainty correlates with error
        errors = np.abs(y_true - y_pred)
        
        # Spearman correlation between uncertainty and error
        from scipy.stats import spearmanr
        corr, pval = spearmanr(uncertainty, errors)
        metrics['uncertainty_error_correlation'] = corr
        metrics['correlation_pvalue'] = pval
        
        # Calibration: fraction of samples within k*std
        for k in [1, 2, 3]:
            within_interval = np.abs(y_true - y_pred) <= k * uncertainty
            metrics[f'coverage_{k}sigma'] = np.mean(within_interval)
            # Expected coverage: 68.27%, 95.45%, 99.73% for 1,2,3 sigma
    
    return metrics


def analyze_uncertainty_quality(
    eval_predictions: Dict[str, np.ndarray],
    eval_targets: np.ndarray,
    ensemble_weights: Dict[str, float],
    method_name: str,
    task: str,
) -> Dict[str, float]:
    """
    Analyze uncertainty quality for an ensemble method.
    
    Args:
        eval_predictions: All model predictions on eval set
        eval_targets: True targets
        ensemble_weights: Selected ensemble weights
        method_name: Name of the method
        task: Task type
        
    Returns:
        Uncertainty quality metrics
    """
    # Get predictions for selected models only
    selected_preds = {k: v for k, v in eval_predictions.items() if k in ensemble_weights}
    
    # Compute uncertainty
    uncertainty = compute_ensemble_uncertainty(selected_preds, ensemble_weights, task)
    
    # Compute ensemble prediction
    ensemble_pred = np.zeros_like(eval_targets, dtype=float)
    for model_key, weight in ensemble_weights.items():
        if model_key in eval_predictions:
            ensemble_pred += eval_predictions[model_key] * weight
    
    # Compute calibration metrics
    calibration = compute_calibration_metrics(
        eval_targets, ensemble_pred, uncertainty['std'], task
    )
    
    # Summary statistics
    results = {
        'method': method_name,
        'n_models_selected': len(ensemble_weights),
        'mean_uncertainty': np.mean(uncertainty['std']),
        'median_uncertainty': np.median(uncertainty['std']),
        'uncertainty_range': np.max(uncertainty['std']) - np.min(uncertainty['std']),
    }
    results.update(calibration)
    
    return results, uncertainty


# =============================================================================
# DATA LOADING
# =============================================================================

def load_predictions_data(
    prediction_dir: str = 'predictions/',
    task: str = 'Dmax',
) -> Tuple[Dict, Dict, Dict, np.ndarray, List[str]]:
    """Load all prediction data."""
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    data_dir = Path(prediction_dir)
    files = sorted(data_dir.glob('preds-*.csv'))
    
    val_predictions = defaultdict(lambda: defaultdict(dict))
    test_predictions = {}
    val_targets = defaultdict(dict)
    test_targets = None
    
    architectures = set()
    task_lower = task.lower()
    
    for i, file in enumerate(files):
        if i % 50 == 0:
            print(f"  Processing {i+1}/{len(files)}")
        
        # Match task in filename
        if f'task={task_lower}' not in file.stem.lower():
            continue
        
        parts = file.stem.split('-')
        metadata = {}
        for part in parts:
            if '=' in part:
                key, value = part.split('=', 1)
                metadata[key] = value
        
        arch = f"model={metadata.get('model', '')}-data={metadata.get('data', '')}-group={metadata.get('group', 'all')}"
        fold = int(metadata.get('fold', 0))
        split = metadata.get('split', '')
        
        architectures.add(arch)
        
        df = pd.read_csv(file)
        preds = df['pred'].values
        targets = df['target'].values
        
        # Transform for DC50 task (regression)
        if task_lower == 'dc50':
            preds = dc50_to_pdc50(preds)
            targets = dc50_to_pdc50(targets)
        # For 'bin' task, predictions are already probabilities
        
        if split == 'val':
            val_predictions[arch][fold] = preds
            val_targets[arch][fold] = targets
        elif split == 'test':
            model_key = f"{arch}-fold={fold}"
            test_predictions[model_key] = preds
            
            if test_targets is None:
                test_targets = targets
    
    print(f"\nData loaded:")
    print(f"  Task: {task}")
    print(f"  Architectures: {len(architectures)}")
    print(f"  Total models: {len(test_predictions)}")
    if test_targets is not None:
        print(f"  Test set size: {len(test_targets)}")
        if task_lower == 'bin':
            print(f"  Class distribution: {np.mean(test_targets):.2%} positive")
    
    return val_predictions, val_targets, test_predictions, test_targets, list(architectures)


def split_test_set(
    test_predictions: Dict[str, np.ndarray],
    test_targets: np.ndarray,
    test_perc: float = 0.5,
    random_seed: int = 42,
) -> Tuple[Dict[str, any], Dict[str, any]]:
    """
    Split test set 50-50 for fair comparison.
    
    Args:
        test_predictions: Dict of model predictions on test set
        test_targets: True targets for test set
        random_seed: Random seed for reproducibility
    
    Returns:
        sel_data: Selection data (predictions and targets)
        eval_data: Evaluation data (predictions and targets)
    """
    n_samples = len(test_targets)
    indices = np.random.RandomState(random_seed).permutation(n_samples)
    split_idx = int(n_samples * test_perc)
    
    sel_indices = indices[:split_idx]
    eval_indices = indices[split_idx:]
    
    # Selection data (for ensemble selection/hillclimbing)
    sel_predictions = {k: v[sel_indices] for k, v in test_predictions.items()}
    sel_targets = test_targets[sel_indices]
    
    # Evaluation data (untouched by any method)
    eval_predictions = {k: v[eval_indices] for k, v in test_predictions.items()}
    eval_targets = test_targets[eval_indices]
    
    print(f"\nTest set split:")
    print(f"  Selection set: {len(sel_targets)} ({len(sel_targets)/n_samples*100:.1f}%) samples")
    print(f"  Evaluation set: {len(eval_targets)} ({len(eval_targets)/n_samples*100:.1f}%) samples")
    
    sel_data = {'predictions': sel_predictions, 'targets': sel_targets}
    eval_data = {'predictions': eval_predictions, 'targets': eval_targets}
    
    return sel_data, eval_data


# =============================================================================
# BASELINES AND METHODS
# =============================================================================

def compute_baselines(
    eval_predictions: Dict[str, np.ndarray],
    eval_targets: np.ndarray,
    architectures: List[str],
    task: str,
) -> Dict[str, float]:
    """Compute baselines on evaluation set ONLY."""
    print("\n" + "=" * 80)
    print(f"COMPUTING BASELINES (on evaluation set, task={task})")
    print("=" * 80)
    
    results = {}
    metric_name = 'Log Loss' if task == 'bin' else 'RMSE'
    
    # Baseline 1: Best single model
    best_score = float('inf')
    best_model = None
    
    for model_key, preds in eval_predictions.items():
        score = compute_metric(eval_targets, preds, task)
        if score < best_score:
            best_score = score
            best_model = model_key
    
    results['best_single'] = best_score
    print(f"\n1. Best Single Model: {best_model}")
    print(f"   {metric_name}: {best_score:.4f}")
    
    # Baseline 2: Average ALL models
    all_preds = np.array(list(eval_predictions.values()))
    avg_pred = np.mean(all_preds, axis=0)
    avg_score = compute_metric(eval_targets, avg_pred, task)
    
    results['average_all'] = avg_score
    print(f"\n2. Average All {len(eval_predictions)} Models:")
    print(f"   {metric_name}: {avg_score:.4f}")
    print(f"   vs Best: {((avg_score - best_score) / best_score * 100):+.2f}%")
    
    # Baseline 3: Best architecture (average of its folds)
    arch_scores = {}
    for arch in architectures:
        arch_preds = []
        for model_key, preds in eval_predictions.items():
            if model_key.startswith(arch):
                arch_preds.append(preds)
        
        if arch_preds:
            arch_avg = np.mean(arch_preds, axis=0)
            arch_score = compute_metric(eval_targets, arch_avg, task)
            arch_scores[arch] = arch_score
    
    best_arch = min(arch_scores, key=arch_scores.get)
    best_arch_score = arch_scores[best_arch]
    
    results['best_architecture'] = best_arch_score
    results['best_architecture_name'] = best_arch
    print(f"\n3. Best Architecture: {best_arch}")
    print(f"   {metric_name}: {best_arch_score:.4f}")
    
    return results


def run_caruana_method(
    sel_data: Dict[str, any],
    eval_data: Dict[str, any],
    task: str,
) -> Tuple[Dict[str, float], float]:
    """Caruana Train on selection set, evaluate on eval set."""
    print("\n" + "=" * 80)
    print("Caruana HELD-OUT TEST SET ENSEMBLE SELECTION")
    print("=" * 80)
    
    print(f"\nTraining on {len(sel_data['targets'])} selection samples")
    print(f"Will evaluate on {len(eval_data['targets'])} UNSEEN evaluation samples")
    
    # Use appropriate metric for selection
    metric = 'rmse' if task != 'bin' else 'rmse'  # Use RMSE for greedy selection even for binary
    # Note: For binary, we optimize on prediction space; log_loss computed post-hoc
    
    selector = EnsembleSelector(metric=metric, verbose=False)
    ensemble_weights = selector.greedy_selection(
        sel_data['predictions'],
        sel_data['targets'],
        n_iterations=100,
        n_bags=10,
    )
    
    print(f"\nSelected {len(ensemble_weights)} models")
    
    # Evaluate on UNSEEN evaluation set
    eval_pred = np.zeros_like(eval_data['targets'], dtype=float)
    for model_key, weight in ensemble_weights.items():
        if model_key in eval_data['predictions']:
            eval_pred += eval_data['predictions'][model_key] * weight
    
    final_score = compute_metric(eval_data['targets'], eval_pred, task)
    
    metric_name = 'Log Loss' if task == 'bin' else 'RMSE'
    print(f"\nFinal {metric_name} on UNSEEN evaluation set: {final_score:.4f}")
    
    # Show top models
    print(f"\nTop 10 selected models:")
    sorted_weights = sorted(ensemble_weights.items(), key=lambda x: x[1], reverse=True)
    for model, weight in sorted_weights[:10]:
        print(f"\tweight: {weight:.4f} -> {model}")
    
    return ensemble_weights, final_score


def run_architecture_caruana_method(
    sel_data: Dict[str, any],
    eval_data: Dict[str, any],
    architectures: List[str],
    task: str,
) -> Tuple[Dict[str, float], float]:
    """ Architecture-level Ensemble selection."""
    print("\n" + "=" * 80)
    print("ARCHITECTURE-LEVEL Ensemble SELECTION")
    print("=" * 80)
    
    # Create architecture meta-models
    arch_sel_predictions = {}
    arch_eval_predictions = {}
    
    for arch in architectures:
        # Selection predictions
        arch_preds = [v for k, v in sel_data['predictions'].items() if k.startswith(arch)]
        if arch_preds:
            arch_sel_predictions[arch] = np.mean(arch_preds, axis=0)
        
        # Evaluation predictions
        arch_preds = [v for k, v in eval_data['predictions'].items() if k.startswith(arch)]
        if arch_preds:
            arch_eval_predictions[arch] = np.mean(arch_preds, axis=0)
    
    # Run selection
    metric = 'rmse' if task != 'bin' else 'rmse'
    selector = EnsembleSelector(metric=metric, verbose=False)
    ensemble_weights = selector.greedy_selection(
        arch_sel_predictions,
        sel_data['targets'],
        n_iterations=50,
        n_bags=10,
    )
    
    print(f"\nSelected {len(ensemble_weights)} architectures")
    
    # Evaluate on held-out portion
    eval_pred = np.zeros_like(eval_data['targets'], dtype=float)
    for arch, weight in ensemble_weights.items():
        if arch in arch_eval_predictions:
            eval_pred += arch_eval_predictions[arch] * weight
    
    final_score = compute_metric(eval_data['targets'], eval_pred, task)
    
    metric_name = 'Log Loss' if task == 'bin' else 'RMSE'
    print(f"\nFinal {metric_name}: {final_score:.4f}")
    
    # Show weights
    print(f"\nArchitecture weights:")
    for arch, weight in sorted(ensemble_weights.items(), key=lambda x: x[1], reverse=True):
        print(f"\tweight: {weight:.4f} -> {arch}")
    
    return ensemble_weights, final_score


# =============================================================================
# RESULTS AND VISUALIZATION
# =============================================================================

def log_results(
    baselines: Dict[str, float],
    method_scores: Dict[str, float],
    task: str,
    output_dir: str = 'plots/',
    uncertainty_results: Optional[List[Dict]] = None,
) -> pd.DataFrame:
    """Log and visualize results."""
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("All methods evaluated on same UNSEEN evaluation set")
    print("=" * 80)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metric_name = 'Log Loss' if task == 'bin' else 'RMSE'
    
    results = {
        'Method': [
            'Baseline: Best Single Model',
            f'Baseline: Average All Models',
            'Baseline: Best Architecture',
            'Caruana Ensemble Selection',
            'Architecture-Level Ensemble Selection'
        ],
        f'Evaluation Set {metric_name}': [
            baselines['best_single'],
            baselines['average_all'],
            baselines['best_architecture'],
            method_scores.get('method2', np.nan),
            method_scores.get('method4', np.nan)
        ]
    }
    
    df = pd.DataFrame(results)
    
    best = baselines['best_single']
    df['vs Best Single (%)'] = ((df[f'Evaluation Set {metric_name}'] - best) / best * 100)
    
    print("\n" + df.to_string(index=False))
    
    # Save
    df.to_csv(output_dir / f'ensemble_comparison_results_{task}.csv', index=False)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    methods = df['Method'].values
    scores = df[f'Evaluation Set {metric_name}'].values
    improvement = df['vs Best Single (%)'].values
    
    colors = ['red', 'gray', 'gray', 'green', 'orange']
    
    # Plot 1: Absolute scores
    ax = axes[0]
    bars = ax.barh(range(len(methods)), scores, color=colors, alpha=0.7)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([m.replace('Baseline: ', '').replace('Method ', 'M') 
                        for m in methods], fontsize=10)
    ax.set_xlabel(f'{metric_name} on Evaluation Set (lower is better)', fontsize=12)
    ax.set_title(f'Fair Comparison - Task: {task.upper()}', fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    
    for i, (bar, val) in enumerate(zip(bars, scores)):
        if not np.isnan(val):
            ax.text(val, i, f' {val:.4f}', va='center', fontsize=10)
    
    # Plot 2: Relative improvement
    ax = axes[1]
    bars = ax.barh(range(len(methods)), improvement, color=colors, alpha=0.7)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([m.replace('Baseline: ', '').replace('Method ', 'M') 
                        for m in methods], fontsize=10)
    ax.set_xlabel('Change vs Best Single (%)', fontsize=12)
    ax.set_title('Performance Relative to Best', fontsize=13, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.invert_yaxis()
    
    for i, (bar, val) in enumerate(zip(bars, improvement)):
        if not np.isnan(val):
            color = 'green' if val < 0 else 'red'
            ax.text(val, i, f' {val:+.2f}%', va='center', fontsize=10, color=color)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'ensemble_comparison_{task}.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to {output_dir / f'ensemble_comparison_{task}.png'}")
    plt.close()
    
    # Uncertainty analysis visualization
    if uncertainty_results:
        plot_uncertainty_analysis(uncertainty_results, task, output_dir)
    
    return df


def plot_uncertainty_analysis(
    uncertainty_results: List[Dict],
    task: str,
    output_dir: Path,
) -> None:
    """Plot uncertainty quantification analysis."""
    df_unc = pd.DataFrame(uncertainty_results)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Number of models vs uncertainty
    ax = axes[0, 0]
    ax.bar(df_unc['method'], df_unc['n_models_selected'], color='steelblue', alpha=0.7)
    ax.set_ylabel('Number of Models Selected')
    ax.set_title('Ensemble Size by Method')
    ax.tick_params(axis='x', rotation=45)
    
    # Plot 2: Mean uncertainty
    ax = axes[0, 1]
    ax.bar(df_unc['method'], df_unc['mean_uncertainty'], color='coral', alpha=0.7)
    ax.set_ylabel('Mean Prediction Uncertainty (Std)')
    ax.set_title('Average Uncertainty by Method')
    ax.tick_params(axis='x', rotation=45)
    
    # Plot 3: Calibration metric
    ax = axes[1, 0]
    if task == 'bin':
        if 'ece' in df_unc.columns:
            ax.bar(df_unc['method'], df_unc['ece'], color='green', alpha=0.7)
            ax.set_ylabel('Expected Calibration Error (lower is better)')
            ax.set_title('Calibration Quality')
    else:
        if 'uncertainty_error_correlation' in df_unc.columns:
            ax.bar(df_unc['method'], df_unc['uncertainty_error_correlation'], color='green', alpha=0.7)
            ax.set_ylabel('Uncertainty-Error Correlation')
            ax.set_title('Uncertainty Quality (higher is better)')
    ax.tick_params(axis='x', rotation=45)
    
    # Plot 4: Coverage (for regression) or additional metric
    ax = axes[1, 1]
    if task != 'bin' and 'coverage_1sigma' in df_unc.columns:
        x = np.arange(len(df_unc))
        width = 0.25
        ax.bar(x - width, df_unc['coverage_1sigma'], width, label='1σ (exp: 68%)', alpha=0.7)
        ax.bar(x, df_unc['coverage_2sigma'], width, label='2σ (exp: 95%)', alpha=0.7)
        ax.bar(x + width, df_unc['coverage_3sigma'], width, label='3σ (exp: 99.7%)', alpha=0.7)
        ax.axhline(y=0.6827, color='blue', linestyle='--', alpha=0.5)
        ax.axhline(y=0.9545, color='orange', linestyle='--', alpha=0.5)
        ax.axhline(y=0.9973, color='green', linestyle='--', alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(df_unc['method'], rotation=45)
        ax.set_ylabel('Coverage')
        ax.set_title('Prediction Interval Coverage')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'Additional metrics\nnot available', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'ensemble_uncertainty_analysis_{task}.png', dpi=300, bbox_inches='tight')
    print(f"Uncertainty analysis saved to {output_dir / f'ensemble_uncertainty_analysis_{task}.png'}")
    plt.close()
    
    # Save uncertainty metrics
    df_unc.to_csv(output_dir / f'ensemble_uncertainty_metrics_{task}.csv', index=False)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 80)
    print("FAIR ENSEMBLE SELECTION COMPARISON")
    print("Supports regression (Dmax, DC50) and binary classification (bin)")
    print("=" * 80)
    
    parser = argparse.ArgumentParser(description="Run comparison of ensemble selection methods.")
    parser.add_argument('--prediction_dir', type=str, default='predictions/',
                        help='Directory containing prediction CSV files.')
    parser.add_argument('--task', type=str, default='Dmax',
                        help='Task identifier: Dmax, DC50, or bin')
    parser.add_argument('--output_dir', type=str, default='ensemble_results/',
                        help='Directory to save output plots and results.')
    parser.add_argument('--hillclimb_perc', type=float, default=0.2,
                        help='Percentage of test set to use for selection (default: 0.5).')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    args = parser.parse_args()
    
    task = args.task.lower()
    if task not in ['dmax', 'dc50', 'bin']:
        raise ValueError("Invalid task. Choose from 'Dmax', 'DC50', or 'bin'.")
    
    print(f"\nTask: {task.upper()}")
    print(f"Metric: {'Log Loss / AUC-ROC' if task == 'bin' else 'RMSE'}")
    
    # Load data
    val_preds, val_targets, test_preds, test_targets, architectures = \
        load_predictions_data(args.prediction_dir, args.task)
    
    if test_targets is None or len(test_preds) == 0:
        print(f"\nERROR: No prediction files found for task '{args.task}'")
        print("Check that files match pattern: preds-*-task={task}-*.csv")
        return None
    
    # Split test set
    sel_data, eval_data = split_test_set(
        test_predictions=test_preds,
        test_targets=test_targets,
        test_perc=args.hillclimb_perc,
        random_seed=args.random_seed,
    )
    
    # Compute baselines
    baselines = compute_baselines(
        eval_data['predictions'],
        eval_data['targets'],
        architectures,
        task,
    )
    
    # Run methods and collect uncertainty
    method_scores = {}
    uncertainty_results = []
    
    # Caruana selection
    weights2 = None
    try:
        weights2, score2 = run_caruana_method(sel_data, eval_data, task)
        method_scores['method2'] = score2
        
        # Save ensemble weights
        save_ensemble_weights(
            weights=weights2,
            output_path=args.output_dir,
            task=task,
            method_name="caruana_ensemble",
            metric_value=score2,
            metadata={
                "selection_set_size": len(sel_data['targets']),
                "eval_set_size": len(eval_data['targets']),
                "hillclimb_perc": args.hillclimb_perc,
                "random_seed": args.random_seed,
            }
        )
        
        # Uncertainty analysis
        unc_metrics, _ = analyze_uncertainty_quality(
            eval_data['predictions'], eval_data['targets'],
            weights2, 'Caruana Ensemble Selection', task
        )
        uncertainty_results.append(unc_metrics)
    except Exception as e:
        print(f"Caruana selection failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Architecture-level ensemble selection
    weights4 = None
    arch_weights_expanded = None
    try:
        weights4, score4 = run_architecture_caruana_method(sel_data, eval_data, architectures, task)
        method_scores['method4'] = score4
        
        # For Architecture-level ensemble selection, weights are at architecture level - need to expand
        arch_weights_expanded = {}
        for arch, weight in weights4.items():
            arch_models = [k for k in eval_data['predictions'].keys() if k.startswith(arch)]
            for model in arch_models:
                arch_weights_expanded[model] = weight / len(arch_models)
        
        # Save architecture-level weights
        save_ensemble_weights(
            weights=weights4,
            output_path=args.output_dir,
            task=task,
            method_name="architecture_level",
            metric_value=score4,
            metadata={
                "selection_set_size": len(sel_data['targets']),
                "eval_set_size": len(eval_data['targets']),
                "architectures": list(architectures),
                "weight_type": "architecture_level",
            }
        )
        
        # Also save expanded model-level weights
        save_ensemble_weights(
            weights=arch_weights_expanded,
            output_path=args.output_dir,
            task=task,
            method_name="architecture_level_expanded",
            metric_value=score4,
            metadata={
                "selection_set_size": len(sel_data['targets']),
                "eval_set_size": len(eval_data['targets']),
                "weight_type": "model_level_from_architecture",
            }
        )
        
        unc_metrics, _ = analyze_uncertainty_quality(
            eval_data['predictions'], eval_data['targets'],
            arch_weights_expanded, 'Architecture-Level Ensemble', task
        )
        uncertainty_results.append(unc_metrics)
    except Exception as e:
        print(f"Architecture-level ensemble selection failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Baseline uncertainty (average all)
    try:
        uniform_weights = {k: 1.0/len(eval_data['predictions']) 
                          for k in eval_data['predictions'].keys()}
        unc_metrics, _ = analyze_uncertainty_quality(
            eval_data['predictions'], eval_data['targets'],
            uniform_weights, 'Baseline: Average All', task
        )
        uncertainty_results.append(unc_metrics)
    except Exception as e:
        print(f"Baseline uncertainty analysis failed: {e}")
    
    # Baseline uncertainty (best architecture)
    try:
        best_arch = baselines['best_architecture_name']
        arch_models = [k for k in eval_data['predictions'].keys() if k.startswith(best_arch)]
        arch_weights = {k: 1.0/len(arch_models) for k in arch_models}
        
        unc_metrics, _ = analyze_uncertainty_quality(
            eval_data['predictions'], eval_data['targets'],
            arch_weights, 'Baseline: Best Architecture', task
        )
        uncertainty_results.append(unc_metrics)
    except Exception as e:
        print(f"Best architecture uncertainty analysis failed: {e}")
    
    # Create comparison
    results_df = log_results(
        baselines, method_scores, task, 
        Path(args.output_dir), uncertainty_results
    )
    
    # Print additional metrics for binary classification
    if task == 'bin':
        print("\n" + "=" * 80)
        print("ADDITIONAL BINARY CLASSIFICATION METRICS")
        print("=" * 80)
        
        # Compute additional metrics for best method
        best_method = min(method_scores, key=method_scores.get)
        if best_method == 'method2':
            weights = weights2
        else:
            weights = arch_weights_expanded
        
        ensemble_pred = np.zeros_like(eval_data['targets'], dtype=float)
        for model_key, weight in weights.items():
            if model_key in eval_data['predictions']:
                ensemble_pred += eval_data['predictions'][model_key] * weight
        
        all_metrics = compute_all_metrics(eval_data['targets'], ensemble_pred, task)
        print(f"\nBest ensemble method ({best_method}):")
        for name, value in all_metrics.items():
            print(f"  {name}: {value:.4f}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nEvaluation approach:")
    print("  • Baselines: evaluated on evaluation set (unseen)")
    print("  • Caruana trained on selection set, evaluated on eval set (unseen)")
    print("  • Ensemble trained on selection set, evaluated on eval set (unseen)")
    print("\nUncertainty quantification:")
    print("  • Ensemble disagreement (std across models)")
    print("  • Calibration metrics (ECE for binary, coverage for regression)")
    print("\nAll methods evaluated on data they haven't seen during selection!")
    
    return results_df


if __name__ == "__main__":
    results = main()
