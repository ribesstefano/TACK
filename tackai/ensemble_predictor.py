"""
Ensemble Predictor for TACK Models
Handles loading and prediction from multiple model types (XGBoost, Lightning)
with weighted averaging and uncertainty quantification.
"""
import os
import re
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Literal
from dataclasses import dataclass, field

import torch
import numpy as np
import pandas as pd
import xgboost as xgb

from tackai.data.datamodule import load_datamodule
from tackai.models.tackai_model import TACKModel

warnings.filterwarnings('ignore')


@dataclass
class EnsemblePrediction:
    """Container for ensemble prediction results.

    All predictions are stored in the **original (denormalized) scale**.
    Two kinds of 95% confidence intervals are provided:

    * **Percentile CI** – non-parametric, based on the 2.5th and 97.5th
      percentiles of the individual model predictions.  Interpretation:
      "most models predict within this range."
    * **SEM CI** – parametric, based on the standard error of the mean
      (assumes approximate normality).  Interpretation: "the true average
      prediction is likely in this range."  Shrinks towards zero as the
      number of models increases.
    """
    # Core predictions (denormalized / original scale)
    weighted_mean: np.ndarray
    uncertainty_std: np.ndarray

    # Individual model info (all denormalized)
    individual_predictions: Dict[str, np.ndarray]
    weights: Dict[str, float]
    model_names: List[str]

    # Task and label info
    task: str = 'dmax'
    label_name: Optional[str] = None

    # Additional uncertainty metrics (computed from denormalized predictions)
    prediction_variance: np.ndarray = field(default=None)
    prediction_range: np.ndarray = field(default=None)
    prediction_iqr: np.ndarray = field(default=None)

    # For binary classification
    predictive_entropy: Optional[np.ndarray] = None

    # Percentile-based 95% CI (non-parametric)
    ci_percentile_lower_95: Optional[np.ndarray] = None
    ci_percentile_upper_95: Optional[np.ndarray] = None

    # Standard Error Method (SEM)-based 95% CI (parametric, normal assumption)
    ci_sem_lower_95: Optional[np.ndarray] = None
    ci_sem_upper_95: Optional[np.ndarray] = None

    def __post_init__(self):
        """Compute additional uncertainty metrics from denormalized predictions."""
        if not self.individual_predictions:
            return

        preds = np.array(list(self.individual_predictions.values()))
        n_models = len(preds)

        self.prediction_variance = np.var(preds, axis=0)
        self.prediction_range = np.max(preds, axis=0) - np.min(preds, axis=0)

        q75 = np.percentile(preds, 75, axis=0)
        q25 = np.percentile(preds, 25, axis=0)
        self.prediction_iqr = q75 - q25

        # --- Percentile-based 95% CI (non-parametric) ---
        self.ci_percentile_lower_95 = np.percentile(preds, 2.5, axis=0)
        self.ci_percentile_upper_95 = np.percentile(preds, 97.5, axis=0)

        # --- SEM-based 95% CI (parametric) ---
        if n_models > 1:
            sem = np.std(preds, ddof=1, axis=0) / np.sqrt(n_models)
        else:
            sem = np.zeros_like(self.weighted_mean)
        self.ci_sem_lower_95 = self.weighted_mean - 1.96 * sem
        self.ci_sem_upper_95 = self.weighted_mean + 1.96 * sem

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        def to_list(arr):
            if arr is None:
                return None
            if isinstance(arr, np.ndarray):
                return arr.tolist()
            return arr

        return {
            'weighted_mean': to_list(self.weighted_mean),
            'uncertainty_std': to_list(self.uncertainty_std),
            'ci_percentile_lower_95': to_list(self.ci_percentile_lower_95),
            'ci_percentile_upper_95': to_list(self.ci_percentile_upper_95),
            'ci_sem_lower_95': to_list(self.ci_sem_lower_95),
            'ci_sem_upper_95': to_list(self.ci_sem_upper_95),
            'prediction_variance': to_list(self.prediction_variance),
            'prediction_range': to_list(self.prediction_range),
            'prediction_iqr': to_list(self.prediction_iqr),
            'predictive_entropy': to_list(self.predictive_entropy),
            'model_names': self.model_names,
            'weights': self.weights,
            'n_models': len(self.model_names),
            'task': self.task,
            'label_name': self.label_name,
        }

    def summary(self) -> str:
        """Return a formatted summary string."""
        lines = [
            f"Task: {self.task.upper()}",
            f"Label: {self.label_name or 'N/A'}",
            f"Number of models: {len(self.model_names)}",
            f"Prediction: {self.weighted_mean[0]:.4f} ± {self.uncertainty_std[0]:.4f}",
            f"95% CI (percentile): [{self.ci_percentile_lower_95[0]:.4f}, {self.ci_percentile_upper_95[0]:.4f}]",
            f"95% CI (SEM):        [{self.ci_sem_lower_95[0]:.4f}, {self.ci_sem_upper_95[0]:.4f}]",
        ]
        return "\n".join(lines)


@dataclass
class SampleInput:
    """Container for a single sample input with all possible features."""
    smiles: Optional[str] = None
    poi_name: Optional[str] = None
    poi_sequence: Optional[str] = None
    ligase_name: Optional[str] = None
    ligase_sequence: Optional[str] = None
    cell_line: Optional[str] = None
    assay_type: Optional[str] = None
    treatment_time: Optional[float] = None
    degrader_type: Optional[str] = None
    
    # For precomputed features
    precomputed_features: Optional[np.ndarray] = None
    
    def to_datamodule_dict(
        self,
        smiles_col: str = "SMILES",
        poi_col: str = "POI_Name",
        poi_sequence_col: str = "POI_Sequence",
        ligase_col: str = "Ligase_Name",
        ligase_sequence_col: str = "Ligase_Sequence",
        cell_line_col: str = "Cell_Line_ID",
        assay_type_col: str = "Assay",
        treatment_time_col: str = "Assay_Time",
        degrader_type_col: str = "Degrader_Type",
    ) -> Dict[str, Any]:
        """Convert to dictionary format expected by datamodule."""
        return {
            smiles_col: self.smiles,
            poi_col: self.poi_name,
            poi_sequence_col: self.poi_sequence,
            ligase_col: self.ligase_name,
            ligase_sequence_col: self.ligase_sequence,
            cell_line_col: self.cell_line,
            assay_type_col: self.assay_type,
            treatment_time_col: self.treatment_time,
            degrader_type_col: self.degrader_type,
        }


# Default values for optional inputs based on common training data.
# NOTE: SMILES, POI (name/sequence), and E3 ligase (name/sequence) are
# considered *required* — no defaults are supplied for them.
DEFAULT_VALUES = {
    'Cell_Line_ID': 'Unknown',
    'Assay': 'Unknown',
    'Assay_Time': 24.0,  # Default 24 hours
    'Degrader_Type': 'PROTAC',
}

# Task labels recognised in model filenames
KNOWN_TASKS = {'dmax', 'dc50', 'bin', 'dmax_bin', 'dc50_bin', 'multitask'}


class EnsemblePredictor:
    """
    Ensemble predictor that loads and combines predictions from multiple models.
    
    Supports:
    - XGBoost models (.json, .ubj, .pkl)
    - PyTorch Lightning models (.ckpt)
    - Weighted averaging with uncertainty quantification
    - Proper denormalization using datamodule transformers
    
    Example:
        >>> predictor = EnsemblePredictor.from_directory(
        ...     model_dir='models/',
        ...     weights={'model1': 0.5, 'model2': 0.5}
        ... )
        >>> sample = SampleInput(
        ...     smiles='CCO',
        ...     poi_name='BRD4',
        ...     poi_sequence='MKTAYIA...',
        ...     ligase_name='CRBN',
        ...     cell_line='HEK293',
        ... )
        >>> result = predictor.predict(sample)
        >>> print(f"Prediction: {result.weighted_mean} ± {result.uncertainty_std}")
    """
    
    def __init__(
        self,
        models: Dict[str, Any],
        datamodules: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None,
        # task: str = 'dmax',
        # label_name: Optional[str] = None,
        device: str = 'cpu',
    ) -> None:
        """
        Initialize the ensemble predictor.
        
        Args:
            models: Dictionary mapping model names to model objects
            datamodules: Dictionary mapping model names to their datamodules
            weights: Optional weights for each model (uniform if None)
            task: Task type ('dmax', 'dc50', 'bin')
            label_name: Name of the label column for denormalization
            device: Device to use for inference ('cpu' or 'cuda')
        """
        self.models = models
        self.datamodules = datamodules
        self.device = device
        
        # Per-model task inferred from filenames / datamodule label names
        self.model_tasks: Dict[str, str] = {}
        for name in models:
            self.model_tasks[name] = self._infer_model_task(name, datamodules.get(name))
        
        # self.task = task.lower()
        # # Infer label name if not provided
        # if label_name is None:
        #     label_name = self._infer_label_name()
        # self.label_name = label_name
        
        # Set up weights, if not provided, use uniform weights per task,
        # creating a dict mapping model name → weight
        self.weights = {}
        if weights is None:
            task2name = {}
            for name, task in self.model_tasks.items():
                if task not in task2name:
                    task2name[task] = []
                task2name[task].append(name)
            # Convert to model_name → weight
            for task, names in task2name.items():
                n = len(names)
                for name in names:
                    self.weights[name] = 1.0 / n
        
        # Validate weights
        for name in self.weights.keys():
            if name not in self.models:
                raise ValueError(f"Weight specified for unknown model: {name}")
        
        # Model type tracking
        self.model_types = {}
        for name, model in self.models.items():
            self.model_types[name] = self._get_model_type(model)
        
        # Validate that every model has a corresponding datamodule
        missing_dm = [name for name in self.models if self.datamodules.get(name) is None]
        if missing_dm:
            raise ValueError(
                f"The following models have no associated datamodule: {missing_dm}. "
                "Each model must have a corresponding datamodule for featurization "
                "and denormalization."
            )
        
        # Pre-compute categorical choices across all datamodules
        self._categorical_choices: Dict[str, List[str]] = self._collect_categorical_choices()
    
    def _infer_label_name(self) -> Optional[str]:
        """Infer label name from task or datamodule."""
        task_to_label = {
            'dmax': 'Dmax (%)',
            'dc50': 'DC50 (nM)',
            'bin': 'Binary Activity',
        }
        
        # Try to get from datamodule first
        for dm in self.datamodules.values():
            if dm is not None and hasattr(dm, 'labels') and dm.labels:
                return dm.labels[0]
        
        return None
    
    # ------------------------------------------------------------------
    # Task inference
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_model_task(model_name: str, datamodule: Optional[Any] = None) -> str:
        """Infer the prediction task for a single model.

        The task is determined by:
        1. The label names stored in the datamodule (most reliable).
        2. Parsing the model filename (``model=<arch>_<task>_protac-...``).
        3. Falling back to ``'dmax'`` if nothing else works.
        """
        # 1. From datamodule label names
        if datamodule is not None and hasattr(datamodule, 'labels') and datamodule.labels:
            label = datamodule.labels[0].lower()
            if 'binary' in label or 'activity' in label:
                return 'bin'
            elif 'pdc50' in label or 'dc50' in label:
                return 'dc50'
            elif 'dmax' in label:
                return 'dmax'

        # 2. From model filename convention:
        #    model=<arch>_<task>_protac-data=...
        #    e.g. model=mlp_dmax_protac-data=...
        m = re.search(r'model=\w+?_(\w+?)_protac', model_name, re.IGNORECASE)
        if m:
            task_str = m.group(1).lower()
            if task_str in KNOWN_TASKS:
                return task_str

        return 'dmax'
    
    # ------------------------------------------------------------------
    # Categorical choices
    # ------------------------------------------------------------------

    def _collect_categorical_choices(self) -> Dict[str, List[str]]:
        """Collect unique category values from all datamodule ordinal encoders.

        Returns a dict mapping column names (e.g. ``'Ligase_Name'``,
        ``'Cell_Line_ID'``, ``'Assay'``) to sorted lists of known
        categories seen during training (across **all** folds / configs).
        """
        merged: Dict[str, set] = {}
        for dm in self.datamodules.values():
            if dm is None:
                continue
            pipeline = getattr(dm, 'category_pipeline', None)
            if pipeline is None:
                continue
            for _name, transformer, cols in pipeline.transformers_:
                ordinal = transformer.named_steps.get('ordinal') if hasattr(transformer, 'named_steps') else None
                if ordinal is None or not hasattr(ordinal, 'categories_'):
                    continue
                for col, cats in zip(cols, ordinal.categories_):
                    if col not in merged:
                        merged[col] = set()
                    merged[col].update(c for c in cats if c is not None)
        # Convert to sorted lists
        return {col: sorted(vals) for col, vals in merged.items()}

    def get_categorical_choices(self) -> Dict[str, List[str]]:
        """Return known categorical values for each ordinal-encoded column.

        Each list is sorted and **does not** include ``'Unknown'`` – the
        caller should add it when building a UI.
        """
        return dict(self._categorical_choices)

    @property
    def available_tasks(self) -> List[str]:
        """Return the distinct set of tasks present in the loaded models."""
        return sorted(set(self.model_tasks.values()))
    
    def get_required_inputs(self, model_name: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Get the required inputs for each model/datamodule.
        
        Args:
            model_name: Specific model name, or None for all models
            
        Returns:
            Dictionary mapping model names to lists of required column names
        """
        required = {}
        
        datamodules = self.datamodules
        if model_name is not None:
            datamodules = {model_name: self.datamodules.get(model_name)}
        
        for name, dm in datamodules.items():
            if dm is None:
                required[name] = ['SMILES']  # Minimum requirement
                continue
            
            req_cols = [dm.smiles_col]  # Always need SMILES
            
            # Check which features this datamodule needs
            if getattr(dm, 'use_poi_sequence_embedding', False):
                req_cols.append(dm.poi_sequence_col)
            if getattr(dm, 'use_poi_name_embedding', False):
                req_cols.append(dm.poi_col)
            if getattr(dm, 'use_poi_precomputed_embedding', False):
                # Depends on embedding ID type
                if getattr(dm, 'poi_embeddings_id_type', 'sequence') == 'sequence':
                    req_cols.append(dm.poi_sequence_col)
                else:
                    req_cols.append(dm.poi_col)
            if getattr(dm, 'use_ligase_name_embedding', False):
                req_cols.append(dm.ligase_col)
            if getattr(dm, 'use_ligase_precomputed_embedding', False):
                req_cols.append(dm.ligase_sequence_col)
            if getattr(dm, 'use_cell_description_embedding', False) or getattr(dm, 'use_cell_name_embedding', False):
                req_cols.append(dm.cell_line_col)
            if getattr(dm, 'use_treatment_time', False):
                req_cols.append(dm.treatment_time_col)
            if getattr(dm, 'use_assay_type_encoding', False):
                req_cols.append(dm.assay_type_col)
            
            required[name] = list(set(req_cols))  # Remove duplicates
        
        return required
    
    def validate_and_fill_defaults(
        self,
        sample_dict: Dict[str, Any],
        datamodule: Optional[Any] = None,
        verbose: bool = True,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Validate inputs and fill in defaults for *optional* features.

        **Required** inputs (SMILES, POI name/sequence, E3 ligase
        name/sequence) will **not** be defaulted – a warning is emitted
        instead and the caller can decide whether to skip the model.

        The method also normalises user-supplied keys so that e.g. both
        ``'Smiles'`` and ``'SMILES'`` are accepted.

        Args:
            sample_dict: Dictionary of input values.
            datamodule: Datamodule to check requirements against.
            verbose: Whether to print warnings about missing inputs.

        Returns:
            A tuple ``(filled_dict, missing_required)`` where
            ``missing_required`` lists column names that are required by
            the datamodule but were not provided by the user.
        """
        dm = datamodule
        if dm is None:
            raise ValueError(
                "A datamodule must be provided for input validation. "
                "Cannot fall back to a reference datamodule."
            )

        # --- Normalise keys --------------------------------------------------
        # Build a map from lower-cased key → datamodule column name so that
        # user dicts with slightly different casing still match.
        dm_col_names = [
            dm.smiles_col, dm.poi_col, dm.poi_sequence_col,
            dm.ligase_col, dm.ligase_sequence_col, dm.cell_line_col,
            dm.treatment_time_col, dm.assay_type_col,
        ]
        lower_to_dm = {c.lower(): c for c in dm_col_names if c}

        filled: Dict[str, Any] = {}
        for k, v in sample_dict.items():
            canonical = lower_to_dm.get(k.lower(), k)
            filled[canonical] = v

        missing_required: List[str] = []
        default_warnings: List[str] = []

        # Columns that are *never* auto-filled – the user must provide them
        required_cols = {dm.smiles_col, dm.poi_col, dm.poi_sequence_col,
                         dm.ligase_col, dm.ligase_sequence_col}

        # (dm_col, is_needed_check)
        col_checks = [
            (dm.smiles_col, lambda: True),
            (dm.poi_col, lambda: (
                getattr(dm, 'use_poi_name_embedding', False)
                or getattr(dm, 'poi_embeddings_id_type', '') != 'sequence'
            )),
            (dm.poi_sequence_col, lambda: (
                getattr(dm, 'use_poi_sequence_embedding', False)
                or getattr(dm, 'use_poi_precomputed_embedding', False)
            )),
            (dm.ligase_col, lambda: getattr(dm, 'use_ligase_name_embedding', False)),
            (dm.ligase_sequence_col, lambda: getattr(dm, 'use_ligase_precomputed_embedding', False)),
            (dm.cell_line_col, lambda: (
                getattr(dm, 'use_cell_description_embedding', False)
                or getattr(dm, 'use_cell_name_embedding', False)
            )),
            (dm.treatment_time_col, lambda: getattr(dm, 'use_treatment_time', False)),
            (dm.assay_type_col, lambda: getattr(dm, 'use_assay_type_encoding', False)),
        ]

        for dm_col, is_needed_fn in col_checks:
            if not is_needed_fn():
                continue

            current_val = filled.get(dm_col)
            is_missing = current_val is None or (isinstance(current_val, str) and current_val.strip() == '')

            if not is_missing:
                continue

            if dm_col in required_cols:
                missing_required.append(dm_col)
            else:
                # Fill optional fields with defaults
                default_val = DEFAULT_VALUES.get(dm_col)
                if default_val is not None:
                    filled[dm_col] = default_val
                    short = str(default_val) if len(str(default_val)) < 30 else str(default_val)[:30] + '...'
                    default_warnings.append(f"  - {dm_col}: using default '{short}'")

        if verbose:
            if missing_required:
                warnings.warn(
                    f"Required input(s) missing: {', '.join(missing_required)}. "
                    "Models that need these features will be skipped.",
                    UserWarning,
                    stacklevel=2,
                )
            if default_warnings:
                print("Note: Some inputs were missing and filled with defaults:")
                for w in default_warnings[:5]:
                    print(w)
                if len(default_warnings) > 5:
                    print(f"  ... and {len(default_warnings) - 5} more")

        return filled, missing_required

    def get_xgb_feature_names(self, datamodule: Any) -> List[str]:
        """
        Get feature names for XGBoost model from a sample featurization.
        
        Args:
            datamodule: The datamodule to use for featurization
            
        Returns:
            List of feature names in order they appear in concatenated features
        """
        return datamodule.get_xgboost_feature_names()
    
    @staticmethod
    def _get_model_type(model: Any) -> str:
        """Determine the type of model."""
        model_class = type(model).__name__
        
        if 'XGB' in model_class or 'Booster' in model_class:
            return 'xgboost'
        elif hasattr(model, 'forward') and hasattr(model, 'eval'):
            return 'lightning'
        else:
            return 'unknown'
    
    @classmethod
    def from_directory(
        cls,
        model_dir: Union[str, Path],
        datamodule_dir: Optional[Union[str, Path]] = None,
        weights: Optional[Dict[str, float]] = None,
        # task: str = 'dmax',
        # label_name: Optional[str] = None,
        device: str = 'cpu',
        pattern: Optional[str] = None,
    ) -> 'EnsemblePredictor':
        """
        Load models from a directory.
        
        Args:
            model_dir: Directory containing model files
            datamodule_dir: Directory containing datamodule state dicts (optional)
            weights: Optional weights for each model
            task: Task type
            label_name: Label column name for denormalization
            device: Device for inference
            pattern: Optional regex pattern to filter model files
            
        Returns:
            Initialized EnsemblePredictor
        """
        model_dir = Path(model_dir)
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        models = {}
        datamodules = {}
        
        # Find model files
        model_files = []
        for ext in ['*.ckpt', '*.json', '*.ubj', '*.pkl']:
            model_files.extend(model_dir.glob(f'**/{ext}'))
        
        # Filter by pattern if provided
        if pattern:
            regex = re.compile(pattern)
            model_files = [f for f in model_files if regex.search(str(f))]
        
        print(f"Found {len(model_files)} model files in {model_dir}")
        
        for i, model_file in enumerate(model_files):
            model_name = model_file.stem
            
            try:
                model, datamodule = cls._load_model_and_datamodule(
                    model_file, device, datamodule_dir
                )
                models[model_name] = model
                datamodules[model_name] = datamodule
                print(f"  Loaded ({i+1}/{len(model_files)}): {model_name} ({cls._get_model_type(model)})")
            except Exception as e:
                print(f"  Failed to load {model_name}: {e}")
                continue
        
        if not models:
            raise ValueError(f"No models could be loaded from {model_dir}")
        
        # return cls(models, datamodules, weights, task, label_name, device)
        return cls(models, datamodules, weights, device)
    
    @classmethod
    def from_weights_file(
        cls,
        weights_file: Union[str, Path],
        model_dir: Union[str, Path],
        datamodule_dir: Optional[Union[str, Path]] = None,
        device: str = 'cpu',
    ) -> 'EnsemblePredictor':
        """
        Load an ensemble predictor from a weights JSON file.
        
        Only loads the models specified in the weights file.
        
        Args:
            weights_file: Path to the JSON weights file
            model_dir: Directory containing model checkpoints
            datamodule_dir: Directory containing datamodule state dicts
            device: Device for inference
            
        Returns:
            Initialized EnsemblePredictor with only the specified models
        """
        import json
        
        weights_file = Path(weights_file)
        model_dir = Path(model_dir)
        
        if not weights_file.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_file}")
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # Load weights file
        with open(weights_file, 'r') as f:
            weights_doc = json.load(f)
        
        weights = weights_doc.get("weights", {})
        task = weights_doc.get("task", "dmax")
        method = weights_doc.get("method", "unknown")
        
        print(f"Loading ensemble from weights file: {weights_file.name}")
        print(f"  Task: {task}")
        print(f"  Method: {method}")
        print(f"  Models to load: {len(weights)}")
        
        if not weights:
            raise ValueError("No weights found in weights file")
        
        # Find all model files in directory
        model_files = []
        for ext in ['*.ckpt', '*.json', '*.ubj', '*.pkl']:
            model_files.extend(model_dir.glob(f'**/{ext}'))
        
        # Build a lookup of model_name -> model_file
        model_lookup = {f.stem: f for f in model_files}
        
        models = {}
        datamodules = {}
        loaded_weights = {}
        
        for model_name, weight in weights.items():
            if model_name not in model_lookup:
                print(f"  ⚠️ Model not found: {model_name}")
                continue
            
            model_file = model_lookup[model_name]
            
            try:
                model, datamodule = cls._load_model_and_datamodule(
                    model_file, task, device, datamodule_dir or model_dir
                )
                models[model_name] = model
                datamodules[model_name] = datamodule
                loaded_weights[model_name] = weight
                print(f"  ✅ Loaded: {model_name[:60]}... (weight={weight:.4f})")
            except Exception as e:
                print(f"  ❌ Failed to load {model_name}: {e}")
                continue
        
        if not models:
            raise ValueError(f"No models could be loaded from weights file")
        
        print(f"\nSuccessfully loaded {len(models)} / {len(weights)} models")
        
        return cls(
            models=models,
            datamodules=datamodules,
            weights=loaded_weights,
            task=task,
            device=device,
        )
    
    @staticmethod
    def _load_model_and_datamodule(
        model_path: Path,
        device: str,
        datamodule_dir: Optional[Path] = None,
    ) -> Tuple[Any, Any]:
        """Load a model and its corresponding datamodule."""
        suffix = model_path.suffix.lower()
        
        if suffix == '.ckpt':
            return EnsemblePredictor._load_lightning_model(
                model_path, device, datamodule_dir
            )
        elif suffix in ['.json', '.ubj', '.pkl']:
            return EnsemblePredictor._load_xgboost_model(
                model_path, datamodule_dir
            )
        else:
            raise ValueError(f"Unsupported model format: {suffix}")
    
    @staticmethod
    def _load_lightning_model(
        model_path: Path,
        device: str,
        datamodule_dir: Optional[Path] = None,
    ) -> Tuple[Any, Any]:
        """Load a PyTorch Lightning model."""
        datamodule = None
        
        # Extract data configuration from model path name
        # Format: model=mlp_dmax_protac-data=XXX-group=YYY-fold=ZZZ.ckpt
        model_name = model_path.stem
        data_config = None
        for part in model_name.split('-'):
            if part.startswith('data='):
                data_config = part.replace('data=', '')
                break
        
        # Extract group and fold from model name
        group = None
        fold = None
        for part in model_name.split('-'):
            if part.startswith('group='):
                group = part.replace('group=', '')
            elif part.startswith('fold='):
                fold = part.replace('fold=', '')
        
        # Try to load datamodule from various locations
        dm_candidates = []
        
        # First, try the naming convention used in ensemble directory
        # Format: datamodule-data=XXX-group=YYY-fold=ZZZ_hparams.yaml / _state.pt
        if data_config and datamodule_dir:
            dm_dir = Path(datamodule_dir)
            dm_base = f"datamodule-data={data_config}-group={group}-fold={fold}"
            dm_candidates.append((dm_dir / f"{dm_base}_hparams.yaml", dm_dir / f"{dm_base}_state.pt"))
        
        # Also try in model's parent directory
        if data_config:
            dm_dir = model_path.parent
            dm_base = f"datamodule-data={data_config}-group={group}-fold={fold}"
            dm_candidates.append((dm_dir / f"{dm_base}_hparams.yaml", dm_dir / f"{dm_base}_state.pt"))
        
        # Try legacy naming conventions as fallback
        if datamodule_dir:
            dm_dir = Path(datamodule_dir)
            dm_candidates.append((dm_dir / f"{model_path.stem}_datamodule.pt", None))
            dm_candidates.append((dm_dir / "datamodule_state.pt", None))
        
        dm_candidates.extend([
            (model_path.parent / f"{model_path.stem}_datamodule.pt", None),
            (model_path.parent / "datamodule_state.pt", None),
            (model_path.parent / "datamodule_hparams.yaml", model_path.parent / "datamodule_state.pt"),
        ])
        
        for candidate in dm_candidates:
            # Handle tuple of (hparams_path, state_path) or (state_path, None)
            if isinstance(candidate, tuple):
                hparams_path, state_path = candidate
            else:
                hparams_path = candidate
                state_path = None
            
            if not hparams_path.exists():
                continue
                
            try:
                if hparams_path.suffix == '.yaml' and state_path and state_path.exists():
                    # Load from YAML + state dict using tackai's load_datamodule
                    datamodule = load_datamodule(hparams_path, state_path)
                    break
                elif hparams_path.suffix == '.pt':
                    # Load state dict directly
                    from tackai.data.datamodule import DegradationComplexDataModule
                    state_dict = torch.load(hparams_path, map_location='cpu')
                    if 'hparams' in state_dict:
                        datamodule = DegradationComplexDataModule(**state_dict['hparams'])
                        datamodule.load_state_dict(state_dict)
                        break
            except Exception as e:
                print(f"    Warning: Failed to load datamodule from {hparams_path}: {e}")
        
        try:
            # Load Lightning model using TACKModel.load_from_checkpoint
            model = TACKModel.load_from_checkpoint(
                str(model_path), 
                map_location=device,
            )
            model.to(device)
            model.eval()
        except Exception as e:
            # Fallback: load directly with torch
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            if 'state_dict' in checkpoint:
                raise NotImplementedError(
                    f"Cannot load Lightning model from checkpoint. Error: {e}"
                )
            model = checkpoint
        
        return model, datamodule
    
    @staticmethod
    def _load_xgboost_model(
        model_path: Path,
        datamodule_dir: Optional[Path] = None,
    ) -> Tuple[Any, Any]:
        """Load an XGBoost model."""        
        suffix = model_path.suffix.lower()
        
        if suffix in ['.json', '.ubj']:
            model = xgb.Booster()
            model.load_model(str(model_path))
            model.n_jobs = 4
        elif suffix == '.pkl':
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        else:
            raise ValueError(f"Unsupported XGBoost format: {suffix}")
        
        # Extract data configuration from model path name
        # Format: model=xgboost_dmax_protac-data=XXX-group=YYY-fold=ZZZ.json
        model_name = model_path.stem
        data_config = None
        for part in model_name.split('-'):
            if part.startswith('data='):
                data_config = part.replace('data=', '')
                break
        
        # Extract group and fold from model name
        group = None
        fold = None
        for part in model_name.split('-'):
            if part.startswith('group='):
                group = part.replace('group=', '')
            elif part.startswith('fold='):
                fold = part.replace('fold=', '')
        
        # Try to load datamodule for XGBoost
        datamodule = None
        dm_candidates = []
        
        # First, try the naming convention used in ensemble directory
        if data_config and datamodule_dir:
            dm_dir = Path(datamodule_dir)
            dm_base = f"datamodule-data={data_config}-group={group}-fold={fold}"
            dm_candidates.append((dm_dir / f"{dm_base}_hparams.yaml", dm_dir / f"{dm_base}_state.pt"))
        
        # Also try in model's parent directory
        if data_config:
            dm_dir = model_path.parent
            dm_base = f"datamodule-data={data_config}-group={group}-fold={fold}"
            dm_candidates.append((dm_dir / f"{dm_base}_hparams.yaml", dm_dir / f"{dm_base}_state.pt"))
        
        # Legacy fallback
        if datamodule_dir:
            dm_candidates.append((Path(datamodule_dir) / f"{model_path.stem}_datamodule.pt", None))
        dm_candidates.append((model_path.parent / f"{model_path.stem}_datamodule.pt", None))
        dm_candidates.append((model_path.parent / "datamodule_state.pt", None))
        
        for candidate in dm_candidates:
            if isinstance(candidate, tuple):
                hparams_path, state_path = candidate
            else:
                hparams_path = candidate
                state_path = None
            
            if not hparams_path.exists():
                continue
                
            try:
                if hparams_path.suffix == '.yaml' and state_path and state_path.exists():
                    from tackai.data.datamodule import load_datamodule
                    datamodule = load_datamodule(hparams_path, state_path)
                    break
                elif hparams_path.suffix == '.pt':
                    from tackai.data.datamodule import DegradationComplexDataModule
                    state_dict = torch.load(hparams_path, map_location='cpu')
                    if 'hparams' in state_dict:
                        datamodule = DegradationComplexDataModule(**state_dict['hparams'])
                        datamodule.load_state_dict(state_dict)
                        break
            except Exception as e:
                print(f"    Warning: Failed to load datamodule from {hparams_path}: {e}")
        
        return model, datamodule
    
    def featurize_input(
        self,
        sample: Union[SampleInput, Dict[str, Any]],
        model_name: Optional[str] = None,
        return_format: Literal['dict', 'xgb', 'pt'] = 'dict',
    ) -> Dict[str, Any]:
        """
        Featurize input for prediction.
        
        Args:
            sample: SampleInput object or dictionary with input data
            model_name: Specific model to use for featurization (if None, uses all)
            return_format: Format for returned features ('dict', 'xgb', 'pt')
            
        Returns:
            Dictionary mapping model names to featurized inputs
        """
        # Convert SampleInput to dict if needed
        if isinstance(sample, SampleInput):
            # Use the first available datamodule's column names for conversion
            ref_dm = None
            if model_name and model_name in self.datamodules:
                ref_dm = self.datamodules[model_name]
            else:
                ref_dm = next(iter(self.datamodules.values()))
            
            sample_dict = sample.to_datamodule_dict(
                smiles_col=ref_dm.smiles_col,
                poi_col=ref_dm.poi_col,
                poi_sequence_col=ref_dm.poi_sequence_col,
                ligase_col=ref_dm.ligase_col,
                ligase_sequence_col=ref_dm.ligase_sequence_col,
                cell_line_col=ref_dm.cell_line_col,
                assay_type_col=ref_dm.assay_type_col,
                treatment_time_col=ref_dm.treatment_time_col,
            )
            precomputed = sample.precomputed_features
        else:
            sample_dict = sample
            precomputed = sample.get('precomputed_features')
        
        featurized = {}
        
        for i, (name, datamodule) in enumerate(self.datamodules.items()):
            if model_name is not None and name != model_name:
                continue
            
            model_type = self.model_types.get(name, 'unknown')
            
            if precomputed is not None:
                featurized[name] = precomputed
            else:
                try:
                    # Validate and fill defaults for this specific datamodule
                    filled_sample, missing_required = self.validate_and_fill_defaults(
                        sample_dict, datamodule, verbose=True
                    )
                    
                    if missing_required:
                        warnings.warn(
                            f"Skipping model '{name}': missing required input(s) "
                            f"{', '.join(missing_required)}.",
                            UserWarning,
                            stacklevel=2,
                        )
                        featurized[name] = None
                        continue
                    
                    # Determine format based on model type
                    if model_type == 'xgboost':
                        features = datamodule.featurize_sample(filled_sample, return_tensor='xgb')
                    elif model_type == 'lightning':
                        features = datamodule.featurize_sample(filled_sample, return_tensor='pt')
                    else:
                        features = datamodule.featurize_sample(filled_sample, return_tensor=return_format)
                    
                    featurized[name] = features
                except Exception as e:
                    print(f"Warning: Featurization failed for {name}: {e}")
                    featurized[name] = None
        
        return featurized
    
    def predict_single_model(
        self,
        model_name: str,
        features: Any,
    ) -> np.ndarray:
        """Get prediction from a single model."""
        model = self.models[model_name]
        model_type = self.model_types[model_name]
        
        if features is None:
            raise ValueError(f"No features available for model {model_name}")
        
        if model_type == 'xgboost':
            return self._predict_xgboost(model, features)
        elif model_type == 'lightning':
            return self._predict_lightning(model, features)
        else:
            raise ValueError(f"Unknown model type for {model_name}")
    
    def _predict_xgboost(self, model: Any, features: Any) -> np.ndarray:
        """Get prediction from XGBoost model.
        
        When *features* is a ``(array, feature_names)`` tuple the method
        builds a ``pandas.DataFrame`` so that columns are aligned to the
        model's expected feature order.  This avoids errors caused by the
        inference code iterating feature keys in a different order than the
        training code.
        """        
        # Handle tuple of (features, feature_names)
        feature_names = None
        if isinstance(features, tuple) and len(features) == 2:
            features, feature_names = features
        
        if isinstance(features, np.ndarray):
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # Try to get the model's expected feature names
            model_feature_names = None
            try:
                model_feature_names = model.feature_names
            except Exception:
                pass
            
            if feature_names is not None and model_feature_names is not None:
                if set(feature_names) == set(model_feature_names) and len(feature_names) == features.shape[1]:
                    # Same features, possibly different order – build a
                    # DataFrame and reorder columns to match the model.
                    df = pd.DataFrame(features, columns=feature_names)
                    df = df[model_feature_names]
                    dmatrix = xgb.DMatrix(df)
                elif len(feature_names) == features.shape[1]:
                    dmatrix = xgb.DMatrix(features, feature_names=feature_names)
                else:
                    print(
                        f"Warning: Feature count mismatch "
                        f"({features.shape[1]} vs {len(feature_names)} names). "
                        f"Using without feature names."
                    )
                    dmatrix = xgb.DMatrix(features)
            elif feature_names is not None and len(feature_names) == features.shape[1]:
                dmatrix = xgb.DMatrix(features, feature_names=feature_names)
            else:
                dmatrix = xgb.DMatrix(features)
        elif isinstance(features, pd.DataFrame):
            dmatrix = xgb.DMatrix(features)
        elif isinstance(features, xgb.DMatrix):
            dmatrix = features
        else:
            raise ValueError(f"Unsupported feature type for XGBoost: {type(features)}")
        
        pred = model.predict(dmatrix)
        return np.atleast_1d(pred)
    
    def _predict_lightning(self, model: Any, features: Any) -> np.ndarray:
        """Get prediction from Lightning model."""
        model.eval()
        
        with torch.no_grad():
            if isinstance(features, dict):
                # Convert dict values to tensors and add batch dimension
                batch = {}
                for k, v in features.items():
                    if isinstance(v, np.ndarray):
                        tensor = torch.from_numpy(v).float()
                        if tensor.ndim == 1:
                            tensor = tensor.unsqueeze(0)
                        batch[k] = tensor.to(self.device)
                    elif isinstance(v, torch.Tensor):
                        tensor = v.float()  # Ensure float32
                        if tensor.ndim == 1:
                            tensor = tensor.unsqueeze(0)
                        batch[k] = tensor.to(self.device)
                    else:
                        batch[k] = v
                
                output = model(batch)
            elif isinstance(features, np.ndarray):
                tensor = torch.from_numpy(features).float().to(self.device)
                if tensor.ndim == 1:
                    tensor = tensor.unsqueeze(0)
                output = model(tensor)
            elif isinstance(features, torch.Tensor):
                features = features.float()  # Ensure float32
                if features.ndim == 1:
                    features = features.unsqueeze(0)
                output = model(features.to(self.device))
            else:
                output = model(features)
            
            # Extract prediction from output
            if isinstance(output, torch.Tensor):
                return output.cpu().numpy().flatten()
            elif isinstance(output, dict):
                # Try common keys for predictions
                for key in ['prediction', 'pred', 'output', 'logits']:
                    if key in output:
                        return output[key].cpu().numpy().flatten()
                # Fallback to first tensor value
                for v in output.values():
                    if isinstance(v, torch.Tensor):
                        return v.cpu().numpy().flatten()
            
            return np.atleast_1d(output)
    
    def denormalize_prediction(
        self,
        prediction: np.ndarray,
        datamodule: Optional[Any] = None,
        task_key: Optional[str] = None,
    ) -> np.ndarray:
        """
        Denormalize prediction using datamodule's inverse transform.
        
        Args:
            prediction: Normalized prediction array
            datamodule: Datamodule to use for inverse transform
            task_key: Key for the label transformer (e.g., 'Dmax', 'DC50', or label name)
            
        Returns:
            Denormalized prediction array
        """
        dm = datamodule
        
        if dm is None:
            raise ValueError(
                "A datamodule must be provided for denormalization. "
                "Cannot fall back to a reference datamodule."
            )
        
        # Check if datamodule has normalization enabled
        if not (getattr(dm, 'normalize_labels', False) or getattr(dm, 'standardize_labels', False)):
            return prediction
        
        # Determine the transformer key
        if task_key is None:
            # Infer the label name from the datamodule if possible
            task_key = dm.labels if hasattr(dm, 'labels') else None
            if isinstance(task_key, list) and len(task_key) == 1:
                task_key = task_key[0]
            else:
                print(f"Warning: Multiple or no labels found in datamodule; cannot infer task key for denormalization. ")
                return prediction
        
        try:
            return dm.inverse_transform_labels(prediction, task_key)
        except (ValueError, KeyError) as e:
            print(f"Warning: Could not denormalize predictions: {e}")
            return prediction
    
    def predict(
        self,
        sample: Union[SampleInput, Dict[str, Any]],
        return_individual: bool = True,
        tasks: Optional[List[str]] = None,
    ) -> Dict[str, EnsemblePrediction]:
        """Make ensemble prediction, returning one result per task.

        Each model's raw prediction is **denormalized first** using its own
        datamodule, then the weighted mean and uncertainty metrics are
        computed in the original (denormalized) scale.

        Args:
            sample: ``SampleInput`` object or dictionary with input data.
            return_individual: Whether to return individual model predictions.
            tasks: Optional subset of tasks to predict.  ``None`` means all
                available tasks.

        Returns:
            Dictionary mapping task name to ``EnsemblePrediction``.
        """
        # Featurize input for all models
        features = self.featurize_input(sample)

        # Gather raw predictions, denormalize each, and group by task
        # Structure: {task_name: {model_name: denormalized_prediction}}
        task_preds: Dict[str, Dict[str, np.ndarray]] = {}
        allowed_tasks = set(tasks) if tasks else None

        for model_name in self.models.keys():
            if model_name not in self.weights:
                continue

            model_task = self.model_tasks.get(model_name, None)
            if allowed_tasks is not None and model_task not in allowed_tasks:
                continue

            model_features = features.get(model_name)
            if model_features is None:
                continue

            try:
                raw_pred = self.predict_single_model(model_name, model_features)
                # Denormalize using this model's own datamodule
                dm = self.datamodules[model_name]
                denorm_pred = self.denormalize_prediction(raw_pred, dm)
                task_preds.setdefault(model_task, {})[model_name] = denorm_pred
            except Exception as e:
                error_msg = str(e)
                if 'feature_names mismatch' in error_msg:
                    error_msg = "feature_names mismatch (training/inference feature encoding incompatible)"
                print(f"Warning: Prediction failed for model={model_name}: {error_msg}")
                continue

        if not task_preds:
            raise RuntimeError("All model predictions failed")

        # Build one EnsemblePrediction per task (all in denormalized scale)
        results = {}
        for task_name, preds_denorm in task_preds.items():
            # Weighted average in denormalized scale
            weighted_sum = np.zeros_like(list(preds_denorm.values())[0], dtype=float)
            weight_sum = 0.0
            for model_name, pred in preds_denorm.items():
                w = self.weights.get(model_name, 0.0)
                weighted_sum += pred * w
                weight_sum += w
            weighted_mean = weighted_sum / weight_sum if weight_sum > 0 else weighted_sum

            # Standard deviation across denormalized predictions
            all_preds = np.array(list(preds_denorm.values()))
            uncertainty_std = np.std(all_preds, axis=0)

            # Binary classification entropy
            predictive_entropy = None
            if task_name == 'bin':
                avg_pred = np.clip(weighted_mean, 1e-7, 1 - 1e-7)
                predictive_entropy = -(
                    avg_pred * np.log(avg_pred)
                    + (1 - avg_pred) * np.log(1 - avg_pred)
                )

            # Task label name
            task_to_label = {
                'dmax': 'Dmax (%)',
                'dc50': 'DC50 (nM)',
                'bin': 'Binary Activity',
            }
            label_name = task_to_label.get(task_name, 'Unknown label name')

            results[task_name] = EnsemblePrediction(
                weighted_mean=weighted_mean,
                uncertainty_std=uncertainty_std,
                individual_predictions=preds_denorm if return_individual else {},
                weights={k: v for k, v in self.weights.items() if k in preds_denorm},
                model_names=list(preds_denorm.keys()),
                task=task_name,
                label_name=label_name,
                predictive_entropy=predictive_entropy,
            )

        return results
    
    def predict_batch(
        self,
        samples: List[Union[SampleInput, Dict[str, Any]]],
    ) -> List[Dict[str, EnsemblePrediction]]:
        """Make batch predictions.

        Args:
            samples: List of SampleInput objects or dictionaries.

        Returns:
            List of dicts mapping task name to ``EnsemblePrediction``
            (one dict per sample).
        """
        results = []
        
        for i, sample in enumerate(samples):
            try:
                result = self.predict(sample)
                results.append(result)
            except Exception as e:
                print(f"Warning: Prediction failed for sample {i}: {e}")
                results.append(None)
        
        return results
    
    def predict_dataframe(
        self,
        df: pd.DataFrame,
        smiles_col: str = "SMILES",
        poi_col: Optional[str] = None,
        poi_sequence_col: Optional[str] = None,
        ligase_col: Optional[str] = None,
        cell_line_col: Optional[str] = None,
        treatment_time_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """Make predictions for a DataFrame.

        Adds one set of prediction columns per task present in the ensemble.
        """
        all_row_results = []

        for idx, row in df.iterrows():
            sample = SampleInput(
                smiles=row.get(smiles_col) if smiles_col in row else None,
                poi_name=row.get(poi_col) if poi_col and poi_col in row else None,
                poi_sequence=row.get(poi_sequence_col) if poi_sequence_col and poi_sequence_col in row else None,
                ligase_name=row.get(ligase_col) if ligase_col and ligase_col in row else None,
                cell_line=row.get(cell_line_col) if cell_line_col and cell_line_col in row else None,
                treatment_time=row.get(treatment_time_col) if treatment_time_col and treatment_time_col in row else None,
            )

            row_dict: Dict[str, Any] = {}
            try:
                task_results = self.predict(sample, return_individual=True)
                for task_name, result in task_results.items():
                    suffix = f"_{task_name}" if len(task_results) > 1 else ""
                    row_dict[f'prediction{suffix}'] = result.weighted_mean[0] if result.weighted_mean is not None else np.nan
                    row_dict[f'uncertainty{suffix}'] = result.uncertainty_std[0] if result.uncertainty_std is not None else np.nan
                    row_dict[f'ci_pctl_lower{suffix}'] = result.ci_percentile_lower_95[0] if result.ci_percentile_lower_95 is not None else np.nan
                    row_dict[f'ci_pctl_upper{suffix}'] = result.ci_percentile_upper_95[0] if result.ci_percentile_upper_95 is not None else np.nan
                    row_dict[f'ci_sem_lower{suffix}'] = result.ci_sem_lower_95[0] if result.ci_sem_lower_95 is not None else np.nan
                    row_dict[f'ci_sem_upper{suffix}'] = result.ci_sem_upper_95[0] if result.ci_sem_upper_95 is not None else np.nan
            except Exception as e:
                row_dict['error'] = str(e)

            all_row_results.append(row_dict)

        result_df = pd.DataFrame(all_row_results)
        return pd.concat([df.reset_index(drop=True), result_df], axis=1)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            'n_models': len(self.models),
            'model_names': list(self.models.keys()),
            'model_types': self.model_types,
            'model_tasks': self.model_tasks,
            'weights': self.weights,
            'available_tasks': self.available_tasks,
            'device': self.device,
            'categorical_choices': self.get_categorical_choices(),
        }
    
    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """Update ensemble weights."""
        total = sum(new_weights.values())
        self.weights = {k: v / total for k, v in new_weights.items() if k in self.models}
    
    def __repr__(self) -> str:
        return (
            f"EnsemblePredictor(n_models={len(self.models)}, "
            f"device='{self.device}')"
        )
