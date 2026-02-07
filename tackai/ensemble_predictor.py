"""
Ensemble Predictor for STAEDA Models
Handles loading and prediction from multiple model types (XGBoost, Lightning)
with weighted averaging and uncertainty quantification.
"""
import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Literal
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings('ignore')


@dataclass
class EnsemblePrediction:
    """Container for ensemble prediction results."""
    # Core predictions (in original scale after denormalization)
    weighted_mean: np.ndarray
    uncertainty_std: np.ndarray
    
    # Raw predictions (normalized, before inverse transform)
    weighted_mean_normalized: np.ndarray
    uncertainty_std_normalized: np.ndarray
    
    # Individual model info
    individual_predictions: Dict[str, np.ndarray]
    individual_predictions_normalized: Dict[str, np.ndarray]
    weights: Dict[str, float]
    model_names: List[str]
    
    # Task and label info
    task: str = 'dmax'
    label_name: Optional[str] = None
    
    # Additional uncertainty metrics (computed in original scale)
    prediction_variance: np.ndarray = field(default=None)
    prediction_range: np.ndarray = field(default=None)
    prediction_iqr: np.ndarray = field(default=None)
    
    # For binary classification
    predictive_entropy: Optional[np.ndarray] = None
    
    # Confidence intervals
    ci_lower_95: Optional[np.ndarray] = None
    ci_upper_95: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Compute additional uncertainty metrics."""
        if not self.individual_predictions:
            return
            
        preds = np.array(list(self.individual_predictions.values()))
        
        self.prediction_variance = np.var(preds, axis=0)
        self.prediction_range = np.max(preds, axis=0) - np.min(preds, axis=0)
        
        q75 = np.percentile(preds, 75, axis=0)
        q25 = np.percentile(preds, 25, axis=0)
        self.prediction_iqr = q75 - q25
        
        # 95% confidence intervals
        self.ci_lower_95 = self.weighted_mean - 1.96 * self.uncertainty_std
        self.ci_upper_95 = self.weighted_mean + 1.96 * self.uncertainty_std
    
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
            'weighted_mean_normalized': to_list(self.weighted_mean_normalized),
            'ci_lower_95': to_list(self.ci_lower_95),
            'ci_upper_95': to_list(self.ci_upper_95),
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
            f"95% CI: [{self.ci_lower_95[0]:.4f}, {self.ci_upper_95[0]:.4f}]",
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


# Default values for missing inputs based on common training data
DEFAULT_VALUES = {
    'POI_Name': 'BRD4',
    'POI_Sequence': 'MSSPQDLKNNIAQEYLTSQVLPGHTPPPPLLKKA',  # Short placeholder, will be extended
    'Ligase_Name': 'VHL',
    'Ligase_Sequence': 'MPRRAENWDE...',  # Placeholder
    'Cell_Line_ID': 'HEK293',
    'Assay': 'Dmax',
    'Assay_Time': 24.0,  # Default 24 hours
    'Degrader_Type': 'PROTAC',
}

# Commonly used BRD4 sequence for placeholder
BRD4_SEQUENCE = (
    "MSSPQDLKNNIAQEYLTSQVLPGHTPPPPLLKKAPKVKPLPPPLPPAPASGQKKQQQQQPQQQ"
    "QPPPPPKKPHMERGNGKEKSTSGKLPNLVNGEGGKPWKIGKKENISSILPMCKIKDLLHSDC"
    "ACLAWSEKDREEKQRLLAIRQQQLLQLEGLQQHQQQLQQQQQQQQQQQQQQQQQQLQQQQQQ"
    "QQQQQLQPPPPPQPHLPPPPQPQLPQQQQQQQQQQQQQQQQQQQQQQQQQQQLQQQQPPPPP"
    "PPPPPPPPPPPPPPQQQQQQQQQQQQQQQQLQQQQQLQQQQQLQQQQQQQQQQQQQQQQLQQ"
    "QLQQQQLQQQQQQQQQQLQQQQQQQQQQQQQQQQQQQQQQQQQQQQLQPQQQPQQLPPPPPP"
)


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
        task: str = 'dmax',
        label_name: Optional[str] = None,
        device: str = 'cpu',
        denormalize: bool = True,
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
            denormalize: Whether to denormalize predictions
        """
        self.models = models
        self.datamodules = datamodules
        self.task = task.lower()
        self.device = device
        self.denormalize = denormalize
        
        # Infer label name if not provided
        if label_name is None:
            label_name = self._infer_label_name()
        self.label_name = label_name
        
        # Set up weights
        if weights is None:
            n_models = len(models)
            self.weights = {name: 1.0 / n_models for name in models.keys()}
        else:
            total = sum(weights.values())
            self.weights = {k: v / total for k, v in weights.items()}
        
        # Validate weights
        for name in self.weights.keys():
            if name not in self.models:
                raise ValueError(f"Weight specified for unknown model: {name}")
        
        # Model type tracking
        self.model_types = {}
        for name, model in self.models.items():
            self.model_types[name] = self._get_model_type(model)
        
        # Get a reference datamodule for shared operations
        self._ref_datamodule = self._get_reference_datamodule()
    
    def _infer_label_name(self) -> Optional[str]:
        """Infer label name from task or datamodule."""
        task_to_label = {
            'dmax': 'Dmax (%) (DC50/Dmax)',
            'dc50': 'pDC50 (DC50/Dmax)',
            'bin': 'Binary_Activity',
        }
        
        # Try to get from datamodule first
        for dm in self.datamodules.values():
            if dm is not None and hasattr(dm, 'labels') and dm.labels:
                return dm.labels[0]
        
        return task_to_label.get(self.task)
    
    def _get_reference_datamodule(self) -> Optional[Any]:
        """Get a reference datamodule for shared operations like denormalization."""
        for dm in self.datamodules.values():
            if dm is not None:
                return dm
        return None
    
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
    ) -> Dict[str, Any]:
        """
        Validate inputs and fill in defaults for missing values.
        
        Args:
            sample_dict: Dictionary of input values
            datamodule: Datamodule to check requirements against
            verbose: Whether to print warnings about missing inputs
            
        Returns:
            Dictionary with defaults filled in for missing values
        """
        dm = datamodule or self._ref_datamodule
        if dm is None:
            return sample_dict
        
        filled = dict(sample_dict)
        missing_warnings = []
        
        # Map standard column names to sample_dict keys and defaults
        col_mappings = [
            (dm.smiles_col, 'SMILES', None),  # SMILES is required, no default
            (dm.poi_col, 'POI_Name', DEFAULT_VALUES.get('POI_Name')),
            (dm.poi_sequence_col, 'POI_Sequence', BRD4_SEQUENCE),
            (dm.ligase_col, 'Ligase_Name', DEFAULT_VALUES.get('Ligase_Name')),
            (dm.ligase_sequence_col, 'Ligase_Sequence', DEFAULT_VALUES.get('Ligase_Sequence')),
            (dm.cell_line_col, 'Cell_Line_ID', DEFAULT_VALUES.get('Cell_Line_ID')),
            (dm.assay_type_col, 'Assay', DEFAULT_VALUES.get('Assay')),
            (dm.treatment_time_col, 'Assay_Time', DEFAULT_VALUES.get('Assay_Time')),
        ]
        
        for dm_col, default_key, default_val in col_mappings:
            # Check if this column is needed
            is_needed = False
            if dm_col == dm.smiles_col:
                is_needed = True
            elif dm_col == dm.poi_col and (getattr(dm, 'use_poi_name_embedding', False) or 
                                           getattr(dm, 'poi_embeddings_id_type', '') != 'sequence'):
                is_needed = True
            elif dm_col == dm.poi_sequence_col and (getattr(dm, 'use_poi_sequence_embedding', False) or
                                                     getattr(dm, 'use_poi_precomputed_embedding', False)):
                is_needed = True
            elif dm_col == dm.ligase_col and getattr(dm, 'use_ligase_name_embedding', False):
                is_needed = True
            elif dm_col == dm.ligase_sequence_col and getattr(dm, 'use_ligase_precomputed_embedding', False):
                is_needed = True
            elif dm_col == dm.cell_line_col and (getattr(dm, 'use_cell_description_embedding', False) or
                                                  getattr(dm, 'use_cell_name_embedding', False)):
                is_needed = True
            elif dm_col == dm.treatment_time_col and getattr(dm, 'use_treatment_time', False):
                is_needed = True
            elif dm_col == dm.assay_type_col and getattr(dm, 'use_assay_type_encoding', False):
                is_needed = True
            
            # Check if value is missing/None
            current_val = filled.get(dm_col)
            if is_needed and (current_val is None or (isinstance(current_val, str) and current_val.strip() == '')):
                if default_val is not None:
                    filled[dm_col] = default_val
                    missing_warnings.append(f"  - {dm_col}: using default '{default_val if len(str(default_val)) < 30 else str(default_val)[:30] + '...'}'") 
                elif dm_col == dm.smiles_col:
                    raise ValueError(f"SMILES is required but not provided")
        
        if verbose and missing_warnings:
            print(f"Note: Some inputs were missing and filled with defaults:")
            for w in missing_warnings[:3]:  # Show max 3 warnings
                print(w)
            if len(missing_warnings) > 3:
                print(f"  ... and {len(missing_warnings) - 3} more")
        
        return filled
    
    def get_xgb_feature_names(self, datamodule: Any, sample_dict: Dict[str, Any]) -> List[str]:
        """
        Get feature names for XGBoost model from a sample featurization.
        
        Args:
            datamodule: The datamodule to use for featurization
            sample_dict: Sample input dictionary
            
        Returns:
            List of feature names in order they appear in concatenated features
        """
        # Get features as dict to see the names
        features = datamodule.featurize_sample(sample_dict, return_tensor='np')
        
        feature_names = []
        for key in sorted(features.keys()):
            value = features[key]
            if hasattr(value, '__len__') and not isinstance(value, str):
                # Multi-dimensional feature
                for i in range(len(value.flatten())):
                    feature_names.append(f"{key}_{i}")
            else:
                feature_names.append(key)
        
        return feature_names
    
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
        task: str = 'dmax',
        label_name: Optional[str] = None,
        device: str = 'cpu',
        pattern: Optional[str] = None,
        denormalize: bool = True,
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
            denormalize: Whether to denormalize predictions
            
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
        
        for model_file in model_files:
            model_name = model_file.stem
            
            try:
                model, datamodule = cls._load_model_and_datamodule(
                    model_file, task, device, datamodule_dir
                )
                models[model_name] = model
                datamodules[model_name] = datamodule
                print(f"  Loaded: {model_name} ({cls._get_model_type(model)})")
            except Exception as e:
                print(f"  Failed to load {model_name}: {e}")
                continue
        
        if not models:
            raise ValueError(f"No models could be loaded from {model_dir}")
        
        return cls(models, datamodules, weights, task, label_name, device, denormalize)
    
    @classmethod
    def from_weights_file(
        cls,
        weights_file: Union[str, Path],
        model_dir: Union[str, Path],
        datamodule_dir: Optional[Union[str, Path]] = None,
        device: str = 'cpu',
        denormalize: bool = True,
    ) -> 'EnsemblePredictor':
        """
        Load an ensemble predictor from a weights JSON file.
        
        Only loads the models specified in the weights file.
        
        Args:
            weights_file: Path to the JSON weights file
            model_dir: Directory containing model checkpoints
            datamodule_dir: Directory containing datamodule state dicts
            device: Device for inference
            denormalize: Whether to denormalize predictions
            
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
            denormalize=denormalize,
        )
    
    @staticmethod
    def _load_model_and_datamodule(
        model_path: Path,
        task: str,
        device: str,
        datamodule_dir: Optional[Path] = None,
    ) -> Tuple[Any, Any]:
        """Load a model and its corresponding datamodule."""
        suffix = model_path.suffix.lower()
        
        if suffix == '.ckpt':
            return EnsemblePredictor._load_lightning_model(
                model_path, task, device, datamodule_dir
            )
        elif suffix in ['.json', '.ubj', '.pkl']:
            return EnsemblePredictor._load_xgboost_model(
                model_path, task, datamodule_dir
            )
        else:
            raise ValueError(f"Unsupported model format: {suffix}")
    
    @staticmethod
    def _load_lightning_model(
        model_path: Path,
        task: str,
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
                    # Load from YAML + state dict using staeda's load_datamodule
                    from tackai.data.datamodule import load_datamodule
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
            # Load Lightning model using STAEDAModel.load_from_checkpoint
            from tackai.models.staeda_model import STAEDAModel
            model = STAEDAModel.load_from_checkpoint(
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
        task: str,
        datamodule_dir: Optional[Path] = None,
    ) -> Tuple[Any, Any]:
        """Load an XGBoost model."""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost is required to load XGBoost models")
        
        suffix = model_path.suffix.lower()
        
        if suffix in ['.json', '.ubj']:
            model = xgb.Booster()
            model.load_model(str(model_path))
        elif suffix == '.pkl':
            import pickle
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
            if self._ref_datamodule is not None:
                sample_dict = sample.to_datamodule_dict(
                    smiles_col=self._ref_datamodule.smiles_col,
                    poi_col=self._ref_datamodule.poi_col,
                    poi_sequence_col=self._ref_datamodule.poi_sequence_col,
                    ligase_col=self._ref_datamodule.ligase_col,
                    ligase_sequence_col=self._ref_datamodule.ligase_sequence_col,
                    cell_line_col=self._ref_datamodule.cell_line_col,
                    assay_type_col=self._ref_datamodule.assay_type_col,
                    treatment_time_col=self._ref_datamodule.treatment_time_col,
                )
            else:
                sample_dict = sample.to_datamodule_dict()
            precomputed = sample.precomputed_features
        else:
            sample_dict = sample
            precomputed = sample.get('precomputed_features')
        
        featurized = {}
        
        for name, datamodule in self.datamodules.items():
            if model_name is not None and name != model_name:
                continue
            
            model_type = self.model_types.get(name, 'unknown')
            
            if precomputed is not None:
                featurized[name] = precomputed
            elif datamodule is not None:
                try:
                    # Validate and fill defaults for this specific datamodule
                    filled_sample = self.validate_and_fill_defaults(
                        sample_dict, datamodule, verbose=False
                    )
                    
                    # Determine format based on model type
                    if model_type == 'xgboost':
                        # For XGBoost, get features as dict first to preserve names
                        features_dict = datamodule.featurize_sample(filled_sample, return_tensor='np')
                        # Store as tuple: (flattened_array, feature_names)
                        feature_list = []
                        feature_names = []
                        for key in sorted(features_dict.keys()):
                            value = features_dict[key]
                            flat_value = value.flatten()
                            feature_list.append(flat_value)
                            for i in range(len(flat_value)):
                                feature_names.append(f"{key}_{i}")
                        features = (np.concatenate(feature_list).astype(np.float32), feature_names)
                    elif model_type == 'lightning':
                        features = datamodule.featurize_sample(filled_sample, return_tensor='pt')
                    else:
                        features = datamodule.featurize_sample(filled_sample, return_tensor=return_format)
                    
                    featurized[name] = features
                except Exception as e:
                    print(f"Warning: Featurization failed for {name}: {e}")
                    featurized[name] = None
            else:
                # No datamodule, pass raw input
                featurized[name] = sample_dict
        
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
        """Get prediction from XGBoost model."""
        import xgboost as xgb
        
        # Handle tuple of (features, feature_names)
        feature_names = None
        if isinstance(features, tuple) and len(features) == 2:
            features, feature_names = features
        
        if isinstance(features, np.ndarray):
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # Try to get feature names from the model if not provided
            if feature_names is None:
                try:
                    model_feature_names = model.feature_names
                    if model_feature_names:
                        feature_names = model_feature_names
                except:
                    pass
            
            # Create DMatrix with feature names if available
            if feature_names is not None:
                # Check if number of features matches
                if len(feature_names) == features.shape[1]:
                    dmatrix = xgb.DMatrix(features, feature_names=feature_names)
                else:
                    # Feature count mismatch - try without names
                    print(f"Warning: Feature count mismatch ({features.shape[1]} vs {len(feature_names)} names). Using without feature names.")
                    dmatrix = xgb.DMatrix(features)
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
        if not self.denormalize:
            return prediction
        
        dm = datamodule or self._ref_datamodule
        
        if dm is None:
            return prediction
        
        # Check if datamodule has normalization enabled
        if not (getattr(dm, 'normalize_labels', False) or getattr(dm, 'standardize_labels', False)):
            return prediction
        
        # Determine the transformer key
        if task_key is None:
            # Try to infer from task or label name
            if self.label_name and self.label_name in getattr(dm, 'label_transformers', {}):
                task_key = self.label_name
            elif self.task.lower() == 'dmax':
                task_key = 'Dmax'
            elif self.task.lower() == 'dc50':
                task_key = 'DC50'
            else:
                task_key = self.label_name
        
        try:
            return dm.inverse_transform_labels(prediction, task_key)
        except (ValueError, KeyError) as e:
            print(f"Warning: Could not denormalize predictions: {e}")
            return prediction
    
    def predict(
        self,
        sample: Union[SampleInput, Dict[str, Any]],
        return_individual: bool = True,
    ) -> EnsemblePrediction:
        """
        Make ensemble prediction.
        
        Args:
            sample: SampleInput object or dictionary with input data
            return_individual: Whether to return individual model predictions
            
        Returns:
            EnsemblePrediction object with weighted mean, uncertainty, and details
        """
        # Featurize input for all models
        features = self.featurize_input(sample)
        
        # Collect predictions from all models
        individual_predictions_normalized = {}
        
        for model_name in self.models.keys():
            if model_name not in self.weights:
                continue
            
            model_features = features.get(model_name)
            if model_features is None:
                # Try to use any available features
                available = [f for f in features.values() if f is not None]
                if available:
                    model_features = available[0]
            
            try:
                pred = self.predict_single_model(model_name, model_features)
                individual_predictions_normalized[model_name] = pred
            except Exception as e:
                error_msg = str(e)
                # Shorten feature_names mismatch errors (they can be very long)
                if 'feature_names mismatch' in error_msg:
                    error_msg = "feature_names mismatch (training/inference feature encoding incompatible)"
                print(f"Warning: Prediction failed for model={model_name}: {error_msg}")
                continue
        
        if not individual_predictions_normalized:
            raise RuntimeError("All model predictions failed")
        
        # Compute weighted average (normalized)
        weighted_sum = np.zeros_like(list(individual_predictions_normalized.values())[0], dtype=float)
        weight_sum = 0.0
        
        for model_name, pred in individual_predictions_normalized.items():
            weight = self.weights.get(model_name, 0.0)
            weighted_sum += pred * weight
            weight_sum += weight
        
        weighted_mean_normalized = weighted_sum / weight_sum if weight_sum > 0 else weighted_sum
        
        # Compute uncertainty in normalized space
        all_preds_normalized = np.array(list(individual_predictions_normalized.values()))
        uncertainty_std_normalized = np.std(all_preds_normalized, axis=0)
        
        # Denormalize predictions
        individual_predictions = {}
        for model_name, pred in individual_predictions_normalized.items():
            dm = self.datamodules.get(model_name) or self._ref_datamodule
            individual_predictions[model_name] = self.denormalize_prediction(pred, dm)
        
        weighted_mean = self.denormalize_prediction(weighted_mean_normalized)
        
        # Compute uncertainty in original scale
        # Use delta method approximation: denormalize mean ± std
        upper = self.denormalize_prediction(weighted_mean_normalized + uncertainty_std_normalized)
        lower = self.denormalize_prediction(weighted_mean_normalized - uncertainty_std_normalized)
        uncertainty_std = (upper - lower) / 2.0
        
        # Compute additional metrics for binary classification
        predictive_entropy = None
        if self.task == 'bin':
            avg_pred = np.clip(weighted_mean, 1e-7, 1 - 1e-7)
            predictive_entropy = -(
                avg_pred * np.log(avg_pred) +
                (1 - avg_pred) * np.log(1 - avg_pred)
            )
        
        return EnsemblePrediction(
            weighted_mean=weighted_mean,
            uncertainty_std=uncertainty_std,
            weighted_mean_normalized=weighted_mean_normalized,
            uncertainty_std_normalized=uncertainty_std_normalized,
            individual_predictions=individual_predictions if return_individual else {},
            individual_predictions_normalized=individual_predictions_normalized if return_individual else {},
            weights={k: v for k, v in self.weights.items() if k in individual_predictions_normalized},
            model_names=list(individual_predictions_normalized.keys()),
            task=self.task,
            label_name=self.label_name,
            predictive_entropy=predictive_entropy,
        )
    
    def predict_batch(
        self,
        samples: List[Union[SampleInput, Dict[str, Any]]],
    ) -> List[EnsemblePrediction]:
        """
        Make batch predictions.
        
        Args:
            samples: List of SampleInput objects or dictionaries
            
        Returns:
            List of EnsemblePrediction objects
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
        """
        Make predictions for a DataFrame.
        
        Args:
            df: Input DataFrame
            smiles_col: Column name for SMILES
            poi_col: Column name for POI name
            poi_sequence_col: Column name for POI sequence
            ligase_col: Column name for ligase name
            cell_line_col: Column name for cell line
            treatment_time_col: Column name for treatment time
            
        Returns:
            DataFrame with predictions added
        """
        results = []
        
        for idx, row in df.iterrows():
            sample = SampleInput(
                smiles=row.get(smiles_col) if smiles_col in row else None,
                poi_name=row.get(poi_col) if poi_col and poi_col in row else None,
                poi_sequence=row.get(poi_sequence_col) if poi_sequence_col and poi_sequence_col in row else None,
                ligase_name=row.get(ligase_col) if ligase_col and ligase_col in row else None,
                cell_line=row.get(cell_line_col) if cell_line_col and cell_line_col in row else None,
                treatment_time=row.get(treatment_time_col) if treatment_time_col and treatment_time_col in row else None,
            )
            
            try:
                result = self.predict(sample, return_individual=True)  # Need individual for CI
                ci_lower = result.ci_lower_95[0] if result.ci_lower_95 is not None else np.nan
                ci_upper = result.ci_upper_95[0] if result.ci_upper_95 is not None else np.nan
                results.append({
                    'prediction': result.weighted_mean[0] if result.weighted_mean is not None else np.nan,
                    'uncertainty': result.uncertainty_std[0] if result.uncertainty_std is not None else np.nan,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                })
            except Exception as e:
                results.append({
                    'prediction': np.nan,
                    'uncertainty': np.nan,
                    'ci_lower': np.nan,
                    'ci_upper': np.nan,
                    'error': str(e),
                })
        
        result_df = pd.DataFrame(results)
        return pd.concat([df.reset_index(drop=True), result_df], axis=1)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            'n_models': len(self.models),
            'model_names': list(self.models.keys()),
            'model_types': self.model_types,
            'weights': self.weights,
            'task': self.task,
            'label_name': self.label_name,
            'device': self.device,
            'denormalize': self.denormalize,
            'has_datamodule': self._ref_datamodule is not None,
        }
    
    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """Update ensemble weights."""
        total = sum(new_weights.values())
        self.weights = {k: v / total for k, v in new_weights.items() if k in self.models}
    
    def __repr__(self) -> str:
        return (
            f"EnsemblePredictor(n_models={len(self.models)}, "
            f"task='{self.task}', label='{self.label_name}', device='{self.device}')"
        )
