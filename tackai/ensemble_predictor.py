"""
Ensemble Predictor for TACK Models
Handles loading and prediction from multiple model types (XGBoost, Lightning)
with weighted averaging and uncertainty quantification.
"""
import gc
import os
import re
import json
import time
import pickle
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Literal
from dataclasses import dataclass, field

# xgboost must be imported before torch to avoid an OpenMP runtime conflict on
# macOS: torch's libtorch sets up Intel OpenMP (libiomp5), and if xgb.Booster()
# is first called afterwards it initialises a second OpenMP runtime → SIGSEGV.
import xgboost as xgb
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from tackai.data.datamodule import DegradationComplexDataModule, load_datamodule
# TACKModel is imported lazily inside _load_lightning_model so that XGBoost-only
# ensembles never trigger the PyTorch/OpenMP initialisation that conflicts with
# XGBoost's own OpenMP runtime on macOS.

warnings.filterwarnings('ignore')


@dataclass
class EnsemblePrediction:
    """Container for ensemble prediction results.

    All predictions are stored in the **original (denormalized) scale**.
    Two kinds of 95% confidence intervals are provided:

    * **Percentile CI** – non-parametric, based on the 2.5th and 97.5th
      percentiles of the individual model predictions.
    * **SEM CI** – parametric, based on the standard error of the mean
      (assumes approximate normality).
    """
    weighted_mean: np.ndarray
    uncertainty_std: np.ndarray

    individual_predictions: Dict[str, np.ndarray]
    weights: Dict[str, float]
    model_names: List[str]

    task: str = 'dmax'
    label_name: Optional[str] = None

    prediction_variance: np.ndarray = field(default=None)
    prediction_range: np.ndarray = field(default=None)
    prediction_iqr: np.ndarray = field(default=None)

    predictive_entropy: Optional[np.ndarray] = None

    ci_percentile_lower_95: Optional[np.ndarray] = None
    ci_percentile_upper_95: Optional[np.ndarray] = None

    ci_sem_lower_95: Optional[np.ndarray] = None
    ci_sem_upper_95: Optional[np.ndarray] = None

    def __post_init__(self):
        if not self.individual_predictions:
            return

        preds = np.array(list(self.individual_predictions.values()))
        n_models = len(preds)

        self.prediction_variance = np.var(preds, axis=0)
        self.prediction_range = np.max(preds, axis=0) - np.min(preds, axis=0)
        self.prediction_iqr = np.percentile(preds, 75, axis=0) - np.percentile(preds, 25, axis=0)

        self.ci_percentile_lower_95 = np.percentile(preds, 2.5, axis=0)
        self.ci_percentile_upper_95 = np.percentile(preds, 97.5, axis=0)

        sem = np.std(preds, ddof=1, axis=0) / np.sqrt(n_models) if n_models > 1 else np.zeros_like(self.weighted_mean)
        self.ci_sem_lower_95 = self.weighted_mean - 1.96 * sem
        self.ci_sem_upper_95 = self.weighted_mean + 1.96 * sem

    def to_dict(self) -> Dict[str, Any]:
        def to_list(arr):
            return arr.tolist() if isinstance(arr, np.ndarray) else arr

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
        return "\n".join([
            f"Task: {self.task.upper()}",
            f"Label: {self.label_name or 'N/A'}",
            f"Number of models: {len(self.model_names)}",
            f"Prediction: {self.weighted_mean[0]:.4f} ± {self.uncertainty_std[0]:.4f}",
            f"95% CI (percentile): [{self.ci_percentile_lower_95[0]:.4f}, {self.ci_percentile_upper_95[0]:.4f}]",
            f"95% CI (SEM):        [{self.ci_sem_lower_95[0]:.4f}, {self.ci_sem_upper_95[0]:.4f}]",
        ])


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


# Feature-key prefixes treated as SMILES-dependent in the SMILES/context split.
SMILES_FEATURE_PREFIXES: Tuple[str, ...] = (
    'Feature_Fingerprint',
    'Feature_Descriptor_',
)

# Default values for optional inputs.
DEFAULT_VALUES = {
    'Cell_Line_ID': 'Unknown cell line.',
    'Assay': 'Unknown',
    'Assay_Time': 24.0,
    'Degrader_Type': 'PROTAC',
}

KNOWN_TASKS = {'dmax', 'dc50', 'bin', 'dmax_bin', 'dc50_bin', 'multitask'}


@dataclass
class PreprocessedContext:
    """Per-model encoded context for fast repeated SMILES screening.

    Produced by :meth:`EnsemblePredictor.transform_context`; pass to
    :meth:`EnsemblePredictor.predict` as the ``context`` argument together
    with a list of SMILES strings to score them against a fixed biological
    context without re-encoding protein / cell-line embeddings.
    """
    context_features: Dict[str, Dict[str, np.ndarray]]
    xgb_layouts: Dict[str, List[Tuple[str, int, int]]]
    xgb_total_dims: Dict[str, int]
    xgb_row_template: Dict[str, np.ndarray]
    xgb_feature_names: Dict[str, List[str]]
    predictor_id: int = 0
    source_context: Dict[str, Any] = field(default_factory=dict)


class EnsemblePredictor:
    """Ensemble predictor that loads and combines predictions from multiple models.

    Supports XGBoost and PyTorch Lightning models with weighted averaging and
    uncertainty quantification. Use :meth:`from_directory` to instantiate.

    Example:
        >>> predictor = EnsemblePredictor.from_directory(
        ...     model_dir='ensembles/dmax',
        ...     weights_file='ensemble_weights_dmax.json',
        ... )
        >>> result = predictor.predict({
        ...     'SMILES': 'CCO', 'POI_Name': 'BRD4', ...
        ... })
    """

    TASK_LABELS = {
        'dmax': 'Dmax (%)',
        'dc50': 'DC50 (nM)',
        'bin': 'Binary Activity',
    }

    def __init__(
        self,
        models: Dict[str, Any],
        datamodules: Dict[str, Any],
        weights: Dict[str, float],
        device: str = 'cpu',
        n_jobs: Optional[int] = None,
    ) -> None:
        """Private constructor. Use :meth:`from_directory` instead.

        Args:
            models: Mapping of model name → model object.
            datamodules: Mapping of model name → fitted DegradationComplexDataModule.
            weights: Mapping of model name → ensemble weight.
            device: Inference device ('cpu' or 'cuda').
            n_jobs: XGBoost thread count (None = all cores).
        """
        self.models = models
        self.datamodules = datamodules
        self.weights = weights
        self.device = 'cuda' if device == 'gpu' else device
        self.n_jobs = n_jobs

        # Lazy-loading state — populated by from_directory(lazy_loading=True)
        self._lazy: bool = False
        self._model_paths: Dict[str, Path] = {}
        self._dm_paths: Dict[str, Tuple[Optional[Path], Path]] = {}
        self._hparam_overrides: Optional[Dict[str, Any]] = None

        if n_jobs is not None and n_jobs > 1:
            omp = os.environ.get('OMP_NUM_THREADS')
            if omp is None or int(omp) < n_jobs:
                os.environ['OMP_NUM_THREADS'] = str(n_jobs)
                if omp is not None:
                    warnings.warn(
                        f"OMP_NUM_THREADS was {omp}, overriding to {n_jobs} "
                        "so XGBoost can use the requested threads.",
                        UserWarning, stacklevel=2,
                    )
            try:
                xgb.set_config(nthread=n_jobs)
            except (TypeError, AttributeError):
                pass

        self.model_tasks = {
            name: self._infer_model_task(name, datamodules.get(name))
            for name in models
        }
        self.model_types = {name: self._get_model_type(model) for name, model in models.items()}

        missing_dm = [name for name in models if datamodules.get(name) is None]
        if missing_dm:
            raise ValueError(
                f"The following models have no associated datamodule: {missing_dm}. "
                "Each model must have a corresponding datamodule for featurization "
                "and denormalization."
            )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_directory(
        cls,
        model_dir: Union[str, Path],
        weights_file: Optional[Union[str, Path]] = None,
        device: str = 'cpu',
        n_jobs: Optional[int] = None,
        pattern: Optional[str] = None,
        hparam_overrides: Optional[Dict[str, Any]] = None,
        lazy_loading: bool = True,
    ) -> 'EnsemblePredictor':
        """Load ensemble from a directory containing models and datamodules.

        Args:
            model_dir: Directory containing model files and paired datamodule
                state dicts (``*_state.pt`` / ``*_hparams.yaml``).
            weights_file: Optional JSON file with per-model weights. Only the
                models listed in the file are loaded; their weights are used for
                the ensemble average. See module docstring for file format.
            device: Inference device (``'cpu'`` or ``'cuda'``).
            n_jobs: XGBoost thread count (``None`` = all cores).
            pattern: Optional regex to filter model file paths.
            hparam_overrides: Optional hparam key/value pairs applied to every
                datamodule before it is instantiated. Useful for fixing stale
                paths stored in old checkpoints, e.g.::

                    EnsemblePredictor.from_directory(
                        'ensembles/dc50',
                        hparam_overrides={
                            'poi_embeddings_file': '/new/embeddings.npz',
                            'ligase_embeddings_file': '/new/embeddings.npz',
                        },
                    )
            lazy_loading: If ``True`` (default), models and datamodules are
                loaded one at a time during inference and immediately freed
                afterwards. This keeps peak memory proportional to a single
                model rather than the whole ensemble. Set to ``False`` to
                load everything eagerly upfront (old behaviour).

        Returns:
            Initialized :class:`EnsemblePredictor`.
        """
        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        device = 'cuda' if device == 'gpu' else device

        # --- Parse weights file -------------------------------------------
        weights = None
        wanted_stems: Optional[set] = None
        if weights_file is not None:
            weights_file = Path(weights_file)
            if not weights_file.exists():
                raise FileNotFoundError(f"Weights file not found: {weights_file}")
            with open(weights_file) as f:
                weights_doc = json.load(f)
            weights = weights_doc.get("weights", {})
            if not weights:
                raise ValueError(f"No 'weights' key found in {weights_file}")
            wanted_stems = set(weights)
            print(
                f"Loaded weights from {weights_file.name} "
                f"(task={weights_doc.get('task', 'unknown')}, "
                f"method={weights_doc.get('method', 'unknown')}, "
                f"{len(weights)} models)"
            )

        # --- Discover model files -----------------------------------------
        model_files: List[Path] = []
        for ext in ['*.ckpt', '*.json', '*.ubj', '*.pkl']:
            model_files.extend(model_dir.glob(f'**/{ext}'))

        if pattern:
            regex = re.compile(pattern)
            model_files = [f for f in model_files if regex.search(str(f))]

        # --- Validate weights against found files -------------------------
        if wanted_stems is not None:
            found_stems = {f.stem for f in model_files}
            missing = wanted_stems - found_stems
            if missing:
                raise ValueError(
                    f"{len(missing)} model(s) from weights file not found in {model_dir}: "
                    + ", ".join(sorted(missing))
                )
            model_files = [f for f in model_files if f.stem in wanted_stems]

        print(f"{'Registering' if lazy_loading else 'Loading'} {len(model_files)} model(s) from {model_dir}")

        # ------------------------------------------------------------------ #
        # LAZY PATH — store paths, infer metadata from filenames/extensions  #
        # ------------------------------------------------------------------ #
        if lazy_loading:
            model_tasks: Dict[str, str] = {}
            model_types: Dict[str, str] = {}
            model_paths: Dict[str, Path] = {}
            dm_paths: Dict[str, Tuple[Optional[Path], Path]] = {}

            for model_file in model_files:
                name = model_file.stem
                paths = cls._find_dm_paths(model_file)
                if paths is None:
                    print(f"  Warning: no datamodule found for {name} — skipping.")
                    continue
                mtype = (
                    'xgboost'
                    if model_file.suffix.lower() in ('.json', '.ubj', '.pkl')
                    else 'lightning'
                )
                model_paths[name] = model_file
                dm_paths[name] = paths
                model_tasks[name] = cls._infer_model_task(name, None)
                model_types[name] = mtype
                print(f"  Registered ({len(model_paths)}/{len(model_files)}): {name} ({mtype})")

            if not model_paths:
                raise ValueError(f"No models with matching datamodules found in {model_dir}")

            if weights is None:
                task_to_names: Dict[str, List[str]] = defaultdict(list)
                for name, task in model_tasks.items():
                    task_to_names[task].append(name)
                weights = {
                    name: 1.0 / len(names)
                    for names in task_to_names.values()
                    for name in names
                }
            else:
                weights = {k: v for k, v in weights.items() if k in model_paths}

            predictor = cls(models={}, datamodules={}, weights=weights, device=device, n_jobs=n_jobs)
            predictor._lazy = True
            predictor._model_paths = model_paths
            predictor._dm_paths = dm_paths
            predictor._hparam_overrides = hparam_overrides
            predictor.model_tasks = model_tasks
            predictor.model_types = model_types
            return predictor

        # ------------------------------------------------------------------ #
        # EAGER PATH — load everything upfront                               #
        # ------------------------------------------------------------------ #
        models: Dict[str, Any] = {}
        datamodules: Dict[str, Any] = {}

        for i, model_file in enumerate(model_files):
            name = model_file.stem
            try:
                model = cls._load_model(model_file, device, n_jobs)
                dm = cls._load_datamodule(model_file, hparam_overrides=hparam_overrides)
                models[name] = model
                datamodules[name] = dm
                print(f"  Loaded ({i+1}/{len(model_files)}): {name} ({cls._get_model_type(model)})")
            except Exception as e:
                print(f"  Failed to load {name}: {e}")

        if not models:
            raise ValueError(f"No models could be loaded from {model_dir}")

        if weights is None:
            task_to_names = defaultdict(list)
            for name in models:
                task = cls._infer_model_task(name, datamodules.get(name))
                task_to_names[task].append(name)
            weights = {
                name: 1.0 / len(names)
                for names in task_to_names.values()
                for name in names
            }
        else:
            weights = {k: v for k, v in weights.items() if k in models}

        return cls(models, datamodules, weights, device, n_jobs)

    # ------------------------------------------------------------------
    # Static loaders
    # ------------------------------------------------------------------

    @staticmethod
    def _find_dm_paths(
        model_path: Path,
    ) -> Optional[Tuple[Optional[Path], Path]]:
        """Return the first existing ``(hparams_path_or_None, state_path)`` pair
        for the datamodule paired with *model_path*, or ``None`` if nothing is found.
        """
        stem = model_path.stem
        parent = model_path.parent

        data_config = group = fold = None
        for part in stem.split('-'):
            if part.startswith('data='):
                data_config = part[5:]
            elif part.startswith('group='):
                group = part[6:]
            elif part.startswith('fold='):
                fold = part[5:]

        candidates: List[Tuple[Optional[Path], Path]] = []
        if data_config:
            dm_base = f"datamodule-data={data_config}-group={group}-fold={fold}"
            candidates.append((None, parent / f"{dm_base}_state.pt"))
            candidates.append((parent / f"{dm_base}_hparams.yaml", parent / f"{dm_base}_state.pt"))

        candidates += [
            (None, parent / f"{stem}_state.pt"),
            (None, parent / "datamodule_state.pt"),
            (parent / "datamodule_hparams.yaml", parent / "datamodule_state.pt"),
        ]

        for hparams_path, state_path in candidates:
            yaml_ok = hparams_path is None or hparams_path.exists()
            if yaml_ok and state_path.exists():
                return (hparams_path, state_path)
        return None

    @staticmethod
    def _load_datamodule(
        model_path: Path,
        hparam_overrides: Optional[Dict[str, Any]] = None,
    ) -> Optional[DegradationComplexDataModule]:
        """Load the datamodule paired with *model_path*.

        Args:
            model_path: Path to the model file whose sibling datamodule to load.
            hparam_overrides: Optional hparam overrides forwarded to
                :func:`load_datamodule` (e.g. to fix stale embedding paths).

        Returns:
            Loaded :class:`DegradationComplexDataModule`, or ``None`` if no
            matching file is found.
        """
        paths = EnsemblePredictor._find_dm_paths(model_path)
        if paths is None:
            return None
        hparams_path, state_path = paths
        try:
            return load_datamodule(
                state_dict_path=state_path,
                hparams_path=hparams_path,
                hparam_overrides=hparam_overrides,
            )
        except Exception as e:
            warnings.warn(f"Failed to load datamodule from {state_path}: {e}")
            return None

    @staticmethod
    def _load_model(
        model_path: Path,
        device: str,
        n_jobs: Optional[int] = None,
    ) -> Any:
        """Load a model from file, dispatching on extension."""
        suffix = model_path.suffix.lower()
        if suffix == '.ckpt':
            return EnsemblePredictor._load_lightning_model(model_path, device)
        elif suffix in ('.json', '.ubj', '.pkl'):
            return EnsemblePredictor._load_xgboost_model(model_path, n_jobs, device)
        raise ValueError(f"Unsupported model format: {suffix}")

    @staticmethod
    def _load_lightning_model(model_path: Path, device: str) -> Any:
        from tackai.models.tack_model import TACKModel  # lazy import: avoids loading PyTorch until needed
        try:
            model = TACKModel.load_from_checkpoint(str(model_path), map_location=device)
        except Exception as e:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            if 'state_dict' in checkpoint:
                raise NotImplementedError(f"Cannot load Lightning model from checkpoint: {e}")
            model = checkpoint
        model.to(device)
        model.eval()
        return model

    @staticmethod
    def _load_xgboost_model(
        model_path: Path,
        n_jobs: Optional[int],
        device: Literal['cpu', 'cuda'] = 'cpu',
    ) -> Any:
        suffix = model_path.suffix.lower()
        if suffix in ('.json', '.ubj'):
            model = xgb.Booster()
            model.load_model(str(model_path))
        else:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

        if isinstance(model, xgb.Booster):
            if n_jobs is not None:
                model.set_param('nthread', n_jobs)
            if device != 'cpu':
                model.set_param('device', device)
        elif hasattr(model, 'set_params'):
            kwargs = {}
            if n_jobs is not None:
                kwargs['n_jobs'] = n_jobs
            if device != 'cpu':
                kwargs['device'] = device
            if kwargs:
                model.set_params(**kwargs)

        return model

    # ------------------------------------------------------------------
    # Task / type helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_model_type(model: Any) -> str:
        cls_name = type(model).__name__
        if 'XGB' in cls_name or 'Booster' in cls_name:
            return 'xgboost'
        if hasattr(model, 'forward') and hasattr(model, 'eval'):
            return 'lightning'
        return 'unknown'

    @staticmethod
    def _infer_model_task(model_name: str, datamodule: Optional[Any] = None) -> str:
        """Infer the prediction task from the datamodule or model filename."""
        if datamodule is not None and hasattr(datamodule, 'labels') and datamodule.labels:
            label = datamodule.labels[0].lower()
            if 'binary' in label or 'activity' in label:
                return 'bin'
            if 'pdc50' in label or 'dc50' in label:
                return 'dc50'
            if 'dmax' in label:
                return 'dmax'

        m = re.search(r'model=\w+?_(\w+?)_protac', model_name, re.IGNORECASE)
        if m and m.group(1).lower() in KNOWN_TASKS:
            return m.group(1).lower()

        return 'dmax'

    @property
    def available_tasks(self) -> List[str]:
        return sorted(set(self.model_tasks.values()))

    # ------------------------------------------------------------------
    # Lazy-loading helpers
    # ------------------------------------------------------------------

    def _iter_model_names(
        self, allowed_tasks: Optional[set] = None
    ):
        """Yield model names from either the lazy path registry or eager dict."""
        names = list(self._model_paths if self._lazy else self.models)
        for name in names:
            if allowed_tasks and self.model_tasks.get(name) not in allowed_tasks:
                continue
            yield name

    def _get_model_and_dm(
        self, name: str
    ) -> Tuple[Any, 'DegradationComplexDataModule']:
        """Return (model, datamodule) — loading from disk in lazy mode."""
        if self._lazy:
            model = self._load_model(self._model_paths[name], self.device, self.n_jobs)
            hparams_path, state_path = self._dm_paths[name]
            dm = load_datamodule(state_path, hparams_path, self._hparam_overrides)
            return model, dm
        return self.models[name], self.datamodules[name]

    def _get_dm(self, name: str) -> 'DegradationComplexDataModule':
        """Return the datamodule only — avoids loading the model in lazy mode."""
        if self._lazy:
            hparams_path, state_path = self._dm_paths[name]
            return load_datamodule(state_path, hparams_path, self._hparam_overrides)
        return self.datamodules[name]

    def _free_if_lazy(self, *objs: Any) -> None:
        """Delete objects and collect garbage only when in lazy mode."""
        if not self._lazy:
            return
        for obj in objs:
            if obj is not None:
                del obj
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Input validation / defaults
    # ------------------------------------------------------------------

    def _validate_and_fill_defaults(
        self,
        sample_dict: Dict[str, Any],
        datamodule: Any,
        verbose: bool = True,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Validate inputs and fill defaults for optional fields.

        Args:
            sample_dict: Raw input dict.
            datamodule: Datamodule that determines required columns.
            verbose: Emit warnings for missing/defaulted fields.

        Returns:
            ``(filled_dict, missing_required)`` — filled dict with defaults
            applied, and list of required columns that were not provided.
        """
        dm = datamodule
        dm_col_names = [
            dm.smiles_col, dm.poi_col, dm.poi_sequence_col,
            dm.ligase_col, dm.ligase_sequence_col, dm.cell_line_col,
            dm.treatment_time_col, dm.assay_type_col,
        ]
        lower_to_dm = {c.lower(): c for c in dm_col_names if c}

        filled = {lower_to_dm.get(k.lower(), k): v for k, v in sample_dict.items()}

        required_cols = {dm.smiles_col, dm.poi_col, dm.poi_sequence_col,
                         dm.ligase_col, dm.ligase_sequence_col}

        col_checks = [
            (dm.smiles_col, lambda: True),
            (dm.poi_col, lambda: (
                dm.poi_features == "name"
                or getattr(dm, 'poi_embeddings_id_type', '') != 'sequence'
            )),
            (dm.poi_sequence_col, lambda: dm.poi_features in ("sequence", "precomputed")),
            (dm.ligase_col, lambda: dm.ligase_features == "name"),
            (dm.ligase_sequence_col, lambda: dm.ligase_features == "precomputed"),
            (dm.cell_line_col, lambda: dm.cell_features in ("description", "name")),
            (dm.treatment_time_col, lambda: getattr(dm, 'use_treatment_time', False)),
            (dm.assay_type_col, lambda: getattr(dm, 'use_assay_type_encoding', False)),
        ]

        missing_required = []
        default_warnings = []

        for dm_col, is_needed_fn in col_checks:
            if not is_needed_fn():
                continue
            current_val = filled.get(dm_col)
            is_missing = (
                (isinstance(current_val, str) and not current_val.strip())
                or (not isinstance(current_val, str) and pd.isna(current_val))
            )
            if not is_missing:
                continue
            if dm_col in required_cols:
                missing_required.append(dm_col)
            else:
                default_val = DEFAULT_VALUES.get(dm_col)
                if default_val is not None:
                    filled[dm_col] = default_val
                    short = str(default_val)[:30]
                    default_warnings.append(f"  - {dm_col}: using default '{short}'")

        if verbose:
            if missing_required:
                warnings.warn(
                    f"Required input(s) missing: {', '.join(missing_required)}. "
                    "Models that need these features will be skipped.",
                    UserWarning, stacklevel=3,
                )
            if default_warnings:
                print("Note: Some inputs were missing and filled with defaults:")
                for w in default_warnings[:5]:
                    print(w)
                if len(default_warnings) > 5:
                    print(f"  ... and {len(default_warnings) - 5} more")

        return filled, missing_required

    def _prepare_sample_dicts(
        self,
        samples: List[Union[SampleInput, Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Convert SampleInput / raw dicts to datamodule-compatible dicts.

        In lazy mode ``self.datamodules`` is empty so we fall back to the
        default TACK column names (identical to ``DegradationComplexDataModule``
        defaults), which ``_validate_and_fill_defaults`` normalises per-model.
        """
        if self.datamodules:
            ref_dm = next(iter(self.datamodules.values()))
            kwargs = dict(
                smiles_col=ref_dm.smiles_col,
                poi_col=ref_dm.poi_col,
                poi_sequence_col=ref_dm.poi_sequence_col,
                ligase_col=ref_dm.ligase_col,
                ligase_sequence_col=ref_dm.ligase_sequence_col,
                cell_line_col=ref_dm.cell_line_col,
                assay_type_col=ref_dm.assay_type_col,
                treatment_time_col=ref_dm.treatment_time_col,
            )
        else:
            kwargs = {}  # SampleInput.to_datamodule_dict() defaults match DM defaults

        return [
            s.to_datamodule_dict(**kwargs) if isinstance(s, SampleInput) else s
            for s in samples
        ]

    # ------------------------------------------------------------------
    # Featurization
    # ------------------------------------------------------------------

    def transform(
        self,
        samples: List[Dict[str, Any]],
        return_timings: bool = False,
    ) -> Union[Dict[str, Any], Tuple[Dict[str, Any], Dict[str, float]]]:
        """Featurize a batch of samples for every loaded model.

        Validates and fills defaults once per datamodule for the whole
        batch, then calls ``datamodule.transform`` (shared embedding
        lookups, one sklearn pass per model).

        Args:
            samples: List of sample dicts (column-name → value).
            return_timings: If True, return a ``(features, timings)`` pair.

        Returns:
            ``{model_name: features}`` where *features* is the format
            expected by the model type (xgb tuple or pt dict).
        """
        featurized: Dict[str, Any] = {}
        timings: Dict[str, list] = defaultdict(list)

        for name, datamodule in self.datamodules.items():
            filled_samples = []
            for i, sample_dict in enumerate(samples):
                t0 = time.time()
                filled, missing = self._validate_and_fill_defaults(
                    sample_dict, datamodule, verbose=(i == 0),
                )
                timings['fill_defaults'].append(time.time() - t0)
                if missing:
                    raise ValueError(
                        f"Sample {i} is missing required input(s) for model '{name}': "
                        f"{', '.join(missing)}."
                    )
                filled_samples.append(filled)

            ret = 'xgb' if self.model_types.get(name) == 'xgboost' else 'pt'
            t0 = time.time()
            featurized[name] = datamodule.transform(filled_samples, return_tensor=ret)
            timings['featurize_batch'].append(time.time() - t0)

        if return_timings:
            return featurized, {k: float(np.mean(v)) for k, v in timings.items()}
        return featurized

    # ------------------------------------------------------------------
    # Per-model inference (private)
    # ------------------------------------------------------------------

    def _predict_xgboost_batch(self, model: Any, feat_list: Any) -> np.ndarray:
        """Run XGBoost prediction on pre-featurized batch output.

        *feat_list* may be a ``(ndarray, feature_names)`` tuple (fast path
        from ``transform``) or a list of per-sample tuples.
        """
        def _dmatrix_from_array(X: np.ndarray, feature_names: Optional[List[str]]) -> xgb.DMatrix:
            model_fnames = getattr(model, 'feature_names', None)
            if feature_names is not None and model_fnames is not None and set(feature_names) == set(model_fnames):
                df = pd.DataFrame(X, columns=list(feature_names))[model_fnames]
                return xgb.DMatrix(df)
            if feature_names is not None and len(feature_names) == X.shape[1]:
                return xgb.DMatrix(pd.DataFrame(X, columns=feature_names))
            return xgb.DMatrix(X)

        if isinstance(model, xgb.Booster):
            if self.n_jobs is not None:
                model.set_param('nthread', self.n_jobs)
            if self.device != 'cpu':
                model.set_param('device', self.device)

        # Fast path: single (matrix, names) tuple
        if (
            isinstance(feat_list, tuple) and len(feat_list) == 2
            and isinstance(feat_list[0], np.ndarray) and feat_list[0].ndim == 2
        ):
            X, feature_names = feat_list
            return model.predict(_dmatrix_from_array(X, list(feature_names)))

        # Per-sample list path
        arrays, feature_names = [], None
        for feat in feat_list:
            if feat is None:
                continue
            if isinstance(feat, tuple) and len(feat) == 2:
                arr, fnames = feat
                if feature_names is None:
                    feature_names = fnames
                arrays.append(arr.reshape(1, -1) if arr.ndim == 1 else arr)
            else:
                arr = np.array(feat)
                arrays.append(arr.reshape(1, -1) if arr.ndim == 1 else arr)

        if not arrays:
            return np.full(len(feat_list), np.nan)

        return model.predict(_dmatrix_from_array(np.vstack(arrays), feature_names))

    def _predict_lightning_batch(self, model: Any, feat_list: Any) -> np.ndarray:
        """Run a single batched forward pass for a Lightning/MLP model."""
        def _extract_output(output, n):
            if isinstance(output, torch.Tensor):
                return output.cpu().numpy().reshape(n, -1)
            if isinstance(output, dict):
                for key in ['prediction', 'pred', 'output', 'logits']:
                    if key in output:
                        return output[key].cpu().numpy().reshape(n, -1)
                for v in output.values():
                    if isinstance(v, torch.Tensor):
                        return v.cpu().numpy().reshape(n, -1)
            return np.array(output).reshape(n, -1)

        # Fast path: feat_list is already a batched Dict[str, Tensor]
        if isinstance(feat_list, dict):
            n = next(iter(feat_list.values())).shape[0]
            model.eval()
            with torch.no_grad():
                output = model.predict_step({k: v.to(self.device) for k, v in feat_list.items()})
            return _extract_output(output, n)

        # Legacy path: list of per-sample feature dicts
        n = len(feat_list)
        valid_indices = [i for i, f in enumerate(feat_list) if f is not None]
        valid_feats = [feat_list[i] for i in valid_indices]

        if not valid_feats:
            return np.full((n, 1), np.nan, dtype=np.float64)

        model.eval()
        with torch.no_grad():
            batch = {}
            for k in valid_feats[0].keys():
                tensors = []
                for feat in valid_feats:
                    v = feat[k]
                    if isinstance(v, np.ndarray):
                        t = torch.from_numpy(v).float()
                    elif isinstance(v, torch.Tensor):
                        t = v.float()
                    else:
                        t = torch.tensor(np.array(v), dtype=torch.float32)
                    tensors.append(t)
                batch[k] = torch.stack(tensors, dim=0).to(self.device)
            output = model.predict_step(batch)

        preds_valid = _extract_output(output, len(valid_feats))
        result = np.full((n, preds_valid.shape[1]), np.nan, dtype=np.float64)
        for out_idx, orig_idx in enumerate(valid_indices):
            result[orig_idx] = preds_valid[out_idx]
        return result

    def _denormalize(self, prediction: np.ndarray, datamodule: Any) -> np.ndarray:
        """Inverse-transform normalized predictions to the original scale.

        A no-op if the datamodule does not normalize labels.

        Args:
            prediction: Model output array.
            datamodule: Datamodule with fitted label transformers.

        Returns:
            Denormalized array.
        """
        dm = datamodule
        if not (getattr(dm, 'normalize_labels', False) or getattr(dm, 'standardize_labels', False)):
            return prediction

        task_key = dm.labels if hasattr(dm, 'labels') else None
        if isinstance(task_key, list) and len(task_key) == 1:
            task_key = task_key[0]
        else:
            warnings.warn(
                "Cannot denormalize: datamodule has multiple or no labels; "
                "returning raw prediction.",
                UserWarning, stacklevel=2,
            )
            return prediction

        try:
            return dm.inverse_transform_labels(prediction, task_key)
        except (ValueError, KeyError) as e:
            raise ValueError(
                f"Failed to denormalize prediction for task '{task_key}': {e}"
            ) from e

    def _infer_and_denormalize(
        self,
        model_name: str,
        feat_list: Any,
        model: Any = None,
        dm: Any = None,
    ) -> np.ndarray:
        """Run inference for one model and denormalize predictions.

        *model* and *dm* may be passed explicitly (lazy path) to avoid
        re-looking them up from ``self.models``/``self.datamodules``.

        Returns an ndarray of shape ``(n_samples,)`` in the original scale.
        """
        if model is None:
            model = self.models[model_name]
        if dm is None:
            dm = self.datamodules[model_name]

        if self.model_types[model_name] == 'xgboost':
            preds = self._predict_xgboost_batch(model, feat_list)
        else:
            preds = self._predict_lightning_batch(model, feat_list)

        preds_flat = preds.reshape(-1, 1) if preds.ndim == 1 else preds
        denorm = np.empty_like(preds_flat, dtype=float)
        for j in range(preds_flat.shape[0]):
            denorm[j] = self._denormalize(preds_flat[j], dm)
        return denorm.flatten() if preds.ndim == 1 else denorm

    # ------------------------------------------------------------------
    # Ensemble assembly (private)
    # ------------------------------------------------------------------

    def _build_ensemble_prediction(
        self,
        preds_by_model: Dict[str, np.ndarray],
        task_name: str,
        return_individual: bool = True,
    ) -> EnsemblePrediction:
        """Build an EnsemblePrediction from per-model denormalized predictions."""
        weighted_sum = np.zeros_like(next(iter(preds_by_model.values())), dtype=float)
        weight_sum = 0.0
        for model_name, pred in preds_by_model.items():
            w = self.weights.get(model_name, 0.0)
            weighted_sum += pred * w
            weight_sum += w
        weighted_mean = weighted_sum / weight_sum if weight_sum > 0 else weighted_sum

        all_preds = np.array(list(preds_by_model.values()))
        uncertainty_std = np.std(all_preds, axis=0)

        predictive_entropy = None
        if task_name == 'bin':
            avg = np.clip(weighted_mean, 1e-7, 1 - 1e-7)
            predictive_entropy = -(avg * np.log(avg) + (1 - avg) * np.log(1 - avg))

        return EnsemblePrediction(
            weighted_mean=weighted_mean,
            uncertainty_std=uncertainty_std,
            individual_predictions=preds_by_model if return_individual else {},
            weights={k: v for k, v in self.weights.items() if k in preds_by_model},
            model_names=list(preds_by_model.keys()),
            task=task_name,
            label_name=self.TASK_LABELS.get(task_name, 'Unknown'),
            predictive_entropy=predictive_entropy,
        )

    def _assemble_batch_results(
        self,
        model_batch_preds: Dict[str, np.ndarray],
        n_samples: int,
        return_individual: bool = True,
    ) -> List[Dict[str, EnsemblePrediction]]:
        """Build per-sample EnsemblePrediction dicts from model-level batch arrays."""
        results = []
        for i in range(n_samples):
            task_preds: Dict[str, Dict[str, np.ndarray]] = {}
            for model_name, batch_pred in model_batch_preds.items():
                pred_i = batch_pred[i] if batch_pred.ndim > 1 else np.array([batch_pred[i]])
                if np.any(np.isnan(pred_i)):
                    raise ValueError(
                        f"Prediction for model '{model_name}' is NaN for sample {i}."
                    )
                task = self.model_tasks.get(model_name)
                task_preds.setdefault(task, {})[model_name] = np.atleast_1d(pred_i)

            if not task_preds:
                raise ValueError(f"No valid predictions for sample {i}.")

            results.append({
                task_name: self._build_ensemble_prediction(preds, task_name, return_individual)
                for task_name, preds in task_preds.items()
            })
        return results

    # ------------------------------------------------------------------
    # Public prediction API
    # ------------------------------------------------------------------

    def predict(
        self,
        samples: Union[
            SampleInput, Dict[str, Any],
            List[Union[SampleInput, Dict[str, Any]]],
            str,
            List[str],
        ],
        context: Optional[PreprocessedContext] = None,
        return_individual: bool = True,
        tasks: Optional[List[str]] = None,
        verbose: bool = False,
        return_timings: bool = False,
    ) -> Union[
        Dict[str, EnsemblePrediction],
        List[Dict[str, EnsemblePrediction]],
    ]:
        """Make ensemble predictions for one or many samples.

        Each model's raw output is always denormalized using its own
        datamodule before the weighted ensemble average is computed.

        When *context* is provided (a :class:`PreprocessedContext` from
        :meth:`transform_context`), *samples* should be a SMILES string or a
        list of SMILES strings. Context features (protein/cell embeddings,
        categorical encodings) are reused as-is; only the SMILES-dependent
        features are recomputed per call. This is the fast path for screening
        many compounds against the same biological context.

        Args:
            samples: Single ``SampleInput``/dict or a list of them for the
                normal path. A single SMILES string or list of SMILES strings
                when *context* is given.
            context: Pre-encoded context from :meth:`transform_context`.
                When supplied, *samples* must be SMILES string(s).
            return_individual: Include per-model predictions in the result.
            tasks: Optional subset of tasks to predict (``None`` = all).
            verbose: Show a progress bar over models.
            return_timings: Return ``(result, timings)`` instead of just the result.

        Returns:
            If *samples* is a single item: ``Dict[task, EnsemblePrediction]``.
            If *samples* is a list: ``List[Dict[task, EnsemblePrediction]]``.
            When *return_timings* is ``True``, a ``(result, timings)`` tuple.
        """
        if context is not None:
            return self._predict_with_context(
                samples, context,
                return_individual=return_individual,
                tasks=tasks,
                verbose=verbose,
                return_timings=return_timings,
            )

        single = not isinstance(samples, list)
        if single:
            samples = [samples]

        n = len(samples)
        if n == 0:
            return ([], {}) if return_timings else []

        if tasks:
            available = set(self.model_tasks.values())
            if not set(tasks).intersection(available):
                raise ValueError(
                    f"No models for requested tasks: {set(tasks)}. Available: {available}."
                )

        timings: Dict[str, float] = {}

        # 1. Normalise inputs
        t0 = time.time()
        sample_dicts = self._prepare_sample_dicts(samples)
        timings['prepare'] = time.time() - t0

        # 2+3. Per-model: (lazy) load → featurize → infer → denorm → (lazy) free
        allowed_tasks = set(tasks) if tasks else None
        model_batch_preds: Dict[str, np.ndarray] = {}
        t_feat = t_infer = 0.0

        for model_name in tqdm(
            list(self._iter_model_names(allowed_tasks)), desc="Predicting", disable=not verbose
        ):
            model, dm = self._get_model_and_dm(model_name)
            try:
                filled_samples = []
                for i, sd in enumerate(sample_dicts):
                    filled, missing = self._validate_and_fill_defaults(sd, dm, verbose=(i == 0))
                    if missing:
                        raise ValueError(
                            f"Sample {i} is missing required input(s) for model '{model_name}': "
                            f"{', '.join(missing)}."
                        )
                    filled_samples.append(filled)

                ret = 'xgb' if self.model_types[model_name] == 'xgboost' else 'pt'
                t0 = time.time()
                feat_list = dm.transform(filled_samples, return_tensor=ret)
                t_feat += time.time() - t0

                t0 = time.time()
                try:
                    model_batch_preds[model_name] = self._infer_and_denormalize(
                        model_name, feat_list, model=model, dm=dm
                    )
                except Exception as e:
                    msg = str(e)
                    if 'feature_names mismatch' in msg:
                        msg = "feature_names mismatch (training/inference encoding incompatible)"
                    raise ValueError(f"Prediction failed for model '{model_name}': {msg}") from e
                t_infer += time.time() - t0
            finally:
                self._free_if_lazy(model, dm)

        timings.update(featurize=t_feat, inference=t_infer)

        # 4. Assemble per-sample results
        t0 = time.time()
        results = self._assemble_batch_results(model_batch_preds, n, return_individual)
        timings['assemble'] = time.time() - t0

        out = results[0] if single else results
        return (out, timings) if return_timings else out

    def _predict_with_context(
        self,
        smiles: Union[str, List[str]],
        context: PreprocessedContext,
        return_individual: bool = True,
        tasks: Optional[List[str]] = None,
        verbose: bool = False,
        return_timings: bool = False,
    ) -> Union[
        List[Dict[str, EnsemblePrediction]],
        Tuple[List[Dict[str, EnsemblePrediction]], Dict[str, float]],
    ]:
        """Fast screening path: score SMILES against a pre-encoded context."""
        if context.predictor_id != id(self):
            raise ValueError(
                "PreprocessedContext was not produced by this predictor instance. "
                "Call transform_context on the same EnsemblePredictor you use to score."
            )

        single = isinstance(smiles, str)
        smiles_list: List[str] = [smiles] if single else smiles
        n = len(smiles_list)
        if n == 0:
            return ([], {}) if return_timings else []

        allowed_tasks = set(tasks) if tasks else None
        if allowed_tasks:
            available = set(self.model_tasks.values())
            if not allowed_tasks.intersection(available):
                raise ValueError(
                    f"No models for requested tasks: {allowed_tasks}. Available: {available}."
                )

        timings: Dict[str, float] = {}
        model_batch_preds: Dict[str, np.ndarray] = {}
        t_smiles = t_infer = 0.0

        for model_name in tqdm(
            list(self._iter_model_names(allowed_tasks)), desc="Predicting", disable=not verbose
        ):
            model, dm = self._get_model_and_dm(model_name)
            try:
                ret = 'xgb' if self.model_types[model_name] == 'xgboost' else 'pt'

                t0 = time.time()
                feat_arg = dm.transform(
                    smiles_list, context=context.context_features[model_name], return_tensor=ret
                )
                t_smiles += time.time() - t0

                t0 = time.time()
                try:
                    model_batch_preds[model_name] = self._infer_and_denormalize(
                        model_name, feat_arg, model=model, dm=dm
                    )
                except Exception as e:
                    msg = str(e)
                    if 'feature_names mismatch' in msg:
                        msg = "feature_names mismatch (training/inference encoding incompatible)"
                    raise ValueError(f"Prediction failed for model '{model_name}': {msg}") from e
                t_infer += time.time() - t0
            finally:
                self._free_if_lazy(model, dm)

        timings.update(smiles_featurize=t_smiles, inference=t_infer)

        t0 = time.time()
        results = self._assemble_batch_results(model_batch_preds, n, return_individual)
        timings['assemble'] = time.time() - t0

        out = results[0] if single else results
        return (out, timings) if return_timings else out

    # ------------------------------------------------------------------
    # Context / SMILES split API — fast path for screening many SMILES
    # ------------------------------------------------------------------

    def transform_context(
        self,
        context: Union[SampleInput, Dict[str, Any]],
        verbose: bool = False,
    ) -> PreprocessedContext:
        """Encode the non-SMILES (context) features once for every model.

        Call once per biological context (POI / ligase / cell line / assay /
        treatment time), then pass the result to :meth:`predict` via the
        ``context`` argument alongside a list of SMILES strings to score.

        Args:
            context: ``SampleInput`` or dict with context columns.
                SMILES is ignored.
            verbose: Print progress per model.

        Returns:
            :class:`PreprocessedContext` ready for :meth:`predict`.
        """
        if isinstance(context, SampleInput):
            context_dict = self._prepare_sample_dicts([context])[0]
        else:
            context_dict = dict(context)

        context_features: Dict[str, Dict[str, np.ndarray]] = {}
        xgb_layouts: Dict[str, List[Tuple[str, int, int]]] = {}
        xgb_total_dims: Dict[str, int] = {}
        xgb_row_template: Dict[str, np.ndarray] = {}
        xgb_feature_names: Dict[str, List[str]] = {}
        source_context: Dict[str, Any] = {}

        for model_name in self._iter_model_names():
            dm = self._get_dm(model_name)
            try:
                if getattr(dm, 'use_tokenizer', False):
                    raise ValueError(
                        f"Model '{model_name}' uses a tokenizer; use predict() directly instead of "
                        "transform_context + predict(context=...)."
                    )

                filled, missing = self._validate_and_fill_defaults(
                    context_dict, dm, verbose=False,
                )
                missing = [c for c in missing if c != dm.smiles_col]
                if missing:
                    raise ValueError(
                        f"Model '{model_name}' is missing required context field(s): "
                        f"{', '.join(missing)}."
                    )
                if verbose:
                    print(f"  Preprocessed context for model '{model_name}'")

                # Use dm.transform() — the same engine as predict — so that the
                # same numeric_pipeline (properly pickled) handles treatment time.
                # Supply a dummy SMILES so the molecular sub-pipeline runs
                # without errors; those features are stripped immediately after.
                dummy = dict(filled)
                dummy[dm.smiles_col] = dummy.get(dm.smiles_col) or 'C'
                all_feats = dm.transform([dummy], return_tensor='dict')  # {key: (1, dim)}
                ctx_feats = {k: v[0] for k, v in all_feats.items()
                             if not k.startswith(SMILES_FEATURE_PREFIXES)}
                context_features[model_name] = ctx_feats

                if self.model_types[model_name] == 'xgboost':
                    layout = dm.get_feature_layout()
                    total_dim = sum(width for _, _, width in layout)
                    template = np.zeros((1, total_dim), dtype=np.float32)

                    for key, offset, width in layout:
                        if key.startswith(SMILES_FEATURE_PREFIXES):
                            continue
                        if key not in ctx_feats:
                            raise ValueError(
                                f"Model '{model_name}' expects context feature '{key}' "
                                "in its XGBoost layout, but the datamodule did not produce it."
                            )
                        val = np.asarray(ctx_feats[key], dtype=np.float32).flatten()
                        if val.shape[0] != width:
                            raise ValueError(
                                f"Context feature '{key}' for model '{model_name}' has width "
                                f"{val.shape[0]} but layout expects {width}."
                            )
                        template[0, offset:offset + width] = val

                    xgb_layouts[model_name] = layout
                    xgb_total_dims[model_name] = total_dim
                    xgb_row_template[model_name] = template
                    xgb_feature_names[model_name] = dm.get_xgboost_feature_names()

                if not source_context:
                    source_context = dict(filled)
            finally:
                self._free_if_lazy(dm)

        return PreprocessedContext(
            context_features=context_features,
            xgb_layouts=xgb_layouts,
            xgb_total_dims=xgb_total_dims,
            xgb_row_template=xgb_row_template,
            xgb_feature_names=xgb_feature_names,
            predictor_id=id(self),
            source_context=source_context,
        )

    # ------------------------------------------------------------------
    # DataFrame helper
    # ------------------------------------------------------------------

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
        """Make predictions for every row in a DataFrame.

        Args:
            df: Input DataFrame with at least a SMILES column.
            smiles_col: Name of the SMILES column.
            poi_col: POI name column (optional).
            poi_sequence_col: POI sequence column (optional).
            ligase_col: Ligase name column (optional).
            cell_line_col: Cell line column (optional).
            treatment_time_col: Treatment time column (optional).

        Returns:
            Input DataFrame with appended prediction columns.
        """
        samples = [
            SampleInput(
                smiles=row.get(smiles_col) if smiles_col in row.index else None,
                poi_name=row.get(poi_col) if poi_col and poi_col in row.index else None,
                poi_sequence=row.get(poi_sequence_col) if poi_sequence_col and poi_sequence_col in row.index else None,
                ligase_name=row.get(ligase_col) if ligase_col and ligase_col in row.index else None,
                cell_line=row.get(cell_line_col) if cell_line_col and cell_line_col in row.index else None,
                treatment_time=row.get(treatment_time_col) if treatment_time_col and treatment_time_col in row.index else None,
            )
            for _, row in df.iterrows()
        ]

        batch_results = self.predict(samples, return_individual=True)

        rows = []
        for task_results in batch_results:
            row_dict: Dict[str, Any] = {}
            if task_results is None:
                row_dict['error'] = 'prediction failed'
            else:
                for task_name, result in task_results.items():
                    suffix = f"_{task_name}" if len(task_results) > 1 else ""
                    row_dict[f'prediction{suffix}'] = result.weighted_mean[0] if result.weighted_mean is not None else np.nan
                    row_dict[f'uncertainty{suffix}'] = result.uncertainty_std[0] if result.uncertainty_std is not None else np.nan
                    row_dict[f'ci_pctl_lower{suffix}'] = result.ci_percentile_lower_95[0] if result.ci_percentile_lower_95 is not None else np.nan
                    row_dict[f'ci_pctl_upper{suffix}'] = result.ci_percentile_upper_95[0] if result.ci_percentile_upper_95 is not None else np.nan
                    row_dict[f'ci_sem_lower{suffix}'] = result.ci_sem_lower_95[0] if result.ci_sem_lower_95 is not None else np.nan
                    row_dict[f'ci_sem_upper{suffix}'] = result.ci_sem_upper_95[0] if result.ci_sem_upper_95 is not None else np.nan
            rows.append(row_dict)

        return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def get_categorical_choices(self) -> Dict[str, List[str]]:
        """Collect known category values from all datamodule ordinal encoders."""
        merged: Dict[str, set] = {}
        for dm in self.datamodules.values():
            if dm is None:
                continue
            pipeline = getattr(dm, 'category_pipeline', None)
            if pipeline is None:
                continue
            for _, transformer, cols in pipeline.transformers_:
                ordinal = transformer.named_steps.get('ordinal') if hasattr(transformer, 'named_steps') else None
                if ordinal is None or not hasattr(ordinal, 'categories_'):
                    continue
                for col, cats in zip(cols, ordinal.categories_):
                    merged.setdefault(col, set()).update(c for c in cats if c is not None)
        return {col: sorted(vals) for col, vals in merged.items()}

    def get_model_info(self) -> Dict[str, Any]:
        names = list(self._model_paths if self._lazy else self.models)
        return {
            'n_models': len(names),
            'model_names': names,
            'model_types': self.model_types,
            'model_tasks': self.model_tasks,
            'weights': self.weights,
            'available_tasks': self.available_tasks,
            'device': self.device,
            'categorical_choices': {} if self._lazy else self.get_categorical_choices(),
        }

    def __repr__(self) -> str:
        n = len(self._model_paths if self._lazy else self.models)
        mode = 'lazy' if self._lazy else 'eager'
        return (
            f"EnsemblePredictor(n_models={n}, "
            f"tasks={self.available_tasks}, device='{self.device}', mode='{mode}')"
        )
