"""DataModule for PROTAC degradation activity prediction (DC50/Dmax)."""
import os
import logging
import pickle
import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Union, Any, Dict, Literal, Tuple

# xgboost must be imported before torch to avoid an OpenMP runtime conflict on
# macOS: torch's libtorch sets up Intel OpenMP (libiomp5), and if xgb.Booster()
# is first called afterwards it initialises a second OpenMP runtime → SIGSEGV.
import xgboost as xgb
import torch
import sklearn.compose._column_transformer as _sklearn_ct
if not hasattr(_sklearn_ct, '_RemainderColsList'):
    # Compatibility shim: _RemainderColsList was removed after sklearn 1.6.x
    # but may appear in pickled ColumnTransformer objects from older checkpoints.
    class _RemainderColsList(list):
        def __init__(self, columns, future_dtype=None):
            super().__init__(columns)
            self.future_dtype = future_dtype
    _sklearn_ct._RemainderColsList = _RemainderColsList
import pandas as pd
from tqdm import tqdm
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
from scipy.stats import gaussian_kde
from scipy.special import expit, logit
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
    QuantileTransformer,
    FunctionTransformer,
)
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.decomposition import PCA
from transformers import AutoTokenizer

from tackai.data.embeddings.protein_embeddings import ProteinEmbedding
from tackai.data.embeddings.cell_embeddings import CellEmbedding
from tackai.data.embeddings.mol_embeddings import MolEmbedding
from tackai.config import load_config_from_yaml


# Processing kinds for a feature, used to decide what can be reused across CV
# folds / ensemble members and what must be recomputed per training fold.
STATELESS = "stateless"  # deterministic per input; disk-cached once and shared
FITTED = "fitted"        # depends on the training fold; refit on every fold

# Feature groups, used to split featurization into a SMILES-dependent
# ("molecular") pass and a SMILES-independent ("context") pass. A fixed
# biological context (POI, E3 ligase, cell line, assay, time) can be
# featurized once and reused across many molecules; see
# ``transform_context`` / ``transform``.
MOLECULAR = "molecular"
CONTEXT = "context"

# Declarative map: datamodule feature flag -> processing metadata. Mirrors the
# fit/transform split in ``setup()``. STATELESS features are produced by a pure
# ``transform`` whose result only depends on the input value (SMILES, sequence,
# cell line) and is cached on disk in ``TACKAI_CACHE``; they are computed once
# and reused for every fold and every ensemble member. FITTED features rely on a
# sklearn estimator (encoder, scaler, PCA, count-vectorizer) that is fit on the
# training fold and therefore must be refit per fold.
#
# Notes:
# - ``embedder_attr`` is the datamodule attribute holding the embedder whose
#   on-disk cache file documents where the stateless result is stored.
# - Raw precomputed protein embeddings are STATELESS; the optional PCA applied
#   on top of them (``use_poi_pca`` / ``use_ligase_pca``) is a separate FITTED
#   step recorded independently.
# - Raw RDKit descriptors are STATELESS (cached), but they are subsequently
#   passed through the FITTED numeric pipeline (a cheap scaler); the expensive
#   part — descriptor computation — is the compute-once concern captured here.
FEATURE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "fingerprint": {
        "name": "fingerprint", "kind": STATELESS, "token": "FP", "group": MOLECULAR,
        "embedder_attr": "fp_embedder", "feature_keys": ["Feature_Fingerprint"],
    },
    "descriptors": {
        "name": "descriptors", "kind": STATELESS, "token": "Mol-Desc", "group": MOLECULAR,
        "embedder_attr": "desc_embedder", "feature_keys": ["Feature_Descriptor_"],
    },
    "poi_precomputed": {
        "name": "poi_precomputed", "kind": STATELESS, "token": "POI-ESM", "group": CONTEXT,
        "embedder_attr": "poi_precomputed_embedding",
        "feature_keys": ["Feature_POI_Precomputed_Embedding"],
    },
    "ligase_precomputed": {
        "name": "ligase_precomputed", "kind": STATELESS, "token": "E3-ESM", "group": CONTEXT,
        "embedder_attr": "ligase_precomputed_embedding",
        "feature_keys": ["Feature_Ligase_Precomputed_Embedding"],
    },
    "cell_description": {
        "name": "cell_description", "kind": STATELESS, "token": "Cell-Text", "group": CONTEXT,
        "embedder_attr": "cell_description_embedding",
        "feature_keys": ["Feature_{cell_line_col}_Description"],
    },
    "poi_sequence": {
        "name": "poi_sequence", "kind": FITTED, "token": "POI-Vec", "group": CONTEXT,
        "embedder_attr": "poi_sequence_embedding",
        "feature_keys": ["Feature_{poi_sequence_col}"],
    },
    "poi_name": {
        "name": "poi_name", "kind": FITTED, "token": "POI-Cat", "group": CONTEXT,
        "embedder_attr": None, "feature_keys": ["Feature_{poi_col}"],
    },
    "ligase_name": {
        "name": "ligase_name", "kind": FITTED, "token": "E3-Cat", "group": CONTEXT,
        "embedder_attr": None, "feature_keys": ["Feature_{ligase_col}"],
    },
    "cell_name": {
        "name": "cell_name", "kind": FITTED, "token": "Cell-Cat", "group": CONTEXT,
        "embedder_attr": None, "feature_keys": ["Feature_{cell_line_col}"],
    },
    "assay_type": {
        "name": "assay_type", "kind": FITTED, "token": "Assay", "group": CONTEXT,
        "embedder_attr": None, "feature_keys": ["Feature_{assay_type_col}"],
    },
    "treatment_time": {
        "name": "treatment_time", "kind": FITTED, "token": "Time", "group": CONTEXT,
        "embedder_attr": None, "feature_keys": ["Feature_{treatment_time_col}"],
    },
    "poi_pca": {
        "name": "poi_pca", "kind": FITTED, "token": "POI-PCA", "group": CONTEXT,
        "embedder_attr": None, "feature_keys": ["Feature_POI_Precomputed_Embedding"],
    },
    "ligase_pca": {
        "name": "ligase_pca", "kind": FITTED, "token": "E3-PCA", "group": CONTEXT,
        "embedder_attr": None, "feature_keys": ["Feature_Ligase_Precomputed_Embedding"],
    },
}


def _is_molecular_feature(key: str) -> bool:
    """Return True if *key* corresponds to a SMILES-dependent (molecular) feature."""
    return key == 'Feature_Fingerprint' or key.startswith('Feature_Descriptor_')


class TorchListDataset(torch.utils.data.Dataset):
    """Minimal PyTorch Dataset wrapping a list of pre-featurized dicts.

    Replaces the HuggingFace Dataset returned by ``featurize_dataset`` so that
    sklearn pipelines and other non-picklable objects are never captured in a
    multiprocessing map closure.
    """

    def __init__(self, data: List[Dict[str, Any]]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: Union[int, str]) -> Any:
        if isinstance(idx, str):
            return [item[idx] for item in self.data]
        return self.data[idx]

    @property
    def column_names(self) -> List[str]:
        return list(self.data[0].keys()) if self.data else []

    @property
    def num_rows(self) -> int:
        return len(self.data)

    def filter(self, condition_fn) -> 'TorchListDataset':
        return TorchListDataset([item for item in self.data if condition_fn(item)])

    def select(self, indices) -> 'TorchListDataset':
        return TorchListDataset([self.data[i] for i in indices])


class DegradationComplexDataModule(pl.LightningDataModule):
    """Lightning DataModule for the TACK PROTAC dataset.

    Handles data loading, featurization (Morgan fingerprints, ESM protein
    embeddings, RDKit descriptors, categorical/numeric sklearn pipelines),
    label normalization, and DataLoader construction. Supports XGBoost,
    MLP, and BERT model types.
    """
    
    def __init__(
        self,
        dataset: Union[str, DatasetDict] = "ailab-bio/TACK",
        # Column names
        smiles_col: str = "SMILES",
        ligase_col: str = "Ligase_Name",
        ligase_sequence_col: str = "Ligase_Sequence",
        poi_col: str = "POI_Name",
        poi_sequence_col: str = "POI_Sequence",
        cell_line_col: str = "Cell_Line_ID",
        assay_type_col: str = "Assay",
        treatment_time_col: str = "Assay_Time",
        treatment_time_dmax_col: str = "Assay_Time",
        treatment_time_dc50_col: str = "Assay_Time",
        treatment_time_ic50_col: str = "Treatment Time (h) (Cellular activities, IC50)",
        labels: Optional[List[str]] = None,
        normalize_labels: bool = False,
        standardize_labels: bool = False,
        impute_labels: bool = False,
        # Molecular feature specification
        fp_size: int = 512,
        radius: int = 16,
        mol_features: Optional[str] = None,  # "fingerprint", "descriptors", "fingerprint+descriptors"
        use_relevant_descriptors: bool = False,
        selected_descriptors: Optional[List[str]] = None,
        # Protein/cell feature specifications
        poi_features: Optional[str] = None,    # "precomputed", "sequence", "name"
        ligase_features: Optional[str] = None, # "precomputed", "name"
        # Precomputed protein embeddings parameters
        poi_embeddings_file: Optional[Union[Path, str]] = None,
        poi_embeddings_format: Literal["npz", "h5"] = "npz",
        poi_embeddings_per_residue: bool = True,
        poi_residue_pooling: Optional[Literal["mean", "sum", "max", "cls", "mean_sqrt_len"]] = "sum",
        poi_embeddings_id_type: Literal["sequence", "uniprot"] = "sequence",
        ligase_embeddings_file: Optional[Union[Path, str]] = None,
        ligase_embeddings_format: Literal["npz", "h5"] = "npz",
        ligase_embeddings_per_residue: bool = True,
        ligase_residue_pooling: Optional[Literal["mean", "sum", "max", "cls", "mean_sqrt_len"]] = "sum",
        ligase_embeddings_id_type: Literal["sequence", "uniprot"] = "sequence",
        # PCA parameters for precomputed embeddings
        use_poi_pca: bool = False,
        poi_pca_n_components: Optional[int] = None,
        use_ligase_pca: bool = False,
        ligase_pca_n_components: Optional[int] = None,
        cell_features: Optional[str] = None,   # "description", "name"
        # Tokenizer parameters (for BERT-based models)
        use_tokenizer: bool = False,
        tokenizer_name: str = "google-bert/bert-base-cased",
        max_length: int = 512,
        prompt_template: Optional[str] = None,
        label_task_col: str = "Value_Type",
        degrader_type_col: Optional[str] = None,
        default_degrader_type: str = "PROTAC",
        include_prompt: bool = False,  # Whether to include 'Prompt' in returned features, mostly used for debugging
        is_bert_multitask: bool = False,
        # Assay-related flags
        use_assay_type_encoding: bool = False,
        use_treatment_time: bool = False,
        # DataLoader params and multiprocessing
        batch_size: int = 32,
        num_workers: int = 0,
        num_proc: int = 1,
        hf_token: Optional[str] = None,
        verbose: int = 0,
        features_order: Literal["sorted", "mol_first"] = "mol_first",
        sort_features: Optional[bool] = None,
        categorical_encoding: Literal['minmax', 'onehot', 'embedding'] = 'minmax',
        # --- Deprecated boolean flags (pre-refactor API) ---
        use_fingerprints: Optional[bool] = None,
        use_descriptors: Optional[bool] = None,
        use_poi_sequence_embedding: Optional[bool] = None,
        use_poi_name_embedding: Optional[bool] = None,
        use_poi_precomputed_embedding: Optional[bool] = None,
        use_ligase_name_embedding: Optional[bool] = None,
        use_ligase_precomputed_embedding: Optional[bool] = None,
        use_cell_description_embedding: Optional[bool] = None,
        use_cell_name_embedding: Optional[bool] = None,
    ):
        """Initialize the datamodule.

        Args:
            dataset: HuggingFace repo string (downloaded in setup()) or a pre-built DatasetDict.
            smiles_col: Column name for SMILES strings.
            ligase_col: Column name for E3 ligase name.
            ligase_sequence_col: Column name for E3 ligase amino acid sequence.
            poi_col: Column name for POI (target protein) name.
            poi_sequence_col: Column name for POI amino acid sequence.
            cell_line_col: Column name for cell line identifier.
            assay_type_col: Column name for assay type.
            treatment_time_col: Default column for treatment time (hours).
            treatment_time_dmax_col: Treatment time column for Dmax labels.
            treatment_time_dc50_col: Treatment time column for DC50 labels.
            treatment_time_ic50_col: Treatment time column for IC50 labels.
            labels: Label column names. Defaults to the four standard TACK columns.
            normalize_labels: Apply quantile normalization to labels.
            standardize_labels: Apply standard scaling to labels.
            impute_labels: Keep NaN-label samples in the dataset (no filtering).
            fp_size: Morgan fingerprint bit size.
            radius: Morgan fingerprint radius.
            mol_features: Molecular feature(s) to include. One of ``"fingerprint"``,
                ``"descriptors"``, or ``"fingerprint+descriptors"`` to combine both.
                ``None`` disables all molecular features (requires ``use_tokenizer``).
            use_relevant_descriptors: Use the curated relevant-descriptor subset.
            selected_descriptors: Explicit list of descriptor names to include.
            poi_features: POI encoding — ``"precomputed"`` (ESM embeddings),
                ``"sequence"`` (amino-acid count TF-IDF), or ``"name"`` (ordinal/one-hot).
                ``None`` omits POI features.
            ligase_features: Ligase encoding — ``"precomputed"`` (ESM embeddings) or
                ``"name"`` (ordinal/one-hot). ``None`` omits ligase features.
            poi_embeddings_file: Path to the POI embedding archive (.npz or .h5).
            poi_embeddings_format: Format of the POI embedding archive.
            poi_embeddings_per_residue: Whether the archive stores per-residue embeddings.
            poi_residue_pooling: Pooling strategy for per-residue POI embeddings.
            poi_embeddings_id_type: Whether to look up embeddings by 'sequence' or 'uniprot'.
            ligase_embeddings_file: Path to the ligase embedding archive.
            ligase_embeddings_format: Format of the ligase embedding archive.
            ligase_embeddings_per_residue: Whether the archive stores per-residue embeddings.
            ligase_residue_pooling: Pooling strategy for per-residue ligase embeddings.
            ligase_embeddings_id_type: Whether to look up embeddings by 'sequence' or 'uniprot'.
            use_poi_pca: Reduce POI precomputed embeddings with PCA (fitted on training data).
            poi_pca_n_components: Number of PCA components for POI embeddings.
            use_ligase_pca: Reduce ligase precomputed embeddings with PCA.
            ligase_pca_n_components: Number of PCA components for ligase embeddings.
            cell_features: Cell line encoding — ``"description"`` (sentence-transformer
                embeddings) or ``"name"`` (ordinal/one-hot). ``None`` omits cell features.
            use_tokenizer: Tokenize a text prompt (BERT-based models).
            tokenizer_name: HuggingFace tokenizer identifier.
            max_length: Maximum token sequence length.
            prompt_template: Optional format string for custom prompt assembly.
            label_task_col: Column holding the task type for BERT multi-task mode.
            degrader_type_col: Column holding the degrader type (e.g. PROTAC, molecular glue).
            default_degrader_type: Fallback degrader type when the column is absent.
            include_prompt: Include the assembled prompt string in the feature dict (debug).
            is_bert_multitask: Enable BERT multi-task mode (single value column + Value_Type).
            use_assay_type_encoding: Include assay type as a categorical feature.
            use_treatment_time: Include treatment time as a scaled numeric feature.
            batch_size: DataLoader batch size.
            num_workers: DataLoader worker count.
            num_proc: Ignored (kept for API compatibility with older call sites).
            hf_token: HuggingFace token for private dataset access.
            verbose: Logging verbosity (0=ERROR, 1=INFO, 2+=DEBUG).
            features_order: Feature ordering strategy. ``'mol_first'`` (default) places
                molecular features (fingerprints / descriptors) before all other
                features (sorted alphabetically within each group). ``'sorted'`` sorts
                all feature keys alphabetically.
            sort_features: Deprecated. Pass ``features_order='sorted'`` instead.
                Will be fixed at ``'mol_first'`` in a future release.
            categorical_encoding: How to encode categorical features —
                'minmax' (ordinal + MinMax), 'onehot', or 'embedding' (ordinal index for nn.Embedding).
        """
        # --- Backward-compatibility: translate legacy boolean flags --------
        _LEGACY_PARAMS = [
            'use_fingerprints', 'use_descriptors',
            'use_poi_sequence_embedding', 'use_poi_name_embedding',
            'use_poi_precomputed_embedding', 'use_ligase_name_embedding',
            'use_ligase_precomputed_embedding',
            'use_cell_description_embedding', 'use_cell_name_embedding',
        ]
        _legacy_used = [
            p for p in _LEGACY_PARAMS if locals()[p] is not None
        ]
        if _legacy_used:
            warnings.warn(
                f"DegradationComplexDataModule received deprecated parameter(s): "
                f"{', '.join(_legacy_used)}. "
                "Replace use_fingerprints/use_descriptors with mol_features, "
                "use_poi_*/use_ligase_*/use_cell_* booleans with poi_features, "
                "ligase_features, and cell_features respectively.",
                DeprecationWarning,
                stacklevel=2,
            )
            # mol_features
            if mol_features is None:
                if use_fingerprints and use_descriptors:
                    mol_features = 'fingerprint+descriptors'
                elif use_fingerprints:
                    mol_features = 'fingerprint'
                elif use_descriptors:
                    mol_features = 'descriptors'
            # poi_features
            if poi_features is None:
                if use_poi_precomputed_embedding:
                    poi_features = 'precomputed'
                elif use_poi_sequence_embedding:
                    poi_features = 'sequence'
                elif use_poi_name_embedding:
                    poi_features = 'name'
            # ligase_features
            if ligase_features is None:
                if use_ligase_precomputed_embedding:
                    ligase_features = 'precomputed'
                elif use_ligase_name_embedding:
                    ligase_features = 'name'
            # cell_features
            if cell_features is None:
                if use_cell_description_embedding:
                    cell_features = 'description'
                elif use_cell_name_embedding:
                    cell_features = 'name'
        # -------------------------------------------------------------------

        super().__init__()
        # Exclude dataset/hf_token (not serializable) and deprecated legacy
        # flags (already translated above) from the saved hyperparameters.
        self.save_hyperparameters(ignore=['dataset', 'hf_token'] + _LEGACY_PARAMS)
        
        # Column names
        self.smiles_col = smiles_col
        self.ligase_col = ligase_col
        self.ligase_sequence_col = ligase_sequence_col
        self.poi_col = poi_col
        self.poi_sequence_col = poi_sequence_col
        self.cell_line_col = cell_line_col
        self.assay_type_col = assay_type_col
        self.treatment_time_col = treatment_time_col
        self.treatment_time_dmax_col = treatment_time_dmax_col
        self.treatment_time_dc50_col = treatment_time_dc50_col
        self.treatment_time_ic50_col = treatment_time_ic50_col
        self.label_task_col = label_task_col
        self.degrader_type_col = degrader_type_col
        
        # Feature specifications
        self.mol_features = mol_features
        self.poi_features = poi_features
        self.ligase_features = ligase_features
        self.cell_features = cell_features
        self.use_treatment_time = use_treatment_time
        self.include_prompt = include_prompt
        self.default_degrader_type = default_degrader_type
        self.normalize_labels = normalize_labels
        self.standardize_labels = standardize_labels
        self.impute_labels = impute_labels
        self.use_assay_type_encoding = use_assay_type_encoding
        self.categorical_encoding = categorical_encoding

        # Store ID type for embeddings lookup
        self.poi_embeddings_id_type = poi_embeddings_id_type
        self.ligase_embeddings_id_type = ligase_embeddings_id_type
        
        # PCA settings
        self.use_poi_pca = use_poi_pca
        self.poi_pca_n_components = poi_pca_n_components
        self.use_ligase_pca = use_ligase_pca
        self.ligase_pca_n_components = ligase_pca_n_components
        
        # Initialize PCA transformers
        self.poi_pca = PCA(n_components=poi_pca_n_components, random_state=42) if use_poi_pca else None
        self.ligase_pca = PCA(n_components=ligase_pca_n_components, random_state=42) if use_ligase_pca else None

        if sort_features is not None:
            warnings.warn(
                "sort_features is deprecated and will be removed in a future release. "
                "The ordering will be fixed at 'mol_first'. "
                "Use features_order='sorted' to preserve alphabetical ordering.",
                DeprecationWarning,
                stacklevel=2,
            )
            if sort_features:
                features_order = "sorted"
        self.features_order = features_order

        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        if verbose == 0:
            self.logger.setLevel(logging.ERROR)
        elif verbose == 1:
            self.logger.setLevel(logging.INFO)
        elif verbose >= 2:
            self.logger.setLevel(logging.DEBUG)
        
        # Tokenizer settings
        self.use_tokenizer = use_tokenizer
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.prompt_template = prompt_template
        self.tokenizer = None  # Will be initialized in setup if needed
        self.is_bert_multitask = is_bert_multitask
        
        # Dictionary mapping feature names to their dimensions
        self.feature_dims = {}
        
        # Create embedders based on flags
        self.fp_embedder = MolEmbedding(
            embeddings_type="fingerprint",
            radius=radius,
            fp_size=fp_size,
            load_from_cache=False,
            filename=f"morgan_fp_radius{radius}_size{fp_size}.npz",
        ) if self.mol_features and "fingerprint" in self.mol_features else None

        # RDKit descriptors embedder
        filename = "rdkit_descriptors.npz"
        if use_relevant_descriptors:
            filename = "rdkit_descriptors_relevant.npz"
        elif selected_descriptors is not None:
            selected_str = "_".join(sorted(selected_descriptors))
            filename = f"rdkit_descriptors_selected_{selected_str}.npz"
        self.desc_embedder = MolEmbedding(
            embeddings_type="rdkit_descriptors",
            use_relevant_descriptors=use_relevant_descriptors,
            selected_descriptors=selected_descriptors,
            load_from_cache=True,
            filename=filename,
        ) if self.mol_features and "descriptors" in self.mol_features else None

        # POI sequence embedding (amino acid count)
        self.poi_sequence_embedding = ProteinEmbedding(
            embeddings_type="amino_acid_count",
            load_from_cache=False,
            filename="protein_embeddings_amino_acid_count.npz",
        ) if self.poi_features == "sequence" else None

        # POI precomputed embedding
        self.poi_precomputed_embedding = ProteinEmbedding(
            embeddings_type="precomputed",
            embeddings_file=poi_embeddings_file,
            embeddings_format=poi_embeddings_format,
            embeddings_per_residue=poi_embeddings_per_residue,
            residue_pooling=poi_residue_pooling,
            load_from_cache=False,
        ) if self.poi_features == "precomputed" else None

        # Ligase precomputed embedding
        self.ligase_precomputed_embedding = ProteinEmbedding(
            embeddings_type="precomputed",
            embeddings_file=ligase_embeddings_file,
            embeddings_format=ligase_embeddings_format,
            embeddings_per_residue=ligase_embeddings_per_residue,
            residue_pooling=ligase_residue_pooling,
            load_from_cache=False,
        ) if self.ligase_features == "precomputed" else None

        # Cell line description embedding (sentence transformer)
        self.cell_description_embedding = CellEmbedding(
            embeddings_type="sentence_transformer",
            pooling="sum",
            load_from_cache=True,
            filename="cell_embeddings_model=sentence-transformer_pooling=sum.npz",
        ) if self.cell_features == "description" else None
        
        # Labels-related column names
        default_labels = [
            "Dmax (%) (DC50/Dmax)",
            "pDC50 (DC50/Dmax)",
            "pIC50 (Cellular activities, IC50)",
            "pIC50 (Protac to Target, IC50)",
        ]
        self.labels = labels or default_labels
        
        # Check if the `labels` columns are in the dataset
        if isinstance(dataset, DatasetDict):
            for label in self.labels:
                if label not in dataset['train'].column_names:
                    raise ValueError(f"Label column '{label}' not found in dataset.")

        self.fp_size = fp_size
        self.radius = radius

        # DataLoader and multiprocessing params
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_proc = num_proc
        
        if not use_tokenizer and mol_features is None:
            raise ValueError("At least one of mol_features or use_tokenizer must be set.")
        
        self.dataset = dataset

        if isinstance(dataset, str):
            self.hf_token = os.getenv("HF_TOKEN") if hf_token is None else hf_token
            self.dataset_name = dataset
        else:
            self.dataset_name = "custom_dataset"

        # TODO: Handle species?
        # self.ordinal_encoders['POI Species'] = OrdinalEncoder(**encoder_args)
        # self.ordinal_encoders['Cell Species'] = OrdinalEncoder(**encoder_args)
 
        # Initialize category and numeric pipelines
        self._create_category_pipeline()
        self._create_numeric_pipelines()

        # Initialize label transformers
        self.label_transformers = {}

        # For BERT-like models, we need to track tasks (Dmax, DC50, etc. in
        # Value_Type column) to get the correct normalizations and predictions
        self.train_tasks = None
        self.val_tasks = None
        self.test_tasks = None

        # Track whether encoders and transformers have been fitted
        self._encoders_fitted = False

    def prepare_data(self):
        """No-op: dataset download happens lazily in setup() to avoid multi-GPU race conditions."""
        pass

    def setup(self, stage: Optional[str] = None):
        """Fit encoders on the training split and featurize all requested splits.

        Encoder fitting (sklearn pipelines, PCA, label transformers) only runs
        when ``stage`` is ``'train'`` or ``None`` and training data is present.
        If encoders were already fitted (e.g. loaded from a checkpoint), fitting
        is skipped.

        Args:
            stage: Lightning stage hint — ``'train'``, ``'validation'``, ``'test'``,
                or ``None`` to process all available splits.
        """
        if isinstance(self.dataset, str):
            # Load dataset from Hugging Face hub
            self.logger.info(f"Loading dataset '{self.dataset_name}' from Hugging Face hub...")
            warn = (f"WARNING: Loading dataset '{self.dataset_name}' from the hub in the setup method. "
                "In distributed training, this may lead to multiple downloads. "
                "Consider passing a DatasetDict object directly to avoid this.")
            self.logger.warning(warn)
            print(warn)
            self.dataset = load_dataset(self.dataset_name, token=self.hf_token)
        
        # Initialize tokenizer if using text mode
        if self.use_tokenizer and self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.logger.debug(f"Initialized tokenizer: {self.tokenizer_name}")
        
        # Fit encoders only if they haven't been fitted yet (e.g., not loaded from checkpoint)
        # and we have training data and we're in training stage
        if not self._encoders_fitted and "train" in self.dataset and (stage == "train" or stage is None):
            # Fit encoders on training data only
            train_df = self.dataset["train"].to_pandas()

            # Fit POI sequence embedding
            if self.poi_features == "sequence" and self.poi_sequence_embedding is not None:
                poi_seqs = list(set(self.dataset["train"][self.poi_sequence_col]))
                self.logger.debug(f"Fitting POI sequence encoder on {len(poi_seqs)} sequences: {poi_seqs[:5]}...")
                self.poi_sequence_embedding.fit(poi_seqs)
                self.logger.debug("POI sequence encoder fitted on training data")
                
            # Collect SMILES for molecular embeddings
            smiles_list = list(set(self.dataset["train"][self.smiles_col]))

            # Fit molecular embeddings
            if self.mol_features and "fingerprint" in self.mol_features:
                self.logger.debug("Fitting fingerprint embedder on training data...")
                self.fp_embedder.transform(smiles_list, update_cache=True)
                self.logger.debug("Fingerprint embedder fitted on training data")
            
            # Collect cell lines for embeddings
            cell_lines = list(self.dataset["train"][self.cell_line_col])
            if "validation" in self.dataset:
                cell_lines += list(self.dataset["validation"][self.cell_line_col])
            if "test" in self.dataset:
                cell_lines += list(self.dataset["test"][self.cell_line_col])
            cell_lines = list(set(cell_lines))
            
            # Fit cell description embedding
            if self.cell_features == "description" and self.cell_description_embedding is not None:
                self.logger.debug(f"Fitting cell description embedding on {len(cell_lines)} cell lines: {cell_lines[:5]}...")
                is_cell_emb_empty = not bool(self.cell_description_embedding.embeddings)
                self.cell_description_embedding.transform(cell_lines, update_cache=is_cell_emb_empty)
                self.logger.debug("Cell description embedding fitted on training data")
            
            # Fit categorical and numeric pipelines
            self._fit_category_pipeline(train_df)
            self._fit_numeric_pipelines(train_df)

            # Fit PCA on precomputed embeddings if requested
            self._fit_pca_on_protein_embeddings(train_df)

            # Fit label transformers if normalizing labels
            if self.normalize_labels or self.standardize_labels:
                self._fit_label_transformers(train_df)

            self._encoders_fitted = True
            self.logger.info("All encoders and transformers have been fitted on training data")
        elif self._encoders_fitted:
            self.logger.info("Encoders already fitted (loaded from checkpoint or previous setup)")
        else:
            self.logger.warning("Skipping encoder fitting: either not in training stage or no training data available")

        # Featurize all individual splits
        self.logger.debug("Featurizing datasets...")
        if (stage == "train" or stage is None) and "train" in self.dataset:
            self.train_dataset = self.featurize_dataset(self.dataset["train"], "train", self.num_proc)
            
            # TODO: Including operators as noise is still a work in progress
            # self.train_dataset = self.dataset["train"].map(
            #     lambda x: self.augment_sample(x, rel_std=0.9, is_log_scale=True, ignore_null_operators=False),
            #     self.num_proc,
            # )
            # self.train_dataset = self.featurize_dataset(self.train_dataset, "train", self.num_proc)
            
            self.logger.debug(f"Training dataset featurization complete: {len(self.train_dataset)} samples.")
            
            # Initialize feature dimensions from first training sample
            if len(self.train_dataset) > 0:
                self._initialize_feature_dims(self.train_dataset[0])

        if (stage == "validation" or stage is None or stage == "test") and "validation" in self.dataset:
            self.val_dataset = self.featurize_dataset(self.dataset["validation"], "validation", self.num_proc)

        if (stage == "test" or stage is None) and "test" in self.dataset:
            self.test_dataset = self.featurize_dataset(self.dataset["test"], "test", self.num_proc)

    def _transform_batch(
        self,
        examples: List[Dict],
        return_tensor: Literal['np', 'pt', 'xgb'] = 'np',
    ) -> List[Dict[str, Any]]:
        """Batch-encode multiple full samples efficiently (internal engine).

        Instead of building a 1-row DataFrame per sample for the sklearn
        pipelines, this method:
        1. Computes raw embeddings once per unique input value within the batch.
        2. Runs the category and numeric sklearn pipelines on **all samples
           at once** (one DataFrame, one ``.transform()`` call each).
        3. Applies PCA in batch (one ``pca.transform(matrix)`` call).

        Args:
            examples: List of sample dicts (column-name → value).
            return_tensor: Output format — ``'np'`` returns a list of per-sample
                feature dicts; ``'pt'`` returns a single batched
                ``Dict[str, Tensor]``; ``'xgb'`` returns a list of
                ``(flat_1D_array, feature_names)`` tuples.

        Returns:
            See *return_tensor* description above.
        """
        n = len(examples)
        if n == 0:
            return []

        # ------------------------------------------------------------------
        # 1. Raw embeddings — compute once per unique value
        # ------------------------------------------------------------------
        fp_results: Dict[str, np.ndarray] = {}
        cell_text_results: Dict[str, np.ndarray] = {}
        poi_vec_results: Dict[str, np.ndarray] = {}
        poi_precomp_results: Dict[str, np.ndarray] = {}
        ligase_precomp_results: Dict[str, np.ndarray] = {}
        smiles_vals = [ex[self.smiles_col] for ex in examples]

        # Fingerprints
        if self.mol_features and "fingerprint" in self.mol_features and self.fp_embedder is not None:
            unique_smiles = list(dict.fromkeys(smiles_vals))
            fp_results = self.fp_embedder.transform(unique_smiles)

        # Cell description embedding
        if self.cell_features == "description" and self.cell_description_embedding is not None:
            unique_cells = list(dict.fromkeys(ex[self.cell_line_col] for ex in examples))
            cell_text_results = self.cell_description_embedding.transform(unique_cells)

        # POI sequence embedding (amino acid count / tfidf)
        if self.poi_features == "sequence" and self.poi_sequence_embedding is not None:
            unique_seqs = list(dict.fromkeys(ex[self.poi_sequence_col] for ex in examples))
            poi_vec_results = self.poi_sequence_embedding.transform(unique_seqs)

        # POI precomputed embedding (raw, before PCA)
        if self.poi_features == "precomputed" and self.poi_precomputed_embedding is not None:
            unique_poi_ids = list(dict.fromkeys(self._get_protein_id(ex, 'poi') for ex in examples))
            poi_precomp_results = self.poi_precomputed_embedding.transform(unique_poi_ids)

        # Ligase precomputed embedding (raw, before PCA)
        if self.ligase_features == "precomputed" and self.ligase_precomputed_embedding is not None:
            unique_lig_ids = list(dict.fromkeys(self._get_protein_id(ex, 'ligase') for ex in examples))
            ligase_precomp_results = self.ligase_precomputed_embedding.transform(unique_lig_ids)

        # ------------------------------------------------------------------
        # 2. Batch PCA — transform all unique embeddings at once
        # ------------------------------------------------------------------
        poi_pca_results: Dict[str, np.ndarray] = {}
        if self.use_poi_pca and self.poi_pca is not None and poi_precomp_results:
            ids_list = list(poi_precomp_results.keys())
            matrix = np.stack([poi_precomp_results[pid] for pid in ids_list])
            pca_out = self.poi_pca.transform(matrix)
            for pid, vec in zip(ids_list, pca_out):
                poi_pca_results[pid] = vec

        ligase_pca_results: Dict[str, np.ndarray] = {}
        if self.use_ligase_pca and self.ligase_pca is not None and ligase_precomp_results:
            ids_list = list(ligase_precomp_results.keys())
            matrix = np.stack([ligase_precomp_results[lid] for lid in ids_list])
            pca_out = self.ligase_pca.transform(matrix)
            for lid, vec in zip(ids_list, pca_out):
                ligase_pca_results[lid] = vec

        # ------------------------------------------------------------------
        # 3. Batch category pipeline
        # ------------------------------------------------------------------
        cat_results: Optional[np.ndarray] = None
        if self.category_pipeline is not None and self.categorical_cols:
            cat_df = pd.DataFrame({
                col: [ex[col] for ex in examples]
                for col in self.categorical_cols
            })
            cat_results = self.category_pipeline.transform(cat_df)  # shape (n, n_cat_features)

        # ------------------------------------------------------------------
        # 4. Batch numeric pipeline
        # ------------------------------------------------------------------
        num_results: Optional[np.ndarray] = None
        if self.numeric_pipeline is not None:
            num_data: Dict[str, Any] = {}
            desc_names: List[str] = []
            desc_matrix: Optional[np.ndarray] = None

            if self.use_treatment_time:
                num_data[self.treatment_time_col] = [ex[self.treatment_time_col] for ex in examples]

            if self.mol_features and "descriptors" in self.mol_features:
                desc_names = self.desc_embedder.get_descriptor_names()
                unique_smiles = list(dict.fromkeys(smiles_vals))
                desc_all = self.desc_embedder.transform(unique_smiles)
                desc_matrix = np.empty((n, len(desc_names)), dtype=np.float32)
                for i, smi in enumerate(smiles_vals):
                    desc_matrix[i] = desc_all[smi]

                for j, name in enumerate(desc_names):
                    num_data[f'Descriptor_{name}'] = desc_matrix[:, j]

            # Fast path: transform treatment-time and descriptor columns
            # directly with cached sklearn parameters, avoiding expensive
            # ColumnTransformer overhead with many 1-column transformers.
            used_fast_path = False
            if self.mol_features and "descriptors" in self.mol_features and desc_matrix is not None and hasattr(self.numeric_pipeline, 'transformers_'):
                try:
                    desc_col_to_idx = {
                        f'Descriptor_{name}': idx for idx, name in enumerate(desc_names)
                    }
                    desc_means = np.zeros(len(desc_names), dtype=np.float64)
                    desc_scales = np.ones(len(desc_names), dtype=np.float64)
                    found_desc_scalers = 0
                    treatment_block = None

                    for trans_name, transformer, cols in self.numeric_pipeline.transformers_:
                        if trans_name == 'remainder' or not cols:
                            continue

                        col = cols[0]
                        if col == self.treatment_time_col:
                            tt_values = np.asarray(num_data[self.treatment_time_col], dtype=np.float64).reshape(-1, 1)
                            treatment_block = transformer.transform(tt_values).astype(np.float32, copy=False)
                            continue

                        desc_idx = desc_col_to_idx.get(col)
                        if desc_idx is None or not isinstance(transformer, StandardScaler):
                            raise TypeError("Unexpected numeric transformer layout")

                        mean = float(np.ravel(transformer.mean_)[0]) if hasattr(transformer, 'mean_') else 0.0
                        scale = float(np.ravel(transformer.scale_)[0]) if hasattr(transformer, 'scale_') else 1.0
                        if scale == 0.0:
                            scale = 1.0

                        desc_means[desc_idx] = mean
                        desc_scales[desc_idx] = scale
                        found_desc_scalers += 1

                    if found_desc_scalers == len(desc_names):
                        desc_scaled = (desc_matrix.astype(np.float64) - desc_means) / desc_scales
                        desc_scaled = desc_scaled.astype(np.float32, copy=False)

                        blocks = []
                        if self.use_treatment_time:
                            if treatment_block is None:
                                raise RuntimeError("Missing treatment-time transformer in numeric pipeline")
                            blocks.append(treatment_block)
                        blocks.append(desc_scaled)

                        num_results = np.hstack(blocks) if len(blocks) > 1 else blocks[0]
                        used_fast_path = True
                except Exception:
                    used_fast_path = False

            if not used_fast_path:
                # Fallback to sklearn implementation for full compatibility.
                num_df = pd.DataFrame(num_data)
                num_results = self.numeric_pipeline.transform(num_df)  # shape (n, n_num_features)

        # Precompute per-column output widths for the category pipeline so
        # that the sample loop can slice cat_results correctly (one-hot
        # encoding produces multiple output columns per input column).
        cat_col_offsets: List[tuple] = []  # [(offset, width), ...]
        if cat_results is not None:
            cached_offsets = getattr(self, '_batch_cat_col_offsets', None)
            if cached_offsets is not None and len(cached_offsets) == len(self.categorical_cols):
                cat_col_offsets = cached_offsets
            else:
                offset = 0
                for trans_name, inner, cols in self.category_pipeline.transformers_:
                    if trans_name == 'remainder' or not cols:
                        continue
                    width = inner.transform(cat_df[cols].iloc[:1]).shape[1]
                    cat_col_offsets.append((offset, width))
                    offset += width
                self._batch_cat_col_offsets = cat_col_offsets

        # ------------------------------------------------------------------
        # 5. Assemble results
        # ------------------------------------------------------------------

        # Fast path for Lightning models: build one batched tensor dict directly
        # from the already-computed matrices, skipping per-sample loop and
        # re-stacking.
        if return_tensor == 'pt':
            batch_out: Dict[str, torch.Tensor] = {}

            if self.mol_features and "fingerprint" in self.mol_features and self.fp_embedder is not None:
                fp_mat = np.stack([fp_results[ex[self.smiles_col]] for ex in examples])
                batch_out['Feature_Fingerprint'] = torch.from_numpy(fp_mat).float()

            if self.cell_features == "description" and self.cell_description_embedding is not None:
                cell_mat = np.stack([cell_text_results[ex[self.cell_line_col]] for ex in examples])
                batch_out[f'Feature_{self.cell_line_col}_Description'] = torch.from_numpy(cell_mat).float()

            if self.poi_features == "sequence" and self.poi_sequence_embedding is not None:
                poi_mat = np.stack([poi_vec_results[ex[self.poi_sequence_col]] for ex in examples])
                batch_out[f'Feature_{self.poi_sequence_col}'] = torch.from_numpy(poi_mat).float()

            if self.poi_features == "precomputed" and self.poi_precomputed_embedding is not None:
                emb_res = poi_pca_results if (self.use_poi_pca and self.poi_pca is not None and poi_pca_results) else poi_precomp_results
                poi_mat = np.stack([emb_res[self._get_protein_id(ex, 'poi')] for ex in examples])
                batch_out['Feature_POI_Precomputed_Embedding'] = torch.from_numpy(poi_mat).float()

            if self.ligase_features == "precomputed" and self.ligase_precomputed_embedding is not None:
                emb_res = ligase_pca_results if (self.use_ligase_pca and self.ligase_pca is not None and ligase_pca_results) else ligase_precomp_results
                lig_mat = np.stack([emb_res[self._get_protein_id(ex, 'ligase')] for ex in examples])
                batch_out['Feature_Ligase_Precomputed_Embedding'] = torch.from_numpy(lig_mat).float()

            if cat_results is not None:
                for j, col in enumerate(self.categorical_cols):
                    col_offset, col_width = cat_col_offsets[j]
                    cat_slice = cat_results[:, col_offset:col_offset + col_width].astype(np.float32)
                    if self.categorical_encoding == 'embedding':
                        cat_slice = cat_slice + 1
                    batch_out[f'Feature_{col}'] = torch.from_numpy(cat_slice)

            if num_results is not None:
                for j, col in enumerate(self.numerical_cols):
                    batch_out[f'Feature_{col}'] = torch.from_numpy(
                        num_results[:, j:j+1].astype(np.float32)
                    )

            if self.use_tokenizer:
                all_ids, all_masks, all_ttids = [], [], []
                for ex in examples:
                    tok = self._tokenize_sample(ex)
                    all_ids.append(tok['input_ids'])
                    all_masks.append(tok['attention_mask'])
                    if 'token_type_ids' in tok:
                        all_ttids.append(tok['token_type_ids'])
                batch_out['input_ids'] = torch.from_numpy(np.stack(all_ids))
                batch_out['attention_mask'] = torch.from_numpy(np.stack(all_masks))
                if all_ttids:
                    batch_out['token_type_ids'] = torch.from_numpy(np.stack(all_ttids))

            return batch_out

        # ------------------------------------------------------------------
        # Per-sample assembly for 'np', 'pt', and 'xgb' modes
        # ------------------------------------------------------------------
        batch_features: List[Dict[str, Any]] = []
        
        xgb_feature_names = None
        if return_tensor == 'xgb' and self.feature_dims:
            xgb_feature_names = self.get_xgboost_feature_names()

        for i, ex in enumerate(examples):
            features: Dict[str, Any] = {}

            if self.mol_features and "fingerprint" in self.mol_features and self.fp_embedder is not None:
                features['Feature_Fingerprint'] = fp_results[ex[self.smiles_col]]

            if self.cell_features == "description" and self.cell_description_embedding is not None:
                features[f'Feature_{self.cell_line_col}_Description'] = cell_text_results[ex[self.cell_line_col]]

            if self.poi_features == "sequence" and self.poi_sequence_embedding is not None:
                features[f'Feature_{self.poi_sequence_col}'] = poi_vec_results[ex[self.poi_sequence_col]]

            if self.poi_features == "precomputed" and self.poi_precomputed_embedding is not None:
                pid = self._get_protein_id(ex, 'poi')
                if self.use_poi_pca and self.poi_pca is not None:
                    features['Feature_POI_Precomputed_Embedding'] = poi_pca_results[pid]
                else:
                    features['Feature_POI_Precomputed_Embedding'] = poi_precomp_results[pid]

            if self.ligase_features == "precomputed" and self.ligase_precomputed_embedding is not None:
                lid = self._get_protein_id(ex, 'ligase')
                if self.use_ligase_pca and self.ligase_pca is not None:
                    features['Feature_Ligase_Precomputed_Embedding'] = ligase_pca_results[lid]
                else:
                    features['Feature_Ligase_Precomputed_Embedding'] = ligase_precomp_results[lid]

            # Category pipeline — slice from batch result using precomputed
            # offsets (handles one-hot expansion correctly)
            if cat_results is not None:
                for j, col in enumerate(self.categorical_cols):
                    col_offset, col_width = cat_col_offsets[j]
                    val = cat_results[i, col_offset:col_offset + col_width]
                    if self.categorical_encoding == 'embedding':
                        val = val + 1  # shift: unknown → 0, known → 1..N
                    features[f'Feature_{col}'] = val

            # Numeric pipeline — slice from batch result
            if num_results is not None:
                for j, col in enumerate(self.numerical_cols):
                    features[f'Feature_{col}'] = num_results[i, j:j+1]

            # Tokenizer
            if self.use_tokenizer:
                tokenized = self._tokenize_sample(ex)
                if not self.include_prompt and 'Prompt' in tokenized:
                    del tokenized['Prompt']
                features.update(tokenized)

            # ----------------------------------------------------------
            # Convert to requested tensor format
            # ----------------------------------------------------------
            if return_tensor == 'np':
                for key, value in features.items():
                    if isinstance(value, torch.Tensor):
                        features[key] = value.numpy()
                    elif not isinstance(value, np.ndarray):
                        features[key] = np.array(value)
            elif return_tensor == 'xgb':
                # Use feature_dims order if available (matches training), else fall back to sorted
                if self.feature_dims:
                    feature_order = [k for k in self.feature_dims if k in features]
                else:
                    feature_order = sorted(features.keys())

                feature_list = []
                for key in feature_order:
                    value = features[key]
                    if isinstance(value, torch.Tensor):
                        value = value.numpy()
                    feature_list.append(value.flatten())
                if not self.feature_dims:
                    feature_names = [
                        f"{key}_{j}"
                        for key in feature_order
                        for j in range(features[key].flatten().shape[0])
                    ]
                else:
                    feature_names = xgb_feature_names
                features = (
                    np.concatenate(feature_list).astype(np.float32),
                    feature_names,
                )

            batch_features.append(features)

        return batch_features

    def _transform_molecular_batch(
        self,
        smiles_list: List[str],
    ) -> List[Dict[str, np.ndarray]]:
        """Batch-compute molecular (SMILES-dependent) features (internal engine).

        Encodes Morgan fingerprints and/or scaled RDKit descriptors for a
        list of SMILES strings.

        Args:
            smiles_list: List of SMILES strings (duplicates allowed).

        Returns:
            List of dicts mapping ``Feature_*`` keys to numpy arrays, one
            entry per input SMILES (molecular features only).
        """
        n = len(smiles_list)
        if n == 0:
            return []

        fp_results: Dict[str, np.ndarray] = {}
        if self.mol_features and "fingerprint" in self.mol_features and self.fp_embedder is not None:
            unique_smiles = list(dict.fromkeys(smiles_list))
            fp_results = self.fp_embedder.transform(unique_smiles)

        mol_num_results: Optional[np.ndarray] = None
        if self.mol_features and "descriptors" in self.mol_features and self.desc_embedder is not None and self.mol_numeric_pipeline is not None:
            desc_names = self.desc_embedder.get_descriptor_names()
            unique_smiles = list(dict.fromkeys(smiles_list))
            desc_all = self.desc_embedder.transform(unique_smiles)
            desc_matrix = np.empty((n, len(desc_names)), dtype=np.float32)
            for i, smi in enumerate(smiles_list):
                desc_matrix[i] = desc_all[smi]

            # Fast path: read StandardScaler params directly from transformers_ to
            # avoid ColumnTransformer.transform() which requires private sklearn
            # attrs (_columns, _remainder) absent on legacy-reconstructed pipelines.
            _used_fast = False
            if hasattr(self.mol_numeric_pipeline, 'transformers_'):
                try:
                    _col_to_idx = {f'Descriptor_{name}': idx for idx, name in enumerate(desc_names)}
                    _means = np.zeros(len(desc_names), dtype=np.float64)
                    _scales = np.ones(len(desc_names), dtype=np.float64)
                    _found = 0
                    for _tname, _t, _cols in self.mol_numeric_pipeline.transformers_:
                        if _tname == 'remainder' or not _cols:
                            continue
                        _idx = _col_to_idx.get(_cols[0])
                        if _idx is None or not isinstance(_t, StandardScaler):
                            raise TypeError("Unexpected mol_numeric_pipeline layout")
                        _means[_idx] = float(np.ravel(_t.mean_)[0]) if hasattr(_t, 'mean_') else 0.0
                        _s = float(np.ravel(_t.scale_)[0]) if hasattr(_t, 'scale_') else 1.0
                        _scales[_idx] = _s if _s != 0.0 else 1.0
                        _found += 1
                    if _found == len(desc_names):
                        mol_num_results = ((desc_matrix.astype(np.float64) - _means) / _scales).astype(np.float32, copy=False)
                        _used_fast = True
                except Exception:
                    pass

            if not _used_fast:
                mol_num_df = pd.DataFrame({
                    f'Descriptor_{name}': desc_matrix[:, j]
                    for j, name in enumerate(desc_names)
                })
                mol_num_results = self.mol_numeric_pipeline.transform(mol_num_df)

        results: List[Dict[str, np.ndarray]] = []
        for i, smi in enumerate(smiles_list):
            feats: Dict[str, np.ndarray] = {}
            if self.mol_features and "fingerprint" in self.mol_features and fp_results:
                feats['Feature_Fingerprint'] = fp_results[smi]
            if mol_num_results is not None:
                for j, col in enumerate(self.mol_numerical_cols):
                    feats[f'Feature_{col}'] = mol_num_results[i, j:j + 1]
            results.append(feats)

        return results

    def transform_context(self, example: Union[Dict, pd.Series]) -> Dict[str, np.ndarray]:
        """Encode all non-molecular (context) features for one sample.

        Encodes protein/cell embeddings, categorical encodings, and treatment
        time. SMILES is not required. Pass the returned dict to
        :meth:`transform` as the ``context`` argument to screen many molecules
        against the same fixed biological context.

        Args:
            example: Dict or ``pd.Series`` with context columns. SMILES is ignored.

        Returns:
            Dict mapping ``Feature_*`` keys to 1-D numpy arrays.
        """
        if isinstance(example, pd.Series):
            example = example.to_dict()

        feats: Dict[str, np.ndarray] = {}

        if self.cell_features == "description" and self.cell_description_embedding is not None:
            feats[f'Feature_{self.cell_line_col}_Description'] = (
                self.cell_description_embedding.transform(example[self.cell_line_col])
            )

        if self.poi_features == "sequence" and self.poi_sequence_embedding is not None:
            feats[f'Feature_{self.poi_sequence_col}'] = (
                self.poi_sequence_embedding.transform(example[self.poi_sequence_col])
            )

        if self.poi_features == "precomputed" and self.poi_precomputed_embedding is not None:
            poi_id = self._get_protein_id(example, 'poi')
            poi_emb = self.poi_precomputed_embedding.transform(poi_id)
            if self.use_poi_pca and self.poi_pca is not None:
                poi_emb = self.poi_pca.transform(poi_emb.reshape(1, -1)).flatten()
            feats['Feature_POI_Precomputed_Embedding'] = poi_emb

        if self.ligase_features == "precomputed" and self.ligase_precomputed_embedding is not None:
            lig_id = self._get_protein_id(example, 'ligase')
            lig_emb = self.ligase_precomputed_embedding.transform(lig_id)
            if self.use_ligase_pca and self.ligase_pca is not None:
                lig_emb = self.ligase_pca.transform(lig_emb.reshape(1, -1)).flatten()
            feats['Feature_Ligase_Precomputed_Embedding'] = lig_emb

        feats.update(self._run_category_pipeline(example))
        feats.update(self._run_context_numeric_pipeline(example))
        return feats

    def _finalize_output(
        self,
        feats_list: List[Dict[str, np.ndarray]],
        return_tensor: Literal['dict', 'xgb', 'pt', 'np'],
    ) -> Any:
        """Convert a list of per-sample feature dicts to the requested output format.

        Args:
            feats_list: Per-sample dicts (Feature_* → np.ndarray).
            return_tensor: Target format — see :meth:`transform` for details.

        Returns:
            Encoded features in the requested format (stacked over samples).
        """
        if not feats_list:
            if return_tensor in ('dict', 'pt'):
                return {}
            return np.empty((0, 0), dtype=np.float32)

        if return_tensor in ('np', 'xgb'):
            # Ordered Feature_* keys only, matching training order
            if self.feature_dims:
                feat_keys = [k for k in self.feature_dims if k in feats_list[0]]
            else:
                feat_keys = self._apply_feature_order(
                    k for k in feats_list[0] if k.startswith('Feature_')
                )
            rows = [
                np.concatenate([f[k].flatten() for k in feat_keys]).astype(np.float32)
                for f in feats_list
            ]
            matrix = np.stack(rows)
            if return_tensor == 'np':
                return matrix
            return matrix, self.get_xgboost_feature_names()

        # 'dict' or 'pt': all feature keys in features_order order
        all_keys = self._apply_feature_order(feats_list[0].keys())
        result: Dict[str, np.ndarray] = {}
        for k in all_keys:
            stacked = np.stack([f[k] for f in feats_list])
            if k.startswith('Feature_'):
                stacked = stacked.astype(np.float32, copy=False)
            result[k] = stacked
        if return_tensor == 'dict':
            return result
        return {k: torch.from_numpy(v) for k, v in result.items()}

    def transform(
        self,
        input: Union[str, Dict, pd.Series, List[str], List[Dict]],
        context: Optional[Dict[str, np.ndarray]] = None,
        batch_size: Optional[int] = None,
        return_tensor: Literal['dict', 'xgb', 'pt', 'np'] = 'np',
    ) -> Any:
        """Encode one or many samples into feature tensors.

        Args:
            input: What to encode. Accepted forms:

                * A single ``dict`` or ``pd.Series`` — full sample with SMILES
                  and context columns.
                * A ``list`` of ``dict`` — batch of full samples.
                * A single SMILES ``str`` or a ``list`` of SMILES ``str`` —
                  molecular-only encoding; *context* must be provided.

            context: Pre-encoded context dict from :meth:`transform_context`.
                When provided, only molecular features (fingerprint /
                descriptors) are computed per SMILES and merged with
                *context*. When ``None``, all features are computed from
                each full sample dict.
            batch_size: If set, process input in chunks of this size and
                concatenate. Useful for large inputs to limit peak memory.
            return_tensor: Output format:

                * ``'dict'`` — ``Dict[str, np.ndarray]`` shape
                  ``(n, feature_dim)`` per key, ordered by
                  ``self.features_order``.
                * ``'np'`` — ``np.ndarray`` shape ``(n, total_features)``,
                  features concatenated in ``self.features_order``.
                * ``'pt'`` — ``Dict[str, torch.Tensor]`` shape
                  ``(n, feature_dim)`` per key.
                * ``'xgb'`` — ``(np.ndarray, List[str])`` 2-D matrix plus
                  feature names, shape ``(n, total_features)``.

        Returns:
            Encoded features in the format requested by *return_tensor*.

        Raises:
            TypeError: If the input type is not recognised.
            ValueError: If a SMILES-only input is provided without *context*.
        """

        # ------------------------------------------------------------------
        # 1. Normalise input → (smiles_list, examples)
        # ------------------------------------------------------------------
        smiles_list: Optional[List[str]] = None
        examples: Optional[List[Dict]] = None

        if isinstance(input, str):
            smiles_list = [input]
        elif isinstance(input, pd.Series):
            examples = [input.to_dict()]
        elif isinstance(input, dict):
            examples = [input]
        elif isinstance(input, list) and len(input) > 0:
            if isinstance(input[0], str):
                smiles_list = input
            elif isinstance(input[0], dict):
                examples = input
            elif isinstance(input[0], pd.Series):
                examples = [s.to_dict() for s in input]
            else:
                raise TypeError(
                    f"List elements must be str, dict, or pd.Series; "
                    f"got {type(input[0]).__name__}."
                )
        elif isinstance(input, list) and len(input) == 0:
            if return_tensor in ('dict', 'pt'):
                return {}
            return np.empty((0, 0), dtype=np.float32)
        else:
            raise TypeError(
                f"input must be str, dict, pd.Series, List[str], or List[dict]; "
                f"got {type(input).__name__}."
            )

        if smiles_list is not None and context is None:
            raise ValueError(
                "context is required when input is a SMILES string or list of "
                "SMILES strings. Call transform_context() first and pass the "
                "result as context."
            )

        # When context is provided but input is full-sample dicts, extract SMILES
        if examples is not None and context is not None:
            smiles_list = [ex[self.smiles_col] for ex in examples]
            examples = None

        # ------------------------------------------------------------------
        # 2a. Full encoding — no context, full-sample dicts
        # ------------------------------------------------------------------
        if context is None:
            assert examples is not None
            chunks = (
                [examples[i:i + batch_size] for i in range(0, len(examples), batch_size)]
                if batch_size else [examples]
            )
            if return_tensor == 'pt':
                # _transform_batch has a native 'pt' fast path that avoids
                # per-sample loops and builds the batched tensor dict directly
                batches = [
                    self._transform_batch(chunk, return_tensor='pt')
                    for chunk in chunks
                ]
                if len(batches) == 1:
                    return batches[0]
                keys = list(batches[0].keys())
                return {k: torch.cat([b[k] for b in batches], dim=0) for k in keys}

            raw: List[Dict[str, np.ndarray]] = []
            for chunk in chunks:
                raw.extend(self._transform_batch(chunk, return_tensor='np'))
            return self._finalize_output(raw, return_tensor)

        # ------------------------------------------------------------------
        # 2b. Molecular-only encoding + context merge
        # ------------------------------------------------------------------
        assert smiles_list is not None
        chunks_smi = (
            [smiles_list[i:i + batch_size] for i in range(0, len(smiles_list), batch_size)]
            if batch_size else [smiles_list]
        )
        mol_raw: List[Dict[str, np.ndarray]] = []
        for chunk in chunks_smi:
            mol_raw.extend(self._transform_molecular_batch(chunk))
        merged = [{**mol_feats, **context} for mol_feats in mol_raw]
        return self._finalize_output(merged, return_tensor)

    def get_feature_layout(self) -> List[Tuple[str, int, int]]:
        """Return the ordered (key, offset, width) layout of the flat XGBoost feature vector.

        Mirrors the concatenation order used by
        ``transform(return_tensor='xgb')`` / ``get_xgboost_feature_names()``,
        derived from ``self.feature_dims``.

        Returns:
            List of ``(feature_key, offset, width)`` tuples.
        """
        if not self.feature_dims:
            raise RuntimeError(
                "feature_dims not initialized. Call setup() before get_feature_layout()."
            )
        layout = []
        offset = 0
        for key, width in self.feature_dims.items():
            layout.append((key, offset, width))
            offset += width
        return layout

    def featurize_dataset(
            self,
            dataset: Union[Dataset, List[Dict[str, Any]]],
            split_name: str = "unknown",
            num_proc: Optional[int] = None,
    ) -> TorchListDataset:
        """ Featurize an entire dataset split and extract labels.

        Always runs sequentially — sklearn pipelines and embedding objects are
        not picklable, so multiprocessing map is intentionally avoided.

        Args:
            dataset: The dataset to featurize (HuggingFace Dataset or list of dicts).
            split_name: Name of the split (for storing tasks).
            num_proc: Ignored; kept for API compatibility.

        Returns:
            Featurized dataset as a TorchListDataset.
        """
        features_list: List[Dict[str, Any]] = []
        tasks_list: List[Any] = []

        all_samples = [dataset[i] for i in range(len(dataset))]
        all_features = self._transform_batch(all_samples, return_tensor='np')
        for sample, features in zip(all_samples, all_features):
            features.update(self._featurize_and_normalize_labels(sample))
            features_list.append(features)

            if self.is_bert_multitask and self.label_task_col in sample:
                tasks_list.append(sample[self.label_task_col])

        self.logger.debug(f"Featurized dataset shape: {len(features_list)} samples")
        self.logger.debug(f"Dataset columns: {list(features_list[0].keys()) if features_list else []}")

        # Filter NaN labels before tensor conversion
        if len(self.labels) == 1 and not self.impute_labels:
            label = self.labels[0]
            initial_size = len(features_list)
            keep = [pd.notnull(f[label]) for f in features_list]
            features_list = [f for f, k in zip(features_list, keep) if k]
            if tasks_list:
                tasks_list = [t for t, k in zip(tasks_list, keep) if k]
            self.logger.debug(f"Removed {initial_size - len(features_list)} samples with NaN labels")

        # Convert numpy arrays and Python scalars to torch tensors
        def _to_torch(val: Any) -> torch.Tensor:
            if isinstance(val, torch.Tensor):
                return val
            if isinstance(val, np.ndarray):
                arr = np.ascontiguousarray(val)
                if np.issubdtype(arr.dtype, np.integer):
                    return torch.from_numpy(arr.astype(np.int64))
                return torch.from_numpy(arr.astype(np.float32))
            if val is None:
                return torch.tensor(float('nan'))
            try:
                return torch.tensor(float(val), dtype=torch.float32)
            except (TypeError, ValueError):
                return torch.tensor(float('nan'))

        torch_list = [{k: _to_torch(v) for k, v in f.items()} for f in features_list]

        # Store tasks for this split
        if split_name == "train":
            self.train_tasks = tasks_list if tasks_list else None
            self.logger.debug(f"Stored {len(tasks_list) if tasks_list else 0} tasks for training split")
        elif split_name == "validation":
            self.val_tasks = tasks_list if tasks_list else None
            self.logger.debug(f"Stored {len(tasks_list) if tasks_list else 0} tasks for validation split")
        elif split_name == "test":
            self.test_tasks = tasks_list if tasks_list else None
            self.logger.debug(f"Stored {len(tasks_list) if tasks_list else 0} tasks for test split")

        return TorchListDataset(torch_list)

    def assemble_prompt(
        self,
        smiles: str,
        target: Optional[str] = None,
        e3: Optional[str] = None,
        cell_line: Optional[str] = None,
        assay_type: Optional[str] = None,
        treatment_time: Optional[float] = None,
        degrader_type: Optional[str] = None,
        label_type: Optional[str] = None,
    ) -> str:
        """ Assemble a text prompt for BERT-based models.
        
        Args:
            smiles: The SMILES string of the molecule.
            target: The target protein name.
            e3: The E3 ligase name.
            cell_line: The cell line identifier.
            treatment_time: The treatment time in hours.
            degrader_type: The type of degrader (e.g., "PROTAC", "molecular glue").
            label_type: The type of value being predicted (e.g., "DC50", "Dmax").
            
        Returns:
            The assembled prompt string.
        """
        if self.prompt_template is not None:
            # Use custom template with placeholder substitution
            prompt = self.prompt_template.format(
                smiles=smiles or "",
                target=target or "unknown target",
                e3=e3 or "unknown E3 ligase",
                cell_line=cell_line or "",
                assay_type=assay_type or "",
                treatment_time=treatment_time or "",
                degrader_type=degrader_type or self.default_degrader_type,
                label_type=label_type or "activity",
            )
        else:
            # Default prompt assembly
            degrader_type = degrader_type or self.default_degrader_type
            
            # Build context parts
            assay_type_part = f", using {assay_type} assay, " if assay_type and pd.notnull(assay_type) else ""
            cell_line_part = f" in {cell_line} cell line " if cell_line and pd.notnull(cell_line) else ""
            treatment_time_part = f" after {int(treatment_time)} hours" if treatment_time and pd.notnull(treatment_time) else ""
            label_type_part = label_type if label_type and pd.notnull(label_type) else " degradation activity "
            target_part = target if target and pd.notnull(target) else " the target protein "
            e3_part = e3 if e3 and pd.notnull(e3) else " an "
            
            prompt = (
                f"Predict {label_type_part} {treatment_time_part} when targeting {target_part} {cell_line_part} "
                f"via {e3_part} E3 recruiter {assay_type_part} for {degrader_type}: {smiles}"
            )
        
        # Clean up extra spaces
        prompt = " ".join(prompt.split()).replace(" ,", ",")
        return prompt.strip()

    def _get_treatment_time(self, example: Dict, label: str) -> Optional[float]:
        """ Get the appropriate treatment time based on the label being predicted."""
        # TODO: Simplify this logic: in multi-task settings we assume the same treatment time
        if "DC50" in label:
            return example.get(self.treatment_time_dc50_col)
        elif "Dmax" in label:
            return example.get(self.treatment_time_dmax_col)
        elif "IC50" in label:
            return example.get(self.treatment_time_ic50_col)
        return example.get(self.treatment_time_col)

    def _tokenize_sample(self, example: Dict) -> Dict:
        """ Tokenize a single example's prompt."""
        # Get degrader type from example or use default
        if self.degrader_type_col and self.degrader_type_col in example:
            degrader_type = example[self.degrader_type_col]
        else:
            degrader_type = self.default_degrader_type
        
        # Get label type from example
        label_type = example.get(self.label_task_col)
        
        # Determine treatment time based on label columns present
        # TODO: The treatment time shouldn't be based on which label columns to
        # chose from, in a multi-task setting.
        treatment_time = None
        for label in self.labels:
            if label in example and pd.notna(example.get(label)):
                treatment_time = self._get_treatment_time(example, label)
                break
        
        # Assemble the prompt
        prompt = self.assemble_prompt(
            smiles=example.get(self.smiles_col),
            target=example.get(self.poi_col),
            e3=example.get(self.ligase_col),
            cell_line=example.get(self.cell_line_col),
            assay_type=example.get(self.assay_type_col),
            treatment_time=treatment_time,
            degrader_type=degrader_type,
            label_type=label_type,
        )
        
        # Tokenize
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='np',
        )
        
        result = {
            'Prompt': prompt,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }
        
        # Add token_type_ids if available
        if 'token_type_ids' in encoding:
            result['token_type_ids'] = encoding['token_type_ids'].flatten()
        
        return result

    def _initialize_feature_dims(self, example: Dict) -> None:
        """ Initialize feature dimensions dictionary from a sample.
        
        Args:
            example: A dictionary representing a single featurized data point.
        """
        self.feature_dims = {}
        
        # Excluded keys that are not features
        excluded_keys = set(self.labels) | {'Prompt', 'input_ids', 'attention_mask', 'token_type_ids'}
        
        for key, value in example.items():
            if key in excluded_keys:
                continue
            if isinstance(value, np.ndarray):
                self.feature_dims[key] = value.flatten().shape[0]
            elif isinstance(value, (int, float)):
                self.feature_dims[key] = 1
            elif hasattr(value, 'shape'):
                self.feature_dims[key] = value.flatten().shape[0]
        
        # Add tokenizer dimensions if applicable
        if self.use_tokenizer:
            self.feature_dims['input_ids'] = self.max_length
            self.feature_dims['attention_mask'] = self.max_length
            if 'token_type_ids' in example:
                self.feature_dims['token_type_ids'] = self.max_length
        
        ordered_keys = self._apply_feature_order(self.feature_dims.keys())
        self.feature_dims = {k: self.feature_dims[k] for k in ordered_keys}

        self.logger.debug(f"Feature dimensions initialized: {self.feature_dims}")

    def _apply_feature_order(self, keys: Iterable[str]) -> List[str]:
        """Return *keys* reordered according to ``self.features_order``.

        Args:
            keys: Feature key names to order.

        Returns:
            Ordered list of feature keys.
        """
        keys = list(keys)
        if self.features_order == "sorted":
            return sorted(keys)
        # mol_first: molecular (SMILES-dependent) keys first, context keys after,
        # both groups sorted alphabetically within themselves.
        mol = sorted(k for k in keys if _is_molecular_feature(k))
        ctx = sorted(k for k in keys if not _is_molecular_feature(k))
        return mol + ctx

    def get_xgboost_feature_names(self) -> List[str]:
        """ Get the list of feature names in the order they are concatenated for XGBoost.
        
        Returns:
            List of feature names.
        """
        # Check if self.feature_dims has been initialized
        if not self.feature_dims:
            raise ValueError("Feature dimensions have not been initialized. Ensure that setup() has been called and the training dataset has been featurized.")

        # For certain features, like sequence embeddings, expand feature names,
        # e.g., name each n-gram
        feature_names_out = []
        for feat, dim in self.feature_dims.items():
            if self.poi_sequence_col in feat or self.ligase_sequence_col in feat:
                if self.poi_sequence_embedding is not None:
                    ngram_names = self.poi_sequence_embedding.sklearn_encoder.get_feature_names_out()
                    feature_names_out.extend([f"{feat}_{ngram}" for ngram in ngram_names])
                else:
                    feature_names_out.extend([f"{feat}_{j}" for j in range(dim)])
            elif "Descriptor" not in feat or dim > 1:
                feature_names_out.extend([f"{feat}_{j}" for j in range(dim)])
            else:
                feature_names_out.append(feat)
        return feature_names_out

    def get_feature_dims(self) -> Dict[str, int]:
        """ Return the dictionary mapping feature names to their dimensions.

        Returns:
            Dictionary mapping feature names to integer dimensions.
        """
        return self.feature_dims.copy()

    def get_categorical_vocab_sizes(self) -> Dict[str, int]:
        """ Return vocab sizes for embedding-encoded categorical features.

        Returns {feature_key: vocab_size} where vocab_size = n_known_categories + 1.
        Index 0 is reserved for unknown categories (after the +1 shift applied in
        _run_category_pipeline). Only meaningful when categorical_encoding == 'embedding'.

        Returns:
            Dictionary mapping feature names to their vocabulary sizes.
        """
        if self.categorical_encoding != 'embedding' or self.category_pipeline is None:
            return {}
        sizes = {}
        for _, transformer, cols in self.category_pipeline.transformers_:
            ordinal = transformer.named_steps.get('ordinal')
            if ordinal is None or not hasattr(ordinal, 'categories_'):
                continue
            for col, cats in zip(cols, ordinal.categories_):
                sizes[f'Feature_{col}'] = len(cats) + 1  # +1 for unknown at index 0
        return sizes

    def get_total_feature_dim(self, exclude_tokenizer: bool = True) -> int:
        """ Calculate the total dimension of all features.
        
        Args:
            exclude_tokenizer: Whether to exclude tokenizer-related features.
            
        Returns:
            Total dimension as an integer.
        """
        excluded = set()
        if exclude_tokenizer:
            excluded = {'input_ids', 'attention_mask', 'token_type_ids', 'Prompt'}

        return sum(dim for name, dim in self.feature_dims.items() if name not in excluded)

    def _resolve_feature_keys(self, key_templates: List[str]) -> List[str]:
        """ Resolve ``Feature_*`` key templates against this config's columns.

        Templates may contain column placeholders (e.g.
        ``"Feature_{poi_col}"``) or a trailing-prefix marker (e.g.
        ``"Feature_Descriptor_"`` matches every descriptor feature key).

        Args:
            key_templates: List of feature-key templates from FEATURE_REGISTRY.

        Returns:
            Concrete feature keys present in ``self.feature_dims`` (when set up),
            otherwise the formatted template keys.
        """
        resolved: List[str] = []
        for template in key_templates:
            key = template.format(
                poi_col=self.poi_col,
                ligase_col=self.ligase_col,
                cell_line_col=self.cell_line_col,
                assay_type_col=self.assay_type_col,
                poi_sequence_col=self.poi_sequence_col,
                treatment_time_col=self.treatment_time_col,
            )
            if key.endswith('_'):
                # Prefix marker: expand to all matching feature_dims keys.
                matches = [k for k in self.feature_dims if k.startswith(key)]
                resolved.extend(matches if matches else [key])
            else:
                resolved.append(key)
        return resolved

    def _is_feature_active(self, key: str) -> bool:
        """Return whether the named FEATURE_REGISTRY entry is enabled by the current config.

        Args:
            key: A key from :data:`FEATURE_REGISTRY`.

        Returns:
            True if the corresponding feature is active.
        """
        checks = {
            "fingerprint":        lambda: bool(self.mol_features and "fingerprint" in self.mol_features),
            "descriptors":        lambda: bool(self.mol_features and "descriptors" in self.mol_features),
            "poi_precomputed":    lambda: self.poi_features == "precomputed",
            "poi_sequence":       lambda: self.poi_features == "sequence",
            "poi_name":           lambda: self.poi_features == "name",
            "ligase_precomputed": lambda: self.ligase_features == "precomputed",
            "ligase_name":        lambda: self.ligase_features == "name",
            "cell_description":   lambda: self.cell_features == "description",
            "cell_name":          lambda: self.cell_features == "name",
            "assay_type":         lambda: self.use_assay_type_encoding,
            "treatment_time":     lambda: self.use_treatment_time,
            "poi_pca":            lambda: self.use_poi_pca,
            "ligase_pca":         lambda: self.use_ligase_pca,
        }
        return checks.get(key, lambda: False)()

    def get_feature_spec(self) -> List[Dict[str, Any]]:
        """ Describe the active processed features and how they are produced.

        Returns one entry per enabled feature with its processing ``kind``
        (``stateless`` vs ``fitted``; see :data:`FEATURE_REGISTRY`), the
        concrete ``Feature_*`` keys it maps to, the total ``dim`` (when the data
        module has been set up), and the on-disk ``cache`` file for stateless,
        disk-cached embedders.

        This is the structured replacement for reverse-parsing ``str(self)``: it
        is recorded verbatim in the run manifest so downstream tooling can tell,
        without string matching, which features were used and which can be
        computed once and shared across folds / ensemble members.

        Returns:
            List of ``{name, kind, token, feature_keys, dim, cache}`` dicts,
            ordered as in FEATURE_REGISTRY.
        """
        spec: List[Dict[str, Any]] = []
        for flag, meta in FEATURE_REGISTRY.items():
            if not self._is_feature_active(flag):
                continue

            feature_keys = self._resolve_feature_keys(meta["feature_keys"])
            dim = sum(self.feature_dims.get(k, 0) for k in feature_keys)

            cache = None
            embedder_attr = meta.get("embedder_attr")
            if embedder_attr is not None:
                embedder = getattr(self, embedder_attr, None)
                if embedder is not None:
                    cache_path = (
                        getattr(embedder, "embeddings_file", None)
                        or getattr(embedder, "filename", None)
                    )
                    cache = str(cache_path) if cache_path is not None else None

            spec.append({
                "name": meta["name"],
                "kind": meta["kind"],
                "token": meta["token"],
                "feature_keys": feature_keys,
                "dim": int(dim) if dim else None,
                "cache": cache,
            })
        return spec

    def _create_category_pipeline(self):
        """ Create sklearn pipeline for categorical feature preprocessing."""
        transformers = []
        
        # Ordinal encoding + shift + MinMax for each categorical feature
        self.categorical_cols = []
        if self.ligase_features == "name":
            self.categorical_cols.append(self.ligase_col)
        if self.poi_features == "name":
            self.categorical_cols.append(self.poi_col)
        if self.cell_features == "name":
            self.categorical_cols.append(self.cell_line_col)
        if self.use_assay_type_encoding:
            self.categorical_cols.append(self.assay_type_col)

        for col in self.categorical_cols:
            if self.categorical_encoding == 'embedding':
                # For PyTorch embeddings: ordinal only, int64 so indices can go
                # directly into nn.Embedding without an extra cast.
                # The +1 shift (unknown → 0, known → 1..N) is applied in
                # _run_category_pipeline() after transformation.
                inner = Pipeline([
                    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
                    ('ordinal', OrdinalEncoder(
                        handle_unknown='use_encoded_value',
                        unknown_value=-1,
                        dtype=np.int64
                    )),
                ])
            elif self.categorical_encoding == 'onehot':
                inner = Pipeline([
                    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
                    ('onehot', OneHotEncoder(
                        handle_unknown='ignore',
                        sparse_output=False
                    )),
                ])
            else:
                # Default 'minmax' mode: ordinal + MinMax scaling to [0, 1]
                inner = Pipeline([
                    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
                    ('ordinal', OrdinalEncoder(
                        handle_unknown='use_encoded_value',
                        unknown_value=-1,
                        dtype=np.int32
                    )),
                    ('minmax', MinMaxScaler())
                ])
            transformers.append((f'{col}_pipeline', inner, [col]))
        
        self.category_pipeline = None
        if transformers:
            self.category_pipeline = ColumnTransformer(
                transformers=transformers,
                remainder='drop',
                sparse_threshold=0,
            )

    def _fit_category_pipeline(self, train_df: pd.DataFrame) -> None:
        """ Fit category pipeline on training data.
        
        Args:
            train_df: Training dataframe.
        """
        if self.category_pipeline is None:
            return

        X = pd.DataFrame({col: train_df[col] for col in self.categorical_cols})
        self.category_pipeline.fit(X)
        self.logger.debug("Category pipeline fitted on training data")

    def _run_category_pipeline(self, example: Union[Dict, pd.Series]) -> Dict[str, np.ndarray]:
        """ Run the category pipeline on a single example.

        Args:
            example: A dictionary representing a single data point.

        Returns:
            A dictionary with processed categorical features.
        """
        if self.category_pipeline is None:
            return {}

        # NOTE: The key is to rely on the columns defined in the
        # `_create_category_pipeline` method.
        X = pd.DataFrame({col: [example[col]] for col in self.categorical_cols})
        transformed = self.category_pipeline.transform(X)

        result = {}
        offset = 0
        for col, (_, inner, _) in zip(self.categorical_cols, self.category_pipeline.transformers_):
            width = inner.transform(X[[col]]).shape[1]
            val = transformed[0, offset:offset + width]
            if self.categorical_encoding == 'embedding':
                val = val + 1  # shift: unknown → 0, known categories → 1..N
            result[f'Feature_{col}'] = val
            offset += width

        return result

    def _create_numeric_pipelines(self):
        """ Create sklearn pipelines for numeric feature preprocessing.

        Builds three independent ColumnTransformer objects (no shared
        transformer instances, so each can be fit/saved/loaded on its own):
        - `context_numeric_pipeline`: treatment time only (SMILES-independent).
        - `mol_numeric_pipeline`: RDKit descriptors only (SMILES-dependent).
        - `numeric_pipeline`: both combined; kept for the legacy single-pass
          featurization path (`_run_numeric_pipeline`, `_transform_batch`).
        """
        default_treatment_time = 24  # hours

        def _treatment_time_transformer() -> Pipeline:
            # NOTE: The second imputer is to handle the case where all
            # values are NaN
            return Pipeline([
                ('mean_imputer', SimpleImputer(strategy='mean', keep_empty_features=True)),
                ('const_imputer', SimpleImputer(strategy='constant', fill_value=default_treatment_time, keep_empty_features=True)),
                ('scaler', StandardScaler())
            ])

        self.context_numerical_cols = []
        context_transformers = []
        if self.use_treatment_time:
            self.context_numerical_cols.append(self.treatment_time_col)
            context_transformers.append((
                'treatment_time_pipeline', _treatment_time_transformer(), [self.treatment_time_col]
            ))

        self.mol_numerical_cols = []
        mol_transformers = []
        combined_transformers = [
            ('treatment_time_pipeline', _treatment_time_transformer(), [self.treatment_time_col])
        ] if self.use_treatment_time else []
        if self.mol_features and "descriptors" in self.mol_features:
            for desc_name in self.desc_embedder.get_descriptor_names():
                col = f'Descriptor_{desc_name}'
                self.mol_numerical_cols.append(col)
                mol_transformers.append((f'descriptor_{desc_name}_pipeline', StandardScaler(), [col]))
                combined_transformers.append((f'descriptor_{desc_name}_pipeline', StandardScaler(), [col]))

        self.numerical_cols = self.context_numerical_cols + self.mol_numerical_cols

        self.context_numeric_pipeline = None
        if context_transformers:
            self.context_numeric_pipeline = ColumnTransformer(
                transformers=context_transformers,
                remainder='drop',
                sparse_threshold=0,
            )

        self.mol_numeric_pipeline = None
        if mol_transformers:
            self.mol_numeric_pipeline = ColumnTransformer(
                transformers=mol_transformers,
                remainder='drop',
                sparse_threshold=0,
            )

        self.numeric_pipeline = None
        if combined_transformers:
            self.numeric_pipeline = ColumnTransformer(
                transformers=combined_transformers,
                remainder='drop',
                sparse_threshold=0,
            )

    def _fit_numeric_pipelines(self, train_df: pd.DataFrame) -> None:
        """ Fit context, molecular, and combined numeric pipelines on training data.

        Computes RDKit descriptors once and reuses them to fit both the
        combined `numeric_pipeline` (legacy single-pass path) and the split
        `mol_numeric_pipeline` (molecular-only batch path).

        Args:
            train_df: Training dataframe.
        """
        data = {}
        if self.use_treatment_time:
            data[self.treatment_time_col] = train_df[self.treatment_time_col]

        if self.mol_features and "descriptors" in self.mol_features:
            descs_list = []
            for _, row in train_df.iterrows():
                # Each descriptor is an array of shape: (num_descriptors,)
                descs = self.desc_embedder.transform(row[self.smiles_col])
                descs_list.append(descs)
            descs_array = np.array(descs_list, dtype=np.float32).T  # Shape: (num_descriptors, num_samples)
            for i, name in enumerate(self.desc_embedder.get_descriptor_names()):
                data[f'Descriptor_{name}'] = descs_array[i]

        numeric_df = pd.DataFrame(data)

        if self.context_numeric_pipeline is not None:
            self.context_numeric_pipeline.fit(numeric_df[self.context_numerical_cols])

        if self.mol_numeric_pipeline is not None:
            self.mol_numeric_pipeline.fit(numeric_df[self.mol_numerical_cols])

        if self.numeric_pipeline is not None:
            self.numeric_pipeline.fit(numeric_df)

        self.logger.debug("Numeric pipelines fitted on training data")

    def _run_numeric_pipeline(self, example: Union[Dict, pd.Series]) -> Dict[str, np.ndarray]:
        """ Run the numeric pipeline on a single example.
        
        Args:
            example: A dictionary representing a single data point.
            
        Returns:
            A dictionary with processed numeric features.
        """
        if self.numeric_pipeline is None:
            return {}

        data = {}
        if self.use_treatment_time:
            data[self.treatment_time_col] = example[self.treatment_time_col]
        
        if self.mol_features and "descriptors" in self.mol_features:
            descs = self.desc_embedder.transform(example[self.smiles_col])
            for i, name in enumerate(self.desc_embedder.get_descriptor_names()):
                data[f'Descriptor_{name}'] = np.array([descs[i]], dtype=np.float32)

        transformed = self.numeric_pipeline.transform(pd.DataFrame([data]))
        
        result = {}
        for i, col in enumerate(self.numerical_cols):
            result[f'Feature_{col}'] = transformed[0, i:i+1]

        return result

    def _run_context_numeric_pipeline(self, example: Union[Dict, pd.Series]) -> Dict[str, np.ndarray]:
        """ Run the context-only numeric pipeline (treatment time) on a single example.

        Args:
            example: A dictionary representing a single data point.

        Returns:
            A dictionary with processed context-only numeric features.
        """
        if self.context_numeric_pipeline is None:
            return {}

        data = {}
        if self.use_treatment_time:
            data[self.treatment_time_col] = example[self.treatment_time_col]

        transformed = self.context_numeric_pipeline.transform(pd.DataFrame([data]))

        result = {}
        for i, col in enumerate(self.context_numerical_cols):
            result[f'Feature_{col}'] = transformed[0, i:i+1]

        return result

    def _create_label_pipeline(self, labels_len: int = 1000) -> Pipeline:
        """ Create pipeline for label normalization."""
        transformers = []
        # if self.impute_labels:
        #     transformers.append(('imputer', SimpleImputer(strategy='mean', add_indicator=True)))
        if self.standardize_labels:
            return StandardScaler()
        elif self.normalize_labels:
            transformers =[
                ('quantile', QuantileTransformer(
                        output_distribution='normal',
                        n_quantiles=min(1000, labels_len),
                        random_state=42,
                )),
                # ('minmax', MinMaxScaler()),
            ]
            return Pipeline(transformers)

    def _fit_label_transformers(self, train_df: pd.DataFrame) -> None:
        """ Fit label transformers on training data.
        
        For BERT multi-task (single label column with Value_Type), fits separate
        transformers for each Value_Type. For other cases, fits one transformer
        per label column.
        
        Args:
            train_df: Training dataframe.
        """        
        if self.is_bert_multitask:
            # BERT multi-task: fit separate transformers for each Value_Type
            label = self.labels[0]
            for value_type in ['Dmax', 'DC50']:
                subset = train_df[train_df[self.label_task_col] == value_type]
                if len(subset) > 0:
                    label_values = subset[label].dropna().values.reshape(-1, 1)
                    if len(label_values) > 0:
                        self.label_transformers[label] = self._create_label_pipeline(len(label_values))
                        self.label_transformers[label].fit(label_values)
                        self.logger.debug(f"Fitted label transformer for Value_Type: {value_type} on {len(label_values)} samples")
        else:
            # Standard case: one transformer per label
            for label in self.labels:
                if label not in train_df.columns:
                    self.logger.warning(f"Label column '{label}' not found in training data, skipping transformer fitting.")
                    continue
                label_values = train_df[label].dropna().values.reshape(-1, 1)
                if len(label_values) > 0:
                    self.label_transformers[label] = self._create_label_pipeline(len(label_values))
                    self.label_transformers[label].fit(label_values)
                    self.logger.debug(f"Fitted label transformer for label: {label} on {len(label_values)} samples")

    def _featurize_and_normalize_labels(self, example: Union[Dict, pd.Series]) -> Dict[str, Any]:
        """ Featurize and normalize labels for a single example.
        
        Args:
            example: A dictionary representing a single data point.
        
        Returns:
            A dictionary with normalized labels.
        """
        labels_dict = {}
        for label in self.labels:
            raw_value = example[label]
            if pd.notnull(raw_value) and (self.normalize_labels or self.standardize_labels):
                # Determine the transformer key
                # TODO: This logic feels wrong...
                if self.is_bert_multitask:
                    transformer_key = example.get(self.label_task_col)
                else:
                    transformer_key = label
                
                if transformer_key and transformer_key in self.label_transformers:
                    label_value = np.array([[raw_value]], dtype=np.float32)
                    labels_dict[label] = self.label_transformers[transformer_key].transform(label_value).flatten()[0]
                else:
                    labels_dict[label] = raw_value
            else:
                labels_dict[label] = raw_value
        return labels_dict

    def inverse_transform_labels(self, y: np.ndarray, key: str) -> np.ndarray:
        """ Inverse transform normalized labels back to original scale.
        
        Args:
            y: The normalized label values as a numpy array.
            key: The transformer key (label name or Value_Type for BERT multi-task).
            
        Returns:
            The inverse transformed label values as a numpy array.
        """
        if not self.normalize_labels and not self.standardize_labels:
            return y
            
        if key not in self.label_transformers:
            raise ValueError(f"No label transformer found for key: {key}. Available keys: {list(self.label_transformers.keys())}")
        
        original_shape = y.shape
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        y_unscaled = self.label_transformers[key].inverse_transform(y)
        
        if len(original_shape) == 1:
            return y_unscaled.flatten()
        return y_unscaled

    def _fit_pca_on_protein_embeddings(self, train_df: pd.DataFrame) -> None:
        """ Fit PCA on precomputed protein embeddings from training data.
        
        Args:
            train_df: Training dataframe.
        """
        # Fit PCA on precomputed embeddings
        if self.use_poi_pca and self.poi_precomputed_embedding is not None:
            self.logger.debug("Fitting PCA on POI precomputed embeddings...")
            # Get IDs for all POIs in training set
            poi_ids = train_df.apply(lambda x: self._get_protein_id(x, 'poi'), axis=1).tolist()
            # Transform to embeddings matrix
            poi_embeddings = self.poi_precomputed_embedding.transform(poi_ids)
            poi_embeddings = np.array([v for v in poi_embeddings.values()])
            # Finally fit PCA
            self.poi_pca.fit(poi_embeddings)
            self.logger.debug(f"POI PCA fitted with {self.poi_pca.n_components_} components")
            
        if self.use_ligase_pca and self.ligase_precomputed_embedding is not None:
            self.logger.debug("Fitting PCA on Ligase precomputed embeddings...")
            # Get IDs for all ligases in training set
            ligase_ids = train_df.apply(lambda x: self._get_protein_id(x, 'ligase'), axis=1).tolist()
            # Transform to embeddings matrix
            ligase_embeddings = self.ligase_precomputed_embedding.transform(ligase_ids)
            ligase_embeddings = np.array([v for v in ligase_embeddings.values()])
            # Finally fit PCA
            self.ligase_pca.fit(ligase_embeddings)
            self.logger.debug(f"Ligase PCA fitted with {self.ligase_pca.n_components_} components")
        
    def cache_mol_descriptors(self):
        """ Precompute and cache molecular descriptors for all SMILES in the dataset."""
        if not self.mol_features and "descriptors" in self.mol_features or self.desc_embedder is None:
            return
        
        all_smiles = set()
        for split in ['train', 'validation', 'test']:
            if split in self.dataset:
                all_smiles.update(self.dataset[split][self.smiles_col])
        
        self.logger.info(f"Caching molecular descriptors for {len(all_smiles)} unique SMILES...")
        self.desc_embedder.transform(list(all_smiles), update_cache=True)
        self.logger.info("Molecular descriptors cached.")

    def cache_cell_description_embeddings(self):
        """ Precompute and cache cell description embeddings for all cell lines in the dataset."""
        if not self.cell_features == "description" or self.cell_description_embedding is None:
            return
        
        all_cell_lines = set()
        for split in ['train', 'validation', 'test']:
            if split in self.dataset:
                all_cell_lines.update(self.dataset[split][self.cell_line_col])
        
        self.logger.info(f"Caching cell description embeddings for {len(all_cell_lines)} unique cell lines...")
        
        all_cell_lines = list(all_cell_lines)
        if self.verbose > 1:
            indexes = list(range(0, len(all_cell_lines), 8))
        else:
            indexes = range(0, len(all_cell_lines), 8)
        
        for i in indexes:
            batch = list(all_cell_lines)[i:i+8]
            self.cell_description_embedding.transform(batch, update_cache=True)
        
        self.logger.info("Cell description embeddings cached.")

    def train_dataloader(self):
        """Return the training DataLoader (shuffled, drop_last=True)."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )
    
    def val_dataloader(self):
        """Return the validation DataLoader (sequential)."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
    
    def test_dataloader(self):
        """Return the test DataLoader (sequential)."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )     

    def get_Xy(self, dataset: TorchListDataset, return_features_names: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """ Convert a featurized TorchListDataset to numpy arrays for XGBoost.

        Args:
            dataset: The featurized TorchListDataset to convert.
            return_features_names: Whether to return the feature names.

        Returns:
            Tuple of (X, y, feature_names) or (X, y).
        """
        self.logger.debug("Extracting feature names for XGBoost DMatrix...")
        first_row = dataset[0]

        excluded_keys = set(self.labels) | {'Prompt', 'input_ids', 'attention_mask', 'token_type_ids'}
        features_names = [k for k in first_row.keys() if k not in excluded_keys]
        features_names = self._apply_feature_order(features_names)

        feature_names_out = []
        for feat in features_names:
            vect = first_row[feat].numpy().flatten()
            if self.poi_sequence_col in feat or self.ligase_sequence_col in feat:
                if self.poi_sequence_embedding is not None:
                    ngram_names = self.poi_sequence_embedding.sklearn_encoder.get_feature_names_out()
                    feature_names_out.extend([f"{feat}_{ngram}" for ngram in ngram_names])
                else:
                    feature_names_out.extend([f"{feat}_{j}" for j in range(len(vect))])
            elif "Descriptor" not in feat or vect.shape[0] > 1:
                feature_names_out.extend([f"{feat}_{j}" for j in range(len(vect))])
            else:
                feature_names_out.append(feat)

        self.logger.debug("Extracting features for XGBoost DMatrix...")
        X = np.vstack([
            np.concatenate([dataset[i][feat].numpy().flatten() for feat in features_names])
            for i in range(len(dataset))
        ])

        include_labels = all(label in dataset.column_names for label in self.labels)
        if not include_labels:
            for label in self.labels:
                if label not in dataset.column_names:
                    self.logger.warning(f"Label column '{label}' not found in dataset, skipping label extraction.")
            y = None
        else:
            self.logger.debug("Extracting labels for XGBoost DMatrix...")
            y = np.hstack([
                np.array([dataset[i][label].item() for i in range(len(dataset))], dtype=np.float32)[:, np.newaxis]
                for label in self.labels
            ]).astype(np.float32)
            self.logger.debug(f"Final feature matrix shape: {X.shape}")
            self.logger.debug(f"Final label matrix shape: {y.shape}")

        if return_features_names:
            return X, y, feature_names_out
        return X, y

    def get_xgboost_dataset(
            self,
            split: str,
            quantile_matrix: bool = False,
            dtrain: Optional[Union[xgb.DMatrix, xgb.QuantileDMatrix]] = None,
    ) -> Union[xgb.DMatrix, xgb.QuantileDMatrix]:
        """ Convert the specified split to an XGBoost DMatrix. """
        if split == "train":
            dataset = self.train_dataset
        elif split == "validation":
            dataset = self.val_dataset
        elif split == "test":
            dataset = self.test_dataset
        else:
            raise ValueError("split must be 'train', 'validation', or 'test'")
        return self.get_xgboost_dmatrix(dataset, quantile_matrix, dtrain, weight=split == "train")
       
    def get_xgboost_dmatrix(
            self,
            dataset: Dataset,
            quantile_matrix: bool = False,
            dtrain: Optional[Union[xgb.DMatrix, xgb.QuantileDMatrix]] = None,
            weight: bool = False,
    ) -> Union[xgb.DMatrix, xgb.QuantileDMatrix]:
        """ Convert a featurized HuggingFace Dataset to an XGBoost DMatrix.
        
        Args:
            dataset: The HuggingFace Dataset to convert.
            quantile_matrix: Whether to use QuantileDMatrix.
            dtrain: Reference DMatrix for QuantileDMatrix (if applicable).
            
        Returns:
            An XGBoost DMatrix or QuantileDMatrix containing features and labels (if available).
        """
        X, y, feature_names = self.get_Xy(dataset)
        
        weights = None
        if weight:
            try:
                # Estimate density of your target variable
                density = gaussian_kde(y)
                weights = 1.0 / density(y)

                # Normalize weights (optional, keeps learning rate effective)
                weights = weights / weights.mean()
            except Exception as e:
                self.logger.warning(f"Failed to compute sample weights: {e}")
                weights = compute_sample_weight(class_weight="balanced", y=y.flatten())
        
        if quantile_matrix:
            return xgb.QuantileDMatrix(data=X, label=y,
                                       feature_names=feature_names, ref=dtrain,
                                       weight=weights)
        return xgb.DMatrix(data=X, label=y, feature_names=feature_names,
                           weight=weights)

        
    def state_dict(self) -> Dict[str, Any]:
        """Return the state dict containing all encoders and configuration.
        
        Compatible with PyTorch Lightning's state management.
        
        Returns:
            Dictionary containing serializable state.
        """
        state = {
            'feature_dims': self.feature_dims.copy(),
            '_encoders_fitted': self._encoders_fitted,
            'hparams': dict(self.hparams),
        }

        # Save encoders and transformers
        if self.category_pipeline is not None:
            state['categorical_cols'] = self.categorical_cols
            state['category_pipeline'] = pickle.dumps(self.category_pipeline)

        if self.numeric_pipeline is not None:
            state['numerical_cols'] = self.numerical_cols
            state['numeric_pipeline'] = pickle.dumps(self.numeric_pipeline)

        if self.context_numeric_pipeline is not None:
            state['context_numerical_cols'] = self.context_numerical_cols
            state['context_numeric_pipeline'] = pickle.dumps(self.context_numeric_pipeline)

        if self.mol_numeric_pipeline is not None:
            state['mol_numerical_cols'] = self.mol_numerical_cols
            state['mol_numeric_pipeline'] = pickle.dumps(self.mol_numeric_pipeline)

        state['state_dict_version'] = 2

        # Helper to compress a dictionary of sklearn estimators using Pickle
        def _serialize_estimators(estimators_dict):
            serialized = {}
            for key, estimator in estimators_dict.items():
                # Pickle dumps directly to bytes
                serialized[key] = pickle.dumps(estimator)
            return serialized

        # Save label transformers
        if self.normalize_labels or self.standardize_labels:
            state['label_transformers'] = _serialize_estimators(self.label_transformers)
            
        # Save POI sequence embedding (TfidfVectorizer)
        if self.poi_features == "sequence" and self.poi_sequence_embedding is not None:
            if hasattr(self.poi_sequence_embedding, 'sklearn_encoder') and \
               hasattr(self.poi_sequence_embedding.sklearn_encoder, 'vocabulary_'):
                state['poi_sequence_embedding_sklearn_encoder'] = pickle.dumps(self.poi_sequence_embedding.sklearn_encoder)
        
        # Save fingerprint embedder info
        if self.mol_features and "fingerprint" in self.mol_features and self.fp_embedder is not None:
            state['fp_embedder'] = {
                'fp_size': self.fp_size,
                'radius': self.radius,
            }
        
        # Save tokenizer name
        if self.use_tokenizer:
            state['tokenizer_name'] = self.tokenizer_name
            state['max_length'] = self.max_length
        
        # Save PCA transformers
        if self.use_poi_pca and self.poi_pca is not None:
            state['poi_pca'] = pickle.dumps(self.poi_pca)
        
        if self.use_ligase_pca and self.ligase_pca is not None:
            state['ligase_pca'] = pickle.dumps(self.ligase_pca)

        return state
    
    @staticmethod
    def _heal_column_transformer(ct: Any) -> Any:
        """Back-populate private attributes on a ColumnTransformer loaded from an old pickle.

        Older sklearn versions used ``sparse_threshold`` instead of the fitted
        ``sparse_output_`` attribute that newer versions expect during transform.
        """
        if ct is None:
            return ct
        if not hasattr(ct, 'sparse_output_'):
            if hasattr(ct, 'sparse_threshold'):
                ct.sparse_output_ = ct.sparse_threshold < 1.0
            else:
                ct.sparse_output_ = False
        return ct

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dict to restore encoders and configuration.
        
        Compatible with PyTorch Lightning's state management.
        
        Args:
            state_dict: Dictionary containing serialized state from state_dict().
        """
        self.feature_dims = state_dict.get('feature_dims', {})
        self._encoders_fitted = state_dict.get('_encoders_fitted', False)
        
        if 'category_pipeline' in state_dict:
            self.categorical_cols = state_dict.get('categorical_cols', [])
            self.category_pipeline = self._heal_column_transformer(
                pickle.loads(state_dict['category_pipeline'])
            )

        if 'numeric_pipeline' in state_dict:
            self.numerical_cols = state_dict.get('numerical_cols', [])
            self.numeric_pipeline = self._heal_column_transformer(
                pickle.loads(state_dict['numeric_pipeline'])
            )

        version = state_dict.get('state_dict_version', 1)
        if version >= 2:
            if 'context_numeric_pipeline' in state_dict:
                self.context_numerical_cols = state_dict.get('context_numerical_cols', [])
                self.context_numeric_pipeline = self._heal_column_transformer(
                    pickle.loads(state_dict['context_numeric_pipeline'])
                )
            if 'mol_numeric_pipeline' in state_dict:
                self.mol_numerical_cols = state_dict.get('mol_numerical_cols', [])
                self.mol_numeric_pipeline = self._heal_column_transformer(
                    pickle.loads(state_dict['mol_numeric_pipeline'])
                )
        elif 'numeric_pipeline' in state_dict:
            # Legacy state dict (version 1): reconstruct split pipelines from the
            # fitted combined pipeline without refitting.
            self._reconstruct_split_pipelines_from_legacy()

        # Restore label transformers
        if (self.normalize_labels or self.standardize_labels) and 'label_transformers' in state_dict:
            for key, bytes_data in state_dict['label_transformers'].items():
                # Pickle loads directly from bytes
                self.label_transformers[key] = pickle.loads(bytes_data)

        # Restore POI sequence embedding (TfidfVectorizer)
        if self.poi_features == "sequence" and 'poi_sequence_embedding_sklearn_encoder' in state_dict:
            if self.poi_sequence_embedding is not None:
                self.poi_sequence_embedding.sklearn_encoder = pickle.loads(state_dict['poi_sequence_embedding_sklearn_encoder'])
        
        # Restore tokenizer (Transformers are best loaded by name)
        if self.use_tokenizer and 'tokenizer_name' in state_dict:
            self.tokenizer_name = state_dict['tokenizer_name']
            self.max_length = state_dict.get('max_length', self.max_length)
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Restore PCA transformers
        if self.use_poi_pca and 'poi_pca' in state_dict:
            self.poi_pca = pickle.loads(state_dict['poi_pca'])
        
        if self.use_ligase_pca and 'ligase_pca' in state_dict:
            self.ligase_pca = pickle.loads(state_dict['ligase_pca'])
        
        self.logger.debug(f"State dict loaded via Pickle. Feature dims: {len(self.feature_dims)}, Encoders fitted: {self._encoders_fitted}")

    def _reconstruct_split_pipelines_from_legacy(self) -> None:
        """Reconstruct context/molecular numeric pipelines from a pre-split (v1) state dict.

        Older payloads only contain the combined ``numeric_pipeline``. Its
        fitted transformers are split by name — ``'treatment_time_pipeline'``
        is context, ``'descriptor_*'`` is molecular — to populate
        ``context_numeric_pipeline`` / ``mol_numeric_pipeline`` without
        needing to re-fit.
        """
        if self.numeric_pipeline is None or not hasattr(self.numeric_pipeline, 'transformers_'):
            return

        context_transformers = []
        mol_transformers = []
        self.context_numerical_cols = []
        self.mol_numerical_cols = []

        for trans_name, transformer, cols in self.numeric_pipeline.transformers_:
            if trans_name == 'remainder' or not cols:
                continue
            if trans_name == 'treatment_time_pipeline':
                context_transformers.append((trans_name, transformer, cols))
                self.context_numerical_cols.extend(cols)
            elif trans_name.startswith('descriptor_'):
                mol_transformers.append((trans_name, transformer, cols))
                self.mol_numerical_cols.extend(cols)

        if context_transformers:
            self.context_numeric_pipeline = ColumnTransformer(
                transformers=context_transformers, remainder='drop', sparse_threshold=0,
            )
            self.context_numeric_pipeline.transformers_ = context_transformers
            self._heal_column_transformer(self.context_numeric_pipeline)
        else:
            self.context_numeric_pipeline = None

        if mol_transformers:
            self.mol_numeric_pipeline = ColumnTransformer(
                transformers=mol_transformers, remainder='drop', sparse_threshold=0,
            )
            self.mol_numeric_pipeline.transformers_ = mol_transformers
            self._heal_column_transformer(self.mol_numeric_pipeline)
        else:
            self.mol_numeric_pipeline = None

        self.logger.debug(
            "Reconstructed split numeric pipelines from a legacy (version 1) state dict."
        )

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """PyTorch Lightning hook to save datamodule state to checkpoint.
        
        Args:
            checkpoint: The checkpoint dictionary.
        """
        checkpoint['datamodule_state_dict'] = self.state_dict()
    
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """PyTorch Lightning hook to load datamodule state from checkpoint.
        
        Args:
            checkpoint: The checkpoint dictionary.
        """
        if 'datamodule_state_dict' in checkpoint:
            self.load_state_dict(checkpoint['datamodule_state_dict'])

    def _get_protein_id(self, example: Union[Dict, pd.Series], protein_type: Literal['poi', 'ligase']) -> str:
        """Get the appropriate protein identifier for embedding lookup.
        
        Args:
            example: Data sample
            protein_type: 'poi' or 'ligase'
            
        Returns:
            Protein identifier (sequence or UniProt ID)
        """
        if protein_type == 'poi':
            id_type = self.poi_embeddings_id_type
            seq_col = self.poi_sequence_col
            uniprot_col = 'POI_UniProt'
        else:  # ligase
            id_type = self.ligase_embeddings_id_type
            seq_col = self.ligase_sequence_col
            uniprot_col = 'Ligase_UniProt'
        
        if id_type == 'sequence':
            protein_id = example.get(seq_col)
            if protein_id is None or (isinstance(protein_id, float) and np.isnan(protein_id)):
                raise ValueError(f"Sequence column '{seq_col}' is missing or NaN for {protein_type}")
            return protein_id
        elif id_type == 'uniprot':
            protein_id = example.get(uniprot_col)
            if protein_id is None or (isinstance(protein_id, float) and np.isnan(protein_id)):
                raise ValueError(f"UniProt column '{uniprot_col}' is missing or NaN for {protein_type}")
            return protein_id
        else:
            raise ValueError(f"Invalid protein ID type: {id_type}")

    def augment_sample(
        self,
        example: Dict[str, Any],
        rel_std: float = 0.05,
        is_log_scale: bool = False,
        ignore_null_operators: bool = True,
    ) -> Dict[str, Any]:
        """Jitter label values in-place to augment training data.

        Applies relative Gaussian noise to each label, then clips the result to
        respect the measurement operator (``>``, ``<``, ``>=``, ``<=``, ``~``).
        Log-space jitter is used when ``is_log_scale=True`` to keep noise
        scale-invariant across orders of magnitude.

        Args:
            example: Sample dict; mutated in-place and returned.
            rel_std: Relative standard deviation of the noise (default 5%).
            is_log_scale: Apply noise in log10-space instead of linear space.
            ignore_null_operators: Skip samples with no operator column.

        Returns:
            The mutated example dict.
        """
        for label in self.labels:
            if f'{label}_Operator' not in example:
                if ignore_null_operators: continue
                operator = '~'
            else:
                operator = example[f'{label}_Operator']
                
            value = example[label]
            if pd.isnull(value) or (pd.isnull(operator) and ignore_null_operators):
                continue

            # 1. Determine the "jitter" magnitude
            # We use a percentage of the current value to keep it scale-invariant
            # Adding a tiny floor to avoid math errors on zero values
            if is_log_scale and value > 0:
                # Apply noise in log-space: log(v) + noise
                log_val = np.log10(value)
                # 0.02 shift in log10 space is roughly a 4.7% change in linear space
                perturbed_log = log_val + np.random.normal(0, rel_std * 0.5)
                new_value = 10**perturbed_log
            else:
                # Standard relative noise
                noise_factor = np.random.normal(0, rel_std)
                new_value = value * (1 + noise_factor)

            # 2. Apply Operator Logic
            # Instead of fixed 5%, we ensure the value moves in the 'safe' direction
            # but stays probabilistically close to the original.
            shift = abs(np.random.normal(rel_std, rel_std * 0.5)) # Positive shift
            
            if operator == '>':
                # Ensure it is at least slightly larger than the original
                final_value = max(value, new_value) * (1 + shift)
            elif operator == '<':
                # Ensure it is at least slightly smaller
                final_value = min(value, new_value) * (1 - shift)
            elif operator == '>=':
                final_value = new_value if new_value >= value else value
            elif operator == '<=':
                final_value = new_value if new_value <= value else value
            else: # '~' or null
                final_value = new_value

            example[label] = final_value

        return example

    def oversample_uniform(self, dataset: TorchListDataset, label: str, n_bins: int = 20) -> TorchListDataset:
        """ Oversample the training dataset to make the label distributions
        more uniform. This is currently not working properly and it is not used.
        """
        col_data = dataset[label]
        values = np.array([t.item() if isinstance(t, torch.Tensor) else t for t in col_data])
        mask = ~np.isnan(values)
        clean_values = values[mask]
        
        # Use equal width bins (linspace) to isolate sparse regions
        bins = np.linspace(clean_values.min(), clean_values.max(), n_bins + 1)
        
        # Map values to bin indices and clip to stay within [0, n_bins-1]
        bin_indices = np.digitize(values, bins, right=False) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        all_indices = np.arange(len(values))
        valid_indices = all_indices[mask]
        valid_bin_indices = bin_indices[mask]

        # Count samples in each bin
        bin_counts = np.bincount(valid_bin_indices, minlength=n_bins)
        max_count = bin_counts.max()

        new_indices = []
        for i in range(n_bins):
            idx_in_bin = valid_indices[valid_bin_indices == i]
            if len(idx_in_bin) == 0:
                continue
                
            # Add original indices for this bin
            new_indices.extend(idx_in_bin.tolist())
            
            # Oversample by picking random indices from this bin until reaching max_count
            reps = max_count - len(idx_in_bin)
            if reps > 0:
                new_indices.extend(np.random.choice(idx_in_bin, size=reps, replace=True))

        return dataset.select(new_indices)

    def __str__(self) -> str:
        """ Return a unique string representation of the data module configuration.
        
        This string can be used as a unique identifier for logging and checkpointing.
        """
        parts = []
        
        # Add feature flags
        if self.mol_features and "fingerprint" in self.mol_features:
            parts.append(f'fp{getattr(self, "fp_size", 512)}r{getattr(self, "radius", 16)}')
        if self.mol_features and "descriptors" in self.mol_features:
            if getattr(self, 'use_relevant_descriptors', False):
                parts.append('desc')
            elif getattr(self, 'selected_descriptors', False):
                parts.append('sel_desc')
            else:
                parts.append('all_desc')
        if getattr(self, 'use_tokenizer', False):
            tokenizer = getattr(self, 'tokenizer_name', 'bert')
            parts.append(f'tok_{tokenizer.split("/")[-1].replace("-", "_")}')
        if self.poi_features == "name":
            if self.categorical_encoding == 'embedding':
                parts.append('poi_pt')
            elif self.categorical_encoding == 'onehot':
                parts.append('poi_onehot')
            else:
                parts.append('poi_ord')
        if self.poi_features == "sequence":
            parts.append('poi_vec')
        if self.poi_features == "precomputed":
            parts.append('poi_emb')
        if self.ligase_features == "precomputed":
            parts.append('lig_emb')
        if self.ligase_features == "name":
            if self.categorical_encoding == 'embedding':
                parts.append('lig_pt')
            elif self.categorical_encoding == 'onehot':
                parts.append('lig_onehot')
            else:
                parts.append('lig_ord')
        if self.cell_features == "name":
            if self.categorical_encoding == 'embedding':
                parts.append('cell_pt')
            elif self.categorical_encoding == 'onehot':
                parts.append('cell_onehot')
            else:
                parts.append('cell_ord')
        if self.cell_features == "description":
            parts.append('cell_text')
        if getattr(self, 'use_assay_type_encoding', False):
            parts.append('assay')
        if getattr(self, 'use_treatment_time', False):
            parts.append('time')
        if getattr(self, 'normalize_labels', False):
            parts.append('norm')
        if getattr(self, 'standardize_labels', False):
            parts.append('std')
        if getattr(self, 'use_poi_pca', False):
            parts.append(f'poi_pca{getattr(self, "poi_pca_n_components", 0.95)}')
        if getattr(self, 'use_ligase_pca', False):
            parts.append(f'lig_pca{getattr(self, "ligase_pca_n_components", 0.95)}')
        
        # Add labels
        labels = getattr(self, 'labels', ['Value'])
        parts.append(f'labels_{"_".join(labels)}')
        
        return '_'.join(parts) if parts else 'default_config'

    def get_hyperparameters(self) -> Dict[str, Any]:
        """Return a dictionary of hyperparameters for logging."""
        return dict(self.hparams)

    @staticmethod
    def convert_dc50_to_pdc50(dc50: float, unit: Literal['nM', 'uM', 'pM', 'M'] = 'nM') -> float:
        """ Convert DC50 in nano Molar to pDC50 (-log10(M)). """
        if pd.isnull(dc50) or dc50 <= 0:
            return np.nan
        unit_factors = {
            'pM': 1e-12,
            'nM': 1e-9,
            'uM': 1e-6,
            'M': 1.0,
        }
        if unit not in unit_factors:
            raise ValueError(f"Invalid unit '{unit}'. Must be one of {list(unit_factors.keys())}.")
        factor = unit_factors.get(unit, 1e-9)
        return -np.log10(dc50 * factor + 1e-20)  # small offset to avoid log10(0)


def load_datamodule(
    state_dict_path: Union[Path, str],
    hparams_path: Optional[Union[Path, str]] = None,
    hparam_overrides: Optional[Dict[str, Any]] = None,
) -> DegradationComplexDataModule:
    """Load a DegradationComplexDataModule from a saved state dict.

    Hyperparameters are sourced in priority order:
    1. ``hparam_overrides`` — any key here wins over everything else.
    2. ``hparams_path`` YAML file — used when provided.
    3. ``hparams`` key embedded in the state dict — used when the YAML file is absent.

    A ``FileNotFoundError`` is raised if neither a YAML file nor an embedded
    ``hparams`` key is available.

    Args:
        state_dict_path: Path to the saved ``.pt`` state dict file.
        hparams_path: Optional path to a YAML file with hyperparameters.
            If ``None``, the state dict must contain an ``'hparams'`` key.
        hparam_overrides: Optional dict of hparam key/value pairs that
            override whatever is loaded from the YAML or the state dict.
            Use this to fix stale paths, e.g.::

                load_datamodule(
                    'model_state.pt',
                    hparam_overrides={
                        'poi_embeddings_file': '/new/path/embeddings.npz',
                        'ligase_embeddings_file': '/new/path/embeddings.npz',
                    },
                )

    Returns:
        A :class:`DegradationComplexDataModule` with all transformers restored.
    """
    state_path = Path(state_dict_path)
    state_dict = torch.load(state_path, map_location='cpu', weights_only=False)

    if hparams_path is not None:
        hparams = load_config_from_yaml(Path(hparams_path))
    elif 'hparams' in state_dict:
        hparams = dict(state_dict['hparams'])
    else:
        raise FileNotFoundError(
            f"No hparams_path was provided and the state dict at '{state_path}' "
            "does not contain an 'hparams' key. Pass hparams_path explicitly."
        )

    if hparam_overrides:
        hparams.update(hparam_overrides)

    dm = DegradationComplexDataModule(dataset=None, **hparams)
    dm.load_state_dict(state_dict)

    # Check if any of the non-None encoders are fitted, this will raise an error
    # if they are not have been fitted, and so that the checkpoints are broken.
    if dm.category_pipeline is not None:
        check_is_fitted(dm.category_pipeline)
    if dm.numeric_pipeline is not None:
        check_is_fitted(dm.numeric_pipeline)
    if dm.context_numeric_pipeline is not None:
        check_is_fitted(dm.context_numeric_pipeline)
    if dm.mol_numeric_pipeline is not None:
        check_is_fitted(dm.mol_numeric_pipeline)
    for transformer in dm.label_transformers.values():
        check_is_fitted(transformer)
    if dm.use_poi_pca and dm.poi_pca is not None:
        check_is_fitted(dm.poi_pca)
    if dm.use_ligase_pca and dm.ligase_pca is not None:
        check_is_fitted(dm.ligase_pca)
    if dm.poi_sequence_embedding is not None:
        if dm.poi_features == "sequence":
            check_is_fitted(dm.poi_sequence_embedding.sklearn_encoder)

    return dm