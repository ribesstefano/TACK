""" DataModule for BERT regression tasks using PyTorch Lightning."""
import os
import logging
import pickle
from pathlib import Path
from typing import List, Optional, Union, Any, Dict, Literal, Tuple

import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import xgboost as xgb
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


class DegradationComplexDataModule(pl.LightningDataModule):
    
    """ Wrapper module to handle data loading and featurization for the TACK
    dataset.
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
        # Molecular embeddings parameters
        fp_size: int = 512,
        radius: int = 16,
        use_fingerprints: bool = False,
        use_descriptors: bool = False,
        use_relevant_descriptors: bool = False,
        selected_descriptors: Optional[List[str]] = None,
        # Protein embedding flags
        use_poi_sequence_embedding: bool = False,  # POI sequence -> amino acid count
        use_poi_name_embedding: bool = False,     # POI name -> ordinal encoding
        use_ligase_name_embedding: bool = False,   # Ligase name -> ordinal encoding (via categorical)
        use_poi_precomputed_embedding: bool = False,  # POI -> precomputed embeddings
        use_ligase_precomputed_embedding: bool = False,  # Ligase -> precomputed embeddings
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
        # Cell line embedding flags
        use_cell_description_embedding: bool = False,   # Cell line -> sentence transformer
        use_cell_name_embedding: bool = False,      # Cell line -> ordinal encoding
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
        sort_features: bool = True,
        categorical_encoding: Literal['minmax', 'onehot', 'embedding'] = 'minmax',
    ):
        super().__init__()
        # Exclude 'dataset' and 'hf_token' from hyperparameters since they shouldn't be serialized
        self.save_hyperparameters(ignore=['dataset', 'hf_token'])
        
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
        
        # Feature inclusion flags
        self.use_fingerprints = use_fingerprints
        self.use_descriptors = use_descriptors
        self.use_poi_sequence_embedding = use_poi_sequence_embedding
        self.use_poi_name_embedding = use_poi_name_embedding
        self.use_ligase_name_embedding = use_ligase_name_embedding
        self.use_poi_precomputed_embedding = use_poi_precomputed_embedding
        self.use_ligase_precomputed_embedding = use_ligase_precomputed_embedding
        self.use_cell_description_embedding = use_cell_description_embedding
        self.use_cell_name_embedding = use_cell_name_embedding
        self.use_treatment_time = use_treatment_time
        self.include_prompt = include_prompt
        self.default_degrader_type = default_degrader_type
        self.normalize_labels = normalize_labels
        self.standardize_labels = standardize_labels
        self.impute_labels = impute_labels
        self.use_assay_type_encoding = use_assay_type_encoding
        self.sort_features = sort_features
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
        ) if use_fingerprints else None

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
        ) if use_descriptors else None

        # POI sequence embedding (amino acid count)
        self.poi_sequence_embedding = ProteinEmbedding(
            embeddings_type="amino_acid_count",
            load_from_cache=False,
            filename="protein_embeddings_amino_acid_count.npz",
        ) if use_poi_sequence_embedding else None
        
        # POI precomputed embedding
        self.poi_precomputed_embedding = ProteinEmbedding(
            embeddings_type="precomputed",
            embeddings_file=poi_embeddings_file,
            embeddings_format=poi_embeddings_format,
            embeddings_per_residue=poi_embeddings_per_residue,
            residue_pooling=poi_residue_pooling,
            load_from_cache=False,
        ) if use_poi_precomputed_embedding else None
        
        # Ligase precomputed embedding
        self.ligase_precomputed_embedding = ProteinEmbedding(
            embeddings_type="precomputed",
            embeddings_file=ligase_embeddings_file,
            embeddings_format=ligase_embeddings_format,
            embeddings_per_residue=ligase_embeddings_per_residue,
            residue_pooling=ligase_residue_pooling,
            load_from_cache=False,
        ) if use_ligase_precomputed_embedding else None
        
        # Cell line description embedding (sentence transformer)
        self.cell_description_embedding = CellEmbedding(
            embeddings_type="sentence_transformer",
            pooling="sum",
            load_from_cache=True,
            filename="cell_embeddings_model=sentence-transformer_pooling=sum.npz",
        ) if use_cell_description_embedding else None
        
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
        
        if not use_tokenizer and not use_fingerprints and not use_descriptors:
            raise ValueError("At least one of use_fingerprints, use_descriptors, or use_tokenizer must be True.")
        
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
        self._create_numeric_pipeline()
        
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
        # Download dataset (only called on one GPU in distributed training)
        pass

    def setup(self, stage: Optional[str] = None):
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
            if self.use_poi_sequence_embedding and self.poi_sequence_embedding is not None:
                poi_seqs = list(set(self.dataset["train"][self.poi_sequence_col]))
                self.logger.debug(f"Fitting POI sequence encoder on {len(poi_seqs)} sequences: {poi_seqs[:5]}...")
                self.poi_sequence_embedding.fit(poi_seqs)
                self.logger.debug("POI sequence encoder fitted on training data")
                
            # Collect SMILES for molecular embeddings
            smiles_list = list(set(self.dataset["train"][self.smiles_col]))

            # Fit molecular embeddings
            if self.use_fingerprints:
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
            if self.use_cell_description_embedding and self.cell_description_embedding is not None:
                self.logger.debug(f"Fitting cell description embedding on {len(cell_lines)} cell lines: {cell_lines[:5]}...")
                self.cell_description_embedding.transform(cell_lines, update_cache=True)
                self.logger.debug("Cell description embedding fitted on training data")
            
            # Fit categorical and numeric pipelines
            self._fit_category_pipeline(train_df)
            self._fit_numeric_pipeline(train_df)

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

    def featurize_sample(self, example: Union[Dict, pd.Series], return_tensor: str = 'np') -> Dict[str, Any]:
        """ Convert a single example to a numerical feature vector.
        
        Args:
            example: A dictionary or pandas Series representing a single data point.

        Returns:
            A dictionary mapping feature names to their numerical values.
        """
        features = {}
        
        # SMILES -> Morgan fingerprint and/or RDKit descriptors
        if self.use_fingerprints and self.fp_embedder is not None:
            features['Feature_Fingerprint'] = self.fp_embedder.transform(example[self.smiles_col])
        
        # Cell line description embedding
        if self.use_cell_description_embedding and self.cell_description_embedding is not None:
            features[f'Feature_{self.cell_line_col}_Description'] = self.cell_description_embedding.transform(example[self.cell_line_col])
        
        # POI sequence embedding
        if self.use_poi_sequence_embedding and self.poi_sequence_embedding is not None:
            features[f'Feature_{self.poi_sequence_col}'] = self.poi_sequence_embedding.transform(example[self.poi_sequence_col])
        
        # POI precomputed embedding
        if self.use_poi_precomputed_embedding and self.poi_precomputed_embedding is not None:
            poi_id = self._get_protein_id(example, 'poi')
            poi_embedding = self.poi_precomputed_embedding.transform(poi_id)
            
            # Apply PCA if enabled
            if self.use_poi_pca and self.poi_pca is not None:
                poi_embedding = self.poi_pca.transform(poi_embedding.reshape(1, -1)).flatten()
            
            features[f'Feature_POI_Precomputed_Embedding'] = poi_embedding
        
        # Ligase precomputed embedding
        if self.use_ligase_precomputed_embedding and self.ligase_precomputed_embedding is not None:
            ligase_id = self._get_protein_id(example, 'ligase')
            ligase_embedding = self.ligase_precomputed_embedding.transform(ligase_id)
            
            # Apply PCA if enabled
            if self.use_ligase_pca and self.ligase_pca is not None:
                ligase_embedding = self.ligase_pca.transform(ligase_embedding.reshape(1, -1)).flatten()
            
            features[f'Feature_Ligase_Precomputed_Embedding'] = ligase_embedding
 
        features.update(self._run_category_pipeline(example))
        features.update(self._run_numeric_pipeline(example))
        
        # Add tokenized features if using tokenizer
        if self.use_tokenizer:
            tokenized = self._tokenize_sample(example)
            # Conditionally include 'Prompt'
            if not self.include_prompt and 'Prompt' in tokenized:
                del tokenized['Prompt']
            features.update(tokenized)

        # NOTE: Labels are NOT normalized here - they are normalized in
        # featurize_dataset to ensure consistency and proper handling of the
        # Value_Type for BERT multi-task

        if return_tensor == 'np':
            # Convert all features to numpy arrays
            for key, value in features.items():
                if isinstance(value, torch.Tensor):
                    features[key] = value.numpy()
                elif not isinstance(value, np.ndarray):
                    features[key] = np.array(value)
        elif return_tensor == 'pt':
            # Convert all features to PyTorch tensors
            for key, value in features.items():
                if isinstance(value, np.ndarray):
                    features[key] = torch.tensor(value)
                else:
                    features[key] = torch.tensor(np.array(value))
        elif return_tensor == 'xgb':
            # Use feature_dims order if available (matches training), else fall back to sorted
            if self.feature_dims:
                feature_order = [k for k in self.feature_dims if k in features]
            else:
                feature_order = sorted(features.keys())

            # Concatenate all features into a single 1D array for XGBoost
            feature_list = []
            for key in feature_order:
                value = features[key]
                if isinstance(value, torch.Tensor):
                    value = value.numpy()
                feature_list.append(value.flatten())
            
            # We return a tuple of (features, feature_names) for XGBoost to keep track of feature names                
            if not self.feature_dims: 
                feature_names = [
                    f"{key}_{i}"
                    for key in feature_order
                    for i in range(features[key].flatten().shape[0])
                ]
            else:
                feature_names = self.get_xgboost_feature_names()  # Ensure feature names are initialized
            features = (
                np.concatenate(feature_list).astype(np.float32),
                feature_names
            )   
        else:
            raise ValueError(f"Invalid return_tensor value: {return_tensor}. Must be 'np' (NumPy), 'pt' (PyTorch), or 'xgb' (XGBoost, i.e., flattened array).")
        return features

    def featurize_samples_batch(
        self,
        examples: List[Dict],
        return_tensor: Literal['np', 'pt', 'xgb'] = 'np',
        shared_cache: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Batch-featurize multiple samples efficiently.

        Instead of building a 1-row DataFrame per sample for the sklearn
        pipelines, this method:
        1. Computes raw embeddings once per unique input value (with optional
           cross-model shared cache).
        2. Runs the category and numeric sklearn pipelines on **all samples
           at once** (one DataFrame, one ``.transform()`` call each).
        3. Applies PCA in batch (one ``pca.transform(matrix)`` call).

        Args:
            examples: List of sample dicts (column-name → value).
            return_tensor: Output format per sample ('np', 'pt', 'xgb').
            shared_cache: Optional dict ``{(embedder_key, input_value): embedding}``
                shared across datamodules so that identical raw embeddings
                are computed only once in an ensemble.

        Returns:
            List of feature dicts (same structure as ``featurize_sample``).
            For ``return_tensor='xgb'`` each entry is a ``(flat_array, feature_names)``
            tuple.
        """
        if shared_cache is None:
            shared_cache = {}

        n = len(examples)
        if n == 0:
            return []

        # ------------------------------------------------------------------
        # 1. Raw embeddings — compute once per unique value, using shared_cache
        # ------------------------------------------------------------------
        fp_results: Dict[str, np.ndarray] = {}
        cell_text_results: Dict[str, np.ndarray] = {}
        poi_vec_results: Dict[str, np.ndarray] = {}
        poi_precomp_results: Dict[str, np.ndarray] = {}
        ligase_precomp_results: Dict[str, np.ndarray] = {}

        # Fingerprints
        if self.use_fingerprints and self.fp_embedder is not None:
            unique_smiles = list({ex[self.smiles_col] for ex in examples})
            for smi in unique_smiles:
                cache_key = ('fp', self.radius, self.fp_size, smi)
                if cache_key in shared_cache:
                    fp_results[smi] = shared_cache[cache_key]
                else:
                    emb = self.fp_embedder.transform(smi)
                    shared_cache[cache_key] = emb
                    fp_results[smi] = emb

        # Cell description embedding
        if self.use_cell_description_embedding and self.cell_description_embedding is not None:
            unique_cells = list({ex[self.cell_line_col] for ex in examples})
            for cell in unique_cells:
                cache_key = ('cell_text', cell)
                if cache_key in shared_cache:
                    cell_text_results[cell] = shared_cache[cache_key]
                else:
                    emb = self.cell_description_embedding.transform(cell)
                    shared_cache[cache_key] = emb
                    cell_text_results[cell] = emb

        # POI sequence embedding (amino acid count / tfidf)
        # NOTE: keyed by id(embedder) because TfidfVectorizer vocabulary differs per fold
        if self.use_poi_sequence_embedding and self.poi_sequence_embedding is not None:
            emb_id = id(self.poi_sequence_embedding)
            unique_seqs = list({ex[self.poi_sequence_col] for ex in examples})
            for seq in unique_seqs:
                cache_key = ('poi_vec', emb_id, seq)
                if cache_key in shared_cache:
                    poi_vec_results[seq] = shared_cache[cache_key]
                else:
                    emb = self.poi_sequence_embedding.transform(seq)
                    shared_cache[cache_key] = emb
                    poi_vec_results[seq] = emb

        # POI precomputed embedding (raw, before PCA)
        if self.use_poi_precomputed_embedding and self.poi_precomputed_embedding is not None:
            unique_poi_ids = list({self._get_protein_id(ex, 'poi') for ex in examples})
            for pid in unique_poi_ids:
                cache_key = ('poi_precomp', pid)
                if cache_key in shared_cache:
                    poi_precomp_results[pid] = shared_cache[cache_key]
                else:
                    emb = self.poi_precomputed_embedding.transform(pid)
                    shared_cache[cache_key] = emb
                    poi_precomp_results[pid] = emb

        # Ligase precomputed embedding (raw, before PCA)
        if self.use_ligase_precomputed_embedding and self.ligase_precomputed_embedding is not None:
            unique_lig_ids = list({self._get_protein_id(ex, 'ligase') for ex in examples})
            for lid in unique_lig_ids:
                cache_key = ('lig_precomp', lid)
                if cache_key in shared_cache:
                    ligase_precomp_results[lid] = shared_cache[cache_key]
                else:
                    emb = self.ligase_precomputed_embedding.transform(lid)
                    shared_cache[cache_key] = emb
                    ligase_precomp_results[lid] = emb

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
            # Build the DataFrame column-by-column from pre-allocated matrices
            # instead of building per-sample dicts. For ~1000 descriptors this
            # avoids millions of tiny numpy-array allocations.
            num_data: Dict[str, Any] = {}
            if self.use_treatment_time:
                num_data[self.treatment_time_col] = [ex[self.treatment_time_col] for ex in examples]
            if self.use_descriptors:
                desc_names = self.desc_embedder.get_descriptor_names()
                desc_matrix = np.empty((n, len(desc_names)), dtype=np.float32)
                for i, ex in enumerate(examples):
                    smi = ex[self.smiles_col]
                    cache_key = ('desc_raw', id(self.desc_embedder), smi)
                    if cache_key in shared_cache:
                        descs = shared_cache[cache_key]
                    else:
                        descs = self.desc_embedder.transform(smi)
                        shared_cache[cache_key] = descs
                    desc_matrix[i] = descs
                for j, name in enumerate(desc_names):
                    num_data[f'Descriptor_{name}'] = desc_matrix[:, j]
            num_df = pd.DataFrame(num_data)
            num_results = self.numeric_pipeline.transform(num_df)  # shape (n, n_num_features)

        # Precompute per-column output widths for the category pipeline so
        # that the sample loop can slice cat_results correctly (one-hot
        # encoding produces multiple output columns per input column).
        cat_col_offsets: List[tuple] = []  # [(offset, width), ...]
        if cat_results is not None:
            offset = 0
            for _, inner, cols in self.category_pipeline.transformers_:
                width = inner.transform(cat_df[cols].iloc[:1]).shape[1]
                cat_col_offsets.append((offset, width))
                offset += width

        # ------------------------------------------------------------------
        # 5. Assemble results
        # ------------------------------------------------------------------

        # Fast path for Lightning models: build one batched tensor dict directly
        # from the already-computed matrices, skipping the per-sample loop and
        # the re-stacking that _predict_lightning_batch would otherwise do.
        if return_tensor == 'pt':
            batch_out: Dict[str, torch.Tensor] = {}

            if self.use_fingerprints and self.fp_embedder is not None:
                fp_mat = np.stack([fp_results[ex[self.smiles_col]] for ex in examples])
                batch_out['Feature_Fingerprint'] = torch.from_numpy(fp_mat).float()

            if self.use_cell_description_embedding and self.cell_description_embedding is not None:
                cell_mat = np.stack([cell_text_results[ex[self.cell_line_col]] for ex in examples])
                batch_out[f'Feature_{self.cell_line_col}_Description'] = torch.from_numpy(cell_mat).float()

            if self.use_poi_sequence_embedding and self.poi_sequence_embedding is not None:
                poi_mat = np.stack([poi_vec_results[ex[self.poi_sequence_col]] for ex in examples])
                batch_out[f'Feature_{self.poi_sequence_col}'] = torch.from_numpy(poi_mat).float()

            if self.use_poi_precomputed_embedding and self.poi_precomputed_embedding is not None:
                emb_res = poi_pca_results if (self.use_poi_pca and self.poi_pca is not None and poi_pca_results) else poi_precomp_results
                poi_mat = np.stack([emb_res[self._get_protein_id(ex, 'poi')] for ex in examples])
                batch_out['Feature_POI_Precomputed_Embedding'] = torch.from_numpy(poi_mat).float()

            if self.use_ligase_precomputed_embedding and self.ligase_precomputed_embedding is not None:
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

            return batch_out

        # ------------------------------------------------------------------
        # Per-sample assembly for 'np', 'pt', and 'xgb' modes
        # ------------------------------------------------------------------
        batch_features: List[Dict[str, Any]] = []
        for i, ex in enumerate(examples):
            features: Dict[str, Any] = {}

            if self.use_fingerprints and self.fp_embedder is not None:
                features['Feature_Fingerprint'] = fp_results[ex[self.smiles_col]]

            if self.use_cell_description_embedding and self.cell_description_embedding is not None:
                features[f'Feature_{self.cell_line_col}_Description'] = cell_text_results[ex[self.cell_line_col]]

            if self.use_poi_sequence_embedding and self.poi_sequence_embedding is not None:
                features[f'Feature_{self.poi_sequence_col}'] = poi_vec_results[ex[self.poi_sequence_col]]

            if self.use_poi_precomputed_embedding and self.poi_precomputed_embedding is not None:
                pid = self._get_protein_id(ex, 'poi')
                if self.use_poi_pca and self.poi_pca is not None:
                    features['Feature_POI_Precomputed_Embedding'] = poi_pca_results[pid]
                else:
                    features['Feature_POI_Precomputed_Embedding'] = poi_precomp_results[pid]

            if self.use_ligase_precomputed_embedding and self.ligase_precomputed_embedding is not None:
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
            elif return_tensor == 'pt':
                for key, value in features.items():
                    if isinstance(value, np.ndarray):
                        features[key] = torch.tensor(value)
                    else:
                        features[key] = torch.tensor(np.array(value))
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
                    feature_names = self.get_xgboost_feature_names()
                features = (
                    np.concatenate(feature_list).astype(np.float32),
                    feature_names,
                )

            batch_features.append(features)

        return batch_features
    
    def featurize_dataset(
            self,
            dataset: Union[Dataset, List[Dict[str, Any]]],
            split_name: str = "unknown",
            num_proc: Optional[int] = None,
    ) -> Dataset:
        """ Featurize an entire dataset split and extract labels.
        
        Args:
            dataset: The dataset to featurize.
            split_name: Name of the split (for storing tasks).
            
        Returns:
            Featurized dataset with torch format.
        """
        features_list = []
        tasks_list = []
        
        if num_proc is None or self.num_proc <= 1:
            for i in range(len(dataset)):
                sample = dataset[i]
                features = self.featurize_sample(sample)
                features.update(self._featurize_and_normalize_labels(sample))
                features_list.append(features)
                
                # Store task for BERT multi-task
                if self.is_bert_multitask and self.label_task_col in sample:
                    tasks_list.append(sample[self.label_task_col])
            
            ds = Dataset.from_list(features_list)
        else:
            cols_to_remove = [c for c in dataset.column_names if not c.startswith('Feature_') and c not in self.labels]
            # Also remove Task column if BERT multi-task
            if self.is_bert_multitask:
                cols_to_remove.append('Task')
            
            # Use dataset.map for multiprocessing
            def featurize_and_normalize_labels_map(example):
                features = self.featurize_sample(example)
                features.update(self._featurize_and_normalize_labels(example))
                # Store task for BERT multi-task
                if self.is_bert_multitask and self.label_task_col in example:
                    return {**features, 'Task': example[self.label_task_col]}
                return features
            
            ds = dataset.map(
                featurize_and_normalize_labels_map,
                num_proc=num_proc,
            )
            if self.is_bert_multitask:
                tasks_list = ds['Task']
            ds = ds.remove_columns(cols_to_remove)
        
        self.logger.debug(f"Featurized dataset shape: {ds.num_rows} samples")
        self.logger.debug(f"Dataset columns: {ds.column_names}")
        
        # If the number of labels is 1, remove samples with NaN labels
        if len(self.labels) == 1 and not self.impute_labels:
            label = self.labels[0]
            initial_size = ds.num_rows
            
            # Filter and also filter tasks_list correspondingly
            non_nan_mask = [pd.notnull(ds[i][label]) for i in range(len(ds))]
            ds = ds.filter(lambda x: pd.notnull(x[label]))
            
            if tasks_list:
                tasks_list = [t for t, keep in zip(tasks_list, non_nan_mask) if keep]
            
            self.logger.debug(f"Removed {initial_size - ds.num_rows} samples with NaN labels")
        
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
            
        return ds.with_format("torch")

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
        
        if self.sort_features:
            self.feature_dims = dict(sorted(self.feature_dims.items()))

        self.logger.debug(f"Feature dimensions initialized: {self.feature_dims}")
    
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

    def _create_category_pipeline(self):
        """ Create sklearn pipeline for categorical feature preprocessing."""
        transformers = []
        
        # Ordinal encoding + shift + MinMax for each categorical feature
        self.categorical_cols = []
        if self.use_ligase_name_embedding:
            self.categorical_cols.append(self.ligase_col)
        if self.use_poi_name_embedding:
            self.categorical_cols.append(self.poi_col)
        if self.use_cell_name_embedding:
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
                    ('ordinal', OrdinalEncoder(
                        handle_unknown='use_encoded_value',
                        unknown_value=-1,
                        dtype=np.int64
                    )),
                ])
            elif self.categorical_encoding == 'onehot':
                inner = Pipeline([
                    ('onehot', OneHotEncoder(
                        handle_unknown='ignore',
                        sparse_output=False
                    )),
                ])
            else:
                # Default 'minmax' mode: ordinal + MinMax scaling to [0, 1]
                inner = Pipeline([
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

    def _create_numeric_pipeline(self):
        """ Create sklearn pipeline for numeric feature preprocessing."""
        transformers = []
        self.numerical_cols = []
        
        # SimpleImputer and StandardScaler for treatment time
        if self.use_treatment_time:
            self.numerical_cols.append(self.treatment_time_col)
            default_treatment_time = 24  # hours
            # NOTE: The second imputer is to handle the case where all values
            # are NaN
            transformers.append((
                'treatment_time_pipeline',
                Pipeline([
                    ('mean_imputer', SimpleImputer(strategy='mean', keep_empty_features=True)),
                    ('const_imputer', SimpleImputer(strategy='constant', fill_value=default_treatment_time, keep_empty_features=True)),
                    ('scaler', StandardScaler())
                ]),
                [self.treatment_time_col]
            ))
        
        # SimpleImputer and StandardScaler for descriptors
        if self.use_descriptors:
            for desc_name in self.desc_embedder.get_descriptor_names():
                self.numerical_cols.append(f'Descriptor_{desc_name}')
                transformers.append((
                    f'descriptor_{desc_name}_pipeline',
                    StandardScaler(),
                    [f'Descriptor_{desc_name}']
                ))
        
        self.numeric_pipeline = None
        if transformers:
            self.numeric_pipeline = ColumnTransformer(
                transformers=transformers,
                remainder='drop',
                sparse_threshold=0,
            )

    def _fit_numeric_pipeline(self, train_df: pd.DataFrame) -> None:
        """ Fit numeric pipeline on training data.
        
        Args:
            train_df: Training dataframe.
        """
        if self.numeric_pipeline is None:
            return
        
        # Get all numeric data into one DataFrame, then fit the pipeline
        data = {}
        if self.use_treatment_time:
            data[self.treatment_time_col] = train_df[self.treatment_time_col]
        
        if self.use_descriptors:
            descs_list = []
            for _, row in train_df.iterrows():
                # Each descriptor is an array of shape: (num_descriptors,)
                descs = self.desc_embedder.transform(row[self.smiles_col])
                descs_list.append(descs)
            descs_array = np.array(descs_list, dtype=np.float32).T  # Shape: (num_descriptors, num_samples)
            for i, name in enumerate(self.desc_embedder.get_descriptor_names()):
                data[f'Descriptor_{name}'] = descs_array[i]
        
        numeric_df = pd.DataFrame(data)
        self.numeric_pipeline.fit(numeric_df)
        self.logger.debug("Numeric pipeline fitted on training data")

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
        
        if self.use_descriptors:
            descs = self.desc_embedder.transform(example[self.smiles_col])
            for i, name in enumerate(self.desc_embedder.get_descriptor_names()):
                data[f'Descriptor_{name}'] = np.array([descs[i]], dtype=np.float32)

        transformed = self.numeric_pipeline.transform(pd.DataFrame([data]))
        
        result = {}
        for i, col in enumerate(self.numerical_cols):
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
    
    # def _create_label_pipeline(self, target_type: str = 'Dmax') -> Pipeline:
    #     """ Create pipeline for label normalization.
        
    #     Args:
    #         target_type: 'Dmax' or 'DC50'
            
    #     Returns:
    #         An sklearn Pipeline for label transformation.
    #     """
    #     if target_type == 'Dmax':
    #         # STRATEGY 1: Dmax (Logit Transform)
    #         # Transforms skewed 0-100% data into a bell-curve shape
    #         return Pipeline([
    #             ('logit_transform', FunctionTransformer(
    #                 func=_dmax_logit_forward,
    #                 inverse_func=_dmax_logit_inverse,
    #                 validate=True,
    #                 check_inverse=False # often safer to disable strict checking for floats
    #             )),
    #             ('scaler', StandardScaler())
    #         ])
    #     elif target_type == 'DC50':
    #         # STRATEGY 2: DC50 (Log10 + Standardization)
    #         # Transforms raw concentrations to pDC50, then Z-scores them.
    #         return Pipeline([
    #             ('to_pdc50', FunctionTransformer(
    #                 func=_dc50_to_pdc50_vectorized,
    #                 inverse_func=_pdc50_to_dc50_inverse,
    #                 kw_args={'unit': 'nM'},        # Pass arguments here
    #                 inv_kw_args={'unit': 'nM'},    # Pass arguments for inverse here
    #                 validate=True
    #             )),
    #             ('scaler', StandardScaler())
    #         ])
            
    #     else:
    #         raise ValueError(f"Unknown target_type: {target_type}. Must be 'Dmax' or 'DC50'.")

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
        if not self.use_descriptors or self.desc_embedder is None:
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
        if not self.use_cell_description_embedding or self.cell_description_embedding is None:
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
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )     

    def get_Xy(self, dataset: Dataset, return_features_names: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """ Convert a featurized HuggingFace Dataset to an XGBoost DMatrix.
        
        Args:
            dataset: The HuggingFace Dataset to convert.
            return_features_names: Whether to return the feature names.
            
        Returns:
            An XGBoost DMatrix containing features and labels (if available).
        """
        # Precompute feature names from the first sample
        self.logger.debug("Extracting feature names for XGBoost DMatrix...")
        first_row = dataset[0]
        
        # Exclude tokenizer-related and label columns from XGBoost features
        excluded_keys = set(self.labels) | {'Prompt', 'input_ids', 'attention_mask', 'token_type_ids'}
        features_names = [k for k in first_row.keys() if k not in excluded_keys]
        if self.sort_features:
            features_names = sorted(features_names)
        
        # For certain features, like sequence embeddings, expand feature names,
        # e.g., name each n-gram
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

        # Parallel extraction of features using map
        def extract_features(example):
            features = [example[feat].numpy().flatten() for feat in features_names]
            return {"feature_vector": np.concatenate(features)}

        self.logger.debug("Extracting features for XGBoost DMatrix...")
        if self.num_proc > 1:
            mapped_ds = dataset.map(extract_features, num_proc=self.num_proc)
        else:
            # Use a bare for loop if num_proc is 1 to avoid map overhead
            features_list = []
            for i in range(len(dataset)):
                sample = dataset[i]
                features = extract_features(sample)
                features_list.append(features)
            mapped_ds = Dataset.from_list(features_list)
        X = np.vstack(mapped_ds["feature_vector"])
        
        # Check if all labels are present
        include_labels = True
        for label in self.labels:
            if label not in dataset.column_names:
                self.logger.warning(f"Label column '{label}' not found in dataset, skipping label extraction.")
                include_labels = False
                break
        
        if not include_labels:
            y = None
        else:
            self.logger.debug("Extracting labels for XGBoost DMatrix...")
            y = np.hstack([
                np.array(dataset.with_format("numpy")[label])[:, np.newaxis] for label in self.labels
            ]).astype(np.float32)
            
            # NOTE: The following imputation cannot be applied during inference
            # and with unknown new data to featurize, unless we save the imputer
            # state during training. For now, we skip it entirely. NaN values in X
            # are independently handled by the featurization pipelines (and also
            # by XGBoost itself). NaN values in y should have been removed during
            # featurization of the datasets.
            # ----------------------------------------------------------------------
            # # Combine X and y to use KNNImputer if there are missing labels
            # combined = np.hstack([X, y])
            # if np.isnan(combined).any():
            #     self.logger.debug("Imputing missing values with KNNImputer...")
            #     # imputer = KNNImputer(n_neighbors=5, keep_empty_features=True)
            #     imputer = IterativeImputer(
            #         skip_complete=True,
            #         max_value=10,
            #         keep_empty_features=True,
            #         random_state=42,
            #     )
            #     combined_imputed = imputer.fit_transform(combined)
            #     X = combined_imputed[:, :X.shape[1]]
            #     y = combined_imputed[:, X.shape[1]:]
            #     print(f"[{split} - After imputation] X shape: {X.shape}, y shape: {y.shape}")

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
        if self.use_poi_sequence_embedding and self.poi_sequence_embedding is not None:
            if hasattr(self.poi_sequence_embedding, 'sklearn_encoder') and \
               hasattr(self.poi_sequence_embedding.sklearn_encoder, 'vocabulary_'):
                state['poi_sequence_embedding_sklearn_encoder'] = pickle.dumps(self.poi_sequence_embedding.sklearn_encoder)
        
        # Save fingerprint embedder info
        if self.use_fingerprints and self.fp_embedder is not None:
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
            self.category_pipeline = pickle.loads(state_dict['category_pipeline'])

        if 'numeric_pipeline' in state_dict:
            self.numerical_cols = state_dict.get('numerical_cols', [])
            self.numeric_pipeline = pickle.loads(state_dict['numeric_pipeline'])

        # 4. Restore label transformers
        if (self.normalize_labels or self.standardize_labels) and 'label_transformers' in state_dict:
            for key, bytes_data in state_dict['label_transformers'].items():
                # Pickle loads directly from bytes
                self.label_transformers[key] = pickle.loads(bytes_data)

        # 5. Restore POI sequence embedding (TfidfVectorizer)
        if self.use_poi_sequence_embedding and 'poi_sequence_embedding_sklearn_encoder' in state_dict:
            if self.poi_sequence_embedding is not None:
                self.poi_sequence_embedding.sklearn_encoder = pickle.loads(state_dict['poi_sequence_embedding_sklearn_encoder'])
        
        # 6. Restore tokenizer (Transformers are best loaded by name)
        if self.use_tokenizer and 'tokenizer_name' in state_dict:
            self.tokenizer_name = state_dict['tokenizer_name']
            self.max_length = state_dict.get('max_length', self.max_length)
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 7. Restore PCA transformers
        if self.use_poi_pca and 'poi_pca' in state_dict:
            self.poi_pca = pickle.loads(state_dict['poi_pca'])
        
        if self.use_ligase_pca and 'ligase_pca' in state_dict:
            self.ligase_pca = pickle.loads(state_dict['ligase_pca'])
        
        self.logger.debug(f"State dict loaded via Pickle. Feature dims: {len(self.feature_dims)}, Encoders fitted: {self._encoders_fitted}")

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
        rel_std: float = 0.05,  # 5% relative standard deviation
        is_log_scale: bool = False,
        ignore_null_operators: bool = True,
    ) -> Dict[str, Any]:
        
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

    def oversample_uniform(self, dataset: Dataset, label: str, n_bins: int = 20) -> Dataset:
        """ Oversample the training dataset to make the label distributions
        more uniform. This is currently not working properly and it is not used.
        """
        values = np.array(dataset[label])
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
        if getattr(self, 'use_fingerprints', False):
            parts.append(f'fp{getattr(self, "fp_size", 512)}r{getattr(self, "radius", 16)}')
        if getattr(self, 'use_descriptors', False):
            if getattr(self, 'use_relevant_descriptors', False):
                parts.append('desc')
            elif getattr(self, 'selected_descriptors', False):
                parts.append('sel_desc')
            else:
                parts.append('all_desc')
        if getattr(self, 'use_tokenizer', False):
            tokenizer = getattr(self, 'tokenizer_name', 'bert')
            parts.append(f'tok_{tokenizer.split("/")[-1].replace("-", "_")}')
        if getattr(self, 'use_poi_name_embedding', False):
            if self.categorical_encoding == 'embedding':
                parts.append('poi_pt')
            elif self.categorical_encoding == 'onehot':
                parts.append('poi_onehot')
            else:
                parts.append('poi_ord')
        if getattr(self, 'use_poi_sequence_embedding', False):
            parts.append('poi_vec')
        if getattr(self, 'use_poi_precomputed_embedding', False):
            parts.append('poi_emb')
        if getattr(self, 'use_ligase_precomputed_embedding', False):
            parts.append('lig_emb')
        if getattr(self, 'use_ligase_name_embedding', False):
            if self.categorical_encoding == 'embedding':
                parts.append('lig_pt')
            elif self.categorical_encoding == 'onehot':
                parts.append('lig_onehot')
            else:
                parts.append('lig_ord')
        if getattr(self, 'use_cell_name_embedding', False):
            if self.categorical_encoding == 'embedding':
                parts.append('cell_pt')
            elif self.categorical_encoding == 'onehot':
                parts.append('cell_onehot')
            else:
                parts.append('cell_ord')
        if getattr(self, 'use_cell_description_embedding', False):
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
    hparams_path: Union[Path, str],
    state_dict_path: Union[Path, str],
) -> DegradationComplexDataModule:
    """ Load a DegradationComplexDataModule from YAML hyperparameters and a saved state dict.
    
    Args:
        hparams_path: Path to the YAML file with hyperparameters.
        state_dict_path: Path to the saved state dict file.
        
    Returns:
        An instance of DegradationComplexDataModule with loaded state.
    """
    hparams_path = Path(hparams_path)
    state_path = Path(state_dict_path)
    dm = DegradationComplexDataModule(
        dataset=None,
        **load_config_from_yaml(hparams_path)
    )
    dm.load_state_dict(torch.load(state_path))

    # Check if any of the non-None encoders are fitted, this will raise an error
    # if they are not have been fitted, and so that the checkpoints are broken.
    if dm.category_pipeline is not None:
        check_is_fitted(dm.category_pipeline)
    if dm.numeric_pipeline is not None:
        check_is_fitted(dm.numeric_pipeline)
    for transformer in dm.label_transformers.values():
        check_is_fitted(transformer)
    if dm.use_poi_pca and dm.poi_pca is not None:
        check_is_fitted(dm.poi_pca)
    if dm.use_ligase_pca and dm.ligase_pca is not None:
        check_is_fitted(dm.ligase_pca)
    if dm.poi_sequence_embedding is not None:
        if dm.use_poi_sequence_embedding:
            check_is_fitted(dm.poi_sequence_embedding.sklearn_encoder)

    return dm

def _dmax_logit_forward(X: np.ndarray) -> np.ndarray:
    """ Forward step for Dmax: 
        1. Scale 0-100 to 0-1
        2. Clip to avoid infinity in logit (0.001 to 0.999)
        3. Apply Logit transform: log(p / (1-p))
        
    Args:
        X: Raw Dmax value or array of values. Must be in [0, 100] range.
        
    Returns:
        Logit-transformed Dmax value or array of values.
    """
    # Ensure float
    X = X.astype(float)
    
    # Scale 0-100 -> 0-1
    X_norm = X / 100.0
    
    # Clip to avoid log(0) or log(1) which yields inf
    # 1e-4 allows dmax values close to 0 and 100 while keeping numbers stable
    epsilon = 1e-4
    X_clipped = np.clip(X_norm, epsilon, 1 - epsilon)
    
    # Logit transform: log(p / (1-p))
    return logit(X_clipped)

def _dmax_logit_inverse(X: np.ndarray) -> np.ndarray:
    """ Inverse step for Dmax:
        1. Apply Sigmoid (inverse of logit)
        2. Scale 0-1 -> 0-100
        
    Args:
        X: Logit-transformed Dmax value or array of values.
        
    Returns:
        Raw Dmax value or array of values.
    """
    # Sigmoid function
    X_sigmoid = expit(X)
    
    # Rescale back to percentage
    return X_sigmoid * 100.0

def _dc50_to_pdc50_vectorized(X: np.ndarray, unit: str = 'nM') -> np.ndarray:
    """ Vectorized version of DC50 -> pDC50 for pipelines. Accepts 2D array (n_samples, 1).
    """
    unit_factors = {
        'pM': 1e-12, 'nM': 1e-9, 'uM': 1e-6, 'M': 1.0
    }
    factor = unit_factors.get(unit, 1e-9)
    
    # Convert to float and handle potential string inputs
    X = X.astype(float)
    
    # Apply transformation: -log10(Concentration)
    # Added 1e-20 inside log to match your scalar implementation's safety
    return -np.log10(X * factor + 1e-20)

def _pdc50_to_dc50_inverse(X: np.ndarray, unit: str = 'nM') -> np.ndarray:
    """ Inverse transformation: pDC50 -> Raw DC50.
    
    Args:
        X: pDC50 value or array of values.
        unit: Unit of concentration ('nM', 'uM', 'pM', 'M').
        
    Returns:
        Raw DC50 value or array of values.
    """
    unit_factors = {
        'pM': 1e-12, 'nM': 1e-9, 'uM': 1e-6, 'M': 1.0
    }
    factor = unit_factors.get(unit, 1e-9)
    
    # Inverse logic: 10^(-pDC50) / factor
    return (10 ** (-X)) / factor