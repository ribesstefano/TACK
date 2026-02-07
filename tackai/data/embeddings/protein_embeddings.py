""" Handles protein embeddings using SciKit-Learn and Hugging Face Transformers. """
import logging
from pathlib import Path
from typing import Optional, List, Union, Literal, Dict

import numpy as np
import torch
import h5py
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OrdinalEncoder

from tackai.data.embeddings.utils import EmbeddingMixin


class ProteinEmbedding(EmbeddingMixin):
    """ Class for handling protein embeddings. """
    
    def __init__(
        self,
        embeddings_type: Literal["amino_acid_count", "ordinal", "transformer", "esm", "boltz2_s", "boltz2_z", "boltz_s_z", "precomputed"] = "esm",
        # Embedding-specific configurations
        count_vect_kwargs: Optional[dict] = None,
        ordinal_enc_kwargs: Optional[dict] = None,
        boltz_output_dir: Optional[Union[Path, str]] = None,
        
        # Precomputed embeddings configurations
        embeddings_file: Optional[Union[Path, str]] = None,
        embeddings_format: Literal["npz", "h5"] = "npz",
        embeddings_per_residue: bool = True,
        residue_pooling: Optional[Literal["mean", "sum", "max", "cls", "mean_sqrt_len"]] = None,
        sequence_id_key: str = "sequences",
        embeddings_key: str = "embeddings",
        
        # Transformer configurations
        pretrained_model: str = "facebook/esm2_t6_8M_UR50D",
        batch_size: int = 16,
        device: Union[int, str] = "cpu",
        pooling: Literal["cls", "mean", "sum", "max", "mean_sqrt_len"] = "sum",
        return_tensors: Literal["pt", "np"] = "np",
        
        # EmbeddingMixin parameters
        embeddings: Optional[Union[Dict[str, np.ndarray], np.ndarray]] = None,
        model: Optional[Union[AutoModel, str]] = None,
        tokenizer: Optional[Union[AutoTokenizer, str]] = None,
        load_from_cache: bool = False,
        filename: Optional[Union[Path, str]] = None,
        cache_dir: Optional[Union[Path, str]] = None,
    ):
        """ Initialize the ProteinEmbedding class.
        
        Args:
            embeddings_type: Type of embeddings to compute consistently for this instance
            count_vect_kwargs: Parameters for TfidfVectorizer (only used if embeddings_type="amino_acid_count")
            boltz_output_dir: Directory containing Boltz2 embeddings files (only used for boltz2 types)
            embeddings_file: Path to file containing precomputed embeddings (npz or h5)
            embeddings_format: Format of the embeddings file ("npz" or "h5")
            embeddings_per_residue: Whether embeddings are per-residue (True) or per-sequence (False)
            residue_pooling: Pooling method for per-residue embeddings. If None and embeddings_per_residue=True, 
                           will remove BOS/EOS tokens but keep per-residue structure
            sequence_id_key: Key name for sequence identifiers in the embeddings file
            embeddings_key: Key name for embeddings array in the embeddings file
            pretrained_model: Name of the pre-trained model to use if tokenizer or model is None
            batch_size: Batch size for transformer encoding
            device: Device to run the model on ("cpu" or "cuda")
            pooling: Pooling method for transformer embeddings
            return_tensors: Format of the returned tensors
            embeddings: Precomputed embeddings or fingerprints
            model: Pre-trained transformer model for embeddings
            tokenizer: Tokenizer for the transformer model
            load_from_cache: Whether to load embeddings from cache
            filename: Path to the file containing embeddings
            cache_dir: Directory to store cached embeddings
        """
        # Set default filename based on embeddings_type if not provided
        if filename is None:
            config = embeddings_type
            if embeddings_type in ["transformer", "esm"]:
                config = config + f"_model={pretrained_model.replace('/', '-')}"
                config = config + f"_pooling={pooling}"
            elif "boltz" in embeddings_type:
                config = config + f"_pooling={pooling}"
            elif embeddings_type == "amino_acid_count":
                args = '_'.join([f"{k}={v.replace('.','').replace(' ','')}" for k, v in count_vect_kwargs.items()])
                config = config + args
            elif embeddings_type == "precomputed":
                if embeddings_file:
                    config = config + f"_file={Path(embeddings_file).stem}"
                if residue_pooling:
                    config = config + f"_pooling={residue_pooling}"
            filename = f"protein_embeddings_{config}.npz"
        
        super().__init__(
            embeddings=embeddings,
            model=model,
            tokenizer=tokenizer,
            load_from_cache=load_from_cache,
            filename=filename,
            cache_dir=cache_dir,
        )

        # Store embedding configuration
        self.embeddings_type = embeddings_type
        self.pretrained_model = pretrained_model
        self.batch_size = batch_size
        self.device = device
        self.pooling = pooling
        self.return_tensors = return_tensors
        
        # Precomputed embeddings configuration
        self.embeddings_file = Path(embeddings_file) if embeddings_file else None
        self.embeddings_format = embeddings_format
        self.embeddings_per_residue = embeddings_per_residue
        self.residue_pooling = residue_pooling
        self.sequence_id_key = sequence_id_key
        self.embeddings_key = embeddings_key
        
        # Load precomputed embeddings if specified
        if embeddings_type == "precomputed" and embeddings_file:
            self._load_precomputed_embeddings()

        # Initialize type-specific components
        self.sklearn_encoder = None
        if embeddings_type == "amino_acid_count":
            count_vect_kwargs = {
                "analyzer": "char",
                "ngram_range": (1, 2),
                "lowercase": False,
            } if count_vect_kwargs is None else count_vect_kwargs
            self.sklearn_encoder = TfidfVectorizer(**count_vect_kwargs)
        elif embeddings_type == "ordinal":
            encoder_args = {
                'handle_unknown': 'use_encoded_value',
                'unknown_value': -1,
                'dtype': np.int32,
            }
            encoder_args.update({} if ordinal_enc_kwargs is None else ordinal_enc_kwargs)
            self.sklearn_encoder = OrdinalEncoder(**encoder_args)
        
        if embeddings_type in ["boltz2_s", "boltz2_z"]:
            if boltz_output_dir is None:
                raise ValueError("boltz_output_dir must be provided for Boltz2 embeddings")
            self.boltz_output_dir = Path(boltz_output_dir)

    def _load_precomputed_embeddings(self):
        """Load precomputed embeddings from npz or h5 file."""
        if not self.embeddings_file.exists():
            raise FileNotFoundError(f"Embeddings file not found: {self.embeddings_file}")
        
        logging.info(f"Loading precomputed embeddings from {self.embeddings_file}")
        
        if self.embeddings_format == "npz":
            self._load_from_npz()
        elif self.embeddings_format == "h5":
            self._load_from_h5()
        else:
            raise ValueError(f"Unsupported embeddings format: {self.embeddings_format}")
        
        logging.info(f"Loaded {len(self.embeddings)} precomputed embeddings")

    def _load_from_npz(self):
        """Load embeddings from npz file."""
        data = np.load(self.embeddings_file, allow_pickle=True)
        
        # Get sequence identifiers and embeddings
        seq_ids = data[self.sequence_id_key]
        embs = data[self.embeddings_key]
        
        # Process each embedding
        for seq_id, emb in zip(seq_ids, embs):
            if emb is None or (isinstance(emb, float) and np.isnan(emb)):
                continue
            processed_emb = self._process_embedding(emb)
            self.embeddings[seq_id] = processed_emb

    def _load_from_h5(self):
        """Load embeddings from h5 file."""
        with h5py.File(self.embeddings_file, 'r') as f:
            # Assume structure: f[sequence_id] = embedding_array
            for seq_id in f.keys():
                emb = f[seq_id][:]
                processed_emb = self._process_embedding(emb)
                self.embeddings[seq_id] = processed_emb

    def _process_embedding(self, emb: np.ndarray) -> np.ndarray:
        """Process a single embedding based on configuration.
        
        Args:
            emb: Raw embedding array
            
        Returns:
            Processed embedding array
        """
        # If per-residue embeddings and pooling is specified, apply pooling
        if self.embeddings_per_residue:
            # Remove BOS/EOS tokens (first and last positions) if present
            # Common in ESM and similar models
            if emb.ndim == 2 and emb.shape[0] > 2:
                emb = emb[1:-1]
            
            # Apply pooling if specified
            if self.residue_pooling is not None:
                emb = self._pool_residue_embeddings(emb)
        
        return emb

    def _pool_residue_embeddings(self, emb: np.ndarray) -> np.ndarray:
        """Pool per-residue embeddings to sequence-level.
        
        Args:
            emb: Per-residue embeddings of shape (L, D)
            
        Returns:
            Pooled embeddings of shape (D,)
        """
        if self.residue_pooling == "mean":
            return np.mean(emb, axis=0)
        elif self.residue_pooling == "sum":
            return np.sum(emb, axis=0)
        elif self.residue_pooling == "max":
            return np.max(emb, axis=0)
        elif self.residue_pooling == "cls":
            # For models with CLS token at position 0 (before removal)
            # This assumes BOS token was not removed
            return emb[0]
        elif self.residue_pooling == "mean_sqrt_len":
            return np.mean(emb, axis=0) / np.sqrt(emb.shape[0])
        else:
            raise ValueError(f"Unsupported residue pooling method: {self.residue_pooling}")

    def transform(
        self,
        sequences: Union[str, List[str]],
        skip_existing: bool = True,
        update_cache: bool = False,
    ) -> Dict[str, Union[np.array, torch.Tensor]]:
        """ Encode sequences strings into fingerprints or embeddings using the configured method.
        
        Args:
            sequences: Sequence string or list of sequences strings
            skip_existing: Whether to skip sequences that are already in the embeddings
            update_cache: Whether to update the cache with new embeddings

        Returns:
            Dict[str, Union[np.array, torch.Tensor]]: Encoded embeddings
        """
        if isinstance(sequences, str):
            seq_list = [sequences]
        elif isinstance(sequences, list):
            seq_list = sequences
        else:
            raise ValueError("Input sequences must be a string or a list of strings.")

        if skip_existing:
            seq_to_encode = [s for s in seq_list if s not in self.embeddings]
            seq_encoded = {s: self.embeddings[s] for s in seq_list if s in self.embeddings}
        else:
            seq_to_encode = seq_list

        if not seq_to_encode:
            embeddings = {}
        else:
            # For precomputed embeddings, missing sequences will raise an error
            if self.embeddings_type == "precomputed":
                missing = [s for s in seq_to_encode if s not in self.embeddings]
                if missing:
                    raise KeyError(f"Sequences not found in precomputed embeddings: {missing[:5]}")
                embeddings = {s: self.embeddings[s] for s in seq_to_encode}
            else:
                embeddings = self._encode_sequences(seq_to_encode)

        if skip_existing:
            all_embeddings = {**seq_encoded, **embeddings}
            embeddings = {s: all_embeddings[s] for s in seq_list}

        # Update instance embeddings
        if len(embeddings) > 0:
            self.embeddings.update(embeddings)

        if update_cache:
            self.save()
        
        if isinstance(sequences, str):
            return embeddings[sequences]

        return embeddings

    def _encode_sequences(self, sequences: List[str]) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """ Internal method to encode sequences based on the configured embeddings_type. """
        if self.embeddings_type in ["amino_acid_count", "ordinal"]:
            return self._encode_sklearn(sequences)
        elif self.embeddings_type in ["transformer", "esm"]:
            return self._encode_with_transformer(sequences)
        elif self.embeddings_type in ["boltz2_s", "boltz2_z"]:
            return self._encode_boltz2(sequences)
        elif self.embeddings_type == "precomputed":
            # This shouldn't be called for precomputed since transform handles it
            return {s: self.embeddings[s] for s in sequences}
        else:
            raise ValueError(f"Unsupported embeddings_type: {self.embeddings_type}")

    def _encode_sklearn(self, sequences: List[str]) -> Dict[str, np.ndarray]:
        """ Encode sequences using amino acid counting. """
        if self.embeddings_type == "amino_acid_count":
            # Reshape for CountVectorizer
            embeddings = self.sklearn_encoder.transform(sequences).toarray()
        elif self.embeddings_type == "ordinal":
            # NOTE: The unknown value is encoded as -1, so after adding 1 it
            # will become 0, perfect for embedding layers.
            seq_reshaped = np.array(sequences).reshape(-1, 1)
            embeddings = self.sklearn_encoder.transform(seq_reshaped)
            embeddings = embeddings + 1
        return {s: e for s, e in zip(sequences, embeddings)}

    def _encode_with_transformer(self, sequences: List[str]) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """ Encode sequences using transformer model. """
        return self.encode_with_transformer(
            strings=sequences,
            tokenizer=self.tokenizer,
            model=self.model,
            pretrained_model=self.pretrained_model,
            batch_size=self.batch_size,
            device=self.device,
            pooling=self.pooling,
            return_tensors=self.return_tensors,
            return_dict=True,
        )

    def _encode_boltz2(self, sequences: List[str]) -> Dict[str, np.ndarray]:
        """ Encode sequences using Boltz2 embeddings. """
        embeddings = {}
        for seq_id in sequences:
            emb_file = self.boltz_output_dir / f"boltz_results_{seq_id}" / "predictions" / seq_id / f"embeddings_{seq_id}.npz"
            if not emb_file.exists():
                raise FileNotFoundError(f"Boltz2 embeddings file {emb_file} not found.")
            
            data = np.load(emb_file)
            s = data['s']
            z = data['z']

            if self.embeddings_type == "boltz2_s":
                embeddings[seq_id] = self.pool_boltz_embeddings(s=s, pooling=self.pooling)
            elif self.embeddings_type == "boltz2_z":
                embeddings[seq_id] = self.pool_boltz_embeddings(z=z, pooling=self.pooling, z_pooling="flatten")
        
        return embeddings

    def fit(self, sequences: List[str]):
        """ Fit the Sci-kit learn encoder on the provided sequences. """
        if self.embeddings_type == "ordinal":
            self.sklearn_encoder.fit(np.array(sequences).reshape(-1, 1))
        elif self.embeddings_type == "amino_acid_count":
            self.sklearn_encoder.fit(sequences)
        else:
            logging.warning("CountVectorizer can only be fitted for embeddings_type='amino_acid_count' or 'ordinal'. Skipping fit.")
            return

    @staticmethod
    def pool_boltz_embeddings(
            s: Optional[np.ndarray] = None,
            z: Optional[np.ndarray] = None,
            pooling: Literal["mean", "sum", "max", "mean_sqrt_len"] = "sum",
            z_pooling: Literal["none", "flatten", "sum_axis0", "sum_axis1"] = "flatten",
    ) -> np.ndarray:
        """ Pool Boltz2 embeddings. The s embeddings are of shape (L, D) and z embeddings are of shape (L, D, D).
        
        Args:
            s (Optional[np.ndarray]): Boltz2 S embeddings.
            z (Optional[np.ndarray]): Boltz2 Z embeddings.
            pooling (Literal["mean", "sum", "max", "mean_sqrt_len"]): Pooling method for reducing over the L dimension.
            z_pooling (Literal["none", "flatten", "sum_axis0", "sum_axis1"]): Pooling method for reducing over one D dimension of the z embeddings.

        Returns:
            np.ndarray: Pooled embeddings.
        """
        
        # Example shapes:
        # s: (1, 350, 384)
        # z: (1, 350, 350, 128)
        # s: (1, 498, 384)
        # z: (1, 498, 498, 128)

        if s is not None and z is not None:
            raise NotImplementedError("Both 's' and 'z' cannot be provided at the same time. Choose one.")
        elif s is not None:
            s = s[0]
            # s is of shape (L, D)
            if pooling == "mean":
                return np.mean(s, axis=0)
            elif pooling == "sum":
                return np.sum(s, axis=0)
            elif pooling == "max":
                return np.max(s, axis=0)
            elif pooling == "mean_sqrt_len":
                return np.mean(s, axis=0) / np.sqrt(s.shape[0])
            else:
                raise ValueError(f"Unsupported pooling method: {pooling}. Choose from 'mean', 'sum', 'max', or 'mean_sqrt_len'.")
        elif z is not None:
            z = z[0]
            # z is of shape (L, L, D)
            if pooling == "mean":
                emb = np.mean(z, axis=(0, 1))
            elif pooling == "sum":
                emb = np.sum(z, axis=(0, 1))
            elif pooling == "max":
                emb = np.max(z, axis=(0, 1))
            elif pooling == "mean_sqrt_len":
                emb = np.mean(z, axis=(0, 1)) / np.sqrt(z.shape[0])

            if z_pooling == "none":
                return emb
            elif z_pooling == "flatten":
                return emb.flatten()
            elif z_pooling == "sum_axis0":
                return np.sum(emb, axis=0)
            elif z_pooling == "sum_axis1":
                return np.sum(emb, axis=1)
        else:
            raise ValueError("Either 's' or 'z' must be provided.")