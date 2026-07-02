""" """
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Union, List, Literal

import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

from tackai.data.utils import get_cache_dir


def encode_with_transformer(
        strings: Union[str, List[str]],
        tokenizer: Optional[AutoTokenizer] = None,
        model: Optional[AutoModel] = None,
        pretrained_model: str = "ailab-bio/PROTAC-Splitter-Encoder",
        batch_size: int = 64,
        device: Union[int, str] = "cpu",
        pooling: Literal["cls", "mean", "sum", "max", "mean_sqrt_len"] = "sum",
        return_tensors: Literal["pt", "np"] = "np",
        return_dict: bool = False,
) -> Union[torch.Tensor, np.ndarray]:
    """ Encode a list of strings into embeddings using a pre-trained transformer model.

    Args:
        strings: A single string or a list of strings to encode.
        tokenizer: Tokenizer for the model.
        model: Pre-trained model to use for encoding.
        pretrained_model: Name of the pre-trained model to load if tokenizer/model are None.
        batch_size: Batch size for encoding.
        device: Device to run the model on ("cpu" or "cuda").
        pooling: Pooling method applied along the sequence-length dimension.
        return_tensors: Return format ("pt" for PyTorch tensors, "np" for NumPy arrays).
        return_dict: If True, return a dict mapping each string to its embedding.

    Returns:
        Tensor or array of shape (num_strings, embedding_dim), or a dict if return_dict=True.
    """
    if isinstance(strings, str):
        strings = [strings]

    if tokenizer is None or model is None:
        logging.warning(f"Loading pre-trained model {pretrained_model} for encoding strings.")
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        model = AutoModel.from_pretrained(pretrained_model)

    model = model.to(device)
    embeddings = []
    for i in range(0, len(strings), batch_size):
        batch = strings[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            if pooling == "cls":
                batch_embeds = outputs.last_hidden_state[:, 0, :].cpu()
            elif pooling == "mean":
                batch_embeds = outputs.last_hidden_state.mean(dim=1).cpu()
            elif pooling == "sum":
                batch_embeds = outputs.last_hidden_state.sum(dim=1).cpu()
            elif pooling == "max":
                batch_embeds = outputs.last_hidden_state.max(dim=1).values.cpu()
            elif pooling == "mean_sqrt_len":
                seq_lengths = inputs["input_ids"].ne(tokenizer.pad_token_id).sum(dim=1, keepdim=True).float()
                batch_embeds = (outputs.last_hidden_state.sum(dim=1) / seq_lengths.sqrt()).cpu()
            else:
                raise ValueError(f"Unsupported pooling method: {pooling}")
            embeddings.append(batch_embeds)

    result = torch.cat(embeddings, dim=0)
    if return_tensors == "np":
        result = result.numpy()

    if return_dict:
        return {s: emb for s, emb in zip(strings, result)}

    return result


class EmbeddingMixin(ABC):
    """ Abstract base class for embedding handlers with caching and serialization support.

    Subclasses implement :meth:`_transform_batch` for their specific encoding strategy.
    The :meth:`transform` template method handles caching, key resolution, and result ordering.
    All instances are picklable and can be saved as part of a datamodule checkpoint.
    """

    def __init__(
            self,
            embeddings: Optional[Union[Dict[str, np.ndarray], np.ndarray]] = None,
            model: Optional[Union[AutoModel, str]] = None,
            tokenizer: Optional[Union[AutoTokenizer, str]] = None,
            load_from_cache: bool = False,
            filename: Optional[Union[Path, str]] = None,
            cache_dir: Optional[Union[Path, str]] = None,
    ):
        if isinstance(embeddings, dict):
            self.embeddings = embeddings
        elif isinstance(embeddings, np.ndarray):
            self.embeddings = {str(i): emb for i, emb in enumerate(embeddings)}
        else:
            self.embeddings = {}

        self.filename = filename
        self.cache_dir = cache_dir
        if load_from_cache and filename is not None:
            self.load(filename=filename, cache_dir=cache_dir)

        self.model = None
        if model is not None:
            if isinstance(model, str):
                self.model = AutoModel.from_pretrained(model)
            else:
                self.model = model
            self.model.eval()

        self.tokenizer = None
        if tokenizer is not None:
            if isinstance(tokenizer, str):
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            else:
                self.tokenizer = tokenizer

    def __getitem__(self, key: str) -> np.ndarray:
        return self.embeddings[key]

    def __contains__(self, key: str) -> bool:
        return key in self.embeddings

    def __len__(self) -> int:
        return len(self.embeddings)

    def shape(self) -> tuple:
        """ Return the common shape of all stored embeddings.

        Returns:
            tuple: Shape of a single embedding. Leading or trailing size-1 dimensions
                are squeezed (e.g. (1, 128) → (128,)).

        Raises:
            ValueError: If no embeddings are stored or shapes are inconsistent.
        """
        if not self.embeddings:
            raise ValueError("No embeddings stored.")
        shapes = {emb.shape for emb in self.embeddings.values()}
        if len(shapes) != 1:
            raise ValueError(f"Embeddings have inconsistent shapes: {shapes}")
        shape = shapes.pop()
        if all(dim == 1 for dim in shape[:-1]):
            return (shape[-1],)
        if all(dim == 1 for dim in shape[1:]):
            return (shape[0],)
        return shape

    def save(
            self,
            filename: Optional[Union[Path, str]] = None,
            cache_dir: Optional[str] = None,
    ):
        """ Save embeddings to a .npz file.

        Args:
            filename: Name of the file. Defaults to the filename set on this instance,
                or "embeddings.npz" if unset.
            cache_dir: Directory to save the file. Defaults to the cache directory set on
                this instance, or the value of the TACKAI_CACHE environment variable.
        """
        if cache_dir is None:
            cache_dir = self.cache_dir if self.cache_dir is not None else get_cache_dir()
        if filename is None:
            filename = self.filename if self.filename is not None else "embeddings.npz"

        filepath = Path(cache_dir) / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        to_save = {
            k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
            for k, v in self.embeddings.items()
        }
        np.savez(filepath, **to_save)
        logging.info(f"Embeddings saved to {filepath}")

    def load(self, filename: Optional[Union[Path, str]] = None, cache_dir: Optional[str] = None):
        """ Load embeddings from a .npz file. """
        cache_dir = get_cache_dir() if cache_dir is None else cache_dir
        filepath = Path(cache_dir) / filename if filename else Path(cache_dir) / self.filename

        if not filepath.exists():
            logging.warning(f"File {filepath} does not exist. Skipping load.")
            return

        loaded = np.load(filepath, allow_pickle=True)
        self.embeddings = {k: v for k, v in loaded.items()}
        logging.info(f"Embeddings loaded from {filepath}")

    def to_numpy(self) -> np.ndarray:
        """ Stack all stored embeddings into a single numpy array. """
        return np.stack(list(self.embeddings.values()), axis=0)

    def to_tensor(self) -> torch.Tensor:
        """ Stack all stored embeddings into a single tensor. """
        tensors = [
            v if isinstance(v, torch.Tensor) else torch.from_numpy(v)
            for v in self.embeddings.values()
        ]
        return torch.stack(tensors, dim=0)

    def preprocess(
        self,
        op: Literal["standardize", "normalize", "minmax"] = "standardize",
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """ Return a preprocessed copy of the embeddings without modifying this instance.

        Args:
            op: The preprocessing operation to apply.
            **kwargs: Additional keyword arguments forwarded to the sklearn scaler.

        Returns:
            Dict[str, np.ndarray]: Preprocessed embeddings keyed by the same keys.
        """
        if op == "standardize":
            scaler = StandardScaler(**kwargs)
        elif op == "normalize":
            scaler = Normalizer(**kwargs)
        elif op == "minmax":
            scaler = MinMaxScaler(**kwargs)
        else:
            raise ValueError(f"Unsupported operation: {op}")

        arr = self.to_numpy()
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        scaled = scaler.fit_transform(arr)
        return dict(zip(self.embeddings.keys(), scaled))

    # --- Template method ---------------------------------------------------------

    def transform(
        self,
        items: Union[str, List[str]],
        skip_existing: bool = True,
        update_cache: bool = False,
    ) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """ Encode items into embeddings, using the cache to skip already-encoded ones.

        Args:
            items: A single item (string) or list of items to encode.
            skip_existing: If True, items already present in ``self.embeddings`` are
                returned from cache without re-encoding.
            update_cache: If True, persist the updated embeddings to disk after encoding.

        Returns:
            A dict mapping each item to its embedding when ``items`` is a list, or a
            single embedding array when ``items`` is a plain string.
        """
        single = isinstance(items, str)
        items_input = [items] if single else list(items)  # original keys, before normalization
        items_list = self._normalize_items(list(items_input))  # normalized keys used for encoding/cache

        if skip_existing:
            to_encode = [x for x in items_list if x not in self.embeddings]
            result: Dict[str, np.ndarray] = {x: self.embeddings[x] for x in items_list if x in self.embeddings}
        else:
            to_encode = items_list
            result = {}

        if to_encode:
            key_map = self._resolve_keys(to_encode)  # normalized → canonical
            canonical_keys = list(dict.fromkeys(key_map.values()))  # preserve order, deduplicate

            # Reuse canonical keys already cached (e.g. not_found_description after fuzzy matching)
            canonical_embs: Dict[str, np.ndarray] = {k: self.embeddings[k] for k in canonical_keys if k in self.embeddings}
            to_encode_canonical = [k for k in canonical_keys if k not in self.embeddings]
            if to_encode_canonical:
                canonical_embs.update(self._transform_batch(to_encode_canonical))

            # Map canonical embeddings back to normalized keys and store
            new_embs = {norm: canonical_embs[canon] for norm, canon in key_map.items()}
            self.embeddings.update(new_embs)
            result.update(new_embs)

        if update_cache:
            self.save()

        # Return keyed by the ORIGINAL (pre-normalization) inputs so callers can look up
        # by whatever they passed in (including None or "" that were normalized internally).
        ordered = {orig: result[norm] for orig, norm in zip(items_input, items_list)}
        return ordered[items_input[0]] if single else ordered

    def _normalize_items(self, items: List[str]) -> List[str]:
        """ Pre-process a list of items before encoding. Override to apply normalization. """
        return items

    def _resolve_keys(self, keys: List[str]) -> Dict[str, str]:
        """ Map original keys to canonical keys used for encoding.

        The default implementation is the identity map. Override to apply fuzzy matching
        or other key normalization (e.g. in CellEmbedding).
        """
        return {k: k for k in keys}

    @abstractmethod
    def _transform_batch(self, keys: List[str]) -> Dict[str, np.ndarray]:
        """ Encode a batch of keys and return a dict mapping each key to its embedding.

        Args:
            keys: Items to encode. Keys that were already in ``self.embeddings`` are
                filtered out before this method is called.

        Returns:
            Dict[str, np.ndarray]: Mapping from each key to its embedding array.
        """

    # --- Shared transformer utility ----------------------------------------------

    @staticmethod
    def encode_with_transformer(
            strings: Union[str, List[str]],
            tokenizer: Optional[AutoTokenizer] = None,
            model: Optional[AutoModel] = None,
            pretrained_model: str = "ailab-bio/PROTAC-Splitter-Encoder",
            batch_size: int = 64,
            device: Union[int, str] = "cpu",
            pooling: Literal["cls", "mean", "sum", "max", "mean_sqrt_len"] = "sum",
            return_tensors: Literal["pt", "np"] = "np",
            return_dict: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        """ Delegates to the module-level :func:`encode_with_transformer`. """
        return encode_with_transformer(
            strings=strings,
            tokenizer=tokenizer,
            model=model,
            pretrained_model=pretrained_model,
            batch_size=batch_size,
            device=device,
            pooling=pooling,
            return_tensors=return_tensors,
            return_dict=return_dict,
        )
