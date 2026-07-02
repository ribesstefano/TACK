""" Handles cell embeddings using SciKit-Learn and Hugging Face Transformers. """
import re
from pathlib import Path
from typing import Optional, List, Union, Literal, Dict, Tuple
import requests
import logging
import json
from difflib import get_close_matches

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sentence_transformers import SentenceTransformer
from thefuzz import process

from tackai.data.utils import get_cache_dir
from tackai.data.embeddings.utils import EmbeddingMixin


class CellEmbedding(EmbeddingMixin):
    """ Class for handling cell line embeddings. """

    def __init__(
        self,
        embeddings_type: Literal["one_hot", "ordinal", "transformer", "sentence_transformer"] = "sentence_transformer",
        not_found_description: str = "Unknown cell line.",

        # One-hot encoding configurations
        onehot_enc_kwargs: Optional[dict] = None,
        ordinal_enc_kwargs: Optional[dict] = None,

        # Transformer configurations
        pretrained_model: str = "sentence-transformers/all-mpnet-base-v1",
        batch_size: int = 64,
        device: Union[int, str] = "cpu",
        pooling: Literal["cls", "mean", "sum", "max", "mean_sqrt_len"] = "sum",
        return_tensors: Literal["pt", "np"] = "np",

        # Cell line processing configurations
        get_cellosaurus_descriptions: bool = True,
        min_similarity_score: float = 90,

        # EmbeddingMixin parameters
        embeddings: Optional[Union[Dict[str, np.ndarray], np.ndarray]] = None,
        model: Optional[Union[AutoModel, str]] = None,
        tokenizer: Optional[Union[AutoTokenizer, str]] = None,
        load_from_cache: bool = False,
        filename: Optional[Union[Path, str]] = None,
        cache_dir: Optional[Union[Path, str]] = None,
        verbose: int = 0,
    ):
        """ Initialize the CellEmbedding class.

        Args:
            embeddings_type: Type of embeddings to compute consistently for this instance.
            not_found_description: Fallback text/embedding used for unrecognized cell lines.
            onehot_enc_kwargs: Parameters for OneHotEncoder (only used if embeddings_type="one_hot").
            ordinal_enc_kwargs: Parameters for OrdinalEncoder (only used if embeddings_type="ordinal").
            pretrained_model: Name of the pre-trained model to use.
            batch_size: Batch size for encoding.
            device: Device to run the model on ("cpu" or GPU index).
            pooling: Pooling method for transformer embeddings.
            return_tensors: Return type of the embeddings ("pt" for PyTorch tensors, "np" for NumPy arrays).
            get_cellosaurus_descriptions: Whether to encode Cellosaurus descriptions instead of raw cell line IDs.
            min_similarity_score: Minimum similarity score for fuzzy matching (0–100).
            embeddings: Precomputed embeddings.
            model: Pre-trained transformer model.
            tokenizer: Tokenizer for the transformer model.
            load_from_cache: Whether to load embeddings from cache.
            filename: Path to the file containing embeddings.
            cache_dir: Directory to store cached embeddings.
            verbose: Logging verbosity (0=ERROR, 1=WARNING, 2=DEBUG).
        """
        if filename is None:
            filename = f"cell_embeddings_{embeddings_type}.npz"

        super().__init__(
            embeddings=embeddings,
            model=model,
            tokenizer=tokenizer,
            load_from_cache=load_from_cache,
            filename=filename,
            cache_dir=cache_dir,
        )

        self.embeddings_type = embeddings_type
        self.not_found_description = not_found_description
        self.pretrained_model = pretrained_model
        self.batch_size = batch_size
        self.device = device
        self.pooling = pooling
        self.return_tensors = return_tensors
        self.get_cellosaurus_descriptions = get_cellosaurus_descriptions
        self.min_similarity_score = min_similarity_score

        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        if verbose == 0:
            self.logger.setLevel(logging.ERROR)
        elif verbose == 1:
            self.logger.setLevel(logging.WARNING)
        elif verbose == 2:
            self.logger.setLevel(logging.DEBUG)

        self.logger.debug("Loading Cellosaurus data...")
        cellosaurus_text = self._get_cellosaurus_text(cache_dir=cache_dir)
        self.logger.debug("Parsing Cellosaurus data...")

        filepath_data = Path(cache_dir or get_cache_dir()) / "cell2data.json"
        filepath_descr = Path(cache_dir or get_cache_dir()) / "cell2description.json"
        filepath_cell_id = Path(cache_dir or get_cache_dir()) / "cell2cell_id.json"
        if filepath_data.exists() and filepath_descr.exists() and filepath_cell_id.exists():
            with open(filepath_data, 'r') as f:
                self.cell2data = json.load(f)
            with open(filepath_descr, 'r') as f:
                self.cell2description = json.load(f)
            with open(filepath_cell_id, 'r') as f:
                self.cell2cell_id = json.load(f)
            self.logger.debug("Loaded Cellosaurus data from cached JSON files.")
        else:
            cell_lines = self._parse_cellosaurus_text(cellosaurus_text)
            self.cell2description = {}
            self.cell2data = {}
            self.cell2cell_id = {}
            self.logger.debug(f"Processing {len(cell_lines)} cell lines from Cellosaurus...")
            for cell_line in cell_lines:
                cell_data, cell_descr = self.clean_cell_line_cellosaurus_entry(cell_line)
                self.cell2data[cell_line['ID']] = cell_data
                self.cell2description[cell_line['ID']] = cell_descr
                self.cell2description[cell_line['AC']] = cell_descr
                self.cell2cell_id[cell_line['ID']] = cell_line['AC']

            with open(filepath_data, 'w') as f:
                json.dump(self.cell2data, f, indent=4)
            self.logger.debug(f"Processed Cellosaurus data saved to {filepath_data}")

            with open(filepath_descr, 'w') as f:
                json.dump(self.cell2description, f, indent=4)
            self.logger.debug(f"Cell line descriptions saved to {filepath_descr}")

            with open(filepath_cell_id, 'w') as f:
                json.dump(self.cell2cell_id, f, indent=4)
            self.logger.debug(f"Cell line ID mappings saved to {filepath_cell_id}")

        self.cell_id2data = {v: self.cell2data[k] for k, v in self.cell2cell_id.items()}

        self.synonym2cell_line = {}
        for cell_id, cell_data in self.cell2data.items():
            if 'SY' in cell_data:
                for synonym in cell_data['SY']:
                    synonym = synonym.strip()
                    if synonym and synonym not in self.synonym2cell_line:
                        self.synonym2cell_line[synonym] = cell_id

        self.sklearn_encoder = None
        if embeddings_type == "one_hot":
            encoder_args = {"handle_unknown": "ignore"}
            encoder_args.update({} if onehot_enc_kwargs is None else onehot_enc_kwargs)
            self.sklearn_encoder = OneHotEncoder(**encoder_args)
        elif embeddings_type == "ordinal":
            encoder_args = {
                'handle_unknown': 'use_encoded_value',
                'unknown_value': -1,
                'dtype': np.int32,
            }
            encoder_args.update({} if ordinal_enc_kwargs is None else ordinal_enc_kwargs)
            self.sklearn_encoder = OrdinalEncoder(**encoder_args)

        if self.sklearn_encoder is not None:
            X = self.get_cell_lines() + list(self.synonym2cell_line.keys())
            self.sklearn_encoder.fit(np.array(X).reshape(-1, 1))

    # --- EmbeddingMixin hooks ----------------------------------------------------

    def _normalize_items(self, items: List[str]) -> List[str]:
        """ Ensure not_found_description is encoded, then normalize None/"" entries. """
        self._ensure_not_found_encoded()
        return [s if s not in [None, ""] else self.not_found_description for s in items]

    def _resolve_keys(self, keys: List[str]) -> Dict[str, str]:
        """ Apply fuzzy matching to map raw cell line names to canonical Cellosaurus IDs. """
        if self.min_similarity_score <= 0:
            return {k: k for k in keys}
        return {s: self.get_fuzzy_cell_line(s, self.min_similarity_score)[0] for s in keys}

    def _transform_batch(self, keys: List[str]) -> Dict[str, np.ndarray]:
        """ Encode a batch of canonical cell line IDs into embeddings. """
        if self.embeddings_type in ["one_hot", "ordinal"]:
            # sklearn encoder is fitted on cell line IDs, not descriptions
            return self._encode_sklearn(keys)

        # Transformer-based methods encode Cellosaurus descriptions
        if self.get_cellosaurus_descriptions:
            descriptions = [self.get_cell_description(s) for s in keys]
            descriptions = [s if s not in [None, ""] else self.not_found_description for s in descriptions]
        else:
            descriptions = keys

        if self.embeddings_type == "transformer":
            return self._encode_with_transformer(descriptions, keys)
        elif self.embeddings_type == "sentence_transformer":
            return self._encode_with_sentence_transformer(descriptions, keys)
        else:
            raise ValueError(f"Unsupported embeddings_type: {self.embeddings_type}")

    def transform(
        self,
        cell_lines: Union[str, List[str], None],
        skip_existing: bool = True,
        update_cache: bool = False,
    ) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """ Encode cell lines into embeddings using the configured method.

        Args:
            cell_lines: Cell line string, list of cell line strings, or None.
                None returns the not_found_description embedding directly.
            skip_existing: Whether to skip already encoded cell lines.
            update_cache: Whether to update the cache with new embeddings.

        Returns:
            Dict[str, Union[np.array, torch.Tensor]] or a single embedding array.
        """
        if cell_lines is None:
            self._ensure_not_found_encoded()
            return self.embeddings[self.not_found_description]
        return super().transform(cell_lines, skip_existing=skip_existing, update_cache=update_cache)

    def _ensure_not_found_encoded(self):
        """ Encode and cache the not_found_description embedding if not already present. """
        if self.not_found_description not in self.embeddings:
            nf_embs = self._transform_batch([self.not_found_description])
            self.embeddings.update(nf_embs)

    # --- Cell line utilities -----------------------------------------------------

    def get_cell_lines(self) -> List[str]:
        """ Get all cell lines available in the embeddings. """
        return list(self.cell2description.keys())

    def get_cell_line_data(self, cell_line: str) -> Dict[str, Union[str, List[str]]]:
        """ Get data for a specific cell line.

        Args:
            cell_line: Cell line ID or name.

        Returns:
            Dict[str, Union[str, List[str]]]: Data for the cell line.
        """
        if cell_line in self.cell2data:
            return self.cell2data[cell_line]
        elif cell_line in self.synonym2cell_line:
            return self.cell2data[self.synonym2cell_line[cell_line]]
        else:
            raise ValueError(f"Cell line {cell_line} not found in the embeddings.")

    def get_cell_line_description(self, cell_line: str) -> str:
        """ Get the description for a specific cell line.

        Args:
            cell_line: Cell line ID or name.

        Returns:
            str: Description of the cell line.
        """
        if cell_line in self.cell2description:
            return self.cell2description[cell_line]
        elif cell_line in self.synonym2cell_line:
            return self.cell2description[self.synonym2cell_line[cell_line]]
        else:
            raise ValueError(f"Cell line {cell_line} not found in the embeddings.")

    def __getitem__(self, key: str) -> np.ndarray:
        """ Get the embedding for a given cell line, with fuzzy matching fallback.

        Args:
            key: Cell line ID or name.

        Returns:
            np.ndarray: Embedding for the cell line.
        """
        if key in self.embeddings:
            return self.embeddings[key]
        return self.embeddings[self.get_fuzzy_cell_line(key)[0]]

    @staticmethod
    def _get_cellosaurus_text(cache_dir: Union[str, Path] = None) -> str:
        """ Download the Cellosaurus text file and return its content. """
        if cache_dir is None:
            cache_dir = get_cache_dir()

        filepath = Path(cache_dir) / "cellosaurus.txt"
        if filepath.exists():
            with open(filepath, 'r') as file:
                return file.read()

        url = "https://ftp.expasy.org/databases/cellosaurus/cellosaurus.txt"
        response = requests.get(url)
        if response.status_code == 200:
            with open(filepath, 'w') as file:
                file.write(response.text)
            return response.text
        else:
            raise ValueError(f"Failed to download Cellosaurus text file. Status code: {response.status_code}")

    @staticmethod
    def _parse_cellosaurus_text(
            cellosaurus_text: str,
    ) -> List[Dict[str, Union[str, List[str]]]]:
        """ Parse a Cellosaurus text file and return a list of cell line entries.

        Args:
            cellosaurus_text: Content of the Cellosaurus text file.

        Returns:
            List[Dict[str, Union[str, List[str]]]]: List of dictionaries containing cell line
                information. Keys include 'ID', 'AC', 'SY', 'DR', 'RX', 'CC', 'OX', 'HI',
                'CA', 'DT'.
        """
        lines = cellosaurus_text.splitlines()

        cell_lines = []
        cell_line_entry = {}
        for line in lines:
            if line.startswith("ID   "):
                if cell_line_entry:
                    cell_lines.append(cell_line_entry)
                    cell_line_entry = {}
                cell_line_entry['ID'] = line[5:].strip()
            elif line.startswith("AC   "):
                cell_line_entry['AC'] = line[5:].strip()
            elif line.startswith("SY   "):
                cell_line_entry['SY'] = line[5:].strip()
            elif line.startswith("DR   "):
                cell_line_entry.setdefault('DR', []).append(line[5:].strip())
            elif line.startswith("RX   "):
                cell_line_entry.setdefault('RX', []).append(line[5:].strip())
            elif line.startswith("CC   "):
                cell_line_entry.setdefault('CC', []).append(line[5:].strip())
            elif line.startswith("OX   "):
                cell_line_entry['OX'] = line[5:].strip()
            elif line.startswith("HI   "):
                cell_line_entry['HI'] = line[5:].strip()
            elif line.startswith("CA   "):
                cell_line_entry['CA'] = line[5:].strip()
            elif line.startswith("DT   "):
                cell_line_entry['DT'] = line[5:].strip()

        if cell_line_entry:
            cell_lines.append(cell_line_entry)

        return cell_lines

    @staticmethod
    def clean_cell_line_cellosaurus_entry(cell_line, cc_headers_to_ignore=None, unique_columns_ranking=None):
        """
        Clean and process a single cell line entry from Cellosaurus data.

        Args:
            cell_line (dict): Single cell line entry from parse_cellosaurus_text
            cc_headers_to_ignore (list): List of CC headers to ignore during processing
            unique_columns_ranking (list): Ordered list of columns by uniqueness ranking

        Returns:
            tuple: (cleaned_cell_data_dict, description_string)
        """
        if cc_headers_to_ignore is None:
            cc_headers_to_ignore = [
                'Miscellaneous',
                'From',
                'Anecdotal',
                'Misspelling',
                'Part of',
                'Registration',
                'Discontinued',
            ]

        if unique_columns_ranking is None:
            unique_columns_ranking = [
                'Genome ancestry', 'Karyotypic information', 'Senescence',
                'Biotechnology', 'Virology', 'Caution', 'Donor information',
                'Sequence variation', 'Characteristics', 'Transfected with',
                'Monoclonal antibody target', 'HLA typing', 'Knockout cell',
                'Microsatellite instability', 'HI', 'Breed/subspecies',
                'Derived from site', 'Population', 'Group',
                'Monoclonal antibody isotype', 'Cell type', 'Transformant',
                'Selected for resistance to', 'CA'
            ]

        cell_data = cell_line.copy()
        for comment in cell_data.get('CC', []):
            cc_header = comment.split(':')[0].strip()
            if cc_header not in cc_headers_to_ignore:
                cc_text = comment.split(':')[1].strip()
                cell_data[cc_header] = cell_data.get(cc_header, '') + cc_text + ' '

        fields_to_ignore = ['CC', 'DT', 'SY']
        features_to_ignore = [
            'Problematic cell line',
            'Omics',
            'AC',
            'OX',
            'Doubling time',
        ]

        cell_description = ""
        for col in unique_columns_ranking:
            if col in fields_to_ignore or col in features_to_ignore:
                continue
            if col in cell_data and cell_data.get(col) is not None:
                cell_description += f"{cell_data[col].strip()}\n"

        cell_description = re.sub(r'\(PubMed=.*?\)', '', cell_description)
        cell_description = re.sub(r'UBERON=.*?\.', '', cell_description)
        cell_description = cell_description.strip()
        cell_description = cell_description.replace(' .', '.')
        cell_description = cell_description.replace('  ', ' ')

        if 'SY' in cell_data:
            cell_data['SY'] = cell_data['SY'].split(';')
            cell_data['SY'] = [syn.strip() for syn in cell_data['SY'] if syn.strip()]

        return cell_data, cell_description

    def get_fuzzy_cell_line(
            self,
            cell_line: str,
            min_similarity_score: float = 90,
            get_list: bool = False,
    ) -> Tuple[str, float]:
        """ Get the closest matching cell line ID among the available cell lines.

        Args:
            cell_line: Cell line ID or name.
            min_similarity_score: Minimum similarity score for fuzzy matching (0–100).
            get_list: If True, return all matches above the minimum similarity score.

        Returns:
            Tuple of (matched cell line ID, score).
        """
        all_cell_lines = list(self.cell2description.keys())
        all_synonyms = list(self.synonym2cell_line.keys())

        if cell_line is None or cell_line == "":
            return self.not_found_description, 0

        if cell_line in self.cell2description:
            return ([cell_line], 100) if get_list else (cell_line, 100)
        elif cell_line in self.synonym2cell_line:
            cell_id = self.synonym2cell_line[cell_line]
            return ([cell_id], 100) if get_list else (cell_id, 100)

        if not get_list:
            closest_match, score = process.extractOne(cell_line, all_cell_lines + all_synonyms)
            if score > min_similarity_score:
                if closest_match in self.cell2description:
                    return closest_match, score
                else:
                    closest_synonym = self.synonym2cell_line.get(closest_match, closest_match)
                    self.logger.debug(f"Using synonym '{closest_match}' for cell line '{cell_line}' with score {score}.")
                    return closest_synonym, score
            else:
                matches = get_close_matches(cell_line, all_cell_lines + all_synonyms, n=1, cutoff=min_similarity_score / 100)
                if matches:
                    closest_match = matches[0]
                    if closest_match in self.cell2description:
                        self.logger.debug(f"Using close match '{closest_match}' for cell line '{cell_line}' with score {score}.")
                        return closest_match, score
                    else:
                        closest_synonym = self.synonym2cell_line.get(closest_match, closest_match)
                        self.logger.debug(f"Using close synonym '{closest_match}' for cell line '{cell_line}' with score {score}.")
                        return closest_synonym, score

                self.logger.debug(f"No suitable match found for cell line '{cell_line}' with minimum score {min_similarity_score}.")
                return self.not_found_description, 0
        else:
            matches = process.extract(cell_line, all_cell_lines + all_synonyms, limit=None)
            filtered_matches = [(m[0], m[1]) for m in matches if m[1] >= min_similarity_score]
            if filtered_matches:
                return filtered_matches, 0
            else:
                self.logger.debug(f"No matches found for cell line '{cell_line}' with minimum score {min_similarity_score}.")
                return [(self.not_found_description, 0)], 0

    def get_cell_description(
            self,
            cell_line: str,
            use_fuzzy_matching: bool = False,
            min_similarity_score: float = 90,
            passthrough_if_not_found: bool = True,
    ) -> str:
        """ Get the description of a cell line.

        Args:
            cell_line: Cell line ID or name.
            use_fuzzy_matching: Whether to use fuzzy matching if exact match not found.
            min_similarity_score: Minimum similarity score for fuzzy matching (0–100).
            passthrough_if_not_found: If True, return the input cell_line if not found;
                otherwise raise an error.

        Returns:
            str: Description of the cell line.
        """
        if not (0 <= min_similarity_score <= 100):
            raise ValueError("min_similarity_score must be between 0 and 100.")

        if cell_line is None and passthrough_if_not_found:
            return self.not_found_description
        elif cell_line in self.cell2description:
            return self.cell2description.get(cell_line, self.not_found_description)
        elif cell_line in self.synonym2cell_line:
            cell_id = self.synonym2cell_line[cell_line]
            return self.cell2description.get(cell_id, self.not_found_description)
        elif use_fuzzy_matching:
            closest_synonym, score = self.get_fuzzy_cell_line(
                cell_line=cell_line,
                min_similarity_score=min_similarity_score,
            )
            return self.cell2description.get(closest_synonym, self.not_found_description)
        elif passthrough_if_not_found:
            return cell_line
        else:
            raise ValueError(f"Cell line \"{cell_line}\" not found in the embeddings.")

    # --- Internal encoding methods -----------------------------------------------

    def _encode_sklearn(self, keys: List[str]) -> Dict[str, np.ndarray]:
        """ Encode cell line IDs using the fitted sklearn encoder. """
        X = np.array(keys).reshape(-1, 1)
        if self.embeddings_type == "one_hot":
            embeddings = self.sklearn_encoder.transform(X).toarray()
        else:  # ordinal
            embeddings = self.sklearn_encoder.transform(X).astype(np.float32)
            # Shift by 1: OrdinalEncoder uses -1 for unknowns, making 0 a valid embedding index
            embeddings = embeddings + 1
        return {k: e for k, e in zip(keys, embeddings)}

    def _encode_with_transformer(self, descriptions: List[str], original_keys: List[str]) -> Dict[str, np.ndarray]:
        """ Encode descriptions using transformer model, keyed by original cell line IDs. """
        embeddings = self.encode_with_transformer(
            strings=descriptions,
            tokenizer=self.tokenizer,
            model=self.model,
            pretrained_model=self.pretrained_model,
            batch_size=self.batch_size,
            device=self.device,
            pooling=self.pooling,
            return_tensors=self.return_tensors,
            return_dict=True,
        )
        return {k: e for k, e in zip(original_keys, embeddings.values())}

    def _encode_with_sentence_transformer(self, descriptions: List[str], original_keys: List[str]) -> Dict[str, np.ndarray]:
        """ Encode descriptions using sentence transformer model, keyed by original cell line IDs. """
        model = self.model if self.model is not None else SentenceTransformer(self.pretrained_model)

        embeddings = model.encode(
            descriptions,
            batch_size=self.batch_size,
            device=self.device,
            output_value="token_embeddings",
        )
        self.logger.debug(f"Embeddings shapes: {', '.join([str(e.shape) for e in embeddings])}")

        if self.pooling == "sum":
            embeddings = [e.sum(axis=0) for e in embeddings]
        elif self.pooling == "mean":
            embeddings = [e.mean(axis=0) for e in embeddings]
        elif self.pooling == "max":
            embeddings = [e.max(axis=0) for e in embeddings]
        elif self.pooling == "mean_sqrt_len":
            embeddings = [e.mean(axis=0) / np.sqrt(e.shape[0]) for e in embeddings]
        else:
            raise ValueError(f"Unsupported pooling method for sentence transformer: {self.pooling}")

        if self.return_tensors == "np":
            embeddings = [e.cpu().numpy() if isinstance(e, torch.Tensor) else e for e in embeddings]

        self.logger.debug(f"Embeddings shapes after pooling: {', '.join([str(e.shape) for e in embeddings])}")

        return {k: e for k, e in zip(original_keys, embeddings)}
