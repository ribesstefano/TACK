"""Biological context blocks, read from the cached PCA-reduced embedding tables.

The tables are produced by ``notebooks/context_embeddings.ipynb`` (cell lines),
``notebooks/protein_pooling_comparison.ipynb`` (POI and E3 ligase) and
``notebooks/assay_embeddings.ipynb`` (assay types), and live in ``TACKAI_CACHE``. They are
already reduced, which is why no PCA runs anywhere in this pipeline.

What happens to a key the tables do not contain depends on whether the vector can be
reconstructed at all:

* **cell** — a missing or blank accession gets the table's ``"Unknown cell line."`` vector;
  an accession that is simply absent raises, because silently substituting "unknown" for a
  real cell line would hide the mistake.
* **assay** — free text is canonised, and an unseen canonical type is embedded on the fly
  with the same sentence-transformer and projected through the cached PCA.
* **poi / e3** — an unseen sequence raises. It cannot be fixed on the fly: the winning
  poolings are ``lse`` (not yet in :class:`~tackai.data.embeddings.protein_embeddings.ProteinEmbedding`)
  and ``mean_rm_pc2``, which is fitted over a whole vocabulary and therefore undefined for a
  single new sequence. Supply the 640-d embedding yourself via :meth:`register_sequence`.
"""
import re
import unicodedata
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from tackai.data.utils import get_cache_dir

CELL_MODEL = "sentence-transformers/all-mpnet-base-v1"
ASSAY_MODEL = "sentence-transformers/all-mpnet-base-v1"
POI_ESM_MODEL = "facebook/esm2_t30_150M_UR50D"

_PROTEIN_STEM = "protein_embeddings_esm_model=facebook-esm2_t30_150M_UR50D"
_ASSAY_STEM = "vocab=open20_model=all-mpnet-base-v1_pooling=mean"

#: Block name -> the cached table and its PCA side file.
CONTEXT_FILES: Dict[str, Dict[str, str]] = {
    "cell": {
        "table": "cell_embeddings_model=sentence-transformer_pooling=mean_pca47.npz",
        "pca_model": "cell_embeddings_model=sentence-transformer_pooling=mean_pca47_model.npz",
    },
    "poi": {
        "table": f"{_PROTEIN_STEM}_layer=18_pooling=lse_window=1022_block=poi_pca51.npz",
        "pca_model": f"{_PROTEIN_STEM}_layer=18_pooling=lse_window=1022_block=poi_pca51_model.npz",
    },
    "e3": {
        "table": f"{_PROTEIN_STEM}_layer=30_pooling=mean_rm_pc2_window=1022_block=e3_pca7.npz",
        "pca_model": f"{_PROTEIN_STEM}_layer=30_pooling=mean_rm_pc2_window=1022_block=e3_pca7_model.npz",
    },
    "combined": {
        "table": f"{_PROTEIN_STEM}_layer=18_pooling=lse_window=1022_block=combined_pca52.npz",
        "pca_model": f"{_PROTEIN_STEM}_layer=18_pooling=lse_window=1022_block=combined_pca52_model.npz",
    },
    "assay": {
        "table": f"assay_embeddings_{_ASSAY_STEM}_pca8.npz",
        "pca_model": f"assay_pca_{_ASSAY_STEM}_pca8.npz",
    },
}

CONTEXT_BLOCKS = ("e3", "cell", "poi", "assay", "assay_time")
SEQUENCE_BLOCKS = ("poi", "e3")


def _is_missing(value) -> bool:
    """True for ``None``, NaN, pandas NA and blank strings."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return value is not value      # pandas NA compares unequal to itself


def normalize_assay(value) -> str:
    """Normalise a raw assay string to the open-vocabulary key.

    Ported from ``notebooks/assay_embeddings.ipynb``: text no rule recognises is passed
    through, so the vocabulary grows with the data instead of collapsing to "other".

    Args:
        value: Raw assay description, possibly missing.

    Returns:
        The canonical assay key, or ``"unknown"`` when the value is missing.
    """
    if _is_missing(value):
        return "unknown"
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    text = re.sub(r"[‐‑–—−]", "-", text)
    text = re.sub(r"\s+", " ", text)
    if text in {"", "nan", "none", "missing", "unknown", "n/a"}:
        return "unknown"
    if "capillary" in text or "simple western" in text:
        return "capillary immunoassay"
    if "htrf" in text:
        return "htrf"
    if "hibit" in text:
        return "hibit"
    if "nanoluc" in text or "nanoluciferase" in text:
        return "nanoluc reporter"
    if "msd" in text or "meso scale" in text:
        return "msd"
    if "flow cytometr" in text or "facs" in text:
        return "flow cytometry"
    if "incell hunter" in text or "incellhunter" in text:
        return "incell hunter"
    if "in-cell" in text or "in cell" in text or "cytoblot" in text:
        return "in-cell immunoassay + western blot" if "/western blot" in text else "in-cell immunoassay"
    if "dot blot" in text:
        return "dot blot"
    western = "western" in text or "immunoblot" in text
    if western and "elisa" in text:
        return "elisa + western blot"
    if western:
        return "western blot"
    if "elisa" in text:
        return "elisa"
    if "high-content" in text or "high content" in text:
        return "high-content imaging"
    if "fluorescence" in text or "immunofluorescence" in text or "imaging" in text:
        return "fluorescence imaging"
    if "mts" in text or "cck8" in text or "celltiter" in text:
        return "viability assay"
    return text


class ContextEncoder:
    """Look up the cached context embedding of every POI, E3 ligase, cell line and assay.

    Args:
        cache_dir: Directory holding the npz tables (default: ``TACKAI_CACHE``).
        protein_space: ``"per_block"`` reads POI from its own 51-d table and E3 from its 7-d
            table; ``"combined"`` reads both from the shared 52-d space.
    """

    NOT_FOUND = "Unknown cell line."

    def __init__(self, cache_dir: Optional[str] = None, protein_space: str = "per_block"):
        if protein_space not in {"per_block", "combined"}:
            raise ValueError(f"protein_space must be 'per_block' or 'combined', got {protein_space!r}")
        self.cache_dir = Path(cache_dir or get_cache_dir())
        self.protein_space = protein_space
        self._tables: Dict[str, Dict[str, np.ndarray]] = {}
        self._sentence_model = None

    def files(self, block: str) -> Dict[str, str]:
        """The cached table and PCA side-file names backing a block.

        Args:
            block: ``"poi"``, ``"e3"``, ``"cell"`` or ``"assay"``.

        Returns:
            Mapping with ``"table"`` and ``"pca_model"`` filenames.
        """
        if block in SEQUENCE_BLOCKS and self.protein_space == "combined":
            return CONTEXT_FILES["combined"]
        if block not in CONTEXT_FILES:
            raise KeyError(f"unknown context block {block!r}")
        return CONTEXT_FILES[block]

    def table(self, block: str) -> Dict[str, np.ndarray]:
        """The block's lookup table, loaded once and kept."""
        if block not in self._tables:
            path = self.cache_dir / self.files(block)["table"]
            if not path.exists():
                raise FileNotFoundError(
                    f"context table for block {block!r} not found: {path}. Set TACKAI_CACHE to "
                    "the directory holding the cached embeddings (see the README)."
                )
            self._tables[block] = dict(np.load(path))
        return self._tables[block]

    def dim(self, block: str) -> int:
        """Width of a block's vectors."""
        if block == "assay_time":
            return 1
        return int(next(iter(self.table(block).values())).shape[0])

    def _pca(self, block: str):
        """``(mean_, components_)`` of the block's cached PCA."""
        side = np.load(self.cache_dir / self.files(block)["pca_model"])
        return side["mean_"], side["components_"]

    def _encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Embed free text with the same sentence-transformer the assay table was built with."""
        if self._sentence_model is None:
            from sentence_transformers import SentenceTransformer
            self._sentence_model = SentenceTransformer(ASSAY_MODEL, device="cpu")
        return np.asarray(self._sentence_model.encode(list(texts)), dtype=np.float32)

    def register_sequence(self, block: str, sequence: str, embedding: np.ndarray) -> None:
        """Add a sequence to a protein table by projecting its full ESM embedding.

        The escape hatch for a POI or ligase the cache does not know: supply the 640-d
        embedding and it is projected with the block's cached PCA, so the vector lands in the
        same space as the trained models expect.

        Args:
            block: ``"poi"`` or ``"e3"``.
            sequence: Amino-acid sequence, used as the lookup key.
            embedding: Full-dimension ESM embedding of that sequence.
        """
        if block not in SEQUENCE_BLOCKS:
            raise ValueError(f"register_sequence applies to {SEQUENCE_BLOCKS}, got {block!r}")
        mean_, components_ = self._pca(block)
        emb = np.asarray(embedding, dtype=np.float64).reshape(-1)
        if emb.shape[0] != mean_.shape[0]:
            raise ValueError(f"{block}: expected a {mean_.shape[0]}-d embedding, got {emb.shape[0]}-d")
        self.table(block)[str(sequence).strip()] = ((emb - mean_) @ components_.T).astype(np.float32)

    def encode(self, block: str, keys: Sequence) -> np.ndarray:
        """Vectors of every key of a block, in order.

        Args:
            block: ``"poi"``, ``"e3"``, ``"cell"`` or ``"assay"``.
            keys: Sequences (POI/E3), Cellosaurus accessions (cell) or raw text (assay).

        Returns:
            Array of shape ``(len(keys), dim(block))``, ``float32``.

        Raises:
            KeyError: For a POI/E3 sequence or a cell accession that is not in the table.
        """
        keys = list(keys)
        if not keys:
            return np.empty((0, self.dim(block)), dtype=np.float32)
        if block == "assay":
            return self._encode_assay(keys)
        table = self.table(block)
        rows = []
        for key in keys:
            if _is_missing(key):
                if block != "cell":
                    raise KeyError(f"{block}: a sequence is required, got a missing value")
                rows.append(table[self.NOT_FOUND])
                continue
            name = str(key).strip()
            if name not in table:
                raise KeyError(
                    f"{block}: {'sequence' if block in SEQUENCE_BLOCKS else 'key'} not in the "
                    f"cached table: {name[:30]}"
                    + ("… — register it with register_sequence(block, sequence, embedding)"
                       if block in SEQUENCE_BLOCKS else "")
                )
            rows.append(table[name])
        return np.stack(rows).astype(np.float32)

    def encodable(self, block: str, keys: Sequence) -> np.ndarray:
        """Which keys of a block this encoder can turn into a vector.

        Lets a caller building a training table drop the rows it cannot encode instead of
        hitting the :meth:`encode` exception. Assay text is always encodable, because unseen
        types fall back to an on-the-fly embedding.

        Args:
            block: Context block name.
            keys: Keys to check.

        Returns:
            Boolean array, one entry per key.
        """
        if block == "assay":
            return np.ones(len(list(keys)), dtype=bool)
        table = self.table(block)
        out = []
        for key in keys:
            if _is_missing(key):
                out.append(block == "cell")        # a missing cell line has a vector; a sequence does not
            else:
                out.append(str(key).strip() in table)
        return np.array(out, dtype=bool)

    def _encode_assay(self, raws: Sequence) -> np.ndarray:
        """Assay vectors, embedding canonical types the table does not hold."""
        table = self.table("assay")
        names = [normalize_assay(r) for r in raws]
        misses = sorted({n for n in names if n not in table})
        if misses:
            mean_, components_ = self._pca("assay")
            projected = (self._encode_texts(misses).astype(np.float64) - mean_) @ components_.T
            fallback = dict(zip(misses, projected.astype(np.float32)))
        else:
            fallback = {}
        return np.stack([table.get(n, fallback.get(n)) for n in names]).astype(np.float32)

    def encode_context(self, record: dict) -> Dict[str, np.ndarray]:
        """Encode one experimental context into its blocks.

        Args:
            record: Mapping with ``poi_seq``, ``e3_seq``, ``cell_id`` (or ``cell_key``),
                ``assay`` and optionally ``assay_time``.

        Returns:
            Mapping of block name to a ``(1, dim)`` array; a missing ``assay_time`` is NaN,
            which the preprocessor imputes with the training median.
        """
        cell = record.get("cell_id", record.get("cell_key"))
        time = record.get("assay_time")
        return {
            "e3": self.encode("e3", [record.get("e3_seq")]),
            "cell": self.encode("cell", [cell]),
            "poi": self.encode("poi", [record.get("poi_seq")]),
            "assay": self.encode("assay", [record.get("assay")]),
            "assay_time": np.array([[np.nan if _is_missing(time) else float(time)]], dtype=np.float32),
        }
