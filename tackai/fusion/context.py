"""Biological context blocks, read from the cached PCA-reduced embedding tables.

The tables are produced by ``notebooks/context_embeddings.ipynb`` (cell lines),
``notebooks/protein_pooling_comparison.ipynb`` (POI and E3 ligase) and
``notebooks/assay_embeddings.ipynb`` (assay types), and live in ``TACKAI_CACHE``. They are
already reduced, which is why no PCA runs anywhere in this pipeline.

They can also be fetched directly from the Hugging Face Hub dataset repo named by
:data:`DEFAULT_CONTEXT_REPO`, via
:meth:`~tackai.fusion.data.FusionData.from_pretrained`, which installs them into
``TACKAI_CACHE`` and verifies their content against a published manifest before using them.
That download never happens implicitly from inside this module: :meth:`ContextEncoder.table`
still raises a plain ``FileNotFoundError`` when a table is missing, so a kernel fit never
blocks on an unexpected network call.

The other direction — staging a cache's tables into a manifest-described folder and uploading
that folder (:func:`stage_context_tables` / :func:`upload_context_tables`) — lives here rather
than in ``scripts/publish_fusion_context.py`` because both the maintainer publish script and
:meth:`~tackai.fusion.data.FusionData.push_to_hub` need the exact same manifest format; the
script is a thin CLI wrapper over these two functions, not its own implementation.

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
import hashlib
import json
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

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

#: Hugging Face Hub dataset repo published by scripts/publish_fusion_context.py and read by
#: FusionData.from_pretrained(). A dataset repo, not a model repo: these tables are an input
#: to every fusion model, never the output of one.
DEFAULT_CONTEXT_REPO = "ailab-bio/TACK-fusion-context"

MANIFEST_FORMAT = "tack-fusion-context/v1"


def sha256_of(path: Path) -> str:
    """Streamed sha256 of a file, so staging never loads a whole npz into memory to hash it."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _npz_table_dims(path: Path) -> Tuple[int, int]:
    """(vector width, number of keys) of a context *table* npz, read from its arrays.

    A table npz holds one vector per lookup key (sequence, cell accession, assay type), all
    the same width.
    """
    with np.load(path) as data:
        keys = list(data.keys())
        dim = int(data[keys[0]].shape[0])
    return dim, len(keys)


def _npz_pca_dims(path: Path) -> Tuple[int, int]:
    """(full pre-PCA dimension, reduced dimension) of a PCA side-file npz.

    A PCA side file holds `mean_` (length = full dimension) and `components_` (shape
    reduced x full) — not per-key vectors, so it has neither a key count nor a single "dim"
    the way a table npz does.
    """
    with np.load(path) as data:
        reduced_dim, full_dim = data["components_"].shape
    return int(full_dim), int(reduced_dim)


def stage_context_tables(cache_dir: Union[str, Path], out_dir: Union[str, Path]) -> Dict:
    """Copy every :data:`CONTEXT_FILES` entry from ``cache_dir`` into ``out_dir``, with a manifest.

    Shared by ``scripts/publish_fusion_context.py`` (publishing a whole cache) and
    :meth:`~tackai.fusion.data.FusionData.push_to_hub` (publishing one instance's own cache),
    so both produce the exact manifest format :meth:`~tackai.fusion.data.FusionData.from_pretrained`
    expects.

    Args:
        cache_dir: Directory holding the cached context tables (``TACKAI_CACHE``).
        out_dir: Staging directory to write into; refused if it already exists.

    Returns:
        The manifest dict also written to ``out_dir / "manifest.json"``.

    Raises:
        SystemExit: If ``out_dir`` already exists, or any :data:`CONTEXT_FILES` entry is
            missing from ``cache_dir`` (every missing file is listed before exiting).
    """
    import tackai

    cache_dir, out_dir = Path(cache_dir), Path(out_dir)
    if out_dir.exists():
        print(f"Staging directory already exists: {out_dir}")
        print("Remove it (or move it aside) before re-staging, to avoid mixing stale and "
              "fresh files.")
        raise SystemExit(1)

    wanted = [(filename, block, role) for block, files in CONTEXT_FILES.items()
              for role, filename in files.items()]
    missing = [filename for filename, _, _ in wanted if not (cache_dir / filename).exists()]
    if missing:
        print(f"{len(missing)} file(s) missing from {cache_dir}:")
        for name in missing:
            print(f"  {name}")
        print("Refusing to publish a partial set of context tables.")
        raise SystemExit(1)

    manifest = {
        "format": MANIFEST_FORMAT,
        "tackai_version": getattr(tackai, "__version__", "unknown"),
        "block_dims": {"assay_time": 1}, "combined_dim": None,
        "models": {"cell": CELL_MODEL, "assay": ASSAY_MODEL, "poi": POI_ESM_MODEL,
                  "e3": POI_ESM_MODEL},
        "files": {},
    }
    out_dir.mkdir(parents=True)
    for filename, block, role in wanted:
        src = cache_dir / filename
        entry = {"block": block, "role": role, "sha256": sha256_of(src),
                 "bytes": src.stat().st_size}
        if role == "table":
            dim, n_keys = _npz_table_dims(src)
            entry["dim"], entry["n_keys"] = dim, n_keys
            if block == "combined":
                manifest["combined_dim"] = dim
            else:
                manifest["block_dims"][block] = dim
        else:
            full_dim, reduced_dim = _npz_pca_dims(src)
            entry["full_dim"], entry["reduced_dim"] = full_dim, reduced_dim
        manifest["files"][filename] = entry
        shutil.copy2(src, out_dir / filename)
        print(f"  staged {filename} ({block}/{role})")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\nStaged {len(wanted)} files + manifest.json -> {out_dir}")
    return manifest


def upload_context_tables(staged_dir: Union[str, Path], repo_id: str, private: bool,
                          commit_message: str, subfolder: Optional[str] = None) -> str:
    """Create (if needed) the dataset repo and upload every file in ``staged_dir``.

    Args:
        staged_dir: Directory written by :func:`stage_context_tables`.
        repo_id: Hugging Face Hub repo id.
        private: Create the repo as private if it does not exist yet.
        commit_message: Commit message for the upload.
        subfolder: Subdirectory within the repo to upload into, so several published
            snapshots can share one repo without colliding.

    Returns:
        The commit sha ``upload_folder`` reports.
    """
    from huggingface_hub import HfApi
    api = HfApi()
    print(f"Authenticated as: {api.whoami()['name']}")
    print(f"Creating (if needed) dataset repo {repo_id} ...")
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=private)
    destination = f"{repo_id}/{subfolder}" if subfolder else repo_id
    print(f"Uploading {staged_dir} -> {destination} ...")
    commit = api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(staged_dir),
                               path_in_repo=subfolder, commit_message=commit_message)
    sha = getattr(commit, "oid", str(commit))
    print(f"Done: https://huggingface.co/datasets/{repo_id} (commit {sha})")
    return sha


CONTEXT_BLOCKS = ("e3", "cell", "poi", "assay", "assay_time")
#: Assay duration [h] used where a measurement does not record one. A fixed constant on
#: purpose: nothing about it is learned from the training rows.
ASSAY_TIME_DEFAULT = 24.0
SEQUENCE_BLOCKS = ("poi", "e3")


def assay_time_or_default(value) -> float:
    """The assay duration in hours, or :data:`ASSAY_TIME_DEFAULT` when it is not recorded."""
    return ASSAY_TIME_DEFAULT if _is_missing(value) else float(value)


def _is_missing(value) -> bool:
    """True for ``None``, NaN, pandas NA and blank strings.

    ``pd.NA`` needs its own branch: it is a singleton, so the usual ``value is not value``
    trick does not catch it, and ``np.isnan(pd.NA)`` raises. Records built from a
    nullable-dtype frame (``Int64``, ``Float64``, ``string``) carry it routinely.
    """
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


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

    @property
    def dims(self) -> Dict[str, int]:
        """Width of every context block, discovered from the cached tables.

        Loads each table once (they are kept), so this is where a change of a PCA width in the
        cache becomes visible.
        """
        return {b: self.dim(b) for b in CONTEXT_BLOCKS}

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
            Mapping of block name to a ``(1, dim)`` array; a missing ``assay_time`` becomes
            :data:`ASSAY_TIME_DEFAULT`.
        """
        cell = record.get("cell_id", record.get("cell_key"))
        time = record.get("assay_time")
        return {
            "e3": self.encode("e3", [record.get("e3_seq")]),
            "cell": self.encode("cell", [cell]),
            "poi": self.encode("poi", [record.get("poi_seq")]),
            "assay": self.encode("assay", [record.get("assay")]),
            "assay_time": np.array([[assay_time_or_default(time)]], dtype=np.float32),
        }
