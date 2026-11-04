"""
Regenerate the cached cell-line embeddings, the cell half of
``notebooks/context_embeddings.ipynb`` with the analysis left out.

Each cell line is embedded from its Cellosaurus description with a sentence-transformer,
PCA-compressed, and written to ``TACKAI_CACHE`` as the three npz files
:data:`tackai.fusion.context.CONTEXT_FILES` names for the ``cell`` block. A key Cellosaurus
does not know keeps its own entry and is embedded from the "unknown" description, so it
resolves to a vector instead of being dropped.

Only the ``cell`` block is regenerated. That notebook also writes ``recruiter`` and ``target``
protein tables with ``pooling=sum``; those are the superseded generation —
``CONTEXT_FILES`` now points at the ``lse`` / ``mean_rm_pc2`` tables that
``scripts/generate_protein_embeddings.py`` writes, so regenerating them here would only add
dead files.

**The vocabulary is taken from the production path, not from the notebook's columns.**
``FusionData.build_table`` resolves a row's cell key as
``Cell_Line_STD_Accession`` -> ``Cell_Line_STD`` -> ``Cell_Line`` (or ``Cell_Line_ID`` ->
``Cell_Line`` in the TACKv2 layout), so a row whose accession is blank is looked up under free
text such as ``"Human PBMCs"``. The notebook reads the accession column alone and so never
builds a vector for those keys, which is why they were being dropped. Building the vocabulary
with ``build_table`` guarantees the table covers exactly the keys the encoder will be asked
for.

    python scripts/generate_cell_embeddings.py \\
        --csv data/AutoTPD-TPDdb.csv data/auto_tpd_plus/*.csv --write

Like the protein script, the number of PCA components is chosen from the data, so a larger
vocabulary can change it and with it the file name (``_pca47`` -> ``_pca49``); the script says
so loudly, and ``--pca-dims`` pins it. Nothing is written without ``--write``, because
refitting the PCA is a new embedding space that invalidates models trained in the old one.

Author: Stefano Ribes
"""
import os
import sys

# torch and xgboost each bundle their own libomp and segfault when both run multi-threaded in
# one process. Must be set before numpy/torch are imported.
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 1))

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")
load_dotenv(find_dotenv(usecwd=True))

from tackai.data.utils import get_cache_dir
from tackai.data.embeddings.utils import EmbeddingMixin

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

CELL_MODEL = "sentence-transformers/all-mpnet-base-v1"
CELL_POOLING = "mean"                       # pooling over the tokens of a cell-line description
NOT_FOUND = "Unknown cell line."            # CellEmbedding's description for a missing cell line
VARIANCE_TARGET, SIMILARITY_TARGET = 0.90, 0.95   # the notebook's PCA rule
PUBLISHED_PCA_DIMS = 47                     # width the published table carries, used only to warn
META_FILE = "context_embeddings_meta.json"


class StoredEmbedding(EmbeddingMixin):
    """Store for the precomputed vectors; saves and loads like the other TACK embeddings."""

    def _transform_batch(self, keys):
        raise KeyError(f"no cached embedding for {keys[:3]}")


def stem() -> str:
    return f"cell_embeddings_model=sentence-transformer_pooling={CELL_POOLING}"


def cache_files(k: int) -> dict:
    return {"full": f"{stem()}.npz", "pca": f"{stem()}_pca{k}.npz",
            "pca_model": f"{stem()}_pca{k}_model.npz"}


def load_vocab(files) -> list:
    """Cell keys the production table actually asks for, most frequent first, "unknown" last.

    Args:
        files: CSV paths.

    Returns:
        List of cell keys.
    """
    from tackai.fusion.data import build_table

    table = build_table(files)
    keys = table["cell_key"].astype("string").str.strip()
    keys = keys.mask(keys.isna() | (keys == "") | (keys.str.lower() == "nan"))
    n_missing = int(keys.isna().sum())
    counts = keys.dropna().value_counts()
    vocab = sorted(counts.index, key=lambda k: (-counts[k], k))
    vocab = [k for k in vocab if k != NOT_FOUND] + [NOT_FOUND]
    logger.info("%d rows read, %d without a cell key (-> %r), %d distinct keys",
                len(table), n_missing, NOT_FOUND, len(vocab))
    return vocab


def cosine_matrix(X: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity of the rows of ``X`` as an (n, n) array in [-1, 1]."""
    U = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return np.clip(U @ U.T, -1.0, 1.0)


def similarity_preservation(X: np.ndarray) -> pd.DataFrame:
    """Variance and similarity structure kept by the first k PCA components, k = 1 ... min(n-1, d)."""
    kmax = min(len(X) - 1, X.shape[1])
    pca = PCA(n_components=kmax, svd_solver="full").fit(X)
    Z, upper = pca.transform(X), np.triu_indices(len(X), k=1)
    full = cosine_matrix(X - pca.mean_)[upper]
    rows = []
    for k in range(1, kmax + 1):
        rows.append({"dims": k,
                     "variance": pca.explained_variance_ratio_[:k].sum(),
                     "similarity_rho": spearmanr(
                         cosine_matrix(Z[:, :k] @ pca.components_[:k])[upper], full).statistic})
    return pd.DataFrame(rows)


def choose_dims(table: pd.DataFrame, variance_target: float, similarity_target) -> int:
    """Fewest components reaching the variance target and, when given, the similarity target."""
    ok = table["variance"] >= variance_target - 1e-12
    if similarity_target is not None:
        ok &= table["similarity_rho"] >= similarity_target - 1e-12
    if not ok.any():
        raise ValueError(f"no number of components reaches variance >= {variance_target} and "
                         f"similarity >= {similarity_target}; loosen the targets or pin --pca-dims")
    return int(table.loc[ok, "dims"].min())


def compare_to_cached(path: Path, table: dict, atol: float) -> str:
    """One-line verdict comparing a freshly built table against the npz already on disk."""
    if not path.exists():
        return f"MISSING   {path.name} is not in the cache"
    with np.load(path, allow_pickle=False) as cached:
        cached_keys, new_keys = set(cached.keys()), set(table)
        shared = cached_keys & new_keys
        if not shared:
            return f"DIFFERS   {path.name}: no key in common with the rebuilt table"
        worst = max(float(np.abs(np.asarray(cached[k]) - table[k]).max()) for k in shared)
    verdict = "OK" if worst <= atol else "DIFFERS"
    note = "" if cached_keys == new_keys else (
        f"; +{len(new_keys - cached_keys)} new / -{len(cached_keys - new_keys)} dropped keys")
    return f"{verdict:9s} {path.name}: {len(shared)} shared keys, max abs diff {worst:.3e}{note}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate the cached cell-line embeddings (generation only, no analysis).")
    parser.add_argument("--csv", nargs="+", required=True, type=Path,
                        help="Curated CSV(s) whose cell-line keys define the vocabulary.")
    parser.add_argument("--cache-dir", default=None,
                        help="Directory to write the npz files to (default: TACKAI_CACHE).")
    parser.add_argument("--pca-dims", type=int, default=None,
                        help="Pin the PCA width instead of choosing it from the data.")
    parser.add_argument("--variance-target", type=float, default=VARIANCE_TARGET,
                        help=f"PCA variance target (default: {VARIANCE_TARGET}).")
    parser.add_argument("--similarity-target", type=float, default=SIMILARITY_TARGET,
                        help="PCA similarity target, or a negative value to drop the criterion "
                             f"(default: {SIMILARITY_TARGET}).")
    parser.add_argument("--recompute", action="store_true",
                        help="Re-encode every key instead of reusing the cached full embeddings.")
    parser.add_argument("--write", action="store_true",
                        help="Actually write the cache files. Without it the script only reports.")
    parser.add_argument("--verify", action="store_true",
                        help="Compare the rebuilt tables against the cached ones instead of "
                             "writing; exits non-zero on any difference.")
    parser.add_argument("--atol", type=float, default=1e-4,
                        help="Absolute tolerance used by --verify. Set above the float32 drift "
                             "of a non-deterministic transformer forward pass as amplified by "
                             "the PCA projection (default: 1e-4).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.write and args.verify:
        raise SystemExit("--write and --verify are mutually exclusive")
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(get_cache_dir())
    missing_csv = [str(p) for p in args.csv if not Path(p).exists()]
    if missing_csv:
        raise FileNotFoundError(f"CSV(s) not found: {missing_csv}")
    similarity_target = None if args.similarity_target < 0 else args.similarity_target

    vocab = load_vocab(args.csv)
    # Union in whatever the cached table already holds, so regenerating from one set of CSVs
    # never drops a key another dataset still looks up. The vectors already exist, so the only
    # cost is keeping them.
    cached_full = cache_dir / cache_files(PUBLISHED_PCA_DIMS)["full"]
    if cached_full.exists() and not args.recompute:
        with np.load(cached_full, allow_pickle=False) as data:
            carried = [k for k in data.keys() if k not in set(vocab)]
        if carried:
            logger.info("carrying %d key(s) held by the cached table but absent from these CSVs",
                        len(carried))
            vocab = [k for k in vocab if k != NOT_FOUND] + carried + [NOT_FOUND]

    from tackai.data.embeddings.cell_embeddings import CellEmbedding

    encoder = CellEmbedding(embeddings_type="sentence_transformer", pretrained_model=CELL_MODEL,
                            pooling=CELL_POOLING, not_found_description=NOT_FOUND,
                            filename=cache_files(PUBLISHED_PCA_DIMS)["full"], cache_dir=cache_dir,
                            load_from_cache=not args.recompute)
    todo = [k for k in vocab if k not in encoder]
    if todo:
        # A key Cellosaurus does not know is embedded from the "unknown" description but keeps
        # its own entry, so it resolves to a vector instead of raising at lookup time.
        known = sum(1 for k in todo if encoder.get_cell_description(k))
        logger.info("encoding %d key(s) not in the cache (%d with a Cellosaurus description, "
                    "%d falling back to %r)", len(todo), known, len(todo) - known, NOT_FOUND)
        encoder.transform(todo)
    logger.info("cell: %d from the cache, %d encoded", len(vocab) - len(todo), len(todo))

    table = {k: np.asarray(encoder[k], dtype=np.float32) for k in vocab}
    matrix = np.stack([table[k] for k in vocab])
    # One row per distinct vector: every unrecognised key shares the "unknown" vector, and
    # fitting the PCA on the duplicates would weight that one point hundreds of times over.
    uniq = np.sort(np.unique(matrix, axis=0, return_index=True)[1])
    fit = matrix[uniq].astype(np.float64)
    if args.pca_dims:
        k, chosen = args.pca_dims, f"pinned at {args.pca_dims}"
    else:
        k = choose_dims(similarity_preservation(fit), args.variance_target, similarity_target)
        chosen = f"chosen from the data ({args.variance_target:.2f} variance / "
        chosen += "no similarity target)" if similarity_target is None else f"{similarity_target:.2f} similarity)"
    pca = PCA(n_components=k, svd_solver="full").fit(fit)
    reduced = dict(zip(vocab, pca.transform(matrix.astype(np.float64)).astype(np.float32)))
    files = cache_files(k)
    logger.info("cell: %d keys, %d distinct vectors of dim %d -> PCA %d dims (%.1f%% variance, %s)",
                len(vocab), len(fit), matrix.shape[1], k,
                100 * pca.explained_variance_ratio_.sum(), chosen)
    if k != PUBLISHED_PCA_DIMS and not args.pca_dims:
        logger.warning("  !! cell chose %d components, but the published table has %d. The file "
                       "name changes to %s, so tackai.fusion.context.CONTEXT_FILES must be "
                       "updated to match (or pass --pca-dims %d to keep the old width).",
                       k, PUBLISHED_PCA_DIMS, files["pca"], PUBLISHED_PCA_DIMS)

    if args.verify:
        status = 0
        for kind, built in (("full", table), ("pca", reduced)):
            verdict = compare_to_cached(cache_dir / files[kind], built, args.atol)
            logger.info("  %s", verdict)
            status |= int(not verdict.startswith("OK"))
        return status
    if not args.write:
        logger.info("would write %s", ", ".join(files.values()))
        logger.info("nothing written (pass --write to overwrite the cache)")
        return 0

    encoder.embeddings = dict(table)
    encoder.save(filename=files["full"], cache_dir=cache_dir)
    StoredEmbedding(embeddings=reduced, filename=files["pca"], cache_dir=cache_dir).save()
    np.savez(cache_dir / files["pca_model"],
             mean_=pca.mean_.astype(np.float32),
             components_=pca.components_.astype(np.float32),
             explained_variance_ratio_=pca.explained_variance_ratio_.astype(np.float32))

    # The meta file also describes the notebook's protein blocks; only the cell entry is ours.
    meta_path = cache_dir / META_FILE
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.setdefault("blocks", {})["cell"] = {
        "n_keys": len(vocab), "n_distinct_vectors": int(len(fit)),
        "embedding_dim": int(matrix.shape[1]), "pca_dims": k,
        "explained_variance": float(pca.explained_variance_ratio_.sum()), "files": files}
    meta["cell"] = {"model": CELL_MODEL, "pooling": CELL_POOLING,
                    "key": "FusionData.build_table cell_key (accession, else cell-line text)",
                    "script": "scripts/generate_cell_embeddings.py",
                    "source": {"files": sorted(str(p) for p in args.csv)},
                    "variance_target": args.variance_target,
                    "similarity_target": similarity_target}
    meta["not_found_description"] = NOT_FOUND
    meta_path.write_text(json.dumps(meta, indent=1))

    for name in list(files.values()) + [META_FILE]:
        logger.info("%9.1f KB  %s", (cache_dir / name).stat().st_size / 1024, name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
