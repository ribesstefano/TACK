"""
Regenerate the cached POI / E3-ligase protein embeddings, the generation half of
``notebooks/protein_pooling_comparison.ipynb`` with the pooling sweep and its analysis left out.

The notebook scores ~1,600 (block, model, layer, pooling) variants and then writes the winner
of each block to ``TACKAI_CACHE``. The winners are now fixed — they are what
:data:`tackai.fusion.context.CONTEXT_FILES` names — so this script skips the sweep and runs
only the three winning combinations in :data:`BLOCK_SPEC`. One ESM pass per sequence serves
all three, because they need just two layers and two poolings between them.

Per block it writes the same three npz files the notebook wrote: the full table, the
PCA-compressed table, and the fitted PCA.

    python scripts/generate_protein_embeddings.py \\
        --csv data/AutoTPD-TPDdb.csv data/auto_tpd_plus/*.csv

Two things to know before running it over a *larger* vocabulary than the published one:

* **The E3 table is not incremental.** ``mean_rm_pc2`` projects out the two leading principal
  components of the block's own table (Arora et al.'s common-component removal), so those
  directions — and therefore every E3 vector — depend on which sequences are in the
  vocabulary. Adding a sequence rewrites the whole block. ``lse``, used by ``poi`` and
  ``combined``, is per-sequence and does not have this property, but their PCA is refitted
  here too, so treat every regenerated table as a new embedding space.
* **The number of PCA components is chosen from the data**, by the notebook's rule: the fewest
  that explain ``--variance-target`` of the variance *and* keep ``--similarity-target`` of the
  similarity structure. A different vocabulary can therefore choose a different ``k``, which
  changes the file name (``_pca51`` -> ``_pca53``) and so needs ``CONTEXT_FILES`` updated.
  The script says so loudly when ``k`` differs from :data:`PUBLISHED_PCA_DIMS`; pass
  ``--pca-dims poi=51`` to pin it instead.

Because a regenerated space invalidates models trained in the old one, the default is
``--dry-run``-safe: nothing is written unless ``--write`` is given.

    python scripts/generate_protein_embeddings.py --csv ... --verify   # reproduce the cache?
    python scripts/generate_protein_embeddings.py --csv ... --write    # overwrite it

Author: Stefano Ribes
"""
import os
import sys

# torch and xgboost each bundle their own libomp and segfault when both run multi-threaded in
# one process. Must be set before numpy/torch are imported. This script never imports xgboost,
# so it is only set when the caller has not chosen a thread count of their own.
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 1))

import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from scipy.special import logsumexp
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")
load_dotenv(find_dotenv(usecwd=True))

from tackai.data.utils import get_cache_dir
from tackai.data.embeddings.utils import EmbeddingMixin
from tackai.data.embeddings.protein_embeddings import sanitize_sequence

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "facebook/esm2_t30_150M_UR50D"   # 640-d, 30 layers: the reference model
WINDOW = 1022                                 # residues per ESM-2 window (1024 tokens minus BOS/EOS)
LSE_TAU = 1.0                                 # temperature of the smooth-max pooling
VARIANCE_TARGET, SIMILARITY_TARGET = 0.90, 0.95   # the notebook's PCA rule

#: block -> (layer, pooling): the winners of the notebook's sweep, i.e. what CONTEXT_FILES names.
BLOCK_SPEC = {"e3": (30, "mean_rm_pc2"), "poi": (18, "lse"), "combined": (18, "lse")}

#: The PCA widths the published tables carry, used only to warn when a rerun disagrees.
PUBLISHED_PCA_DIMS = {"e3": 7, "poi": 51, "combined": 52}

SEQ_COLUMNS = {"e3": "Recruiter_Sequence", "poi": "Degradation_Target_Sequence"}
GENE_COLUMNS = {"e3": "Recruiter_Gene", "poi": "Degradation_Target_Gene"}

#: Poolings applied per sequence, vs. those fitted over a whole block's table.
SEQUENCE_POOLINGS = ("lse", "mean")
TABLE_POOLINGS = {"mean_rm_pc2": 2}      # name -> number of leading components removed
TABLE_BASE = "mean"                      # the per-sequence pooling a table pooling starts from


class StoredEmbedding(EmbeddingMixin):
    """Store for the precomputed vectors; saves and loads like the other TACK embeddings."""

    def _transform_batch(self, keys):
        raise KeyError(f"no cached embedding for {keys[:3]}")


# ------------------------------- vocabulary ------------------------------------------------- #
def load_sequences(files) -> pd.DataFrame:
    """Collect the distinct recruiter and target sequences of every CSV, with gene names and row counts.

    Args:
        files: CSV paths.

    Returns:
        DataFrame indexed by sanitised sequence with the columns ``block`` (``"e3"``, ``"poi"``
        or ``"both"``), ``gene`` (most frequent gene name, upper case), ``rows`` (CSV rows) and
        ``length``.
    """
    records = {}
    for path in files:
        header = pd.read_csv(path, nrows=0).columns
        for block, seq_col in SEQ_COLUMNS.items():
            gene_col = GENE_COLUMNS[block]
            if seq_col not in header:
                continue
            df = pd.read_csv(path, usecols=[c for c in (seq_col, gene_col) if c in header],
                             dtype="string")
            seqs = df[seq_col].str.strip()
            genes = (df[gene_col].str.strip().str.upper() if gene_col in df
                     else pd.Series(pd.NA, index=df.index, dtype="string"))
            for raw, gene in zip(seqs, genes):
                if pd.isna(raw) or not raw or raw.lower() == "nan":
                    continue
                rec = records.setdefault(sanitize_sequence(raw),
                                         {"blocks": set(), "genes": {}, "rows": 0})
                rec["blocks"].add(block)
                rec["rows"] += 1
                if not pd.isna(gene) and gene:
                    rec["genes"][gene] = rec["genes"].get(gene, 0) + 1
    if not records:
        raise ValueError("no sequences found: none of the CSVs has a recruiter or target sequence column")
    rows = [{"sequence": s,
             "block": "both" if len(r["blocks"]) == 2 else next(iter(r["blocks"])),
             "gene": max(r["genes"], key=r["genes"].get) if r["genes"] else "?",
             "rows": r["rows"], "length": len(s)}
            for s, r in records.items()]
    return pd.DataFrame(rows).set_index("sequence").sort_values("rows", ascending=False)


def build_vocab(seq_info: pd.DataFrame) -> dict:
    """Per-block sequence lists. ``combined`` is E3 first, then the POI sequences not already in it."""
    vocab = {"e3": [s for s in seq_info.index if seq_info.loc[s, "block"] in ("e3", "both")],
             "poi": [s for s in seq_info.index if seq_info.loc[s, "block"] in ("poi", "both")]}
    e3_set = set(vocab["e3"])
    vocab["combined"] = vocab["e3"] + [s for s in vocab["poi"] if s not in e3_set]
    return vocab


# ------------------------------- pooling ---------------------------------------------------- #
def pool_lse(H: np.ndarray, tau: float = LSE_TAU) -> np.ndarray:
    """Smooth maximum over residues: ``tau * log(mean(exp(H / tau)))``, computed stably."""
    return tau * (logsumexp(H / tau, axis=0) - np.log(len(H)))


def pool_mean(H: np.ndarray) -> np.ndarray:
    return H.mean(axis=0)


POOL_FN = {"lse": pool_lse, "mean": pool_mean}


def remove_top_components(X: np.ndarray, k: int) -> np.ndarray:
    """Remove the first ``k`` principal components of a table (common-component removal).

    Args:
        X: Array of shape (n, d), one row per sequence.
        k: Number of leading components to project out.

    Returns:
        Array of the same shape, with the k leading directions of the centred table removed.
    """
    pca = PCA(n_components=min(k, min(X.shape) - 1), svd_solver="full").fit(X)
    C = pca.components_
    return X - (X - pca.mean_) @ C.T @ C


# ------------------------------- the ESM pass ----------------------------------------------- #
def window_sequence(seq: str, size: int = WINDOW):
    """Split a sequence into ``ceil(len / size)`` balanced windows (lengths differ by at most one)."""
    if not seq:
        raise ValueError("cannot embed an empty sequence")
    bounds = np.linspace(0, len(seq), math.ceil(len(seq) / size) + 1).round().astype(int)
    return [seq[a:b] for a, b in zip(bounds[:-1], bounds[1:])]


def residue_states(seq: str, model, tokenizer, layers, device: str) -> dict:
    """Per-residue hidden states of one sequence, per requested layer.

    One forward pass per window; BOS/EOS are dropped and the windows concatenated, so the
    result has one row per residue of ``seq``. Attentions are not requested — unlike the
    notebook's sweep, no pooling here needs them, and asking for them costs a lot of memory
    on a 1022-residue window.

    Args:
        seq: Amino-acid sequence.
        model: A loaded ESM model in eval mode.
        tokenizer: Its tokenizer.
        layers: 1-based layer indices to keep.
        device: Torch device the model is on.

    Returns:
        Dict ``layer -> H`` with ``H`` of shape (len(seq), D), float32.
    """
    import torch

    parts = {int(l): [] for l in layers}
    for window in window_sequence(seq):
        inputs = tokenizer(window, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        for l in parts:
            h = out.hidden_states[l][0].float().cpu().numpy()   # (T, D), T = len(window) + 2
            parts[l].append(h[1:-1])
        del out
    return {l: np.concatenate(p).astype(np.float32) for l, p in parts.items()}


def pooled_vectors(sequences, layers, poolings, model_name: str, device: str,
                   cache_path: Path, recompute: bool) -> dict:
    """Pool every (layer, pooling) of every sequence, with an on-disk cache keyed per entry.

    Re-running after adding sequences only encodes the new ones, which matters because a full
    pass over ~1,100 sequences is the expensive part of this script.

    Returns:
        Dict ``(layer, pooling) -> {sequence: vector}``.
    """
    wanted = {f"{l}|{p}|{s}" for l in layers for p in poolings for s in sequences}
    store = {}
    if cache_path.exists():
        # Every key is kept, not just the wanted ones: this file is also the pooling sweep's
        # cache (every model, layer and strategy the notebook scored), and it is rewritten
        # below. Narrowing it to what this run needs would silently throw the rest away.
        with np.load(cache_path, allow_pickle=False) as data:
            store = {k: data[k] for k in data.keys()}
        logger.info("pooled cache: %d of %d needed vectors reusable from %s (%d entries in total)",
                    len(wanted & set(store)), len(wanted), cache_path.name, len(store))
    if recompute:
        # Drop only what this run is asked to redo, leaving the rest of the sweep intact.
        store = {k: v for k, v in store.items() if k not in wanted}
    todo = sorted({s for s in sequences
                   if any(f"{l}|{p}|{s}" not in store for l in layers for p in poolings)})
    if todo:
        import torch
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).eval().to(device)
        t0 = time.time()
        for n, seq in enumerate(todo, 1):
            for l, H in residue_states(seq, model, tokenizer, layers, device).items():
                for p in poolings:
                    store[f"{l}|{p}|{seq}"] = np.asarray(POOL_FN[p](H), dtype=np.float32)
            if n % 25 == 0 or n == len(todo):
                logger.info("  encoded %d/%d sequences (%.0fs elapsed)", n, len(todo), time.time() - t0)
        del model
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Written beside the target and renamed over it: this file carries the whole pooling
        # sweep, and a crash midway through a direct write would leave it truncated.
        # The name must still end in .npz: np.savez appends the suffix when it does not,
        # and would then write somewhere other than where the rename looks for it.
        tmp = cache_path.with_name(cache_path.stem + ".tmp.npz")
        np.savez(tmp, **store)
        tmp.replace(cache_path)
        logger.info("pooled %d sequences x %d layers x %d poolings in %.0fs -> %s",
                    len(todo), len(layers), len(poolings), time.time() - t0, cache_path.name)
    return {(l, p): {s: store[f"{l}|{p}|{s}"] for s in sequences}
            for l in layers for p in poolings}


def table_for(block: str, vocab: dict, pooled: dict) -> np.ndarray:
    """The (n_sequences, dim) table of one block, in vocabulary order.

    A table-level pooling is fitted on this block's own ``TABLE_BASE`` table, which is what
    makes ``combined`` a block of its own rather than a concatenation of the other two.
    """
    layer, pooling = BLOCK_SPEC[block]
    seqs = vocab[block]
    if pooling in SEQUENCE_POOLINGS:
        return np.stack([pooled[(layer, pooling)][s] for s in seqs]).astype(np.float64)
    base = np.stack([pooled[(layer, TABLE_BASE)][s] for s in seqs]).astype(np.float64)
    return np.asarray(remove_top_components(base, TABLE_POOLINGS[pooling]), dtype=np.float64)


# ------------------------------- PCA width -------------------------------------------------- #
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


def stem_of(block: str) -> str:
    layer, pooling = BLOCK_SPEC[block]
    return (f"protein_embeddings_esm_model={MODEL_NAME.replace('/', '-')}"
            f"_layer={layer}_pooling={pooling}_window={WINDOW}_block={block}")


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
    return (f"{verdict:9s} {path.name}: {len(shared)} shared keys, "
            f"max abs diff {worst:.3e}{note}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate the cached POI/E3 protein embeddings (generation only, no sweep).")
    parser.add_argument("--csv", nargs="+", required=True, type=Path,
                        help="Curated CSV(s) whose recruiter/target sequence columns define the vocabulary.")
    parser.add_argument("--blocks", nargs="+", default=list(BLOCK_SPEC), choices=list(BLOCK_SPEC),
                        help="Blocks to write (default: all of e3, poi, combined).")
    parser.add_argument("--cache-dir", default=None,
                        help="Directory to write the npz files to (default: TACKAI_CACHE).")
    parser.add_argument("--pooled-cache", default=None,
                        help="Path of the reusable pooled-vector npz "
                             "(default: <cache-dir>/protein_pooling/pooled_esm2_150M.npz).")
    parser.add_argument("--model", default=MODEL_NAME, help=f"ESM model id (default: {MODEL_NAME}).")
    parser.add_argument("--device", default="cpu", help="Torch device for the ESM pass (default: cpu).")
    parser.add_argument("--pca-dims", nargs="+", default=[], metavar="BLOCK=K",
                        help="Pin a block's PCA width instead of choosing it from the data, "
                             "e.g. --pca-dims poi=51 combined=52 e3=7.")
    parser.add_argument("--variance-target", type=float, default=VARIANCE_TARGET,
                        help=f"PCA variance target (default: {VARIANCE_TARGET}).")
    parser.add_argument("--similarity-target", type=float, default=SIMILARITY_TARGET,
                        help=f"PCA similarity target, or a negative value to drop the criterion "
                             f"(default: {SIMILARITY_TARGET}).")
    parser.add_argument("--recompute", action="store_true",
                        help="Ignore the pooled-vector cache and run the ESM passes again.")
    parser.add_argument("--write", action="store_true",
                        help="Actually write the cache files. Without it the script only reports "
                             "what it would write, because a regenerated table is a new "
                             "embedding space that invalidates models trained in the old one.")
    parser.add_argument("--verify", action="store_true",
                        help="Compare the rebuilt tables against the cached ones instead of "
                             "writing; exits non-zero on any difference.")
    parser.add_argument("--atol", type=float, default=1e-5,
                        help="Absolute tolerance used by --verify (default: 1e-5).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.write and args.verify:
        raise SystemExit("--write and --verify are mutually exclusive")
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(get_cache_dir())
    missing = [str(p) for p in args.csv if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"CSV(s) not found: {missing}")
    pinned = {}
    for item in args.pca_dims:
        block, _, value = item.partition("=")
        if block not in BLOCK_SPEC or not value.isdigit():
            raise SystemExit(f"--pca-dims expects BLOCK=K with BLOCK in {list(BLOCK_SPEC)}, got {item!r}")
        pinned[block] = int(value)
    similarity_target = None if args.similarity_target < 0 else args.similarity_target

    seq_info = load_sequences(args.csv)
    vocab = build_vocab(seq_info)
    both = int((seq_info["block"] == "both").sum())
    logger.info("%d distinct sequences need an ESM pass (%d appear as both recruiter and target)",
                len(vocab["combined"]), both)
    for block in BLOCK_SPEC:
        lengths = [len(s) for s in vocab[block]]
        logger.info("  %-9s %4d sequences, median %4d aa, max %5d aa, %d longer than %d",
                    block, len(lengths), int(np.median(lengths)), max(lengths),
                    sum(l > WINDOW for l in lengths), WINDOW)

    layers = sorted({BLOCK_SPEC[b][0] for b in args.blocks})
    poolings = sorted({BLOCK_SPEC[b][1] if BLOCK_SPEC[b][1] in SEQUENCE_POOLINGS else TABLE_BASE
                       for b in args.blocks})
    pooled_cache = (Path(args.pooled_cache) if args.pooled_cache
                    else cache_dir / "protein_pooling" / "pooled_esm2_150M.npz")
    logger.info("ESM pass: layers %s, poolings %s, device %s", layers, poolings, args.device)
    pooled = pooled_vectors(vocab["combined"], layers, poolings, args.model, args.device,
                            pooled_cache, args.recompute)

    status = 0
    written, info = [], {}
    for block in args.blocks:
        layer, pooling = BLOCK_SPEC[block]
        X = table_for(block, vocab, pooled)
        # The PCA is fitted on distinct rows only: duplicated sequences would otherwise weight
        # one point many times over.
        uniq = np.sort(np.unique(X, axis=0, return_index=True)[1])
        if block in pinned:
            k = pinned[block]
            chosen = f"pinned at {k}"
        else:
            k = choose_dims(similarity_preservation(X[uniq]), args.variance_target, similarity_target)
            chosen = f"chosen from the data ({args.variance_target:.2f} variance / "
            chosen += "no similarity target)" if similarity_target is None else f"{similarity_target:.2f} similarity)"
        pca = PCA(n_components=k, svd_solver="full").fit(X[uniq])
        full = {s: v.astype(np.float32) for s, v in zip(vocab[block], X)}
        reduced = {s: v.astype(np.float32) for s, v in zip(vocab[block], pca.transform(X))}

        stem = stem_of(block)
        files = {"full": f"{stem}.npz", "pca": f"{stem}_pca{k}.npz",
                 "pca_model": f"{stem}_pca{k}_model.npz"}
        logger.info("%s: layer %s / %s -> %d-d, PCA %d dims (%.1f%% variance, %s)",
                    block, layer, pooling, X.shape[1], k,
                    100 * pca.explained_variance_ratio_.sum(), chosen)
        if k != PUBLISHED_PCA_DIMS.get(block) and block not in pinned:
            logger.warning("  !! %s chose %d components, but the published table has %d. The file "
                           "name changes to %s, so tackai.fusion.context.CONTEXT_FILES must be "
                           "updated to match (or pass --pca-dims %s=%d to keep the old width).",
                           block, k, PUBLISHED_PCA_DIMS[block], files["pca"], block,
                           PUBLISHED_PCA_DIMS[block])

        if args.verify:
            for kind, built in (("full", full), ("pca", reduced)):
                verdict = compare_to_cached(cache_dir / files[kind], built, args.atol)
                logger.info("  %s", verdict)
                status |= int(not verdict.startswith("OK"))
            continue
        info[block] = {"model": args.model, "layer": int(layer), "pooling": pooling,
                       "window": WINDOW, "n_sequences": len(vocab[block]),
                       "embedding_dim": int(X.shape[1]), "pca_dims": k,
                       "explained_variance": float(pca.explained_variance_ratio_.sum()),
                       "files": files}
        if not args.write:
            logger.info("  would write %s", ", ".join(files.values()))
            continue

        StoredEmbedding(embeddings=full, filename=files["full"], cache_dir=cache_dir).save()
        StoredEmbedding(embeddings=reduced, filename=files["pca"], cache_dir=cache_dir).save()
        np.savez(cache_dir / files["pca_model"],
                 mean_=pca.mean_.astype(np.float32),
                 components_=pca.components_.astype(np.float32),
                 explained_variance_ratio_=pca.explained_variance_ratio_.astype(np.float32))
        written.extend(files.values())

    if args.write:
        meta_file = "protein_pooling_meta.json"
        meta = {"script": "scripts/generate_protein_embeddings.py",
                "notebook": "notebooks/protein_pooling_comparison.ipynb",
                "variance_target": args.variance_target, "similarity_target": similarity_target,
                "pinned_pca_dims": pinned,
                "source": {"files": sorted(str(p) for p in args.csv)},
                "blocks": info}
        (cache_dir / meta_file).write_text(json.dumps(meta, indent=1))
        written.append(meta_file)
        for name in written:
            logger.info("%9.1f KB  %s", (cache_dir / name).stat().st_size / 1024, name)
    elif not args.verify:
        logger.info("nothing written (pass --write to overwrite the cache)")
    return status


if __name__ == "__main__":
    sys.exit(main())
