"""Shared dataset loading for the context-embedding scripts.

The vocabulary behind every cached context table (POI/E3 sequences, cell-line keys, assay
types) comes from the **AutoTPD+ dataset published on the Hugging Face Hub as
``ailab-bio/AutoTPDplus``**, config ``autotpdplus-protac``:

    protac = load_dataset("ailab-bio/AutoTPDplus", "autotpdplus-protac")

Nothing is read from or written to a CSV: :func:`load_frames` hands the loaded split
straight to the readers as a DataFrame, so a regeneration is reproducible from a repo id, a
config and a revision. The repo is public, so no token is needed. ``python-dotenv`` still
loads the repo's ``.env`` (:func:`load_env`) because the scripts read ``TACKAI_CACHE`` and
``HF_HOME`` from it.

``datasets`` caches the download under ``HF_HOME``, so a second run — and a Slurm job on a
node with no outbound network — reuses it without touching the Hub. Populate that cache on a
login node first, and export ``HF_DATASETS_OFFLINE=1`` in the job to make an accidental
fetch fail loudly instead of hanging.

Author: Stefano Ribes
"""
import json
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
from dotenv import find_dotenv, load_dotenv

logger = logging.getLogger(__name__)

#: Repo root, i.e. two levels above scripts/embeddings/.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: The reference dataset the published context tables were built from.
DEFAULT_DATASET_REPO = "ailab-bio/AutoTPDplus"

#: The config taken from it: the living PROTAC release. The repo also publishes the frozen
#: ``autotpd-protac`` benchmark and the molecular-glue configs; none of them is used here.
DEFAULT_DATASET_CONFIG = "autotpdplus-protac"

#: That config publishes its rows as a single ``train`` split.
DATASET_SPLIT = "train"


def load_env(repo_root: Path = REPO_ROOT) -> None:
    """Load ``<repo_root>/.env`` and the nearest ``.env`` above the cwd into the environment.

    The scripts read ``TACKAI_CACHE`` (where the tables are written) and ``HF_HOME`` (where
    the dataset is cached) from there. Existing variables win, as ``load_dotenv`` does not
    override.
    """
    load_dotenv(Path(repo_root) / ".env")
    load_dotenv(find_dotenv(usecwd=True))


def warn_if_new_generation(cache_dir: Path, block: str, table: str) -> None:
    """Warn when a run writes a table other than the one the cache's ``manifest.json`` names.

    The PCA width is chosen from the data and is part of the file name, so a new name means a
    new embedding space, which models trained in the old one cannot use. The encoder keeps
    reading the old table until the new generation is staged and installed.

    Args:
        cache_dir: Directory the tables are written to.
        block: Context block being written.
        table: File name of the PCA-reduced table this run writes.
    """
    path = Path(cache_dir) / "manifest.json"
    if not path.exists():
        return
    current = [name for name, entry in json.loads(path.read_text()).get("files", {}).items()
               if entry.get("block") == block and entry.get("role") == "table"]
    if current and table not in current:
        logger.warning("  !! %s writes %s, but %s names %s. This is a new embedding space: the "
                       "encoder keeps reading the old table until this generation is staged "
                       "with publish_fusion_context.py and installed with "
                       "FusionData.from_pretrained.", block, table, path.name, current[0])


def add_source_arguments(parser) -> None:
    """Add the ``--dataset`` / ``--dataset-config`` / ``--dataset-revision`` group.

    Args:
        parser: The script's :class:`argparse.ArgumentParser`.
    """
    parser.add_argument("--dataset", default=DEFAULT_DATASET_REPO,
                        help=f"Hugging Face dataset repo the rows are loaded from "
                             f"(default: {DEFAULT_DATASET_REPO}, public).")
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG,
                        help=f"Config of --dataset to load "
                             f"(default: {DEFAULT_DATASET_CONFIG}).")
    parser.add_argument("--dataset-revision", default=None,
                        help="Revision of --dataset to load (default: the main branch). "
                             "Pin it to make a regeneration reproducible.")
    parser.add_argument("--dataset-files", nargs="+", default=None,
                        help="Explicit training files inside the dataset repo; use this when "
                             "the repo also contains a separate test CSV.")


def _resolve_revision(args) -> str:
    """The commit this run loads: ``--dataset-revision`` when given, else where the branch points.

    Resolved *before* loading, so the recorded provenance is the revision actually read and
    not whatever the branch has moved to by the time the meta is written — AutoTPD+ is a
    living release and does move. Falls back to the branch name when the Hub is unreachable,
    which is the offline case where ``datasets`` serves its cache.
    """
    if args.dataset_revision:
        return args.dataset_revision
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(args.dataset).sha
    except Exception:
        logger.warning("  could not reach the Hub to resolve a commit; recording \"main\". "
                       "Pass --dataset-revision to pin the source exactly.")
        return "main"


def load_frames(args) -> List[pd.DataFrame]:
    """Load ``--dataset``'s config and return it as the frames the readers take.

    Returned as a list because every reader downstream takes a sequence of sources, and
    carrying ``attrs["name"]`` so :func:`tackai.fusion.data.build_table` can label its
    ``source`` column without a filename to fall back on.

    Records the resolved commit on ``args`` so :func:`source_record` reports the revision this
    run actually read.
    """
    from datasets import load_dataset

    config = None if args.dataset_config == "default" else args.dataset_config
    args.resolved_revision = _resolve_revision(args)
    logger.info("source: %s@%s, config %s", args.dataset, args.resolved_revision, config)
    files = getattr(args, "dataset_files", None)
    if files:
        # Resolve explicit files through the pinned Hub snapshot, then use a built-in
        # reader. In offline mode datasets cannot discover a repo's default config:
        # "default" can become a different cache key from the actual named config.
        from huggingface_hub import hf_hub_download

        paths = [hf_hub_download(args.dataset, name, repo_type="dataset",
                                 revision=args.resolved_revision) for name in files]
        readers = {".csv": "csv", ".parquet": "parquet", ".json": "json", ".jsonl": "json"}
        formats = {readers.get(Path(name).suffix.lower()) for name in files}
        if len(formats) != 1 or None in formats:
            raise ValueError("--dataset-files must contain one supported format: CSV, Parquet or JSON")
        ds = load_dataset(formats.pop(), data_files={DATASET_SPLIT: paths})[DATASET_SPLIT]
    else:
        ds = load_dataset(args.dataset, config, revision=args.resolved_revision)[DATASET_SPLIT]
    frame = ds.to_pandas()
    frame.attrs["name"] = config or args.dataset
    logger.info("  %d rows x %d columns", len(frame), frame.shape[1])
    return [frame]


def source_record(args) -> Dict:
    """What the meta JSON records about this run's source: repo id, config, revision.

    ``revision`` is the commit :func:`load_frames` resolved and loaded; ``pinned`` says whether
    it came from ``--dataset-revision`` or from wherever the branch stood at load time.
    """
    return {"dataset": args.dataset, "repo_type": "dataset",
            "config": args.dataset_config, "split": DATASET_SPLIT,
            "revision": getattr(args, "resolved_revision", args.dataset_revision or "main"),
            "pinned": bool(args.dataset_revision),
            "files": getattr(args, "dataset_files", None)}
