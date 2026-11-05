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


def add_source_arguments(parser) -> None:
    """Add the ``--dataset`` / ``--dataset-config`` / ``--revision`` group.

    Args:
        parser: The script's :class:`argparse.ArgumentParser`.
    """
    parser.add_argument("--dataset", default=DEFAULT_DATASET_REPO,
                        help=f"Hugging Face dataset repo the rows are loaded from "
                             f"(default: {DEFAULT_DATASET_REPO}, public).")
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG,
                        help=f"Config of --dataset to load "
                             f"(default: {DEFAULT_DATASET_CONFIG}).")
    parser.add_argument("--revision", default=None,
                        help="Revision of --dataset to load (default: the main branch). "
                             "Pin it to make a regeneration reproducible.")


def _resolve_revision(args) -> str:
    """The commit this run loads: ``--revision`` when given, else what the branch points at now.

    Resolved *before* loading, so the recorded provenance is the revision actually read and
    not whatever the branch has moved to by the time the meta is written — AutoTPD+ is a
    living release and does move. Falls back to the branch name when the Hub is unreachable,
    which is the offline case where ``datasets`` serves its cache.
    """
    if args.revision:
        return args.revision
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(args.dataset).sha
    except Exception:
        logger.warning("  could not reach the Hub to resolve a commit; recording \"main\". "
                       "Pass --revision to pin the source exactly.")
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

    config = args.dataset_config
    args.resolved_revision = _resolve_revision(args)
    logger.info("source: %s@%s, config %s", args.dataset, args.resolved_revision, config)
    ds = load_dataset(args.dataset, config, revision=args.resolved_revision)[DATASET_SPLIT]
    frame = ds.to_pandas()
    frame.attrs["name"] = config
    logger.info("  %d rows x %d columns", len(frame), frame.shape[1])
    return [frame]


def source_record(args) -> Dict:
    """What the meta JSON records about this run's source: repo id, config, revision.

    ``revision`` is the commit :func:`load_frames` resolved and loaded; ``pinned`` says whether
    it came from ``--revision`` or from wherever the branch stood at load time.
    """
    return {"dataset": args.dataset, "repo_type": "dataset",
            "config": args.dataset_config, "split": DATASET_SPLIT,
            "revision": getattr(args, "resolved_revision", args.revision or "main"),
            "pinned": bool(args.revision)}
