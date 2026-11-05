"""
Aggregate the generators' part manifests, stage the tables they name, and publish them to the
Hugging Face Hub (ailab-bio/TACK-Context-Encoding).

Each ``generate_*_embeddings.py`` script writes its own part manifest into TACKAI_CACHE
(``manifest_protein.json``, ``manifest_cell.json``, ``manifest_assay.json``), so run all three
at least once first. Then two phases, run separately so the staged folder can be inspected
before anything is uploaded:

    # 1. Stage locally (pure file copies + a generated manifest, no network writes)
    python scripts/embeddings/publish_fusion_context.py stage

    # 2. Review scripts/hf_staging/fusion_context/, then upload
    python scripts/embeddings/publish_fusion_context.py upload

Staging takes the table names from the part manifests, so an older generation still sitting in
the cache under another PCA width is not published. It refuses to write a partial repo: a part
manifest that is missing, or a file that is missing or was rewritten since its generator
recorded it, would otherwise surface far from this script, at inference time. The generated
manifest.json records each file's sha256, byte size and the dimensions read out of the npz
itself (never parsed from the filename), the block widths in the same `block_dims` shape
FusionEnsemble.save() writes, so it can be fed straight to FusionEnsemble._check_layout, and
each block's provenance as its generator recorded it.

Requires `huggingface_hub` to be installed and an authenticated session
(`hf auth login`, or an HF_TOKEN in the environment) with write access to the
ailab-bio org for the `upload` phase.

This is a maintainer tool for publishing a freshly generated TACKAI_CACHE. To publish the tables
one already-built FusionData instance is using, call its own `push_to_hub()` instead
(`tackai/fusion/data.py`) — both end in the same `stage_context_tables` /
`upload_context_tables` functions in `tackai/fusion/context.py`.

Author: Stefano Ribes
"""
import argparse
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _dataset import load_env

# HF_TOKEN (and TACKAI_CACHE) come from the repo's .env; HfApi reads HF_TOKEN from the
# environment, so the upload phase needs no --token flag. Existing variables win.
load_env()

from tackai.data.utils import get_cache_dir
from tackai.fusion.context import (DEFAULT_CONTEXT_REPO, sha256_of,  # noqa: F401  (re-exported for tests)
                                   stage_from_part_manifests, upload_context_tables)

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGING_ROOT = REPO_ROOT / "hf_staging" / "fusion_context"


def stage(cache_dir: Path, out_dir: Path, repo_id: str = DEFAULT_CONTEXT_REPO) -> Dict:
    """Thin CLI wrapper over :func:`tackai.fusion.context.stage_from_part_manifests`."""
    return stage_from_part_manifests(cache_dir, out_dir, repo_id=repo_id)


def cmd_stage(args: argparse.Namespace) -> None:
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(get_cache_dir())
    stage(cache_dir, Path(args.out), args.repo_id)


def upload(staged_dir: Path, repo_id: str, private: bool, commit_message: str) -> str:
    """Thin CLI wrapper over :func:`tackai.fusion.context.upload_context_tables`."""
    return upload_context_tables(staged_dir, repo_id, private, commit_message)


def cmd_upload(args: argparse.Namespace) -> None:
    staged_dir = Path(args.staged)
    if not staged_dir.exists():
        print(f"No staging directory found at {staged_dir}. Run the 'stage' phase first.")
        raise SystemExit(1)
    upload(staged_dir, args.repo_id, args.private, args.commit_message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    stage_p = sub.add_parser("stage")
    stage_p.add_argument("--cache-dir", default=None,
                         help="Defaults to TACKAI_CACHE / get_cache_dir().")
    stage_p.add_argument("--out", default=str(STAGING_ROOT))
    stage_p.add_argument("--repo-id", default=DEFAULT_CONTEXT_REPO,
                         help="Only used for the README text; the actual upload target is "
                              "chosen by the 'upload' phase's own --repo-id.")
    stage_p.set_defaults(func=cmd_stage)

    upload_p = sub.add_parser("upload")
    upload_p.add_argument("--repo-id", default=DEFAULT_CONTEXT_REPO)
    upload_p.add_argument("--staged", default=str(STAGING_ROOT))
    upload_p.add_argument("--private", action="store_true")
    upload_p.add_argument("--commit-message", default="Update fusion context embeddings")
    # No --subfolder: the context tables do not depend on the task, so one repo root holds
    # one generation of them. The per-task subfolders belong to the ensemble repo, where each
    # task has its own fitted models.
    upload_p.set_defaults(func=cmd_upload)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
