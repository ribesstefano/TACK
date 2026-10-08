"""
Stage and publish the fusion pipeline's context embedding tables to the
Hugging Face Hub (ailab-bio/TACK-fusion-context).

Two phases, run separately so the staged folder can be inspected before
anything is uploaded:

    # 1. Stage locally (pure file copies + a generated manifest, no network writes)
    python scripts/publish_fusion_context.py stage

    # 2. Review hf_staging/fusion_context/, then upload
    python scripts/publish_fusion_context.py upload

Staging resolves every (block, role) pair of tackai.fusion.context.CONTEXT_FILES against
TACKAI_CACHE (or --cache-dir) and refuses to write a partial repo: a cache missing even one
PCA side file breaks ContextEncoder.register_sequence and the assay fallback path at
inference time, far from this script. The generated manifest.json records each file's
sha256, byte size, and the dimensions read out of the npz itself (never parsed from the
filename), plus the block widths in the same `block_dims` shape FusionEnsemble.save() writes,
so it can be fed straight to FusionEnsemble._check_layout.

Requires `huggingface_hub` to be installed and an authenticated session
(`hf auth login`, or an HF_TOKEN in the environment) with write access to the
ailab-bio org for the `upload` phase.

This is a maintainer tool for publishing an arbitrary TACKAI_CACHE. To publish the tables one
already-built FusionData instance is using, call its own `push_to_hub()` instead
(`tackai/fusion/data.py`) — both this script and that method call the same
`stage_context_tables` / `upload_context_tables` functions in `tackai/fusion/context.py`.

Author: Stefano Ribes
"""
import argparse
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tackai.data.utils import get_cache_dir
from tackai.fusion.context import (DEFAULT_CONTEXT_REPO, sha256_of,  # noqa: F401  (re-exported for tests)
                                   stage_context_tables, upload_context_tables)

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGING_ROOT = REPO_ROOT / "hf_staging" / "fusion_context"


def stage(cache_dir: Path, out_dir: Path, repo_id: str = DEFAULT_CONTEXT_REPO) -> Dict:
    """Thin CLI wrapper over :func:`tackai.fusion.context.stage_context_tables`."""
    return stage_context_tables(cache_dir, out_dir, repo_id=repo_id)


def cmd_stage(args: argparse.Namespace) -> None:
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(get_cache_dir())
    stage(cache_dir, Path(args.out), args.repo_id)


def upload(staged_dir: Path, repo_id: str, private: bool, commit_message: str,
          subfolder: str = None) -> str:
    """Thin CLI wrapper over :func:`tackai.fusion.context.upload_context_tables`."""
    return upload_context_tables(staged_dir, repo_id, private, commit_message,
                                 subfolder=subfolder)


def cmd_upload(args: argparse.Namespace) -> None:
    staged_dir = Path(args.staged)
    if not staged_dir.exists():
        print(f"No staging directory found at {staged_dir}. Run the 'stage' phase first.")
        raise SystemExit(1)
    upload(staged_dir, args.repo_id, args.private, args.commit_message, subfolder=args.subfolder)


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
    upload_p.add_argument("--subfolder", default=None,
                          help="Upload into this subdirectory of the repo, so several "
                               "published snapshots can share one repo.")
    upload_p.set_defaults(func=cmd_upload)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
