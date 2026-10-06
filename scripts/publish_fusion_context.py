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

Author: Stefano Ribes
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tackai
from tackai.data.utils import get_cache_dir
from tackai.fusion.context import (ASSAY_MODEL, CELL_MODEL, CONTEXT_FILES, DEFAULT_CONTEXT_REPO,
                                   POI_ESM_MODEL)

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGING_ROOT = REPO_ROOT / "hf_staging" / "fusion_context"
MANIFEST_FORMAT = "tack-fusion-context/v1"


def sha256_of(path: Path) -> str:
    """Streamed sha256 of a file, so staging never loads a whole npz into memory to hash it."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def npz_table_dims(path: Path) -> Tuple[int, int]:
    """(vector width, number of keys) of a context *table* npz, read from its arrays.

    A table npz holds one vector per lookup key (sequence, cell accession, assay type), all
    the same width.
    """
    with np.load(path) as data:
        keys = list(data.keys())
        dim = int(data[keys[0]].shape[0])
    return dim, len(keys)


def npz_pca_dims(path: Path) -> Tuple[int, int]:
    """(full pre-PCA dimension, reduced dimension) of a PCA side-file npz.

    A PCA side file holds `mean_` (length = full dimension) and `components_` (shape
    reduced x full) — not per-key vectors, so it has neither a key count nor a single "dim"
    the way a table npz does.
    """
    with np.load(path) as data:
        reduced_dim, full_dim = data["components_"].shape
    return int(full_dim), int(reduced_dim)


def stage(cache_dir: Path, out_dir: Path) -> Dict:
    """Copy every CONTEXT_FILES entry from `cache_dir` into `out_dir`, with a manifest.

    Args:
        cache_dir: Directory holding the cached context tables (TACKAI_CACHE).
        out_dir: Staging directory to write into; refused if it already exists.

    Returns:
        The manifest dict also written to `out_dir / "manifest.json"`.

    Raises:
        SystemExit: If `out_dir` already exists, or any CONTEXT_FILES entry is missing from
            `cache_dir` (every missing file is listed before exiting).
    """
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
            dim, n_keys = npz_table_dims(src)
            entry["dim"], entry["n_keys"] = dim, n_keys
            if block == "combined":
                manifest["combined_dim"] = dim
            else:
                manifest["block_dims"][block] = dim
        else:
            full_dim, reduced_dim = npz_pca_dims(src)
            entry["full_dim"], entry["reduced_dim"] = full_dim, reduced_dim
        manifest["files"][filename] = entry
        shutil.copy2(src, out_dir / filename)
        print(f"  staged {filename} ({block}/{role})")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\nStaged {len(wanted)} files + manifest.json -> {out_dir}")
    return manifest


def cmd_stage(args: argparse.Namespace) -> None:
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(get_cache_dir())
    stage(cache_dir, Path(args.out))


def upload(staged_dir: Path, repo_id: str, private: bool, commit_message: str) -> str:
    """Create (if needed) the dataset repo and upload every file in `staged_dir`.

    Args:
        staged_dir: Directory written by `stage()`.
        repo_id: Hugging Face Hub repo id.
        private: Create the repo as private if it does not exist yet.
        commit_message: Commit message for the upload.

    Returns:
        The commit sha `upload_folder` reports.
    """
    from huggingface_hub import HfApi
    api = HfApi()
    print(f"Authenticated as: {api.whoami()['name']}")
    print(f"Creating (if needed) dataset repo {repo_id} ...")
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=private)
    print(f"Uploading {staged_dir} -> {repo_id} ...")
    commit = api.upload_folder(repo_id=repo_id, repo_type="dataset",
                               folder_path=str(staged_dir), commit_message=commit_message)
    sha = getattr(commit, "oid", str(commit))
    print(f"Done: https://huggingface.co/datasets/{repo_id} (commit {sha})")
    return sha


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
    stage_p.set_defaults(func=cmd_stage)

    upload_p = sub.add_parser("upload")
    upload_p.add_argument("--repo-id", default=DEFAULT_CONTEXT_REPO)
    upload_p.add_argument("--staged", default=str(STAGING_ROOT))
    upload_p.add_argument("--private", action="store_true")
    upload_p.add_argument("--commit-message", default="Update fusion context embeddings")
    upload_p.set_defaults(func=cmd_upload)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
