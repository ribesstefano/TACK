"""
Stage and publish TACK's pretrained ensembles + shared cache assets to the
Hugging Face Hub (ailab-bio/TACK-ensembles, ailab-bio/TACK-cache).

Two phases, run separately so the staged folders can be inspected before
anything is uploaded:

    # 1. Stage locally (pure file copies, no network writes)
    python scripts/publish_hf.py stage --all

    # 2. Review hf_staging/ensembles/ and hf_staging/cache/, then upload
    python scripts/publish_hf.py upload --all

Staging reuses collect_ensemble_checkpoints.py's file-discovery logic so only
the checkpoints actually referenced by each ensemble_weights_*.json are
copied (not the full outputs/ training tree). The cache directory is
reconciled from the union of cache/ and cache/tack/ (see README's "Pre-trained
Models & Cache Files" section for why these two currently disagree) plus the
repo-root boltz_embeddings_*.npz files.

Requires `huggingface_hub` to be installed and an authenticated session
(`hf auth login`, or an HF_TOKEN in the environment) with write access to the
ailab-bio org for the `upload` phase.

Author: Stefano Ribes
"""
import argparse
import shutil
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from collect_ensemble_checkpoints import find_model_file, find_datamodule_files  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGING_ROOT = REPO_ROOT / "hf_staging"

ENSEMBLE_REPO_ID = "ailab-bio/TACK-ensembles"
CACHE_REPO_ID = "ailab-bio/TACK-cache"

# (task, strategy) -> outputs/tack2_<task>_scaffold/ensemble_<strategy>/
ENSEMBLES = [
    ("dmax", "caruana"),
    ("dmax", "best_arch"),
    ("dc50", "caruana"),
    ("dc50", "best_arch"),
    ("bin", "caruana"),
    ("bin", "best_arch"),
]

# Cache files to reconcile from cache/ vs cache/tack/ (prefer cache/tack/, the
# directory actually pointed at by TACKAI_CACHE in .env, falling back to the
# stale top-level cache/ for anything missing there).
CACHE_FILENAMES = [
    "cell2cell_id.json",
    "cell2data.json",
    "cell2description.json",
    "cellosaurus.txt",
    "cell_embeddings_model=sentence-transformer_pooling=sum.npz",
    "morgan_fp_radius16_size512.npz",
]

BOLTZ_FILENAMES = [
    "boltz_embeddings_s_mean.npz",
    # "boltz_embeddings_s_mean_geneimputed.npz",
    # "boltz_embeddings_s_sum.npz",
]


def stage_ensemble(task: str, strategy: str) -> None:
    src_dir = REPO_ROOT / "outputs" / f"tack2_{task}_scaffold" / f"ensemble_{strategy}"
    weights_candidates = list(src_dir.glob("ensemble_weights_*.json"))
    if not weights_candidates:
        print(f"  SKIP {task}/{strategy}: no ensemble_weights_*.json found in {src_dir}")
        return
    if len(weights_candidates) > 1:
        print(f"  WARNING {task}/{strategy}: {len(weights_candidates)} weights files found, using {weights_candidates[0].name}")
    weights_file = weights_candidates[0]

    import json
    weights = json.loads(weights_file.read_text()).get("weights", {})
    if not weights:
        print(f"  SKIP {task}/{strategy}: empty 'weights' in {weights_file}")
        return

    checkpoints_dir = src_dir / "checkpoints"
    out_dir = STAGING_ROOT / "ensembles" / f"{task}_{strategy}"
    ckpt_out = out_dir / "checkpoints"
    ckpt_out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weights_file, out_dir / weights_file.name)

    n_ok, missing = 0, []
    for model_name in weights:
        model_file = find_model_file(model_name, checkpoints_dir)
        hparams_file, state_file = find_datamodule_files(model_name, checkpoints_dir)
        if model_file is None or (hparams_file is None and state_file is None):
            missing.append(model_name)
            continue
        shutil.copy2(model_file, ckpt_out / model_file.name)
        if hparams_file:
            shutil.copy2(hparams_file, ckpt_out / hparams_file.name)
        if state_file:
            shutil.copy2(state_file, ckpt_out / state_file.name)
        n_ok += 1

    print(f"  {task}/{strategy}: staged {n_ok}/{len(weights)} models -> {out_dir}")
    if missing:
        print(f"    missing: {missing}")


def stage_cache() -> None:
    out_dir = STAGING_ROOT / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_tack = REPO_ROOT / "cache" / "tack"
    cache_top = REPO_ROOT / "cache"
    for name in CACHE_FILENAMES:
        src = cache_tack / name if (cache_tack / name).exists() else cache_top / name
        if not src.exists():
            print(f"  WARNING: cache file not found in cache/tack/ or cache/: {name}")
            continue
        shutil.copy2(src, out_dir / name)
        print(f"  cache: {name} <- {src.relative_to(REPO_ROOT)}")

    for name in BOLTZ_FILENAMES:
        src = REPO_ROOT / name
        if not src.exists():
            print(f"  WARNING: boltz embeddings file not found at repo root: {name}")
            continue
        shutil.copy2(src, out_dir / name)
        print(f"  cache: {name} <- {src.relative_to(REPO_ROOT)}")


def cmd_stage(args: argparse.Namespace) -> None:
    if STAGING_ROOT.exists():
        print(f"Staging directory already exists: {STAGING_ROOT}")
        print("Remove it (or move it aside) before re-staging, to avoid mixing stale and fresh files.")
        sys.exit(1)

    if args.all or args.cache:
        print("Staging shared cache assets ...")
        stage_cache()
    if args.all or args.ensembles:
        print("Staging ensembles ...")
        for task, strategy in ENSEMBLES:
            stage_ensemble(task, strategy)

    print(f"\nDone. Inspect {STAGING_ROOT} before running the 'upload' phase.")


def cmd_upload(args: argparse.Namespace) -> None:
    if not STAGING_ROOT.exists():
        print(f"No staging directory found at {STAGING_ROOT}. Run the 'stage' phase first.")
        sys.exit(1)

    from huggingface_hub import HfApi
    api = HfApi()
    print(f"Authenticated as: {api.whoami()['name']}")

    if args.all or args.cache:
        cache_dir = STAGING_ROOT / "cache"
        print(f"Creating (if needed) dataset repo {CACHE_REPO_ID} ...")
        api.create_repo(CACHE_REPO_ID, repo_type="dataset", exist_ok=True, private=args.private)
        print(f"Uploading {cache_dir} -> {CACHE_REPO_ID} ...")
        api.upload_large_folder(repo_id=CACHE_REPO_ID, repo_type="dataset", folder_path=str(cache_dir))

    if args.all or args.ensembles:
        ensembles_dir = STAGING_ROOT / "ensembles"
        print(f"Creating (if needed) model repo {ENSEMBLE_REPO_ID} ...")
        api.create_repo(ENSEMBLE_REPO_ID, repo_type="model", exist_ok=True, private=args.private)
        print(f"Uploading {ensembles_dir} -> {ENSEMBLE_REPO_ID} ...")
        api.upload_large_folder(repo_id=ENSEMBLE_REPO_ID, repo_type="model", folder_path=str(ensembles_dir))

    print("\nDone.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    for name, fn in [("stage", cmd_stage), ("upload", cmd_upload)]:
        p = sub.add_parser(name)
        p.add_argument("--all", action="store_true", help="Both cache and ensembles.")
        p.add_argument("--cache", action="store_true", help="Only the shared cache assets.")
        p.add_argument("--ensembles", action="store_true", help="Only the ensemble checkpoints.")
        if name == "upload":
            p.add_argument("--private", action="store_true", help="Create the HF repo(s) as private.")
        p.set_defaults(func=fn)

    args = parser.parse_args()
    if not (args.all or args.cache or args.ensembles):
        parser.error("pass --all, --cache, and/or --ensembles")
    args.func(args)


if __name__ == "__main__":
    main()
