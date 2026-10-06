# Fusion Context Embeddings on the Hub Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish the fusion pipeline's PCA-reduced context embedding tables to a new Hugging
Face dataset repo (`ailab-bio/TACK-fusion-context`) via a staging/upload script, and add
`FusionData.from_pretrained` so any machine can install them into `TACKAI_CACHE` and get a
working, encoder-only `FusionData`.

**Architecture:** A standalone `scripts/publish_fusion_context.py` stages the ten files named
in `tackai.fusion.context.CONTEXT_FILES` plus a generated `manifest.json` (content hashes,
per-block dimensions, source models), then uploads that staged folder as one commit. On the
consuming side, `FusionData.from_pretrained` resolves a local directory or downloads a Hub
snapshot, copies the manifest's files into the cache (verifying content hashes, never
silently overwriting a locally-refitted table), and returns `FusionData(table=None, ...)`.

**Tech Stack:** `huggingface_hub` (already a dependency, `>=1.13.0`), `numpy`, stdlib
`hashlib`/`json`/`shutil`/`argparse`. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-06-fusion-context-hub-design.md`

## Global Constraints

- The Hub repo is new and separate: `ailab-bio/TACK-fusion-context`, `repo_type="dataset"` —
  never reuse or write into `ailab-bio/TACK-cache`.
- The repo layout is flat; every published filename is exactly the string
  `tackai.fusion.context.CONTEXT_FILES[block][role]` already uses — no renaming, no nested
  directories.
- Staging publishes all ten `CONTEXT_FILES` entries or none: a partial set is refused with
  every missing file named in the output.
- Installing a file whose content hash differs from the manifest's record raises, unless the
  caller explicitly passes `force_download=True`.
- `ContextEncoder.table()` never downloads anything itself; all network access is initiated
  from `FusionData.from_pretrained` or the publish script, never implicitly during a fit.
- Tests run via `OMP_NUM_THREADS=1 pytest` (set by `test/conftest.py`); nothing in this plan
  needs `requires_cache` — everything is exercised against the synthetic `fake_cache` fixture.

## Review Focus

- `huggingface_hub` not installed, with a repo id (not a local directory) given — should
  raise with guidance (install the package, or pass a local directory), not a bare
  `ModuleNotFoundError`. (A download that fails for another reason — repo not found, offline —
  is left to propagate `huggingface_hub`'s own exception unchanged: this matches
  `EnsemblePredictor.from_pretrained` and `FusionEnsemble.from_pretrained`, which only special-case
  the import itself.) → Task 4, Part B.
- A `repo_id` that resolves to an existing local directory with no `manifest.json` (a
  mistyped path) — should raise `FileNotFoundError` naming that directory, not a `KeyError`
  from reading a missing key. → Task 4.
- `cache_dir` that does not exist yet (a fresh machine's first run) — should be created
  automatically, not raise `OSError`. → Task 4.
- A file already in the cache under the published name but with different bytes (the
  classic "locally refitted PCA" case) — should raise naming the file, not silently
  overwrite it or silently mix PCA fits. → Task 4.
- `descriptors=None` passed through `from_pretrained` (fingerprint-only featurizer) — should
  be honored (`dims["descriptors"] == 0`), not silently reset to the default 217. → Task 4.

---

### Task 1: Fix `fake_cache` — it is missing the `combined` PCA side file

**Files:**
- Modify: `test/conftest.py` (the `fake_cache` fixture)
- Create: `test/test_fusion_hub.py` (new test module; this task writes its first test)

**Interfaces:**
- Consumes: `fusion_fixtures.COMBINED_FILE`, `_write_pca_model` (both already defined in
  `test/conftest.py` / `test/fusion_fixtures.py`).
- Produces: a `fake_cache` fixture that writes all ten files `CONTEXT_FILES` names, so every
  later task can stage a complete set from it.

`fake_cache` currently calls `_write_pca_model` for `cell`, `poi`, `e3` and `assay`, but never
for `combined` — nothing has read that file before. Staging (Task 3) refuses a partial set, so
this gap must close first.

- [ ] **Step 1: Write the failing test**

```python
"""Publishing the fusion context embeddings to the Hub, and installing them back into
TACKAI_CACHE via FusionData.from_pretrained."""
from tackai.fusion.context import CONTEXT_FILES


def test_fake_cache_has_every_context_files_entry(fake_cache):
    for block, files in CONTEXT_FILES.items():
        for role, filename in files.items():
            assert (fake_cache / filename).exists(), f"missing {block}/{role}: {filename}"
```

Save this as the start of `test/test_fusion_hub.py`.

- [ ] **Step 2: Run it and confirm it fails**

Run: `OMP_NUM_THREADS=1 pytest test/test_fusion_hub.py -v`
Expected: FAIL — the `combined` `pca_model` file
(`protein_embeddings_esm_model=facebook-esm2_t30_150M_UR50D_layer=18_pooling=lse_window=1022_block=combined_pca52_model.npz`)
does not exist in `fake_cache`.

- [ ] **Step 3: Add the missing fixture line**

In `test/conftest.py`, in the `fake_cache` fixture, after the existing
`_write_pca_model(cache / ASSAY_PCA_FILE, 768, 8, 9)` line, add:

```python
    _write_pca_model(cache / COMBINED_FILE.replace(".npz", "_model.npz"), 640, 52, 10)
```

This requires `COMBINED_FILE` to be imported from `fusion_fixtures` in `conftest.py` — check
the existing import line and add it if it is not already there:

```python
from fusion_fixtures import (ASSAY_FILE, ASSAY_PCA_FILE, ASSAYS, CELL_FILE, CELLS,
                             COMBINED_FILE, E3_FILE, NOT_FOUND, POI_FILE, SEQS, SMILES)
```

(`COMBINED_FILE` is already in this import in the current file — confirm it's there; if it
is, only the new `_write_pca_model` call is needed.)

- [ ] **Step 4: Run it and confirm it passes**

Run: `OMP_NUM_THREADS=1 pytest test/test_fusion_hub.py -v`
Expected: PASS

- [ ] **Step 5: Run the full suite to confirm no regressions**

Run: `OMP_NUM_THREADS=1 pytest -q`
Expected: same pass count as before this change, plus the one new test. Nothing reads the
`combined` PCA side file today, so no existing test's behavior should change.

- [ ] **Step 6: Commit**

```bash
git add test/conftest.py test/test_fusion_hub.py
git commit -m "test(fusion): fake_cache now writes the combined PCA side file

CONTEXT_FILES names ten files; fake_cache wrote nine. Nothing read the
tenth before, but staging (next) refuses a partial set."
```

---

### Task 2: `DEFAULT_CONTEXT_REPO` constant

**Files:**
- Modify: `tackai/fusion/context.py`
- Modify: `tackai/fusion/__init__.py`
- Modify: `test/test_fusion_hub.py`

**Interfaces:**
- Produces: `tackai.fusion.context.DEFAULT_CONTEXT_REPO: str`, re-exported as
  `tackai.fusion.DEFAULT_CONTEXT_REPO`. Later tasks (3, 4) use this as the default repo id.

- [ ] **Step 1: Write the failing test**

Append to `test/test_fusion_hub.py`:

```python
def test_default_context_repo_is_exported():
    from tackai.fusion import DEFAULT_CONTEXT_REPO
    assert DEFAULT_CONTEXT_REPO == "ailab-bio/TACK-fusion-context"
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `OMP_NUM_THREADS=1 pytest test/test_fusion_hub.py::test_default_context_repo_is_exported -v`
Expected: FAIL with `ImportError: cannot import name 'DEFAULT_CONTEXT_REPO'`

- [ ] **Step 3: Add the constant**

In `tackai/fusion/context.py`, right after the `CONTEXT_FILES` dict definition (before
`CONTEXT_BLOCKS = (...)`), add:

```python
#: Hugging Face Hub dataset repo published by scripts/publish_fusion_context.py and read by
#: FusionData.from_pretrained(). A dataset repo, not a model repo: these tables are an input
#: to every fusion model, never the output of one.
DEFAULT_CONTEXT_REPO = "ailab-bio/TACK-fusion-context"
```

- [ ] **Step 4: Export it**

In `tackai/fusion/__init__.py`, change:

```python
from tackai.fusion.context import CONTEXT_FILES, ContextEncoder, normalize_assay
```

to:

```python
from tackai.fusion.context import (CONTEXT_FILES, DEFAULT_CONTEXT_REPO, ContextEncoder,
                                   normalize_assay)
```

and add `"DEFAULT_CONTEXT_REPO"` to `__all__`.

- [ ] **Step 5: Run it and confirm it passes**

Run: `OMP_NUM_THREADS=1 pytest test/test_fusion_hub.py -v`
Expected: PASS (both tests)

- [ ] **Step 6: Commit**

```bash
git add tackai/fusion/context.py tackai/fusion/__init__.py test/test_fusion_hub.py
git commit -m "feat(fusion): add DEFAULT_CONTEXT_REPO"
```

---

### Task 3: `scripts/publish_fusion_context.py` — stage and upload

**Files:**
- Create: `scripts/publish_fusion_context.py`
- Modify: `test/test_fusion_hub.py`

**Interfaces:**
- Consumes: `tackai.fusion.context.CONTEXT_FILES`, `CELL_MODEL`, `ASSAY_MODEL`,
  `POI_ESM_MODEL`, `DEFAULT_CONTEXT_REPO` (Task 2); `tackai.data.utils.get_cache_dir`;
  `tackai.__version__`.
- Produces (for Task 4's tests, which stage a `fake_cache` to get a local source directory to
  install from): `stage(cache_dir: Path, out_dir: Path) -> dict` — writes `out_dir` with the
  ten files plus `manifest.json`, and returns the manifest dict. Manifest shape:
  `{"format", "tackai_version", "block_dims": {block: int}, "combined_dim": int, "models":
  {...}, "files": {filename: {"block", "role", "sha256", "bytes", ...}}}`. `files` entries for
  `role == "table"` also carry `"dim"` and `"n_keys"`; entries for `role == "pca_model"` carry
  `"full_dim"` and `"reduced_dim"` instead (a PCA side file stores `mean_`/`components_`, not
  per-key vectors, so it has no `n_keys` and its own notion of "dim" is the pair
  full→reduced).

A PCA side-file npz (written by `_write_pca_model` in `test/conftest.py`, and by the real
`notebooks/*.ipynb`) holds `mean_`, `components_` and `explained_variance_ratio_` — not one
vector per key. Reading its first array's `.shape[0]` (as you would for a table npz) would
read `mean_`'s length, i.e. the *full* pre-PCA dimension, and report the file's own array
count (3) as `n_keys`. Both numbers would be meaningless. The two npz shapes need two
different readers.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_fusion_hub.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import publish_fusion_context as pfc  # noqa: E402

from fusion_fixtures import (ASSAY_FILE, ASSAY_PCA_FILE, CELL_FILE, COMBINED_FILE, E3_FILE,
                             POI_FILE)


def _all_context_filenames():
    from tackai.fusion.context import CONTEXT_FILES
    return [filename for files in CONTEXT_FILES.values() for filename in files.values()]


def test_stage_copies_all_ten_files_and_writes_a_manifest(fake_cache, tmp_path):
    out = tmp_path / "staged"
    manifest = pfc.stage(fake_cache, out)
    for name in _all_context_filenames():
        assert (out / name).exists()
    assert (out / "manifest.json").exists()
    assert manifest["block_dims"] == {"e3": 7, "cell": 47, "poi": 51, "assay": 8,
                                      "assay_time": 1}
    assert manifest["combined_dim"] == 52


def test_stage_hashes_match_the_staged_bytes(fake_cache, tmp_path):
    out = tmp_path / "staged"
    manifest = pfc.stage(fake_cache, out)
    for name, entry in manifest["files"].items():
        assert pfc.sha256_of(out / name) == entry["sha256"]


def test_stage_pca_model_entries_record_the_reduced_dimension(fake_cache, tmp_path):
    from tackai.fusion.context import CONTEXT_FILES
    manifest = pfc.stage(fake_cache, tmp_path / "staged")
    for block, files in CONTEXT_FILES.items():
        table_dim = manifest["files"][files["table"]]["dim"]
        reduced_dim = manifest["files"][files["pca_model"]]["reduced_dim"]
        assert table_dim == reduced_dim, block


def test_stage_refuses_a_partial_set(fake_cache, tmp_path, capsys):
    missing_name = POI_FILE.replace(".npz", "_model.npz")
    (fake_cache / missing_name).unlink()
    with pytest.raises(SystemExit):
        pfc.stage(fake_cache, tmp_path / "staged")
    assert missing_name in capsys.readouterr().out


def test_stage_refuses_an_existing_output_directory(fake_cache, tmp_path):
    out = tmp_path / "staged"
    out.mkdir()
    with pytest.raises(SystemExit):
        pfc.stage(fake_cache, out)
```

This needs `import pytest` at the top of `test/test_fusion_hub.py` — add it alongside the
existing imports (`from tackai.fusion.context import CONTEXT_FILES`).

- [ ] **Step 2: Run them and confirm they fail**

Run: `OMP_NUM_THREADS=1 pytest test/test_fusion_hub.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'publish_fusion_context'`

- [ ] **Step 3: Write `scripts/publish_fusion_context.py` (stage half)**

```python
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    stage_p = sub.add_parser("stage")
    stage_p.add_argument("--cache-dir", default=None,
                         help="Defaults to TACKAI_CACHE / get_cache_dir().")
    stage_p.add_argument("--out", default=str(STAGING_ROOT))
    stage_p.set_defaults(func=cmd_stage)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `OMP_NUM_THREADS=1 pytest test/test_fusion_hub.py -v`
Expected: PASS for every `test_stage_*` test and the two from earlier tasks.

- [ ] **Step 5: Write the failing test for `upload()`**

Append to `test/test_fusion_hub.py`:

```python
def test_upload_calls_create_repo_and_upload_folder(fake_cache, tmp_path, monkeypatch):
    out = tmp_path / "staged"
    pfc.stage(fake_cache, out)
    calls = {}

    class FakeApi:
        def whoami(self):
            return {"name": "test-user"}

        def create_repo(self, repo_id, repo_type, exist_ok, private):
            calls["create_repo"] = (repo_id, repo_type, exist_ok, private)

        def upload_folder(self, repo_id, repo_type, folder_path, commit_message):
            calls["upload_folder"] = (repo_id, repo_type, folder_path, commit_message)
            return type("Commit", (), {"oid": "abc123"})()

    monkeypatch.setattr("huggingface_hub.HfApi", lambda: FakeApi())
    sha = pfc.upload(out, "ailab-bio/TACK-fusion-context", private=False,
                     commit_message="test commit")
    assert calls["create_repo"] == ("ailab-bio/TACK-fusion-context", "dataset", True, False)
    assert calls["upload_folder"] == ("ailab-bio/TACK-fusion-context", "dataset", str(out),
                                      "test commit")
    assert sha == "abc123"
```

- [ ] **Step 6: Run it and confirm it fails**

Run: `OMP_NUM_THREADS=1 pytest test/test_fusion_hub.py::test_upload_calls_create_repo_and_upload_folder -v`
Expected: FAIL with `AttributeError: module 'publish_fusion_context' has no attribute 'upload'`

- [ ] **Step 7: Add `upload()` and the `upload` CLI subcommand**

In `scripts/publish_fusion_context.py`, add after `stage()`:

```python
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
```

Then, in `main()`, after the `stage_p` block and before `args = parser.parse_args()`, add:

```python
    upload_p = sub.add_parser("upload")
    upload_p.add_argument("--repo-id", default=DEFAULT_CONTEXT_REPO)
    upload_p.add_argument("--staged", default=str(STAGING_ROOT))
    upload_p.add_argument("--private", action="store_true")
    upload_p.add_argument("--commit-message", default="Update fusion context embeddings")
    upload_p.set_defaults(func=cmd_upload)
```

- [ ] **Step 8: Run the tests and confirm they pass**

Run: `OMP_NUM_THREADS=1 pytest test/test_fusion_hub.py -v`
Expected: PASS, all tests in the file so far.

- [ ] **Step 9: Commit**

```bash
git add scripts/publish_fusion_context.py test/test_fusion_hub.py
git commit -m "feat(fusion): publish_fusion_context.py stage/upload script"
```

---

### Task 4: `FusionData.from_pretrained`

**Files:**
- Modify: `tackai/fusion/data.py`
- Modify: `test/test_fusion_hub.py`

**Interfaces:**
- Consumes: `pfc.stage` (Task 3, used by tests to produce a local source directory);
  `ContextEncoder`, `DEFAULT_CONTEXT_REPO` (already importable from `tackai.fusion.context`);
  `get_cache_dir` (already imported in `data.py`).
- Produces: `FusionData.from_pretrained(repo_id=DEFAULT_CONTEXT_REPO, *, revision=None,
  token=None, cache_dir=None, force_download=False, protein_space="per_block",
  featurizer=None, descriptors=DESCRIPTOR_NAMES) -> FusionData`, returning an instance with
  `table is None`.

#### Part A: local-directory happy path

- [ ] **Step 1: Write the failing tests**

Append to `test/test_fusion_hub.py`:

```python
import json

import numpy as np

from tackai.fusion.data import FusionData


def test_from_pretrained_installs_into_an_empty_cache(fake_cache, tmp_path):
    staged = tmp_path / "staged"
    pfc.stage(fake_cache, staged)
    target = tmp_path / "fresh_cache"          # does not exist yet
    data = FusionData.from_pretrained(staged, cache_dir=target)
    assert data.table is None
    assert data.dims == {"fingerprint": 1024, "descriptors": 217, "e3": 7, "cell": 47,
                         "poi": 51, "assay": 8, "assay_time": 1}
    for name in _all_context_filenames():
        assert (target / name).exists()
    with pytest.raises(ValueError, match="no table"):
        _ = data.X


def test_from_pretrained_can_encode_a_context(fake_cache, tmp_path, tiny_records):
    staged = tmp_path / "staged"
    pfc.stage(fake_cache, staged)
    data = FusionData.from_pretrained(staged, cache_dir=tmp_path / "fresh_cache")
    row = tiny_records[0]
    ctx = data.encode_context({"poi_seq": row["poi_seq"], "e3_seq": row["e3_seq"],
                               "cell_id": row["cell_id"], "assay": row["assay"],
                               "assay_time": row["assay_time"]})
    assert ctx.shape == (1, len(data.context_columns))


def test_from_pretrained_second_call_is_a_no_op(fake_cache, tmp_path):
    staged = tmp_path / "staged"
    pfc.stage(fake_cache, staged)
    target = tmp_path / "fresh_cache"
    FusionData.from_pretrained(staged, cache_dir=target)
    before = {p.name: p.stat().st_mtime_ns for p in target.iterdir()}
    FusionData.from_pretrained(staged, cache_dir=target)
    after = {p.name: p.stat().st_mtime_ns for p in target.iterdir()}
    assert before == after


def test_from_pretrained_supports_the_combined_protein_space(fake_cache, tmp_path):
    """The combined table is published even though the default protein_space never reads
    it; protein_space="combined" must still work after a from-Hub install (spec test #9)."""
    staged = tmp_path / "staged"
    pfc.stage(fake_cache, staged)
    data = FusionData.from_pretrained(staged, cache_dir=tmp_path / "fresh_cache",
                                      protein_space="combined")
    assert data.dims["poi"] == 52 and data.dims["e3"] == 52
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `OMP_NUM_THREADS=1 pytest test/test_fusion_hub.py -k from_pretrained -v`
Expected: FAIL with `AttributeError: type object 'FusionData' has no attribute 'from_pretrained'`

- [ ] **Step 3: Implement the local-directory happy path**

In `tackai/fusion/data.py`:

1. Add `shutil` to the stdlib imports:

```python
import hashlib
import json
import shutil
from pathlib import Path
```

2. Add `DEFAULT_CONTEXT_REPO` to the `tackai.fusion.context` import:

```python
from tackai.fusion.context import (CONTEXT_BLOCKS, DEFAULT_CONTEXT_REPO, ContextEncoder,
                                   assay_time_or_default)
```

3. Add a module-level helper, near `_first_valid` (before `build_table`):

```python
def _file_sha256(path: Path) -> str:
    """Streamed sha256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
```

4. Add the classmethod to `FusionData`, directly after `from_csv` (before `_drop_unencodable`):

```python
    @classmethod
    def from_pretrained(cls, repo_id: Union[str, Path] = DEFAULT_CONTEXT_REPO, *,
                        revision: Optional[str] = None,
                        token: Optional[str] = None,
                        cache_dir: Optional[Union[str, Path]] = None,
                        force_download: bool = False,
                        protein_space: str = "per_block",
                        featurizer: Optional[MolEncoder] = None,
                        descriptors: Optional[Sequence[str]] = DESCRIPTOR_NAMES) -> "FusionData":
        """Build an encoder-only FusionData from the published context embedding tables.

        Downloads (or reuses a local directory of) the context tables named in
        :data:`~tackai.fusion.context.CONTEXT_FILES`, installs them into the local cache, and
        returns a :class:`FusionData` with no table — ready for :meth:`encode`,
        :meth:`encode_context` and :meth:`assemble`, but not for :attr:`X`, :attr:`groups` or
        :meth:`target` (use :meth:`from_csv` for a training table).

        Args:
            repo_id: A Hugging Face Hub dataset repo id, or a local directory written by
                ``scripts/publish_fusion_context.py``'s ``stage`` phase (default:
                :data:`~tackai.fusion.context.DEFAULT_CONTEXT_REPO`).
            revision: Hub revision, for a repo id.
            token: Hub token, for a private repo.
            cache_dir: Directory to install the tables into (default: ``TACKAI_CACHE`` via
                :func:`get_cache_dir`). Created if it does not exist.
            force_download: Overwrite a cached file whose content differs from the published
                one, instead of raising. A mismatch almost always means the cache already
                holds tables from a different, locally refitted PCA.
            protein_space: Passed to :class:`ContextEncoder`.
            featurizer: Molecular featuriser (default: a fresh :class:`MolEncoder`).
            descriptors: RDKit descriptor names for the default featuriser, or ``None`` for
                fingerprints only.

        Returns:
            A :class:`FusionData` with ``table=None``.

        Raises:
            FileNotFoundError: If the resolved source holds no ``manifest.json``.
            ValueError: If a cached file's content differs from the manifest's record of it
                (and ``force_download`` is false), or if the installed tables' widths
                disagree with the manifest's ``block_dims``.
        """
        source = Path(repo_id)
        if not source.exists():
            try:
                from huggingface_hub import snapshot_download
            except ImportError as e:
                raise ImportError(
                    "huggingface_hub is required to download from the Hub; install it, or "
                    "pass a local directory to from_pretrained() instead."
                ) from e
            source = Path(snapshot_download(repo_id=str(repo_id), repo_type="dataset",
                                            revision=revision, token=token))

        manifest_path = source / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"no manifest.json found in {source}; this does not look like a context "
                "embedding repo published by scripts/publish_fusion_context.py"
            )
        manifest = json.loads(manifest_path.read_text())

        target = Path(cache_dir) if cache_dir is not None else Path(get_cache_dir())
        target.mkdir(parents=True, exist_ok=True)
        for filename, entry in manifest["files"].items():
            dest = target / filename
            if dest.exists():
                dest_hash = _file_sha256(dest)
                if dest_hash == entry["sha256"]:
                    continue
                if not force_download:
                    raise ValueError(
                        f"{filename} already exists in {target} with different content "
                        f"(cached sha256 {dest_hash[:12]}…, published sha256 "
                        f"{entry['sha256'][:12]}…). This usually means the cache holds "
                        "tables from a different, locally refitted PCA. Pass "
                        "force_download=True to overwrite, or point cache_dir elsewhere."
                    )
            shutil.copy2(source / filename, dest)

        encoder = ContextEncoder(cache_dir=target, protein_space=protein_space)
        data = cls(table=None, encoder=encoder, featurizer=featurizer, descriptors=descriptors)

        expected = manifest.get("block_dims", {})
        bad = {b: (expected[b], data.dims[b]) for b in expected
              if b in data.dims and expected[b] != data.dims[b]}
        if bad:
            detail = ", ".join(f"{b}: published with {e}, the cache now has {a}"
                               for b, (e, a) in sorted(bad.items()))
            raise ValueError(
                f"block layout mismatch after installing the published context tables "
                f"({detail}). The cache already held tables for one or more blocks from a "
                "different PCA fit; clear them from the cache or point cache_dir elsewhere."
            )
        return data
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `OMP_NUM_THREADS=1 pytest test/test_fusion_hub.py -k from_pretrained -v`
Expected: PASS for all three Part-A tests.

#### Part B: error paths

- [ ] **Step 5: Write the failing tests**

Append to `test/test_fusion_hub.py`:

```python
def test_from_pretrained_rejects_a_tampered_cached_file(fake_cache, tmp_path):
    staged = tmp_path / "staged"
    pfc.stage(fake_cache, staged)
    target = tmp_path / "fresh_cache"
    FusionData.from_pretrained(staged, cache_dir=target)
    rng = np.random.default_rng(0)
    np.savez(target / POI_FILE, **{s: rng.normal(size=51).astype(np.float32) for s in "xy"})
    with pytest.raises(ValueError, match=POI_FILE):
        FusionData.from_pretrained(staged, cache_dir=target)


def test_from_pretrained_force_download_overwrites(fake_cache, tmp_path):
    staged = tmp_path / "staged"
    pfc.stage(fake_cache, staged)
    target = tmp_path / "fresh_cache"
    FusionData.from_pretrained(staged, cache_dir=target)
    rng = np.random.default_rng(0)
    np.savez(target / POI_FILE, **{s: rng.normal(size=51).astype(np.float32) for s in "xy"})
    data = FusionData.from_pretrained(staged, cache_dir=target, force_download=True)
    assert data.dims["poi"] == 51


def test_from_pretrained_rejects_a_missing_manifest(tmp_path):
    empty_source = tmp_path / "empty_source"
    empty_source.mkdir()
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        FusionData.from_pretrained(empty_source, cache_dir=tmp_path / "cache")


def test_from_pretrained_rejects_a_block_width_mismatch(fake_cache, tmp_path):
    staged = tmp_path / "staged"
    pfc.stage(fake_cache, staged)
    manifest_path = staged / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["block_dims"]["cell"] = 48
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="cell"):
        FusionData.from_pretrained(staged, cache_dir=tmp_path / "fresh_cache")


def test_from_pretrained_descriptors_none_gives_fingerprint_only(fake_cache, tmp_path):
    staged = tmp_path / "staged"
    pfc.stage(fake_cache, staged)
    data = FusionData.from_pretrained(staged, cache_dir=tmp_path / "fresh_cache",
                                      descriptors=None)
    assert data.dims["descriptors"] == 0


def test_from_pretrained_gives_guidance_when_huggingface_hub_is_missing(tmp_path, monkeypatch):
    """A repo id (not a local path) with huggingface_hub unimportable must raise a clear
    ImportError, not a bare traceback. Setting the module to None in sys.modules is the
    standard trick to make `from huggingface_hub import ...` raise ImportError on demand."""
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(ImportError, match="huggingface_hub"):
        FusionData.from_pretrained("ailab-bio/TACK-fusion-context", cache_dir=tmp_path / "cache")
```

- [ ] **Step 6: Run them and confirm they fail**

Run: `OMP_NUM_THREADS=1 pytest test/test_fusion_hub.py -k from_pretrained -v`
Expected: `test_from_pretrained_gives_guidance_when_huggingface_hub_is_missing` FAILs if the
`except ImportError` branch in Step 3 is missing or its message doesn't mention
`huggingface_hub`; the other four (`_rejects_a_tampered_cached_file`,
`_force_download_overwrites`, `_rejects_a_missing_manifest`, `_rejects_a_block_width_mismatch`,
`_descriptors_none_gives_fingerprint_only`) should already be green, since Step 3's
implementation covers all of these branches already. Run the file and read the actual output;
this step exists to confirm each case individually, with its own assertion, before trusting
the Step-3 implementation.

- [ ] **Step 7: Fix any that fail**

If any test fails, the most likely cause is a message-matching mismatch (e.g. `pytest.raises
(..., match=POI_FILE)` failing because `POI_FILE` contains regex metacharacters like `.` — in
that case use `re.escape(POI_FILE)` in the test's `match=` argument instead). Adjust the test
or the implementation as needed and re-run until all pass.

- [ ] **Step 8: Run the full file and confirm all pass**

Run: `OMP_NUM_THREADS=1 pytest test/test_fusion_hub.py -v`
Expected: PASS, every test in the file.

#### Part C: repo-id (Hub) branch

- [ ] **Step 9: Write the failing test**

Append to `test/test_fusion_hub.py`:

```python
def test_from_pretrained_uses_snapshot_download_for_a_repo_id(fake_cache, tmp_path, monkeypatch):
    staged = tmp_path / "staged"
    pfc.stage(fake_cache, staged)
    calls = {}

    def fake_snapshot_download(repo_id, repo_type, revision, token):
        calls["args"] = (repo_id, repo_type, revision, token)
        return str(staged)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    data = FusionData.from_pretrained("ailab-bio/TACK-fusion-context",
                                      cache_dir=tmp_path / "fresh_cache")
    assert calls["args"] == ("ailab-bio/TACK-fusion-context", "dataset", None, None)
    assert data.dims["cell"] == 47
```

- [ ] **Step 10: Run it and confirm it passes**

Run: `OMP_NUM_THREADS=1 pytest test/test_fusion_hub.py::test_from_pretrained_uses_snapshot_download_for_a_repo_id -v`
Expected: PASS — Step 3's implementation already calls `snapshot_download` exactly this way
when `Path(repo_id)` does not exist as a local path. This test exists to pin that contract so
a future refactor cannot silently change the keyword arguments
(`repo_type="dataset"` in particular — the legacy `EnsemblePredictor` uses `repo_type="model"`
by default, and mixing them up would make every download 404).

- [ ] **Step 11: Run the full test suite**

Run: `OMP_NUM_THREADS=1 pytest -q`
Expected: every test passes, including the full `test_fusion_hub.py` module and everything
else in the suite (confirms Task 1's fixture change and the new imports in `data.py` broke
nothing elsewhere).

- [ ] **Step 12: Commit**

```bash
git add tackai/fusion/data.py test/test_fusion_hub.py
git commit -m "feat(fusion): FusionData.from_pretrained"
```

---

### Task 5: Documentation

**Files:**
- Modify: `.env.example`
- Modify: `tackai/fusion/context.py` (module docstring)
- Create: `docs/guide/fusion-context.md`
- Modify: `mkdocs.yml`

No new automated tests — these are documentation changes. Each step is verified by reading
the rendered result.

- [ ] **Step 1: `.env.example`**

In `.env.example`, after the existing paragraph about `HF_TOKEN` (which already mentions
`ailab-bio/TACK-ensembles` / `ailab-bio/TACK-cache`), extend that sentence to also name the
new repo:

```
# Optional: Set up the Hugging Face token. Unused for the public TACK dataset
# and the public ailab-bio/TACK-ensembles / ailab-bio/TACK-cache /
# ailab-bio/TACK-fusion-context repos.
# HF_TOKEN=/your/HF_TOKEN/value/here
```

(This is a one-word-list edit to the existing comment line — add
`/ ailab-bio/TACK-fusion-context` to it.)

- [ ] **Step 2: `tackai/fusion/context.py` module docstring**

The module docstring currently reads (first paragraph):

```python
"""Biological context blocks, read from the cached PCA-reduced embedding tables.

The tables are produced by ``notebooks/context_embeddings.ipynb`` (cell lines),
``notebooks/protein_pooling_comparison.ipynb`` (POI and E3 ligase) and
``notebooks/assay_embeddings.ipynb`` (assay types), and live in ``TACKAI_CACHE``. They are
already reduced, which is why no PCA runs anywhere in this pipeline.

What happens to a key the tables do not contain depends on whether the vector can be
reconstructed at all:
```

Insert a new paragraph between the first and the "What happens..." paragraph:

```python
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

What happens to a key the tables do not contain depends on whether the vector can be
reconstructed at all:
```

- [ ] **Step 3: New guide page**

Create `docs/guide/fusion-context.md`:

```markdown
# Fusion Context Cache (Experimental)

The fusion surrogates in `tackai.fusion` (notebook-driven, under active development — see
`notebooks/fusion_comparison.ipynb`) read their biological context (cell line, POI, E3
ligase, assay) from a handful of small, PCA-reduced embedding tables cached in
`TACKAI_CACHE`. `FusionData.from_pretrained` downloads them for you instead of requiring the
notebooks that produce them:

```python
from tackai.fusion import FusionData

data = FusionData.from_pretrained()   # ailab-bio/TACK-fusion-context by default
ctx = data.encode_context({
    "poi_seq": "MAGEG...", "e3_seq": "MEPVR...",
    "cell_id": "CVCL_0031", "assay": "western blot", "assay_time": 24.0,
})
```

This installs the tables into `TACKAI_CACHE` (honoring `HF_HOME` for the download itself) and
returns a `FusionData` with no training table — it is ready for `encode`, `encode_context` and
`assemble`, but not for `X`, `groups` or `target()` (those need `FusionData.from_csv` and the
curated CSVs).

Passing a local directory instead of a repo id (one written by
`scripts/publish_fusion_context.py`'s `stage` phase) skips the network entirely:

```python
data = FusionData.from_pretrained("/path/to/staged/fusion_context")
```

If a file already in your cache differs from the one being installed — most often because you
locally refitted a PCA — `from_pretrained` raises rather than silently mixing the two. Pass
`force_download=True` to replace it deliberately.
```

- [ ] **Step 4: Add it to the nav**

In `mkdocs.yml`, in the `User Guide` section, add a line after `Command-Line Interface`:

```yaml
  - User Guide:
      - Loading a Pretrained Ensemble: guide/loading-models.md
      - Making Predictions: guide/making-predictions.md
      - Fast Screening (Reusing Context): guide/screening.md
      - Loading the TACK Dataset: guide/loading-dataset.md
      - Command-Line Interface: guide/cli.md
      - Fusion Context Cache: guide/fusion-context.md
```

- [ ] **Step 5: Sanity-check the docstring change**

Run: `OMP_NUM_THREADS=1 python -c "import tackai.fusion.context; print(tackai.fusion.context.__doc__[:400])"`
Expected: prints the module docstring, including the new paragraph, with no syntax errors.

- [ ] **Step 6: Commit**

```bash
git add .env.example tackai/fusion/context.py docs/guide/fusion-context.md mkdocs.yml
git commit -m "docs(fusion): document FusionData.from_pretrained and the new Hub repo"
```

---

## Final verification (all tasks complete)

- [ ] Run `OMP_NUM_THREADS=1 pytest -q` — full suite green.
- [ ] Run `OMP_NUM_THREADS=1 pytest test/test_fusion_hub.py -v` — every test in the new module
  passes individually, with names visible.
- [ ] `git log --oneline -6` shows the five commits from Tasks 1–5 in order.
- [ ] `python scripts/publish_fusion_context.py stage --help` and
  `python scripts/publish_fusion_context.py upload --help` both print usage without error
  (confirms the CLI wiring, not just the tested functions, is intact).
