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
than in ``scripts/embeddings/publish_fusion_context.py`` because both the maintainer publish
script and :meth:`~tackai.fusion.data.FusionData.push_to_hub` need the exact same manifest
format; the script is a thin CLI wrapper over these functions, not its own implementation.

A cache can hold several generations of a table, since each PCA width gets its own file name,
so the files alone do not say which one is current. Each generator in ``scripts/embeddings/``
therefore records what it wrote in its own part manifest (:func:`write_part_manifest`), and
the publish script aggregates those into the final ``manifest.json``
(:func:`stage_from_part_manifests`). Part manifests never replace ``manifest.json`` in the
cache: that file names the tables the encoder reads, and only an install changes it.

Both the ``per_block`` and ``combined`` protein tables are always staged together — a reader
may want either — so the files on disk cannot tell a space apart; the manifest's
``protein_space`` key is what lets :meth:`~tackai.fusion.data.FusionData.from_pretrained`
reproduce the publisher's own choice rather than guessing ``per_block``. Only
:meth:`~tackai.fusion.data.FusionData.push_to_hub` (one already-built encoder) can state this
unambiguously; ``stage_context_tables`` called directly over a whole cache leaves it ``None``.

What happens to a key the tables do not contain depends on whether the vector can be
reconstructed at all:

* **cell** — a missing or blank accession gets the table's ``"Unknown cell line."`` vector;
  an accession that is simply absent raises, because silently substituting "unknown" for a
  real cell line would hide the mistake.
* **assay** — free text is canonised by :func:`normalize_assay` into one of the
  :data:`ASSAY_VOCAB` keys, every one of which the table holds, so there is nothing to embed
  on the fly: an unrecognised readout becomes ``"other"`` and a value that is missing, or
  says the method was not recorded, becomes ``"unknown"``. A key the table nonetheless lacks
  — a cache older than the vocabulary — still falls back to the ``"unknown"`` vector with a
  one-time warning listing the types the table does hold.
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
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from tackai.data.utils import get_cache_dir

CELL_MODEL = "sentence-transformers/all-mpnet-base-v1"
ASSAY_MODEL = "sentence-transformers/all-mpnet-base-v1"
POI_ESM_MODEL = "facebook/esm2_t30_150M_UR50D"

# ------------------------------- assay vocabulary ------------------------------------------- #
# The assay block is a *closed* vocabulary: :func:`normalize_assay` folds every raw string into
# one of these keys, so a table built from ASSAY_VOCAB covers every value the encoder can be
# asked for. ``"other"`` absorbs a real but unrecognised readout; ``"unknown"`` is reserved for
# a missing value or text that states the method was not recorded.
#
# To add a category:
#   1. add its key to ASSAY_VOCAB and the pattern selecting it to ASSAY_RULES. Order matters --
#      the first match wins, so a compound rule ("elisa + western blot") goes above its parts,
#      and a specific platform above the generic family it belongs to;
#   2. write its descriptions in ``scripts/embeddings/generate_assay_embeddings.py``, which
#      refuses to run while a key has none;
#   3. regenerate the table with that script and publish it. The vocabulary size is part of the
#      file name, so a new category lands the table under a name no release has used.

#: Every assay key the cached table holds, in a fixed order. The size is part of the file name.
ASSAY_VOCAB: Tuple[str, ...] = (
    # antibody-based
    "western blot", "in-cell immunoassay", "in-cell immunoassay + western blot",
    "capillary immunoassay", "elisa", "elisa + western blot", "dot blot", "msd", "htrf",
    "alphalisa", "gel densitometry",
    # luminescent / complementation reporters
    "hibit", "nanobit", "nanoluc reporter", "luciferase reporter", "incell hunter",
    # fluorescent reporters and imaging
    "fluorescent reporter", "egfp/mcherry reporter assay",
    "fluoresyn nanobody biosensor (gfp immunostaining)", "fluorescence imaging",
    "high-content imaging", "flow cytometry",
    # label-free
    "mass spectrometry",
    # cell health, catch-all, and absent
    "viability assay", "other", "unknown",
)

#: Text stating that the method was not recorded. Checked before :data:`ASSAY_RULES`, so an
#: explicit "not specified" resolves to ``"unknown"`` rather than to ``"other"``.
_ASSAY_UNSPECIFIED = re.compile(r"not specified|unspecified|not reported|not stated|\bn/?a\b")

#: ``(key, pattern)`` in priority order; the first pattern matching the normalised text wins.
ASSAY_RULES: Tuple[Tuple[str, str], ...] = (
    # Compound readouts first: either single rule would otherwise claim them.
    # "in cell" is anchored at a word boundary: without it the pattern also fires inside
    # unrelated text such as "within cells".
    ("in-cell immunoassay + western blot", r"\bin[- ]?cell.*(western|immunoblot)"
                                          r"|(western|immunoblot).*\bin[- ]?cell"),
    ("elisa + western blot", r"elisa.*(western|immunoblot|\bwb\b)"
                             r"|(western|immunoblot|\bwb\b).*elisa"),
    # Named platforms, each above the family it belongs to.
    ("capillary immunoassay", r"capillary|simple western|\bjess\b|\bwes\b|\bceia\b|proteinsimple"),
    ("htrf", r"htrf"),
    ("alphalisa", r"alpha\s?lisa|amplified luminescence proximity"),
    ("msd", r"\bmsd\b|meso\s?scale"),
    ("hibit", r"hibit"),
    ("nanobit", r"nanobit"),
    ("nanoluc reporter", r"nanoluc|nanoluciferase|nano-?glo"),
    ("incell hunter", r"incell\s?hunter|enzyme fragment complementation|\befc\b"),
    ("luciferase reporter", r"luciferase|firefly|renilla"),
    ("flow cytometry", r"flow cytometr|\bfacs\b"),
    ("high-content imaging", r"high[- ]?content|operetta"),
    ("in-cell immunoassay", r"\bin[- ]?cell|cytoblot"),
    ("dot blot", r"dot blot"),
    # "wb" is matched as a whole word, not a substring: it is the abbreviation AutoTPD-TPDdb
    # uses, but it also occurs inside unrelated tokens ("wbc"), which must not match.
    ("western blot", r"western|immunoblot|\bwb\b|licor"),
    ("elisa", r"elisa"),
    ("gel densitometry", r"densitometr|sds-?page|coomassie"),
    # A named fluorescent tag is a more specific signal than the microscope reading it, so the
    # reporter rules sit above generic imaging.
    ("fluoresyn nanobody biosensor (gfp immunostaining)", r"nanobody|fluoresyn"),
    ("egfp/mcherry reporter assay", r"mcherry"),
    ("fluorescent reporter", r"\be?gfp\b|egfp|fluorescent protein|\byfp\b|mneongreen"),
    ("fluorescence imaging", r"immunofluorescen|fluorescen|imaging|microscop|incucyte"
                             r"|immunostain"),
    ("mass spectrometry", r"mass spec|proteomic|lc-?ms|dia-?ms|\bms\b|\bprm\b|\btmt\b"),
    ("viability assay", r"\bmts\b|cck-?8|celltiter|viability|cytotox|\bmtt\b"),
)

_ASSAY_RULES_COMPILED = tuple((key, re.compile(pat)) for key, pat in ASSAY_RULES)

#: Hugging Face Hub dataset repo published by scripts/publish_fusion_context.py and read by
#: FusionData.from_pretrained(). A dataset repo, not a model repo: these tables are an input
#: to every fusion model, never the output of one.
DEFAULT_CONTEXT_REPO = "ailab-bio/TACK-Context-Encoding"

MANIFEST_FORMAT = "tack-context-encoding/v1"

#: Blocks backed by a lookup table. ``assay_time`` is a scalar passed straight through.
TABLE_BLOCKS = ("cell", "poi", "e3", "combined", "assay")

#: Roles every block publishes: the embedding table and the PCA that produced it.
TABLE_ROLES = ("table", "pca_model")

PART_MANIFEST_FORMAT = "tack-context-encoding/part/v1"

#: Generator -> the part manifest it writes into the cache, the blocks it owns, and its script.
CONTEXT_GENERATORS: Dict[str, Dict] = {
    "protein": {"manifest": "manifest_protein.json", "blocks": ("poi", "e3", "combined"),
                "script": "scripts/embeddings/generate_protein_embeddings.py"},
    "cell": {"manifest": "manifest_cell.json", "blocks": ("cell",),
             "script": "scripts/embeddings/generate_cell_embeddings.py"},
    "assay": {"manifest": "manifest_assay.json", "blocks": ("assay",),
              "script": "scripts/embeddings/generate_assay_embeddings.py"},
}

# A published filename, as strict as the real ones allow: the stems carry '=', '-', '_' and
# '.', and nothing else. Anchored, so a separator or a parent reference cannot appear anywhere.
_SAFE_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._=+-]*\.npz$")


def _safe_basename(name: str, where: str) -> str:
    """Validate a filename read out of a manifest before it is ever joined to a path.

    A manifest key names a download; it is data, not code, and on an untrusted or corrupted
    repo it is attacker-controlled. Requiring a bare npz basename means such a key can neither
    escape the cache directory via a traversal path nor get installed as some other file.

    Args:
        name: The candidate filename, exactly as the manifest spells it.
        where: Path of the manifest, for the error message.

    Returns:
        The validated name, unchanged.

    Raises:
        ValueError: If the name is not a plain ``.npz`` basename.
    """
    if not isinstance(name, str) or ".." in name or not _SAFE_FILENAME.match(name):
        raise ValueError(
            f"{where} names a context file that is not a plain .npz filename: {name!r}. "
            "Manifest entries must be bare basenames; a path is never installed."
        )
    return name


def resolve_context_files(manifest: Dict, where: str = "manifest",
                          blocks: Sequence[str] = TABLE_BLOCKS) -> Dict[str, Dict[str, str]]:
    """Map each block to its table and PCA side file, as the manifest itself declares them.

    The published filenames encode each block's PCA width, so regenerating the tables renames
    all of them. Every manifest entry already records the ``block`` it belongs to and its
    ``role``, so the names are read from there rather than carried as a hardcoded copy that a
    regeneration would silently invalidate. A width this code pinned would only ever be a
    stale second opinion; the tables state their own, and
    :meth:`~tackai.fusion.data.FusionData.from_pretrained` checks them against the manifest's
    ``block_dims`` and the ensemble against its own ``n_features_in_``.

    Args:
        manifest: The parsed ``manifest.json``.
        where: Path of the manifest, for error messages.
        blocks: Blocks that must be present.

    Returns:
        ``{block: {"table": filename, "pca_model": filename}}``.

    Raises:
        ValueError: If ``files`` is missing or malformed, if a block has no entry for a role,
            if it has more than one (two generations staged together, which would otherwise
            pair tables from different PCA fits), or if a filename is not a bare basename.
    """
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError(f"{where} has no 'files' mapping; it does not look like a context "
                         "embedding repo published by stage_context_tables()")

    candidates: Dict[str, Dict[str, list]] = {}
    for name, entry in files.items():
        if not isinstance(entry, dict):
            continue
        block, role = entry.get("block"), entry.get("role")
        if block in blocks and role in TABLE_ROLES:
            candidates.setdefault(block, {}).setdefault(role, []).append(name)

    resolved: Dict[str, Dict[str, str]] = {}
    for block in blocks:
        for role in TABLE_ROLES:
            found = sorted(candidates.get(block, {}).get(role, []))
            if not found:
                raise ValueError(
                    f"{where} declares no {role} for block {block!r}. The published repo and "
                    f"this tackai version disagree about which blocks exist: it lists "
                    f"{sorted(candidates)} and {block!r} is required."
                )
            if len(found) > 1:
                raise ValueError(
                    f"{where} declares {len(found)} files as the {role} for block {block!r}: "
                    f"{found}. Exactly one is required — publishing two generations side by "
                    "side would pair tables from different PCA fits."
                )
            resolved.setdefault(block, {})[role] = _safe_basename(found[0], where)
    return resolved


def read_context_manifest(source: Union[str, Path]) -> Dict:
    """Parse the ``manifest.json`` of a staged, downloaded or installed context directory.

    Args:
        source: Directory expected to hold ``manifest.json``.

    Returns:
        The parsed manifest.

    Raises:
        FileNotFoundError: If the directory holds no manifest.
    """
    path = Path(source) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"no manifest.json found in {source}; this does not look like a context embedding "
            "repo published by scripts/publish_fusion_context.py. The manifest is what names "
            "the tables, so a directory without one cannot be read — install the tables with "
            "FusionData.from_pretrained(), or pass --context-repo to fetch them."
        )
    return json.loads(path.read_text())


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


def _file_entry(path: Path, block: str, role: str) -> Dict:
    """Manifest entry of one context file, with its dimensions read out of the npz itself."""
    entry = {"block": block, "role": role, "sha256": sha256_of(path),
             "bytes": path.stat().st_size}
    if role == "table":
        entry["dim"], entry["n_keys"] = _npz_table_dims(path)
    else:
        entry["full_dim"], entry["reduced_dim"] = _npz_pca_dims(path)
    return entry


def write_part_manifest(cache_dir: Union[str, Path], generator: str,
                        files: Dict[str, Dict[str, str]],
                        provenance: Dict[str, Dict]) -> Dict:
    """Record the tables one generator script just wrote, in its own manifest in the cache.

    Blocks this run did not write keep their existing entries, so
    ``generate_protein_embeddings.py --blocks poi`` does not forget ``e3`` and ``combined``.

    Args:
        cache_dir: Directory the generator wrote into.
        generator: A :data:`CONTEXT_GENERATORS` key.
        files: Block -> ``{"table": filename, "pca_model": filename}``, for the blocks written.
        provenance: Block -> how it was made. Must hold ``"model"``, which becomes the block's
            entry under ``models`` in the published manifest; the rest is recorded as given.

    Returns:
        The part manifest also written to ``cache_dir / CONTEXT_GENERATORS[generator]["manifest"]``.

    Raises:
        ValueError: If ``files`` names a block this generator does not own.
    """
    import tackai

    spec = CONTEXT_GENERATORS[generator]
    foreign = sorted(set(files) - set(spec["blocks"]))
    if foreign:
        raise ValueError(f"generator {generator!r} owns {spec['blocks']}, not {foreign}")

    cache_dir = Path(cache_dir)
    path = cache_dir / spec["manifest"]
    previous = json.loads(path.read_text()) if path.exists() else {}
    part = {
        "format": PART_MANIFEST_FORMAT,
        "generator": generator,
        "script": spec["script"],
        "files": {name: entry for name, entry in previous.get("files", {}).items()
                  if entry.get("block") not in files},
        "provenance": {block: record for block, record in previous.get("provenance", {}).items()
                       if block not in files},
    }
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for block, roles in files.items():
        for role in TABLE_ROLES:
            part["files"][roles[role]] = _file_entry(cache_dir / roles[role], block, role)
        part["provenance"][block] = {
            **provenance[block], "generated_at": generated_at,
            "tackai_version": getattr(tackai, "__version__", "unknown")}
    path.write_text(json.dumps(part, indent=1))
    return part


def aggregate_part_manifests(cache_dir: Union[str, Path]) -> Tuple[Dict[str, Dict[str, str]],
                                                                   Dict[str, Dict]]:
    """Merge every generator's part manifest, checking each file still matches its record.

    Args:
        cache_dir: Directory holding the tables and the part manifests.

    Returns:
        ``(files, provenance)``: block -> ``{"table": filename, "pca_model": filename}``, and
        block -> the provenance its generator recorded.

    Raises:
        SystemExit: If a part manifest is missing or incomplete, or a file it lists is missing
            or was rewritten after it was recorded (every problem is listed before exiting).
    """
    cache_dir = Path(cache_dir)
    files: Dict[str, Dict[str, str]] = {}
    provenance: Dict[str, Dict] = {}
    problems = []
    for spec in CONTEXT_GENERATORS.values():
        path, script = cache_dir / spec["manifest"], spec["script"]
        if not path.exists():
            problems.append(f"{spec['manifest']} is missing: run {script}")
            continue
        part = json.loads(path.read_text())
        if part.get("format") != PART_MANIFEST_FORMAT:
            problems.append(f"{spec['manifest']} has format {part.get('format')!r}, expected "
                            f"{PART_MANIFEST_FORMAT!r}: run {script}")
            continue
        try:
            resolved = resolve_context_files(part, where=str(path), blocks=spec["blocks"])
        except ValueError as err:
            problems.append(f"{err} Run {script}.")
            continue
        for block, roles in resolved.items():
            for name in roles.values():
                if not (cache_dir / name).exists():
                    problems.append(f"{name} is missing: run {script}")
                elif sha256_of(cache_dir / name) != part["files"][name]["sha256"]:
                    problems.append(f"{name} differs from {spec['manifest']}, so it was rewritten "
                                    f"after that run: run {script}")
            files[block] = roles
            provenance[block] = part["provenance"][block]
    if problems:
        print(f"{len(problems)} problem(s) with the context tables in {cache_dir}:")
        for problem in problems:
            print(f"  {problem}")
        print("Refusing to publish a partial or inconsistent set of context tables.")
        raise SystemExit(1)
    return files, provenance


def stage_from_part_manifests(cache_dir: Union[str, Path], out_dir: Union[str, Path], *,
                              repo_id: str = DEFAULT_CONTEXT_REPO) -> Dict:
    """Stage the tables the generators' part manifests name, with an aggregated manifest.

    Args:
        cache_dir: Directory holding the tables and the part manifests.
        out_dir: Staging directory to write into; refused if it already exists.
        repo_id: Hub repo id the README points readers at.

    Returns:
        The manifest also written to ``out_dir / "manifest.json"``.

    Raises:
        SystemExit: See :func:`aggregate_part_manifests` and :func:`stage_context_tables`.
    """
    files, provenance = aggregate_part_manifests(cache_dir)
    return stage_context_tables(cache_dir, out_dir, repo_id=repo_id, files=files,
                                provenance=provenance)


def _readme_text(repo_id: str, manifest: Dict) -> str:
    """Small human-facing card for the staged directory: what this is, how to load it."""
    blocks = ", ".join(f"{b} ({d}-d)" for b, d in sorted(manifest["block_dims"].items())
                       if b != "assay_time")
    protein_space = manifest.get("protein_space")
    space_note = (f" Published from a `protein_space=\"{protein_space}\"` encoder; "
                  "`from_pretrained` reproduces that by default." if protein_space
                  else "")
    return f"""# {repo_id}

PCA-reduced context embedding tables for the `tackai` fusion surrogates (`tackai.fusion`):
one lookup table plus a PCA side-file per block ({blocks}), and `manifest.json` recording
each file's sha256, byte size and dimensions.{space_note}

These tables are **inputs** to every fusion model, never the output of one — a fitted
ensemble is only meaningful against the exact tables it was built with, so this repo is
versioned by commit rather than by file content.

## Loading

```python
from tackai.fusion.data import FusionData

data = FusionData.from_pretrained("{repo_id}")
context = data.transform_context({{"poi_seq": ..., "e3_seq": ..., "cell_id": ..., "assay": ...}})
```

`from_pretrained` installs these files into `TACKAI_CACHE` and verifies them against this
repo's `manifest.json` before use.

Generated by `tackai.fusion.context.stage_context_tables` (tackai version
{manifest["tackai_version"]}) — do not hand-edit; re-stage and re-upload instead.
"""


def stage_context_tables(cache_dir: Union[str, Path], out_dir: Union[str, Path], *,
                         repo_id: str = DEFAULT_CONTEXT_REPO,
                         files: Dict[str, Dict[str, str]],
                         protein_space: Optional[str] = None,
                         provenance: Optional[Dict[str, Dict]] = None) -> Dict:
    """Copy every context table from ``cache_dir`` into ``out_dir``, with a manifest and a
    small README.

    Shared by ``scripts/publish_fusion_context.py`` (publishing a whole cache) and
    :meth:`~tackai.fusion.data.FusionData.push_to_hub` (publishing one instance's own cache),
    so both produce the exact manifest format :meth:`~tackai.fusion.data.FusionData.from_pretrained`
    expects.

    Both the ``per_block`` and ``combined`` protein tables are always staged — a reader may
    want either — but which one a *particular* publishing :class:`~tackai.fusion.data.FusionData`
    instance actually encodes with is otherwise invisible from the files alone (they are
    byte-identical either way). Recording it in the manifest lets :meth:`~tackai.fusion.data.FusionData.from_pretrained`
    reproduce the publisher's own space by default instead of silently falling back to
    ``per_block`` and handing back a context vector of the wrong width.

    Args:
        cache_dir: Directory holding the cached context tables (``TACKAI_CACHE``).
        out_dir: Staging directory to write into; refused if it already exists.
        files: Block -> ``{"table": filename, "pca_model": filename}`` to stage.
        repo_id: Hub repo id the README points readers at (purely descriptive here — it is
            not uploaded to until :func:`upload_context_tables` runs).
        protein_space: The ``protein_space`` of the encoder these tables are being staged
            for (``"per_block"`` or ``"combined"``), recorded in the manifest so
            :meth:`~tackai.fusion.data.FusionData.from_pretrained` can default to it. ``None``
            (the default, used by the maintainer CLI when staging a whole cache rather than
            one encoder's) means the manifest does not commit to either space, and a reader
            without its own ``protein_space`` falls back to ``"per_block"``.
        provenance: Block -> how it was made, published under ``"provenance"``. Each record's
            ``"model"`` also fills the manifest's ``"models"``.

    Returns:
        The manifest dict also written to ``out_dir / "manifest.json"``.

    Raises:
        ValueError: If ``protein_space`` is not ``None``, ``"per_block"`` or ``"combined"``.
        SystemExit: If ``out_dir`` already exists, or any file to stage is missing from
            ``cache_dir`` (every missing file is listed before exiting).
    """
    import tackai

    if protein_space not in (None, "per_block", "combined"):
        raise ValueError(f"protein_space must be None, 'per_block' or 'combined', "
                         f"got {protein_space!r}")

    cache_dir, out_dir = Path(cache_dir), Path(out_dir)
    if out_dir.exists():
        print(f"Staging directory already exists: {out_dir}")
        print("Remove it (or move it aside) before re-staging, to avoid mixing stale and "
              "fresh files.")
        raise SystemExit(1)

    wanted = [(filename, block, role)
              for block, roles in files.items()
              for role, filename in roles.items()]
    missing = [filename for filename, _, _ in wanted if not (cache_dir / filename).exists()]
    if missing:
        print(f"{len(missing)} file(s) missing from {cache_dir}:")
        for name in missing:
            print(f"  {name}")
        print("Refusing to publish a partial set of context tables.")
        raise SystemExit(1)

    models = ({"cell": CELL_MODEL, "assay": ASSAY_MODEL, "poi": POI_ESM_MODEL,
               "e3": POI_ESM_MODEL} if provenance is None
              else {block: record["model"] for block, record in provenance.items()})
    manifest = {
        "format": MANIFEST_FORMAT,
        "tackai_version": getattr(tackai, "__version__", "unknown"),
        "protein_space": protein_space,
        "block_dims": {"assay_time": 1}, "combined_dim": None,
        "models": models,
        "files": {},
    }
    if provenance is not None:
        manifest["provenance"] = provenance
    out_dir.mkdir(parents=True)
    for filename, block, role in wanted:
        src = cache_dir / filename
        entry = _file_entry(src, block, role)
        if role == "table":
            if block == "combined":
                manifest["combined_dim"] = entry["dim"]
            else:
                manifest["block_dims"][block] = entry["dim"]
        manifest["files"][filename] = entry
        shutil.copy2(src, out_dir / filename)
        print(f"  staged {filename} ({block}/{role})")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out_dir / "README.md").write_text(_readme_text(repo_id, manifest))
    print(f"\nStaged {len(wanted)} files + manifest.json + README.md -> {out_dir}")
    return manifest


def upload_context_tables(staged_dir: Union[str, Path], repo_id: str, private: bool,
                          commit_message: str) -> str:
    """Create (if needed) the dataset repo and make its tables exactly those in ``staged_dir``.

    Any ``.npz`` the staged folder does not hold is deleted in the same commit, so ``main`` holds
    one generation and a snapshot download fetches nothing stale. Older generations stay
    reachable through their commit.

    Args:
        staged_dir: Directory written by :func:`stage_context_tables`.
        repo_id: Hugging Face Hub repo id.
        private: Create the repo as private if it does not exist yet.
        commit_message: Commit message for the upload.

    Returns:
        The commit sha ``upload_folder`` reports.
    """
    from huggingface_hub import HfApi
    api = HfApi()
    print(f"Authenticated as: {api.whoami()['name']}")
    print(f"Creating (if needed) dataset repo {repo_id} ...")
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=private)
    print(f"Uploading {staged_dir} -> {repo_id} ...")
    commit = api.upload_folder(repo_id=repo_id, repo_type="dataset",
                               folder_path=str(staged_dir), commit_message=commit_message,
                               delete_patterns="*.npz")
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


#: U -> C (selenocysteine), O -> K (pyrrolysine), B -> D (Asx), Z -> E (Glx), J -> L (Xle).
_SEQUENCE_REPLACEMENTS = (("U", "C"), ("O", "K"), ("B", "D"), ("Z", "E"), ("J", "L"))


def _sanitize_sequence_key(seq) -> str:
    """Normalise non-standard amino-acid codes, as the embedding generator does.

    A duplicate of :func:`tackai.data.embeddings.protein_embeddings.sanitize_sequence`, kept
    here on purpose: that module imports ``torch`` and ``h5py`` at module level, and
    ``tackai.fusion`` is deliberately importable without either.
    ``test_sanitisation_agrees_with_the_canonical_implementation`` pins the two together.
    """
    if not isinstance(seq, str):
        return ""
    seq = seq.strip()
    for old, new in _SEQUENCE_REPLACEMENTS:
        seq = seq.replace(old, new)
    return seq


def normalize_assay(value) -> str:
    """Normalise a raw assay string to one of the :data:`ASSAY_VOCAB` keys.

    The vocabulary is closed, so every return value has a vector in the cached table:

    * a missing value, or text saying the method was not recorded, gives ``"unknown"``;
    * the first rule of :data:`ASSAY_RULES` whose pattern matches gives that rule's key.
      Abbreviations are folded into the type they stand for — ``"wb"`` is western blot —
      because the vocabulary already holds that signal;
    * any other non-blank text gives ``"other"``: a real readout this vocabulary has no
      category for, which is a different thing from not knowing the method at all.

    Add a category by extending :data:`ASSAY_VOCAB` and :data:`ASSAY_RULES` — see the comment
    above them — rather than by special-casing here.

    Args:
        value: Raw assay description, possibly missing.

    Returns:
        One of :data:`ASSAY_VOCAB`.
    """
    if _is_missing(value):
        return "unknown"
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    text = re.sub(r"[‐‑–—−]", "-", text)
    text = re.sub(r"\s+", " ", text)
    if text in {"", "nan", "none", "missing", "unknown", "n/a"} or _ASSAY_UNSPECIFIED.search(text):
        return "unknown"
    for key, pattern in _ASSAY_RULES_COMPILED:
        if pattern.search(text):
            return key
    return "other"


class ContextEncoder:
    """Look up the cached context embedding of every POI, E3 ligase, cell line and assay.

    Args:
        cache_dir: Directory holding the npz tables (default: ``TACKAI_CACHE``).
        protein_space: ``"per_block"`` reads POI from its own 51-d table and E3 from its 7-d
            table; ``"combined"`` reads both from the shared 52-d space.
    """

    NOT_FOUND = "Unknown cell line."

    def __init__(self, cache_dir: Optional[str] = None, protein_space: str = "per_block",
                 files: Optional[Dict[str, Dict[str, str]]] = None):
        if protein_space not in {"per_block", "combined"}:
            raise ValueError(f"protein_space must be 'per_block' or 'combined', got {protein_space!r}")
        self.cache_dir = Path(cache_dir or get_cache_dir())
        self.protein_space = protein_space
        # The names come from the manifest installed alongside the tables, never from a copy
        # kept in this module: a cache can hold more than one generation, and only the
        # manifest says which table goes with which PCA fit.
        self._files = files if files is not None else resolve_context_files(
            read_context_manifest(self.cache_dir), where=str(self.cache_dir / "manifest.json"))
        self._tables: Dict[str, Dict[str, np.ndarray]] = {}
        self._warned_assay_misses: set = set()

    @property
    def context_files(self) -> Dict[str, Dict[str, str]]:
        """Every block's table and PCA side-file names, as the manifest declares them."""
        return self._files

    def files(self, block: str) -> Dict[str, str]:
        """The cached table and PCA side-file names backing a block.

        Args:
            block: ``"poi"``, ``"e3"``, ``"cell"`` or ``"assay"``.

        Returns:
            Mapping with ``"table"`` and ``"pca_model"`` filenames.
        """
        if block in SEQUENCE_BLOCKS and self.protein_space == "combined":
            return self._files["combined"]
        if block not in self._files:
            raise KeyError(f"unknown context block {block!r}")
        return self._files[block]

    def _lookup_key(self, block: str, key, table: Dict[str, np.ndarray]) -> str:
        """The table key a raw key resolves to: the stripped string, or, for a protein block
        that does not hold it, its sanitised form.

        The protein tables are keyed by sanitised sequences, so a sequence carrying a
        non-standard residue is stored under a key the stripped string never matches. The
        stripped form is still tried first, so a table holding the raw string keeps working.
        """
        name = str(key).strip()
        if name in table or block not in SEQUENCE_BLOCKS:
            return name
        sanitized = _sanitize_sequence_key(name)
        return sanitized if sanitized in table else name

    def fingerprint(self) -> str:
        """Short content hash of the cached tables this encoder reads.

        Names and widths do not identify a table: refitting a PCA over a larger vocabulary
        keeps both and changes every vector. Anything that caches the output of this encoder
        has to include this in its key, or it serves vectors from a space that no longer
        exists — see :meth:`~tackai.fusion.data.FusionData._cache_dir`.

        Returns:
            The first 12 hex characters of a sha256 over the protein space and the content of
            every block table, in a fixed order.
        """
        h = hashlib.sha256(self.protein_space.encode())
        for name in sorted({self.files(b)["table"] for b in CONTEXT_BLOCKS if b != "assay_time"}):
            h.update(name.encode())
            h.update(sha256_of(self.cache_dir / name).encode())
        return h.hexdigest()[:12]

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
            name = self._lookup_key(block, key, table)
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
        hitting the :meth:`encode` exception. Assay text is always encodable, because an
        unseen canonical type falls back to the table's ``"unknown"`` vector.

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
                out.append(self._lookup_key(block, key, table) in table)
        return np.array(out, dtype=bool)

    def _encode_assay(self, raws: Sequence) -> np.ndarray:
        """Assay vectors; a canonical type the table does not hold resolves to ``"unknown"``."""
        table = self.table("assay")
        names = [normalize_assay(r) for r in raws]
        misses = sorted({n for n in names if n not in table} - self._warned_assay_misses)
        if misses:
            available = ", ".join(sorted(table))
            warnings.warn(
                f"assay type(s) not in the cached table, falling back to \"unknown\": "
                f"{', '.join(misses)}. Available assay types: {available}",
                stacklevel=2,
            )
            self._warned_assay_misses.update(misses)
        return np.stack([table[n] if n in table else table["unknown"]
                         for n in names]).astype(np.float32)

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
