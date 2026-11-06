"""Shared constants for the ``tackai.fusion`` tests.

Kept out of ``conftest.py`` so test modules can import them directly. The module lives
in ``test/`` (which pytest puts on ``sys.path``) rather than a ``test`` package, because
``test`` is a standard-library package name and importing ``test.conftest`` would shadow it.
"""

from tackai.fusion.context import (ASSAY_VOCAB, CONTEXT_GENERATORS,
                                   resolve_context_files, write_part_manifest)
from tackai.fusion.data import BLOCK_ORDER, block_index  # noqa: F401  (re-exported for tests)
from tackai.fusion.mol_encoder import FP_SIZE

#: Widths of a FusionData built on the synthetic cache below and the default featuriser.
#: ``test_fusion_data.py`` checks that they agree with what the library discovers.
BLOCK_DIMS = {"fingerprint": FP_SIZE, "descriptors": 216, "e3": 24, "cell": 56, "poi": 78,
              "assay": 8, "assay_time": 1}

#: Width of the shared POI+E3 table, which ``protein_space="combined"`` reads instead of the
#: two per-block ones. Assert against this rather than a literal: the published tables are
#: refitted periodically, and a test that repeats a width goes stale the same way the library
#: used to.
COMBINED_DIM = 86
DEFAULT_BLOCKS = block_index(BLOCK_DIMS)

CELL_FILE = "cell_embeddings_model=sentence-transformer_pooling=mean_pca56.npz"
POI_FILE = ("protein_embeddings_esm_model=facebook-esm2_t30_150M_UR50D_layer=18"
            "_pooling=lse_window=1022_block=poi_pca78.npz")
COMBINED_FILE = ("protein_embeddings_esm_model=facebook-esm2_t30_150M_UR50D_layer=18"
                 "_pooling=lse_window=1022_block=combined_pca86.npz")
E3_FILE = ("protein_embeddings_esm_model=facebook-esm2_t30_150M_UR50D_layer=30"
           "_pooling=mean_rm_pc2_window=1022_block=e3_pca24.npz")
NOT_FOUND = "Unknown cell line."

# Short stand-in sequences; the real tables are keyed by full amino-acid sequences.
SEQS = {
    "poi": ["MAGEGDQQDAAHNMGNHLPLLPAESEEEDEMEVEDQ",
            "MPRRAENWDEAEVGAEEAGVEEYGPEEDGGEESGAEE",
            "MHKTASQRLFPGPSYQNIKSIMEDSTILSDWTNSNK",
            "MSAEVIHQVEEALDTDEKEMLLFLCRDVAIDVVPPN"],
    "e3": ["MEPVRRSSRLSAQKQQQQQQQQAEDEEMEVEDQDSK",
           "MAAGSIEPVRRSSRLSAQKQQQQQAEDEEMEVEDQ"],
}

CELLS = ["CVCL_0031", "CVCL_0395", "CVCL_0291", "CVCL_0062"]

#: The real assay vocabulary: it is closed, so the synthetic table holds exactly the keys
#: ``normalize_assay`` can return. Tests index the first few, and every one of those is a key.
ASSAYS = list(ASSAY_VOCAB)

SMILES = [
    "CC(C)(C)c1ccc(cc1)C(=O)NC1CCN(CC1)C(=O)c1ccccc1",
    "O=C(Nc1ccc(cc1)S(=O)(=O)N)C1CCN(CC1)Cc1ccccc1",
    "COc1ccc(cc1)C(=O)N1CCN(CC1)c1ncccn1",
    "CC(=O)Nc1ccc(cc1)C(=O)N1CCC(CC1)Oc1ccccc1",
    "Clc1ccc(cc1)C(=O)NC1CCN(CC1)C(=O)OC(C)(C)C",
    "OC(=O)C1CCN(CC1)C(=O)c1ccc(cc1)N1CCOCC1",
    "CN1CCN(CC1)c1ccc(cc1)NC(=O)c1ccc(Cl)cc1",
    "CCOC(=O)c1ccc(cc1)N1CCN(CC1)C(=O)c1ccccc1",
]


#: Name templates of the published context files, parameterised by PCA width. The published
#: repo encodes each block's width in its filename, so a regenerated table arrives under a name
#: no previous release used. Tests build sources at arbitrary widths from these.
_PROTEIN_STEM = "protein_embeddings_esm_model=facebook-esm2_t30_150M_UR50D"
_ASSAY_STEM = f"vocab=open{len(ASSAY_VOCAB)}_model=all-mpnet-base-v1_pooling=mean"
NAME_TEMPLATES = {
    "cell": ("cell_embeddings_model=sentence-transformer_pooling=mean_pca{d}.npz",
             "cell_embeddings_model=sentence-transformer_pooling=mean_pca{d}_model.npz"),
    "poi": (f"{_PROTEIN_STEM}_layer=18_pooling=lse_window=1022_block=poi_pca{{d}}.npz",
            f"{_PROTEIN_STEM}_layer=18_pooling=lse_window=1022_block=poi_pca{{d}}_model.npz"),
    "e3": (f"{_PROTEIN_STEM}_layer=30_pooling=mean_rm_pc2_window=1022_block=e3_pca{{d}}.npz",
           f"{_PROTEIN_STEM}_layer=30_pooling=mean_rm_pc2_window=1022_block=e3_pca{{d}}_model.npz"),
    "combined": (f"{_PROTEIN_STEM}_layer=18_pooling=lse_window=1022_block=combined_pca{{d}}.npz",
                 f"{_PROTEIN_STEM}_layer=18_pooling=lse_window=1022_block=combined_pca{{d}}"
                 "_model.npz"),
    "assay": (f"assay_embeddings_{_ASSAY_STEM}_pca{{d}}.npz",
              f"assay_pca_{_ASSAY_STEM}_pca{{d}}.npz"),
}

# Built from the template: the assay file name carries the vocabulary size, so adding an assay
# category renames it and a literal here would silently stop matching.
ASSAY_FILE, ASSAY_PCA_FILE = (t.format(d=8) for t in NAME_TEMPLATES["assay"])

#: Keys each block's table is written with, matching the synthetic cache in conftest.
BLOCK_KEYS = {"cell": CELLS + [NOT_FOUND], "poi": SEQS["poi"], "e3": SEQS["e3"],
              "combined": SEQS["poi"] + SEQS["e3"], "assay": ASSAYS}

#: Width of the embedding each block's PCA reduces, before reduction.
FULL_DIMS = {"cell": 768, "poi": 640, "e3": 640, "combined": 640, "assay": 768}


def write_context_source(out_dir, dims, *, protein_space=None, mutate=None):
    """Write a published-shaped context source: npz tables plus the manifest describing them.

    Mirrors what :func:`tackai.fusion.context.stage_context_tables` publishes, but at widths
    the caller chooses, so a test can present a generation the library has never seen.

    Args:
        out_dir: Directory to create and write into.
        dims: Block name -> PCA width, e.g. ``{"cell": 40, "poi": 33, ...}``.
        protein_space: Value to record under ``"protein_space"`` in the manifest.
        mutate: Optional callable receiving the finished manifest dict for in-place edits
            before it is written, for tests that need a malformed one.

    Returns:
        The manifest that was written, as a dict.
    """
    import hashlib
    import json
    from pathlib import Path

    import numpy as np

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "tack-context-encoding/v1",
        "tackai_version": "test",
        "protein_space": protein_space,
        "block_dims": {"assay_time": 1},
        "combined_dim": None,
        "models": {},
        "files": {},
    }
    for seed, (block, dim) in enumerate(sorted(dims.items())):
        table_name, pca_name = (t.format(d=dim) for t in NAME_TEMPLATES[block])
        rng = np.random.default_rng(seed + 1)
        keys = BLOCK_KEYS[block]
        np.savez(out_dir / table_name,
                 **{k: rng.normal(size=dim).astype(np.float32) for k in keys})
        full = FULL_DIMS[block]
        np.savez(out_dir / pca_name,
                 mean_=rng.normal(size=full).astype(np.float32),
                 components_=rng.normal(size=(dim, full)).astype(np.float32),
                 explained_variance_ratio_=np.full(dim, 1.0 / dim, dtype=np.float32))
        if block == "combined":
            manifest["combined_dim"] = dim
        else:
            manifest["block_dims"][block] = dim
        for name, role in ((table_name, "table"), (pca_name, "pca_model")):
            path = out_dir / name
            entry = {"block": block, "role": role,
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "bytes": path.stat().st_size}
            if role == "table":
                entry["dim"], entry["n_keys"] = dim, len(keys)
            else:
                entry["full_dim"], entry["reduced_dim"] = full, dim
            manifest["files"][name] = entry
    if mutate is not None:
        mutate(manifest)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def write_part_manifests(cache_dir, manifest):
    """Record the tables ``manifest`` names as if each generator script had just written them.

    Args:
        cache_dir: Directory holding the tables.
        manifest: The manifest returned by :func:`write_context_source`.
    """
    for generator, spec in CONTEXT_GENERATORS.items():
        files = resolve_context_files(manifest, blocks=spec["blocks"])
        write_part_manifest(cache_dir, generator, files,
                            {block: {"model": f"test/{generator}", "source": {"dataset": "test"}}
                             for block in files})


def build_ensemble(factory, data, task, n_members=3, n_folds=3):
    """Fit one member per fold and wrap them in a FusionEnsemble, for tests.

    Deliberately test-only: the package's own ensemble-fitting entry point is being written
    separately, and this must not pre-empt its name or its behaviour.

    A fold whose training labels hold a single value is skipped, because ``fit_member`` refuses
    it by design. The tiny test table has one such fold in its activity task; the real tables
    have none. That refusal is tested directly in ``test_fusion_training.py``.

    Args:
        factory: Callable returning a fresh estimator, called with ``random_state``; or a list
            of them, one per member.
        data: The :class:`~tackai.fusion.data.FusionData` to draw rows and splits from.
        task: Task name.
        n_members: Number of folds to fit (ignored when ``factory`` is a list).
        n_folds: Folds per repeat.

    Returns:
        A fitted :class:`~tackai.fusion.ensemble.FusionEnsemble`.
    """
    import numpy as np
    from functools import partial

    from tackai.fusion.data import TASK_TYPES
    from tackai.fusion.ensemble import FusionEnsemble
    from tackai.fusion.training import fit_member

    _, X, y, _ = data.task_rows(task)
    folds = [f for f in data.splits(task, 1, n_folds)[0] if len(np.unique(y[f[0]])) > 1]
    makers = list(factory) if isinstance(factory, (list, tuple)) else None
    count = len(makers) if makers is not None else min(n_members, len(folds))
    members = [fit_member(partial(makers[k] if makers else factory, task_type=TASK_TYPES[task]),
                          X, y, folds[k][0], random_state=k)
               for k in range(count)]
    return FusionEnsemble(members, data, task)
