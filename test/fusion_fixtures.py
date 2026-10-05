"""Shared constants for the ``tackai.fusion`` tests.

Kept out of ``conftest.py`` so test modules can import them directly. The module lives
in ``test/`` (which pytest puts on ``sys.path``) rather than a ``test`` package, because
``test`` is a standard-library package name and importing ``test.conftest`` would shadow it.
"""

CELL_FILE = "cell_embeddings_model=sentence-transformer_pooling=mean_pca47.npz"
POI_FILE = ("protein_embeddings_esm_model=facebook-esm2_t30_150M_UR50D_layer=18"
            "_pooling=lse_window=1022_block=poi_pca51.npz")
COMBINED_FILE = ("protein_embeddings_esm_model=facebook-esm2_t30_150M_UR50D_layer=18"
                 "_pooling=lse_window=1022_block=combined_pca52.npz")
E3_FILE = ("protein_embeddings_esm_model=facebook-esm2_t30_150M_UR50D_layer=30"
           "_pooling=mean_rm_pc2_window=1022_block=e3_pca7.npz")
ASSAY_FILE = "assay_embeddings_vocab=open20_model=all-mpnet-base-v1_pooling=mean_pca8.npz"
ASSAY_PCA_FILE = "assay_pca_vocab=open20_model=all-mpnet-base-v1_pooling=mean_pca8.npz"

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

ASSAYS = ["western blot", "htrf", "hibit", "capillary immunoassay", "elisa", "unknown",
          "nanoluc reporter", "flow cytometry", "in-cell immunoassay", "msd",
          "high-content imaging", "fluorescence imaging", "dot blot", "viability assay",
          "incell hunter", "elisa + western blot", "in-cell immunoassay + western blot",
          "mass spectrometry", "immunoprecipitation", "reporter assay"]

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
