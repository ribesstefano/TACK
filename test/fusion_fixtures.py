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
