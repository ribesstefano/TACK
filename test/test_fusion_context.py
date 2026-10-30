"""Context embeddings read from the cached, already PCA-reduced npz tables."""
import warnings

import numpy as np
import pytest

from fusion_fixtures import ASSAYS, CELLS, SEQS
from tackai.fusion.context import ContextEncoder, normalize_assay


def test_normalize_assay_ported_rules():
    assert normalize_assay("Western Blot") == "western blot"
    assert normalize_assay("WESTERN BLOT / ELISA") == "elisa + western blot"
    assert normalize_assay("Simple Western") == "capillary immunoassay"
    assert normalize_assay(None) == "unknown"
    assert normalize_assay("") == "unknown"
    assert normalize_assay(float("nan")) == "unknown"
    assert normalize_assay("HiBiT assay") == "hibit"
    assert normalize_assay("some novel readout") == "some novel readout"


def test_normalize_assay_maps_wb_abbreviation():
    """`wb` is western blot: AutoTPD-TPDdb uses the abbreviation, and without this rule those
    rows fall back to the "unknown" assay vector instead of the one they share with
    "western blot"."""
    assert normalize_assay("WB") == "western blot"
    assert normalize_assay("wb") == "western blot"
    assert normalize_assay(" WB ") == "western blot"
    assert normalize_assay("WB/ELISA") == "elisa + western blot"
    # Matched as a word, so a longer token that merely contains "wb" is still passed through.
    assert normalize_assay("wbc count") == "wbc count"


def test_sequences_with_non_standard_residues_resolve_to_their_sanitised_key(fake_cache):
    """The protein tables are keyed by ``sanitize_sequence`` output, so a sequence carrying a
    non-standard residue has to be looked up through the same normalisation or it is
    unreachable — which silently dropped 15 AutoTPD-TPDdb rows over one selenocysteine."""
    enc = ContextEncoder()
    stored = SEQS["poi"][3]
    assert "C" in stored
    raw = stored.replace("C", "U", 1)        # selenocysteine; sanitize_sequence maps U -> C
    assert raw not in enc.table("poi")
    assert enc.encodable("poi", [raw]).tolist() == [True]
    np.testing.assert_array_equal(enc.encode("poi", [raw]), enc.encode("poi", [stored]))


def test_sanitisation_agrees_with_the_canonical_implementation():
    """``context.py`` keeps its own copy to stay free of the torch/h5py imports that
    ``protein_embeddings`` pulls in; this pins the two together."""
    from tackai.data.embeddings.protein_embeddings import sanitize_sequence
    from tackai.fusion.context import _sanitize_sequence_key

    for seq in ["MUOBZJCK", "  MACDEK  ", "MACDEK", "", "UUU"]:
        assert _sanitize_sequence_key(seq) == sanitize_sequence(seq)


def test_fingerprint_tracks_the_content_of_the_tables(fake_cache):
    """The fingerprint is what lets a derived cache notice that a table was refitted.

    Refitting a PCA over a larger vocabulary leaves the table's name and width alone while
    changing every vector, so anything caching encoder output has to key on content.
    """
    from tackai.fusion.context import CONTEXT_FILES

    enc = ContextEncoder()
    before = enc.fingerprint()
    assert before == ContextEncoder().fingerprint()        # same cache, same answer

    path = fake_cache / CONTEXT_FILES["poi"]["table"]
    with np.load(path) as data:
        tables = {k: data[k] for k in data.files}
    tables[next(iter(tables))] += 1.0                      # same keys, same width, new vectors
    np.savez(path, **tables)
    assert ContextEncoder().fingerprint() != before


def test_dims_match_the_cached_tables(fake_cache):
    enc = ContextEncoder()
    assert (enc.dim("poi"), enc.dim("e3"), enc.dim("cell"), enc.dim("assay")) == (78, 24, 56, 8)


def test_sequences_are_looked_up_by_stripped_string(fake_cache):
    enc = ContextEncoder()
    a = enc.encode("poi", [SEQS["poi"][0]])
    b = enc.encode("poi", ["  " + SEQS["poi"][0] + "\n"])
    assert a.shape == (1, 78) and np.array_equal(a, b)


def test_unseen_sequence_raises_keyerror_naming_the_block(fake_cache):  # Review Focus 1
    enc = ContextEncoder()
    with pytest.raises(KeyError, match="poi"):
        enc.encode("poi", ["MKKKWWWNOTINTABLE"])
    with pytest.raises(KeyError, match="e3"):
        enc.encode("e3", ["MKKKWWWNOTINTABLE"])


def test_missing_cell_line_gets_the_not_found_vector(fake_cache):
    enc = ContextEncoder()
    expected = enc.encode("cell", [ContextEncoder.NOT_FOUND])
    for missing in (None, "", "   ", np.nan):
        assert np.array_equal(enc.encode("cell", [missing]), expected)


def test_unknown_cell_accession_raises(fake_cache):
    with pytest.raises(KeyError, match="cell"):
        ContextEncoder().encode("cell", ["CVCL_9999"])


def test_known_assay_comes_from_the_table(fake_cache):
    enc = ContextEncoder()
    assert enc.encode("assay", ["Western blot"]).shape == (1, 8)
    assert np.array_equal(enc.encode("assay", ["Western blot"]), enc.encode("assay", ["western blot"]))


def test_unseen_assay_falls_back_to_unknown_with_one_warning(fake_cache):
    """A canonical type absent from the table resolves to the table's "unknown" vector."""
    enc = ContextEncoder()
    with pytest.warns(UserWarning, match="a brand new readout"):
        out = enc.encode("assay", ["a brand new readout"])
    assert np.array_equal(out, enc.encode("assay", ["unknown"]))
    # The same miss does not warn again.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        enc.encode("assay", ["a brand new readout"])


def test_register_sequence_projects_a_full_embedding(fake_cache):
    enc = ContextEncoder()
    rng = np.random.default_rng(0)
    emb = rng.normal(size=640).astype(np.float32)
    enc.register_sequence("poi", "MNEWSEQ", emb)
    out = enc.encode("poi", ["MNEWSEQ"])
    side = np.load(enc.cache_dir / enc.files("poi")["pca_model"])
    assert np.allclose(out[0], (emb - side["mean_"]) @ side["components_"].T, rtol=1e-5, atol=1e-5)


def test_combined_space_is_selectable(fake_cache):
    enc = ContextEncoder(protein_space="combined")
    assert enc.dim("poi") == 86 and enc.dim("e3") == 86
    assert enc.encode("e3", [SEQS["e3"][0]]).shape == (1, 86)


def test_encode_context_returns_every_context_block(fake_cache):
    enc = ContextEncoder()
    row = enc.encode_context({"poi_seq": SEQS["poi"][0], "e3_seq": SEQS["e3"][0],
                              "cell_id": CELLS[0], "assay": "western blot", "assay_time": 24.0})
    assert set(row) == {"poi", "e3", "cell", "assay", "assay_time"}
    assert row["poi"].shape == (1, 78) and row["assay_time"].shape == (1, 1)


def test_missing_assay_time_is_the_24h_default_not_an_error(fake_cache):
    row = ContextEncoder().encode_context({"poi_seq": SEQS["poi"][0], "e3_seq": SEQS["e3"][0],
                                           "cell_id": CELLS[0], "assay": "htrf"})
    assert (row["assay_time"] == 24.0).all()


def test_tables_are_loaded_once(fake_cache):
    enc = ContextEncoder()
    enc.encode("poi", [SEQS["poi"][0]])
    before = enc._tables["poi"]
    enc.encode("poi", [SEQS["poi"][1]])
    assert enc._tables["poi"] is before


def test_pandas_na_counts_as_missing(fake_cache):
    """Records from a nullable-dtype frame carry pd.NA, which is a singleton, not a NaN."""
    import pandas as pd

    enc = ContextEncoder()
    row = enc.encode_context({"poi_seq": SEQS["poi"][0], "e3_seq": SEQS["e3"][0],
                              "cell_id": pd.NA, "assay": pd.NA, "assay_time": pd.NA})
    assert (row["assay_time"] == 24.0).all()
    assert np.array_equal(row["cell"], enc.encode("cell", [ContextEncoder.NOT_FOUND]))
    assert np.array_equal(row["assay"], enc.encode("assay", ["unknown"]))


def test_dims_are_discovered_from_the_cached_tables(fake_cache):
    from tackai.fusion.context import CONTEXT_BLOCKS
    per_block = ContextEncoder().dims
    assert list(per_block) == list(CONTEXT_BLOCKS)
    assert per_block == {"e3": 24, "cell": 56, "poi": 78, "assay": 8, "assay_time": 1}
    combined = ContextEncoder(protein_space="combined").dims
    assert combined["e3"] == combined["poi"] == 86 and combined["cell"] == 56
