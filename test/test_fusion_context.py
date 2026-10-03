"""Context embeddings read from the cached, already PCA-reduced npz tables."""
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


def test_dims_match_the_cached_tables(fake_cache):
    enc = ContextEncoder()
    assert (enc.dim("poi"), enc.dim("e3"), enc.dim("cell"), enc.dim("assay")) == (51, 7, 47, 8)


def test_sequences_are_looked_up_by_stripped_string(fake_cache):
    enc = ContextEncoder()
    a = enc.encode("poi", [SEQS["poi"][0]])
    b = enc.encode("poi", ["  " + SEQS["poi"][0] + "\n"])
    assert a.shape == (1, 51) and np.array_equal(a, b)


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


def test_unseen_assay_is_embedded_on_the_fly_and_projected(fake_cache, monkeypatch):
    """A canonical type absent from the table uses the sentence-transformer and cached PCA."""
    calls = []

    def fake_encode_texts(texts):
        calls.append(list(texts))
        return np.ones((len(texts), 768), dtype=np.float32)

    enc = ContextEncoder()
    monkeypatch.setattr(enc, "_encode_texts", fake_encode_texts)
    out = enc.encode("assay", ["a brand new readout"])
    assert out.shape == (1, 8) and np.isfinite(out).all()
    assert calls == [["a brand new readout"]]


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
    assert enc.dim("poi") == 52 and enc.dim("e3") == 52
    assert enc.encode("e3", [SEQS["e3"][0]]).shape == (1, 52)


def test_encode_context_returns_every_context_block(fake_cache):
    enc = ContextEncoder()
    row = enc.encode_context({"poi_seq": SEQS["poi"][0], "e3_seq": SEQS["e3"][0],
                              "cell_id": CELLS[0], "assay": "western blot", "assay_time": 24.0})
    assert set(row) == {"poi", "e3", "cell", "assay", "assay_time"}
    assert row["poi"].shape == (1, 51) and row["assay_time"].shape == (1, 1)


def test_missing_assay_time_is_nan_not_an_error(fake_cache):
    row = ContextEncoder().encode_context({"poi_seq": SEQS["poi"][0], "e3_seq": SEQS["e3"][0],
                                           "cell_id": CELLS[0], "assay": "htrf"})
    assert np.isnan(row["assay_time"]).all()


def test_tables_are_loaded_once(fake_cache):
    enc = ContextEncoder()
    enc.encode("poi", [SEQS["poi"][0]])
    before = enc._tables["poi"]
    enc.encode("poi", [SEQS["poi"][1]])
    assert enc._tables["poi"] is before
