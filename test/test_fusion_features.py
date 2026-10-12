"""On-the-fly molecular features, which must match what the training cache holds."""
import numpy as np
import pytest

from fusion_fixtures import SMILES
from tackai.fusion.mol_encoder import DESCRIPTOR_NAMES, MolEncoder


def test_shapes_and_dtypes():
    fp, desc, ok = MolEncoder().featurize(SMILES)
    assert fp.shape == (len(SMILES), 1024) and fp.dtype == np.float32
    assert desc.shape == (len(SMILES), 217) and desc.dtype == np.float32
    assert ok.all() and len(DESCRIPTOR_NAMES) == 217


def test_fingerprint_is_binary():
    fp, _, _ = MolEncoder().featurize(SMILES)
    assert set(np.unique(fp)) <= {0.0, 1.0}


def test_invalid_smiles_does_not_raise_and_is_flagged():  # Review Focus 2
    fp, desc, ok = MolEncoder().featurize(["not_a_molecule", SMILES[0], ""])
    assert list(ok) == [False, True, False]
    assert np.all(fp[0] == 0) and np.all(desc[0] == 0)


def test_empty_input_returns_empty_arrays():  # Review Focus 5
    fp, desc, ok = MolEncoder().featurize([])
    assert fp.shape == (0, 1024) and desc.shape == (0, 217) and ok.shape == (0,)


def test_duplicates_are_computed_once_and_cached_across_calls():
    f = MolEncoder()
    f.featurize([SMILES[0], SMILES[0], SMILES[1]])
    assert f.stats["computed"] == 2
    f.featurize([SMILES[0], SMILES[1]])
    assert f.stats["computed"] == 2 and f.stats["cache_hits"] >= 2


def test_rows_are_in_input_order_with_repeats():
    fp, _, _ = MolEncoder().featurize([SMILES[1], SMILES[0], SMILES[1]])
    assert np.array_equal(fp[0], fp[2]) and not np.array_equal(fp[0], fp[1])


def test_descriptor_sentinel_replaces_non_finite_and_huge_values():
    _, desc, _ = MolEncoder().featurize(SMILES)
    assert np.isfinite(desc).all()
    assert desc.max() <= 1e20


def test_bit_exact_with_tackai_mol_embedding(tmp_path):
    """The whole point: these features must equal what the training cache holds."""
    from tackai.data.embeddings.mol_embeddings import MolEmbedding
    fp_ref = MolEmbedding(embeddings_type="fingerprint", radius=16, fp_size=1024,
                          cache_dir=str(tmp_path)).transform(SMILES[:3])
    de_ref = MolEmbedding(embeddings_type="rdkit_descriptors",
                          cache_dir=str(tmp_path)).transform(SMILES[:3])
    fp, desc, ok = MolEncoder().featurize(SMILES[:3])
    assert ok.all()
    for i, s in enumerate(SMILES[:3]):
        assert np.array_equal(fp[i], fp_ref[s])
        assert np.allclose(desc[i], de_ref[s], rtol=0, atol=0)


def test_process_pool_gives_the_same_values():
    a, b, _ = MolEncoder(n_workers=1).featurize(SMILES)
    f = MolEncoder(n_workers=2)
    c, d, _ = f.featurize(SMILES)
    f.close()
    assert np.array_equal(a, c) and np.allclose(b, d)


def test_the_featuriser_reports_its_molecular_widths():
    from tackai.fusion.mol_encoder import MOL_BLOCKS, N_DESC, MolEncoder
    dims = MolEncoder(fp_size=256).dims
    assert list(dims) == list(MOL_BLOCKS) and dims == {"fingerprint": 256, "descriptors": N_DESC}
    fp, desc, _ = MolEncoder(fp_size=256).featurize(["CCO"])
    assert (fp.shape[1], desc.shape[1]) == (256, N_DESC)


def test_descriptor_subset_follows_rdkit_order():
    enc = MolEncoder(descriptors=["TPSA", "MolWt"])
    full = MolEncoder()
    assert enc.descriptors == [n for n in full.descriptors if n in {"TPSA", "MolWt"}]
    _, desc, _ = enc.featurize(SMILES[:2])
    _, ref, _ = full.featurize(SMILES[:2])
    idx = [full.descriptors.index(n) for n in enc.descriptors]
    assert desc.shape == (2, 2) and enc.dims["descriptors"] == 2
    assert np.array_equal(desc, ref[:, idx])


def test_no_descriptors_gives_fingerprint_only():
    enc = MolEncoder(descriptors=None)
    fp, desc, ok = enc.featurize(SMILES[:2])
    assert desc.shape == (2, 0) and enc.dims["descriptors"] == 0 and ok.all()
    assert enc.featurize([])[1].shape == (0, 0)


def test_unknown_descriptor_rejected():
    with pytest.raises(ValueError, match="NotADescriptor"):
        MolEncoder(descriptors=["MolWt", "NotADescriptor"])


def test_descriptor_subset_with_workers():
    a = MolEncoder(descriptors=["MolWt", "TPSA"], n_workers=2)
    try:
        _, d2, _ = a.featurize(SMILES)
    finally:
        a.close()
    _, d1, _ = MolEncoder(descriptors=["MolWt", "TPSA"]).featurize(SMILES)
    assert np.array_equal(d1, d2)


def test_descriptor_left_out_of_the_default_is_still_requestable():
    from rdkit.Chem import Descriptors
    omitted = [n for n, _ in Descriptors._descList if n not in DESCRIPTOR_NAMES]
    for name in omitted:                       # empty if the default is the full RDKit list
        assert MolEncoder(descriptors=[name]).descriptors == [name]
