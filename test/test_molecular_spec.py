import numpy as np
import pytest
from rdkit import rdBase

from tackai.fusion.molecular_spec import (
    CANONICALIZATION_VERSION,
    MolecularFeatureEncoder,
    MolecularFeatureSpec,
)
from tackai.fusion.mol_encoder import DESCRIPTOR_NAMES, MolEncoder


def test_default_spec_encoder_is_bit_exact_with_existing_mol_encoder():
    smiles = ["CCO", "C[C@H](O)Cl", "c1ccccc1"]
    spec_encoder = MolecularFeatureEncoder()
    baseline = MolEncoder()
    a_fp, a_desc, a_ok = spec_encoder.featurize(smiles)
    b_fp, b_desc, b_ok = baseline.featurize(smiles)
    assert spec_encoder.spec.radius == 16
    assert spec_encoder.spec.size == 1024
    assert spec_encoder.spec.count is False
    assert spec_encoder.spec.chirality is True
    assert spec_encoder.spec.descriptor_names == tuple(DESCRIPTOR_NAMES)
    assert spec_encoder.spec.canonicalization_version == CANONICALIZATION_VERSION
    assert spec_encoder.spec.rdkit_version == rdBase.rdkitVersion
    assert np.array_equal(a_ok, b_ok)
    assert np.array_equal(a_fp, b_fp)
    assert np.array_equal(a_desc, b_desc)


@pytest.mark.parametrize(
    "name,radius,size,count",
    [
        ("morgan_r2_2048", 2, 2048, False),
        ("morgan_r2_4096", 2, 4096, False),
        ("morgan_r3_2048", 3, 2048, False),
        ("morgan_r3_4096", 3, 4096, False),
        ("count_r2_2048", 2, 2048, True),
        ("count_r2_4096", 2, 4096, True),
    ],
)
def test_presets_declare_and_emit_requested_fingerprint_shape(name, radius, size, count):
    spec = MolecularFeatureSpec.preset(name)
    fp, desc, ok = MolecularFeatureEncoder(spec, share_ipc=False).featurize(["CCCO"])
    assert (spec.radius, spec.size, spec.count) == (radius, size, count)
    assert fp.shape == (1, size)
    assert desc.shape == (1, len(DESCRIPTOR_NAMES))
    assert ok.tolist() == [True]
    assert np.issubdtype(fp.dtype, np.floating)
    if count:
        assert fp.max() > 1
    else:
        assert set(np.unique(fp)) <= {0.0, 1.0}


def test_count_fingerprint_retains_repeated_motif_counts():
    spec = MolecularFeatureSpec.preset("count_r2_2048")
    fp, _, ok = MolecularFeatureEncoder(spec, share_ipc=False).featurize(["CCCO"])
    assert ok[0]
    assert np.count_nonzero(fp[0] > 1) >= 1


def test_count_encoder_process_workers_match_in_process():
    spec = MolecularFeatureSpec.preset("count_r2_2048")
    smiles = ["CCCO", "CCCOCC", "c1ccccc1", "C[C@H](O)Cl"]
    one = MolecularFeatureEncoder(spec, n_workers=1, share_ipc=False)
    many = MolecularFeatureEncoder(spec, n_workers=2, share_ipc=False)
    try:
        fp_one, desc_one, ok_one = one.featurize(smiles)
        fp_many, desc_many, ok_many = many.featurize(smiles)
    finally:
        many.close()
        one.close()
    assert np.array_equal(fp_one, fp_many)
    assert np.array_equal(desc_one, desc_many)
    assert np.array_equal(ok_one, ok_many)


def test_invalid_input_keeps_alignment_and_returns_zero_features():
    encoder = MolecularFeatureEncoder(MolecularFeatureSpec.preset("count_r2_2048"),
                                      share_ipc=False)
    fp, desc, ok = encoder.featurize(["not a smiles", "CCO", "", "CCO"])
    assert ok.tolist() == [False, True, False, True]
    assert fp.shape == (4, 2048) and desc.shape == (4, len(DESCRIPTOR_NAMES))
    assert np.all(fp[[0, 2]] == 0) and np.all(desc[[0, 2]] == 0)
    assert np.array_equal(fp[1], fp[3])


def test_spec_json_round_trip_reconstructs_equivalent_encoder():
    spec = MolecularFeatureSpec.preset("count_r2_4096")
    restored = MolecularFeatureSpec.from_json(spec.to_json())
    encoder = MolecularFeatureEncoder.from_json(spec.to_json(), share_ipc=False)
    fp_a, desc_a, _ = MolecularFeatureEncoder(spec, share_ipc=False).featurize(["CCCO"])
    fp_b, desc_b, _ = encoder.featurize(["CCCO"])
    assert restored == spec
    assert np.array_equal(fp_a, fp_b)
    assert np.array_equal(desc_a, desc_b)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("rdkit_version", "0.0", "RDKit version mismatch"),
        ("canonicalization_version", "future-v2", "canonicalization version"),
        ("schema_version", 999, "schema version"),
    ],
)
def test_spec_loader_rejects_runtime_or_contract_version_mismatch(field, value, error):
    payload = MolecularFeatureSpec().to_dict()
    payload[field] = value
    with pytest.raises(ValueError, match=error):
        MolecularFeatureSpec.from_dict(payload)


def test_spec_loader_rejects_descriptor_order_mismatch():
    payload = MolecularFeatureSpec().to_dict()
    payload["descriptor_names"] = list(reversed(payload["descriptor_names"]))
    with pytest.raises(ValueError, match="descriptor order"):
        MolecularFeatureSpec.from_dict(payload)


def test_spec_loader_rejects_missing_or_extra_fields():
    payload = MolecularFeatureSpec().to_dict()
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="fields differ"):
        MolecularFeatureSpec.from_dict(payload)
    with pytest.raises(ValueError, match="invalid molecular feature spec JSON"):
        MolecularFeatureSpec.from_json("{")


def test_descriptor_selection_records_rdkit_order_not_requested_order():
    with pytest.raises(ValueError, match="descriptor order"):
        MolecularFeatureSpec(descriptor_names=("TPSA", "MolWt"))
    ordered = tuple(name for name in DESCRIPTOR_NAMES if name in {"TPSA", "MolWt"})
    spec = MolecularFeatureSpec(descriptor_names=ordered)
    assert MolecularFeatureEncoder(spec).spec.descriptor_names == ordered


def test_count_and_bit_specs_use_distinct_fusion_block_caches(fake_cache, tiny_csv):
    from tackai.fusion.data import FusionData

    bit_spec = MolecularFeatureSpec(radius=2, size=2048)
    count_spec = MolecularFeatureSpec(radius=2, size=2048, count=True)
    bit_data = FusionData.from_csv(
        [tiny_csv], featurizer=MolecularFeatureEncoder(bit_spec, share_ipc=False), cache=True)
    count_data = FusionData.from_csv(
        [tiny_csv], featurizer=MolecularFeatureEncoder(count_spec, share_ipc=False), cache=True)
    assert bit_data._cache_dir() != count_data._cache_dir()
    fp = count_data.X[:, count_data.index["fingerprint"]]
    assert np.any(fp > 1)

    # The established MolEncoder default remains the same configuration and output.
    legacy = MolEncoder()
    assert (legacy.radius, legacy.fp_size) == (16, 1024)
    old_fp, old_desc, old_ok = legacy.featurize(["CCO", "C[C@H](O)Cl"])
    new_fp, new_desc, new_ok = MolecularFeatureEncoder().featurize(["CCO", "C[C@H](O)Cl"])
    assert np.array_equal(old_ok, new_ok)
    assert np.array_equal(old_fp, new_fp)
    assert np.array_equal(old_desc, new_desc)
