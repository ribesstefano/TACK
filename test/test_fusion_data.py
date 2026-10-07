"""The data handler: targets, scaffold groups, splits, and one encoder for train and inference."""
import numpy as np
import pytest

from fusion_fixtures import BLOCK_DIMS, BLOCK_ORDER, CELLS, SEQS, SMILES, block_index
from tackai.fusion.data import DMAX_THR, PDC50_THR, FusionData, make_targets, scaffold_groups

N_COLS = sum(BLOCK_DIMS.values())


def test_make_targets_scales_dmax_and_logs_dc50():
    t = make_targets([95.0, 60.0, np.nan], [10.0, 1000.0, 100.0])
    assert np.allclose(t["dmax"][:2], [0.95, 0.60])
    assert np.isnan(t["dmax"][2])
    assert np.allclose(t["pdc50"], [8.0, 6.0, 7.0])


def test_make_targets_non_positive_dc50_is_undefined():
    t = make_targets([50.0, 50.0], [0.0, -5.0])
    assert np.isnan(t["pdc50"]).all()


def test_activity_needs_both_or_a_decisive_one():
    t = make_targets([95.0, 95.0, 60.0, np.nan, np.nan],
                     [10.0, 10000.0, np.nan, 10.0, 10000.0])
    assert t["activity"][0] == 1.0        # dmax > .8 and pdc50 > 6
    assert t["activity"][1] == 0.0        # pdc50 = 5 -> inactive
    assert t["activity"][2] == 0.0        # dmax .6 < .8 alone is decisive
    assert np.isnan(t["activity"][3])     # pdc50 = 8 alone cannot decide active
    assert t["activity"][4] == 0.0        # pdc50 = 5 alone is decisive


def test_thresholds_are_the_documented_ones():
    assert (DMAX_THR, PDC50_THR) == (0.80, 6.0)


def test_scaffold_groups_share_a_group_for_one_scaffold():
    g, info = scaffold_groups([SMILES[0], SMILES[0], SMILES[1]])
    assert g[0] == g[1]
    assert info["n_failed_rows"] == 0


def test_acyclic_molecule_gets_its_own_group():
    g, info = scaffold_groups(["CCCC", "CCCCC", SMILES[0]])
    assert g[0] != g[1] and info["n_failed_rows"] == 2


def test_from_csv_builds_the_design_matrix(fake_cache, tiny_csv):
    data = FusionData.from_csv([tiny_csv], cache=False)
    assert data.X.shape == (24, N_COLS)
    assert data.X.dtype == np.float32
    assert len(data.groups) == 24 and len(data.smiles) == 24


def test_blocks_land_in_their_columns(fake_cache, tiny_csv):
    data = FusionData.from_csv([tiny_csv], cache=False)
    idx = block_index(BLOCK_DIMS)
    fp = data.X[:, idx["fingerprint"]]
    assert set(np.unique(fp)) <= {0.0, 1.0}
    assert np.isfinite(data.X[:, idx["cell"]]).all()


def test_task_rows_drops_undefined_targets(fake_cache, tiny_csv):
    data = FusionData.from_csv([tiny_csv], cache=False)
    idx, X, y, g = data.task_rows("dmax")
    assert np.isfinite(y).all() and len(idx) == len(y) == len(X) == len(g)
    assert len(y) < 24     # the fixture has NaN Dmax rows


def test_splits_never_split_a_scaffold_group(fake_cache, tiny_csv):
    data = FusionData.from_csv([tiny_csv], cache=False)
    _, _, y, g = data.task_rows("pdc50")
    splits = data.splits("pdc50", n_repeats=2, n_folds=3)
    assert len(splits) == 2 and len(splits[0]) == 3
    for repeat in splits:
        covered = np.concatenate([te for _, te in repeat])
        assert sorted(covered) == list(range(len(y)))
        for tr, te in repeat:
            assert not set(g[tr]) & set(g[te])
            assert not set(tr) & set(te)


def test_encode_reproduces_the_training_row_for_identical_inputs(fake_cache, tiny_csv):
    """The inference path must agree with the training path, with mol features on the fly."""
    data = FusionData.from_csv([tiny_csv], cache=False)
    row = data.table.iloc[0]
    X_inf = data.encode([{"smiles": row["smiles"], "poi_seq": row["poi_seq"],
                          "e3_seq": row["e3_seq"], "cell_id": row["cell_key"],
                          "assay": row["assay_raw"], "assay_time": row["assay_time"]}])
    assert np.allclose(X_inf[0], data.X[0], rtol=0, atol=0, equal_nan=True)


def test_assemble_broadcasts_one_context_over_many_smiles(fake_cache, tiny_csv):
    data = FusionData.from_csv([tiny_csv], cache=False)
    ctx = data.encode_context({"poi_seq": SEQS["poi"][0], "e3_seq": SEQS["e3"][0],
                               "cell_id": CELLS[0], "assay": "western blot", "assay_time": 24.0})
    X = data.assemble(ctx, SMILES[:4])
    assert X.shape == (4, N_COLS)
    assert np.array_equal(X[:, data.context_columns], np.repeat(ctx, 4, axis=0))
    direct = data.encode([{"smiles": s, "poi_seq": SEQS["poi"][0], "e3_seq": SEQS["e3"][0],
                           "cell_id": CELLS[0], "assay": "western blot", "assay_time": 24.0}
                          for s in SMILES[:4]])
    assert np.allclose(X, direct, equal_nan=True)


def test_encode_accepts_a_dataframe(fake_cache, tiny_csv, tiny_records):
    import pandas as pd
    data = FusionData.from_csv([tiny_csv], cache=False)
    assert np.allclose(data.encode(pd.DataFrame(tiny_records)), data.encode(tiny_records),
                       equal_nan=True)


def test_encode_empty_returns_empty_matrix(fake_cache, tiny_csv):  # Review Focus 5
    data = FusionData.from_csv([tiny_csv], cache=False)
    assert data.encode([]).shape == (0, N_COLS)


def test_context_and_mol_columns_partition_the_matrix(fake_cache, tiny_csv):
    data = FusionData.from_csv([tiny_csv], cache=False)
    both = np.sort(np.concatenate([data.context_columns, data.mol_columns]))
    assert np.array_equal(both, np.arange(N_COLS))


def test_context_blocks_are_cached_between_constructions(fake_cache, tiny_csv):
    first = FusionData.from_csv([tiny_csv], cache=True)
    second = FusionData.from_csv([tiny_csv], cache=True)
    assert np.allclose(first.X, second.X, equal_nan=True)
    assert (fake_cache / "fusion_blocks").is_dir()


def _csv_with_unencodable_rows(tmp_path):
    """A table where one row has an unknown POI, one an unknown cell line, one no ligase."""
    import pandas as pd
    base = {"SMILES": SMILES[0], "Recruiter": "CRBN", "Recruiter_Sequence": SEQS["e3"][0],
            "Degradation_Target_Uniprot": "P00001", "Degradation_Target_Sequence": SEQS["poi"][0],
            "Cell_Line_ID": CELLS[0], "Cell_Line": "HeLa", "Assay": "western blot",
            "DC50": 100.0, "DC50_units": "nM", "DC50_h": 24.0, "Dmax_h": 24.0, "Dmax": 90.0}
    rows = [dict(base) for _ in range(6)]
    rows[1]["Degradation_Target_Sequence"] = "MUNKNOWNSEQUENCE"
    rows[1]["Degradation_Target_Uniprot"] = "P99999"
    rows[2]["Cell_Line_ID"] = "CVCL_9999"
    rows[3]["Recruiter_Sequence"] = None
    rows[4]["Degradation_Target_Sequence"] = None
    rows[4]["Degradation_Target_Uniprot"] = "P88888"
    path = tmp_path / "messy.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_from_csv_drops_rows_whose_context_is_not_in_the_cache(fake_cache, tmp_path):
    """A 4.6% slice of the real dev data cannot be encoded; losing it must not be a crash."""
    data = FusionData.from_csv([_csv_with_unencodable_rows(tmp_path)], cache=False)
    assert len(data.table) == 2 and data.X.shape[0] == 2
    assert data.dropped["total"] == 4
    assert data.dropped["poi"] == 2 and data.dropped["cell"] == 1 and data.dropped["e3"] == 1


def test_from_csv_can_raise_instead_of_dropping(fake_cache, tmp_path):
    """on_missing='raise' names the block it could not encode (whichever comes first)."""
    with pytest.raises(KeyError, match=r"(poi|e3|cell)"):
        FusionData.from_csv([_csv_with_unencodable_rows(tmp_path)], cache=False,
                            on_missing="raise")


def test_a_clean_table_drops_nothing(fake_cache, tiny_csv):
    data = FusionData.from_csv([tiny_csv], cache=False)
    assert data.dropped["total"] == 0


def test_block_cache_distinguishes_the_fingerprint_radius(fake_cache, tiny_csv):
    """Two radii must not share a cache directory, or a radius ablation compares nothing."""
    from tackai.fusion.mol_encoder import MolEncoder

    a = FusionData.from_csv([tiny_csv], featurizer=MolEncoder(radius=16), cache=True)
    b = FusionData.from_csv([tiny_csv], featurizer=MolEncoder(radius=2), cache=True)
    idx = block_index(BLOCK_DIMS)
    assert not np.array_equal(a.X[:, idx["fingerprint"]], b.X[:, idx["fingerprint"]])

    direct = MolEncoder(radius=2).featurize(a.table["smiles"].tolist())[0]
    assert np.array_equal(b.X[:, idx["fingerprint"]], direct)


# -- blocks_indexes: the layout an estimator must be given -----------------------------------------

def test_blocks_indexes_cover_every_column_once_in_block_order(fake_cache, tiny_csv):
    data = FusionData.from_csv([tiny_csv], cache=False)
    assert list(data.blocks_indexes) == BLOCK_ORDER
    flat = np.concatenate([data.blocks_indexes[b] for b in BLOCK_ORDER])
    assert np.array_equal(flat, np.arange(data.n_columns))


def test_blocks_indexes_is_a_copy(fake_cache, tiny_csv):
    data = FusionData.from_csv([tiny_csv], cache=False)
    mine = data.blocks_indexes
    mine["poi"] = np.arange(3)
    mine["e3"][0] = -1
    assert len(data.index["poi"]) == data.dims["poi"] and data.index["e3"][0] >= 0


def test_the_gp_follows_the_layout_of_data_with_non_default_widths(fake_cache, tiny_csv):
    from tackai.fusion.context import ContextEncoder
    from tackai.fusion.gp import GPInteraction
    data = FusionData.from_csv([tiny_csv], encoder=ContextEncoder(protein_space="combined"),
                               cache=False)
    assert data.dims != BLOCK_DIMS and data.n_columns != N_COLS
    _, X, y, _ = data.task_rows("pdc50")
    est = GPInteraction(blocks=data.blocks_indexes, n_restarts=1, n_iter=5,
                        max_hyper_points=40).fit(X, y)
    assert est.dims_ == data.dims
    assert np.isfinite(est.predict(X)).all()


def test_the_layout_is_discovered_not_assumed(fake_cache, tiny_csv):
    """dims = featuriser's molecular widths + the encoder's discovered context widths."""
    from tackai.fusion.context import ContextEncoder
    from tackai.fusion.mol_encoder import MolEncoder
    data = FusionData.from_csv([tiny_csv], cache=False)
    assert data.dims == BLOCK_DIMS                      # the fixture's literals match reality
    assert list(data.dims) == BLOCK_ORDER
    wide = FusionData(featurizer=MolEncoder(fp_size=256),
                      encoder=ContextEncoder(protein_space="combined"))
    assert wide.dims == {**MolEncoder(fp_size=256).dims, **ContextEncoder(
        protein_space="combined").dims}
    assert wide.dims["fingerprint"] == 256 and wide.dims["poi"] == wide.dims["e3"] == 52
    assert wide.n_columns == sum(wide.dims.values())
    assert np.array_equal(np.concatenate(list(wide.blocks_indexes.values())),
                          np.arange(wide.n_columns))


def test_descriptors_argument_reaches_the_featurizer():
    from tackai.fusion.mol_encoder import MolEncoder
    d = FusionData(descriptors=["MolWt"])
    assert d.featurizer.descriptors == ["MolWt"] and d.dims["descriptors"] == 1
    assert FusionData(descriptors=None).dims["descriptors"] == 0
    with pytest.raises(ValueError, match="not both"):
        FusionData(featurizer=MolEncoder(), descriptors=["MolWt"])
