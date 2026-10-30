"""Scientific boundaries for the frozen surrogate experiment protocol."""
import numpy as np
import pandas as pd
import pytest

from tackai.fusion.data import build_table
from tackai.fusion.labels import endpoint_bounds, research_targets
from tackai.fusion.research import assign_partitions, identity_groups, write_once_json


def test_dc50_bounds_reverse_and_preserve_strict_threshold():
    lo, hi, exact = endpoint_bounds([1000.] * 5, ["<", "<=", ">", ">=", "="], dc50=True)
    assert lo[0] > 6 and lo[1] == 6
    assert hi[2] < 6 and hi[3] == 6
    assert lo[4] == hi[4] == 6
    assert exact.tolist() == [False, False, False, False, True]


def test_unknown_approximate_and_unphysical_not_exact():
    lo, hi, exact = endpoint_bounds([80, 80, 120, -1], ["~", "mystery", "", ""])
    assert np.isnan(lo).all() and np.isnan(hi).all()
    assert not exact.any()


def test_activity_bounds_and_times(tiny_csv):
    table = build_table([tiny_csv]).iloc[:5].copy()
    table["dmax_raw_pct"] = [80., 80., 90., 90., 50.]
    table["Dmax_operator"] = [">", ">=", "", "", ""]
    table["dc50_nM"] = [100., 100., 100., 100., np.nan]
    table["DC50_operator"] = ""
    table["Dmax_h"] = 24.
    table["DC50_h"] = [24., 24., 48., np.nan, np.nan]
    labels, bounds = research_targets(table, "censored")
    assert labels["activity"][0] == 1
    assert np.isnan(labels["activity"][1:4]).all()
    assert labels["activity"][4] == 0
    assert np.isnan(labels["dmax"][:2]).all()
    assert bounds["dmax"][0, 0] > .8


def test_metadata_survives_without_changing_compatibility(tiny_csv):
    table = build_table([tiny_csv])
    assert table.source_row.tolist() == list(range(len(table)))
    assert "DC50_operator" in table and "DOI" in table
    np.testing.assert_allclose(table.dmax_raw_pct, table.dmax_pct, equal_nan=True)


def test_identity_transitivity_and_missing_keys():
    groups = identity_groups([1, 2, 3, 4, 5], ["A", "A", "B", "C", "D"],
                             ["", "K", "K", "", ""])
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[4] and groups[3] != groups[0]


def test_shared_partitions_are_group_disjoint_and_deterministic():
    groups = np.repeat(np.arange(100), 3)
    roles = assign_partitions(groups)
    np.testing.assert_array_equal(roles, assign_partitions(groups))
    assert set(roles) == {"search", "stacking", "calibration", "confirmation"}
    for group in np.unique(groups):
        assert len(set(roles[groups == group])) == 1


def test_provenance_reuse_rejects_changes(tmp_path):
    path = tmp_path / "manifest.json"
    write_once_json(path, {"version": 1})
    write_once_json(path, {"version": 1})
    with pytest.raises(ValueError, match="Provenance mismatch"):
        write_once_json(path, {"version": 2})
