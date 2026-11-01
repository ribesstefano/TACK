"""Research tools must never materialize reserved-partition labels."""
import json
import sys

import numpy as np
import pandas as pd
import pytest

from scripts.surrogate_research import audit_research_labels as audit
from scripts.surrogate_research import benchmark_graph_surrogate as graph
from scripts.surrogate_research import benchmark_neural_surrogates as neural
from tackai.fusion.research import read_npz_rows, sha256


@pytest.fixture
def snapshot(tmp_path):
    roles = np.array(["search", "confirmation", "search", "stacking", "calibration"])
    np.save(tmp_path / "partition.npy", roles)
    np.save(tmp_path / "source_partition.npy", np.array(["train", "not_search", "eval", "not_search", "not_search"]))
    np.save(tmp_path / "groups.npy", np.arange(len(roles)))
    dims = {key: 1 for key in ("fingerprint", "descriptors", "e3", "cell", "poi", "assay", "assay_time")}
    np.save(tmp_path / "baseline.npy", np.zeros((len(roles), len(dims)), dtype="float32"))
    data = {}
    for policy in ("compat", "exact", "censored"):
        for task in ("dmax", "pdc50", "activity"):
            data[f"{policy}_{task}"] = np.arange(len(roles), dtype=float)
        for task in ("dmax", "pdc50"):
            data[f"{policy}_{task}_bounds"] = np.arange(len(roles) * 2, dtype=float).reshape(-1, 2)
    np.savez(tmp_path / "labels.npz", **data)
    pd.DataFrame({"canonical_smiles": ["C"] * len(roles), "partition": roles,
                  "source_partition": ["train", "not_search", "eval", "not_search", "not_search"],
                  "e3_raw": ["VHL"] * len(roles), "source": ["a"] * len(roles),
                  "publication": ["p"] * len(roles), "Dmax_h": [24] * len(roles),
                  "DC50_h": [24] * len(roles), "Dmax_operator": [""] * len(roles),
                  "DC50_operator": [""] * len(roles), "dmax_raw_pct": [80] * len(roles),
                  "dc50_nM": [100] * len(roles)}).to_csv(tmp_path / "rows.csv", index=False)
    files = ("groups.npy", "partition.npy", "source_partition.npy", "baseline.npy", "labels.npz", "rows.csv")
    (tmp_path / "manifest.json").write_text(json.dumps({"artifact_hashes": {f: sha256(tmp_path / f) for f in files},
                                                       "features": {"baseline": {"dims": dims}}}))
    return tmp_path, roles


def test_reader_preserves_two_dimensional_rows_order_and_duplicates(tmp_path):
    x = np.array([[1., 2.], [3., 4.], [5., 6.]])
    np.savez(tmp_path / "labels.npz", bounds=x, point=x[:, 0])
    np.testing.assert_array_equal(read_npz_rows(tmp_path / "labels.npz", "bounds", [2, 0, 2]), x[[2, 0, 2]])
    np.testing.assert_array_equal(read_npz_rows(tmp_path / "labels.npz", "point", [2, 0]), x[[2, 0], 0])
    assert read_npz_rows(tmp_path / "labels.npz", "bounds", []).shape == (0, 2)
    with pytest.raises(ValueError, match="outside"):
        read_npz_rows(tmp_path / "labels.npz", "bounds", [3])
    np.savez_compressed(tmp_path / "compressed.npz", bounds=x)
    with pytest.raises(ValueError, match="uncompressed"):
        read_npz_rows(tmp_path / "compressed.npz", "bounds", [0])
    np.savez(tmp_path / "fortran.npz", bounds=np.asfortranarray(x))
    with pytest.raises(ValueError, match="C-order"):
        read_npz_rows(tmp_path / "fortran.npz", "bounds", [0])


def test_neural_and_graph_only_request_search_rows(snapshot, monkeypatch):
    path, roles = snapshot
    calls = []
    def spy(file, key, indices):
        calls.append((key, np.asarray(indices).copy()))
        assert np.array_equal(indices, [0, 2])
        return read_npz_rows(file, key, indices)
    monkeypatch.setattr(neural, "read_npz_rows", spy)
    monkeypatch.setattr(graph, "read_npz_rows", spy)
    y, yeval, bounds = neural._load_search_targets(path, roles, "censored")
    assert np.isnan(y[roles != "search"]).all()
    assert np.isnan(yeval[roles != "search"]).all()
    assert np.isnan(bounds[roles != "search"]).all()
    np.testing.assert_array_equal(bounds[[0, 2], 0], [[0, 1], [4, 5]])
    _, _, targets, _ = graph._load_snapshot(path, "baseline", "exact")
    assert np.isnan(targets[roles != "search"]).all()
    assert len(calls) == 11


def test_label_audit_only_requests_search_rows(snapshot, tmp_path, monkeypatch):
    path, _ = snapshot
    calls = []
    def spy(file, key, indices):
        calls.append(key)
        assert np.array_equal(indices, [0, 2])
        return read_npz_rows(file, key, indices)
    monkeypatch.setattr(audit, "read_npz_rows", spy)
    monkeypatch.setattr(sys, "argv", ["audit", "--snapshot", str(path), "--output", str(tmp_path / "audit.json")])
    audit.main()
    assert len(calls) == 9
    assert json.loads((tmp_path / "audit.json").read_text())["scope"].startswith("search rows")
