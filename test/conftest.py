"""Shared fixtures: a synthetic TACKAI_CACHE with the real filenames, and small tables."""
import os

# macOS: torch and xgboost each bundle their own libomp and segfault when both run
# multi-threaded in one process. This must be set before either is imported.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import pytest

from fusion_fixtures import ASSAYS, CELLS, SEQS, SMILES, write_context_source, write_part_manifests


@pytest.fixture
def fake_cache(tmp_path, monkeypatch):
    """A TACKAI_CACHE holding the five npz tables under their real names, plus the manifest
    that names them — the shape ``FusionData.from_pretrained`` installs, and the shape a bare
    ``ContextEncoder`` needs, since the filenames are read from the manifest rather than
    hardcoded — and the part manifests the generator scripts leave behind. Returns the Path."""
    cache = tmp_path / "cache"
    manifest = write_context_source(cache, {"cell": 56, "poi": 78, "e3": 24, "combined": 86,
                                            "assay": 8})
    write_part_manifests(cache, manifest)
    monkeypatch.setenv("TACKAI_CACHE", str(cache))
    return cache


@pytest.fixture
def tiny_records():
    """Eight inference-shaped records covering repeated contexts and molecules."""
    return [
        {"smiles": SMILES[i], "poi_seq": SEQS["poi"][i % 4], "e3_seq": SEQS["e3"][i % 2],
         "cell_id": CELLS[i % 4], "assay": ASSAYS[i % 6], "assay_time": 24.0 + i}
        for i in range(8)
    ]


@pytest.fixture
def tiny_csv(tmp_path):
    """A development-shaped CSV with the columns the table builder reads."""
    rows = []
    for i in range(24):
        rows.append({
            "SMILES": SMILES[i % len(SMILES)],
            "Recruiter": ["CRBN", "VHL"][i % 2],
            "Recruiter_Sequence": SEQS["e3"][i % 2],
            "Degradation_Target_Uniprot": f"P{i % 4:05d}",
            "Degradation_Target_Sequence": SEQS["poi"][i % 4],
            "Cell_Line_ID": CELLS[i % 4],
            "Cell_Line": "HeLa",
            "Assay": ASSAYS[i % 6],
            "DC50": [10.0, 100.0, 1000.0, np.nan][i % 4],
            "DC50_units": "nM",
            "DC50_h": 24.0,
            "Dmax_h": 24.0,
            "Dmax": [95.0, 60.0, np.nan, 80.0][i % 4],
        })
    path = tmp_path / "dev.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
