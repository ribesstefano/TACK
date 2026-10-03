"""Shared fixtures: a synthetic TACKAI_CACHE with the real filenames, and small tables."""
import os

# macOS: torch and xgboost each bundle their own libomp and segfault when both run
# multi-threaded in one process. This must be set before either is imported.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import pytest

from fusion_fixtures import (ASSAY_FILE, ASSAY_PCA_FILE, ASSAYS, CELL_FILE, CELLS,
                             COMBINED_FILE, E3_FILE, NOT_FOUND, POI_FILE, SEQS, SMILES)


def _write(path, keys, dim, seed):
    rng = np.random.default_rng(seed)
    np.savez(path, **{k: rng.normal(size=dim).astype(np.float32) for k in keys})


def _write_pca_model(path, full_dim, out_dim, seed):
    rng = np.random.default_rng(seed)
    np.savez(path, mean_=rng.normal(size=full_dim).astype(np.float32),
             components_=rng.normal(size=(out_dim, full_dim)).astype(np.float32),
             explained_variance_ratio_=np.full(out_dim, 1.0 / out_dim, dtype=np.float32))


@pytest.fixture
def fake_cache(tmp_path, monkeypatch):
    """A TACKAI_CACHE holding the five npz tables under their real names. Returns the Path."""
    cache = tmp_path / "cache"
    cache.mkdir()
    _write(cache / CELL_FILE, CELLS + [NOT_FOUND], 47, 1)
    _write(cache / POI_FILE, SEQS["poi"], 51, 2)
    _write(cache / COMBINED_FILE, SEQS["poi"] + SEQS["e3"], 52, 3)
    _write(cache / E3_FILE, SEQS["e3"], 7, 4)
    _write(cache / ASSAY_FILE, ASSAYS, 8, 5)
    _write_pca_model(cache / CELL_FILE.replace(".npz", "_model.npz"), 768, 47, 6)
    _write_pca_model(cache / POI_FILE.replace(".npz", "_model.npz"), 640, 51, 7)
    _write_pca_model(cache / E3_FILE.replace(".npz", "_model.npz"), 640, 7, 8)
    _write_pca_model(cache / ASSAY_PCA_FILE, 768, 8, 9)
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
