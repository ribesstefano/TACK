"""Opt-in checks against the real TACKAI_CACHE: filenames, keys and dimensions.

These are the only tests that touch the downloaded cache, so a renamed npz or a changed PCA
width is caught by the suite rather than at runtime. Run with ``pytest -m requires_cache``.
"""
from pathlib import Path

import numpy as np
import pytest
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

from tackai.data.utils import get_cache_dir
from tackai.fusion.context import CONTEXT_FILES, ContextEncoder

pytestmark = pytest.mark.requires_cache

CACHE = Path(get_cache_dir())
DEV_FILES = [Path("data/yaochen/AutoTPDplus-PROTAC-training.csv"),
             Path("data/yaochen/TACKv2.csv")]

MISSING_TABLES = [spec["table"] for spec in CONTEXT_FILES.values()
                  if not (CACHE / spec["table"]).exists()]
needs_cache = pytest.mark.skipif(bool(MISSING_TABLES),
                                 reason=f"TACKAI_CACHE lacks {MISSING_TABLES}")
needs_csvs = pytest.mark.skipif(not all(f.exists() for f in DEV_FILES),
                                reason="development CSVs are not present")


@needs_cache
def test_real_tables_have_the_documented_dimensions():
    enc = ContextEncoder()
    assert enc.dim("cell") == 47
    assert enc.dim("poi") == 51
    assert enc.dim("e3") == 7
    assert enc.dim("assay") == 8
    assert ContextEncoder(protein_space="combined").dim("poi") == 52


@needs_cache
def test_real_cache_has_the_unknown_cell_line_key():
    assert ContextEncoder().encode("cell", [None]).shape == (1, 47)


@needs_cache
def test_real_assay_table_covers_the_common_readouts():
    enc = ContextEncoder()
    for raw in ["Western blot", "HiBiT", "HTRF", "simple western", None]:
        assert enc.encode("assay", [raw]).shape == (1, 8)


@needs_cache
@needs_csvs
def test_development_table_encodes_end_to_end():
    """The real dev data builds, and the documented share of rows survives encoding."""
    from tackai.fusion.data import FusionData

    data = FusionData.from_csv(DEV_FILES, verbose=True)
    assert data.X.shape[1] == 1355
    assert data.dropped["read"] == 8432
    assert data.dropped["total"] / data.dropped["read"] < 0.06    # ~4.6% on this snapshot
    assert len(data.table) == data.dropped["read"] - data.dropped["total"]

    # Every embedding block is complete; assay_time is the one block allowed to be missing
    # (not every paper reports a timepoint), and the preprocessor imputes it inside each fold.
    for block in ("e3", "cell", "poi", "assay"):
        assert np.isfinite(data.X[:, data.index[block]]).all(), f"{block} has missing values"
    assert np.isnan(data.X[:, data.index["assay_time"]]).any()

    for task, floor in [("dmax", 4000), ("pdc50", 6500), ("activity", 4000)]:
        idx, X, y, groups = data.task_rows(task)
        assert len(y) > floor and np.isfinite(y).all()
        splits = data.splits(task, n_repeats=1, n_folds=5)
        for train, test in splits[0]:
            assert not set(groups[train]) & set(groups[test])
