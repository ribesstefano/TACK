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

    # The matrix is complete. Not every paper reports a timepoint; those rows carry the
    # constant default of 24 h, which nothing learned from the data.
    assert np.isfinite(data.X).all()
    unreported = data.table["assay_time"].isna().to_numpy()
    print(f"assay_time unreported in {unreported.sum()} of {len(unreported)} rows")
    assert (data.X[unreported, data.index["assay_time"][0]] == 24.0).all()

    for task, floor in [("dmax", 4000), ("pdc50", 6500), ("activity", 4000)]:
        idx, X, y, groups = data.task_rows(task)
        assert len(y) > floor and np.isfinite(y).all()
        splits = data.splits(task, n_repeats=1, n_folds=5)
        for train, test in splits[0]:
            assert not set(groups[train]) & set(groups[test])


@needs_cache
@needs_csvs
def test_gp_member_fits_the_real_dmax_fold():
    """Regression: this exact fit raised "leading minor of order 1066 is not positive-definite".

    The development table measures the same compound in the same context repeatedly, so the
    kernel of the 1200-row hyper-parameter sample is singular; the fit only survives because
    the likelihood noise has a floor.
    """
    from sklearn.metrics import r2_score

    from tackai.fusion.data import FusionData
    from tackai.fusion.gp import GPInteraction

    data = FusionData.from_csv(DEV_FILES)
    _, X, y, groups = data.task_rows("dmax")
    train, test = data.splits("dmax", n_repeats=1, n_folds=5)[0][0]

    gp = GPInteraction(blocks=data.blocks_indexes, random_state=0).fit(X[train], y[train])
    mean, std = gp.predict(X[test], return_std=True)
    assert np.isfinite(mean).all() and (std > 0).all()
    assert gp.model_.noise_ >= 1e-3
    assert r2_score(y[test], mean) > 0.2        # the published M4 reaches ~.47 over 25 folds
