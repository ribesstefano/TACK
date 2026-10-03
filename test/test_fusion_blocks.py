"""The block layout and the fold-internal preprocessor (no PCA anywhere)."""
import numpy as np
import pytest

from tackai.fusion.blocks import (BLOCK_DIMS, BLOCK_ORDER, DENSE_BLOCKS, MOL_BLOCKS,
                                  SMALL_BLOCKS, BlockPreprocessor, block_index)


def make_X(n=40, seed=0):
    rng = np.random.default_rng(seed)
    idx = block_index(BLOCK_DIMS)
    X = np.zeros((n, sum(BLOCK_DIMS.values())), dtype=np.float64)
    X[:, idx["fingerprint"]] = rng.integers(0, 2, (n, 1024))
    X[:, idx["descriptors"]] = rng.normal(0, 1, (n, 217))
    X[:, idx["descriptors"][0]] = rng.normal(0, 1, n) * 1e18   # Ipc-scale column
    for b in ("poi", "e3", "cell", "assay"):
        X[:, idx[b]] = rng.normal(size=(n, BLOCK_DIMS[b]))
    X[:, idx["assay_time"]] = rng.choice([12.0, 24.0, np.nan], (n, 1))
    return X, idx


def test_block_index_is_contiguous_and_in_block_order():
    idx = block_index(BLOCK_DIMS)
    assert list(idx) == BLOCK_ORDER
    flat = np.concatenate([idx[b] for b in BLOCK_ORDER])
    assert np.array_equal(flat, np.arange(sum(BLOCK_DIMS.values())))
    assert sum(BLOCK_DIMS.values()) == 1355


def test_block_sets_partition_the_layout():
    assert set(DENSE_BLOCKS) | set(SMALL_BLOCKS) == set(BLOCK_ORDER)
    assert not set(DENSE_BLOCKS) & set(SMALL_BLOCKS)
    assert set(MOL_BLOCKS) <= set(DENSE_BLOCKS)


def test_molecule_blocks_are_not_scaled_only_divided_by_sqrt_width():
    X, idx = make_X()
    Z = BlockPreprocessor().fit(X).transform(X)
    assert np.allclose(Z["fingerprint"], X[:, idx["fingerprint"]] / np.sqrt(1024))
    assert np.allclose(Z["descriptors"], X[:, idx["descriptors"]] / np.sqrt(217))


def test_no_pca_dims_are_preserved():
    X, _ = make_X()
    pre = BlockPreprocessor().fit(X)
    assert pre.dims_ == BLOCK_DIMS


def test_context_blocks_are_standardised():
    X, _ = make_X()
    Z = BlockPreprocessor().fit(X).transform(X)
    assert np.allclose(Z["cell"].mean(axis=0), 0, atol=1e-8)
    assert np.allclose(Z["cell"].std(axis=0) * np.sqrt(47), 1, atol=1e-6)


def test_constant_context_column_does_not_produce_inf():  # Review Focus 4
    X, idx = make_X()
    X[:, idx["cell"][3]] = 7.0            # a single cell line -> zero variance
    Z = BlockPreprocessor().fit(X).transform(X)
    assert np.isfinite(Z["cell"]).all()
    assert np.allclose(Z["cell"][:, 3], 0.0)


def test_fitted_on_training_rows_only():
    """Statistics come from the fit rows only: changing the held-out rows cannot move them."""
    X, _ = make_X(n=60)
    tr, te = np.arange(30), np.arange(30, 60)
    pre = BlockPreprocessor().fit(X[tr])
    baseline = pre.transform(X[te])["cell"]

    X_shifted = X.copy()
    X_shifted[te] += 100.0
    pre_same_train = BlockPreprocessor().fit(X_shifted[tr])
    assert np.allclose(pre_same_train.transform(X[te])["cell"], baseline, rtol=1e-10, atol=1e-12)

    assert np.allclose(pre.transform(X)["cell"][:30], pre.transform(X[tr])["cell"])


def test_assay_time_is_median_imputed_and_standardised():
    X, idx = make_X()
    Z = BlockPreprocessor().fit(X).transform(X)
    assert np.isfinite(Z["assay_time"]).all()
    assert abs(float(Z["assay_time"].mean())) < 1e-8


def test_nan_in_dense_block_is_mean_imputed():
    X, idx = make_X()
    X[0, idx["poi"][0]] = np.nan
    Z = BlockPreprocessor().fit(X).transform(X)
    assert np.isfinite(Z["poi"]).all()


def test_transform_blocks_subset_matches_full_transform():
    X, _ = make_X()
    pre = BlockPreprocessor().fit(X)
    full = pre.transform(X)
    part = pre.transform_blocks(X, only=["cell", "poi"])
    assert set(part) == {"cell", "poi"}
    for b in part:
        assert np.allclose(part[b], full[b])


def test_concat_follows_block_order():
    X, _ = make_X()
    pre = BlockPreprocessor().fit(X)
    A = pre.concat(pre.transform(X))
    assert A.shape == (40, 1355)
    assert np.allclose(A[:, :1024], pre.transform(X)["fingerprint"])
