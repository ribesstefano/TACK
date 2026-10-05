"""The block layout and the fold-internal preprocessor (no PCA anywhere)."""
import numpy as np
import pytest

from tackai.fusion.blocks import (BLOCK_DIMS, BLOCK_ORDER, DENSE_BLOCKS, LEGACY_SCALE_BLOCKS,
                                  MOL_BLOCKS, SMALL_BLOCKS, BlockPreprocessor, block_index)


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


def test_no_block_is_standardised_by_default():
    """The context arrives PCA-reduced and the planned tree members ignore scaling."""
    X, idx = make_X()
    Z = BlockPreprocessor(dtype="float64").fit(X).transform(X)
    for block in ("e3", "cell", "poi", "assay"):
        assert np.array_equal(Z[block], X[:, idx[block]] / np.sqrt(BLOCK_DIMS[block])), block


def test_scale_blocks_restores_the_legacy_standardisation():
    X, _ = make_X()
    Z = BlockPreprocessor(scale_blocks=LEGACY_SCALE_BLOCKS).fit(X).transform(X)
    assert np.allclose(Z["cell"].mean(axis=0), 0, atol=1e-6)
    assert np.allclose(Z["cell"].std(axis=0) * np.sqrt(47), 1, atol=1e-4)


def test_constant_context_column_does_not_produce_inf():  # Review Focus 4
    X, idx = make_X()
    X[:, idx["cell"][3]] = 7.0            # a single cell line -> zero variance
    Z = BlockPreprocessor(scale_blocks=LEGACY_SCALE_BLOCKS, dtype="float64").fit(X).transform(X)
    assert np.isfinite(Z["cell"]).all()
    assert np.allclose(Z["cell"][:, 3], 0.0)
    assert np.isfinite(BlockPreprocessor().fit(X).transform(X)["cell"]).all()


def test_fitted_on_training_rows_only():
    """Statistics come from the fit rows only: changing the held-out rows cannot move them."""
    X, _ = make_X(n=60)
    tr, te = np.arange(30), np.arange(30, 60)
    pre = BlockPreprocessor(scale_blocks=LEGACY_SCALE_BLOCKS, dtype="float64").fit(X[tr])
    baseline = pre.transform(X[te])["cell"]

    X_shifted = X.copy()
    X_shifted[te] += 100.0
    pre_same_train = BlockPreprocessor(scale_blocks=LEGACY_SCALE_BLOCKS, dtype="float64").fit(X_shifted[tr])
    assert np.allclose(pre_same_train.transform(X[te])["cell"], baseline, rtol=1e-10, atol=1e-12)

    assert np.allclose(pre.transform(X)["cell"][:30], pre.transform(X[tr])["cell"])


def test_assay_time_is_median_imputed_and_not_standardised():
    X, idx = make_X()
    Z = BlockPreprocessor(dtype="float64").fit(X).transform(X)
    assert np.isfinite(Z["assay_time"]).all()
    observed = X[:, idx["assay_time"]][~np.isnan(X[:, idx["assay_time"]])]
    assert set(np.unique(Z["assay_time"])) <= {12.0, 24.0, float(np.median(observed))}


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


def four_pass_reference(pre, X, only):
    """The transform as it was before the fused pass: imputer, scaler, divide, copy."""
    X = np.asarray(X, dtype=np.float64)
    out = {}
    for b in only:
        imp, sc, width = pre.steps_[b]
        Z = imp.transform(X[:, pre.blocks_[b]])
        if sc is not None:
            Z = sc.transform(Z)
        out[b] = np.ascontiguousarray(Z / width)
    return out


def make_X_with_gaps():
    X, idx = make_X()
    X[3, idx["poi"][2]] = np.nan
    X[5, idx["cell"][0]] = np.nan
    X[7, idx["e3"][1]] = np.nan
    X[9, idx["descriptors"][4]] = np.nan
    return X, idx


@pytest.mark.parametrize("scale_blocks", [None, LEGACY_SCALE_BLOCKS])
def test_fused_transform_is_bit_identical_to_the_four_pass_reference(scale_blocks):
    """Removing the redundant passes must not move a single bit (at float64)."""
    X, _ = make_X_with_gaps()
    pre = BlockPreprocessor(scale_blocks=scale_blocks, dtype="float64").fit(X)
    Z, ref = pre.transform(X), four_pass_reference(pre, X, pre.blocks_)
    for block in ref:
        assert np.array_equal(Z[block], ref[block]), block


def test_float32_output_tracks_the_float64_reference():
    X, _ = make_X_with_gaps()
    pre = BlockPreprocessor().fit(X)
    Z, ref = pre.transform(X), four_pass_reference(pre, X, pre.blocks_)
    for block in ref:
        assert Z[block].dtype == np.float32 and Z[block].flags.c_contiguous, block
        assert np.allclose(Z[block], ref[block], rtol=1e-6, atol=1e-30), block


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_transform_never_writes_into_the_callers_array(dtype):  # Review Focus 4
    X, _ = make_X_with_gaps()
    pre = BlockPreprocessor(scale_blocks=LEGACY_SCALE_BLOCKS, dtype=dtype).fit(X)
    for arr in (X.copy(), X.astype(np.float32)):
        before = arr.copy()
        pre.transform(arr)
        pre.transform_blocks(arr, only=["cell", "fingerprint"])
        assert np.array_equal(arr, before, equal_nan=True)


def test_statistics_are_fitted_in_float64():
    X, idx = make_X()
    col = idx["descriptors"][0]                       # the Ipc-scale column, ~1e18
    X[0, col] = np.nan
    pre = BlockPreprocessor().fit(X.astype(np.float32))
    stats = pre.steps_["descriptors"][0].statistics_
    assert stats.dtype == np.float64
    assert stats[0] == pytest.approx(np.nanmean(X.astype(np.float32).astype(np.float64)[:, col]))


def test_a_preprocessor_pickled_before_dtype_existed_behaves_as_float64():  # Review Focus 1
    X, _ = make_X()
    old = BlockPreprocessor().fit(X)
    del old.dtype                                      # an artifact written by the old code
    assert old.transform(X)["cell"].dtype == np.float64


def test_non_contiguous_blocks_transform_identically():
    X, idx = make_X_with_gaps()
    reversed_layout = {b: idx[b][::-1].copy() for b in idx}
    straight = BlockPreprocessor(dtype="float64").fit(X)
    flipped = BlockPreprocessor(blocks=reversed_layout, dtype="float64").fit(X)
    assert isinstance(straight._columns("cell"), slice)
    assert isinstance(flipped._columns("cell"), np.ndarray)
    Zs, Zf = straight.transform(X), flipped.transform(X)
    for block in Zs:
        assert np.array_equal(Zs[block][:, ::-1], Zf[block]), block


def two_halves(**kw):
    X, _ = make_X(n=60)
    return X, BlockPreprocessor(**kw).fit(X[:30]), BlockPreprocessor(**kw).fit(X[30:])


def test_consensus_averages_the_fitted_statistics():
    _, a, b = two_halves(scale_blocks=LEGACY_SCALE_BLOCKS, dtype="float64")
    merged = BlockPreprocessor.consensus([a, b], only=["cell", "assay_time"])
    for block in ("cell", "assay_time"):
        (ia, sa, _), (ib, sb, _) = a.steps_[block], b.steps_[block]
        imp, sc, _ = merged.steps_[block]
        assert np.allclose(imp.statistics_, (ia.statistics_ + ib.statistics_) / 2)
        assert np.allclose(sc.mean_, (sa.mean_ + sb.mean_) / 2)
        assert np.allclose(sc.scale_, (sa.scale_ + sb.scale_) / 2)
    assert not np.allclose(a.steps_["cell"][1].mean_, merged.steps_["cell"][1].mean_)


def test_consensus_does_not_depend_on_member_order():
    X, a, b = two_halves(scale_blocks=LEGACY_SCALE_BLOCKS, dtype="float64")
    ab = BlockPreprocessor.consensus([a, b]).transform(X)
    ba = BlockPreprocessor.consensus([b, a]).transform(X)
    for block in ab:
        assert np.array_equal(ab[block], ba[block]), block


def test_consensus_of_a_single_preprocessor_is_that_preprocessor():
    X, _ = make_X_with_gaps()
    pre = BlockPreprocessor(scale_blocks=LEGACY_SCALE_BLOCKS, dtype="float64").fit(X)
    Z, Zc = pre.transform(X), BlockPreprocessor.consensus([pre]).transform(X)
    for block in Z:
        assert np.array_equal(Z[block], Zc[block]), block


def test_consensus_rejects_members_that_disagree_on_scaling():  # Review Focus 5
    X, _ = make_X(n=60)
    scaled = BlockPreprocessor(scale_blocks=LEGACY_SCALE_BLOCKS).fit(X)
    plain = BlockPreprocessor().fit(X)
    with pytest.raises(ValueError, match="cell"):
        BlockPreprocessor.consensus([scaled, plain], only=["cell"])


def test_consensus_rejects_members_with_different_columns():  # Review Focus 5
    X, idx = make_X(n=60)
    shuffled = {b: idx[b][::-1].copy() for b in idx}
    with pytest.raises(ValueError, match="poi"):
        BlockPreprocessor.consensus([BlockPreprocessor().fit(X),
                                     BlockPreprocessor(blocks=shuffled).fit(X)], only=["poi"])


def test_transform_signature_ignores_the_imputer_but_not_the_scaler():
    _, a, b = two_halves()
    assert a.steps_["descriptors"][0].statistics_.tolist() != \
        b.steps_["descriptors"][0].statistics_.tolist()
    for block in ("fingerprint", "descriptors"):
        assert a.transform_signature(block) == b.transform_signature(block)
    _, sa, sb = two_halves(scale_blocks=("descriptors",))
    assert sa.transform_signature("descriptors") != sb.transform_signature("descriptors")
