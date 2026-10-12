"""SMOGN resampling for imbalanced regression: relevance, bin balancing, synthetic cases."""
import time
import warnings

import numpy as np
import pytest

from tackai.fusion.smogn import relevance, smogn


def skewed(n=300, d=6, seed=0):
    """A target with a long right tail (the rare, relevant cases) and features that follow it."""
    rng = np.random.default_rng(seed)
    y = rng.exponential(1.0, n)
    X = np.column_stack([y * rng.uniform(0.5, 1.5) + rng.normal(0, 0.3, n) for _ in range(d)])
    X[:, 0] = np.arange(n)                      # unique id: lets a test recognise original rows
    return X.astype(np.float32), y


# ------------------------------------------------------------------ relevance

def test_relevance_is_one_beyond_the_whisker_and_zero_at_the_median():
    _, y = skewed()
    phi = relevance(y, xtrm_type="high")
    q1, med, q3 = np.quantile(y, [0.25, 0.5, 0.75])
    outliers = y > q3 + 1.5 * (q3 - q1)
    assert outliers.any() and np.all(phi[outliers] == 1.0)
    assert phi[np.argmin(np.abs(y - med))] == pytest.approx(0.0, abs=1e-3)
    assert phi.min() >= 0.0 and phi.max() <= 1.0


def test_relevance_rises_monotonically_from_the_median_to_the_whisker():
    _, y = skewed()
    order = np.argsort(y)
    above = y[order] >= np.median(y)
    phi = relevance(y, xtrm_type="high")[order][above]
    assert np.all(np.diff(phi) >= -1e-12)


def test_relevance_low_only_ignores_the_high_tail():
    _, y = skewed()
    assert np.all(relevance(y, xtrm_type="low") == 0.0)      # exponential has no low outliers


def test_relevance_without_any_extreme_is_an_error():
    with pytest.raises(ValueError, match="relevan"):
        smogn(*skewed(), rel_xtrm_type="low")


# ------------------------------------------------------------------ resampling

def test_balance_equalises_rare_and_normal_bins():
    X, y = skewed()
    rare = relevance(y) >= 0.5
    Xr, yr = smogn(X, y, rel_thres=0.5, pert=1e-9, random_state=0)
    b = round(len(y) / 2)                                    # two bins: normal, rare
    assert rare.sum() < b                                    # the premise: the rare bin is small
    rare_after = int((yr >= y[rare].min() - 1e-6).sum())
    assert abs(rare_after - b) <= 2                          # ... and SMOGN grew it to the target size
    assert abs(len(yr) - 2 * b) <= 4


def test_undersampling_keeps_distinct_original_rows():
    """The normal bin is only thinned: a random subset of its rows, each at most once."""
    X, y = skewed()
    thr = y[relevance(y) >= 0.5].min()
    Xr, yr = smogn(X, y, pert=1e-9, random_state=0)
    normal_ids = X[y < thr, 0]
    kept = Xr[yr < thr - 1e-6, 0]
    assert len(kept) == len(np.unique(kept))                 # nothing duplicated
    assert np.isin(kept, normal_ids).all()                   # nothing invented
    assert 0 < len(kept) < len(normal_ids)                   # ... and the bin was thinned


def test_synthetic_cases_stay_inside_the_data_range():
    X, y = skewed()
    Xr, yr = smogn(X, y, random_state=1)
    assert yr.min() >= y.min() - 1e-6 and yr.max() <= y.max() + 1e-6
    assert np.all(Xr >= X.min(axis=0) - 1e-5) and np.all(Xr <= X.max(axis=0) + 1e-5)
    assert np.isfinite(Xr).all() and np.isfinite(yr).all()


def test_interpolation_keeps_features_and_target_consistent():
    """On an exactly linear relation, every interpolated (and barely perturbed) case stays on it."""
    rng = np.random.default_rng(3)
    t = np.concatenate([rng.uniform(0, 1, 200), rng.uniform(5, 6, 20)])    # 20 rare, far-out cases
    X = np.column_stack([np.arange(len(t)), t, 3 * t]).astype(np.float64)
    y = 2 * t + 1
    Xr, yr = smogn(X, y, k=3, pert=1e-9, random_state=0)
    new = ~np.isin(Xr[:, 0], X[:, 0])                        # the id column is interpolated too
    assert new.sum() > 20                                    # many synthetic cases were made
    assert np.allclose(yr, 2 * Xr[:, 1] + 1, atol=1e-6)
    assert np.allclose(Xr[:, 2], 3 * Xr[:, 1], atol=1e-6)


def test_synthetic_cases_are_not_just_copies():
    X, y = skewed()
    Xr, _ = smogn(X, y, random_state=0)
    assert len(np.unique(Xr[:, 1:], axis=0)) > len(np.unique(X[:, 1:], axis=0)) * 0.9


def test_same_seed_same_output_different_seed_differs():
    X, y = skewed()
    a, b = smogn(X, y, random_state=7), smogn(X, y, random_state=7)
    c = smogn(X, y, random_state=8)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert a[0].shape != c[0].shape or not np.array_equal(a[0], c[0])


def test_inputs_are_not_modified_and_dtype_is_kept():
    X, y = skewed()
    X0, y0 = X.copy(), y.copy()
    Xr, yr = smogn(X, y, random_state=0)
    assert np.array_equal(X, X0) and np.array_equal(y, y0)
    assert Xr.dtype == X.dtype and yr.dtype == y.dtype


def test_extreme_method_gives_the_rare_bin_more_than_balance():
    X, y = skewed()
    thr = y[relevance(y) >= 0.5].min() - 1e-6
    n_bal = (smogn(X, y, samp_method="balance", random_state=0)[1] >= thr).sum()
    n_ext = (smogn(X, y, samp_method="extreme", random_state=0)[1] >= thr).sum()
    assert n_ext > n_bal


def test_huge_float32_values_do_not_overflow():
    """Descriptor columns reach 1e20, whose float32 square overflows."""
    X, y = skewed()
    X[:, 1] = X[:, 1] * 1e19
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Xr, yr = smogn(X, y, random_state=0)
    assert np.isfinite(Xr).all() and Xr[:, 1].max() <= X[:, 1].max()


def test_constant_features_and_ties_are_handled():
    X, y = skewed()
    X = np.column_stack([X, np.ones(len(y), dtype=np.float32)])          # a constant column
    y = np.round(y, 1)                                                   # many tied targets
    Xr, yr = smogn(X, y, random_state=0)
    assert np.all(Xr[:, -1] == 1.0) and np.isfinite(Xr).all()


@pytest.mark.parametrize("kwargs, message", [
    (dict(k=0), "k"),
    (dict(pert=0.0), "pert"),
    (dict(pert=1.5), "pert"),
    (dict(rel_thres=0.0), "rel_thres"),
    (dict(samp_method="nope"), "samp_method"),
    (dict(rel_xtrm_type="nope"), "rel_xtrm_type"),
    (dict(rel_coef=0.0), "rel_coef"),
])
def test_bad_arguments_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        smogn(*skewed(), **kwargs)


def test_bad_inputs_are_rejected():
    X, y = skewed()
    with pytest.raises(ValueError, match="rows"):
        smogn(X, y[:-1])
    Xn, yn = X.copy(), y.copy()
    Xn[3, 2] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        smogn(Xn, y)
    yn[3] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        smogn(X, yn)


# ------------------------------------------------------------------ speed

def test_wide_matrix_is_fast():
    """A subset of the real design matrix's shape (many columns, a few thousand rows) in seconds."""
    rng = np.random.default_rng(0)
    n, d = 1200, 1400
    y = rng.exponential(1.0, n)
    X = (rng.random((n, d)) < 0.05).astype(np.float32)
    X[:, :50] += y[:, None].astype(np.float32)
    t = time.perf_counter()
    Xr, yr = smogn(X, y, random_state=0)
    assert time.perf_counter() - t < 10.0
    assert Xr.shape[1] == d and len(Xr) == len(yr)
