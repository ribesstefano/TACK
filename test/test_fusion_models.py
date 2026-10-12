"""GPInteraction behind the fit/predict interface: M4."""
import numpy as np
import pytest

from fusion_fixtures import BLOCK_DIMS, DEFAULT_BLOCKS, block_index
from tackai.fusion.gp import GPInteraction


def synth(n=90, seed=0, binary=False):
    """A design matrix in the real block layout with a learnable signal."""
    rng = np.random.default_rng(seed)
    idx = block_index(BLOCK_DIMS)
    X = np.zeros((n, sum(BLOCK_DIMS.values())), dtype=np.float64)
    X[:, idx["fingerprint"]] = rng.integers(0, 2, (n, 1024))
    X[:, idx["descriptors"]] = rng.normal(size=(n, 217))
    X[:, idx["descriptors"][0]] *= 1e18                      # Ipc-scale column
    for b in ("poi", "e3", "cell", "assay"):
        X[:, idx[b]] = rng.normal(size=(n, BLOCK_DIMS[b]))
    X[:, idx["assay_time"]] = rng.choice([6.0, 24.0], (n, 1))
    signal = (X[:, idx["fingerprint"][:8]].sum(axis=1)
              + 2.0 * X[:, idx["poi"][0]] * X[:, idx["fingerprint"][0]]
              + X[:, idx["cell"][0]])
    groups = rng.integers(0, 12, n)
    if binary:
        return X, (signal > np.median(signal)).astype(float), groups
    return X, signal + 0.05 * rng.normal(size=n), groups


def fast_gp(**kw):
    return GPInteraction(**{"blocks": DEFAULT_BLOCKS, **kw}, n_restarts=1, n_iter=15, max_hyper_points=60)


def calibrated(est, X, y):
    """Attach a Platt map fitted on ``(X, y)``, as FusionEnsemble.calibrate does per member."""
    if not est.native_binary:
        from sklearn.linear_model import LogisticRegression
        score = est._predict_model(est.model_, X)
        est.calibrator_ = LogisticRegression(C=1e4).fit(score[:, None], y.astype(int))
    return est


def test_fit_returns_self_and_predict_has_the_right_shape():
    X, y, g = synth()
    est = fast_gp(task_type="regression", random_state=0)
    assert est.fit(X, y) is est
    pred = est.predict(X)
    assert pred.shape == (len(y),) and np.isfinite(pred).all()


def test_predictions_are_in_original_target_units():
    X, y, g = synth()
    y = y * 100.0 + 500.0
    est = fast_gp(random_state=0).fit(X, y)
    pred = est.predict(X)
    assert abs(pred.mean() - y.mean()) < 0.5 * y.std()


def test_learns_better_than_predicting_the_mean():
    X, y, g = synth(n=120)
    tr, te = np.arange(90), np.arange(90, 120)
    est = fast_gp(random_state=0).fit(X[tr], y[tr])
    pred = est.predict(X[te])
    assert np.mean((pred - y[te]) ** 2) < np.mean((y[tr].mean() - y[te]) ** 2)


def test_gp_reports_std_in_original_units():
    X, y, g = synth()
    est = fast_gp(random_state=0).fit(X, y * 100.0)
    mean, std = est.predict(X, return_std=True)
    assert est.supports_std and std.shape == mean.shape and (std > 0).all()
    far = X.copy()
    far[:, block_index(BLOCK_DIMS)["poi"]] += 40.0
    _, std_far = est.predict(far, return_std=True)
    assert std_far.mean() > std.mean()


def test_binary_task_returns_probabilities():
    from sklearn.metrics import roc_auc_score
    X, y, g = synth(binary=True)
    est = calibrated(fast_gp(task_type="binary", random_state=0).fit(X, y), X, y)
    p = est.predict(X)
    assert ((p >= 0) & (p <= 1)).all()
    assert roc_auc_score(y, p) > 0.7


def test_gp_binary_std_is_a_probability_interval():
    X, y, g = synth(binary=True)
    est = calibrated(fast_gp(task_type="binary", random_state=0).fit(X, y), X, y)
    p, std = est.predict(X, return_std=True)
    assert ((p >= 0) & (p <= 1)).all() and (std >= 0).all() and (std <= 1).all()


def test_no_target_scaling_is_recorded():
    """The GP learns its own mean; nothing rescales y."""
    X, y, g = synth()
    est = fast_gp(random_state=0).fit(X, y)
    assert not hasattr(est, "y_mean_") and not hasattr(est, "y_std_")


def test_predictions_are_in_the_units_y_was_given_in():
    X, y, g = synth()
    est = fast_gp(random_state=0).fit(X, y + 500.0)
    pred = est.predict(X)
    assert abs(pred.mean() - (y.mean() + 500.0)) < 0.5 * y.std()


def test_the_fit_never_sees_the_test_rows():
    X, y, g = synth(n=120)
    tr, te = np.arange(90), np.arange(90, 120)
    est = fast_gp(random_state=0).fit(X[tr], y[tr])
    baseline = est.predict(X[te])
    again = fast_gp(random_state=0).fit(X[tr], y[tr])
    assert np.allclose(again.predict(X[te]), baseline, rtol=1e-8, atol=1e-10)


def test_gp_exposes_its_kernel_report():
    X, y, g = synth()
    est = fast_gp(random_state=0).fit(X, y)
    rep = est.kernel_report_
    assert "prod:mol*poi" in rep["weight"] and rep["noise"] > 0


def test_same_seed_same_predictions():
    X, y, g = synth()
    a = fast_gp(random_state=3).fit(X, y).predict(X)
    b = fast_gp(random_state=3).fit(X, y).predict(X)
    assert np.allclose(a, b, rtol=1e-8, atol=1e-10)


def test_kernel_report_describes_the_fitted_model():
    X, y, g = synth(binary=True)
    est = fast_gp(task_type="binary", random_state=0).fit(X, y)
    assert est.kernel_report_["weight"] == est.model_.kernel_report()["weight"]


def test_an_uncalibrated_binary_gp_refuses_to_predict():
    """A latent score is not a probability, and returning it as one would be silent nonsense."""
    X, y, g = synth(binary=True)
    est = fast_gp(task_type="binary", random_state=0).fit(X, y)
    with pytest.raises(ValueError, match="uncalibrated"):
        est.predict(X)


def test_refitting_clears_a_calibration():
    """A calibrator fitted against the old model must not survive onto a new one."""
    X, y, g = synth(binary=True)
    est = calibrated(fast_gp(task_type="binary", random_state=0).fit(X, y), X, y)
    assert est.calibrator_ is not None
    est.fit(X, y)
    assert est.calibrator_ is None


def test_fit_refuses_the_old_positional_groups_argument():
    """fit(X, y, groups) was the old call; validation is keyword-only so it cannot be misread."""
    X, y, g = synth()
    with pytest.raises(TypeError):
        fast_gp(random_state=0).fit(X, y, g)


def test_report_is_the_single_path_to_reported_units():
    """predict() must agree with report() on the model's own score."""
    X, y, g = synth()
    est = fast_gp(random_state=0).fit(X, y * 100.0)
    score = est._predict_model(est.model_, X)
    value, std = est.report(score)
    assert np.allclose(value, est.predict(X))
    assert std.shape == value.shape and not std.any()


def test_report_pushes_a_gp_interval_through_the_calibrator():
    X, y, g = synth(binary=True)
    est = calibrated(fast_gp(task_type="binary", random_state=0).fit(X, y), X, y)
    score, score_std = est._predict_model(est.model_, X, return_std=True)
    value, std = est.report(score, score_std)
    assert ((value >= 0) & (value <= 1)).all() and (std >= 0).all() and (std <= 1).all()
    assert np.allclose(value, est.predict(X))
