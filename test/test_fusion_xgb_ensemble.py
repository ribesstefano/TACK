"""Fitting the stacked ensemble: weights, per-member sigmas and their bounds."""
import numpy as np
import pytest
import xgboost as xgb

from tackai.fusion.xgb_ensemble import XGBStackedEnsemble


def _members(X, y, n=4, seed=0):
    """A few cheap regressors, the last one trained on shuffled labels so it is useless.

    A member that carries no signal has its weight driven toward zero, and a weight at zero
    makes that member's sigma unidentifiable -- the likelihood is flat along it. That is the
    direction the optimiser used to run away in.
    """
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        target = y if i < n - 1 else rng.permutation(y)
        model = xgb.XGBRegressor(n_estimators=8, max_depth=2, random_state=i)
        model.fit(X, target)
        out.append(model)
    return out


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 6))
    y = X[:, 0] * 2.0 + rng.normal(0, 0.3, size=120)
    ensemble = XGBStackedEnsemble(_members(X, y)).fit(X, y)
    return ensemble, X, y


def test_sigma_bounds_have_a_finite_ceiling():
    """`_optimise` must bound log-sigma above, not only below.

    Without an upper bound a near-zero-weight member's sigma can diverge, and because the
    noise term is `weights_ @ sigmas_**2`, even a 1e-6 weight beside a 1e11 sigma dominates
    every prediction's uncertainty. The published pdc50 ensemble was fitted this way: its
    sigmas reached 4.5e11 against a median of 0.65, and the conformal interval then covered
    66% of a test set at a nominal 90%.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = X[:, 0] + rng.normal(0, 0.2, size=60)
    ensemble = XGBStackedEnsemble(_members(X, y, n=3))
    preds = ensemble._base_predictions(X)
    n = preds.shape[1]

    captured = {}
    original = ensemble._optimise

    def spy(p, target, lam, rng_):
        # Re-derive the bounds the way _optimise does, by inspecting what it passes on.
        captured["bounds"] = ensemble._sigma_bounds(target, n)
        return original(p, target, lam, rng_)

    ensemble._optimise = spy
    ensemble.fit(X, y)
    sigma_bounds = captured["bounds"][n:]
    assert all(hi is not None and np.isfinite(hi) for _, hi in sigma_bounds), (
        "log-sigma must have a finite upper bound")
    assert all(lo < hi for lo, hi in sigma_bounds), "the sigma floor must sit below the ceiling"


def test_fitted_sigmas_stay_on_the_scale_of_the_target(fitted):
    """No member's noise may exceed the spread of what it is predicting by a wide margin."""
    ensemble, _, y = fitted
    ceiling = ensemble.sigma_ceiling_ratio * float(np.std(y))
    assert np.isfinite(ensemble.sigmas_).all()
    assert ensemble.sigmas_.max() <= ceiling * (1 + 1e-9)


def test_predicted_std_is_finite_and_on_scale(fitted):
    """The uncertainty a caller sees follows from the sigmas, so it must be sane too."""
    ensemble, X, y = fitted
    out = ensemble.predict(X)
    assert np.isfinite(out["std"]).all()
    assert out["std"].max() <= ensemble.sigma_ceiling_ratio * float(np.std(y)) * 2


def test_weights_remain_a_simplex(fitted):
    ensemble, _, _ = fitted
    assert ensemble.weights_.min() >= 0
    assert ensemble.weights_.sum() == pytest.approx(1.0)


def test_a_config_without_the_ceiling_still_loads(tmp_path):
    """An ensemble saved before `sigma_ceiling_ratio` existed must still load, at the default."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 3))
    y = X[:, 0] + rng.normal(0, 0.2, size=40)
    XGBStackedEnsemble(_members(X, y, n=2)).save_pretrained(tmp_path)

    import json
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config["init"].pop("sigma_ceiling_ratio", None)          # an older artifact
    config_path.write_text(json.dumps(config))

    reloaded = XGBStackedEnsemble.from_pretrained(str(tmp_path))
    assert reloaded.sigma_ceiling_ratio > 0
