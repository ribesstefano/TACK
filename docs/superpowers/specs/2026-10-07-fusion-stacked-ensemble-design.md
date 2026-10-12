# Stacked ensemble weights + uncertainty for `FusionEnsemble`

Status: approved design, core scope (see §8 for what's deferred).

## 1. Problem and scope

`FusionEnsemble` (`tackai/fusion/ensemble.py`) currently combines members with
either equal or manually-supplied weights (`set_weights`), and reports
uncertainty via the law of total variance over the members' own predictive
variances and their disagreement (`_aggregate`). Neither the weights nor the
variance scale are *fit* to any held-out data — they are picked once, outside
the class, or defaulted to uniform.

This spec adds a learned alternative: weights and per-member uncertainty
scales fit by maximum likelihood on a held-out stacking set, with post-hoc
calibration, as three new methods on `FusionEnsemble`:

- `fit_stacking` — learn weights (and, for regression, per-member variance
  scales) from a held-out set.
- `calibrate_stacking` — calibrate the resulting uncertainty on a second
  held-out set (conformal interval for regression, temperature for
  classification).
- `predict_stacked` — score new rows with the fitted weights and calibrated
  uncertainty.

This is **additive**: `set_weights`, `predict`, `predict_matrix`, `calibrate`
(the existing Platt-scaling calibration) and `_aggregate` are untouched and
remain the default path. The new methods are an alternative a caller opts
into explicitly.

**Members in scope**: `GPInteraction`, `xgboost.XGBRegressor`,
`xgboost.XGBClassifier` — i.e. `FusionEnsemble.members` may now mix GP and
plain XGBoost members. No `XGBoostFusion`/M7 wrapper class is restored;
XGBoost members are bare sklearn-API XGBoost estimators, used as-is.

**Tasks in scope**: both regression (`dmax`, `pdc50`) and binary
classification (`activity`).

This follows [Section references below refer to the pasted implementation
spec "Ensemble stacking with uncertainty" that motivated this design] — the
numbered sections below (§3 "Per-model sigma", §4 "Regression", §5
"Classification", §6 "Calibration") map directly to that spec's §3–§6.

## 2. Member dispatch

The existing prediction path (`predict`, `predict_matrix`, `_member_scores`,
`_aggregate`, `calibrate`, the context-fold fast path) is GP-shaped
throughout: it relies on `GPInteraction`-specific attributes (`.model_`,
`.report()`, `.native_binary`, `.calibrator_`) and the GP-only
`fold_context`/`predict_in_context` fast path. None of that is touched by
this work, and none of it is generalized to XGBoost members — a raw
`XGBRegressor`/`XGBClassifier` cannot be scored through `predict()` or
`predict_matrix()`. It can only be scored through the three new stacking
methods below.

A single private helper handles dispatch, used only by the new code:

```python
def _stacking_predict(self, member, X) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """(mean, sigma) for one member, in its own reported units.

    - GPInteraction: member.predict(X, return_std=True) — already handles
      task_type and calibration internally via .report().
    - xgb.XGBRegressor: member.predict(X); sigma is None here — Option A's
      per-model constant sigma is computed once in fit_stacking from D_fit
      residuals, not per call.
    - xgb.XGBClassifier: member.predict_proba(X)[:, 1]; sigma is always None.

    Raises TypeError naming the member and its type for anything else.
    """
```

## 3. Per-model sigma

Only Option A (constant residual RMSE, refined jointly with the weights) is
implemented, matching the original spec's instruction to implement Option A
as the default and only working option and leave B/C/D unimplemented. B/C/D
are not stubbed inside this new code at all — they simply don't exist yet;
there is nothing in `fit_stacking`'s signature that gestures at them.

- **GP members**: sigma is the GP's own predictive std returned by
  `GPInteraction.predict(X, return_std=True)` — latent variance plus noise
  variance, already combined inside `AdditiveProductGP.predict`. No
  additional sigma construction needed.
- **XGBoost regressor members**: sigma is a single scalar per member — the
  RMSE of that member on `D_fit` — computed once in `fit_stacking` and then
  treated exactly like a GP member's sigma column (constant across rows for
  this member) when building the `S` matrix for the mixture fit. The joint
  MLE's per-member scale `s_i` (§4) still multiplies it, so the initial RMSE
  value only needs to be right up to a constant factor, per the original
  spec's framing.
- **Classification members** (GP or XGBoost): no sigma; only the probability
  is used.

## 4. Regression: mixture MLE

Pure NumPy + `scipy.optimize.minimize(method="L-BFGS-B")` with an
analytically-derived gradient. No torch/JAX dependency is added for this —
the objective is small (2N-1 parameters) and a hand gradient is
straightforward for a Gaussian-mixture log-likelihood; introducing an
autodiff graph here would be more machinery than the problem needs.

Parameterization, objective, and optimization follow the original spec's §4
exactly:

- Unconstrained `theta`, `phi` (length N each) → softmax weights `w_i`,
  `exp` scales `s_i`; `sigma_min = 1e-3 * std(y_fit)`; floored
  `sigma_tilde_ij = max(s_i * sigma_i(x_j), sigma_min)`.
- Objective: penalized mean negative log-likelihood of the Gaussian mixture,
  log-sum-exp over `log w_i + log N(y_j; f_i(x_j), sigma_tilde_ij^2)`, plus
  `lambda * sum_i (w_i - 1/N)^2`.
- 5 restarts (`theta` perturbed `N(0, 0.5^2)`, `phi` initialized per the
  original spec: 0 for a real per-row sigma, log-RMSE for the XGBoost
  constant-sigma case), keep the lowest objective, stop at relative change
  `< 1e-8`.
- `lambda` selected from `{0, 0.01, 0.1, 1, 10}` by 5-fold CV inside
  `D_fit`, scoring unpenalized held-out NLL; refit on all of `D_fit` at the
  selected value.

Stores on `self`: `weights_` (dict, member name → weight), `scales_` (dict,
member name → fitted `s_i`), `lambda_`, `stacking_cv_log_` (the per-lambda CV
scores).

## 5. Classification: log-loss-pooled probabilities

Same softmax parameterization of `w` over `theta` alone (no `phi`/sigma —
classification needs none). The unpenalized loss is convex in `w`, so one
L-BFGS run from uniform weights is enough — no restarts. `lambda` selected
by the same 5-fold CV procedure, scoring held-out log loss instead of NLL.

Stores `weights_` on `self` (same attribute as regression — a given
`FusionEnsemble` is fit for exactly one task, so there is no collision), and
`stacking_cv_log_`.

## 6. Data splits (`fit_stacking`)

```python
def fit_stacking(
    self, X=None, y=None, *, groups=None,
    X_fit=None, y_fit=None, X_cal=None, y_cal=None, X_test=None, y_test=None,
    lambdas=(0.0, 0.01, 0.1, 1.0, 10.0), n_restarts=5, seed=0,
) -> "FusionEnsemble":
```

Two mutually exclusive calling conventions:

- **Auto-split (default)**: pass one `(X, y[, groups])`, already held out
  from every member's own training rows (the caller's responsibility —
  `FusionEnsemble` has no record of what its members were trained on, so
  this is documented, not checked). Split internally 60/20/20 into
  `D_fit`/`D_cal`/`D_test`. When `groups` is given, split with
  `GroupShuffleSplit` (applied twice, 60/40 then 50/50 of the remainder),
  matching the scaffold-grouped convention already used in
  `fusion/data.py::make_splits`. Without `groups`, use `train_test_split`,
  stratified by `y` for the binary task. `D_test` is kept but unused this
  pass (no `score()` method yet — see §8); it is still split off so a
  future reporting method has it available without re-splitting.
- **Explicit split**: pass `X_fit`/`y_fit`/`X_cal`/`y_cal` directly (and
  optionally `X_test`/`y_test`), skipping the internal split entirely.

Raises `ValueError` if neither convention is fully specified, or if both
are (e.g. `X` and `X_fit` both given).

For each member, `_stacking_predict` runs on `D_fit`; XGBoost regressor
members additionally get their Option-A sigma computed from `D_fit`
residuals as described in §3.

`fit_stacking` stores `X_cal_`/`y_cal_` (and `X_test_`/`y_test_`, if
available) on `self` once the split is resolved — from the internal split
or from the caller's explicit `X_cal`/`y_cal` — so `calibrate_stacking` can
run with no arguments in the common case. These are the literal arrays,
not indices, since `fit_stacking` has no single array to index into once
the explicit-split convention is used.

## 7. Calibration and prediction

### `calibrate_stacking`

```python
def calibrate_stacking(self, *, X_cal=None, y_cal=None, alpha=0.1) -> "FusionEnsemble":
```

Must be called after `fit_stacking`. Defaults to the `X_cal_`/`y_cal_`
`fit_stacking` stored (see §6); `X_cal`/`y_cal` here are only needed to
override that stored split with a different one.

- **Regression**: normalized split-conformal quantile `q_hat_` from
  `D_cal`'s `e_j = |y_j - mu(x_j)| / sigma(x_j)`, the
  `ceil((n+1)(1-alpha))`-th smallest; `UserWarning` + `q_hat_ = inf` if that
  index exceeds `n` (documents D_cal is too small for the requested
  `alpha` — no CV+ fallback this pass, see §8). Also fits the closed-form
  variance-scale `c_^2 = mean(e_j^2)` for a calibrated sigma alongside the
  interval.
- **Classification**: one temperature `T` minimizing log loss on `D_cal`'s
  pooled probability; applied (stored as `temperature_`) only if it beats
  the uncalibrated log loss by more than the bootstrap standard error of
  the difference, else `temperature_ = 1.0`.

### `predict_stacked`

```python
def predict_stacked(self, X) -> dict:
```

Raises if called before `fit_stacking`. Dispatches every member via
`_stacking_predict`, combines with `weights_`/`scales_`.

- Regression → `{"mean", "std", "std_noise", "std_disagreement", "lower",
  "upper"}` — `mean`/`std` from §4's mixture formulas (noise + disagreement
  components kept separate as in the original spec), `lower`/`upper` from
  the conformal interval if `calibrate_stacking` ran, else `None`/absent
  with a note in the docstring that the interval needs calibration first.
- Classification → `{"proba", "entropy_total", "entropy_aleatoric",
  "entropy_epistemic"}` — pooled probability (temperature-adjusted if
  calibrated), binary entropy decomposition per the original spec's §5,
  epistemic term clamped at zero.

## 8. Deferred (not part of this pass)

Explicitly out of scope, to keep this review-sized:

- **Variant B** for both tasks (least-squares weights + two-scalar
  calibration for regression; logit stacking for classification).
- **CV+ fallback** for a stacking set too small for a clean three-way split.
- **Section 7's full model-selection/reporting apparatus**: the
  repeated-5-fold Nadeau-Bengio model-selection comparison against
  baselines, the bootstrap-diagnostic refit, and a `score()` method with
  bootstrapped 95% CIs over the full accuracy/probabilistic-quality/
  calibration metric table (RMSE, CRPS, PIT, ECE, reliability diagrams,
  etc.).
- A generic standalone `StackedEnsemble` class usable outside
  `FusionEnsemble`, and the two spec's "thin helper" functions for building
  `preds`/`sigmas` matrices from an arbitrary list of models — superseded
  here by `_stacking_predict`'s direct dispatch inside `FusionEnsemble`.

These are reference for later work, not promises inside this spec's code —
no `NotImplementedError` stubs are added for them.

## 9. Testing

New tests in `test/test_fusion_ensemble.py` (or a new
`test/test_fusion_stacking.py` if the existing file is already large)
covering, at minimum:

- One member equals truth + small noise, others random → its weight
  exceeds 0.95 at `lambda=0`.
- All members identical → output equals the single member; weights sum to
  1; disagreement component is 0.
- Analytic gradient vs. central finite differences on the mixture NLL,
  relative error below 1e-5.
- Binary entropy decomposition on random probabilities: total equals
  aleatoric + epistemic within 1e-10; epistemic non-negative.
- Conformal coverage on simulated exchangeable regression data, repeated;
  mean coverage at `alpha=0.1` at least `0.90 - 2*SE`.
- A member reproduces one label exactly → fit finishes with finite loss
  (confirms the variance floor).
- Same seed, two `fit_stacking` runs → identical `weights_`/`scales_` and
  `predict_stacked` output.
- Mixed GP + XGBoost membership: `fit_stacking` runs end-to-end on a small
  synthetic design matrix with one `GPInteraction` and one
  `XGBRegressor`/`XGBClassifier` member.
- `_stacking_predict` raises `TypeError` for an unrecognized member type.
- `fit_stacking` raises `ValueError` when given both or neither calling
  convention.
