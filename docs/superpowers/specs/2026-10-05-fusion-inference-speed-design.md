# Fusion ensemble inference: float32 defaults, one shared context, measured speed — design

**Date:** 2026-10-05
**Branch:** `fast-surrogates`
**Status:** awaiting review
**Supersedes parts of:** `2026-10-03-fusion-surrogates-design.md` (block preprocessing table, GP dtype)

## Goal

Make `FusionEnsemble` inference fast and semantically consistent, and prove both with a
notebook that measures before and after:

1. **float32** becomes the default dtype for the fusion models and every operation in them.
2. **One canonical biological context** is enforced across all members, whatever fold or data
   each was fitted on.
3. **The context blocks are no longer standardised** — they arrive PCA-reduced from the cache,
   and the planned XGBoost members would not use a scaler anyway.
4. The redundant work found in the inference path is removed.
5. Inference cost is timed from 1 to 25 members, so the price of the 25-member ensemble you
   expect is on the record rather than inferred from the inherited default of 5.

Accuracy is measured and reported throughout, but it is not a gate: the user has accepted
degradation here and will retrain with hyper-parameter optimisation separately.

## Why

The ensemble's documented purpose is a screening / reinforcement-learning loop: one fixed
context, a stream of new molecules, hundreds of thousands of scores. Three things stand in the
way today.

**The inference path repeats work.** Per `predict(smiles, context=ctx)` with M members:

| # | Location | Redundancy |
|---|---|---|
| R1 | `ensemble.py:429-441` | `mol_row` is a full-width (n×1355) **float64** scratch array with 114 context columns left zero, and `mol_row[valid]` is a fancy-index copy made **inside** the member loop → M copies per batch |
| R2 | `blocks.py:143-157` | `transform_blocks` makes ~4 passes per block per member: fancy-index copy (blocks are contiguous — a slice view would do), `imp.transform`, `/width`, `ascontiguousarray` |
| R3 | `gp.py:493-497` | `predict_in_context` recomputes the **train-side** terms every batch: `Z_train_["descriptors"] / ls` and its row norms, and `(B*B).sum(1)` over the (n_train×1024) fingerprint block. `params_` are frozen after `load_state`, so these are byte-identical every call, for every member |
| R4 | `gp.py` / `blocks.py` | features are float32 but upcast to float64, so the dominant cross-covariance matmul runs at half the achievable throughput |
| R5 | `gp.py:509` | `torch.any(weight != 0)` per block per batch — decidable once in `fold_context` |
| R6 | `data.py:encode` / `ensemble.py:471` | `encode()` featurises and discards `ok`, so the records path calls `featurize()` a second time to recover it |

**Members disagree about the context.** Each member's `BlockPreprocessor` was fitted on its own
fold, so today two members are fed numerically different values for the *same* experimental
context. Measured on `ensembles/fusion_dmax`:

| block | scaler `scale_` max rel. Δ | scaler `mean_` max rel. Δ | imputer stat max abs Δ |
|---|---|---|---|
| e3 | 17.7% | 14.8% | 0.045 |
| cell | 9.5% | 12.8% | 0.016 |
| poi | 12.6% | 20.5% | 1.66 |
| assay | 5.2% | 9.8% | 0.021 |
| assay_time | 7.6% | 17.3% | **6.0 hours** |

That is a reproducibility defect, not a tuning detail: the ensemble's answer depends on which
folds its members happened to draw.

**The scaling it disagrees about should not be there.** `e3`, `cell`, `poi` and `assay` are read
from cached **PCA-reduced** tables (`context.py:38-59`), so they are already centred and ordered
by decreasing variance. Standardising them re-inflates the low-variance components, costs a
fitted scaler per member per block, and is useless to the tree members planned next.

## Decisions

| Decision | Choice | Rationale / rejected alternative |
|---|---|---|
| Default dtype | `float32` throughout `AdditiveProductGP`, including the fit and the Cholesky | User's call. Rejected: float32 inference over float64 storage (keeps 2× memory and adds a per-batch cast) |
| Cholesky failure | Automatic promotion to float64 for the factorisation and solve, recorded on the model | Two recent commits fixed non-positive-definite kernels in *float64*; float32 has ~7 digits, so a retry is required, not optional |
| Preprocessor statistics | **Fitted in float64, applied in float32** | A raw `Ipc` descriptor near 1e18 would lose the imputer mean entirely if the mean were accumulated in float32. Fit happens once; transform happens per batch |
| Jitter ladder | Scales with dtype: float32 starts 1e-5, ceiling 1e-1; float64 keeps 1e-6, ceiling 1e-2 | float32 eps ≈ 1.2e-7, so a 1e-6 jitter is below the noise of a unit-scale kernel |
| Context scaling | `scale_blocks` defaults to `()` — no dense block is standardised | User's call; inputs are pre-PCA'd and the planned XGBoost members ignore scaling |
| `assay_time` scaling | **Kept standardised** | It is the one context block that is *not* PCA-reduced — raw hours, 0-168, feeding a *linear* kernel where absolute magnitude sets the term's weight. Overturnable on review |
| `÷ sqrt(width)` | Kept for every dense block | A deterministic constant, not a fitted statistic; it stops a 1024-wide block dominating a shared-lengthscale kernel, and is monotone per column so trees are unaffected |
| Canonical context | **Consensus over members**: per block, the mean of the fitted imputer statistics and, when present, of the scaler `mean_`/`scale_` | User's call. Independent of any single fold, needs no extra data, and works on the artifacts already on disk. Rejected: member 0's preprocessor (tied to one fold); a full-dataset refit (forces a manifest migration for no measured gain) |
| Scope of the canonical context | All three prediction paths — cached-context, records, and `predict_matrix` | "The same context on all members" has to hold however the batch arrived, or two paths disagree |
| Old artifacts | Still load and score | The consensus is derived *from the members*, so an ensemble fitted with scaling gets a scaling consensus automatically. No manifest version bump needed |
| Member count | Time 1-25 members, by **cloning** the 5 fitted members up to 25 | Inference cost depends on member count, `n_train` and the block widths — not on parameter values — so cloned members give faithful timing without hours of GP fits. Rejected: fitting 25 real members (needed only for an accuracy curve, which is out of scope) |
| Accuracy | **Measured and reported, never a gate** | User, 2026-10-05: degradation is expected and accepted; the ensembles will be retrained later with hyper-parameter optimisation. No refit is delivered here |
| Old behaviour reachable | `FusionEnsemble(shared_context=False)`, `AdditiveProductGP(dtype="float64")`, `BlockPreprocessor(scale_blocks=...)` | Each semantic change must stay measurable against its predecessor in a single notebook run, and the superseded path must stay testable. The new value is the default in every case |

## Mechanism: the canonical context

`FusionEnsemble` gains `context_pre_`, built once at construction from the members:

- **Validation.** Every member must agree on the context blocks' column indices, their widths,
  and whether a scaler is present for each block. A mismatch raises `ValueError` with the
  offending block named — the same contract as the existing `_check_layout`.
- **Consensus.** For each context block, the arithmetic mean of `imp.statistics_` across members;
  when the members carry a scaler, the arithmetic mean of `mean_` and of `scale_`, with
  `scale_` floored away from zero exactly as `_fit_scaler` does.
- **Use.** `transform_context` transforms the encoded context row **once** with `context_pre_`
  and hands the same arrays to every member — replacing M per-member transforms. The per-member
  `fold_context` call stays per-member, because the GP kernels differ, but every fold is now
  built from identical context values.
- **Consistency.** `_predict_from_records` and `predict_matrix` split their design matrix: the
  molecular blocks go through each member's own preprocessor (they were fitted on raw molecular
  columns and must stay that way), the context blocks go through `context_pre_`.

This is a **prediction-changing** change. Its size is a deliverable: the notebook reports the
shift on held-out rows, per task, against the current per-member behaviour.

## Changes by file

- **`tackai/fusion/blocks.py`** — `scale_blocks` default `()`; fit statistics in float64 and
  transform to a configurable dtype defaulting to float32; slice views for contiguous blocks;
  skip the imputer for a block fitted with no missing values; fuse the remaining passes.
  New: a constructor for a consensus preprocessor from several fitted ones.
- **`tackai/fusion/gp.py`** — `dtype` default `torch.float32`; dtype-scaled jitter ladder;
  float64 promotion path for the Cholesky and the triangular solve, recorded on the model;
  train-side kernel terms precomputed in `load_state` and reused by `predict` /
  `predict_in_context`; nonzero `block_weight` blocks resolved in `fold_context`.
- **`tackai/fusion/ensemble.py`** — `context_pre_` and its validation; `transform_context`
  transforms once; the `[valid]` copy hoisted out of the member loop; mol-only float32 scratch
  array; the records path consumes `ok` from `encode`.
- **`tackai/fusion/data.py`** — `encode` returns `(X, ok)` so the records path featurises once.
- **`tackai/fusion/models.py`** — no behavioural change expected; it inherits the new
  preprocessor and GP defaults. `XGBoostFusion` is untouched.

## Notebook

`notebooks/ensemble_inference_speed.ipynb`, outputs to
`notebooks/ensemble_inference_speed_results/` (CSV + PNG), executed end to end.

1. **Setup** — `OMP_NUM_THREADS=1` before imports (the known torch/xgboost libomp clash in this
   `.venv`); load the saved ensembles as the float64 / per-member-context **baseline**.
2. **Where the time goes** — featurise / block-transform / GP kernel+mean / std solve /
   aggregate, at a fixed batch size, plus a `cProfile` table. Cold vs warm featuriser cache.
3. **Scaling** — µs/molecule and mol/s vs batch size (1 → 1024) for the cached-context path with
   and without `std`, the records path, and `predict_matrix`.
4. **Cost of each redundancy** — one micro-benchmark per R1-R6, so each change has a number
   before it is made.
5. **After** — the same curves on the new code, overlaid; speedup bars.
6. **Members 1-25** — latency and throughput against member count, with the 5 fitted members
   cloned up to 25, so the cost of the ensemble size you expect is on the record.
7. **What it cost in accuracy — reported, not gated.** Held-out predictions under
   float32 vs float64 and shared vs per-member context, as a correlation and a max-shift
   number per task, plus how often the Cholesky had to be promoted to float64. This section
   documents the damage for the later retrain; it does not block anything.
8. **Conclusions.**

### Baseline capture

Notebook §2-4 measure code that will not exist after the change, so their CSVs are generated
and committed **before** the package changes land, each row carrying the commit it was measured
at. A later re-run reproduces §5-8 and reads §2-4 from those CSVs. The semantic changes stay
measurable in a single run through the `shared_context`, `dtype` and `scale_blocks` toggles.

## Verification

- The existing fusion suite (127 tests) stays green. Tests that assert float64 or per-member
  context are updated as part of the work, each with the reason recorded — not loosened.
- New tests: consensus construction and its validation failure; the canonical context being
  used on all three paths; float32 as the default dtype; the float64 promotion path on a
  deliberately singular kernel; `scale_blocks=()` as the default; an old-style scaled ensemble
  still loading and scoring.
- The pure redundancy removals (R1-R3, R5, R6) are gated on **bit-exactness** against the
  pre-change path at the same dtype: they must not change a single returned value. The three
  semantic changes (float32, shared context, no scaling) are expected to change values and are
  gated only on the suite staying green.
- Speed claims come from the notebook's recorded CSVs, not from assertions.
- Accuracy is reported per task for the record. It is explicitly **not** a pass/fail criterion,
  per the user's 2026-10-05 instruction.

## Risks

- **float32 Cholesky on a 2627×2627 kernel with exactly repeated rows.** The development table
  measures the same compound in the same context repeatedly, so `K` is genuinely singular and
  the `NOISE_FLOOR` of 1e-3 is what keeps it factorisable. Mitigated by the dtype-scaled jitter
  ladder and the float64 promotion; the promotion rate is reported rather than hidden.
- **Dropping standardisation changes the fit**, so the three saved ensembles are stale under the
  new defaults. They remain loadable and scoreable — which is what notebook §1 depends on — but
  they are no longer fitted the way the code now fits. Accepted: the retrain with
  hyper-parameter optimisation is a separate, later piece of work.
- **The shared context shifts predictions**, because each GP's lengthscales were fitted in its
  own scaled space. Measured and reported in §7; accepted as degradation for now.
- **Accuracy is unguarded for the duration.** Between this work and the retrain, the saved
  ensembles score with semantics they were not fitted under. Anything that consumes
  `ensembles/fusion_*` for a scientific number in that window gets worse answers than the
  published ones. Flagged here so the retrain is not forgotten.
