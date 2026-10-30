# TACK model development and training plan

Review date: 2026-10-30. Scope: the fast fusion models for pDC50, Dmax and activity in the top-level TACK checkout. This is a proposed research programme; no training jobs or model publication were performed for this review.

## Recommendation

Start by making model selection reflect generalization to new chemical series and experimental sources, then compare improved molecular fingerprints and alternative learners under that protocol. Advance a small multitask neural model and a molecular graph model only after establishing these baselines. Preserve the existing POI, E3, cell-line, assay and treatment-time representations throughout. More trials of the current XGBoost recipe alone are unlikely to resolve the observed gap, although the relative contributions of distribution shift, labels and model capacity still need measurement.

## 1. What the latest runs show

The inspected checkout is `f355a74`. The downstream `TACKOS/extern/TACK` checkout is at `5962e68`; it is a different code revision and was not modified. Current HEAD includes an uncertainty-bound fix newer than the original calibrated artifacts, so HEAD alone does not establish their training provenance.

The latest complete three-task training series is `autotpd4`, launched October 29 and completed October 29–30:

| Task | Slurm job | Development labels | CV pool | Stacking + calibration pool | Test labels |
|---|---:|---:|---:|---:|---:|
| Dmax | 4613364 | 4,508 | 3,875 | 633 | 136 |
| pDC50 | 4613365 | 7,125 | 6,175 | 950 | 199 |
| Activity | 4613366 | 4,667 | 4,075 | 592 | 158 |

Training uses `data/AutoTPD-TPDdb.csv` and `data/auto_tpd_plus/AutoTPDplus-PROTAC.csv`; evaluation uses `data/auto_tpd_plus/PROTAC-test.csv`. The recipe runs 5 repeats × 5 outer folds, with 50 Optuna trials and 3 inner folds per outer fold. It tunes tree count rather than using early stopping by default. Protein space defaults to `combined`.

| Task | Outer validation: mean ± SD across 25 members | Final calibrated ensemble on test |
|---|---|---|
| Dmax | RMSE 0.2076 ± 0.0154; R² 0.3949 ± 0.0792 | RMSE **0.2005**; MAE 0.1645; R² **−0.2804** |
| pDC50 | RMSE 0.8631 ± 0.0443; R² 0.5041 ± 0.0538 | RMSE **1.3283**; MAE 1.0523; R² **−0.0377** |
| Activity | ROC-AUC 0.8731 ± 0.0300; AP 0.8251 ± 0.0427 | ROC-AUC **0.6412**; AP **0.5746**; accuracy 0.5380 |

Here AP is average precision, stored as `pr_auc` by the scripts. Fold SD is descriptive variability, not a confidence interval: folds/repeats share observations. The validation column describes individual members, not an independently evaluated stacked ensemble.

Dmax's test RMSE is similar to validation RMSE despite the much worse R². The test target variance is smaller, so R² alone exaggerates the apparent change in absolute error. Both negative regression R² values mean worse squared error than the retrospective test-mean predictor; that predictor is a diagnostic reference, not a deployable baseline. A baseline fitted only on development is still required.

### Stacking and uncertainty are not solving point accuracy

The saved before/after report compares uniform ensembles with fitted/calibrated ensembles:

- Dmax RMSE improves from 0.2074 to 0.2005. Its nominal 90% interval covers 94.1% of test observations but has mean width 0.7222, or 72.2 percentage points degraded.
- pDC50 RMSE changes from 1.3249 to 1.3283. The original calibrated artifact has mean reported standard deviation about 57.7 million, and 65.8% coverage for nominal 90% intervals. Its config contains a member sigma around 4.46 × 10¹¹. These uncertainty values are numerically pathological.
- Activity ROC-AUC improves from 0.5917 to 0.6412, while Brier score worsens from 0.2782 to 0.3141. Better ranking does not imply better probabilities. This comparison combines weight fitting and temperature calibration, so it does not isolate either operation.

A newer local `autotpd4_pdc50_refit` artifact, timestamped October 30 at 12:25, records the new sigma ceiling. Its test RMSE is 1.328294 and R² −0.037734: effectively unchanged point accuracy. The old report's enormous sigma should not be attributed to this refit. Recompute its coverage, interval width and probability-quality diagnostics separately; the inspected refit metric file only reports point metrics. Local logs report successful publication and reload verification of the original three ensembles; remote deployment state and publication of the later refit were not independently checked.

Older `fast_xgb_*` runs report substantially stronger test results (mean member R² 0.538 Dmax, 0.470 pDC50; activity ROC-AUC 0.936). They are historical results, not evidence that the new recipe regressed on an identical benchmark. Input datasets, overlap exclusions, context caches and calibration protocols must be reconciled before making that comparison.

### Evidence of shift and data limitations

Raw CSV inspection, before the loader's filtering, shows:

- Test: 209 rows from 15 DOIs; 169 VHL rows; 101 SMARCA2/SMARCA4 rows; 104 capillary-immunoassay rows.
- AutoTPD-TPDdb: 8,664 rows, dominated by CRBN (6,866), with AR and ESR1 prominent. Its `DOI` field is unavailable/empty in this inspection; discover alternate provenance fields before attempting publication grouping.
- AutoTPDplus-PROTAC: 6,719 rows, 907 nonempty distinct DOIs, 3,337 CRBN and 2,124 VHL rows; 3,375 rows lack `Assay_STD`.

These are marginal raw counts, not matched eligible-task distributions or proof of a causal explanation. They support a distribution-shift audit. The test is also small and clustered by publication, so row-level intervals would overstate precision.

The latest logs exclude 348 development rows matching 52 test-overlapping SMILES identities and drop one unparseable SMILES row. They subsequently report 12,808 rows considered for context encoding, 689 dropped, and 12,119 retained; POI/E3 failure counts overlap. The test loader retains 205 rows, with task-specific missing labels giving the counts above. Audit exclusions by source and endpoint; do not assume every raw row contributes to training.

## 2. Fixed constraints and scientific contracts

- Freeze the exact current context tables, PCA transforms, sequence lookup vocabulary, `protein_space="combined"`, assay normalization and time encoding. Hash their contents. Do not regenerate protein/cell/assay embeddings or refit their PCA for this programme.
- Current combined protein tables supply 86 dimensions each for POI and E3; cell uses 56, assay 8, and time is scalar. Verify actual artifact/table dimensions before training. Learned projections inside a new model may consume these fixed vectors; they do not require new input representations.
- Keep Dmax as fraction degraded and pDC50 as `9 − log10(DC50[nM])`. Preserve activity as **Dmax > 0.80 AND pDC50 > 6.0**, including strict boundary behavior. A single known failing endpoint establishes inactivity; a single passing endpoint does not establish activity.
- Do not add binding measurements, structures, omics, target identifiers as new predictive features, or different context encoders. Source/DOI/target names may support audits and splits, not new input blocks.
- Molecular changes are explicit alternatives: retain the old radius-16, 1,024-bit chirality-aware Morgan fingerprint and existing RDKit descriptor order as a reproducible baseline. Version new encoders; never change an existing artifact's input semantics in place.
- Keep treatment context and provenance in record identity. Do not collapse conflicting measurements across targets, E3s, cell lines, assays or times.

## 3. Phase A — trustworthy data and validation first

### A1. Freeze and audit the benchmark

Save dataset hashes, encoder hashes, feature specification, dependency versions, code revision plus relevant uncommitted script hashes, seeds and explicit row/group assignments beside each experiment. Existing fast-model resume behavior checks filenames, not input identity; reject resume on a mismatch. Keep metadata outside directories interpreted as model-member JSON collections.

Preserve the existing molecule-level test exclusions. Verify canonical/connectivity overlap and near-neighbor chemical-series overlap across partitions. Keep publication groups where available; unknown DOI must not be fabricated or treated as an independent publication per row. Check alternative source identifiers and document the reduced coverage of publication-based evaluation.

The existing test has already guided diagnosis. Freeze it as a historical benchmark and stop scoring every trial/fold on it. Select models using development-only validation. Reserve an untouched publication/temporal cohort, if reliable dates/provenance exist, or prospectively collected data for final confirmation. Without such a cohort, explicitly report historical-test improvements as exploratory rather than a fresh unbiased result.

### A2. Correct the model-selection splits

1. Pass scaffold/chemical-series groups into the inner Optuna folds. Currently `objective()` uses `StratifiedKFold`/`KFold`, even though outer CV uses `StratifiedGroupKFold`.
2. Make stacking-fit and calibration sets group-disjoint, as well as disjoint from all base-model training. The NPZ already saves groups, but the calibration script ignores them when halving the pool with `train_test_split`.
3. Group the internal regularization selection in `XGBStackedEnsemble.fit`; it currently partitions a random row permutation.
4. For multitask models, assign each molecule/group once across all endpoints **before** applying task masks. Independent per-task splits can let a held-out molecule's other endpoint enter shared-model training.
5. Fit scaling, imputation, feature selection, density weights, tail thresholds and any molecular dimensionality reduction on the training partition only. The current Dmax tail cutoff is computed on the entire CV pool before the outer split; make it fold-local.

Use scaffold-grouped validation as the primary screen and publication/source holdouts as a shift stress test. Add leave-target-family-out diagnostics only where there are enough independent groups; existing vectors may be available for a target without its labels being available in training. Report this distinction.

### A3. Audit labels without changing representations

`build_table()` currently clips Dmax to [0,100], ignores inequality/range operators, averages `DC50_h` and `Dmax_h`, and does not deduplicate measurements. Audit out-of-range values, inequalities, approximate values, repeated exports, conflicts and endpoint-time mismatches. There are nonempty operator columns in all three raw files.

Run a curated exact-measurement baseline, then a censor-aware alternative. Reverse inequality direction when converting DC50 bounds to pDC50. Use interval/one-sided losses for supported censored endpoints; do not treat bounds as exact observations. Derive activity only when bounds establish the threshold result, otherwise mask that label. Separate approximate values from exact or censored values. Report coverage changes rather than silently removing difficult rows.

Preserve the scalar time representation. Compare current mean-time compatibility behavior with endpoint-specific use of the already available hours; this is a separately documented label/context correction. Do not force joint activity labels from incompatible times. Retain unresolved conflicting measurements, with optional replicate/source weights fitted without validation/test information.

Deliverable: a frozen split file, data audit and baseline report. No large search should precede this.

## 4. Phase B — economical model and molecular-feature comparisons

Use identical eligible observations and split assignments for paired comparisons. First vary molecular features with one fixed learner; then vary learners with the strongest few feature sets. Avoid a full Cartesian search.

| Experiment | Proposed comparison | Purpose |
|---|---|---|
| B0 | Development-mean/median regressors, prevalence classifier; regularized linear baseline | Establish deployable floors and expose bias |
| B1 | Current XGBoost/features under corrected splits, including uniform ensemble | Separate protocol changes from architecture changes |
| B2 | Morgan radius 2/3, 2,048/4,096 bits; count fingerprints for finalists; same descriptors | Test local chemistry, folding collisions and repeated substructures |
| B3 | CatBoost and ExtraTrees using the same fixed input blocks | Test different inductive biases without changing context |
| B4 | Small residual MLP with separate molecular/context projections | Learn molecular–context interactions on fixed vectors |
| B5 | Optional existing additive/product-kernel GP on a bounded subset | Check smooth interactions; avoid full exact-GP scaling initially |

Radius 16 is unusually broad relative to the proposed local fingerprints; its harm here is a hypothesis, not an established bug. Measure bit occupancy/collision proxies and validation performance. Retain chirality. RDKit supports explicit radius, size, chirality and count fingerprints ([official API](https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html)).

For XGBoost, include unweighted training (`alpha_w=0`) explicitly, stronger regularization, shallower depths (2–8), `reg_alpha`, and a bounded CPU thread count. Compare squared error against a robust loss. Tune classification class weighting against no weighting; current positive weighting is automatic. Align the regression selection objective with RMSE/MAE and assess rare-region performance separately: current Dmax tuning averages rare/common RMSE, which need not minimize overall test RMSE.

For the MLP, start with 2–3 layers, width 128–256, dropout and weight decay, and interaction terms between projected molecule and context vectors. Use training-only scaling and early stopping within inner training. These are candidate settings, not promised improvements.

Use context-only and molecule-only ablations for diagnosis, never as an excuse to add new context representations. Report errors by source, target family, E3, assay, time and nearest-training-molecule similarity, with sample/group counts.

## 5. Phase C — multitask and graph models

Advance this phase if simple baselines leave a meaningful gap and the audit supports usable shared supervision.

First train a shared MLP with pDC50, Dmax and activity heads on the frozen feature blocks. Mask missing labels, normalize regression losses using training-only statistics, and tune modest task weights within inner validation. Compare against three independent heads/models; shared training can cause negative transfer. The activity label is derived from endpoints, so it is an auxiliary consistency signal rather than independent evidence.

Next replace the molecular fingerprint branch with a directed message-passing graph encoder, concatenate the same context vectors and optionally the existing descriptors, and compare single-task versus masked multitask heads. Preserve atom/bond stereochemistry. Start with depth 3–5 and hidden size 128–256, then test whether a larger molecular receptive field helps these large molecules. Chemprop provides a multitask regression example; a mixed regression/classification head will require explicit implementation and verification rather than assuming a CLI switch ([official multitask example](https://chemprop.readthedocs.io/en/main/multi_task.html)).

Keep Dmax bounded through an appropriate output/loss, with explicit handling of zero and one; pDC50 remains continuous. For censor-aware experiments use a likelihood or distance-to-valid-interval objective whose direction is tested. Never impute unknown endpoint labels as zero. Do not multiply independent endpoint exceedance probabilities to define activity without checking the dependence assumption.

Only after whole-molecule baselines, consider POI-ligand/linker/E3-ligand molecular encodings derived from the existing SMILES. Validate decomposition confidence and retain a whole-molecule fallback. This adds preprocessing risk and is lower priority. Frozen pretrained molecular embeddings are another optional later comparison; document their training-data provenance and overlap risk. No new biological representation is needed.

## 6. Selection, calibration and decision rules

Use RMSE as the primary regression criterion, MAE/R² and rank correlation as secondary metrics. Report Dmax errors in both fractions and percentage points. Use activity AP as the primary ranking metric, with ROC-AUC, log loss, Brier score, prevalence baseline and precision/recall at a threshold selected without test data.

Compare uniform averaging, strongly regularized convex weights optimized for prediction error, and the existing likelihood-based stacker. Select ensemble form using development-only nested evaluation. Evaluate uncalibrated, weights-only and calibrated outputs separately. Conformal scaling should be assessed for coverage and width, not expected to improve regression means. For activity compare no calibration, temperature scaling and, if sample sizes support it, a simple alternative selected within development.

Use group bootstrap intervals for paired metric differences, resampling publication or chemical-series groups as appropriate. Do not treat 25 member predictions on the same test rows as 25 independent experiments. Report small or single-class slices descriptively when metrics are undefined or unstable. Distribution shift limits any claim of nominal conformal coverage.

Proposed advancement criteria, fixed before the search:

- At least 5% relative regression RMSE improvement over the corrected B1 baseline, or +0.03 activity AP, with supportive paired uncertainty and no substantial degradation on the source-shift validation suite.
- Inspect MAE, high-potency/high-Dmax subsets and probability quality; do not advance an AP winner solely on ranking if its probabilities deteriorate materially.
- Calibrated intervals must be finite and have useful width; assess empirical 90% coverage with group-level uncertainty. Wider intervals alone are not success.
- Prefer the simpler candidate when differences are inconclusive. These are proposed practical gates, not expected gains or guaranteed test targets.

For final confirmation, select and freeze the recipe, train base models without touching stacking/calibration groups, fit the chosen ensemble, calibrate once, then evaluate the locked candidate on the reserved confirmation set. Keep a common independent holdout for comparisons of the complete pipeline. Do not refit base models on calibration rows without creating a new valid calibration design.

## 7. Compute stages and implementation work

Start with 1 repeat × 5 outer folds × 3 grouped inner folds and about 20 trials for a small shortlist. For reference, the current recipe already implies 25 × 50 × 3 = **3,750 inner model fits per task**, plus 25 refits. Repeating it for every new feature/model combination would be expensive without identifying the failure mode.

Promote only 2–3 configurations per task to 5 × 5 outer evaluation with 30–50 inner trials where justified. Measure time and memory in the initial pilot before assigning a total allocation. Neural candidates begin with one fold and a few seeds; only survivors receive the full comparable split schedule. Keep trials parallel only within explicit Slurm CPU/memory limits.

CPU baselines belong on Arrhenius CPU allocations. Neural/GNN work requires a GH200 allocation and an aarch64-compatible environment/container with GPU access. Preserve the repository's `uv` lock/environment separation and sklearn `<1.8` ceiling. Clear inherited `PYTHONPATH`; set `OMP_NUM_THREADS=1` before mixing torch and XGBoost. Keep caches under project storage and prefetch artifacts on the login node. Do not reuse x86_64 compiled environments on GH200 nodes.

Implementation locations:

- `scripts/train_fast_models.py`: grouped inner folds, fold-local weighting, evaluation separation and safe resume checks.
- `scripts/calibrate_fast_models.py` and `tackai/fusion/xgb_ensemble.py`: group-aware fit/calibration, separate point/probability/coverage reports.
- `tackai/fusion/data.py`: preserved operators/provenance, stable joint task masks and group assignments.
- `tackai/fusion/mol_encoder.py` plus artifact serialization: explicit versioned molecular alternatives, without changing legacy defaults.
- New model adapters: shared-feature MLP and graph encoder, with portable preprocessing and inference metadata.

Tests should cover split/group separation across tasks, censoring direction and threshold boundaries, molecular serialization parity, finite uncertainties, feature-schema rejection and checkpoint reload prediction equality. Before promotion, test real model inference and HTTP batching/order/invalid-molecule behavior. Preserve regression `mean` and activity `proba`; new model families will need an inference adapter because the current loader assumes XGBoost members. Benchmark batch throughput for downstream RL before deployment.

Save experiments locally and disable the current Slurm scripts' automatic production Hub push for this research series. Publish a separately versioned candidate only after review and validation. Keep the frozen context artifact paired with each model; update the downstream submodule/container deliberately if promotion requires it.

## 8. Review checkpoints

1. **Audit checkpoint:** agree on eligible labels, fixed context snapshot, grouped splits and primary metrics.
2. **CPU checkpoint:** compare corrected XGBoost, new fingerprints, CatBoost/ExtraTrees and simple baselines; decide whether neural work is warranted.
3. **Neural checkpoint:** compare shared MLP and graph models against the best CPU result, including latency and negative-transfer checks.
4. **Confirmation checkpoint:** freeze one candidate per endpoint or a demonstrably better shared model; calibrate, evaluate once and prepare a model card with uncertainty and domain limitations.

## Local evidence inspected

- [Training script](../scripts/train_fast_models.py), [calibration script](../scripts/calibrate_fast_models.py), [Dmax Slurm recipe](../scripts/retrain_fast_models_dmax.slurm).
- [Dmax run log](../logs/tack-retrain-dmax.4613364.log), [pDC50 run log](../logs/tack-retrain-pdc50.4613365.log), [activity run log](../logs/tack-retrain-activity.4613366.log).
- [Dmax fold metrics](../models/fast_xgb_autotpd4_dmax/fold_metrics.csv), [pDC50 fold metrics](../models/fast_xgb_autotpd4_pdc50/fold_metrics.csv), [activity fold metrics](../models/fast_xgb_autotpd4_activity/fold_metrics.csv).
- [Before/after report](../outputs/calibration_parity/calibration_metrics.json), [pDC50 refit point metrics](../models/ensembles/autotpd4_pdc50_refit_eval_metrics.json), and the original/refit ensemble configs.
- [Target and split code](../tackai/fusion/data.py), [context specification](../tackai/fusion/context.py), [molecular features](../tackai/fusion/mol_encoder.py), [stacking implementation](../tackai/fusion/xgb_ensemble.py).

Metrics above were read from saved artifacts/logs and fold aggregates recomputed from CSVs; models were not retrained or inference rerun. Raw dataset summaries precede filtering. Ignored local model/log artifacts may not be present in a fresh clone; archive the evidence with the experiment record before sharing the plan externally.
