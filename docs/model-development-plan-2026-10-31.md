# TACK: updated model development and training plan

Review date: 2026-10-31. Scope: pDC50, Dmax and activity in the top-level TACK fast-fusion pipeline. This review incorporates the local experiments available after the [October 30 plan](model-development-plan-2026-10-30.md). It proposes the next work; it does not establish a production-ready replacement.

## Recommendation

Prioritize label quality and generalization across chemical series and experimental sources, then tune a small, task-specific shortlist. Keep every biological-context representation fixed. The strongest current leads are smaller-radius fingerprints with XGBoost for pDC50, and exact-label training with ExtraTrees/CatBoost for activity. Dmax needs further investigation: its source-holdout performance remains close to a simple baseline. A shared neural model is an experiment, not the default successor to XGBoost.

## 1. Evidence from current models and newer experiments

### Historical test performance

The saved `autotpd4` calibrated ensemble metrics are:

| Task | Test labels | Main metrics |
|---|---:|---|
| Dmax | 136 | RMSE 0.2005; MAE 0.1645; R² −0.2804 |
| pDC50 | 199 | RMSE 1.3283; MAE 1.0523; R² −0.0377 |
| Activity | 158 | ROC-AUC 0.6412; average precision 0.5746; accuracy 0.5380 |

Dmax is a fraction, so RMSE 0.2005 means 20.05 percentage points degraded. Negative R² indicates worse squared error than the retrospective test mean; it does not by itself identify the cause. Dmax's narrower test-label distribution also makes its R² poor despite similar absolute validation/test RMSE.

The original pDC50 uncertainty artifact was numerically pathological. The later local refit has essentially unchanged point error (RMSE 1.328294); bounding uncertainty parameters alone did not improve predictions. Probability quality and interval usefulness must be evaluated separately from ranking and point error.

The historical test is already part of model diagnosis. Its 15 raw DOI groups also occur in development metadata, despite molecule exclusions. Treat it as a historical benchmark, not a new source-independent confirmation set. Older model scores cannot be compared directly without reconciling datasets, exclusions and context versions.

### New development evidence

Completed corrected-baseline experiments use five grouped outer folds, grouped inner tuning with 20 trials, and a separate source stress split. Their mean fold RMSE is 0.2112 for Dmax and 0.8049 for pDC50; mean fold activity AP is 0.7413. Source-holdout results below use the same exact-label evaluation policy across candidates:

| Task / metric | Corrected tuned current recipe | Promising completed pilot | Pilot source score |
|---|---:|---|---:|
| Dmax RMSE ↓ | 0.3022 | ExtraTrees, existing molecular features, exact labels | 0.2869 |
| pDC50 RMSE ↓ | 1.0813 | XGBoost, radius-3 / 4,096-bit Morgan, exact labels | 1.0410 |
| Activity AP ↑ | 0.2610 | ExtraTrees, existing molecular features, exact labels | 0.5526 |

These are shortlist signals, not equally tuned final comparisons. The activity source prevalence baseline is 0.2561; Dmax's development-mean baseline has source RMSE 0.2917. Dmax therefore remains particularly weak under source shift.

The label-policy comparison is informative: ExtraTrees activity source AP changes from 0.2675 with compatibility training labels to 0.5526 with exact-label training, evaluated on identical rows. The latter also has Brier score 0.1661 versus 0.2220. This supports further label-policy testing; changed training coverage and composition may contribute to the effect.

For pDC50, exact-label XGBoost with radius-3 / 4,096-bit fingerprints gives pooled out-of-fold RMSE 0.7348. Do not compare pooled scores directly with unweighted mean fold scores. Existing bootstrap intervals compare pilots against the untuned compatibility XGBoost baseline: they must be recomputed against the tuned comparator before promotion. Source-level intervals are wide; the pDC50 source improvement interval against the pilot baseline includes zero.

Small single-task and shared MLP pilots already exist. The shared radius-2 model gives Dmax source RMSE 0.2815 but mean fold RMSE 0.2236, worse than the corrected tree baseline. It gives pDC50 source RMSE 1.1232 and activity AP 0.3347. Censor-aware neural pilots have not consistently improved these results. More neural capacity is therefore not yet justified by evidence.

At inspection, tuned shortlist array **4685712** had three tasks running and three pending. Collect its results before starting duplicate searches. This review did not submit additional jobs.

## 2. Fixed inputs and scientific definitions

- Freeze the current POI, E3, cell-line, assay and scalar time representations, including lookup tables, PCA spaces and normalization. The existing research snapshot records context fingerprint `1de55fc2283b`. No new omics, structures or biological features.
- Keep the existing descriptor order. Molecular alternatives may change fingerprints or replace their branch with a graph encoder; save a separate versioned feature specification for each trained artifact.
- Preserve Dmax as fraction degraded and pDC50 as `9 − log10(DC50 in nM)`.
- Preserve activity as **Dmax > 0.80 AND pDC50 > 6.0**. One known failing endpoint establishes inactivity; a missing endpoint plus a passing endpoint does not establish activity. Do not silently switch to the programme prose's 100 nM threshold.
- Preserve target, E3, cell, assay, treatment time and provenance in record identity. Source metadata is for grouping and audits, not additional model inputs.

## 3. Stage A: finish data and evaluation checks

Reuse the existing frozen `research_v1` snapshot rather than recreate splits. It has 12,119 encodable development rows and 4,465 molecular/scaffold components:

| Partition | Rows | Groups | Use |
|---|---:|---:|---|
| Search | 8,759 | 3,124 | Feature, learner and hyperparameter selection |
| Stacking | 1,260 | 447 | Fit ensemble weights after recipe selection |
| Calibration | 1,027 | 447 | Calibrate the fixed ensemble |
| Internal confirmation | 1,073 | 447 | Evaluate the locked pipeline once |

These are all-row counts before task masks. Shared partitions prevent multitask leakage through another endpoint. Internal confirmation is not prospectively collected external data.

Finish audits of repeated exports, conflicting measurements, source imbalance, censoring and endpoint-time mismatches. Raw TPDdb includes 192 out-of-range Dmax values; exclude these from the exact-label baseline instead of silently clipping. Keep excluded rows and reasons in an audit table. Do not collapse conflicting contexts or fabricate missing identifiers.

Retain exact-label and compatibility controls. Evaluate one-sided/interval labels through explicit censor-aware objectives, reversing inequality direction when converting DC50 to pDC50. Approximate measurements need their own policy. Compare endpoint-specific reported hours against compatibility mean-time handling while retaining the same scalar time representation; activity requires compatible endpoint times.

All preprocessing, class/density weights, thresholds and early stopping must be fitted within training folds. Keep molecular components together in inner tuning, outer validation, stacking and calibration. Preserve the source stress split's publication separation and molecular-overlap purge.

**Deliverable:** updated audit with task-specific row counts, exclusions, split hashes and checks of cross-partition overlap.

## 4. Stage B: train and compare the shortlist

Complete the existing six-configuration tuned shortlist:

| Task | Candidates |
|---|---|
| pDC50 | XGBoost with radius-2 / 2,048-bit and radius-3 / 4,096-bit fingerprints |
| Dmax | ExtraTrees and CatBoost with current molecular features |
| Activity | ExtraTrees and CatBoost with current molecular features |

Use exact-label training and the identical evaluation rows for the principal comparison. Retain corrected current XGBoost, dummy predictors and useful existing molecular-feature controls. Do not force all three endpoints to share a learner or molecular representation.

Compare unweighted versus density/class weighting, appropriate regularization and squared versus robust regression objectives where supported. Examine high-potency/high-Dmax errors separately instead of optimizing a tail metric while claiming overall RMSE gains.

Recompute pooled out-of-fold scores and paired bootstrap differences against the **tuned** baseline. Resample chemical components for chemical validation and publications for source validation. Report per-source, E3, target-family, assay, time and molecular-similarity slices with sample and group counts. Context-only and molecule-only pilots are diagnostic controls.

Advance at most two candidates per endpoint to repeated grouped evaluation, targeting five repeats of five folds. Use 30–50 inner trials only if the initial tuning curves justify more search. Include every attempted configuration and failed run in the experiment ledger. Record wall time, peak memory and inference latency before expanding the allocation.

**Deliverable:** paired comparison table, repeated-split stability and one frozen recipe per endpoint, or a documented decision that improvement remains inconclusive.

## 5. Stage C: bounded neural and molecular-graph experiments

Keep the existing MLP results as a negative-transfer control. Allow a small follow-up on loss weights/regularization only when it tests a specific explanation for current errors. Mask missing targets and normalize losses with training-only statistics; activity is derived supervision, not an independent biological measurement.

Because Dmax source generalization remains weak, run a controlled directed-message-passing molecular encoder pilot. Replace only the molecular branch; concatenate the same frozen context vectors and optionally existing descriptors. Preserve stereochemistry. Compare single-task and masked multitask heads, with bounded Dmax output and continuous pDC50 output.

Begin with one grouped fold, a source stress evaluation, depth 3–5 and a modest hidden width. Budget a few seeds before full cross-validation. Stop if gains disappear across seeds or require unacceptable inference cost. Fragment-aware or pretrained molecular encodings are later options, requiring a separate preprocessing/provenance audit; they are not prerequisites for the first improved candidate.

Use CPU Slurm allocations for tree models and small neural pilots where measured runtime permits. GPU work requires a GH200 allocation and a matching aarch64 environment/container. Preserve project-local locks and the sklearn `<1.8` ceiling; keep caches on project storage.

## 6. Stage D: ensemble, calibrate and confirm

Select ensemble form within development evaluation: compare uniform averaging with strongly regularized convex stacking. Fit final weights only on reserved stacking groups. Calibrate afterward on independent calibration groups. Base models must not have trained on either holdout.

Report regression RMSE, MAE, R², rank correlation, 90% interval coverage and width. For activity report AP, ROC-AUC, Brier score, log loss and precision/recall at a development-selected threshold. Separate raw, weights-only and calibrated predictions. Check finite outputs and useful uncertainty under source shift; nominal conformal coverage is not guaranteed under that shift.

Proposed promotion gates, fixed before further confirmation:

1. At least 5% relative regression RMSE improvement or +0.03 activity AP over corrected tuned XGBoost, with supportive paired uncertainty and no material source-shift deterioration.
2. No material degradation in regression MAE, activity probability quality or important sufficiently populated slices. Prefer the simpler model when differences are inconclusive.
3. Finite, useful uncertainty, reproducible artifact reloads and acceptable measured batch latency.

Freeze the complete recipe, evaluate internal confirmation once, then report historical-test performance as exploratory. If a candidate fails, document the failure; repeated tuning against confirmation would convert it into development data. A new independently collected cohort remains necessary for a strong external-generalization claim.

## 7. Handoff and review artifacts

Deliver versioned model artifacts with molecular/context contracts, training and split manifests, paired evaluation tables, calibration diagnostics, latency measurements and a model card documenting domains of weakness. Verify live HTTP batch ordering, invalid-molecule handling and regression `mean` versus activity `proba` outputs before downstream adoption. Preserve the existing models for rollback.

Review checkpoints are: data-policy audit, tuned CPU comparison, optional graph decision, and locked final evaluation. No new candidate should be described as improved on the historical test until that evaluation actually exists.

## Local evidence

- [Original detailed review](model-development-plan-2026-10-30.md) and [raw data audit](surrogate-data-audit-2026-10-30.md).
- Saved test metrics: [Dmax](../models/ensembles/autotpd4_dmax_calibrated_eval_metrics.json), [pDC50](../models/ensembles/autotpd4_pdc50_calibrated_eval_metrics.json), [activity](../models/ensembles/autotpd4_activity_calibrated_eval_metrics.json), [pDC50 refit](../models/ensembles/autotpd4_pdc50_refit_eval_metrics.json).
- [Frozen snapshot manifest](../models/research_v1/manifest.json), [snapshot verification](../models/research_v1_verification.json), [pilot summary](../models/research_screen_summary.json).
- Corrected tuned baseline metrics: [Dmax](../models/research_tuned_baseline/dmax/metrics.json), [pDC50](../models/research_tuned_baseline/pdc50/metrics.json), [activity](../models/research_tuned_baseline/activity/metrics.json).
- [Benchmark runner](../scripts/surrogate_research/benchmark_surrogates.py), [shortlist recipe](../scripts/surrogate_research/slurm/tune_surrogate_shortlist.slurm), [neural benchmark](../scripts/surrogate_research/benchmark_neural_surrogates.py), [label policies](../tackai/fusion/labels.py), [default molecular encoder](../tackai/fusion/mol_encoder.py).

This review read local artifacts and recomputed baseline fold aggregates without retraining models. Ignored model/log files may be absent from fresh clones; archive the referenced evidence with any shared report. Production deployment state was not independently checked.
