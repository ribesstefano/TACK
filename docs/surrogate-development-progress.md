# Surrogate development progress

## Scope and current status

The active objective is to deliver a fast, efficient and reliable Dmax, pDC50 and
activity surrogate, with usable uncertainty, for TACKOS. The research branch is
`research/reliable-surrogates`. The full [development plan](model-development-plan-2026-10-30.md)
remains in force. No new candidate has passed model promotion or replaced production.

## 2026-10-30: validation and first experiments

- Preserved the original models and unrelated local files. Added grouped inner
  tuning, fold-local tail thresholds, bounded workers, opt-in historical-test
  evaluation and resume provenance checks to the existing trainer.
- Added group-disjoint stacking/calibration and internal stacking regularization
  selection. The calibration tool checks saved provenance and reports probability
  quality and interval coverage/width. Legacy test splitting requires explicit opt-in.
- Preserved raw endpoint operators, times, source rows and publication/reference
  metadata without adding them to predictive inputs. Added exact/censored label
  policies and strict pDC50/Dmax threshold tests; the compatibility labels remain
  available separately.
- Added molecular feature specifications, including count fingerprints, with
  descriptor order/RDKit checks and cache separation from binary fingerprints.
- Generated the [raw data audit](surrogate-data-audit-2026-10-30.md). All 15 raw
  historical-test DOI groups also occur in development metadata. TPDdb supplies
  92 alternate `Reference` groups; its DOI field is empty. Its 192 out-of-range
  raw Dmax measurements need an explicit label policy.

### Frozen experiment

Preparation job **4642398 completed** in 2m06s, with peak recorded RSS about 836 MB.
`models/research_v1/manifest.json` records data hashes, code hashes, dependency
versions, context fingerprint `1de55fc2283b`, feature contracts and artifact hashes.
The actual context tables are copied under `models/research_v1/context/`.

The snapshot contains 12,119 context-encodable development rows and 4,465 connected
scaffold/molecular identity groups after historical-test molecule exclusions.
All tasks share partition assignments before task masks are applied:

| Role | Rows | Groups |
|---|---:|---:|
| Search | 8,759 | 3,124 |
| Stacking | 1,260 | 447 |
| Calibration | 1,027 | 447 |
| Internal confirmation | 1,073 | 447 |

The confirmation partition is reserved for new recipes; it is not a prospectively
collected external test. Only its counts/label availability have been inspected.
The development source stress split has 4,983 training rows and 3,111 evaluation
rows, with 665 chemically related rows purged from its training side.

Snapshot verification job **4642736 completed** successfully: all artifact/context
hashes match, no standardized historical-test molecule overlap was found, source
training/evaluation share neither publication IDs nor molecular components, and
32 sampled rows in each of six feature matrices match live inference encoding.
Evidence: `models/research_v1_verification.json` and its Slurm log.

### Tests and jobs

- **4642589:** 77 tests passed in 40.49s, covering research labels/partitions,
  molecular specifications, grouped training/calibration and existing fusion
  data/XGBoost ensemble behavior.
- **4642736:** three additional benchmark-metric edge-case tests passed.
- **4642478:** initial 27-candidate fixed-parameter pilot array (three tasks;
  dummy, linear, ExtraTrees and six XGBoost molecular representations). Uses only
  development CV and source holdout. Active/completion state must be read from
  Slurm and each candidate's `COMPLETE` file.
- **4642674:** retries only the two failed dummy-regression candidates after fixing
  undefined Spearman correlation serialization; output goes to a separate retry
  directory. The failed originals are retained for diagnosis.
- **4642767:** exact-label training comparisons for ExtraTrees and XGBoost, on the
  same exact-label evaluation rows. This separates training-label policy from
  representation changes; it is not a new test set.

Preliminary complete candidates show substantial source shift. For example,
ExtraTrees activity mean fold AP is 0.7247 but source-holdout AP is 0.2675; the
source prevalence baseline is 0.2561. Dmax source RMSE for the initial XGBoost
baseline is 0.3368 versus 0.2917 for the development-mean dummy. These are pilot
results, not a tuned comparison or a confidence interval.

### Follow-up experiments

The same source-holdout activity rows show a large improvement under exact-label
training: ExtraTrees AP rises from 0.2675 to 0.5526; XGBoost with the original
molecular features rises from 0.2330 to 0.5220. This is evidence for investigating
label handling, not proof that all generalization issues have been resolved.

- **4642949:** three tuned corrected-B1 baseline jobs are running (20 trials per
  outer/source fold). The comparator retains the current recipe's search space,
  density weighting and classification weighting, with grouped inner folds and
  training-only rare-region thresholds. Evaluation uses the same exact-label
  rows as the new candidates. Five benchmark tests passed before training.
- **4643011:** paired group-bootstrap summary of all completed pilot candidates
  is queued; output is `models/research_pilot_summary.json` (500 draws for this
  screening report). It explicitly lists incomplete original jobs.
- The initial implementation is committed as `a78e635`; model/cache artifacts
  remain ignored and production artifacts remain untouched.

## Outstanding work before completion

1. Finish and summarize all pilots with pooled out-of-fold scores and paired
   chemical-group bootstrap intervals. Assess exact-label effects and source shift.
2. Establish the properly tuned corrected B1 baseline; tune a small shortlist,
   compare remaining planned learners/ablations, and repeat finalists across splits.
3. Implement and evaluate shared/single-task neural and graph candidates if the
   CPU comparisons leave a meaningful gap, as required by the plan's phase gates.
4. Evaluate censor-aware losses and endpoint-time policy where evidence supports
   them. Preserve the fixed context representations and shared task partitions.
5. Select complete pipelines on development evidence; fit and calibrate on their
   reserved groups, evaluate the locked candidates on internal confirmation and
   the historical benchmark with appropriate limitations. Prospective external
   confirmation remains unavailable in the current files.
6. Verify finite, useful uncertainty, source/slice behavior, round-trip artifact
   parity, live HTTP inference and TACKOS batch latency before promoting a candidate.
   Publish only a separately versioned, reviewed candidate; preserve rollback.

No objective-completion claim is supported yet. Keep expensive ignored artifacts,
job logs and frozen data in project storage; do not add them to Git.

## 2026-10-31: tuned comparisons and deployment preparation

This section supersedes the running/queued statuses in the October 30 notes.
The corrected baseline, original/exact pilots, CatBoost/ablation pilots, and
single/shared MLP pilots have completed. Their outputs are still development
experiments; no confirmation or new historical-test evaluation has been run.

- **4685712:** six tuned shortlist configurations submitted; first three tasks
  running and remaining three pending at the latest inspection. Do not duplicate
  these jobs. They compare ExtraTrees/CatBoost for Dmax/activity and radius-2/3
  XGBoost for pDC50, with exact training labels.
- **4685802:** 14 benchmark/paired-summary tests passed. The summarizer now accepts
  an explicit reference root, checks matching rows, labels, groups and folds,
  rejects incomplete repeats and group-crossing folds, and identifies comparator
  paths in each result.
- **4685807:** completed comparison of 79 candidates against the tuned B1 baseline
  with 1,000 paired bootstrap draws. The two failed original dummy runs remain
  explicitly incomplete; their successful retry artifacts are included.
  Evidence: `models/research_tuned_comparison.json`.
- **4685823:** consolidated research/fusion regression checks passed: **96 tests**
  in 18.80 seconds, including masked neural supervision and artifact round trips.
- **4685835:** development slice reports generated; two slice tests passed.
  They read metadata and existing development predictions only, reject rows from
  reserved partitions, retain group counts and flag small slices. Subsequent
  reports add paired E3 intervals and a CatBoost activity comparison.

### Paired results against the tuned baseline

All CV values here are pooled out-of-fold metrics, not means of fold metrics.
Positive differences indicate improvement; intervals are percentile bootstrap
intervals conditional on these development experiments, not corrections for
search over many candidate models.

| Candidate | View | Candidate / baseline | Improvement, 95% bootstrap interval |
|---|---|---|---|
| Exact ExtraTrees Dmax | CV RMSE | 0.2095 / 0.2111 | 0.0016 [−0.0063, 0.0082] |
| Exact ExtraTrees Dmax | Source RMSE | 0.2869 / 0.3022 | 0.0154 [0.0034, 0.0252] |
| Exact radius-3/4096 XGBoost pDC50 | CV RMSE | 0.7348 / 0.8086 | 0.0738 [0.0506, 0.0971] |
| Exact radius-3/4096 XGBoost pDC50 | Source RMSE | 1.0410 / 1.0813 | 0.0403 [−0.0145, 0.1063] |
| Exact ExtraTrees activity | CV AP | 0.7361 / 0.7244 | 0.0117 [−0.0110, 0.0342] |
| Exact ExtraTrees activity | Source AP | 0.5526 / 0.2610 | 0.2916 [0.0156, 0.4382] |

The activity source gain hides an important E3 difference. CRBN AP improves from
0.2599 to 0.5915 (636 rows, 31 publication groups), but VHL falls from 0.4456 to
0.3088 (75 rows, 21 publication groups). This matters because the historical test
is VHL-heavy. Do not promote the aggregate winner without addressing this slice.
Dmax's AR and BTK source slices also regress despite aggregate improvement.

Follow-up source E3 bootstraps (jobs **4685837** and **4685838**) give an activity
ExtraTrees VHL AP difference interval [−0.2574, 0.1282], so the apparent decrease
is uncertain, not established harm. CatBoost's VHL point AP is 0.3836 versus the
baseline 0.4456, also inconclusive [−0.2191, 0.1275]. Neither aggregate score is
sufficient to dismiss the VHL concern.

Search-only label attrition audit **4685913** completed and wrote
`models/research_label_attrition.json`. VHL compatibility activity has 843 labels
(392 positive), while exact activity has 532 (200 positive). Of the 192 lost
compatibility positives, 171 have a missing endpoint time, 2 have differing known
times, and 48 lack an exact endpoint (reasons overlap). CRBN loses 306 positives,
of which 122 lack time and 218 lack an exact endpoint. There are no conflicting
observed exact/compatibility activity labels in these E3 slices: filtering and
coverage change, rather than observed-label flips, drive this policy comparison.
This makes endpoint-time missingness and cohort selection important next checks;
it does not justify inventing missing times or treating uncertain positives as true.

A raw search-row time audit found 3,342 rows with both endpoint times known,
including 21 with differing times; 3,369 have only one time and 2,048 have none.
Only 15 of the differing-time rows have a Dmax value and 21 have a DC50 value;
among rows with just one time, seven Dmax-valued rows and 20 DC50-valued rows
currently inherit the other endpoint's time. These counts precede exact-label
eligibility checks. Endpoint-specific time handling is scientifically worth
documenting, but this small directly affected subset is unlikely by itself to
explain the aggregate error. The reproducible search-only audit is
`models/research_label_attrition_v2.json` (job **4711866**).

### Remaining implementation work now underway

A separate Chemprop graph probe replaces only the molecular fingerprint branch.
It is explicitly a partial pilot (one chemical fold plus source evaluation), not
full cross-validation. Initial smoke tests exposed API/forward/loss issues; fix
and rerun them before accepting any graph result. Graph dependency additions are
restricted to the research extra. The lock retains existing dependency versions;
the local tackai metadata is corrected from stale 1.0.0 to 2.0.0.

A portable, torch-free candidate artifact for alternative tree learners is under
review. In particular, any residual scale used by normalized conformal scores
must be fitted on stacking/training data and frozen **before** conformal calibration.
Fitting that scale on the same conformal labels would invalidate the intended
split-conformal interpretation. Prediction pooling must preserve each estimator's
actual semantics, and activity entropy outputs must remain coherent after scaling.
These artifacts have not yet been integrated into serving or fitted as final models.

After correcting the graph forward path, Chemprop API calls and loss scaling,
five focused graph tests passed. They check stereochemistry, masked supervision,
normalization/early stopping, singleton/empty inference and safe artifact reload.
Job **4685914** is the active bounded pilot in
`models/research_graph_pilot_run2/dmpnn_probe`; earlier failed or incomplete attempts
remain preserved. The graph is not yet a validated candidate. Commit **3a039c4**
records the tested benchmark, neural pilot, slice reporting and updated review work.

The [updated review plan](model-development-plan-2026-10-31.md) records current
evidence and priorities. The original objective still includes repeated finalist
evaluation, independent calibration/confirmation, portable inference, HTTP/TACKOS
verification, latency measurement and a model card. None is waived by pilot gains.

## Tuned shortlist and repeated evaluation

All six tasks of array **4685712** completed, as did the three bound-established
activity label pilots in **4685919**. Paired summary **4693035** covers all nine
completed runs against the tuned current-recipe baseline; evidence is in
`models/research_tuned_shortlist_comparison.json`. Intervals are paired chemical
component bootstrap for CV and publication bootstrap for the source split.

| Candidate | Pooled CV / baseline | Source / baseline | Source improvement, 95% interval |
|---|---|---|---|
| pDC50 tuned XGB radius 2, 2048 | RMSE 0.7185 / 0.8086 | 1.0667 / 1.0813 | 0.0147 [−0.0579, 0.1016] |
| pDC50 tuned XGB radius 3, 4096 | RMSE 0.7241 / 0.8086 | 1.0424 / 1.0813 | 0.0389 [−0.0215, 0.1131] |
| Dmax tuned ExtraTrees | RMSE 0.2088 / 0.2111 | 0.2827 / 0.3022 | 0.0196 [0.0045, 0.0286] |
| Activity tuned ExtraTrees | AP 0.7459 / 0.7244 | 0.5566 / 0.2610 | 0.2956 [−0.0157, 0.4269] |

The activity source interval spans zero despite its large point estimate. The
bound-established activity label pilots produced source AP 0.4005 (ExtraTrees),
0.3067 (CatBoost) and 0.3065 (XGBoost), versus 0.5566 for tuned exact ExtraTrees;
these are not matched-tuning comparisons and do not justify dropping censoring
research generally.

The graph partial pilot **4685914** completed. One shared-model chemical fold has
Dmax RMSE 0.2148, pDC50 RMSE 0.8053 and activity AP 0.7410. The source scores are
0.3357, 1.1558 and 0.3539 respectively. It does not provide evidence to promote a
graph model; its one fold and short training are a resource-bounded screen only.

Array **4693047** evaluates four finalists and their three corrected baseline
comparators under the same five-repeat/five-fold schedule, with 20 grouped inner
trials. It uses only search/source validation groups; confirmation remains reserved.
The longer runs must complete before robust repeated-split selection. They may
exceed the twelve-hour per-task allocation: inspect each Slurm status and preserved
fold ledger before deciding whether to resume or alter resource limits.

Tuned E3 slice reports (**4693339**) show activity CatBoost has VHL source AP
0.4699 versus the baseline 0.4456, while tuned ExtraTrees has 0.3397; both paired
VHL intervals include zero. CRBN AP is 0.4013 for CatBoost and 0.5949 for
ExtraTrees. Dmax CatBoost also has a slightly better pooled CV point estimate
than ExtraTrees. These opposing slice signals justify retaining both as options;
array **4693464** adds CatBoost activity and Dmax to the same repeated protocol.
The reports are exploratory and are not corrected for trying multiple models.

A research candidate wrapper and explicit recipe-driven fit tool are being tested.
The regression residual scale is fitted on stacking groups and frozen before the
conformal quantile is fitted on calibration groups. A separate serving path can
load trusted local calibrated candidates with different per-task molecular specs,
after validating the frozen context and feature dimensions. It does not change the
default published-model loader. These are not yet a promoted surrogate; the
candidate-serving and fit tests must pass, followed by model-level validation,
calibration, locked confirmation, live HTTP and downstream performance checks.

Focused artifact, fit, graph and benchmark tests passed (**4693034**, 19 tests).
Candidate and existing HTTP TestClient checks passed (**4693174**, 30 tests).
A real loopback candidate-server/REINVENT-client test passed (**4693284**); it
also detected and verified a fix for stale port files after SIGTERM. The separate
TACKOS worktree `TACKOS-surrogate-research`, branch `research/tack-surrogate-http`,
contains commit `131e666`, a standard-library HTTP predictor and opt-in Dmax
reward config. Its five focused predictor/reward tests passed (**4693395**), and
its original checkout's unrelated changes remain untouched. Cross-repository
loopback test **4693318** is still running; full TACKOS RL/container validation
has not been performed.

`scripts/surrogate_research/freeze_surrogate_recipe.py` can extract explicitly selected, complete
grouped-fold settings with source-run hashes. The candidate fit recipe now pins
the snapshot manifest SHA-256 and rejects another snapshot before training. This
tool does not automatically choose a winner; do not freeze a final recipe until
the repeated validation and slice review have finished. Candidate use and trust
assumptions are documented in `docs/surrogate-candidate-artifacts.md`.
Recipe freezing and snapshot-lock tests passed in job **4693636** (8 tests).

## 2026-11-01: repeated validation and controlled follow-up

Arrays **4693047** and **4693464** completed all nine five-repeat/five-fold
experiments. Job **4711191** produced matched pooled out-of-fold comparisons
against the repeated corrected baseline, and **4711205** produced E3/source
slices. Both use 1,000 paired bootstrap draws. The candidate training policy is
exact-label; the current-XGB reference uses compatibility labels, so differences
include label policy and cannot be attributed solely to model or fingerprints.

| Candidate | Pooled CV / baseline | CV improvement (95% interval) | Source / baseline |
|---|---|---|---|
| pDC50 XGB r2/2048 | RMSE 0.7108 / 0.7799 | 0.0691 [0.0528, 0.0874] | 1.0179 / 1.0506 |
| pDC50 XGB r3/4096 | RMSE 0.7091 / 0.7799 | 0.0708 [0.0545, 0.0885] | 1.0274 / 1.0506 |
| Dmax CatBoost | RMSE 0.2013 / 0.2074 | 0.0060 [0.0009, 0.0107] | 0.2969 / 0.3408 |
| Dmax ExtraTrees | RMSE 0.2051 / 0.2074 | 0.0022 [−0.0044, 0.0074] | 0.2871 / 0.3408 |
| Activity CatBoost | AP 0.7684 / 0.7392 | 0.0292 [0.0093, 0.0549] | 0.4606 / 0.2292 |
| Activity ExtraTrees | AP 0.7545 / 0.7392 | 0.0153 [−0.0046, 0.0386] | 0.5426 / 0.2292 |

Only pDC50 passes the prespecified numeric CV gate. The source pDC50 paired
intervals include zero. Direct radius-2 versus radius-3 comparisons (job
**4711242**) are inconclusive on CV and source holdout, so radius 2 is the
provisional lower-width candidate, not a proven better representation. Dmax
CatBoost's 2.9% CV reduction misses the 5% gate, and activity CatBoost's +0.0292
misses the +0.03 gate. Activity source VHL AP is 0.2693 for CatBoost and 0.3295
for ExtraTrees versus 0.3936 for the baseline (75 rows, 21 publications); both
intervals include zero but do not exclude material degradation.

As a descriptive Dmax threshold check, scoring `Dmax > 0.80` at a fixed
predicted-Dmax threshold of 0.80 gave repeated-CV recall 0.528 for CatBoost,
0.450 for ExtraTrees and 0.406 for the current baseline; the corresponding
source-holdout recalls were 0.413, 0.180 and 0.382. These thresholds were not
optimized and no uncertainty interval is attached. They do not override the
primary RMSE gate, but show that a model can change the high-degradation signal
even when its aggregate RMSE changes little.

Exact-label current-XGB comparators (**4711281**, summarized in **4716929**)
completed on identical folds. For pDC50, radius 2 achieved CV RMSE 0.7108
versus 0.7331 (3.0% lower; paired 95% improvement interval [0.0122, 0.0331]);
source RMSE was 1.0179 versus 1.0688 (interval [0.0054, 0.0945]). Radius 3
had CV RMSE 0.7091 and source RMSE 1.0274. Thus much of the larger gain over
the compatibility-trained reference is associated with label-policy changes;
the matched-label candidate's CV gain is below 5%. Its representation and
training settings both differ from the comparator. Dmax CatBoost improved CV RMSE
from 0.2064 to 0.2013 (2.5%), while activity CatBoost changed CV AP from
0.7675 to 0.7684. Neither supports promotion against this matched baseline.
These are development comparisons, not confirmation or test results.
Matched-baseline biological and source slice reports (**4716990**) also flag
activity CatBoost on source-held-out VHL: AP 0.2693 versus 0.4272 for matched
XGBoost, difference −0.1580, publication-bootstrap interval [−0.2439, −0.0009]
over 75 rows and 21 publications. This exploratory, unadjusted subgroup result
reinforces the concern about activity generalization; it is not a confirmation
test. ExtraTrees VHL AP was 0.3295, with its paired interval crossing zero.
The radius-2 pDC50 fold parameters were saved as a *pre-calibration* recipe in
`models/research_recipes/pdc50_r2_precal.json`; no final candidate has been
selected. A search-only endpoint-head activity ranking experiment (**4711307**,
with dependent combine/summary jobs **4711308** and **4711320**) tests a fixed
joint-threshold score without assuming endpoint independence. Its output is an
uncalibrated ranking score, not an activity probability. Failed initial test/job
chain **4711298–4711302** was corrected before resubmission.

The endpoint ranking jobs completed: pooled CV AP was 0.6735 with CatBoost Dmax
or 0.6904 with ExtraTrees Dmax, versus 0.7392 for direct activity XGBoost.
The paired chemical-group intervals for improvement were [−0.0967, −0.0309]
and [−0.0753, −0.0181]. Source AP was 0.2797 and 0.2825 versus 0.2292, with
publication intervals crossing zero. This fixed bottleneck score is an
uncalibrated ranking heuristic and does not warrant promotion.

Benchmark label access now reads only search-row offsets from the frozen NPZ;
it rejects source-stress rows outside search. The search-only dummy smoke run
**4711267** produced all 4,524 prediction records bit-identical to the earlier
baseline. Earlier benchmark runners materialized whole label arrays but used
only search/source rows for scoring; no confirmation metrics were computed.
The cross-repository live check **4711192** passed after earlier filesystem-stalled
job **4693318** timed out. It exercised the actual TACK candidate server,
REINVENT client and isolated TACKOS HTTP predictor on synthetic models; it is
not a full TACKOS RL run.
TACKOS commit `9270f54` adds invalid-row consistency checks; focused tests
**4711806** and the cross-repository loopback check **4711807** passed.

The isolated TACKOS branch's lean REINVENT configuration and HTTP predictor
tests passed together (**4711635**, 18 tests). Its changed files pass Ruff. A
full `ruff check tackos/ reinvent_entrypoint/` reports three pre-existing issues
in untouched files (`reinvent_entrypoint/cli.py` and
`tackos/data/preprocessing/preprocess_pipeline.py`); no unrelated fixes were made.

The provisional pDC50 search-only bundle is being fitted in **4711370**. Its
first attempt **4711347** stopped before training because the snapshot's RDKit
distribution version (`2026.3.1`) and runtime's equivalent formatted version
(`2026.03.1`) were compared literally. The check now compares distribution
versions and records the runtime string in the molecular contract; focused job
**4711368** passed 12 tests. The independent group-held-out stacking study
**4711418** depends on bundle completion and compares uniform against fixed
regularized convex strategies without reading calibration or confirmation labels.
Current conformal intervals target row-marginal coverage under exchangeability;
group-disjoint splits alone do not prove new-series coverage.

Before the stacking study runs, use uniform pDC50 weights unless a convex option
reduces pooled held-out stacking RMSE by at least 1%, does not worsen macro-group
RMSE, and has a paired chemical-group bootstrap interval entirely below zero
(the report defines delta as candidate minus uniform). If multiple options meet
these criteria, prefer the strongest shrinkage. This is a conservative selection
rule, not an independent confirmation: trying multiple penalties on the same
groups makes the reported intervals exploratory, and bootstrapping fixed
out-of-fold predictions does not capture refitting variability. Interval
methodology and alpha must also be fixed before reading calibration outcomes.

The fixed input-dependent scale study **4711809** passed its synthetic checks
(**4711804**) and completed using stacking labels only. Against global-scale
group-maximum intervals, its mean width fell from 2.8478 to 2.5842 pDC50 units
(9.26%), while group-all coverage changed from 0.8889 to 0.8711. The paired
coverage difference was −0.0178 [−0.0444, 0.0089], and the width difference was
−0.2637 [−0.3436, −0.1868]. It narrowly misses the prespecified 10% width gate;
retain the simpler global scale provisionally. These are development estimates,
not final-calibration results.

A Dmax interval-loss XGBoost pilot **4711416** uses exact measurements as
zero-width intervals and censored measurements as bounds; its comparison with
fixed-parameter exact-label XGBoost is scheduled as **4711417**. This is a
development pilot, not a claim that censored labels or interval uncertainty
have improved. The new tools never treat a bound as an exact point label.

Both jobs completed. Interval-loss pooled CV RMSE was 0.2078 versus 0.2091 for
fixed-parameter exact-label XGBoost, improvement 0.0013 with chemical-group
bootstrap interval [−0.0022, 0.0052]. Source RMSE was 0.3245 versus 0.3095,
improvement −0.0150 [−0.0394, 0.0009] by publication bootstrap. This one-repeat
pilot does not support advancing the interval-loss recipe.

Activity source-weighting experiment **4711453** reuses the completed exact
CatBoost fold parameters and computes bounded VHL source-balancing weights from
each training fold only. Source and E3 identifiers do not enter its feature
matrix. Paired overall and E3 slice reports **4711454** and **4711455** depend
on completion and compare against the otherwise identical unweighted model.
The completed comparison found pooled CV AP 0.7664 versus 0.7684 unweighted
(difference −0.0020; paired 95% interval [−0.0074, 0.0031]). Source-holdout
AP was 0.4705 versus 0.4606 (difference 0.0100; interval [−0.0308, 0.0809]);
the VHL slice contained only 75 rows and its interval also crossed zero. This
does not establish a benefit from source weighting.

The fold-specific similarity diagnostic **4711526** computes each held-out
molecule's maximum chirality-aware radius-2 Morgan Tanimoto similarity to that
fold's labeled training molecules; these fingerprints are diagnostic only.
For provisional pDC50 r2, CV RMSE is 0.929 versus 0.907 for the baseline in the
0.3–0.5 similarity band (193 rows), with a wide paired interval crossing zero;
for repeated CV, the per-row diagnostic averages its fold-specific similarities.
At >0.7 similarity the point gains are clearer. The smallest source band has
only 14 rows. These are exploratory slices, not evidence of a hard applicability
threshold. Activity's report also completed; AP depends on prevalence and should
be compared between models within a band, not across bands. The other ongoing
jobs must be reviewed before any deployment claim.

A separate group-held-out conformal-method study (**4711558**, dependent on the
search-only pDC50 bundle) compares row-level and group-maximum quantiles on
stacking groups only. Its five synthetic tests passed (**4711530**). The new
locked internal-confirmation evaluator has passed four synthetic tests
(**4711564**), but has **not** been run against the real confirmation labels.
It creates a persistent marker immediately before reading those labels and
reports row and group-all coverage with cluster-bootstrap uncertainty.

The provisional pDC50 bundle **4711370** completed, followed by stacking study
**4711418** and conformal study **4711558**. Uniform pooling had held-out
stacking RMSE 0.7027; none of the convex strategies met the predeclared
improvement rule, so uniform is the provisional choice. At nominal 90%, the
development study estimated row/group-all coverage of 0.888/0.822 for row
quantiles versus 0.939/0.889 for group-maximum quantiles. Mean interval widths
were 2.22 and 2.85 pDC50 units respectively. The extra group-level protection
therefore has a substantial width cost; these are grouped development estimates,
not guarantees under distribution shift.
A 2.85-unit pDC50 interval corresponds to roughly a 700-fold DC50 range; the
practical value of that width needs explicit evaluation rather than treating
coverage alone as success.

The candidate artifact and locked-recipe format now support `group_max` as an
explicit regression calibration unit. Its scores are maxima within each
calibration group; the associated target is simultaneous coverage of all
observed rows in a new exchangeable group. The original row mode remains the
default. Focused group-calibration and compatibility checks **4711609** passed
25 tests. The final pDC50 interval method is not yet fixed and no real
calibration or confirmation labels have been evaluated for it.

Before inspecting the pending input-dependent scale comparison, advance that
experimental method only if its mean interval width is at least 10% lower than
the global-scale group-maximum method, its group-all coverage is no more than
two percentage points lower, and paired group-bootstrap intervals support a
width reduction while excluding a coverage loss larger than five points. This
is an exploratory development rule, not a coverage guarantee; both scales use
the same held-out group folds. Otherwise retain the simpler global scale.

The search-only 25-member pDC50 bundle occupies about 54 MiB. A CPU microbenchmark
(**4711634**) scored 128 distinct search molecules under one fixed context: cold
featurization plus prediction took 2.43 s, and warm-cache median/p95 took
47.8/50.0 ms. Model deserialization took 0.27 s and peak process RSS was about
775 MiB. This excludes Python startup, HTTP and RL; it is not a comparison on
matched hardware against the previous model. A real loopback latency probe
(**4711664**) uses a clearly marked temporary calibration derived only from
stacking groups to measure the service path; it is not a final artifact.
That loopback probe completed: startup was 2.42 s, the cold 128-molecule
request 2.27 s, and warm request median/p95 47.9/117.2 ms over ten repeats.
The p95 is based on only ten observations and excludes any RL workload.
A second temporary probe (**4711680**) exercised the new group-maximum interval
path through the live service: startup 2.52 s, cold 2.35 s, warm median/p95
48.9/130.0 ms. It used a disjoint split of stacking groups only and was deleted
after timing; these are not final calibration results.
A third probe (**4711750**) additionally called the actual isolated TACKOS HTTP
predictor against the same temporary group-calibrated server: warm prediction
median/p95 was 47.8/49.8 ms for 128 molecules over ten calls. This tests the
adapter on the provisional model, not a complete TACKOS training run.
