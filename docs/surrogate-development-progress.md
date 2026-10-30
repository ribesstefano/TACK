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
