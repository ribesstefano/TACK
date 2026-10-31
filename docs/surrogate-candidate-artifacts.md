# Research candidate artifacts

This path is for explicitly selected, locally trained research candidates. It
does not select a model, download an artifact or replace the published ensembles.
The candidate is not ready for production until the locked evaluation, calibration,
latency and downstream checks in the development plan have passed.

`scripts/fit_surrogate_candidate.py` takes a frozen snapshot and a JSON recipe.
The recipe pins the endpoint, molecular feature matrix, label policy, shared
grouped fold schedule, a learner and parameter overrides for every member,
stacking strategy and calibration alpha. Each base member trains only on search
rows. The optional convex weights and empirical regression residual scale use
the reserved stacking groups. The independent calibration groups fit the
conformal quantile or classification temperature. Confirmation labels are not
read by the fit tool.

Defaults not listed in `params` are defined by the fit tool; its source hash and
the dependency versions are recorded with the artifact.

A compact schema illustration follows. Its deliberately tiny fold/estimator
counts are for understanding the format, not a recommended training recipe:

```json
{
  "schema_version": 1,
  "task": "dmax",
  "feature": "baseline",
  "label_policy": "exact",
  "cv": {"repeats": 1, "folds": 2, "seeds": [20261030]},
  "members": [
    {"family": "extratrees", "repeat": 0, "fold": 0, "seed": 11,
     "params": {"n_estimators": 10, "min_samples_leaf": 3}},
    {"family": "extratrees", "repeat": 0, "fold": 1, "seed": 12,
     "params": {"n_estimators": 10, "min_samples_leaf": 3}}
  ],
  "stacking": {"weights": "uniform", "l2": 0.1},
  "calibration": {"alpha": 0.1}
}
```

For a locked recipe, run on a CPU allocation from the repository root:

```bash
env -u PYTHONPATH OMP_NUM_THREADS=1 .venv/bin/python scripts/fit_surrogate_candidate.py \
  --snapshot models/research_v1 --recipe /path/to/locked-recipe.json \
  --output /path/to/new-candidate/dmax
```

The output directory must not already exist. A parent directory may contain
separate `dmax/`, `pdc50/` and `activity/` candidates, each with its own molecular
specification. Save the frozen context tables with the candidates and retain the
snapshot and recipe hashes in the experiment record. Predictions from a single
member have no between-model disagreement; the empirical residual scale is not
a learned decomposition of biological noise.

To test a set of calibrated **trusted local** candidate artifacts, the server can
use their separate molecular features while enforcing the saved context fingerprint:

```bash
tack serve --candidate-dir /path/to/new-candidate --local-context \
  --protein-space combined --context /path/to/context.json \
  --port 0 --port-file /path/to/tack.port
```

`TACKAI_CACHE` must point at the exact frozen context tables. The server reports
per-task matrix widths under `/info`, shares featurization among identical
molecular specifications, returns regression `mean` and classification `proba`,
and keeps invalid SMILES aligned as JSON `null`. Use a loopback address, wait for
`/health`, and clean up the service process and port file from the Slurm job.

Candidate files contain joblib/pickle data. The recorded hash detects accidental
changes, but does not authenticate a producer: load artifacts only from a
trusted source. Model package and RDKit version checks are enforced on load.
The existing `--repo-id` and `--checkpoints-dir` paths remain available for the
original XGBoost ensembles.
