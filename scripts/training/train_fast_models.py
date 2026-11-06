"""Train and export the three fast TACK tasks using immutable HF context tables.

Run from any directory with the training image (see REPRODUCING.md). Pin both
--dataset-revision and --context-revision for an offline, reproducible experiment.
Context tables are read directly from the HF snapshot, without overwriting TACKAI_CACHE.

A common Butina test holdout is reserved first, then a scaffold-grouped calibration
holdout per task. Nested scaffold CV tunes ordinary RMSE or average precision;
early stopping replaces a separate search over tree counts by default. One repeat
of five folds and twenty trials is the default. The test split never selects parameters.

Every run exports ensembles/<task> locally, with model files, weights/calibration,
the exact feature layout, context revision and activity thresholds in config.json.
--push-to-hub optionally publishes those ensembles under task subfolders.
Activity means DC50 < 100 nM AND Dmax > 80%; incomplete labels can establish
inactivity but cannot establish activity. Old 1-uM activity runs must be retrained.

Use a new --output-dir when the provenance changes. Existing model/params pairs
are reused only when the recorded training and feature contracts agree.
"""
import os
import sys

# Set before importing either runtime; XGBoost's explicit n_jobs controls its own pool.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import hashlib
from importlib.metadata import version
import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "embeddings"))
from _dataset import REPO_ROOT, add_source_arguments, load_env, load_frames, source_record

# Load HF_TOKEN / TACKAI_CACHE from the repo's .env before any tackai/huggingface_hub call.
# Anchored on this file rather than the cwd so it also works from a SLURM job submitted
# from elsewhere; variables already in the environment take precedence.
load_env()

from tackai.fusion.data import (DEFAULT_CONTEXT_REPO, DMAX_THR, PDC50_THR, FusionData,
                                make_splits, scaffold_groups)
from tackai.fusion.mol_encoder import DESCRIPTOR_NAMES, MolEncoder
from tackai.fusion.xgb_ensemble import XGBStackedEnsemble
from tackai.training.splitting import DEFAULT_BUTINA_CUTOFF, butina_clusters

RDLogger.DisableLog("rdApp.*")   # an unparseable SMILES is handled, not reported per row

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TAIL = "low"  # which end of the regression target is the rare region
TAIL_Q = 0.10  # quantile defining it
TASKS = ("activity", "dmax", "pdc50")


def suggest_params(trial: "optuna.Trial", task: str, tune_n_estimators: bool = False) -> dict:
    p = dict(
        max_depth=trial.suggest_int("max_depth", 2, 6),
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        min_child_weight=trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 0.1, 100.0, log=True),
    )
    if tune_n_estimators:
        # Log scale: the useful range spans two orders of magnitude, and the step from 50 to
        # 100 trees matters far more than the one from 1900 to 2000. Searching it jointly with
        # learning_rate is the point — the two trade off against each other, and the
        # early-stopping alternative fixes the pair implicitly instead of scoring it.
        p["n_estimators"] = trial.suggest_int("n_estimators", 50, 2000, log=True)
    return p


def make_model(params: dict, task: str, n_estimators: int, seed: int, scale_pos_weight=1.0,
               n_jobs: int = 1, device: str = "cpu") -> xgb.XGBModel:
    """An XGBoost model for a task. ``params["n_estimators"]`` wins over the argument, so a
    tuned count is used as tuned and the argument stays the fallback for the
    early-stopping path."""
    kw = dict(params)
    kw.setdefault("n_estimators", n_estimators)
    kw.update(n_jobs=n_jobs, random_state=seed, tree_method="hist", device=device)
    if task == "activity":
        return xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric="aucpr", **kw)
    return xgb.XGBRegressor(eval_metric="rmse", **kw)


def tail_mask(y: np.ndarray, cut: float) -> np.ndarray:
    return y <= cut if TAIL == "low" else y >= cut


def regression_scores(y_true: np.ndarray, y_pred: np.ndarray, cut) -> dict:
    rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
    sc = dict(
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=rmse(y_true, y_pred),
        r2=float(r2_score(y_true, y_pred)),
    )
    if cut is not None:
        m = tail_mask(y_true, cut)
        sc["rmse_rare"] = rmse(y_true[m], y_pred[m]) if m.sum() > 1 else np.nan
        sc["rmse_common"] = rmse(y_true[~m], y_pred[~m]) if (~m).sum() > 1 else np.nan
    return sc


def binary_scores(y_true: np.ndarray, proba: np.ndarray, pred: np.ndarray) -> dict:
    return dict(
        roc_auc=float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) == 2 else np.nan,
        pr_auc=float(average_precision_score(y_true, proba)),
        f1=float(f1_score(y_true, pred, zero_division=0)),
        precision=float(precision_score(y_true, pred, zero_division=0)),
        recall=float(recall_score(y_true, pred, zero_division=0)),
    )


def objective(trial: "optuna.Trial", X: np.ndarray, y: np.ndarray, task: str, cut, seed: int,
              n_inner: int, tune_n_estimators: bool = False, *, groups: np.ndarray,
              n_jobs: int = 1, device: str = "cpu") -> float:
    """Inner-CV score of one hyper-parameter draw (lower is better).

    With ``tune_n_estimators`` the tree count is part of the search space and early stopping
    is switched off: the trial has to be scored at the number of trees it proposed, or the
    search would optimise a parameter that never reaches the model. The recorded
    ``n_estimators`` user attribute is then the suggested value rather than the mean stopping
    iteration, so the caller refits identically either way.
    """
    params = suggest_params(trial, task, tune_n_estimators)
    splitter = make_splits(y, groups, n_repeats=1, n_folds=n_inner, seeds=[seed])[0]
    scores, iters = [], []
    for tr, va in splitter:
        fit_kw = {} if tune_n_estimators else {"eval_set": [(X[va], y[va])]}
        model = make_model(params, task, 2000, seed, n_jobs=n_jobs, device=device)
        if not tune_n_estimators:
            model.set_params(early_stopping_rounds=50)
        model.fit(X[tr], y[tr], verbose=False, **fit_kw)
        if task == "activity":
            s = -average_precision_score(y[va], model.predict_proba(X[va])[:, 1])
        else:
            sc = regression_scores(y[va], model.predict(X[va]), cut)
            # Tune the metric we report. Tail scores remain diagnostics, not a different
            # objective which trades away accuracy on most of the population.
            s = sc["rmse"]
        scores.append(s)
        if not tune_n_estimators:
            iters.append(model.best_iteration + 1)
    trial.set_user_attr("n_estimators",
                        int(params["n_estimators"]) if tune_n_estimators else int(np.mean(iters)))
    return float(np.mean(scores))


def canonical_smiles(smiles: str):
    """Canonical SMILES of a molecule, or ``None`` when RDKit cannot parse it."""
    try:
        mol = Chem.MolFromSmiles(str(smiles))
    except Exception:
        return None
    return Chem.MolToSmiles(mol) if mol is not None else None


def test_overlap_smiles(dev_frames, test_file) -> set:
    """Raw SMILES strings in the development tables whose molecule also occurs in the test CSV.

    Three identities are unioned, because each catches cases the others miss:

    * **raw SMILES** — the only handle on a molecule RDKit cannot parse, and 158 rows of
      ``AutoTPD-PROTAC`` are unparseable;
    * **canonical SMILES** — catches the same structure written differently;
    * **Connectivity_Key** — catches salts, tautomers and stereoisomers of one connectivity,
      which canonicalisation keeps apart.

    Returned as raw strings so the result can be handed to
    ``FusionData.from_csv(exclude_smiles=...)``, which matches on the table's own SMILES
    column. Every row of an excluded molecule goes, not only the matching one.

    Args:
        dev_frames: Development tables as DataFrames.
        test_file: Held-out test CSV path.

    Returns:
        Set of raw SMILES strings to exclude.
    """
    # str() per value rather than .astype(str).str.strip(): the CSVs disagree on the dtype of
    # these columns, and .str accessors yield NaN (a float) for a non-string entry.
    def strings(frame, column):
        if column not in frame:
            return pd.Series("", index=frame.index, dtype=object)
        return frame[column].map(lambda v: "" if pd.isna(v) else str(v).strip())

    test = pd.read_csv(test_file, low_memory=False)
    test_raw = {s for s in strings(test, "SMILES") if s}
    test_canon = {c for c in (canonical_smiles(s) for s in test_raw) if c}
    test_conn = {c for c in strings(test, "Connectivity_Key") if c}

    excluded, by_key = set(), {"raw": 0, "canonical": 0, "connectivity": 0}
    for dev in dev_frames:
        smiles = strings(dev, "SMILES")
        conn = strings(dev, "Connectivity_Key")
        for i, s in smiles.items():
            if not s or s in excluded:
                continue
            if s in test_raw:
                excluded.add(s); by_key["raw"] += 1
                continue
            canon = canonical_smiles(s)
            if canon and canon in test_canon:
                excluded.add(s); by_key["canonical"] += 1
                continue
            if test_conn and conn.get(i, "") in test_conn:
                excluded.add(s); by_key["connectivity"] += 1
    logger.info("test overlap: %d distinct molecule(s) to exclude "
                "(matched by raw SMILES %d, canonical %d, connectivity key %d)",
                len(excluded), by_key["raw"], by_key["canonical"], by_key["connectivity"])
    return excluded


def butina_test_mask(smiles: Sequence[str], frac: float, cutoff: float, seed: int) -> np.ndarray:
    """Boolean mask of the rows held out as the test set by Taylor-Butina clustering.

    Whole clusters are taken, in a seeded random order, until ``frac`` of the rows are
    reserved; the last cluster is taken whole even if that overshoots, because splitting one
    would put near-neighbours of a test molecule into training, which is exactly what the
    clustering is there to prevent. Rows are the unit of the target fraction, not molecules:
    one molecule contributes as many rows as it has measured contexts.

    Args:
        smiles: Per-row SMILES of the development table (duplicates expected).
        frac: Target fraction of rows to hold out.
        cutoff: Taylor-Butina distance cutoff (1 - Tanimoto similarity).
        seed: Seed of the cluster order, and so of the split.

    Returns:
        Boolean array, ``True`` for a test row.
    """
    clusters = butina_clusters(list(smiles), cutoff=cutoff)
    rng = np.random.default_rng(seed)
    order = rng.permutation(np.unique(clusters))
    target = frac * len(clusters)
    mask = np.zeros(len(clusters), dtype=bool)
    for cid in order:
        if mask.sum() >= target:
            break
        mask |= clusters == cid
    logger.info("Butina test split (cutoff=%.2f, seed=%d): %d of %d rows held out (%.1f%%) in "
                "%d of %d clusters", cutoff, seed, int(mask.sum()), len(mask),
                100 * mask.mean(), len(set(clusters[mask])), len(set(clusters)))
    return mask


def test_scaffold_overlap_smiles(dev_frames: Sequence[pd.DataFrame], test_file: str) -> set[str]:
    """Development molecules sharing a generic Murcko scaffold with an external test row."""
    dev = set().union(*(set(f.SMILES.dropna().astype(str).str.strip()) for f in dev_frames)) - {""}
    test = set(pd.read_csv(test_file).SMILES.dropna().astype(str).str.strip()) - {""}
    _, info = scaffold_groups(sorted(dev | test))
    scaffolds = info["scaffold"]
    held_out = {scaffolds[s] for s in test}
    excluded = {s for s in dev if scaffolds[s] in held_out}
    logger.info("test scaffold overlap: excluding %d development molecules", len(excluded))
    return excluded


def holdout_calibration(y: np.ndarray, groups: np.ndarray, frac: float, seed: int):
    """Split a task's rows into (CV-training, calibration) without splitting a scaffold group.

    The calibration set has to be carved out *before* the cross-validation, not after: every
    row the 5x5 CV covers is in-train for about 20 of the 25 members, so calibrating on it
    would fit the stacking weights and the conformal quantile on values the members have
    already memorised — narrow intervals that do not hold on new molecules. Grouping by
    scaffold matters for the same reason at one remove: a near-duplicate scaffold either side
    of the split leaks just as effectively as the row itself.

    Args:
        y: Task targets, used only for stratification.
        groups: Scaffold group id per row.
        frac: Target fraction of the rows to hold out; realised as one fold of
            ``round(1 / frac)`` grouped, stratified folds, so the achieved fraction is the
            closest a whole number of folds allows (0.1 -> 10 folds, 0.33 -> 3).
        seed: Seed of the split.

    Returns:
        ``(train_idx, calib_idx)`` index arrays into the task's own rows.
    """
    n_folds = max(2, int(round(1.0 / frac)))
    train_idx, calib_idx = make_splits(y, groups, n_repeats=1, n_folds=n_folds, seeds=[seed])[0][0]
    return train_idx, calib_idx


def build_ensemble(task_dir: Path, X_cal, y_cal, seed: int,
                   groups_cal=None, weighting: str = "uniform") -> XGBStackedEnsemble:
    """Assemble a task's trained folds into an ensemble, calibrated when a holdout is given.

    The two steps must not share rows — the conformal quantile has to be measured on
    residuals of an ensemble whose weights are already fixed — so the calibration holdout is
    halved between them. Without a holdout the members keep uniform weights and the ensemble
    is returned uncalibrated: ``predict`` then reports model disagreement only, with no
    conformal interval.

    Args:
        task_dir: Directory of the task's ``*.json`` fold models.
        X_cal: Calibration design matrix, or ``None`` for no calibration.
        y_cal: Calibration targets, or ``None``.
        seed: Seed of the fit/calibrate halving.

    Returns:
        The assembled :class:`XGBStackedEnsemble`.
    """
    ensemble = XGBStackedEnsemble.from_pretrained(str(task_dir))
    if X_cal is None or not len(y_cal):
        logger.warning("No calibration holdout: the ensemble keeps uniform weights and is left "
                       "uncalibrated (no conformal quantile / temperature).")
        return ensemble
    if groups_cal is None:
        raise ValueError("Calibration scaffold groups are required to prevent leakage.")
    fit, conf = next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
                     .split(X_cal, y_cal, groups_cal))
    X_fit, X_conf, y_fit, y_conf = X_cal[fit], X_cal[conf], y_cal[fit], y_cal[conf]
    logger.info("Calibrating %d-member ensemble: %d rows to fit the stacking weights, %d to "
                "calibrate", len(ensemble.models), len(y_fit), len(y_conf))
    if weighting == "stacked":
        ensemble.fit(X_fit, y_fit)
    elif weighting == "uniform":
        if ensemble.task == "regression":
            residuals = ensemble._base_predictions(X_fit) - y_fit[:, None]
            ensemble.sigmas_ = np.maximum(np.sqrt(np.mean(residuals**2, axis=0)), 1e-12)
    else:
        raise ValueError(f"Unknown ensemble weighting {weighting!r}")
    ensemble.calibrate(X_conf, y_conf)
    return ensemble


def check_provenance(path: Path, record: dict) -> None:
    """Write this run's provenance, or refuse to resume models trained on different inputs.

    A re-run skips every fold whose files exist, so new folds would otherwise join models
    trained on other tables or rows. A revision recorded as ``"main"`` could not be resolved
    (no network), so it is not compared; pin ``--dataset-revision`` to compare it offline too.

    Args:
        path: The output directory's ``provenance.json``.
        record: What this run is using.

    Raises:
        SystemExit: If the directory holds models trained on different inputs.
    """
    if path.exists():
        old = json.loads(path.read_text())
        # Earlier v2 runs always used molecule-only external-test exclusion.
        for item in (old, record):
            if "training" in item:
                item["training"] = {"exclude_test_scaffolds": False, **item["training"]}
        # .get, because a directory written before a key existed is still resumable: a
        # missing key compares unequal only if the new run actually sets it differently.
        keys = ["context_fingerprint", "protein_space", "test_split", "fingerprint",
                "training", "activity_thresholds", "dataset_identity", "test_sha256"]
        revisions = old["source"]["revision"], record["source"]["revision"]
        changed = [k for k in keys if old.get(k) != record.get(k)]
        if "main" not in revisions and revisions[0] != revisions[1]:
            changed.append("dataset revision")
        if changed:
            raise SystemExit(f"{path.parent} holds models trained on different inputs "
                             f"({', '.join(changed)} changed). Use a new --output-dir.")
    path.write_text(json.dumps(record, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train fusion-surrogate XGBoost models with 5x5 CV and per-fold Optuna tuning.",
    )
    add_source_arguments(parser)
    parser.add_argument("--test-file", default=None,
                         help="Held-out test CSV for the reported test metrics and the overlap "
                              "exclusion. Without it the test set is carved out of the "
                              "development rows by Taylor-Butina clustering instead "
                              "(--test-frac, --butina-cutoff, --butina-seed).")
    parser.add_argument("--output-dir", required=True,
                         help="Directory to save trained models, their params, and fold_metrics.csv.")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS),
                         help="Tasks to train (default: all of activity, dmax, pdc50).")
    parser.add_argument("--protein-space", default="combined", choices=["per_block", "combined"],
                         help="Passed to ContextEncoder.")
    parser.add_argument("--fp-size", type=int, default=2024,
                        help="Morgan fingerprint length (default: 2024, the workspace convention). "
                             "Chirality is always included; geometry is restored from ensemble metadata.")
    parser.add_argument("--fp-radius", type=int, default=8,
                        help="Morgan fingerprint radius (default: 8).")
    parser.add_argument("--include-ipc", action=argparse.BooleanOptionalAction, default=False,
                        help="Include the numerically unstable Ipc/AvgIpc descriptors for legacy "
                             "reproduction (default: excluded from new training).")
    parser.add_argument("--featurize-workers", type=int, default=1,
                        help="Worker processes for molecular featurisation (1 = in-process). "
                             "Featurisation is a one-off cost per run, cached under "
                             "TACKAI_CACHE and keyed by the fingerprint settings.")
    parser.add_argument("--n-repeats", type=int, default=1, help="Outer CV repeats (default: 1).")
    parser.add_argument("--n-folds", type=int, default=5, help="Outer CV folds per repeat.")
    parser.add_argument("--n-trials", type=int, default=20, help="Optuna trials per fold.")
    parser.add_argument("--ensemble-weighting", choices=["uniform", "stacked"], default="uniform",
                        help="Uniform averaging (default) or learned mixture weights.")
    parser.add_argument("--n-inner", type=int, default=3, help="Inner CV folds used inside the Optuna objective.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--n-jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
                        help="XGBoost threads per fit (default: Slurm allocation, otherwise 1).")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                        help="XGBoost training device; cuda requires a GPU allocation and --nv.")
    parser.add_argument("--tune-n-estimators", dest="tune_n_estimators",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="Let Optuna search the number of trees instead of fixing it by "
                             "early stopping on the inner validation fold (default: off). "
                             "--tune-n-estimators enables the separate tree-count search.")
    parser.add_argument("--test-frac", type=float, default=0.1,
                        help="Only without --test-file: fraction of the development rows held "
                             "out as the test set, taken as whole Butina clusters "
                             "(default: 0.1). 0 trains on every row and reports no test "
                             "metrics.")
    parser.add_argument("--butina-cutoff", type=float, default=DEFAULT_BUTINA_CUTOFF,
                        help="Taylor-Butina distance cutoff (1 - Tanimoto similarity) used to "
                             "cluster development molecules. Smaller values form tighter "
                             "clusters; this is not an all-pairs similarity exclusion.")
    parser.add_argument("--butina-seed", type=int, default=42,
                        help="Seed deciding which Butina clusters become the test set. Fixed "
                             "independently of --seed so the test set is identical across "
                             "model and feature-set experiments.")
    parser.add_argument("--calibration-frac", type=float, default=0.1,
                        help="Fraction of the remaining development rows held out, grouped by "
                             "scaffold, as the calibration set that fits the ensemble's "
                             "stacking weights and conformal quantile; the CV members never "
                             "see it (default: 0.1). 0 disables the holdout, and then no "
                             "calibration is performed at all.")
    parser.add_argument("--calibration-seed", type=int, default=9000,
                        help="Seed of the calibration holdout split. Deliberately outside the "
                             "outer-CV repeat seeds (1000..1004) so the holdout is not a copy "
                             "of one of the CV folds.")
    parser.add_argument("--exclude-test-overlap", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Drop development rows whose molecule also appears in --test-file, "
                             "matched on raw SMILES, canonical SMILES and Connectivity_Key "
                             "(default: on).")
    parser.add_argument("--exclude-test-scaffolds", action=argparse.BooleanOptionalAction,
                        default=False, help="Also exclude development molecules sharing a generic "
                        "Murcko scaffold with --test-file (default: off). Useful when expanding "
                        "the training source around a previously fixed test set.")
    parser.add_argument("--context-repo-id", default=DEFAULT_CONTEXT_REPO,
                         help="Hub dataset repo to download the context tables from.")
    parser.add_argument("--context-revision", default=None,
                         help="Revision of --context-repo-id (default: the main branch).")
    parser.add_argument("--push-to-hub", default=None, metavar="REPO_ID",
                        help="Publish the locally exported "
                             "XGBStackedEnsemble, calibrate it on the calibration holdout, "
                             "save it under <output-dir>/ensembles/<task> and push it to this "
                             "Hugging Face Hub repo id under the subfolder <task>. The repo is "
                             "created private unless --no-private is given.")
    parser.add_argument("--private", action=argparse.BooleanOptionalAction, default=True,
                        help="Create the --push-to-hub repo as private (default: on).")
    args = parser.parse_args()
    if not 0.0 <= args.test_frac < 1.0:
        raise SystemExit(f"--test-frac must be in [0, 1), got {args.test_frac}")
    if not 0.0 <= args.calibration_frac < 1.0:
        raise SystemExit(f"--calibration-frac must be in [0, 1), got {args.calibration_frac}")
    if min(args.n_repeats, args.n_trials, args.n_jobs) < 1 or min(args.n_folds, args.n_inner) < 2:
        raise SystemExit("repeats/trials/jobs must be positive; folds/inner folds must be >= 2")
    if args.exclude_test_scaffolds and not args.test_file:
        raise SystemExit("--exclude-test-scaffolds requires --test-file")
    return args


def main():
    args = parse_args()
    selected_descriptors = [name for name in DESCRIPTOR_NAMES
                            if args.include_ipc or name not in ("Ipc", "AvgIpc")]
    if not os.environ.get("HF_TOKEN"):
        logger.warning("HF_TOKEN is not set (looked in %s and the cwd); private Hub repos will fail.",
                       REPO_ROOT / ".env")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Read the immutable Hub snapshot directly: do not overwrite the shared embedding
    # cache, and do not force downloads on offline compute nodes.
    from huggingface_hub import snapshot_download
    context_path = Path(args.context_repo_id)
    if not context_path.is_dir():
        context_path = Path(snapshot_download(args.context_repo_id, repo_type="dataset",
                                             revision=args.context_revision))
    context_revision = context_path.name
    ctx = FusionData.from_pretrained(context_path, cache_dir=context_path,
                                     protein_space=args.protein_space).encoder
    dev_frames = load_frames(args)
    check_provenance(outdir / "provenance.json", {
        "source": source_record(args),
        "context_repo": args.context_repo_id,
        "context_revision": context_revision,
        "context_fingerprint": ctx.fingerprint(),
        "protein_space": args.protein_space,
        # The models' input width: mixing two fingerprint geometries in one directory would
        # produce members that cannot be stacked, so a resume compares this.
        "fingerprint": {"radius": args.fp_radius, "size": args.fp_size,
                        "include_chirality": True},
        "test_file": args.test_file,
        "test_sha256": (hashlib.sha256(Path(args.test_file).read_bytes()).hexdigest()
                        if args.test_file else None),
        "dataset_identity": {"repo": args.dataset, "config": args.dataset_config,
                             "files": args.dataset_files,
                             "frames": [hashlib.sha256(pd.util.hash_pandas_object(f, index=True)
                                                       .values.tobytes()).hexdigest()
                                        for f in dev_frames]},
        "activity_thresholds": {"dmax": DMAX_THR, "pdc50": PDC50_THR},
        "training": {"version": 2, "inner_split": "scaffold", "metric": "rmse_or_ap",
                     "seed": args.seed, "n_repeats": args.n_repeats, "n_folds": args.n_folds,
                     "device": args.device,
                     "n_inner": args.n_inner, "n_trials": args.n_trials,
                     "tune_n_estimators": args.tune_n_estimators,
                     "calibration_frac": args.calibration_frac,
                     "calibration_seed": args.calibration_seed,
                     "ensemble_weighting": args.ensemble_weighting,
                     "exclude_test_overlap": args.exclude_test_overlap,
                     "exclude_test_scaffolds": args.exclude_test_scaffolds,
                     "versions": {p: version(p) for p in ("xgboost", "scikit-learn", "rdkit", "numpy")},
                     "descriptors": selected_descriptors},
        "test_split": ({"source": "file", "file": args.test_file} if args.test_file else
                       {"source": "butina", "frac": args.test_frac,
                        "cutoff": args.butina_cutoff, "seed": args.butina_seed}),
    })

    # One featuriser for both tables: the fingerprint geometry is part of the design matrix,
    # so a test table built with another one is not scoreable by these models. MolEncoder
    # always includes chirality.
    logger.info("Molecular features: Morgan radius %d, %d bits, chirality included, "
                "+ %d RDKit descriptors", args.fp_radius, args.fp_size, len(selected_descriptors))
    featurizer = MolEncoder(radius=args.fp_radius, fp_size=args.fp_size,
                            n_workers=args.featurize_workers, descriptors=selected_descriptors)

    exclude = (test_overlap_smiles(dev_frames, args.test_file)
               if args.test_file and args.exclude_test_overlap else None)
    if args.exclude_test_scaffolds:
        exclude = set(exclude or ()) | test_scaffold_overlap_smiles(dev_frames, args.test_file)
    dev_dt = FusionData.from_csv(dev_frames, encoder=ctx, featurizer=featurizer,
                                 exclude_smiles=exclude, verbose=True)
    test_dt = (FusionData.from_csv([args.test_file], encoder=ctx, featurizer=featurizer,
                                   verbose=True)
               if args.test_file else None)
    (outdir / "data_summary.json").write_text(json.dumps({
        "source_rows": sum(len(f) for f in dev_frames),
        "encoded_rows": len(dev_dt.table), "excluded_overlap_rows": dev_dt.excluded,
        "block_dims": dev_dt.dims,
        "tasks": {task: {"n_labels": int(np.isfinite(dev_dt.target(task)).sum()),
                         "target_mean": float(np.nanmean(dev_dt.target(task)))}
                  for task in args.tasks},
    }, indent=2))

    # No test file: carve the test set out of development by clustering the molecules, before
    # the calibration holdout and the CV see any row. Clustering once over the whole
    # development table (rather than per task) keeps one test set across tasks, which is what
    # makes the three tasks' test metrics comparable -- and what the `activity` target, a
    # function of the other two, requires to stay consistent.
    if test_dt is None and args.test_frac > 0:
        logger.info("No --test-file: deriving the test split from the development molecules")
        test_mask = butina_test_mask(dev_dt.smiles, args.test_frac, args.butina_cutoff,
                                     args.butina_seed)
    else:
        test_mask = None
        if test_dt is None:
            logger.warning("No --test-file and --test-frac 0: no test metrics, and no molecule "
                           "is reserved, so evaluating on a test set later may include training "
                           "molecules.")

    records = []

    for task in args.tasks:
        task_dir = outdir / task
        task_dir.mkdir(parents=True, exist_ok=True)

        row_idx, X_task, y_task, groups_task = dev_dt.task_rows(task=task)

        # Step 1: the test rows, reserved first and never seen again below. ``pool_rows``
        # keeps the development row index of every row that survives, so the calibration npz
        # records indices into the development table rather than into this pool.
        if test_dt is not None:
            test_rows, X_test, y_test, test_groups = test_dt.task_rows(task=task)
            np.savez(task_dir / "test.npz", X=X_test, y=y_test,
                     groups=test_groups.astype(str), row_index=test_rows,
                     feature_names=np.array(test_dt.feature_names),
                     **({"benchmark_row_id": test_dt.table.benchmark_row_id.to_numpy()[test_rows]}
                        if "benchmark_row_id" in test_dt.table else {}),
                     scaffold_keys=np.array([test_dt.scaffold_info["scaffold"][s]
                                             for s in test_dt.smiles[test_rows]], dtype=str))
            X_all, y_all, groups_all, pool_rows = X_task, y_task, groups_task, row_idx
        elif test_mask is not None:
            is_test = test_mask[row_idx]
            X_test, y_test = X_task[is_test], y_task[is_test]
            X_all, y_all, groups_all = (X_task[~is_test], y_task[~is_test], groups_task[~is_test])
            pool_rows = row_idx[~is_test]
            np.savez(task_dir / "test.npz", X=X_test, y=y_test,
                     groups=groups_task[is_test].astype(str), row_index=row_idx[is_test],
                     feature_names=np.array(dev_dt.feature_names),
                     scaffold_keys=np.array([dev_dt.scaffold_info["scaffold"][s]
                                             for s in dev_dt.smiles[row_idx[is_test]]], dtype=str))
            logger.info("%s: %d of %d rows held out as the Butina test set", task,
                        len(y_test), len(y_task))
        else:
            X_test = y_test = None
            X_all, y_all, groups_all, pool_rows = X_task, y_task, groups_task, row_idx

        # Step 2: carve the calibration set out of what is left, before cross-validating, so
        # the ensemble's stacking weights and conformal quantile are fitted on rows no member
        # was trained on. Saved next to the models for calibrate_fast_models.py.
        if args.calibration_frac > 0:
            cv_idx, calib_idx = holdout_calibration(y_all, groups_all, args.calibration_frac,
                                                    args.calibration_seed)
            np.savez(task_dir / "calibration.npz", X=X_all[calib_idx], y=y_all[calib_idx],
                     groups=groups_all[calib_idx].astype(str), row_index=pool_rows[calib_idx],
                     scaffold_keys=np.array([dev_dt.scaffold_info["scaffold"][s]
                                             for s in dev_dt.smiles[pool_rows[calib_idx]]], dtype=str))
            logger.info("%s: %d development rows -> %d for CV, %d held out for calibration "
                        "(%d scaffold groups, disjoint from CV)", task, len(y_all), len(cv_idx),
                        len(calib_idx), len(set(groups_all[calib_idx])))
        else:
            cv_idx = np.arange(len(y_all))
            calib_idx = np.array([], dtype=int)
            logger.info("%s: %d development rows, no calibration holdout and no calibration",
                        task, len(y_all))

        # Step 3: train on the rest.
        X, y, groups = X_all[cv_idx], y_all[cv_idx], groups_all[cv_idx]
        np.savez(task_dir / "split_rows.npz", cv_row_index=pool_rows[cv_idx],
                 calibration_row_index=pool_rows[calib_idx])
        # The tail/cut-based rare-vs-common RMSE split only applies to dmax (a bounded
        # fraction, where a tail quantile is meaningful); skip it for activity and pdc50.
        cut = float(np.quantile(y, TAIL_Q if TAIL == "low" else 1 - TAIL_Q)) if task == "dmax" else None

        # Split the CV-training rows only: dev_dt.splits() would span the calibration rows too.
        cv_splits = make_splits(y, groups, n_repeats=args.n_repeats, n_folds=args.n_folds)
        for i, repeat in enumerate(cv_splits):
            for j, (train_idx, val_idx) in enumerate(repeat):
                key = f"{task}_repeat{i}_fold{j}"
                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]

                model_file = task_dir / f"{key}.json"
                params_file = task_dir / f"{key}.params.json"
                if model_file.exists() and params_file.exists():
                    logger.info("Loading existing model for %s from %s", key, model_file)
                    model = xgb.XGBClassifier() if task == "activity" else xgb.XGBRegressor()
                    model.load_model(model_file)
                    with open(params_file, "r") as f:
                        params_data = json.load(f)
                    best = params_data.get("best_params", {})
                else:
                    logger.info("Training new model for %s", key)
                    seed = args.seed + 100 * i + j
                    study = optuna.create_study(
                        direction="minimize",
                        sampler=optuna.samplers.TPESampler(seed=seed),
                    )
                    baseline = dict(max_depth=3, learning_rate=0.05, subsample=1.0,
                                    colsample_bytree=1.0, min_child_weight=5.0, reg_lambda=10.0)
                    if args.tune_n_estimators:
                        baseline["n_estimators"] = 500
                    study.enqueue_trial(baseline)
                    study.optimize(
                        lambda t: objective(t, X_train, y_train, task, cut, seed, args.n_inner,
                                            args.tune_n_estimators, groups=groups[train_idx],
                                            n_jobs=args.n_jobs, device=args.device),
                        n_trials=args.n_trials, show_progress_bar=sys.stderr.isatty(),
                    )
                    best = study.best_params
                    n_est = study.best_trial.user_attrs["n_estimators"]

                    model = make_model(best, task, n_est, seed, n_jobs=args.n_jobs,
                                       device=args.device).fit(X_train, y_train)

                    model.save_model(model_file)
                    params_file.write_text(json.dumps(
                        dict(best_params=best, n_estimators=n_est,
                             inner_score=study.best_value, cut=cut), indent=2))

                # "calib" is the held-out calibration set: no member was trained on it, so a
                # per-member score there is the honest one to compare against "val".
                splits = [("val", X_val, y_val)]
                if X_test is not None and len(y_test):
                    splits.append(("test", X_test, y_test))
                if len(calib_idx):
                    splits.append(("calib", X_all[calib_idx], y_all[calib_idx]))
                for split, Xe, ye in splits:
                    if task == "activity":
                        proba = model.predict_proba(Xe)[:, 1]
                        pred = (proba >= 0.5).astype(int)
                        sc = binary_scores(ye, proba, pred)
                    else:
                        pred = model.predict(Xe)
                        sc = regression_scores(ye, pred, cut)
                    records.append(dict(task=task, repeat=i, fold=j, split=split,
                                         n_train=len(train_idx),
                                         n_estimators=model.get_booster().num_boosted_rounds(),
                                         **sc))
                # Keep completed folds reviewable even if an allocation ends mid-task.
                metrics_tmp = outdir / ".fold_metrics.csv.tmp"
                pd.DataFrame(records).to_csv(metrics_tmp, index=False)
                metrics_tmp.replace(outdir / "fold_metrics.csv")
                logger.info("Completed %s (%d trees)", key, model.get_booster().num_boosted_rounds())

        # Always export a locally loadable ensemble; publishing is optional.
        X_cal = X_all[calib_idx] if len(calib_idx) else None
        y_cal = y_all[calib_idx] if len(calib_idx) else None
        ensemble = build_ensemble(task_dir, X_cal, y_cal, args.seed,
                                  groups_all[calib_idx] if len(calib_idx) else None,
                                  weighting=args.ensemble_weighting)
        ensemble.target_name = task
        run_record = json.loads((outdir / "provenance.json").read_text())
        ensemble.training_provenance = {
            "source": run_record["source"],
            "source_content_hashes": run_record["dataset_identity"]["frames"],
            "training": {k: v for k, v in run_record["training"].items() if k != "descriptors"},
        }
        ensemble.feature_spec = {
            "fingerprint": {"radius": args.fp_radius, "size": args.fp_size},
            "descriptors": featurizer.descriptors,
            "context_repo": args.context_repo_id,
                "context_revision": context_revision,
            "context_fingerprint": ctx.fingerprint(),
            "protein_space": ctx.protein_space, "block_dims": dev_dt.dims,
            "activity_thresholds": {"dmax": DMAX_THR, "pdc50": PDC50_THR},
        }
        # Saved outside task_dir: XGBStackedEnsemble._from_pretrained reads every .json in
        # a directory as a member, so an ensemble saved beside the fold models would be
        # reloaded as a double-counted mixture of the two.
        ensemble_dir = outdir / "ensembles" / task
        ensemble.save_pretrained(ensemble_dir)
        logger.info("Saved the %s ensemble to %s", task, ensemble_dir)
        reloaded = XGBStackedEnsemble.from_pretrained(str(ensemble_dir))
        if X_test is not None and len(y_test):
            output = reloaded.predict(X_test)
            pred = output["proba" if task == "activity" else "mean"]
            np.savez(ensemble_dir / "test_predictions.npz", y=y_test, prediction=pred)
            sc = (binary_scores(y_test, pred, (pred >= 0.5).astype(int)) if task == "activity"
                  else regression_scores(y_test, pred, cut))
            records.append(dict(task=task, split="ensemble_test", **sc))
        if args.push_to_hub:
            logger.info("Pushing %s to %s/%s", task, args.push_to_hub, task)
            ensemble.push_to_hub(args.push_to_hub, private=args.private, subfolder=task)

    results = pd.DataFrame(records)
    results.to_csv(outdir / "fold_metrics.csv", index=False)
    logger.info("Wrote %s", outdir / "fold_metrics.csv")


if __name__ == "__main__":
    main()
