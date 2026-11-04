#!/usr/bin/env python
"""Train one (task, dataset) cell of the five-training-set PROTAC dataset comparison (v2).

v2 of ``train_dataset_comparison.py``, kept as a separate file so the original
run (script, notebook, checkpoints, predictions under ``outputs/dataset_comparison/``)
stays untouched as a reference. Two changes, both aimed at the root causes found
when the v1 results were analyzed (see ``notebooks/dataset_comparison_analysis_v1.ipynb``):
test performance was dominated by cross-paper target-prior mismatch and a mostly-unseen
Assay feature, not by which training set was used.

1. **Paper-grouped tuning validation split.** ``--tune``'s per-seed validation carve
   used to be a random ROW split, so ~94% of validation rows shared a paper (DOI)
   with the training rows -- Optuna was selecting hyperparameters for same-paper
   interpolation, while the held-out test set is entirely out-of-paper. The split
   is now grouped by DOI (``sklearn.model_selection.GroupShuffleSplit``), so no
   paper appears on both sides.
2. **Coarse Assay-family remapping.** The raw ``Assay`` column has ~50 near-duplicate
   free-text spellings; one-hot encoding it meant 60% of test rows (mostly
   "capillary electrophoresis", never seen in training) got an all-zero context
   vector. ``assay_family()`` maps every raw value onto one of four families
   (Western-Immunoblot / ELISA-MSD / Luminescence-Reporter / Fluorescence-Flow) or
   "Unknown", applied identically to every training pool and the test set before
   featurization, so the encoder actually has training examples for what the test
   set's dominant assay type methodologically is.

Otherwise identical to v1: same five named training pools (``DATASETS``) plus the
downsampled arm (``DOWNSAMPLE_ARMS``), same per-seed checkpointing under
``<out-dir>/.seed_cache/`` (``--no-resume`` to ignore it), same feature/model config.

Usage:
    # one cell of the experiment
    python scripts/train_dataset_comparison_v2.py --task bin --dataset PROTAC-DB --tune

    # the size-matched downsampled arm (requires --tune; see DOWNSAMPLE_ARMS)
    python scripts/train_dataset_comparison_v2.py --task bin --dataset DEV-PMC-LLM-downsampled --tune

    # populate the shared embedding cache before launching a parallel array
    python scripts/train_dataset_comparison_v2.py --warmup-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from datasets import Dataset, DatasetDict
from sklearn.model_selection import GroupShuffleSplit

from tackai.data.datamodule import DegradationComplexDataModule
from tackai.data.tasks import load_task_dataset
from tackai.training.xgboost_models import train_xgboost, tune_xgboost_hyperparameters

warnings.filterwarnings("ignore")
logger = logging.getLogger("train_dataset_comparison_v2")

# --------------------------------------------------------------------------
# Experiment definition
# --------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "sets"
YAOCHEN_DIR = REPO_ROOT / "data" / "yaochen"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "dataset_comparison_v2"
TMP_DIR = DEFAULT_OUT_DIR / ".tmp"

# Whiteboard sets #1-#5. NOTE: "PROTAC-DB" used to point at
# data/sets/PROTACDB-baseline-cleaned-std.csv, a stale, differently-sized base+PMC
# union (2219 rows / 1399 compounds) predating this file's data/yaochen/ equivalent
# (2098 rows / 1350 compounds, a clean Record_ID union of DEV-NO-PMC + DEV-PMC).
DATASETS: Dict[str, Path] = {
    "DEV-NO-PMC": YAOCHEN_DIR / "PROTACDB_basedata.csv",           # #1: base only
    "AutoTPD+": YAOCHEN_DIR / "AutoTPDplus-PROTAC-training.csv",   # #2: base + AutoTPD PMC-LLM
    "PROTAC-DB": YAOCHEN_DIR / "PROTACDB-training.csv",            # #3: base + PROTAC-DB's own PMC
    "DEV-PMC-LLM": YAOCHEN_DIR / "AutoTPD-PROTAC.csv",             # #4: AutoTPD PMC-LLM, isolated
    "DEV-PMC": YAOCHEN_DIR / "PROTACDB_pmc.csv",                   # #5: PROTAC-DB's own PMC, isolated
}
TEST_CSV = DATA_DIR / "PROTAC-test.csv"

# A 6th arm, not a real CSV: per seed, retrain DEV-PMC-LLM on a fresh subsample
# drawn down to DEV-PMC's cleaned pool size for that task, isolating a data-SOURCE
# effect (#4 vs #5) from a data-QUANTITY effect. Value is (source_dataset, size_ref_dataset).
DOWNSAMPLE_ARMS: Dict[str, tuple] = {
    "DEV-PMC-LLM-downsampled": ("DEV-PMC-LLM", "DEV-PMC"),
}

# DEV-PMC rows that are exact-duplicate measurements of a DEV-NO-PMC row (same
# compound, target, cell, assay and both readouts) -- re-scraped by both curation
# pipelines rather than genuinely new PMC content. Dropped before use as set #5.
DEDUP_AGAINST_BASE = {
    "DEV-PMC": "DEV-NO-PMC",
}
DEDUP_SIGNATURE_COLS = ["Compound_Key", "Degradation_Target_Gene",
                        "Cell_Line_STD_Accession", "DC50", "Dmax", "Assay"]

# Coarse assay families the raw free-text ``Assay`` column is remapped onto (after
# dedup, which still checks the raw text -- see load_clean_frames). Order matters:
# first keyword match wins, so e.g. "NanoLuc reporter assay" hits Luminescence-Reporter
# before the generic "reporter"/"assay" wording could reach a later bucket.
ASSAY_FAMILY_KEYWORDS = [
    ("Luminescence-Reporter", ("hibit", "nanoluc", "nanoglo", "luciferase", "nluc")),
    ("Western-Immunoblot", ("western", "immunoblot", "capillary", "jess", "cytoblot", "dot blot")),
    ("ELISA-MSD", ("elisa", "msd")),
    ("Fluorescence-Flow", ("flow cytometry", "immunofluorescence", "egfp", "mcherry")),
]


def assay_family(raw) -> str:
    """Map one raw ``Assay`` string onto a coarse family, or ``"Unknown"`` if missing/unmatched."""
    if pd.isna(raw):
        return "Unknown"
    s = str(raw).lower()
    for family, keywords in ASSAY_FAMILY_KEYWORDS:
        if any(k in s for k in keywords):
            return family
    return "Other"


COMPOUND_COL = "Connectivity_Key"          # clustering key for the bootstrap
CELL_COL = "Cell_Line_STD_Accession"       # subgroup key for the seen/unseen split

TASKS = ["Dmax", "DC50", "bin"]
LABEL_COL = {"Dmax": "Dmax", "DC50": "pDC50", "bin": "Activity"}
DISPLAY = {"Dmax": "Dmax (%)", "DC50": "pDC50 (-log10 M)", "bin": "Activity"}
OBJECTIVE = {"Dmax": "reg:squarederror", "DC50": "reg:squarederror", "bin": "binary:logistic"}
RAW_TASK = {"Dmax": "dmax", "DC50": "dc50", "bin": "bin"}

# These raw CSVs use different column names than the published HF TACK dataset.
COL = dict(
    poi_col="Degradation_Target_Gene", poi_sequence_col="Degradation_Target_Sequence",
    ligase_col="Recruiter_Gene", ligase_sequence_col="Recruiter_Sequence",
)

# Context blocks per configs/data/cell_text_poi_countvec-onehot.yaml; the molecular
# block keeps the fingerprint (the biggest transfer win in the feature probe).
DATA_CONFIG = dict(
    mol_features="fingerprint+descriptors", fp_size=512, radius=16,
    use_relevant_descriptors=False,
    poi_features="sequence",        # amino-acid count vectoriser
    ligase_features="name",         # one-hot
    cell_features="description",    # sentence-transformer text embedding
    cell_line_col=CELL_COL,
    use_assay_type_encoding=True, use_treatment_time=True,
    categorical_encoding="onehot",
    normalize_labels=True, standardize_labels=False, impute_labels=True,
)
# Classification labels are used as-is (cf. the bin-*.yaml configs).
BIN_LABEL_OVERRIDES = dict(normalize_labels=False, standardize_labels=False, impute_labels=False)

# configs/model/xgboost.yaml / xgboost-bin.yaml, minus early stopping: this design
# trains on the whole pool, so there is no validation split to stop against.
XGB_PARAMS = dict(
    tree_method="hist", learning_rate=0.01, subsample=0.60, colsample_bytree=0.80,
    min_child_weight=20, max_depth=5, gamma=1.5, reg_alpha=1.5, reg_lambda=1.5,
)
NUM_BOOST_ROUND = 2000
N_SEEDS_DEFAULT = 25
VAL_FRAC_DEFAULT = 0.2          # M%: --tune's per-seed validation carve-out
N_TUNING_TRIALS_DEFAULT = 20    # matches tune_xgboost_hyperparameters's use in cross_validation.py
TUNE_EARLY_STOPPING_ROUNDS = 30

TIME_COL_PRIORITY = {"Dmax": ("Dmax_h", "DC50_h"), "DC50": ("DC50_h", "Dmax_h"),
                     "bin": ("Dmax_h", "DC50_h")}


def output_paths(out_dir: Path, task: str, dataset_name: str) -> tuple:
    """Return the ``(predictions, metadata)`` paths for one (task, dataset) cell."""
    slug = dataset_name.replace("+", "plus").replace("-", "_")
    base = Path(out_dir) / f"preds_{task}_{slug}"
    return base.with_suffix(".parquet"), base.with_suffix(".meta.json")


def seed_cache_dir(out_dir: Path, task: str, dataset_name: str, cache_key: dict) -> Path:
    """Per-seed checkpoint directory for one (task, dataset) cell.

    Keyed on a hash of ``cache_key`` (everything that affects a seed's result:
    source pool, subsample size, feature/model config, tuning budget) rather
    than just (task, dataset), so a config change can't silently resume
    checkpoints computed under a different config -- it just gets a fresh,
    differently-named directory instead.
    """
    slug = dataset_name.replace("+", "plus").replace("-", "_")
    digest = hashlib.sha1(json.dumps(cache_key, sort_keys=True, default=str).encode()).hexdigest()[:10]
    return Path(out_dir) / ".seed_cache" / f"{task}_{slug}_{digest}"


def data_config_for(task: str) -> dict:
    """Feature config for one task: shared everywhere except label handling."""
    cfg = dict(DATA_CONFIG)
    if task == "bin":
        cfg.update(BIN_LABEL_OVERRIDES)
    return cfg


def xgb_params_for(task: str, seed: int, n_jobs: int) -> dict:
    """XGBoost parameters for one task and seed."""
    params = dict(XGB_PARAMS, objective=OBJECTIVE[task], seed=seed, n_jobs=n_jobs)
    if task != "bin":
        params["base_score"] = 0.0     # labels are normalized to mean 0
    return params


# --------------------------------------------------------------------------
# Data preparation
# --------------------------------------------------------------------------
def _row_signature(df: pd.DataFrame) -> pd.Series:
    """Content signature per row over ``DEDUP_SIGNATURE_COLS``, for exact-duplicate checks."""
    cols = [c for c in DEDUP_SIGNATURE_COLS if c in df.columns]
    return df[cols].fillna("NA").astype(str).agg("|".join, axis=1)


def load_clean_frames() -> tuple:
    """Load the CSVs, drop null-SMILES rows, remove train/test compound leakage,
    drop exact-duplicate measurements per ``DEDUP_AGAINST_BASE``, then remap
    ``Assay`` onto its coarse family (see ``assay_family``).

    The Assay remap runs LAST, after the dedup check, since dedup compares raw
    Assay text (two genuinely different assays that happen to share a family
    should not look like duplicates).

    Returns:
        ``(cleaned, test_df)`` where ``cleaned`` maps dataset name to its
        leakage-filtered, family-remapped training frame, and ``test_df`` is
        likewise family-remapped.
    """
    test_df = pd.read_csv(TEST_CSV)
    n0 = len(test_df)
    test_df = test_df.dropna(subset=["SMILES"]).reset_index(drop=True)
    logger.info("test: dropped %d/%d rows with a missing SMILES", n0 - len(test_df), n0)
    test_keys = set(test_df[COMPOUND_COL].dropna())

    raw = {}
    cleaned = {}
    for name, path in DATASETS.items():
        df = pd.read_csv(path)
        n0 = len(df)
        df = df.dropna(subset=["SMILES"]).reset_index(drop=True)
        is_leak = df[COMPOUND_COL].isin(test_keys)
        logger.info("%s: dropped %d missing-SMILES, %d leaked compounds",
                    name, n0 - len(df), int(is_leak.sum()))
        raw[name] = df
        cleaned[name] = df[~is_leak].reset_index(drop=True)

    for name, base_name in DEDUP_AGAINST_BASE.items():
        base_sig = set(_row_signature(raw[base_name]))
        is_dup = _row_signature(cleaned[name]).isin(base_sig)
        if is_dup.any():
            logger.info("%s: dropped %d exact-duplicate measurements already in %s",
                        name, int(is_dup.sum()), base_name)
            cleaned[name] = cleaned[name][~is_dup].reset_index(drop=True)

    for name, df in cleaned.items():
        df["Assay"] = df["Assay"].map(assay_family)
        fam_counts = df["Assay"].value_counts().to_dict()
        logger.info("%s: Assay remapped to families %s", name, fam_counts)
    test_df["Assay"] = test_df["Assay"].map(assay_family)
    logger.info("test: Assay remapped to families %s", test_df["Assay"].value_counts().to_dict())
    return cleaned, test_df


def prepare_task_pool(df: pd.DataFrame, task: str, tmp_tag: str) -> pd.DataFrame:
    """Apply tackai's real task label-prep pipeline to a raw dataframe.

    Args:
        df: Cleaned, leakage-filtered raw dataframe.
        task: One of ``"Dmax"``, ``"DC50"``, ``"bin"``.
        tmp_tag: Filename stem for the temporary CSV handed to ``load_task_dataset``.

    Returns:
        The full labelled pool (train + held-out recombined), with a ``pDC50``
        column added and null-label rows dropped for the ``DC50`` task.
    """
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    tmp_csv = TMP_DIR / f"{tmp_tag}_{task}.csv"
    df.to_csv(tmp_csv, index=False)
    ds, held_out_ds, _labels = load_task_dataset(
        task=RAW_TASK[task], model_type="xgboost", custom_dataset_csv=str(tmp_csv),
        group="random", held_out_frac=0.0, held_out_seed=42,
    )
    full = pd.concat([ds.to_pandas(), held_out_ds.to_pandas()], ignore_index=True)
    if task == "DC50":
        full["pDC50"] = full["DC50"].apply(
            lambda v: DegradationComplexDataModule.convert_dc50_to_pdc50(v, unit="nM")
        )
        full = full.dropna(subset=["pDC50"]).reset_index(drop=True)
    return full


def add_assay_time(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """Synthesize the unified ``Assay_Time`` column the datamodule expects."""
    primary, fallback = TIME_COL_PRIORITY[task]
    df = df.copy()
    df["Assay_Time"] = df[primary].fillna(df[fallback])
    return df


def build_datamodule(train_df: pd.DataFrame, test_df: pd.DataFrame,
                     task: str) -> DegradationComplexDataModule:
    """Build and fit one DataModule with no validation split."""
    dd = DatasetDict({
        "train": Dataset.from_pandas(add_assay_time(train_df, task), preserve_index=False),
        "test": Dataset.from_pandas(add_assay_time(test_df, task), preserve_index=False),
    })
    dm = DegradationComplexDataModule(
        dataset=dd, labels=[LABEL_COL[task]], verbose=0, **COL, **data_config_for(task)
    )
    dm.setup()
    return dm


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train_one_seed(dm: DegradationComplexDataModule, task: str, seed: int, n_jobs: int,
                   dtrain: Optional[xgb.DMatrix], num_boost_round: int, tune: bool,
                   val_frac: float, n_tuning_trials: int,
                   subsample_n: Optional[int] = None,
                   groups: Optional[np.ndarray] = None) -> xgb.Booster:
    """Train one seed's model, either on the full pool or via a tuned val split.

    Args:
        dm: Fitted datamodule (features/encoders shared across every seed).
        task: Task key (``"Dmax"``, ``"DC50"`` or ``"bin"``).
        seed: Seed for both the XGBoost model and, if ``tune``, the validation pick
            (and, if ``subsample_n`` is set, the pool subsample).
        n_jobs: XGBoost thread count.
        dtrain: Pre-built full-pool training DMatrix, used when not ``tune``.
        num_boost_round: Boosting-round cap (fixed rounds if not ``tune``, else the
            cap early stopping trains up to).
        tune: If set, carve ``val_frac`` of the pool off as validation (the pick
            depends on ``seed``) and run Optuna against it instead of training on
            the full pool with fixed hyperparameters.
        val_frac: M, the validation fraction (only used if ``tune``).
        n_tuning_trials: Optuna trial budget for this seed (only used if ``tune``).
        subsample_n: If set and smaller than the pool, draw a fresh ``subsample_n``-row
            subsample of the pool for this seed (without replacement) before the
            val-split carve, so the training-set SIZE is held constant across seeds
            while its CONTENT varies. Only supported on the ``tune`` path (there is
            no full-pool ``dtrain`` to subsample on the non-``tune`` path).
        groups: Length-``len(dm.train_dataset)`` array (e.g. source-paper DOI) to
            group the val-split carve by, via ``GroupShuffleSplit`` -- no group ends
            up on both sides, unlike v1's plain random row split. Falls back to a
            random row split if ``None`` or if the (possibly subsampled) pool has
            fewer than 2 distinct groups.

    Returns:
        The trained ``xgb.Booster``.
    """
    if not tune:
        if subsample_n is not None:
            raise ValueError("subsample_n requires tune=True")
        return xgb.train(xgb_params_for(task, seed, n_jobs), dtrain,
                         num_boost_round=num_boost_round, verbose_eval=False)

    rng = np.random.default_rng(seed)
    n_full = len(dm.train_dataset)
    if subsample_n is not None and subsample_n < n_full:
        pool_idx = rng.choice(n_full, size=subsample_n, replace=False)
    else:
        pool_idx = np.arange(n_full)

    n = len(pool_idx)
    n_val = max(1, round(val_frac * n))
    pool_groups = np.asarray(groups)[pool_idx] if groups is not None else None
    if pool_groups is not None and len(np.unique(pool_groups)) >= 2:
        gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
        train_pos, val_pos = next(gss.split(np.zeros(n), groups=pool_groups))
        val_idx, train_idx = pool_idx[val_pos].tolist(), pool_idx[train_pos].tolist()
    else:
        if pool_groups is not None:
            logger.warning("only %d distinct group(s) in this seed's pool; "
                           "falling back to a random row split", len(np.unique(pool_groups)))
        perm = pool_idx[rng.permutation(n)]
        val_idx, train_idx = perm[:n_val].tolist(), perm[n_val:].tolist()
    dtrain_seed = dm.get_xgboost_dmatrix(dm.train_dataset.select(train_idx), weight=True)
    dval_seed = dm.get_xgboost_dmatrix(dm.train_dataset.select(val_idx), weight=False)

    base_config = dict(
        model_config=xgb_params_for(task, seed, n_jobs),
        training_config=dict(num_boost_round=num_boost_round,
                             early_stopping_rounds=TUNE_EARLY_STOPPING_ROUNDS),
    )
    tuned_config = tune_xgboost_hyperparameters(
        dtrain_seed, dval_seed, n_trials=n_tuning_trials, base_config=base_config, seed=seed,
    )
    tuned_config["model_config"]["seed"] = seed
    return train_xgboost(dtrain_seed, dval_seed, tuned_config)


def train_cell(task: str, dataset_name: str, seeds: List[int], n_jobs: int,
               num_boost_round: int = NUM_BOOST_ROUND, tune: bool = False,
               val_frac: float = VAL_FRAC_DEFAULT,
               n_tuning_trials: int = N_TUNING_TRIALS_DEFAULT,
               out_dir: Optional[Path] = None, resume: bool = True) -> tuple:
    """Train every seed for one (task, dataset) and return a tidy prediction table.

    Args:
        task: Task key (``"Dmax"``, ``"DC50"`` or ``"bin"``).
        dataset_name: Which training pool to fit on -- a key of ``DATASETS``, or of
            ``DOWNSAMPLE_ARMS`` (in which case the *source* dataset's pool is used,
            downsampled per seed to the size-reference dataset's cleaned pool size;
            see ``DOWNSAMPLE_ARMS`` and ``train_one_seed``'s ``subsample_n``).
        seeds: Random seeds; one model is trained per seed.
        n_jobs: XGBoost thread count, normally the Slurm CPU allocation.
        num_boost_round: Fixed boosting rounds (no validation split to early-stop on),
            or the cap early stopping trains up to when ``tune`` is set.
        tune: If set, each seed carves ``val_frac`` of the pool off as validation,
            grouped by source paper (DOI) so no paper appears on both sides (see
            ``train_one_seed``'s ``groups``), and tunes XGBoost hyperparameters
            with Optuna against it, instead of training on the full pool with
            fixed hyperparameters. Both the validation pick and the model's own
            random init therefore change together across seeds. Required
            (``ValueError`` otherwise) when ``dataset_name`` is a
            ``DOWNSAMPLE_ARMS`` key.
        val_frac: M, the validation fraction (only used if ``tune``).
        n_tuning_trials: Optuna trial budget per seed (only used if ``tune``).
        out_dir: If given, enables per-seed checkpointing under
            ``out_dir/.seed_cache/`` (see ``seed_cache_dir``) -- each seed's
            predictions are written as soon as that seed finishes, so a killed
            or timed-out run only has to redo the seeds it hadn't reached.
            ``None`` disables checkpointing (e.g. for ad hoc/notebook use).
        resume: If ``out_dir`` is set, whether to read existing checkpoints for
            this exact config (``True``, the default) or ignore them and
            retrain every seed from scratch, overwriting any stale ones.

    Returns:
        ``(predictions, meta)``. ``predictions`` has one row per (seed, test row)
        with columns matching ``tackai.evaluation.compare``'s schema plus
        ``compound``/``cell``/``cell_seen``. ``meta`` records the fitted feature
        spec and split sizes, so the analysis notebook can report them without
        paying for featurization again.
    """
    cleaned, test_raw = load_clean_frames()

    subsample_n = None
    if dataset_name in DOWNSAMPLE_ARMS:
        if not tune:
            raise ValueError(f"{dataset_name!r} is a DOWNSAMPLE_ARMS entry and requires tune=True")
        source_name, ref_name = DOWNSAMPLE_ARMS[dataset_name]
        ref_tag = f"train_{ref_name.replace('+', 'plus').replace('-', '_')}"
        ref_pool = prepare_task_pool(cleaned[ref_name], task, tmp_tag=ref_tag)
        subsample_n = len(ref_pool)
    else:
        source_name = dataset_name

    tag = f"train_{source_name.replace('+', 'plus').replace('-', '_')}"
    train_pool = prepare_task_pool(cleaned[source_name], task, tmp_tag=tag)
    test_pool = prepare_task_pool(test_raw, task, tmp_tag="test")
    if subsample_n is not None and subsample_n >= len(train_pool):
        logger.warning("%s: size-reference pool (%d rows) is not smaller than %s's own "
                       "pool (%d rows); every seed will just use the full pool",
                       dataset_name, subsample_n, source_name, len(train_pool))

    t0 = time.time()
    dm = build_datamodule(train_pool, test_pool, task)
    logger.info("featurized in %.1fs | train=%d test=%d n_features=%d",
                time.time() - t0, len(dm.dataset["train"]), len(dm.dataset["test"]),
                dm.get_total_feature_dim())

    label = LABEL_COL[task]
    # Only needed on the fixed-hyperparameter path; --tune slices dm.train_dataset
    # into a fresh (weighted) train/(unweighted) val DMatrix pair per seed instead.
    dtrain = None if tune else dm.get_xgboost_dataset("train")
    dtest = dm.get_xgboost_dataset("test")
    test_df = dm.dataset["test"].to_pandas()
    y_true = np.asarray(test_df[label], dtype=float)
    compounds = test_df[COMPOUND_COL].to_numpy()
    cells = test_df[CELL_COL].astype(str).to_numpy()
    seen_cells = set(dm.dataset["train"].to_pandas()[CELL_COL].dropna().astype(str))
    cell_seen = np.isin(cells, list(seen_cells))

    # Row-aligned with dm.train_dataset (dm.dataset["train"] and its featurized
    # counterpart share row order -- the same assumption test_df/dtest already rely
    # on above). Rows with a missing DOI each get their own singleton group instead
    # of being merged into one giant "NaN" group.
    train_doi = dm.dataset["train"].to_pandas()["DOI"]
    groups = train_doi.where(train_doi.notna(), "MISSING-DOI-" + train_doi.index.astype(str)).to_numpy()

    cell_cache_dir = None
    if out_dir is not None:
        cache_key = dict(
            source_dataset=source_name, subsample_n=subsample_n,
            data_config=data_config_for(task), num_boost_round=num_boost_round,
            tune=tune, val_frac=val_frac, n_tuning_trials=n_tuning_trials,
            # xgb_params minus the per-seed 'seed' and speed-only 'n_jobs' fields,
            # which don't affect what the checkpoint is a checkpoint OF.
            xgb_params={k: v for k, v in xgb_params_for(task, 0, 0).items()
                       if k not in ("seed", "n_jobs")},
        )
        cell_cache_dir = seed_cache_dir(out_dir, task, dataset_name, cache_key)
        cell_cache_dir.mkdir(parents=True, exist_ok=True)

    seed_frames = []
    for seed in seeds:
        seed_path = cell_cache_dir / f"seed_{seed:03d}.parquet" if cell_cache_dir else None
        if resume and seed_path is not None and seed_path.exists():
            seed_frames.append(pd.read_parquet(seed_path))
            logger.info("  seed %2d loaded from checkpoint (%s)", seed, seed_path)
            continue

        t1 = time.time()
        model = train_one_seed(dm, task, seed, n_jobs, dtrain, num_boost_round,
                               tune, val_frac, n_tuning_trials, subsample_n=subsample_n,
                               groups=groups)
        preds = model.predict(dtest)
        preds = preds.flatten() if preds.ndim > 1 else preds
        y_raw = np.asarray(dm.inverse_transform_labels(preds, label)).flatten()
        if task == "bin":
            prob, y_pred = y_raw, (y_raw >= 0.5).astype(float)
        else:
            prob, y_pred = np.full_like(y_raw, np.nan, dtype=float), y_raw
        seed_df = pd.DataFrame(
            dict(task=task, method=dataset_name, fold=seed, set="test", compound=c,
                 cell=cl, cell_seen=bool(cs), target=float(t), pred=float(p), prob=float(pr))
            for c, cl, cs, t, p, pr in zip(compounds, cells, cell_seen, y_true, y_pred, prob)
        )
        if seed_path is not None:
            tmp_seed_path = seed_path.with_suffix(".parquet.tmp")
            seed_df.to_parquet(tmp_seed_path, index=False)
            tmp_seed_path.replace(seed_path)      # atomic: a killed write never leaves a corrupt checkpoint
        seed_frames.append(seed_df)
        logger.info("  seed %2d done in %.1fs", seed, time.time() - t1)

    records_df = pd.concat(seed_frames, ignore_index=True) if seed_frames else pd.DataFrame()

    meta = dict(
        task=task, dataset=dataset_name, source_dataset=source_name, n_seeds=len(seeds),
        num_boost_round=num_boost_round,
        n_train_rows=len(dm.dataset["train"]), n_test_rows=len(dm.dataset["test"]),
        n_train_compounds=int(dm.dataset["train"].to_pandas()[COMPOUND_COL].nunique()),
        n_test_compounds=int(test_df[COMPOUND_COL].nunique()),
        n_features=int(dm.get_total_feature_dim()),
        feature_spec=dm.get_feature_spec(),
        data_config=data_config_for(task),
        xgb_params=xgb_params_for(task, 0, n_jobs),
        tune=tune,
        subsample_n=subsample_n,
    )
    if tune:
        meta.update(val_frac=val_frac, n_tuning_trials=n_tuning_trials)
    return records_df, meta


def warmup() -> None:
    """Featurize every (task, dataset) once, serially, to populate the shared cache.

    The embedding caches under ``TACKAI_CACHE`` are written with a non-atomic
    ``np.savez``, so concurrent array elements computing a missing embedding could
    corrupt them. Running this once before the array means every element then only
    reads a warm cache.
    """
    cleaned, test_raw = load_clean_frames()
    for task in TASKS:
        test_pool = prepare_task_pool(test_raw, task, tmp_tag="test")
        for name in DATASETS:
            tag = f"train_{name.replace('+', 'plus').replace('-', '_')}"
            pool = prepare_task_pool(cleaned[name], task, tmp_tag=tag)
            t0 = time.time()
            dm = build_datamodule(pool, test_pool, task)
            logger.info("warmed %-6s %-11s in %5.1fs (%d features)",
                        task, name, time.time() - t0, dm.get_total_feature_dim())
    logger.info("cache warm; the array can now run without racing on np.savez")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=TASKS, help="Task to train.")
    p.add_argument("--dataset", choices=list(DATASETS) + list(DOWNSAMPLE_ARMS),
                   help="Training pool to fit on.")
    p.add_argument("--n-seeds", type=int, default=N_SEEDS_DEFAULT,
                   help=f"Number of seeds (default {N_SEEDS_DEFAULT}).")
    p.add_argument("--num-boost-round", type=int, default=NUM_BOOST_ROUND)
    p.add_argument("--n-jobs", type=int, default=16, help="XGBoost threads.")
    p.add_argument("--tune", action="store_true",
                   help="Per seed, carve --val-frac of the pool off as validation "
                        "(the pick depends on that seed) and tune XGBoost "
                        "hyperparameters with Optuna against it, instead of training "
                        "on the full pool with fixed hyperparameters.")
    p.add_argument("--val-frac", type=float, default=VAL_FRAC_DEFAULT,
                   help=f"Validation fraction M with --tune (default {VAL_FRAC_DEFAULT}).")
    p.add_argument("--n-tuning-trials", type=int, default=N_TUNING_TRIALS_DEFAULT,
                   help=f"Optuna trials per seed with --tune (default {N_TUNING_TRIALS_DEFAULT}).")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--warmup-only", action="store_true",
                   help="Only populate the shared embedding cache, then exit.")
    p.add_argument("--overwrite", action="store_true",
                   help="Retrain even if the output parquet already exists.")
    p.add_argument("--no-resume", action="store_true",
                   help="Ignore existing per-seed checkpoints under --out-dir/.seed_cache/ "
                        "and retrain every seed from scratch (default: resume from them).")
    args = p.parse_args()

    # force=True: importing tackai installs root handlers, which would make a
    # plain basicConfig() a silent no-op and swallow every progress line below.
    logging.basicConfig(level=logging.INFO, force=True,
                        format="%(asctime)s %(levelname)s %(message)s",
                        stream=sys.stdout)

    if args.warmup_only:
        warmup()
        return

    if not args.task or not args.dataset:
        p.error("--task and --dataset are required unless --warmup-only is given")
    if args.dataset in DOWNSAMPLE_ARMS and not args.tune:
        p.error(f"--dataset {args.dataset} requires --tune")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path, _ = output_paths(args.out_dir, args.task, args.dataset)
    if out_path.exists() and not args.overwrite:
        logger.info("%s already exists; nothing to do (use --overwrite to force)", out_path)
        return

    tune_msg = (f" val_frac={args.val_frac} n_tuning_trials={args.n_tuning_trials}"
               if args.tune else "")
    logger.info("task=%s dataset=%s n_seeds=%d n_jobs=%d rounds=%d tune=%s%s",
                args.task, args.dataset, args.n_seeds, args.n_jobs, args.num_boost_round,
                args.tune, tune_msg)
    t0 = time.time()
    df, meta = train_cell(args.task, args.dataset, list(range(args.n_seeds)),
                          n_jobs=args.n_jobs, num_boost_round=args.num_boost_round,
                          tune=args.tune, val_frac=args.val_frac,
                          n_tuning_trials=args.n_tuning_trials,
                          out_dir=args.out_dir, resume=not args.no_resume)
    tmp_path = out_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp_path, index=False)
    tmp_path.replace(out_path)      # atomic publish: readers never see a partial file

    meta_path = out_path.with_suffix(".meta.json")
    tmp_meta = meta_path.with_suffix(".json.tmp")
    tmp_meta.write_text(json.dumps(meta, indent=2, default=str))
    tmp_meta.replace(meta_path)
    logger.info("wrote %s (%d rows) in %.1fs", out_path, len(df), time.time() - t0)


if __name__ == "__main__":
    main()
