"""
Discovery, labeling, and loading of cross-validation predictions.

There is no run-manifest system: ``run_cv_experiment``
(:mod:`tackai.training.cross_validation`) names prediction files directly, one
per (model, data config, task, CV group, fold, split)::

    preds-model=<model_name>-task=<task>-group=<group>-fold=<N>-split=<val|test>.csv

written under a run's ``predictions_dir`` (``tack train``'s
``predictions_dir=`` override). Each row already carries its own
``method``/``task``/``group``/``fold``/``split`` bookkeeping columns (see
``training/persistence.py::_add_run_columns``), so this module's job is purely
to discover, consolidate, and relabel those files -- not to reconstruct run
identity from anything else.

``tack collect`` (:func:`tackai.training.persistence.collect_predictions`)
concatenates one run identity's per-fold files into a single collapsed
``preds-model=...-task=...-group=....csv`` (folds/splits stay as row-level
columns), optionally deleting the per-fold sources. Both loading functions
here transparently support that layout: for each run identity they prefer the
per-fold files when present, falling back to the collapsed file otherwise (see
:func:`_iter_identity_sources`) -- so evaluation keeps working whether or not
``tack collect`` has run.

This module is the single home for:

1. **Canonical method labels.** :func:`feature_label` / :func:`method_label`
   build the published display name (e.g. ``"XGB-DMAX Cell-Text E3-OneHot
   Mol-Desc POI-Vec Time"``) from a resolved data config dict. Because
   prediction CSVs only carry the raw ``method`` string baked in at train time
   (e.g. ``"xgboost_dc50_protac-data=fp512r16_poi_onehot_..."``),
   :func:`load_predictions` / :func:`discover_runs` prettify it on a
   best-effort basis by reading the matching
   ``datamodule-data=...-group=..._hparams.yaml`` from a sibling
   ``checkpoints_dir`` when one is available; otherwise the raw string is kept.

2. **Loading predictions.** :func:`load_predictions` reads every per-fold
   prediction CSV under a predictions directory and consolidates them (rename
   columns, normalize task names, threshold XGBoost binary probabilities, mark
   val/test, drop folds-incomplete methods, convert DC50→pDC50) into one tidy
   long DataFrame.

:func:`discover_runs` summarizes the run identities found under a predictions
directory into a per-run table.
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger("tackai.evaluation.loader")

# Model-type token -> display prefix.
MODEL_DISPLAY = {"xgboost": "XGB", "xgb": "XGB", "mlp": "MLP", "bert": "BERT"}

# Task token -> display suffix appended to the model prefix.
TASK_SUFFIX = {"dmax": "-DMAX", "dc50": "-DC50", "bin": "-BIN"}

# Matches the per-fold filename `run_cv_experiment` writes
# (persistence.py::save_predictions / collect_predictions' `_PER_FOLD_RE`).
_PER_FOLD_RE = re.compile(
    r"^preds-model=(?P<model_name>.+)-task=(?P<task>[^-=]+)-group=(?P<group>[^-=]+)"
    r"-fold=(?P<fold>\d+)-split=(?P<split>val|test)\.csv$"
)

# Matches the collapsed filename `collect_predictions`/`tack collect` writes:
# every fold and split for one run identity concatenated into a single CSV,
# with `fold`/`split` kept as row-level columns instead of filename tokens.
_COLLAPSED_RE = re.compile(
    r"^preds-model=(?P<model_name>.+)-task=(?P<task>[^-=]+)-group=(?P<group>[^-=]+)\.csv$"
)

# Matches a run's raw `method` string, e.g.
# "xgboost_qr_dc50_protac-data=fp512r16_poi_onehot_..._DC50"
# -> model_type=xgboost, variant=qr, task=dc50, data_name=fp512r16_poi_onehot_..._DC50
# (see the `model_name` construction in cross_validation.run_cv_experiment).
_MODEL_NAME_RE = re.compile(
    r"^(?P<model_type>xgboost|mlp|bert)_(?:(?P<variant>qr|mve)_)?"
    r"(?P<task>[a-z0-9]+)_protac-data=(?P<data_name>.+)$"
)


# ---------------------------------------------------------------------------
# DC50 conversion
# ---------------------------------------------------------------------------

def dc50_to_pdc50(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert DC50 in nM to pDC50 (``-log10(M)``)."""
    return -np.log10(np.asarray(x, dtype=float) * 1e-9 + 1e-12)


# ---------------------------------------------------------------------------
# Canonical feature tokens (shared by structured and filename labelers)
# ---------------------------------------------------------------------------

def _categorical_suffix(encoding: Optional[str]) -> str:
    """Map a ``categorical_encoding`` value to its display suffix."""
    return {"embedding": "Emb", "onehot": "OneHot"}.get(encoding, "Ord")


def feature_label(data_config: Dict[str, Any]) -> str:
    """ Build the space-joined feature-token label from a structured data config.

    Args:
        data_config (Dict[str, Any]): A resolved ``configs/data/*.yaml`` dict
            (or ``DegradationComplexDataModule`` hyperparameters).

    Returns:
        str: Sorted, space-joined feature tokens (e.g. ``"Cell-Text Mol-Desc Time"``).
    """
    enc = data_config.get("categorical_encoding", "minmax")
    cat = _categorical_suffix(enc)
    tokens: List[str] = []

    mol_features = data_config.get("mol_features", "")
    if mol_features and "fingerprint" in mol_features:
        tokens.append("FP")
    if mol_features and "descriptors" in mol_features:
        tokens.append("Mol-Desc")
    if data_config.get("poi_features") == "name":
        tokens.append(f"POI-{cat}")
    if data_config.get("poi_features") == "sequence":
        tokens.append("POI-Vec")
    if data_config.get("poi_features") == "precomputed":
        tokens.append("POI-ESM-S")
    if data_config.get("ligase_features") == "name":
        tokens.append(f"E3-{cat}")
    if data_config.get("ligase_features") == "precomputed":
        tokens.append("E3-ESM-S")
    if data_config.get("cell_features") == "name":
        tokens.append(f"Cell-{cat}")
    if data_config.get("cell_features") == "description":
        tokens.append("Cell-Text")
    if data_config.get("use_poi_pca") or data_config.get("use_ligase_pca"):
        tokens.append("POI/E3-ESM-S-PCA")
    if data_config.get("use_treatment_time") or data_config.get("use_assay_type_encoding"):
        tokens.append("Time")

    return " ".join(sorted(tokens))


def method_label(
    model_type: str,
    task: str,
    data_config: Dict[str, Any],
    variant: str = "plain",
) -> str:
    """ Build the full display label ``"<MODEL[-VAR][-TASK]> <features>"``.

    Args:
        model_type (str): ``xgboost`` / ``mlp`` / ``bert``.
        task (str): Task name (``dmax`` / ``dc50`` / ``bin`` / …).
        data_config (Dict[str, Any]): Resolved data config used for the feature tokens.
        variant (str): ``plain`` / ``qr`` / ``mve`` / ``bin``.

    Returns:
        str: Display label matching the evaluation-notebook vocabulary.
    """
    prefix = MODEL_DISPLAY.get(model_type, model_type.upper())
    if variant == "qr":
        prefix += "-QR"
    elif variant == "mve":
        prefix += "-MVE"
    prefix += TASK_SUFFIX.get(task, "")
    feats = feature_label(data_config)
    return f"{prefix} {feats}".strip()


# ---------------------------------------------------------------------------
# Pretty labels from datamodule hparams (best effort)
# ---------------------------------------------------------------------------

def _parse_model_name(model_name: str) -> Optional[Tuple[str, str, str]]:
    """Parse a run's raw ``method`` string into ``(model_type, variant, data_name)``.

    Returns:
        Optional[Tuple[str, str, str]]: ``None`` if ``model_name`` doesn't match
        the ``<model_type>[_<variant>]_<task>_protac-data=<data_name>`` shape
        (e.g. a foreign/manually-added prediction file).
    """
    m = _MODEL_NAME_RE.match(model_name)
    if not m:
        return None
    return m.group("model_type"), m.group("variant") or "plain", m.group("data_name")


@lru_cache(maxsize=None)
def _load_data_config_cached(
    checkpoints_dir: str, data_name: str, group: str
) -> Optional[Dict[str, Any]]:
    """Best-effort load of the resolved data config for one ``(data_name, group)``.

    Reads whichever ``datamodule-data=<data_name>-group=<group>-fold=*_hparams.yaml``
    sorts first -- the data config is identical across folds, only the fitted
    state differs -- so any one of them suffices.

    Args:
        checkpoints_dir (str): Directory holding the ``datamodule-*_hparams.yaml`` files.
        data_name (str): The ``data=...`` slug embedded in the run's ``method`` string.
        group (str): CV grouping strategy (``random`` / ``scaffold`` / ``butina``).

    Returns:
        Optional[Dict[str, Any]]: The parsed hparams dict, or ``None`` if no matching
        file exists (e.g. checkpoints weren't kept for this run).
    """
    matches = sorted(
        Path(checkpoints_dir).glob(f"datamodule-data={data_name}-group={group}-fold=*_hparams.yaml")
    )
    if not matches:
        return None
    with open(matches[0]) as fh:
        return yaml.safe_load(fh)


def _pretty_method_label(
    model_name: str,
    task: str,
    group: str,
    checkpoints_dir: Optional[Path],
) -> str:
    """Best-effort :func:`method_label`-prettified name; falls back to ``model_name``.

    Args:
        model_name (str): The raw ``method`` string baked into the prediction CSVs.
        task (str): Raw (un-normalized) task token, e.g. ``dc50``.
        group (str): CV grouping strategy for this run.
        checkpoints_dir (Optional[Path]): Directory to look up the data config in;
            ``None`` skips prettification entirely.

    Returns:
        str: The prettified label, or ``model_name`` unchanged if the run's data
        config can't be found or ``model_name`` doesn't match the expected shape.
    """
    if checkpoints_dir is None:
        return model_name
    parsed = _parse_model_name(model_name)
    if parsed is None:
        return model_name
    model_type, variant, data_name = parsed
    data_config = _load_data_config_cached(str(checkpoints_dir), data_name, group)
    if data_config is None:
        return model_name
    return method_label(model_type, task, data_config, variant)


def _resolve_checkpoints_dir(
    predictions_dir: Path, checkpoints_dir: Optional[Union[str, Path]], pretty_labels: bool
) -> Optional[Path]:
    """Resolve the checkpoints directory used for label prettification.

    Args:
        predictions_dir (Path): The predictions directory being loaded.
        checkpoints_dir (Optional[Union[str, Path]]): Explicit override, or ``None``
            to try the conventional sibling ``checkpoints/`` directory.
        pretty_labels (bool): If ``False``, prettification is skipped unconditionally.

    Returns:
        Optional[Path]: The resolved directory, or ``None`` if prettification is
        disabled or no such directory can be found.
    """
    if not pretty_labels:
        return None
    if checkpoints_dir is not None:
        return Path(checkpoints_dir)
    default_dir = predictions_dir.parent / "checkpoints"
    return default_dir if default_dir.is_dir() else None


# ---------------------------------------------------------------------------
# File discovery (per-fold, with a collapsed/aggregated-CSV fallback)
# ---------------------------------------------------------------------------

def _iter_identity_sources(predictions_dir: Path) -> Dict[Tuple[str, str, str], List[Path]]:
    """Group every prediction CSV under ``predictions_dir`` by run identity.

    Each ``(model_name, task, group)`` identity maps to the file(s) that make
    up its predictions: its per-fold files
    (``preds-...-fold=<N>-split={val,test}.csv``) when present, or -- only
    when none of those are on disk for that identity (e.g. ``tack collect``
    ran without ``--keep-sources``) -- its single collapsed file
    (``preds-...-task=...-group=....csv``, folds/splits as row columns). An
    identity present in both forms uses only the per-fold files, since
    ``run_cv_experiment`` writes the collapsed file as a non-destructive
    convenience copy (``collect_predictions(remove_sources=False)``) and
    reading both would double-count every row.

    Args:
        predictions_dir (Path): Directory holding the ``preds-*.csv`` files.

    Returns:
        Dict[Tuple[str, str, str], List[Path]]: Identity -> list of files to
        read for it (one file for a collapsed identity, one-or-more for a
        per-fold identity).
    """
    per_fold: Dict[Tuple[str, str, str], List[Path]] = {}
    for path in sorted(predictions_dir.glob("preds-*-fold=*-split=*.csv")):
        m = _PER_FOLD_RE.match(path.name)
        if m is None:
            continue
        key = (m.group("model_name"), m.group("task"), m.group("group"))
        per_fold.setdefault(key, []).append(path)

    sources: Dict[Tuple[str, str, str], List[Path]] = dict(per_fold)
    for path in sorted(predictions_dir.glob("preds-model=*-task=*-group=*.csv")):
        m = _COLLAPSED_RE.match(path.name)
        if m is None:
            continue
        key = (m.group("model_name"), m.group("task"), m.group("group"))
        if key in sources:
            continue  # per-fold files already cover this identity
        sources[key] = [path]

    return sources


def _identity_fold_ids(paths: List[Path]) -> set:
    """Return the distinct fold ids covered by one identity's source file(s).

    Per-fold files encode the fold in the filename; a collapsed file carries
    it as a row-level ``fold`` column instead, so that one file is read to
    recover it.

    Args:
        paths (List[Path]): One identity's files, as returned by
            :func:`_iter_identity_sources`.

    Returns:
        set: The distinct fold ids found.
    """
    folds: set = set()
    for path in paths:
        m = _PER_FOLD_RE.match(path.name)
        if m is not None:
            folds.add(int(m.group("fold")))
        else:
            folds.update(pd.read_csv(path, usecols=["fold"])["fold"].astype(int).unique())
    return folds


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------

def discover_runs(
    predictions_dir: Union[str, Path],
    checkpoints_dir: Optional[Union[str, Path]] = None,
    pretty_labels: bool = True,
) -> pd.DataFrame:
    """ Summarize the run identities available for evaluation.

    Args:
        predictions_dir (Union[str, Path]): Directory holding the prediction
            CSVs -- either per-fold (``preds-*-fold=*-split=*.csv``) or
            collapsed/aggregated (``preds-model=...-task=...-group=....csv``,
            from ``tack collect``); see :func:`_iter_identity_sources`.
        checkpoints_dir (Optional[Union[str, Path]]): Directory holding
            ``datamodule-*_hparams.yaml`` files, used to prettify method labels
            (see :func:`load_predictions`). Defaults to a sibling ``checkpoints/``
            next to ``predictions_dir`` if one exists.
        pretty_labels (bool): Whether to attempt label prettification at all.

    Returns:
        pd.DataFrame: One row per run identity: ``method, model_type,
        model_variant, task, group, data_config_name, n_folds``.
    """
    predictions_dir = Path(predictions_dir)
    resolved_ckpt_dir = _resolve_checkpoints_dir(predictions_dir, checkpoints_dir, pretty_labels)

    identity_sources = _iter_identity_sources(predictions_dir)

    rows: List[Dict[str, Any]] = []
    for (model_name, task, group), paths in sorted(identity_sources.items()):
        parsed = _parse_model_name(model_name)
        model_type, variant, data_name = parsed if parsed else (None, None, None)
        label = _pretty_method_label(model_name, task, group, resolved_ckpt_dir)
        rows.append({
            "method": label,
            "model_type": model_type,
            "model_variant": variant,
            "task": task,
            "group": group,
            "data_config_name": data_name,
            "n_folds": len(_identity_fold_ids(paths)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Prediction loading
# ---------------------------------------------------------------------------

# Raw task token -> normalized task name used throughout evaluation.
_TASK_NORMALIZE = {
    "dmax": "Dmax", "dc50": "DC50", "bin": "bin",
    "binary_class": "bin", "heldout": "bin", "multitask": "bin",
}

# Lowercase CLI input -> internal normalized task name.
TASK_ALIASES: Dict[str, str] = {"dmax": "Dmax", "dc50": "DC50", "bin": "bin"}


def normalize_task(task: str) -> str:
    """Normalize a user-supplied task name (lowercase or display) to internal form."""
    return TASK_ALIASES.get(task.lower(), task)


def _normalize_task(series: pd.Series) -> pd.Series:
    """Normalize raw task tokens to display task names."""
    out = series.astype(str)
    for raw, norm in _TASK_NORMALIZE.items():
        out = out.str.replace(raw, norm, regex=False)
    return out


def load_predictions(
    predictions_dir: Union[str, Path],
    tasks: Optional[Sequence[str]] = None,
    methods: Optional[Sequence[str]] = None,
    convert_dc50: bool = True,
    require_complete_folds: bool = True,
    checkpoints_dir: Optional[Union[str, Path]] = None,
    pretty_labels: bool = True,
) -> pd.DataFrame:
    """ Load all CV predictions into one tidy long DataFrame.

    Reads each run identity's prediction file(s) -- either its per-fold CSVs
    (``preds-model=...-task=...-group=...-fold=<N>-split={val,test}.csv``,
    written by :func:`tackai.training.cross_validation.run_cv_experiment`) or,
    if those are no longer on disk, its collapsed/aggregated CSV (from
    ``tack collect``; see :func:`_iter_identity_sources`). Each row already
    carries its own bookkeeping columns; this function only consolidates them
    and renames two to the vocabulary the rest of ``tackai.evaluation``
    expects: the CSV's ``group`` (CV grouping strategy) becomes ``cv_group``,
    and its ``split`` (val/test) becomes ``set``.

    Args:
        predictions_dir (Union[str, Path]): Directory holding the prediction
            CSVs -- per-fold and/or collapsed (a run's ``predictions_dir``,
            e.g. ``outputs/<run>/predictions``).
        tasks (Optional[Sequence[str]]): If given, keep only these normalized
            tasks (e.g. ``["Dmax"]``).
        methods (Optional[Sequence[str]]): If given, keep only these method
            labels (matched post-prettification, when enabled).
        convert_dc50 (bool): Convert DC50 targets/preds to pDC50.
        require_complete_folds (bool): Drop ``(method, task)`` pairs that have
            fewer folds than the maximum observed (paired tests need equal folds).
        checkpoints_dir (Optional[Union[str, Path]]): Directory holding
            ``datamodule-*_hparams.yaml`` files, used to prettify the raw
            ``method`` strings via :func:`method_label`. Defaults to a sibling
            ``checkpoints/`` next to ``predictions_dir`` if one exists.
        pretty_labels (bool): Whether to attempt the ``checkpoints_dir``-based
            prettification at all; if ``False`` (or no such directory is found),
            the raw ``method`` strings baked into the CSVs are kept as-is.

    Returns:
        pd.DataFrame: Long DataFrame with at least ``[method, task, set, fold,
        cv_group, target, pred, prob]`` plus any extra columns present in the
        prediction CSVs (e.g. ``row_id``).
    """
    predictions_dir = Path(predictions_dir)
    if tasks is not None:
        tasks = [normalize_task(t) for t in tasks]
        unknown = [t for t in tasks if t not in ("Dmax", "DC50", "bin")]
        if unknown:
            raise ValueError(f"Unknown tasks {unknown}; valid: dmax, dc50, bin")

    identity_sources = _iter_identity_sources(predictions_dir)
    if not identity_sources:
        raise FileNotFoundError(
            f"No prediction files (per-fold preds-*-fold=*-split=*.csv, or "
            f"collapsed preds-model=...-task=...-group=....csv) found under {predictions_dir}"
        )
    files = [path for paths in identity_sources.values() for path in paths]

    resolved_ckpt_dir = _resolve_checkpoints_dir(predictions_dir, checkpoints_dir, pretty_labels)

    frames: List[pd.DataFrame] = []
    required_cols = {"target", "pred", "fold", "group", "method", "task", "split"}

    for path in files:
        df = pd.read_csv(path)
        missing = required_cols - set(df.columns)
        if missing:
            logger.warning("Skipping %s: missing expected columns %s", path.name, sorted(missing))
            continue

        raw_task = str(df["task"].iloc[0])
        norm_task = _normalize_task(df["task"])
        if tasks is not None and norm_task.iloc[0] not in tasks:
            continue

        # XGBoost binary: 'pred' holds probabilities; threshold for the label.
        if (norm_task == "bin").any() and "prob" not in df.columns:
            df["prob"] = df["pred"].copy()
            df["pred"] = (df["prob"] >= 0.5).astype(int)

        group_value = str(df["group"].iloc[0])
        label = _pretty_method_label(df["method"].iloc[0], raw_task, group_value, resolved_ckpt_dir)
        if methods is not None and label not in methods:
            continue

        df["task"] = norm_task
        df = df.rename(columns={"group": "cv_group", "split": "set"})
        df["method"] = label

        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No usable prediction files under {predictions_dir} "
            f"(tasks={tasks!r}, methods={methods!r})."
        )

    results = pd.concat(frames, ignore_index=True)

    if require_complete_folds and not results.empty:
        max_folds = results.groupby("method")["fold"].nunique().max()
        drop_idx: List[int] = []
        for (method, task), grp in results.groupby(["method", "task"]):
            n = grp["fold"].nunique()
            if n < max_folds:
                logger.warning(
                    "Method '%s' task '%s' has %d folds (expected %d); dropping.",
                    method, task, n, max_folds,
                )
                drop_idx.extend(grp.index.tolist())
        if drop_idx:
            results = results.drop(drop_idx).reset_index(drop=True)

    if convert_dc50 and (results["task"] == "DC50").any():
        mask = results["task"] == "DC50"
        for col in ("target", "pred", "pred_lower", "pred_upper"):
            if col in results.columns:
                results.loc[mask, col] = dc50_to_pdc50(results.loc[mask, col].to_numpy())

    return results
