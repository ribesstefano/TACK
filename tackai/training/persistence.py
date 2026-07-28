"""
Persistence helpers for data modules and cross-validation prediction outputs.
"""
import re
from pathlib import Path
from typing import Dict, FrozenSet, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import DatasetDict

from tackai import DegradationComplexDataModule
from tackai.config import save_config_to_yaml
from tackai.data.ids import ROW_ID_COLUMN

# Matches a per-fold prediction filename and captures the stable prefix (the
# model/task/group identity) that survives collapsing across folds and splits.
_PER_FOLD_RE = re.compile(r"^(?P<prefix>.+?)-fold=[^-]+-split=[^-]+\.csv$")

# Cache of the (fold, split) pairs recorded in each combined file, keyed by
# (path, mtime_ns) so an immutable combined file is parsed once per run rather
# than once per fold during the resume check.
_COMBINED_FOLD_SPLIT_CACHE: Dict[Tuple[str, int], FrozenSet[Tuple[int, str]]] = {}


def combined_predictions_path(
    results_dir: Path, model_name: str, task: str, group: str
) -> Path:
    """Return the path of the fold-collapsed prediction file for a run.

    One file per ``(model, task, group)`` identity — the same prefix the
    per-fold files share, minus the ``-fold=...-split=...`` suffix.

    Args:
        results_dir: Directory holding the prediction files.
        model_name: Model identity component of the filename.
        task: Task identity component of the filename.
        group: CV grouping-strategy component of the filename.

    Returns:
        Path to the ``preds-model=...-task=...-group=....csv`` combined file.
    """
    return Path(results_dir) / f"preds-model={model_name}-task={task}-group={group}.csv"


def _combined_fold_split_set(combined_path: Path) -> FrozenSet[Tuple[int, str]]:
    """Return the ``(fold, split)`` pairs recorded in a combined predictions file.

    The result is cached on the file's path and modification time, so the
    per-fold resume check reads each (immutable during a run) combined file only
    once instead of re-parsing it for every fold.

    Args:
        combined_path: Path to a fold-collapsed prediction file.

    Returns:
        A frozenset of ``(fold, split)`` pairs. A missing file, or a legacy /
        foreign file lacking the ``fold`` or ``split`` columns, yields an empty
        set (rather than raising) so the resume check degrades to "retrain this
        fold" instead of aborting the whole run.
    """
    try:
        stat = combined_path.stat()
    except OSError:
        return frozenset()
    key = (str(combined_path), stat.st_mtime_ns)
    cached = _COMBINED_FOLD_SPLIT_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        df = pd.read_csv(combined_path, usecols=['fold', 'split'])
        pairs: FrozenSet[Tuple[int, str]] = frozenset(
            zip((int(f) for f in df['fold']), (str(s) for s in df['split']))
        )
    except (ValueError, KeyError):
        pairs = frozenset()
    _COMBINED_FOLD_SPLIT_CACHE[key] = pairs
    return pairs


def fold_predictions_exist(
    results_dir: Path,
    model_name: str,
    tasks: List[str],
    group: str,
    fold_id: int,
    splits: Sequence[str] = ('val', 'test'),
) -> bool:
    """Return True if a fold's predictions already exist for every task/split.

    Checks the per-fold files first (present mid-run, before collection) and
    falls back to the fold-collapsed combined file (present after a completed
    run, once the per-fold files have been removed). This keeps the fold-skip
    resume logic working across both states.

    Args:
        results_dir: Directory holding the prediction files.
        model_name: Model identity used to build the filenames.
        tasks: Task names to check (a multitask run expands to several).
        group: CV grouping strategy used to build the filenames.
        fold_id: Fold index to look for.
        splits: Short split labels that must all be present for the fold to
            count as complete.

    Returns:
        True if, for every ``task`` and every ``split`` in ``splits``, a
        prediction exists (as a per-fold file or a row in the combined file);
        False as soon as any is missing.
    """
    results_dir = Path(results_dir)
    for task in tasks:
        combined_path = combined_predictions_path(results_dir, model_name, task, group)
        combined_pairs = None
        for split_name in splits:
            per_fold = results_dir / (
                f"preds-model={model_name}-task={task}-group={group}"
                f"-fold={fold_id}-split={split_name}.csv"
            )
            if per_fold.exists():
                continue
            if combined_path.exists():
                if combined_pairs is None:
                    combined_pairs = _combined_fold_split_set(combined_path)
                if (fold_id, split_name) in combined_pairs:
                    continue
            return False
    return True

# Maps the short split label used in filenames to the DatasetDict split key.
_DS_SPLIT = {'val': 'validation', 'test': 'test'}


def _split_row_ids(fold_dataset: DatasetDict, split_name: str, n: int):
    """Return the ``row_id`` list for a split, aligned by position.

    Predictions are produced in the same row order as ``fold_dataset[split]``,
    so a positional take is a valid join key. Returns ``None`` (with a warning)
    when the ids are unavailable or the lengths disagree, in which case the
    caller writes predictions without a ``row_id`` column rather than a
    misaligned one.

    Args:
        fold_dataset: The fold's DatasetDict (with 'validation'/'test' splits).
        split_name: Short split label ('val' or 'test').
        n: Expected number of rows (length of the predictions arrays).

    Returns:
        A list of ``row_id`` values of length ``n``, or ``None``.
    """
    ds_split = _DS_SPLIT.get(split_name)
    if fold_dataset is None or ds_split not in fold_dataset:
        return None
    if ROW_ID_COLUMN not in fold_dataset[ds_split].column_names:
        print(f"Warning: '{ROW_ID_COLUMN}' missing from {ds_split} split; "
              "predictions saved without row ids.")
        return None
    row_ids = list(fold_dataset[ds_split][ROW_ID_COLUMN])
    if len(row_ids) != n:
        print(f"Warning: row_id count ({len(row_ids)}) != prediction count "
              f"({n}) for {ds_split} split; predictions saved without row ids.")
        return None
    return row_ids


def _add_run_columns(
    df: pd.DataFrame,
    fold_dataset: DatasetDict,
    split_name: str,
    fold_id: int,
    group: str,
    model_name: str,
    n: int,
    task: str = None,
) -> pd.DataFrame:
    """Attach the per-run provenance columns (and ``row_id``) to a preds frame.

    Every prediction file carries the same ``fold``/``group``/``method``/
    ``split`` bookkeeping (plus ``task`` for the single- and multi-task layouts;
    the BERT layout sets it per ``Value_Type`` afterwards) and, when the ids are
    available and aligned, a leading ``row_id`` join key. Centralizing the block
    keeps the three ``save_predictions`` branches from diverging.

    Args:
        df: Predictions dataframe; its rows must be in ``fold_dataset[split]``
            row order (so the positional ``row_id`` take is a valid join key).
        fold_dataset: The fold's DatasetDict, source of the ``row_id`` values.
        split_name: Short split label ('val' or 'test'), stored in 'split'.
        fold_id: Fold identifier, stored in 'fold'.
        group: CV grouping strategy, stored in 'group'.
        model_name: Model name, stored in 'method'.
        n: Number of rows (length of the predictions arrays).
        task: Task name for the 'task' column; omitted (None) when the caller
            sets 'task' itself (the BERT per-``Value_Type`` case).

    Returns:
        The same ``df``, with the provenance columns added and, if resolvable, a
        ``row_id`` column inserted at position 0.
    """
    df['fold'] = fold_id
    df['group'] = group
    df['method'] = model_name
    if task is not None:
        df['task'] = task
    df['split'] = split_name

    row_ids = _split_row_ids(fold_dataset, split_name, n)
    if row_ids is not None:
        df.insert(0, ROW_ID_COLUMN, row_ids)
    return df


def save_data_module(
    data_module: DegradationComplexDataModule,
    save_dir: Path,
    name: str,
) -> None:
    """Save data module hyperparameters and state to files.

    Args:
        data_module: The data module to save.
        save_dir: Directory to save the files.
        name: Base name for the saved files.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save hyperparameters as YAML
    hparams_path = save_dir / f"{name}_hparams.yaml"
    hparams = data_module.get_hyperparameters()
    save_config_to_yaml(hparams, hparams_path)

    # Save state dictionary (label transformers, etc.)
    state_path = save_dir / f"{name}_state.pt"
    state_dict = data_module.state_dict()
    torch.save(state_dict, state_path)


def load_data_module_state(
    data_module: DegradationComplexDataModule,
    save_dir: Path,
    name: str,
) -> DegradationComplexDataModule:
    """Load data module state from saved files.

    Args:
        data_module: The data module to load state into.
        save_dir: Directory containing the saved files.
        name: Base name of the saved files.

    Returns:
        Data module with loaded state.
    """
    state_path = save_dir / f"{name}_state.npz"
    if state_path.exists():
        state_dict = dict(np.load(state_path, allow_pickle=True))
        data_module.load_state_dict(state_dict)
    return data_module


def save_predictions(
    predictions: Dict[str, np.ndarray],
    data_module: DegradationComplexDataModule,
    model_type: str,
    model_name: str,
    task_name: str,
    group: str,
    fold_id: int,
    results_dir: Path,
    fold_dataset: DatasetDict,
) -> None:
    """Save prediction results to CSV files.

    Args:
        predictions: Dictionary with predictions and targets.
        data_module: Data module with task information.
        model_type: Type of model.
        model_name: Name of the model.
        task_name: Name of the task.
        group: Grouping strategy.
        fold_id: Fold identifier.
        results_dir: Directory to save results.
    """
    if task_name == 'multitask' and len(data_module.labels) > 1:
        # Multi-task MLP: save separate files for each label
        for label in data_module.labels:
            for split_name in ['val', 'test']:
                preds_key = f'{split_name}_preds_{label}'
                targets_key = f'{split_name}_targets_{label}'

                if preds_key not in predictions:
                    print(f"Warning: Predictions for key '{preds_key}' not found. Skipping...")
                    continue

                n = len(predictions[preds_key])
                preds_df = pd.DataFrame({
                    'target': predictions[targets_key],
                    'pred': predictions[preds_key],
                })
                _add_run_columns(
                    preds_df, fold_dataset, split_name, fold_id, group,
                    model_name, n, task=label.lower(),
                )

                preds_path = results_dir / f"preds-model={model_name}-task={label.lower()}-group={group}-fold={fold_id}-split={split_name}.csv"
                preds_df.to_csv(preds_path, index=False)
                print(f"Saved {label} predictions to: {preds_path}")

    elif model_type == 'bert' and data_module.val_tasks is not None:
        # BERT with Value_Type: split by Value_Type
        val_value_types = data_module.val_tasks
        test_value_types = data_module.test_tasks

        for split_name, value_types in [('val', val_value_types), ('test', test_value_types)]:
            n = len(predictions[f'{split_name}_preds'])
            preds_df = pd.DataFrame({
                'value_type': value_types,
                'target': predictions[f'{split_name}_targets'],
                'pred': predictions[f'{split_name}_preds'],
            })
            # 'task' is set per Value_Type below, so leave it off here.
            _add_run_columns(
                preds_df, fold_dataset, split_name, fold_id, group,
                model_name, n,
            )

            for value_type in ['Dmax', 'DC50']:
                type_df = preds_df[preds_df['value_type'] == value_type].copy()
                if len(type_df) == 0:
                    continue

                type_df['task'] = value_type.lower()
                type_df = type_df.drop(columns=['value_type'])

                preds_path = results_dir / f"preds-model={model_name}-task={value_type.lower()}-group={group}-fold={fold_id}-split={split_name}.csv"
                type_df.to_csv(preds_path, index=False)
                print(f"Saved {value_type} predictions to: {preds_path}")
    else:
        # Single-task: save single slim file. Rows are keyed by ``row_id`` (a
        # stable content hash) rather than by copying the full assay metadata,
        # which is recoverable from the source dataset via that key.
        for split_name in ['val', 'test']:
            n = len(predictions[f'{split_name}_preds'])
            preds_df = pd.DataFrame({
                'target': predictions[f'{split_name}_targets'],
                'pred': predictions[f'{split_name}_preds'],
            })

            # Add uncertainty estimates if available: lower and upper quantiles
            if 'val_preds_quantiles_lower' in predictions and split_name == 'val':
                preds_df['pred_lower'] = predictions['val_preds_quantiles_lower']
                preds_df['pred_upper'] = predictions['val_preds_quantiles_upper']
            if 'test_preds_quantiles_lower' in predictions and split_name == 'test':
                preds_df['pred_lower'] = predictions['test_preds_quantiles_lower']
                preds_df['pred_upper'] = predictions['test_preds_quantiles_upper']

            # Add uncertainty estimates if available: predicted variance (for MVE)
            if 'val_uct' in predictions and split_name == 'val':
                preds_df['pred_variance'] = predictions['val_uct']
            if 'test_uct' in predictions and split_name == 'test':
                preds_df['pred_variance'] = predictions['test_uct']

            _add_run_columns(
                preds_df, fold_dataset, split_name, fold_id, group,
                model_name, n, task=task_name,
            )

            preds_path = results_dir / f"preds-model={model_name}-task={task_name}-group={group}-fold={fold_id}-split={split_name}.csv"
            preds_df.to_csv(preds_path, index=False)
            print(f"Saved predictions to: {preds_path}")


def collect_predictions(
    predictions_dir: Path,
    remove_sources: bool = True,
    pattern: str = "preds-*-fold=*-split=*.csv",
) -> List[Path]:
    """Collapse per-fold prediction CSVs into one file per model/task/config.

    Per-fold files sharing a ``preds-model=...-task=...-group=...`` prefix are
    concatenated across every fold *and* split into a single
    ``{prefix}.csv``. Each combined file is self-describing — its ``fold`` and
    ``split`` columns preserve the information dropped from the filename — so a
    ``row_id`` legitimately recurs (once per fold/split it took part in), which
    is exactly what downstream CV aggregation consumes.

    A directory may hold several runs (e.g. a sweep loop over configs); each
    ``(model, task, group)`` identity yields its own combined file. Only files
    with a ``-fold=...-split=...`` suffix are gathered, so already-collapsed
    outputs are never re-ingested and the operation is idempotent.

    An existing combined file is *merged* rather than overwritten: its rows are
    concatenated with the freshly-gathered per-fold rows and exact duplicates are
    dropped. This makes a partial re-run (which only regenerates a subset of
    folds) additive — folds collected by an earlier run but not retrained this
    time are preserved instead of being clobbered.

    Args:
        predictions_dir: Directory holding the ``preds-*.csv`` files.
        remove_sources: If True (default), delete the per-fold files of each
            group after its combined file is written successfully. Note that
            downstream ensemble selection (``scripts/ensemble_comparison.py``)
            reads the per-fold ``-split=`` files, so pass False to keep them.
        pattern: Glob selecting the per-fold files to gather.

    Returns:
        The list of combined-file paths written (one per identity).

    Raises:
        FileNotFoundError: If no per-fold file matches ``pattern``.
    """
    predictions_dir = Path(predictions_dir)

    # Group per-fold files by their stable model/task/group prefix.
    groups: Dict[str, List[Path]] = {}
    for f in sorted(predictions_dir.glob(pattern)):
        match = _PER_FOLD_RE.match(f.name)
        if match is None:
            continue
        groups.setdefault(match.group("prefix"), []).append(f)

    if not groups:
        raise FileNotFoundError(
            f"No per-fold files matching '{pattern}' in {predictions_dir}"
        )

    written: List[Path] = []
    for prefix, files in groups.items():
        out_path = predictions_dir / f"{prefix}.csv"

        # Merge with any previously-collected folds so nothing collected by an
        # earlier run is lost, then drop exact-duplicate rows (a fold present in
        # both the existing combined file and the current per-fold files).
        frames = []
        if out_path.exists():
            frames.append(pd.read_csv(out_path))
        frames.extend(pd.read_csv(f) for f in files)
        combined = pd.concat(frames, ignore_index=True).drop_duplicates(
            ignore_index=True
        )

        combined.to_csv(out_path, index=False)
        print(f"Collected {len(files)} files ({len(combined)} rows) into: {out_path}")
        written.append(out_path)

        if remove_sources:
            for f in files:
                f.unlink()
            print(f"Removed {len(files)} per-fold prediction files for {prefix}.")

    return written
