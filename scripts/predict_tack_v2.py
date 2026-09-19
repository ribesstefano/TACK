"""
Append TACK ensemble predictions (Dmax, DC50, binary activity) as columns to
an input CSV of PROTAC compounds.

The input CSV may use either the raw curated column names (as in tack_2.csv,
e.g. ``Degradation_Target``/``Recruiter``) or the canonical ``ailab-bio/TACK``
schema (``POI_Name``/``Ligase_Name``/...) -- see ``canonicalize_columns()``.
Column semantics and the sample-construction pattern follow
notebooks/datamodule_dev.ipynb and notebooks/ensemble_predictor_tutorial.ipynb.

Usage:
    python scripts/predict_tack_v2.py \\
        --input-csv tack_2.csv \\
        --dmax-ensemble-dir ensembles/dmax_caruana_ensemble \\
        --dc50-ensemble-dir ensembles/dc50_caruana_ensemble \\
        --bin-ensemble-dir ensembles/bin_caruana_ensemble \\
        --output-csv tack_v2_with_predictions.csv
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from tackai import EnsemblePredictor, SampleInput
from tackai.ensemble_predictor import EnsemblePrediction

logger = logging.getLogger("predict_tack_v2")

REPO_ROOT = Path(__file__).resolve().parent.parent

# Columns whose emptiness is worth flagging upfront: a row missing one of
# these will make any ensemble member that needs it fail loudly rather than
# silently, so surfacing gaps before prediction starts saves a confusing trip
# through the per-row fallback path (see run_ensemble()).
DIAGNOSTIC_COLUMNS = ["SMILES", "POI_Name", "POI_Sequence", "Ligase_Name", "Ligase_Sequence"]

# tack_2.csv (raw curated CSV) uses different column names than
# DegradationComplexDataModule/EnsemblePredictor expect by default, which
# match the ailab-bio/TACK Hugging Face dataset schema -- see the renaming
# cell in notebooks/datamodule_dev.ipynb.
RAW_TO_CANONICAL_COLUMNS = {
    "Degradation_Target": "POI_Name",
    "Degradation_Target_Sequence": "POI_Sequence",
    "Recruiter": "Ligase_Name",
    "Recruiter_Sequence": "Ligase_Sequence",
}

# EnsemblePredictor task key -> output column prefix.
TASK_PREFIXES = {"dmax": "Dmax", "dc50": "DC50", "bin": "Binary"}


def _setup_logging(verbose: bool) -> None:
    """Configure root logging to stdout (SLURM's ``.out``).

    Uses stdout rather than logging's stderr default so normal progress
    lands in ``.out`` and ``.err`` is reserved for actual warnings/tracebacks
    -- otherwise every INFO line ends up in ``.err``, which looks like a
    failure even on a successful run.
    """
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.addHandler(handler)


def _log_environment(args: argparse.Namespace) -> None:
    """Log run context useful for diagnosing failures from the SLURM logs alone."""
    logger.info(f"Command: {' '.join(sys.argv)}")
    logger.info(f"Resolved args: {vars(args)}")
    logger.info(f"Python {sys.version.split()[0]} | torch {torch.__version__} | "
                f"cuda_available={torch.cuda.is_available()}")
    slurm_keys = ["SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE"]
    slurm_env = {k: os.environ[k] for k in slurm_keys if k in os.environ}
    if slurm_env:
        logger.info(f"SLURM context: {slurm_env}")

    for label, path in [
        ("input CSV", args.input_csv),
        ("Dmax ensemble dir", args.dmax_ensemble_dir),
        ("DC50 ensemble dir", args.dc50_ensemble_dir),
        ("binary ensemble dir", args.bin_ensemble_dir),
        ("embeddings file", args.embeddings_file),
    ]:
        logger.info(f"{label}: {path} ({'exists' if path.exists() else 'MISSING'})")


def _log_data_quality(df: pd.DataFrame) -> None:
    """Log per-column missingness for the fields ensemble members most commonly require."""
    for col in DIAGNOSTIC_COLUMNS:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found -- any model requiring it will fail per-row.")
            continue
        n_missing = int(df[col].isna().sum())
        if n_missing:
            logger.warning(f"Column '{col}': {n_missing}/{len(df)} rows missing.")
        else:
            logger.debug(f"Column '{col}': no missing values.")


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw curated columns (tack_2.csv-style) to the canonical schema.

    Only renames when the source column is present and the canonical name is
    not already there, so CSVs that already use the canonical schema (e.g.
    exported from the ailab-bio/TACK dataset) pass through unchanged. A
    missing ``Assay_Time`` is derived from ``Dmax_h``, falling back to
    ``DC50_h`` where the former is missing.
    """
    rename = {
        src: dst for src, dst in RAW_TO_CANONICAL_COLUMNS.items()
        if src in df.columns and dst not in df.columns
    }
    if rename:
        logger.info(f"Renaming columns to canonical schema: {rename}")
        df = df.rename(columns=rename)

    if "Assay_Time" not in df.columns and ("Dmax_h" in df.columns or "DC50_h" in df.columns):
        dmax_h = pd.to_numeric(df["Dmax_h"], errors="coerce") if "Dmax_h" in df.columns \
            else pd.Series(np.nan, index=df.index)
        dc50_h = pd.to_numeric(df["DC50_h"], errors="coerce") if "DC50_h" in df.columns \
            else pd.Series(np.nan, index=df.index)
        df["Assay_Time"] = dmax_h.fillna(dc50_h)
        logger.info("Derived Assay_Time from Dmax_h (falling back to DC50_h where missing)")

    return df


def build_samples(df: pd.DataFrame) -> List[SampleInput]:
    """Convert dataframe rows into `SampleInput` objects for `EnsemblePredictor`."""
    def _val(row: pd.Series, col: str):
        if col not in row.index or pd.isna(row[col]):
            return None
        return row[col]

    return [
        SampleInput(
            smiles=_val(row, "SMILES"),
            poi_name=_val(row, "POI_Name"),
            poi_sequence=_val(row, "POI_Sequence"),
            ligase_name=_val(row, "Ligase_Name"),
            ligase_sequence=_val(row, "Ligase_Sequence"),
            cell_line=_val(row, "Cell_Line_ID"),
            assay_type=_val(row, "Assay"),
            treatment_time=_val(row, "Assay_Time"),
        )
        for _, row in df.iterrows()
    ]


def find_weights_file(ensemble_dir: Path) -> Optional[Path]:
    """Auto-detect a Caruana-style `ensemble_weights_*.json` inside *ensemble_dir*.

    Only matches when there is exactly one JSON file directly inside the
    directory (as produced by scripts/collect_ensemble_checkpoints.py);
    "best_arch"-style ensembles keep model checkpoints at the top level
    instead and are left unweighted (uniform averaging).
    """
    top_level_jsons = list(ensemble_dir.glob("*.json"))
    return top_level_jsons[0] if len(top_level_jsons) == 1 else None


def run_ensemble(
    task_key: str,
    ensemble_dir: Path,
    samples: List[SampleInput],
    weights_file: Optional[Path],
    embeddings_file: Optional[Path],
    device: str,
    n_jobs: Optional[int],
    batch_size: int,
) -> pd.DataFrame:
    """Load one ensemble and predict for every sample, in batches of *batch_size*.

    Returns a DataFrame of new ``{prefix}_*`` columns, row-aligned with
    *samples*, so callers can just ``pd.concat`` it onto the input DataFrame.
    """
    hparam_overrides = None
    if embeddings_file is not None and embeddings_file.exists():
        hparam_overrides = {
            "poi_embeddings_file": str(embeddings_file),
            "ligase_embeddings_file": str(embeddings_file),
        }

    weights_file = weights_file or find_weights_file(ensemble_dir)
    logger.info(f"Loading {task_key} ensemble from {ensemble_dir} (weights={weights_file})")
    if embeddings_file is not None:
        logger.info(
            f"  embeddings override {'applied' if hparam_overrides else 'NOT applied'} "
            f"(embeddings_file={embeddings_file}, exists={embeddings_file.exists()})"
        )

    t0 = time.time()
    predictor = EnsemblePredictor.from_directory(
        ensemble_dir,
        weights_file=weights_file,
        device=device,
        n_jobs=n_jobs,
        hparam_overrides=hparam_overrides,
        # Eager: load every model once upfront so the batch loop below reuses
        # them instead of reloading the whole ensemble from disk per batch.
        lazy_loading=False,
    )
    info = predictor.get_model_info()
    logger.info(f"  {info['n_models']} models loaded | available_tasks={info['available_tasks']} "
                f"({time.time() - t0:.1f}s)")

    n = len(samples)
    prefix = TASK_PREFIXES[task_key]
    cols = {
        "prediction": np.full(n, np.nan),
        "uncertainty": np.full(n, np.nan),
        "ci_lower_95": np.full(n, np.nan),
        "ci_upper_95": np.full(n, np.nan),
    }

    def _store(i: int, result: Optional[EnsemblePrediction]) -> None:
        if result is None:
            return
        cols["prediction"][i] = result.weighted_mean[0]
        cols["uncertainty"][i] = result.uncertainty_std[0]
        cols["ci_lower_95"][i] = result.ci_percentile_lower_95[0]
        cols["ci_upper_95"][i] = result.ci_percentile_upper_95[0]

    def _predict_row(offset: int, sample: SampleInput) -> bool:
        try:
            result = predictor.predict(sample, tasks=[task_key], verbose=False)
            _store(offset, result.get(task_key))
            return True
        except Exception as row_err:
            logger.warning(f"  Row {offset} failed for {task_key}: {row_err}")
            return False

    t0 = time.time()
    n_batches = (n + batch_size - 1) // batch_size
    n_row_errors = 0
    for b in tqdm(range(n_batches), desc=f"{prefix} batches"):
        start = b * batch_size
        end = min(start + batch_size, n)
        batch = samples[start:end]
        # A single malformed row (bad SMILES, missing required sequence, ...)
        # makes the whole batch call raise, so fall back to one-row-at-a-time
        # for just this batch -- the rest of the CSV still runs at full speed.
        try:
            batch_results = predictor.predict(batch, tasks=[task_key], verbose=False)
            for j, task_results in enumerate(batch_results):
                _store(start + j, (task_results or {}).get(task_key))
        except Exception as e:
            logger.warning(
                f"Batch {b + 1}/{n_batches} ({task_key}, rows {start}-{end - 1}) failed ({e}); "
                "falling back to one-row-at-a-time for this batch only."
            )
            for j, sample in enumerate(batch):
                if not _predict_row(start + j, sample):
                    n_row_errors += 1

    if n_row_errors:
        logger.warning(f"  {task_key}: {n_row_errors}/{n} rows failed even in per-row fallback.")

    n_ok = int(np.sum(~np.isnan(cols["prediction"])))
    logger.info(f"  {task_key}: {n_ok}/{n} rows predicted successfully "
                f"({n_batches} batch(es) of <={batch_size}, {time.time() - t0:.1f}s)")

    return pd.DataFrame({f"{prefix}_{k}": v for k, v in cols.items()})


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append TACK ensemble predictions (Dmax, DC50, binary activity) to an input CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-csv", type=Path, default=REPO_ROOT / "tack_2.csv",
                        help="Input CSV of PROTAC compounds (must contain a SMILES column).")
    parser.add_argument("--output-csv", type=Path,
                        default=REPO_ROOT / "tack_v2_with_predictions.csv",
                        help="Where to write the CSV with appended prediction columns.")
    parser.add_argument("--dmax-ensemble-dir", type=Path,
                        default=REPO_ROOT / "ensembles" / "dmax_caruana_ensemble",
                        help="Directory with the Dmax ensemble checkpoints + datamodule states.")
    parser.add_argument("--dc50-ensemble-dir", type=Path,
                        default=REPO_ROOT / "ensembles" / "dc50_caruana_ensemble",
                        help="Directory with the DC50 ensemble checkpoints + datamodule states.")
    parser.add_argument("--bin-ensemble-dir", type=Path,
                        default=REPO_ROOT / "ensembles" / "bin_caruana_ensemble",
                        help="Directory with the binary-activity ensemble checkpoints + datamodule states.")
    parser.add_argument("--dmax-weights", type=Path, default=None,
                        help="Ensemble weights JSON for Dmax (auto-detected inside --dmax-ensemble-dir if omitted).")
    parser.add_argument("--dc50-weights", type=Path, default=None,
                        help="Ensemble weights JSON for DC50 (auto-detected inside --dc50-ensemble-dir if omitted).")
    parser.add_argument("--bin-weights", type=Path, default=None,
                        help="Ensemble weights JSON for binary activity (auto-detected inside --bin-ensemble-dir if omitted).")
    parser.add_argument("--embeddings-file", type=Path, default=REPO_ROOT / "embeddings.npz",
                        help="Precomputed POI/ligase sequence embeddings (.npz) for embedding-based ensemble members.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Inference device (cuda/cpu).")
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="XGBoost thread count (default: all cores).")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Rows predicted per batch. Bounds the cost of the per-row fallback "
                             "to a single batch instead of the whole CSV when a row fails.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable DEBUG-level logging.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    _setup_logging(args.verbose)
    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be a positive integer, got {args.batch_size}")
    _log_environment(args)

    t_start = time.time()
    df = pd.read_csv(args.input_csv)
    logger.info(f"Loaded {len(df)} rows from {args.input_csv}")
    if "SMILES" not in df.columns:
        raise ValueError(f"Input CSV must contain a SMILES column. Found: {list(df.columns)}")

    df = canonicalize_columns(df).reset_index(drop=True)
    _log_data_quality(df)
    samples = build_samples(df)

    ensemble_specs = [
        ("dmax", args.dmax_ensemble_dir, args.dmax_weights),
        ("dc50", args.dc50_ensemble_dir, args.dc50_weights),
        ("bin", args.bin_ensemble_dir, args.bin_weights),
    ]

    result_df = df
    ran_any = False
    for task_key, ensemble_dir, weights_file in ensemble_specs:
        if not ensemble_dir.exists():
            logger.warning(f"Skipping {task_key}: directory not found: {ensemble_dir}")
            continue
        ran_any = True
        pred_df = run_ensemble(
            task_key=task_key,
            ensemble_dir=ensemble_dir,
            samples=samples,
            weights_file=weights_file,
            embeddings_file=args.embeddings_file,
            device=args.device,
            n_jobs=args.n_jobs,
            batch_size=args.batch_size,
        )
        result_df = pd.concat([result_df, pred_df], axis=1)

    if not ran_any:
        raise RuntimeError(
            "None of the three ensemble directories were found -- nothing to predict. "
            "Check --dmax-ensemble-dir/--dc50-ensemble-dir/--bin-ensemble-dir."
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.output_csv, index=False)
    logger.info(f"Saved {len(result_df)} rows ({len(result_df.columns)} columns) to {args.output_csv} "
                f"(total {time.time() - t_start:.1f}s)")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.getLogger("predict_tack_v2").exception("predict_tack_v2.py FAILED")
        sys.exit(1)
