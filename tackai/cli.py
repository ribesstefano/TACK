"""
Command-line interface for TACK.

Exposes four subcommands through the ``tack`` console script:

- ``tack train`` — run nested cross-validation training. Accepts Hydra-style
  ``key=value`` overrides composed from ``configs/train.yaml`` (e.g.
  ``tack train model=mlp data=fp task=dmax``). For multi-run sweeps use the
  Hydra launcher directly: ``python scripts/train_models.py -m ...``.
- ``tack predict`` — run weighted ensemble inference on an input CSV. The CSV
  must contain a ``SMILES`` column; optional context columns (``POI_Name``,
  ``POI_Sequence``, ``Ligase_Name``, ``Cell_Line_ID``/``Cell_Line``,
  ``Assay_Time``) are auto-detected when present.
- ``tack collect`` — concatenate a training run's per-fold ``preds-*.csv``
  files into a single ``predictions.csv`` keyed by ``row_id``.
- ``tack evaluate`` — discover runs under a ``predictions_dir``, compute
  per-fold metrics, rank method+data configs via ``autorank`` (on the
  validation set), and write ``report.md``/``ranking.csv``/``metrics.csv``
  plus figures to an output directory.
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from tackai.ensemble_predictor import EnsemblePredictor

logger = logging.getLogger("tackai.cli")

CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")

# Candidate column names probed when a context column is not given explicitly.
_COLUMN_CANDIDATES = {
    "poi_col": ["POI_Name", "Target", "POI"],
    "poi_sequence_col": ["POI_Sequence", "Target_Sequence"],
    "ligase_col": ["Ligase_Name", "E3_Ligase", "E3"],
    "cell_line_col": ["Cell_Line_ID", "Cell_Line", "Cell"],
    "treatment_time_col": ["Assay_Time", "Treatment_Time", "Time"],
}


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------

def train_from_cfg(cfg: DictConfig) -> None:
    """Run a cross-validation training experiment from a composed config.

    Bridges the Hydra/OmegaConf config to the dict-based training pipeline:
    resolves the ``data`` and ``model`` groups to plain dicts, prepares the
    task dataset, and delegates to :func:`tackai.training.run_cv_experiment`.
    """
    import pytorch_lightning as pl

    from tackai.data.tasks import load_task_dataset
    from tackai.training import run_cv_experiment

    pl.seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("high")

    data_config = OmegaConf.to_container(cfg.data, resolve=True)
    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    model_type = model_config.pop("model_type")

    # Runtime overrides shared by every data config
    data_config["num_proc"] = cfg.num_proc
    data_config["batch_size"] = cfg.batch_size

    n_tuning_trials = cfg.n_tuning_trials
    if n_tuning_trials is None:
        n_tuning_trials = 20 if model_type == "xgboost" else 100

    dataset, held_out_dataset, labels = load_task_dataset(
        task=cfg.task,
        model_type=model_type,
        custom_dataset_csv=cfg.custom_dataset_csv,
        group=cfg.group,
        held_out_frac=cfg.held_out_frac,
        held_out_seed=cfg.held_out_seed,
    )

    checkpoints_dir = Path(cfg.checkpoint_dir)
    predictions_dir = Path(cfg.predictions_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    run_cv_experiment(
        dataset=dataset,
        held_out_dataset=held_out_dataset,
        model_type=model_type,
        data_config=data_config,
        model_config=model_config,
        task_name=cfg.task,
        labels=labels,
        group=cfg.group,
        checkpoints_dir=checkpoints_dir,
        results_dir=predictions_dir,
        tune_hyperparameters=cfg.tune_hyperparameters,
        n_tuning_trials=n_tuning_trials,
    )


def train(argv: Optional[List[str]] = None) -> None:
    """Compose ``configs/train.yaml`` with the given overrides and train.

    This is the single-run convenience path. For sweeps, call the Hydra
    launcher (``python scripts/train_models.py -m ...``) which supports
    ``--multirun`` natively.
    """
    argv = list(sys.argv[1:] if argv is None else argv)

    if "-h" in argv or "--help" in argv:
        print(
            "Usage: tack train [KEY=VALUE ...]\n\n"
            "Train CV models via Hydra config composition. Common overrides:\n"
            "  model=<stem>        a file under configs/model/ (e.g. mlp, xgboost)\n"
            "  data=<stem>         a file under configs/data/ (e.g. fp, simple)\n"
            "  task=<task>         dmax | dc50 | bin | dmax_bin | dc50_bin | multitask\n"
            "  group=<strategy>    random | scaffold | butina\n"
            "  batch_size=64 seed=42 num_proc=1\n"
            "  tune_hyperparameters=true n_tuning_trials=20\n"
            "  checkpoint_dir=./checkpoints predictions_dir=./predictions\n\n"
            "Inspect the merged config or run sweeps with the Hydra launcher:\n"
            "  python scripts/train_models.py --cfg job\n"
            "  python scripts/train_models.py -m model=xgboost,mlp data=fp,simple\n"
        )
        return

    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="train", overrides=argv)
    train_from_cfg(cfg)


# ----------------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------------

def _resolve_column(
    df: pd.DataFrame, given: Optional[str], candidates: List[str]
) -> Optional[str]:
    """Return the column to use: the explicit one, else the first candidate present."""
    if given is not None:
        return given
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def predict(argv: Optional[List[str]] = None) -> None:
    """Run weighted ensemble inference on an input CSV."""
    parser = argparse.ArgumentParser(
        prog="tack predict",
        description="Run ensemble inference on an input CSV (requires a SMILES column).",
    )
    model_source = parser.add_mutually_exclusive_group(required=True)
    model_source.add_argument("--checkpoints-dir",
                        help="Local directory with model checkpoints and datamodule states.")
    model_source.add_argument("--repo-id",
                        help="Hugging Face Hub repo id to load the ensemble from, "
                             "e.g. ailab-bio/TACK-ensembles.")
    parser.add_argument("--subfolder", default=None,
                        help="Subfolder within --repo-id holding one ensemble's "
                             "checkpoints, e.g. dmax_caruana. Ignored with --checkpoints-dir.")
    parser.add_argument("--input-csv", required=True,
                        help="Input CSV; must contain a SMILES column.")
    parser.add_argument("--output-csv", required=True,
                        help="Where to write predictions.")
    parser.add_argument("--weights", default=None,
                        help="Ensemble weights JSON; restricts inference to the listed models.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Inference device (cuda/cpu).")
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="Threads for XGBoost inference.")
    parser.add_argument("--smiles-col", default="SMILES",
                        help="Name of the SMILES column.")
    parser.add_argument("--poi-col", default=None,
                        help="POI/target name column (auto-detected if omitted).")
    parser.add_argument("--poi-sequence-col", default=None,
                        help="POI sequence column (auto-detected if omitted).")
    parser.add_argument("--ligase-col", default=None,
                        help="E3 ligase name column (auto-detected if omitted).")
    parser.add_argument("--cell-line-col", default=None,
                        help="Cell line column (auto-detected if omitted).")
    parser.add_argument("--treatment-time-col", default=None,
                        help="Treatment time column (auto-detected if omitted).")
    args = parser.parse_args(argv)

    df = pd.read_csv(args.input_csv)
    logger.info(f"Loaded {len(df)} rows from {args.input_csv}")

    # SMILES is required; fall back to any SMILES-like column.
    smiles_col = args.smiles_col
    if smiles_col not in df.columns:
        smiles_like = [c for c in df.columns if "smiles" in c.lower()]
        if not smiles_like:
            raise ValueError(
                f"Required SMILES column '{args.smiles_col}' not found. "
                f"Available columns: {list(df.columns)}"
            )
        smiles_col = smiles_like[0]
        logger.info(f"Using detected SMILES column: {smiles_col}")

    resolved = {
        key: _resolve_column(df, getattr(args, key), candidates)
        for key, candidates in _COLUMN_CANDIDATES.items()
    }

    predictor = EnsemblePredictor.from_pretrained(
        args.checkpoints_dir if args.checkpoints_dir else args.repo_id,
        subfolder=args.subfolder,
        weights_file=args.weights,
        device=args.device,
        n_jobs=args.n_jobs,
    )

    info = predictor.get_model_info()
    logger.info(
        f"Loaded {info['n_models']} models | tasks={info['available_tasks']} "
        f"| device={info['device']}"
    )

    # Warn about columns the loaded models expect but the CSV does not provide.
    required_cols, _ = predictor.get_required_inputs()
    missing = sorted(c for c in required_cols if c and c not in df.columns)
    if missing:
        logger.warning(
            f"Input CSV is missing columns the models expect: {missing}. "
            "Affected features fall back to defaults."
        )

    result_df = predictor.predict_dataframe(
        df=df,
        smiles_col=smiles_col,
        poi_col=resolved["poi_col"],
        poi_sequence_col=resolved["poi_sequence_col"],
        ligase_col=resolved["ligase_col"],
        cell_line_col=resolved["cell_line_col"],
        treatment_time_col=resolved["treatment_time_col"],
    )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(result_df)} predictions to {output_path}")


# ----------------------------------------------------------------------------
# Collect predictions
# ----------------------------------------------------------------------------

def collect(argv: Optional[List[str]] = None) -> None:
    """Collapse per-fold prediction CSVs into one file per model/task/config."""
    from tackai.training import collect_predictions

    parser = argparse.ArgumentParser(
        prog="tack collect",
        description="Collapse per-fold preds-*.csv files into one file per "
                    "model/task/config (folds and splits become columns).",
    )
    parser.add_argument("--predictions-dir", required=True,
                        help="Directory holding the preds-*.csv files.")
    parser.add_argument("--pattern", default="preds-*-fold=*-split=*.csv",
                        help="Glob selecting the per-fold files to gather.")
    parser.add_argument("--keep-sources", action="store_true",
                        help="Keep the per-fold files instead of removing them "
                             "after the combined file is written.")
    args = parser.parse_args(argv)

    collect_predictions(
        predictions_dir=Path(args.predictions_dir),
        remove_sources=not args.keep_sources,
        pattern=args.pattern,
    )


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------

def evaluate(argv: Optional[List[str]] = None) -> None:
    """Discover runs, compute metrics, rank via autorank, and write a report."""
    from tackai.evaluation.report import run_evaluation

    parser = argparse.ArgumentParser(
        prog="tack evaluate",
        description="Evaluate a training run's predictions: discover model+data "
                    "configs, compute per-fold metrics, rank them via autorank "
                    "(on the validation set), and write a report.",
    )
    parser.add_argument("--predictions-dir", required=True,
                        help="A run's predictions directory: preds-*.csv files, "
                             "per-fold and/or collapsed by `tack collect`.")
    parser.add_argument("--output-dir", required=True,
                        help="Destination for report.md, ranking.csv, metrics.csv, "
                             "runs.csv, and figures/.")
    parser.add_argument("--checkpoints-dir", default=None,
                        help="A run's checkpoints directory (datamodule-*_hparams.yaml "
                             "files), used to prettify method labels. Defaults to a "
                             "sibling 'checkpoints/' next to --predictions-dir.")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Tasks to evaluate (dmax/dc50/bin). Default: all present.")
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Exact method-label whitelist. Default: all discovered.")
    parser.add_argument("--subset", default="val", choices=["val", "test"],
                        help="Which split's descriptive metrics/plots to report. "
                             "Statistical ranking always uses 'val'. Default: val.")
    parser.add_argument("--metric", default=None,
                        help="Ranking metric key. Default per task: RMSE / ROC-AUC.")
    parser.add_argument("--model-family", default=None,
                        help="Restrict ranking to one family, e.g. 'xgb' or 'mlp'.")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level for the omnibus comparison.")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip rendering figures (boxplots/CD diagrams/ROC-PR).")
    args = parser.parse_args(argv)

    result = run_evaluation(
        predictions_dir=args.predictions_dir,
        output_dir=args.output_dir,
        checkpoints_dir=args.checkpoints_dir,
        tasks=args.tasks,
        subset=args.subset,
        metric=args.metric,
        methods=args.methods,
        model_family=args.model_family,
        alpha=args.alpha,
        make_plots=not args.no_plots,
    )
    logger.info(f"Wrote evaluation report to: {result['report']}")


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def main() -> None:
    """``tack`` console entry point: dispatch to ``train``/``predict``/``collect``/``evaluate``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        prog="tack",
        description="TACK training and ensemble inference CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "train", add_help=False,
        help="Train CV models (Hydra overrides, e.g. model=mlp data=fp task=dmax).",
    )
    subparsers.add_parser(
        "predict", add_help=False,
        help="Run ensemble inference on an input CSV.",
    )
    subparsers.add_parser(
        "collect", add_help=False,
        help="Concatenate a run's per-fold prediction CSVs into one file.",
    )
    subparsers.add_parser(
        "evaluate", add_help=False,
        help="Rank a run's model+data configs via autorank and write a report.",
    )

    args, extra = parser.parse_known_args()
    if args.command == "train":
        train(extra)
    elif args.command == "predict":
        predict(extra)
    elif args.command == "collect":
        collect(extra)
    elif args.command == "evaluate":
        evaluate(extra)


if __name__ == "__main__":
    main()
