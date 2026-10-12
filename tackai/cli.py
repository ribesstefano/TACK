"""
Command-line interface for TACK.

Exposes five subcommands through the ``tack`` console script:

- ``tack train`` — run nested cross-validation training. Accepts Hydra-style
  ``key=value`` overrides composed from ``configs/train.yaml`` (e.g.
  ``tack train model=mlp data=fp task=dmax``). ``tack train`` uses the Hydra
  compose API directly, not the multirun launcher, so for sweeps use a shell
  loop over ``tack train`` (see ``configs/train.yaml``'s header comment).
- ``tack predict`` — run weighted ensemble inference on an input CSV. The CSV
  must contain a ``SMILES`` column; optional context columns (``POI_Name``,
  ``POI_Sequence``, ``Ligase_Name``, ``Cell_Line_ID``/``Cell_Line``,
  ``Assay_Time``) are auto-detected when present.
- ``tack serve`` — serve one fusion ensemble (``tackai.fusion``) over HTTP, so a client
  in a *different* Python environment can score molecules without installing ``tackai``
  beside its own dependencies. Unlike every other subcommand this needs the ``serve``
  extra, not ``training``; see :mod:`tackai.serve` for why a server rather than a
  per-call subprocess.
- ``tack collect`` — concatenate a training run's per-fold ``preds-*.csv``
  files into a single ``predictions.csv`` keyed by ``row_id``.
- ``tack evaluate`` — discover runs under a ``predictions_dir``, compute
  per-fold metrics, rank method+data configs via ``autorank`` (on the
  validation set), and write ``report.md``/``ranking.csv``/``metrics.csv``
  plus figures to an output directory.

Every subcommand needs an extra a plain ``pip install tackai`` does not bring in:
``training`` (``pip install "tackai[training]"``) for all of them but ``tack serve``,
which needs ``serve``. So that ``tack --help`` and the argument parsers still work
without either, nothing heavier than ``pandas`` is imported at module level: each
subcommand imports its dependencies on entry, through :func:`_training_stack`, which
turns a missing one into an actionable install message naming the right extra.
"""
import argparse
import contextlib
import json
import logging
import os
import socket
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, List, Optional

import pandas as pd

if TYPE_CHECKING:
    from omegaconf import DictConfig

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


@contextlib.contextmanager
def _training_stack(command: str, extra: str = "training") -> Iterator[None]:
    """Turn a missing optional dependency into an actionable install message.

    Wrap the imports a subcommand needs; an ``ImportError`` raised inside becomes a
    ``SystemExit`` naming the extra to install instead of a bare traceback.

    Args:
        command: Subcommand name, for the message (e.g. ``"train"``).
        extra: Name of the extra that provides the missing dependencies. Defaults to
            ``"training"``, which covers every subcommand but ``serve``.
    """
    try:
        yield
    except ImportError as e:
        raise SystemExit(
            f"`tack {command}` needs the optional {extra} dependencies ({e}). "
            f'Install them with:  pip install "tackai[{extra}]"'
        ) from e


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------

def train_from_cfg(cfg: "DictConfig") -> None:
    """Run a cross-validation training experiment from a composed config.

    Bridges the Hydra/OmegaConf config to the dict-based training pipeline:
    resolves the ``data`` and ``model`` groups to plain dicts, prepares the
    task dataset, and delegates to :func:`tackai.training.run_cv_experiment`.
    """
    with _training_stack("train"):
        import pytorch_lightning as pl
        import torch
        from omegaconf import OmegaConf

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

    This is a single-run entry point (Hydra compose API, not the multirun
    launcher). For sweeps, loop over ``tack train`` invocations from the
    shell — see ``configs/train.yaml``'s header comment for an example.
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
            "This uses the Hydra compose API directly, not the multirun launcher.\n"
            "For sweeps, loop over `tack train` from the shell, e.g.:\n"
            "  for enc in minmax onehot embedding; do\n"
            "      tack train model=xgboost data=fp task=dmax data.categorical_encoding=$enc\n"
            "  done\n"
        )
        return

    with _training_stack("train"):
        from hydra import compose, initialize_config_dir

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
    parser.add_argument("--device", default=None,
                        help="Inference device (cuda/cpu). Default: cuda if available, "
                             "else cpu.")
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

    with _training_stack("predict"):
        import torch

        from tackai.ensemble_predictor import EnsemblePredictor

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

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
        device=device,
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
# Serve
# ----------------------------------------------------------------------------

def _load_context(spec: Optional[str]) -> Optional[dict]:
    """Read a default context from a JSON file, or from a literal JSON object.

    Accepts both because the two uses differ: a file is what a reproducible run
    script wants, a literal is what you type while trying the server out.

    Args:
        spec: Path to a JSON file, or a JSON object starting with ``{``. ``None``
            returns ``None``.

    Returns:
        The parsed mapping, or ``None``.
    """
    if spec is None:
        return None
    text = spec if spec.lstrip().startswith("{") else Path(spec).read_text()
    context = json.loads(text)
    if not isinstance(context, dict):
        raise SystemExit(f"--context must be a JSON object, got {type(context).__name__}.")
    return context


def serve(argv: Optional[List[str]] = None) -> None:
    """Serve one fusion ensemble over HTTP for out-of-environment clients."""
    parser = argparse.ArgumentParser(
        prog="tack serve",
        description="Serve a fusion ensemble over HTTP, so a client in another Python "
                    "environment (e.g. REINVENT) can score molecules without installing "
                    "tackai beside its own dependencies.",
    )
    model_source = parser.add_mutually_exclusive_group(required=True)
    model_source.add_argument("--checkpoints-dir",
                        help="Local directory holding the ensemble's model files.")
    model_source.add_argument("--repo-id",
                        help="Hugging Face Hub repo id to load the ensemble from, "
                             "e.g. ailab-bio/TACK-FastEnsemble.")
    parser.add_argument("--task", action="append", dest="tasks",
                        choices=["dmax", "pdc50", "activity"],
                        help="Task to serve; repeat for several. Each is loaded from the "
                             "subfolder of the same name, so the task name is the "
                             "subfolder. Default: all three, which costs about 18%% more "
                             "per request than one because they share a featurisation; "
                             "name a single task if you only ever read one.")
    parser.add_argument("--context", default=None,
                        help="Default experimental context as a JSON file path or a literal "
                             "JSON object, with keys poi_seq/e3_seq/cell_id/assay/assay_time. "
                             "Lets clients post only SMILES, which is the normal arrangement "
                             "for a generative run whose context never changes.")
    context_source = parser.add_mutually_exclusive_group()
    context_source.add_argument("--context-repo", default=None,
                        help="Hub dataset repo or local directory holding the context "
                             "embedding tables (default: ailab-bio/TACK-Context-Encoding). "
                             "Its manifest records the protein embedding space the tables "
                             "were built in, which is what keeps the design matrix the width "
                             "the published ensembles expect. Cached files are reused.")
    context_source.add_argument("--local-context", action="store_true",
                        help="Read the context tables straight from TACKAI_CACHE with no "
                             "manifest and no network access, for an offline compute node. "
                             "The protein space then comes from --protein-space, and a wrong "
                             "choice is what produces a feature-count mismatch.")
    parser.add_argument("--protein-space", default=None, choices=["per_block", "combined"],
                        help="POI/E3 embedding space, which fixes those blocks' widths. "
                             "Normally inferred from the context manifest; set it explicitly "
                             "only with --local-context, or to override the manifest.")
    parser.add_argument("--token", default=None,
                        help="Hugging Face token, for private repos. Prefer the HF_TOKEN "
                             "environment variable, which avoids the token reaching the "
                             "process list.")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind address (default: 127.0.0.1). Loopback keeps the endpoint "
                             "off the network — it has no authentication, so on a shared node "
                             "0.0.0.0 would expose it to every other user.")
    parser.add_argument("--port", type=int, default=8777,
                        help="Bind port (default: 8777). Pass 0 to let the OS choose a free "
                             "one and combine with --port-file to tell the client about it.")
    parser.add_argument("--port-file", default=None,
                        help="Write the bound port to this file once listening. The file is "
                             "the handshake for --port 0, so several jobs on one node need no "
                             "port bookkeeping.")
    parser.add_argument("--uds", default=None,
                        help="Listen on this Unix domain socket path instead of a TCP port, "
                             "sidestepping port allocation entirely on a shared node. "
                             "Mutually exclusive with --port-file.")
    parser.add_argument("--n-workers", type=int, default=1,
                        help="Featuriser worker processes (default: 1). Raise only for very "
                             "large batches; each worker re-imports rdkit.")
    parser.add_argument("--log-level", default="info",
                        help="uvicorn log level (default: info).")
    args = parser.parse_args(argv)

    if args.uds and args.port_file:
        parser.error("--uds and --port-file are mutually exclusive: a Unix socket has no port.")

    # Must precede any xgboost/torch import: the two bundle separate libomp copies that
    # segfault when both run multi-threaded in one process (see CLAUDE.md).
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    context = _load_context(args.context)

    with _training_stack("serve", extra="serve"):
        import uvicorn

        from tackai.fusion.context import DEFAULT_CONTEXT_REPO
        from tackai.fusion.data import TASKS
        from tackai.serve.app import create_app, load_service
        from tackai.serve.schemas import ContextSpec

    # The published ensembles and the published context tables are a matched pair: the
    # tables' manifest records the protein space the models were trained in. Defaulting
    # to the bare local cache instead silently builds a narrower matrix, which xgboost
    # then rejects on every request.
    if args.local_context:
        context_repo = None
    else:
        context_repo = args.context_repo or DEFAULT_CONTEXT_REPO

    tasks = args.tasks or list(TASKS)
    try:
        service = load_service(
            model=args.checkpoints_dir or args.repo_id,
            tasks=tasks,
            context=ContextSpec(**context) if context is not None else None,
            context_repo=context_repo,
            protein_space=args.protein_space,
            token=args.token,
            n_workers=args.n_workers,
        )
    except ValueError as e:
        # A layout or context mismatch: report it as a message, not a traceback, since
        # every one of them is fixed by a different flag rather than by a code change.
        raise SystemExit(f"tack serve: {e}") from e
    app = create_app(service)
    config = uvicorn.Config(app, log_level=args.log_level, workers=1)

    if args.uds:
        config.uds = args.uds
        logger.info(f"Serving {', '.join(tasks)} on unix:{args.uds}")
        uvicorn.Server(config).run()
        return

    # Bind before handing the socket to uvicorn so --port 0 can be resolved to the real
    # port and published; uvicorn alone only reports it to the log.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((args.host, args.port))
    except OSError as e:
        raise SystemExit(
            f"Could not bind {args.host}:{args.port} ({e}). Another `tack serve` may "
            "already hold it — pass --port 0 --port-file PATH to pick a free one."
        ) from e
    port = sock.getsockname()[1]
    if args.port_file:
        port_path = Path(args.port_file)
        port_path.parent.mkdir(parents=True, exist_ok=True)
        port_path.write_text(str(port))
    logger.info(f"Serving {', '.join(tasks)} on http://{args.host}:{port} (docs at /docs)")
    try:
        uvicorn.Server(config).run(sockets=[sock])
    finally:
        if args.port_file:
            Path(args.port_file).unlink(missing_ok=True)


# ----------------------------------------------------------------------------
# Collect predictions
# ----------------------------------------------------------------------------

def collect(argv: Optional[List[str]] = None) -> None:
    """Collapse per-fold prediction CSVs into one file per model/task/config."""
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

    with _training_stack("collect"):
        from tackai.training import collect_predictions

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

    with _training_stack("evaluate"):
        from tackai.evaluation.report import run_evaluation

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
        "serve", add_help=False,
        help="Serve a fusion ensemble over HTTP for out-of-environment clients.",
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
    elif args.command == "serve":
        serve(extra)
    elif args.command == "collect":
        collect(extra)
    elif args.command == "evaluate":
        evaluate(extra)


if __name__ == "__main__":
    main()
