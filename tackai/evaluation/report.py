"""
One-shot evaluation: discover runs, compute metrics, rank, and write a report.

:func:`run_evaluation` is the engine behind ``tack evaluate``. Given a run's
``predictions_dir`` (its ``tack train ... predictions_dir=...`` output, holding
``preds-model=...-task=...-group=...-fold=<N>-split=<val|test>.csv`` files —
see :mod:`tackai.evaluation.loader`), it:

1. discovers the available runs directly from those prediction filenames,
2. loads predictions into one tidy table,
3. computes per-fold metrics for every task (all metrics reported),
4. runs the ``autorank`` omnibus comparison grouped by CV strategy and model
   family (best + equivalent set, CD diagram),
5. renders the kept bespoke plots from :mod:`tackai.models_comparison`, and
6. writes ``report.md``, ``ranking.csv``, ``metrics.csv``, ``runs.csv``
   (and figures) to the output directory.

Heavy, optional dependencies (matplotlib, seaborn, autorank) are imported lazily
so that merely importing this module is cheap.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from tackai.evaluation.loader import discover_runs, dc50_to_pdc50, load_predictions, normalize_task

logger = logging.getLogger("tackai.evaluation.report")

CLASSIFICATION_CUTOFFS = {"Dmax": 80.0, "DC50": float(dc50_to_pdc50(100.0))}


# ---------------------------------------------------------------------------
# Plotting (each guarded; failures degrade gracefully)
# ---------------------------------------------------------------------------

def _save_cd_diagram(result, out_path: Path) -> Optional[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from tackai.evaluation.compare import plot_cd_diagram
        plot_cd_diagram(result)
        plt.savefig(out_path, bbox_inches="tight")
        plt.close("all")
        return out_path
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not render CD diagram %s: %s", out_path.name, exc)
        return None


def _save_boxplots(metrics_df: pd.DataFrame, metric_cols: Sequence[str], out_path: Path) -> Optional[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from tackai.models_comparison import make_boxplots_nonparametric
        usable = [m for m in metric_cols if m in metrics_df.columns and not metrics_df[m].isna().all()]
        if not usable or metrics_df["method"].nunique() < 2:
            return None
        # models_comparison's routines key the repeated-measures subject as
        # 'cv_cycle'; all_fold_metrics() names the same thing 'fold'.
        make_boxplots_nonparametric(metrics_df.rename(columns={"fold": "cv_cycle"}), usable)
        plt.savefig(out_path, bbox_inches="tight")
        plt.close("all")
        return out_path
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not render boxplots %s: %s", out_path.name, exc)
        return None


def _save_roc_pr(sub: pd.DataFrame, out_path: Path) -> Optional[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from tackai.models_comparison import make_curve_plots
        make_curve_plots(sub, val_col="target", prob_col="prob")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close("all")
        return out_path
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not render ROC/PR curves %s: %s", out_path.name, exc)
        return None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_evaluation(
    predictions_dir: Union[str, Path],
    output_dir: Union[str, Path],
    checkpoints_dir: Optional[Union[str, Path]] = None,
    tasks: Optional[Sequence[str]] = None,
    subset: str = "val",
    metric: Optional[str] = None,
    methods: Optional[Sequence[str]] = None,
    model_family: Optional[str] = None,
    alpha: float = 0.05,
    make_plots: bool = True,
) -> Dict[str, object]:
    """Evaluate all runs and write a report in one go.

    Args:
        predictions_dir: A run's predictions directory, holding
            ``preds-model=...-task=...-group=...-fold=<N>-split=<val|test>.csv``
            files (``tack train``'s ``predictions_dir=`` output).
        output_dir: Destination for ``report.md`` and artifacts.
        checkpoints_dir: A run's checkpoints directory, holding
            ``datamodule-*_hparams.yaml`` files, used to prettify method labels
            (see :func:`tackai.evaluation.loader.load_predictions`). Defaults to
            a sibling ``checkpoints/`` next to ``predictions_dir`` if one exists;
            pass ``pretty_labels=False``-equivalent by pointing this at a
            nonexistent path to keep the raw ``method`` strings instead.
        tasks: Normalized tasks to evaluate (``dmax``/``dc50``/``bin``; default all).
        subset: ``"val"`` or ``"test"`` — which split's *descriptive* metrics
            table, boxplot-subject numbers, and ROC/PR curves to show. The
            statistical ranking (autorank omnibus test + equivalent-set +
            CD diagram) always uses ``"val"`` regardless of this value, since
            the ``"test"`` set is identical across folds (see
            :func:`tackai.evaluation.compare.find_equivalent_best_set`).
        metric: Ranking metric key (default per task: RMSE / ROC-AUC).
        methods: Optional whitelist of exact method labels.
        model_family: Restrict to one model family: ``"xgb"`` or ``"mlp"``.
            Methods whose display name does not start with the corresponding
            prefix are excluded from ranking (but still present in metrics).
        alpha: Significance level for the omnibus comparison.
        make_plots: Whether to render figures.

    Returns:
        ``{output_dir, report, runs, ranking, metrics, per_task}``.
    """
    from tackai.evaluation.compare import all_fold_metrics, build_full_report, ALL_METRICS

    predictions_dir = Path(predictions_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    if make_plots:
        fig_dir.mkdir(parents=True, exist_ok=True)

    norm_tasks: Optional[List[str]] = None
    if tasks is not None:
        norm_tasks = [normalize_task(t) for t in tasks]

    # 1. Discover runs.
    runs = discover_runs(predictions_dir, checkpoints_dir=checkpoints_dir)
    runs.to_csv(output_dir / "runs.csv", index=False)
    logger.info("Discovered %d run(s).", len(runs))

    # 2. Load predictions.
    preds = load_predictions(
        predictions_dir, tasks=norm_tasks, methods=methods, checkpoints_dir=checkpoints_dir
    )

    # Apply model-family filter (xgb → "XGB…", mlp → "MLP…").
    if model_family is not None:
        prefix = model_family.upper()
        preds = preds[preds["method"].str.upper().str.startswith(prefix)].reset_index(drop=True)
        if preds.empty:
            logger.warning("No predictions remain after filtering for model_family=%r.", model_family)

    available_tasks = sorted(preds["task"].unique())
    eval_tasks = [t for t in (norm_tasks or available_tasks) if t in available_tasks]

    # 3-5. Per task: all-metrics table + ranking + plots.
    all_metrics_frames: List[pd.DataFrame] = []
    all_ranking_frames: List[pd.DataFrame] = []
    per_task: Dict[str, object] = {}
    figures_all: Dict[str, Dict[str, Optional[Path]]] = {}

    for task in eval_tasks:
        task_slug = task.lower()
        task_preds = preds[preds["task"] == task]
        n_methods = task_preds["method"].nunique() if not task_preds.empty else 0
        if n_methods == 0:
            logger.info("Skipping task %s (no predictions).", task)
            continue

        # Descriptive all-metrics table: follows the requested `subset`.
        metrics_df = all_fold_metrics(task_preds, task=task, subset=subset)
        if not metrics_df.empty:
            metrics_df.insert(0, "task", task)
            all_metrics_frames.append(metrics_df)

        # Statistical testing (ranking + Friedman/ANOVA-annotated boxplots) is
        # always computed on 'val' regardless of `subset` -- the 'test' set is
        # the same fixed rows scored by every fold's model, so folds aren't an
        # independent repeated measure there (see compare.find_equivalent_best_set).
        val_metrics_df = (
            metrics_df if subset == "val" else all_fold_metrics(task_preds, task=task, subset="val")
        )

        # Ranking (grouped by cv_group × model_family) — single autorank call.
        ranking = pd.DataFrame()
        equivalence: Dict[str, object] = {}
        figures: Dict[str, Optional[Path]] = {}

        if n_methods >= 2:
            ranking, equiv_map = build_full_report(
                task_preds, subsets=["val"], tasks=[task],
                metric=metric, alpha=alpha, return_equivalence=True,
            )
            if not ranking.empty:
                all_ranking_frames.append(ranking)

            # Use cached EquivalenceResult objects for CD diagrams (no re-ranking).
            for (s, cv_grp, fam, t), res in equiv_map.items():
                grp_label = cv_grp or "all"
                fam_label = fam or "all"
                key = f"{grp_label}/{fam_label}"
                equivalence[key] = res
                if make_plots:
                    cd_fname = f"cd_{task_slug}_{grp_label}_{fam_label}_{s}.png"
                    figures[f"cd_{key}"] = _save_cd_diagram(res, fig_dir / cd_fname)

        if make_plots and not val_metrics_df.empty:
            metric_cols = ALL_METRICS.get(task, [])
            bp_path = fig_dir / f"boxplots_{task_slug}_val.png"
            figures["boxplots"] = _save_boxplots(val_metrics_df, metric_cols, bp_path)
            if task == "bin":
                roc_path = fig_dir / f"roc_pr_{task_slug}_{subset}.png"
                figures["roc_pr"] = _save_roc_pr(
                    task_preds[task_preds["set"] == subset], roc_path
                )

        figures_all[task] = figures
        per_task[task] = {"ranking": ranking, "metrics": metrics_df,
                          "equivalence": equivalence, "figures": figures}

    # Write combined CSVs.
    metrics = pd.concat(all_metrics_frames, ignore_index=True) if all_metrics_frames else pd.DataFrame()
    ranking_all = pd.concat(all_ranking_frames, ignore_index=True) if all_ranking_frames else pd.DataFrame()
    if not metrics.empty:
        metrics.to_csv(output_dir / "metrics.csv", index=False)
    if not ranking_all.empty:
        ranking_all.to_csv(output_dir / "ranking.csv", index=False)

    # Write combined report.md.
    report_path = output_dir / "report.md"
    _write_report(report_path, runs, ranking_all, per_task, metrics, figures_all, subset, model_family)
    logger.info("Wrote evaluation report to %s", report_path)

    return {
        "output_dir": output_dir, "report": report_path,
        "runs": runs, "ranking": ranking_all, "metrics": metrics,
        "per_task": per_task,
    }


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def _write_report(
    path: Path,
    runs: pd.DataFrame,
    ranking: pd.DataFrame,
    per_task: Dict[str, object],
    metrics: pd.DataFrame,
    figures: Dict[str, Dict[str, Optional[Path]]],
    subset: str,
    model_family: Optional[str],
) -> None:
    from datetime import datetime

    lines: List[str] = []
    lines.append("# TACK evaluation report\n")
    fam_note = f" · model family: `{model_family}`" if model_family else ""
    lines.append(
        f"_Generated {datetime.now().isoformat(timespec='seconds')} "
        f"— descriptive metrics on the `{subset}` set{fam_note}._\n"
    )
    if subset != "val":
        lines.append(
            "_Statistical ranking (best/equivalent-set below) always uses the "
            "`val` set regardless of the above — see "
            "`tackai.evaluation.compare.find_equivalent_best_set`._\n"
        )

    # Runs summary: counts only, no file list.
    lines.append(f"\n**{len(runs)} run(s) found.**\n")

    # Per-task sections.
    for task, info in per_task.items():
        lines.append(f"\n## {task}\n")
        equivalence = info["equivalence"]
        task_metrics = info["metrics"]

        if equivalence:
            for grp_label, res in equivalence.items():
                lines.append(f"### CV group: {grp_label}\n")
                lines.append(
                    f"- **Best ({res.metric})**: `{res.best_method}` "
                    f"(omnibus p = {res.omnibus_pvalue:.3g})"
                )
                equ = [m for m in res.equivalent_methods if m != res.best_method]
                if equ:
                    lines.append("- Statistically equivalent: " + ", ".join(f"`{m}`" for m in equ))
                else:
                    lines.append("- No other method is statistically equivalent to the best.")
                try:
                    lines.append("\n" + res.per_method_table.round(4).to_markdown())
                except Exception:
                    pass
                lines.append("")
        else:
            lines.append("_No slice had >=2 comparable methods._\n")

        if not task_metrics.empty:
            metric_cols = [c for c in task_metrics.columns if c not in ("task", "method", "fold", "cv_group")]
            if metric_cols:
                summary = task_metrics.groupby("method")[metric_cols].agg(["mean", "std"]).round(4)
                lines.append(f"\n#### Metrics summary (mean ± std across folds, `{subset}` set)\n")
                lines.append(summary.to_markdown())
                lines.append("")

        task_figs = figures.get(task, {})
        if any(v is not None for v in task_figs.values()):
            for name, fig_path in task_figs.items():
                if fig_path is not None:
                    rel = Path("figures") / Path(fig_path).name
                    lines.append(f"![{task} {name}]({rel})")
            lines.append("")

    if not ranking.empty:
        lines.append("\n## Full ranking table\n")
        lines.append(ranking.round(4).to_markdown(index=False))
        lines.append("")

    path.write_text("\n".join(lines))
