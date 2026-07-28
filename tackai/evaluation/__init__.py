"""
Evaluation toolkit for TACK cross-validation experiments.

Public API (import directly, e.g. ``from tackai.evaluation import load_predictions``):

- Labeling / loading (:mod:`tackai.evaluation.loader`):
  ``feature_label``, ``method_label``, ``discover_runs``, ``load_predictions``,
  ``dc50_to_pdc50``.
- Statistical comparison (:mod:`tackai.evaluation.compare`, needs ``autorank``):
  ``build_full_report``, ``find_equivalent_best_set``, ``metric_pivot``,
  ``plot_cd_diagram``, ``EquivalenceResult``.
- Orchestration (:mod:`tackai.evaluation.report`): ``run_evaluation``.

``loader`` is imported eagerly (only pandas/numpy/PyYAML). ``compare`` and
``report`` are resolved lazily via ``__getattr__`` so importing this package
does not pull in ``autorank``/matplotlib until those features are actually used.
"""
from __future__ import annotations

from tackai.evaluation.loader import (
    dc50_to_pdc50,
    discover_runs,
    feature_label,
    load_predictions,
    method_label,
    normalize_task,
)

_LAZY = {
    "all_fold_metrics": "tackai.evaluation.compare",
    "build_full_report": "tackai.evaluation.compare",
    "find_equivalent_best_set": "tackai.evaluation.compare",
    "metric_pivot": "tackai.evaluation.compare",
    "plot_cd_diagram": "tackai.evaluation.compare",
    "EquivalenceResult": "tackai.evaluation.compare",
    "run_evaluation": "tackai.evaluation.report",
}

__all__ = [
    "all_fold_metrics", "dc50_to_pdc50", "discover_runs",
    "feature_label", "load_predictions", "method_label", "normalize_task",
    "build_full_report", "find_equivalent_best_set", "metric_pivot",
    "plot_cd_diagram", "EquivalenceResult", "run_evaluation",
]


def __getattr__(name: str):
    if name in _LAZY:
        import importlib
        mod = importlib.import_module(_LAZY[name])
        value = getattr(mod, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'tackai.evaluation' has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_LAZY))
