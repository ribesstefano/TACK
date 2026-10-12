"""Fast fusion surrogates for PROTAC degradation activity.

The two methods that won the fusion comparison (``notebooks/fusion_comparison.ipynb``):
an additive-kernel Gaussian process with cross-block product kernels, and regularised
gradient-boosted trees. Both run on pre-reduced context embeddings with no PCA applied
inside this pipeline.

:class:`FusionData` plus :class:`~tackai.fusion.xgb_ensemble.XGBStackedEnsemble` are the
default, minimal-dependency path: build a design matrix with the former and fit or serve
gradient-boosted members with the latter, neither of which needs ``torch``. The GP surrogate
(:class:`GPInteraction` / :class:`AdditiveProductGP`) and the generic
:class:`FusionEnsemble` that imports it do need ``torch`` (the ``gp`` extra); importing
either name below only loads ``torch`` the first time it is actually accessed, so
``import tackai.fusion`` itself never requires it.
"""
from tackai.fusion.context import (CONTEXT_FILES, DEFAULT_CONTEXT_REPO, ContextEncoder,
                                   normalize_assay)
from tackai.fusion.data import (BLOCK_ORDER, TASK_LABELS, TASK_TYPES, TASKS, FusionData, build_table,
                                make_targets, scaffold_groups)
from tackai.fusion.training import check_labels, fit_member, validation_split
from tackai.fusion.mol_encoder import DESCRIPTOR_NAMES, MolEncoder
from tackai.fusion.xgb_ensemble import XGBStackedEnsemble

# PEP 562 lazy-import map: public name -> defining module. tackai.fusion.gp imports torch
# at module level, and tackai.fusion.ensemble imports tackai.fusion.gp, so both stay out of
# the eager imports above — otherwise `import tackai.fusion` would require torch just to
# reach FusionData / XGBStackedEnsemble, the names that don't need it.
_LAZY: dict[str, str] = {
    "AdditiveProductGP": "tackai.fusion.gp",
    "GPInteraction":     "tackai.fusion.gp",
    "FusionEnsemble":    "tackai.fusion.ensemble",
    "FusionPrediction":  "tackai.fusion.ensemble",
    "FusionContext":     "tackai.fusion.ensemble",
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib
        mod = importlib.import_module(_LAZY[name])
        value = getattr(mod, name)
        globals()[name] = value  # cache so subsequent accesses bypass __getattr__
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(globals()) + list(_LAZY)


__all__ = ["BLOCK_ORDER", "MolEncoder", "DESCRIPTOR_NAMES",
           "ContextEncoder", "CONTEXT_FILES", "DEFAULT_CONTEXT_REPO", "normalize_assay",
           "FusionData", "build_table", "make_targets", "scaffold_groups", "TASKS",
           "TASK_TYPES", "TASK_LABELS", "XGBStackedEnsemble",
           "AdditiveProductGP", "GPInteraction",
           "FusionEnsemble", "FusionPrediction", "FusionContext",
           "check_labels", "validation_split", "fit_member"]
