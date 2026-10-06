"""Fast fusion surrogates for PROTAC degradation activity.

The two methods that won the fusion comparison (``notebooks/fusion_comparison.ipynb``):
an additive-kernel Gaussian process with cross-block product kernels, and regularised
gradient-boosted trees. Both run on pre-reduced context embeddings with no PCA applied
inside this pipeline.
"""
from tackai.fusion.context import (CONTEXT_FILES, DEFAULT_CONTEXT_REPO, ContextEncoder,
                                   normalize_assay)
from tackai.fusion.data import (BLOCK_ORDER, TASK_LABELS, TASK_TYPES, TASKS, FusionData, build_table,
                                make_targets, scaffold_groups)
from tackai.fusion.gp import AdditiveProductGP
from tackai.fusion.models import FusionEstimator, GPInteraction, XGBoostFusion
from tackai.fusion.training import check_labels, fit_member, validation_split
from tackai.fusion.ensemble import FusionContext, FusionEnsemble, FusionPrediction
from tackai.fusion.mol_encoder import DESCRIPTOR_NAMES, MolEncoder

__all__ = ["BLOCK_ORDER", "MolEncoder", "DESCRIPTOR_NAMES",
           "ContextEncoder", "CONTEXT_FILES", "DEFAULT_CONTEXT_REPO", "normalize_assay",
           "FusionData", "build_table", "make_targets", "scaffold_groups", "TASKS",
           "TASK_TYPES", "TASK_LABELS", "AdditiveProductGP", "FusionEstimator", "GPInteraction", "XGBoostFusion",
           "FusionEnsemble", "FusionPrediction", "FusionContext",
           "check_labels", "validation_split", "fit_member"]
