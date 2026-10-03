"""Fast fusion surrogates for PROTAC degradation activity.

The two methods that won the fusion comparison (``notebooks/fusion_comparison.ipynb``):
an additive-kernel Gaussian process with cross-block product kernels, and regularised
gradient-boosted trees. Both run on pre-reduced context embeddings with no PCA applied
inside this pipeline.
"""
from tackai.fusion.blocks import (BLOCK_DIMS, BLOCK_ORDER, DENSE_BLOCKS, MOL_BLOCKS,
                                  SMALL_BLOCKS, BlockPreprocessor, block_index)

__all__ = ["BLOCK_ORDER", "BLOCK_DIMS", "DENSE_BLOCKS", "SMALL_BLOCKS", "MOL_BLOCKS",
           "BlockPreprocessor", "block_index"]
