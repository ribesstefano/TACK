"""Block layout of the fusion design matrix and its fold-internal preprocessor.

The design matrix is the horizontal concatenation of seven blocks in :data:`BLOCK_ORDER`.
No PCA is applied anywhere in this pipeline: the biological context arrives already
reduced from ``TACKAI_CACHE`` (see :mod:`tackai.fusion.context`) and the molecular blocks
stay raw, so trees split on individual Morgan bits and the GP sees untransformed features.
"""
from typing import Dict, Optional, Sequence

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

BLOCK_ORDER = ["fingerprint", "descriptors", "e3", "cell", "poi", "assay", "assay_time"]

#: Blocks that are imputed, optionally standardised, and divided by ``sqrt(width)``.
DENSE_BLOCKS = ["fingerprint", "descriptors", "e3", "cell", "poi", "assay"]
#: Blocks passed through at their own scale (the GP gives them a linear kernel).
SMALL_BLOCKS = ["assay_time"]
#: The molecular blocks, which receive no scaler at all.
MOL_BLOCKS = ["fingerprint", "descriptors"]

BLOCK_DIMS = {
    "fingerprint": 1024,   # Morgan r=16, includeChirality=True
    "descriptors": 217,    # every rdkit Descriptors._descList entry
    "e3": 7,               # esm2_150M layer 30, mean_rm_pc2, PCA 7
    "cell": 47,            # all-mpnet-base-v1 mean pooling, PCA 47
    "poi": 51,             # esm2_150M layer 18, lse, PCA 51
    "assay": 8,            # all-mpnet-base-v1 mean pooling, PCA 8
    "assay_time": 1,       # mean(DC50_h, Dmax_h)
}


def block_index(dims: Dict[str, int]) -> Dict[str, np.ndarray]:
    """Column indices of every block, laid out contiguously in :data:`BLOCK_ORDER`.

    Args:
        dims: Mapping of block name to its width.

    Returns:
        Mapping of block name to an integer array of column indices.
    """
    out, start = {}, 0
    for b in BLOCK_ORDER:
        width = dims[b]
        out[b] = np.arange(start, start + width)
        start += width
    return out


class BlockPreprocessor(BaseEstimator):
    """Per-block imputation and scaling, fitted on the rows passed to :meth:`fit` only.

    Dense blocks are mean-imputed and divided by ``sqrt(width)`` so that no block dominates
    a shared-lengthscale kernel by width alone; the blocks named in ``scale_blocks`` are
    additionally standardised. The molecular blocks are deliberately left unstandardised:
    Morgan bits are already on one scale, and standardising them would let a bit set in a
    handful of rows dominate every distance. ``assay_time`` is median-imputed and standardised.

    Args:
        blocks: Mapping of block name to column indices (default: contiguous
            :data:`BLOCK_DIMS` layout).
        scale_blocks: Dense blocks that are standardised (default: every dense block except
            :data:`MOL_BLOCKS`).
    """

    def __init__(self, blocks: Optional[Dict[str, np.ndarray]] = None,
                 scale_blocks: Optional[Sequence[str]] = None):
        self.blocks = blocks
        self.scale_blocks = scale_blocks

    def fit(self, X, y=None) -> "BlockPreprocessor":
        """Fit the per-block imputers and scalers on ``X``.

        Args:
            X: Design matrix of shape ``(n_rows, n_columns)``.
            y: Ignored; present for scikit-learn compatibility.

        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float64)
        self.blocks_ = {b: np.asarray(i) for b, i in (self.blocks or block_index(BLOCK_DIMS)).items()}
        scaled = self._selected_scale_blocks()
        self.steps_, self.dims_ = {}, {}
        for b, idx in self.blocks_.items():
            Xb = X[:, idx]
            if b in DENSE_BLOCKS:
                imp = SimpleImputer(strategy="mean", keep_empty_features=True).fit(Xb)
                sc = self._fit_scaler(imp.transform(Xb)) if b in scaled else None
                self.steps_[b] = (imp, sc, np.sqrt(Xb.shape[1]))
            else:
                imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(Xb)
                self.steps_[b] = (imp, self._fit_scaler(imp.transform(Xb)), 1.0)
            self.dims_[b] = Xb.shape[1]
        return self

    @staticmethod
    def _fit_scaler(Z) -> StandardScaler:
        """Standardiser whose zero-variance columns map to 0.0 instead of +/-inf."""
        sc = StandardScaler().fit(Z)
        sc.scale_ = np.where(sc.scale_ == 0, 1.0, sc.scale_)
        return sc

    def _selected_scale_blocks(self) -> set:
        """The dense blocks to standardise; unknown names are an error."""
        default = [b for b in DENSE_BLOCKS if b not in MOL_BLOCKS]
        chosen = set(default if self.scale_blocks is None else self.scale_blocks)
        unknown = chosen - set(DENSE_BLOCKS)
        if unknown:
            raise ValueError(f"scale_blocks must be dense blocks {DENSE_BLOCKS}, got {sorted(unknown)}")
        return chosen

    def transform(self, X) -> Dict[str, np.ndarray]:
        """Processed blocks of ``X``.

        Args:
            X: Design matrix with the columns this preprocessor was fitted on.

        Returns:
            Mapping of block name to its processed ``float64`` array.
        """
        return self.transform_blocks(X, only=self.blocks_)

    def transform_blocks(self, X, only: Sequence[str]) -> Dict[str, np.ndarray]:
        """Processed values of a subset of the blocks.

        Used by the ensemble's context cache, which transforms the context blocks once and
        only pushes the molecular blocks through the preprocessor for every new batch.

        Args:
            X: Design matrix with the columns this preprocessor was fitted on.
            only: Names of the blocks to transform.

        Returns:
            Mapping of the requested block names to their processed arrays.
        """
        X = np.asarray(X, dtype=np.float64)
        out = {}
        for b in only:
            imp, sc, width = self.steps_[b]
            Z = imp.transform(X[:, self.blocks_[b]])
            if sc is not None:
                Z = sc.transform(Z)
            out[b] = Z / width
        return out

    def concat(self, Z: Dict[str, np.ndarray], names: Optional[Sequence[str]] = None) -> np.ndarray:
        """Concatenate processed blocks in :data:`BLOCK_ORDER`.

        Args:
            Z: Mapping of block name to processed array.
            names: Blocks to concatenate (default: those present in ``Z``, in block order).

        Returns:
            One ``float64`` array of shape ``(n_rows, sum of widths)``.
        """
        names = list(names or [b for b in BLOCK_ORDER if b in Z])
        return np.concatenate([Z[b] for b in names], axis=1)
