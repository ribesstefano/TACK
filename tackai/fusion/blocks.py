"""Block layout of the fusion design matrix and its fold-internal preprocessor.

The design matrix is the horizontal concatenation of seven blocks in :data:`BLOCK_ORDER`.
No PCA is applied anywhere in this pipeline: the biological context arrives already
reduced from ``TACKAI_CACHE`` (see :mod:`tackai.fusion.context`) and the molecular blocks
stay raw, so trees split on individual Morgan bits and the GP sees untransformed features.
No block is standardised either: the context is already centred by its PCA.
"""
import copy
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

#: What the first version of the pipeline standardised. Kept so the superseded behaviour stays
#: reachable (and testable) through ``scale_blocks``; nothing uses it by default.
LEGACY_SCALE_BLOCKS = ("e3", "cell", "poi", "assay", "assay_time")

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
    """Per-block imputation and ``1/sqrt(width)`` scaling, fitted on the rows given to :meth:`fit`.

    Nothing is standardised by default. The context blocks arrive PCA-reduced, hence already
    centred; the molecular blocks must stay raw so a tree can split on an individual Morgan
    bit and a GP kernel sees untransformed descriptors; and ``assay_time`` feeds a linear
    kernel that learns its own scale. Dense blocks are mean-imputed and divided by
    ``sqrt(width)`` so no block dominates a shared-lengthscale kernel by width alone;
    ``assay_time`` is median-imputed. Imputation statistics are fitted in float64 (a raw
    ``Ipc`` near 1e18 would lose its mean in float32) and applied in ``dtype``.

    Args:
        blocks: Mapping of block name to column indices (default: contiguous
            :data:`BLOCK_DIMS` layout).
        scale_blocks: Blocks to standardise (default: none). :data:`LEGACY_SCALE_BLOCKS` is
            the set the first version used.
        dtype: Floating-point type of :meth:`transform`'s output (``"float32"`` or
            ``"float64"``). A preprocessor unpickled without this attribute behaves as
            ``"float64"``, which is what it was fitted under.
    """

    def __init__(self, blocks: Optional[Dict[str, np.ndarray]] = None,
                 scale_blocks: Optional[Sequence[str]] = None, dtype: str = "float32"):
        self.blocks = blocks
        self.scale_blocks = scale_blocks
        self.dtype = dtype

    def _dtype(self) -> np.dtype:
        """The output dtype; float64 for an object pickled before ``dtype`` existed."""
        return np.dtype(getattr(self, "dtype", "float64"))

    def fit(self, X, y=None) -> "BlockPreprocessor":
        """Fit the per-block imputers (and any requested scalers) on ``X``.

        Args:
            X: Design matrix of shape ``(n_rows, n_columns)``.
            y: Ignored; present for scikit-learn compatibility.

        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float64)
        self.blocks_ = {b: np.asarray(i) for b, i in
                        (self.blocks or block_index(BLOCK_DIMS)).items()}
        scaled = self._selected_scale_blocks()
        self.steps_, self.dims_ = {}, {}
        for b, idx in self.blocks_.items():
            Xb = X[:, idx]
            dense = b in DENSE_BLOCKS
            imp = SimpleImputer(strategy="mean" if dense else "median",
                                keep_empty_features=True).fit(Xb)
            sc = self._fit_scaler(imp.transform(Xb)) if b in scaled else None
            self.steps_[b] = (imp, sc, np.sqrt(Xb.shape[1]) if dense else 1.0)
            self.dims_[b] = Xb.shape[1]
        return self

    @staticmethod
    def _fit_scaler(Z) -> StandardScaler:
        """Standardiser whose zero-variance columns map to 0.0 instead of +/-inf."""
        sc = StandardScaler().fit(Z)
        sc.scale_ = np.where(sc.scale_ == 0, 1.0, sc.scale_)
        return sc

    def _selected_scale_blocks(self) -> set:
        """The blocks to standardise; unknown names are an error."""
        chosen = set(() if self.scale_blocks is None else self.scale_blocks)
        unknown = chosen - set(DENSE_BLOCKS) - set(SMALL_BLOCKS)
        if unknown:
            raise ValueError(f"scale_blocks must be among {DENSE_BLOCKS + SMALL_BLOCKS}, "
                             f"got {sorted(unknown)}")
        return chosen

    def _columns(self, block: str):
        """Where a block lives in the design matrix: a slice (a view) when it is a run."""
        idx = self.blocks_[block]
        if len(idx) and np.all(np.diff(idx) == 1):
            return slice(int(idx[0]), int(idx[-1]) + 1)
        return idx

    def transform_block(self, block: str, Xb) -> np.ndarray:
        """Process one block from an array holding only that block's columns.

        One copy, then in-place imputation and division. The copy is never skipped: the
        result must not alias the caller's rows, and it must be C-contiguous because it goes
        straight into BLAS matmuls, which are not bit-reproducible across memory layouts.

        Args:
            block: Block name.
            Xb: Array of shape ``(n_rows, width)``.

        Returns:
            Fresh C-contiguous array of this preprocessor's dtype.
        """
        imp, sc, width = self.steps_[block]
        dtype = self._dtype()
        Z = np.array(Xb, dtype=dtype, order="C")
        missing = np.isnan(Z)
        if missing.any():
            rows, cols = np.nonzero(missing)
            Z[rows, cols] = imp.statistics_.astype(dtype)[cols]
        if sc is not None:
            Z -= sc.mean_.astype(dtype)
            Z /= sc.scale_.astype(dtype)
        Z /= dtype.type(width)
        return Z

    def transform(self, X) -> Dict[str, np.ndarray]:
        """Processed blocks of ``X``.

        Args:
            X: Design matrix with the columns this preprocessor was fitted on.

        Returns:
            Mapping of block name to its processed array.
        """
        return self.transform_blocks(X, only=self.blocks_)

    def transform_blocks(self, X, only: Sequence[str]) -> Dict[str, np.ndarray]:
        """Processed values of a subset of the blocks.

        Args:
            X: Design matrix with the columns this preprocessor was fitted on.
            only: Names of the blocks to transform.

        Returns:
            Mapping of the requested block names to their processed arrays.
        """
        X = np.asarray(X)
        return {b: self.transform_block(b, X[:, self._columns(b)]) for b in only}

    def transform_signature(self, block: str) -> tuple:
        """Everything that makes :meth:`transform_block` member-specific on NaN-free input.

        The imputer is left out on purpose: it only acts on NaN. Two preprocessors with equal
        signatures produce identical output for a NaN-free block, so the ensemble can
        transform it once for both.

        Args:
            block: Block name.

        Returns:
            A hashable tuple.
        """
        _, sc, width = self.steps_[block]
        scaler = None if sc is None else (sc.mean_.tobytes(), sc.scale_.tobytes())
        return (str(self._dtype()), float(width), scaler)

    @classmethod
    def consensus(cls, preprocessors: Sequence["BlockPreprocessor"],
                  only: Optional[Sequence[str]] = None) -> "BlockPreprocessor":
        """One preprocessor whose statistics are the mean of several fitted ones.

        Members fitted on different folds disagree about the statistics of the *same* context.
        The consensus gives every member one answer, independent of which rows it was fitted
        on. Blocks outside ``only`` keep the first preprocessor's steps and should not be used.

        Args:
            preprocessors: Fitted preprocessors, all with the same layout.
            only: Blocks to average (default: all).

        Returns:
            A new fitted preprocessor; the inputs are not modified.

        Raises:
            ValueError: If a block's columns, width or scaler presence differ.
        """
        first = preprocessors[0]
        merged = copy.copy(first)
        merged.steps_ = dict(first.steps_)
        for block in (list(only) if only is not None else list(first.blocks_)):
            imp0, sc0, width0 = first.steps_[block]
            for pre in preprocessors[1:]:
                imp, sc, width = pre.steps_[block]
                if not np.array_equal(pre.blocks_[block], first.blocks_[block]):
                    raise ValueError(f"block {block!r}: members read different columns of the "
                                     "design matrix, so there is no common context to share")
                if width != width0 or (sc is None) != (sc0 is None):
                    raise ValueError(f"block {block!r}: members disagree on its width or on "
                                     "whether it is standardised; refit them with one setting")
            imp_c = copy.deepcopy(imp0)
            imp_c.statistics_ = np.mean([p.steps_[block][0].statistics_
                                         for p in preprocessors], axis=0)
            sc_c = None
            if sc0 is not None:
                sc_c = copy.deepcopy(sc0)
                sc_c.mean_ = np.mean([p.steps_[block][1].mean_ for p in preprocessors], axis=0)
                sc_c.scale_ = np.mean([p.steps_[block][1].scale_ for p in preprocessors],
                                      axis=0)
                sc_c.var_ = sc_c.scale_ ** 2
            merged.steps_[block] = (imp_c, sc_c, width0)
        return merged

    def concat(self, Z: Dict[str, np.ndarray], names: Optional[Sequence[str]] = None) -> np.ndarray:
        """Concatenate processed blocks in :data:`BLOCK_ORDER`.

        Args:
            Z: Mapping of block name to processed array.
            names: Blocks to concatenate (default: those present in ``Z``, in block order).

        Returns:
            One array of shape ``(n_rows, sum of widths)``.
        """
        names = list(names or [b for b in BLOCK_ORDER if b in Z])
        return np.concatenate([Z[b] for b in names], axis=1)
