"""Exact Gaussian process with additive block kernels and cross-block product kernels.

This is M4 of the fusion comparison, written directly in torch rather than through GPyTorch
for one reason: **the pairwise squared distances never change during the marginal-likelihood
fit** — only the lengthscales do. Caching them per block turns every Adam step into an
elementwise ``exp`` plus a Cholesky, which matters now that no PCA runs upstream and the
fingerprint block is 1024 columns wide instead of 64.

Blocks listed in ``ard_blocks`` are the exception: per-column lengthscales make the distance
depend on the parameters, and an ``N x N x D`` cache is not affordable, so those blocks
recompute ``X / lengthscale`` distances each step. That is cheap for the one block that needs
it (217 descriptor columns, one matmul) and is exactly what keeps a raw ``Ipc`` column from
swamping the kernel.

The covariance is

    K = sum_b s_b . RBF_b  +  s_lin . <x_small, x_small>  +  sum_(a*b) s_ab . K_a . K_b

where the molecular kernel of a product term is ``RBF_fingerprint + RBF_descriptors``, as in
the notebook.
"""
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch

#: The blocks whose sum forms the molecular kernel of a product term.
MOL_KERNEL_BLOCKS = ("fingerprint", "descriptors")
DEFAULT_INTERACTIONS = ("mol*poi", "mol*cell", "poi*cell")

#: Smallest observation noise the likelihood may use. The development table measures the same
#: compound in the same context repeatedly, so K has exactly repeated rows and is singular;
#: without a floor the optimiser shrinks the noise until the Cholesky fails outright. The
#: GPyTorch original carried the same constraint (``GreaterThan(1e-3)``).
NOISE_FLOOR = 1e-3

#: The precisions a GP can run in.
SUPPORTED_DTYPES = {"float32": torch.float32, "float64": torch.float64}


def resolve_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    """A supported precision as a torch dtype; anything else is refused up front.

    Args:
        dtype: ``"float32"``, ``"float64"`` or the matching torch dtype.

    Returns:
        The torch dtype.

    Raises:
        ValueError: For any other precision, before any state has been changed.
    """
    name = str(dtype).replace("torch.", "")
    if name not in SUPPORTED_DTYPES:
        raise ValueError(f"dtype must be float32 or float64, got {dtype!r}")
    return SUPPORTED_DTYPES[name]


#: Where each precision's jitter ladder starts and the largest value it tries before giving up
#: (float32 then promotes to float64, whose ladder is the original). float32's epsilon is 1.2e-7,
#: so the float64 starting jitter of 1e-6 would sit below the rounding noise of a unit-scale kernel.
JITTER_START = {torch.float32: 1e-5, torch.float64: 1e-6}
JITTER_CEILING = {torch.float32: 1e-3, torch.float64: 1e-2}


def _ladder(start: float, ceiling: float):
    """Jitter values ``start, 10*start, ...`` up to ``ceiling``; ``0 * 10`` is 0, so restart at 1e-8."""
    jitter = start
    while jitter <= ceiling * (1 + 1e-9):
        yield jitter
        jitter = max(jitter * 10, 1e-8)


def _try_cholesky(K: torch.Tensor, start: float, ceiling: float):
    """Cholesky factor of ``K + jitter*I`` for the first jitter that works, else None."""
    eye = torch.eye(K.shape[0], dtype=K.dtype)
    for jitter in _ladder(start, ceiling):
        try:
            return torch.linalg.cholesky(K + eye * jitter)
        except Exception:
            continue
    return None


class AdditiveProductGP:
    """Exact GP over processed blocks, with cached distances and exact posterior variance.

    Args:
        dims: Mapping of block name to width (as produced by ``BlockPreprocessor.dims_``).
        interactions: Product kernel terms, each ``"<a>*<b>"`` where a side is either a block
            name or ``"mol"`` (the sum of the molecular RBFs).
        ard_blocks: Blocks given one lengthscale per column.
        rbf_blocks: Blocks given an RBF kernel (default: every block except ``linear_blocks``).
        linear_blocks: Blocks given a linear kernel.
        jitter: Starting diagonal jitter for the Cholesky factorisation (default: 1e-5 in
            float32, 1e-6 in float64).
        dtype: Torch dtype for the computation, float32 by default. A factorisation float32
            cannot do is retried in float64 and recorded in ``promoted_``.
    """

    def __init__(self, dims: Dict[str, int], *,
                 interactions: Sequence[str] = DEFAULT_INTERACTIONS,
                 ard_blocks: Sequence[str] = ("descriptors",),
                 rbf_blocks: Optional[Sequence[str]] = None,
                 linear_blocks: Sequence[str] = ("assay_time",),
                 jitter: Optional[float] = None,
                 dtype: Union[str, torch.dtype] = torch.float32):
        self.dims = dict(dims)
        self.linear_blocks = [b for b in linear_blocks if b in self.dims]
        self.rbf_blocks = list(rbf_blocks) if rbf_blocks is not None else [
            b for b in self.dims if b not in self.linear_blocks]
        self.ard_blocks = [b for b in ard_blocks if b in self.rbf_blocks]
        self.interactions = [t for t in interactions if self._term_available(t)]
        self.jitter = jitter
        self.dtype = resolve_dtype(dtype)
        self.promoted_ = False
        self.promotions_ = 0
        self.factorisations_ = 0
        self.terms = ([f"rbf:{b}" for b in self.rbf_blocks]
                      + [f"linear:{b}" for b in self.linear_blocks]
                      + [f"prod:{t}" for t in self.interactions])

    def _base_jitter(self, dtype: Optional[torch.dtype] = None) -> float:
        """Starting jitter: the user's, else the precision's default."""
        return self.jitter if self.jitter is not None else JITTER_START[dtype or self.dtype]

    # ---------------------------------------------------------------- layout helpers

    def _sides(self, term: str) -> Tuple[str, str]:
        a, b = term.split("*")
        return a, b

    def _term_available(self, term: str) -> bool:
        """True when both sides of a product term have blocks present in ``dims``."""
        for side in self._sides(term):
            needed = MOL_KERNEL_BLOCKS if side == "mol" else (side,)
            if not all(b in self.dims for b in needed):
                return False
        return True

    # ---------------------------------------------------------------- parameters

    def _init_params(self, rng: np.random.Generator) -> Dict[str, torch.Tensor]:
        """Random starting point for one restart, in log space."""
        params = {}
        for b in self.rbf_blocks:
            size = self.dims[b] if b in self.ard_blocks else 1
            value = rng.normal(0.0, 0.4, size=size)
            params[f"ls:{b}"] = torch.tensor(value, dtype=self.dtype, requires_grad=True)
        offset = np.log(max(len(self.terms), 1))
        for term in self.terms:
            params[f"sc:{term}"] = torch.tensor(rng.normal(0.0, 0.4) - offset,
                                                dtype=self.dtype, requires_grad=True)
        params["noise"] = torch.tensor(np.log(0.5 - NOISE_FLOOR), dtype=self.dtype,
                                       requires_grad=True)
        params["mean"] = torch.tensor(0.0, dtype=self.dtype, requires_grad=True)
        return params

    def _column_scales(self, Z: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Per-column spread of every ARD block, used to anchor its lengthscales.

        Measured once on the fit rows and carried in the fitted state, so a model reloaded or
        re-conditioned on other rows keeps the parameterisation it was fitted with.
        """
        scales = {}
        for block in self.ard_blocks:
            column = np.asarray(Z[block], dtype=np.float64)
            spread = column.std(axis=0)
            spread = np.where(spread > 0, spread, 1.0)      # a constant column needs no scaling
            scales[block] = torch.as_tensor(spread, dtype=self.dtype)
        return scales

    @staticmethod
    def _noise(params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Observation noise, never below :data:`NOISE_FLOOR`."""
        return NOISE_FLOOR + torch.exp(params["noise"])

    # ---------------------------------------------------------------- kernel pieces

    def _to_tensor(self, Z: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        return {b: torch.as_tensor(np.asarray(Z[b]), dtype=self.dtype) for b in self.dims}

    @staticmethod
    def _sq_dists(A: torch.Tensor, B: torch.Tensor, same: bool = False,
                  b_sq: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Pairwise squared euclidean distances, clamped at zero for numerical safety.

        Args:
            A: Left rows.
            B: Right rows.
            same: ``A`` and ``B`` are the same rows. The diagonal is then forced to exactly
                zero and the matrix symmetrised: ``|a|^2 + |b|^2 - 2a.b`` cancels to absolute
                error ``eps * |a|^2``, which for a raw descriptor column near 1e18 is ~1e19
                instead of 0, and an RBF whose self-covariance is 0 rather than 1 is not a
                valid kernel at all.
            b_sq: Precomputed ``(B * B).sum(1)``; identical to computing it here.

        Returns:
            Squared distance matrix.
        """
        b2 = (B * B).sum(1) if b_sq is None else b_sq
        d2 = (A * A).sum(1)[:, None] + b2[None, :] - 2.0 * (A @ B.T)
        d2 = d2.clamp_min(0.0)
        if same:
            d2 = 0.5 * (d2 + d2.T)
            d2 = d2 - torch.diag(d2.diagonal())
        return d2

    def _cache_distances(self, Za: Dict[str, torch.Tensor], Zb: Dict[str, torch.Tensor],
                         same: bool = False, train: Optional[dict] = None) -> Dict[str, torch.Tensor]:
        """Squared distances of the isotropic RBF blocks, computed once per (Za, Zb) pair.

        ARD blocks are absent on purpose: their distances depend on the lengthscales and are
        recomputed per step in :meth:`_rbf`.
        """
        return {b: self._sq_dists(Za[b], Zb[b], same=same,
                                  b_sq=None if train is None else train["sq"][b])
                for b in self.rbf_blocks if b not in self.ard_blocks}

    def _rbf(self, block: str, params: Dict[str, torch.Tensor],
             Za: Dict[str, torch.Tensor], Zb: Dict[str, torch.Tensor],
             cached: Dict[str, torch.Tensor], same: bool = False,
             train: Optional[dict] = None) -> torch.Tensor:
        """RBF kernel of one block, from the cached distances or recomputed for ARD."""
        ls = self._lengthscale(block, params)
        if block in self.ard_blocks:
            if train is None:
                right, right_sq = Zb[block] / ls, None
            else:
                right, right_sq = train["ard"][block], train["sq"][block]
            d2 = self._sq_dists(Za[block] / ls, right, same=same, b_sq=right_sq)
            return torch.exp(-0.5 * d2)
        return torch.exp(-0.5 * cached[block] / (ls[0] ** 2))

    def _lengthscale(self, block: str, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Lengthscale of a block, anchored to the spread of its own columns.

        For an ARD block the raw parameter multiplies the column spread measured at fit time,
        so ``x / lengthscale`` starts out order 1 whatever the column's units are. Without
        that anchor a raw descriptor column near 1e18 divided by a lengthscale near 1 makes
        the squared-distance cancellation lose every bit of the result. The data itself is
        never rescaled; only the parameterisation is.
        """
        raw = torch.exp(params[f"ls:{block}"])
        scale = self.ls_scale_.get(block) if hasattr(self, "ls_scale_") else None
        return raw if scale is None else raw * scale

    def _assemble(self, params: Dict[str, torch.Tensor],
                  Za: Dict[str, torch.Tensor], Zb: Dict[str, torch.Tensor],
                  cached: Dict[str, torch.Tensor], same: bool = False,
                  train: Optional[dict] = None) -> torch.Tensor:
        """The full covariance between two sets of rows (noise-free)."""
        rbf = {b: self._rbf(b, params, Za, Zb, cached, same=same, train=train)
               for b in self.rbf_blocks}
        total = None
        for b in self.rbf_blocks:
            term = torch.exp(params[f"sc:rbf:{b}"]) * rbf[b]
            total = term if total is None else total + term
        for b in self.linear_blocks:
            term = torch.exp(params[f"sc:linear:{b}"]) * (Za[b] @ Zb[b].T)
            total = term if total is None else total + term
        for inter in self.interactions:
            left, right = self._sides(inter)
            k_left = self._side_kernel(left, rbf)
            k_right = self._side_kernel(right, rbf)
            term = torch.exp(params[f"sc:prod:{inter}"]) * k_left * k_right
            total = term if total is None else total + term
        return total

    @staticmethod
    def _side_kernel(side: str, rbf: Dict[str, torch.Tensor]) -> torch.Tensor:
        """One side of a product term: a block's RBF, or the molecular sum."""
        if side == "mol":
            return sum(rbf[b] for b in MOL_KERNEL_BLOCKS)
        return rbf[side]

    def _diag(self, params: Dict[str, torch.Tensor], Z: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Prior variance of every row (the diagonal of the covariance with itself)."""
        rbf = {b: torch.ones(len(next(iter(Z.values()))), dtype=self.dtype)
               for b in self.rbf_blocks}      # RBF(x, x) == 1 for every block
        total = None
        for b in self.rbf_blocks:
            term = torch.exp(params[f"sc:rbf:{b}"]) * rbf[b]
            total = term if total is None else total + term
        for b in self.linear_blocks:
            term = torch.exp(params[f"sc:linear:{b}"]) * (Z[b] * Z[b]).sum(1)
            total = term if total is None else total + term
        for inter in self.interactions:
            left, right = self._sides(inter)
            term = (torch.exp(params[f"sc:prod:{inter}"])
                    * self._side_kernel(left, rbf) * self._side_kernel(right, rbf))
            total = term if total is None else total + term
        return total

    def _factor(self, K: torch.Tensor):
        """Cholesky factor with escalating jitter, promoting float32 to float64 if it must.

        Args:
            K: Noisy kernel matrix.

        Returns:
            ``(L, promoted)``; ``L`` is None when even float64 cannot factorise ``K``.
        """
        self.factorisations_ = getattr(self, "factorisations_", 0) + 1
        L = _try_cholesky(K, self._base_jitter(K.dtype), JITTER_CEILING[K.dtype])
        if L is not None or K.dtype == torch.float64:
            return L, False
        L = _try_cholesky(K.to(torch.float64), self._base_jitter(torch.float64),
                          JITTER_CEILING[torch.float64])
        if L is not None:
            self.promotions_ = getattr(self, "promotions_", 0) + 1
        return L, L is not None

    def _cholesky(self, K: torch.Tensor) -> torch.Tensor:
        """Cholesky factor of the noisy kernel; raises if even float64 cannot factorise it."""
        L, promoted = self._factor(K)
        if L is None:
            raise RuntimeError("GP kernel is not positive definite even with "
                               f"{JITTER_CEILING[torch.float64]:g} jitter in float64")
        self.promoted_ = promoted
        return L

    # ---------------------------------------------------------------- fitting

    def _neg_log_mll(self, params, Z, y, cached) -> torch.Tensor:
        """Negative exact log marginal likelihood.

        Returns a non-finite value rather than raising when the kernel cannot be factorised
        even with escalated jitter, so one bad step discards its restart instead of aborting
        a whole ensemble fit.
        """
        n = len(y)
        eye = torch.eye(n, dtype=self.dtype)
        K = self._assemble(params, Z, Z, cached, same=True) + eye * self._noise(params)
        L, _ = self._factor(K)
        if L is None:
            return torch.tensor(float("inf"), dtype=self.dtype)
        resid = (y - params["mean"]).unsqueeze(1).to(L.dtype)
        alpha = torch.cholesky_solve(resid, L)
        return (0.5 * (resid * alpha).sum() + torch.log(torch.diagonal(L)).sum()
                + 0.5 * n * np.log(2.0 * np.pi))

    def fit(self, Z: Dict[str, np.ndarray], y, *, n_restarts: int = 3, n_iter: int = 60,
            lr: float = 0.1, seed: int = 0, max_hyper_points: int = 1200,
            state: Optional[dict] = None) -> dict:
        """Fit the hyper-parameters, then condition the GP on every training row.

        The marginal likelihood is optimised on at most ``max_hyper_points`` rows (the exact
        GP that follows uses all of them), which is what makes repeated fits affordable.

        Args:
            Z: Processed blocks of the training rows.
            y: Training targets.
            n_restarts: Random restarts of the marginal-likelihood optimisation.
            n_iter: Adam steps per restart.
            lr: Adam learning rate.
            seed: Seed for the restarts.
            max_hyper_points: Rows used for the hyper-parameter fit.
            state: Reuse these hyper-parameters instead of searching.

        Returns:
            The fitted hyper-parameter state, suitable for :meth:`load_state`.
        """
        y = np.asarray(y, dtype=float)
        if state is None:
            self.ls_scale_ = self._column_scales(Z)
            rng = np.random.default_rng(seed)
            n = len(y)
            sub = (np.arange(n) if n <= max_hyper_points
                   else np.sort(rng.choice(n, max_hyper_points, replace=False)))
            Z_sub = self._to_tensor({b: np.asarray(Z[b])[sub] for b in self.dims})
            y_sub = torch.as_tensor(y[sub], dtype=self.dtype)
            cached = self._cache_distances(Z_sub, Z_sub, same=True)
            best = (np.inf, None)
            for restart in range(n_restarts):
                torch.manual_seed(seed * 1000 + restart)
                params = self._init_params(np.random.default_rng(seed * 1000 + restart))
                opt = torch.optim.Adam(params.values(), lr=lr)
                for _ in range(n_iter):
                    opt.zero_grad()
                    loss = self._neg_log_mll(params, Z_sub, y_sub, cached)
                    if not torch.isfinite(loss):
                        break          # this restart has wandered off; keep the best so far
                    loss.backward()
                    opt.step()
                with torch.no_grad():
                    final = float(self._neg_log_mll(params, Z_sub, y_sub, cached))
                if np.isfinite(final) and final < best[0]:
                    best = (final, {k: v.detach().clone() for k, v in params.items()})
            if best[1] is None:
                raise RuntimeError("GP marginal-likelihood optimisation diverged in every restart")
            state = {"params": best[1], "neg_mll": best[0], "n_hyper": int(len(y_sub)),
                     "ls_scale": self.ls_scale_}
        self.load_state(state, Z, y)
        return state

    def load_state(self, state: dict, Z: Dict[str, np.ndarray], y) -> None:
        """Condition on ``(Z, y)`` with given hyper-parameters, without refitting them.

        Args:
            state: A state returned by :meth:`fit`.
            Z: Processed blocks of the training rows.
            y: Training targets.
        """
        y = np.asarray(y, dtype=float)
        self.state_ = state
        self.ls_scale_ = state.get("ls_scale", {})
        self.params_ = {k: v.detach().clone() for k, v in state["params"].items()}
        self.Z_train_ = self._to_tensor(Z)
        self.y_train_ = torch.as_tensor(y, dtype=self.dtype)
        self.n_train_ = len(y)
        self.n_hyper_ = int(state.get("n_hyper", len(y)))
        self.noise_ = float(self._noise(self.params_))
        with torch.no_grad():
            cached = self._cache_distances(self.Z_train_, self.Z_train_, same=True)
            K = self._assemble(self.params_, self.Z_train_, self.Z_train_, cached, same=True)
            K = K + torch.eye(self.n_train_, dtype=self.dtype) * self.noise_
            self.chol_ = self._cholesky(K)
            resid = (self.y_train_ - self.params_["mean"]).unsqueeze(1).to(self.chol_.dtype)
            self.alpha_ = torch.cholesky_solve(resid, self.chol_).to(self.dtype)
        self._train_side_ = self._build_train_side()

    def _build_train_side(self) -> dict:
        """Quantities of the training rows that no query batch changes.

        The hyper-parameters are frozen once the GP is conditioned, so the row norms of the
        isotropic blocks and the lengthscale-scaled ARD blocks are the same for every batch
        and every member call. They are exactly the values ``_sq_dists`` would recompute.
        """
        with torch.no_grad():
            sq, ard = {}, {}
            for b in self.rbf_blocks:
                if b in self.ard_blocks:
                    scaled = self.Z_train_[b] / self._lengthscale(b, self.params_)
                    ard[b], sq[b] = scaled, (scaled * scaled).sum(1)
                else:
                    sq[b] = (self.Z_train_[b] * self.Z_train_[b]).sum(1)
            return {"sq": sq, "ard": ard}

    def _train_side(self) -> dict:
        """The train-side cache; rebuilt lazily for a GP pickled before it existed."""
        if self.__dict__.get("_train_side_") is None:
            self._train_side_ = self._build_train_side()
        return self._train_side_

    def astype(self, dtype: Union[str, torch.dtype], jitter: Optional[float] = None):
        """Re-condition this fitted GP in another precision, keeping its hyper-parameters.

        The kernel is rebuilt and refactorised in ``dtype``. ``promoted_`` is True when float32
        could not factorise it and float64 did; False only means float32 succeeded, possibly
        after adding jitter up to 1e-3. Promotion refactorises the kernel as assembled in
        ``dtype``, so it cannot undo rounding already in that kernel, and ``alpha_`` is stored
        in ``dtype`` either way.

        The cast overwrites the stored training rows and hyper-parameters, so it cannot be
        undone: float64 -> float32 -> float64 is not the original model (reload it from disk
        for that).

        Args:
            dtype: ``"float32"`` or ``"float64"`` (or the torch dtype).
            jitter: Starting jitter (default: the new precision's own).

        Raises:
            ValueError: For any other precision, before anything is changed.

        Returns:
            self
        """
        dtype = resolve_dtype(dtype)
        state = self.state_
        cast = {**state,
                "params": {k: v.to(dtype) for k, v in state["params"].items()},
                "ls_scale": {k: v.to(dtype) for k, v in state.get("ls_scale", {}).items()}}
        Z = {b: a.double().numpy() for b, a in self.Z_train_.items()}
        y = self.y_train_.double().numpy()
        self.dtype, self.jitter = dtype, jitter
        self.load_state(cast, Z, y)
        return self

    # ---------------------------------------------------------------- prediction

    def kernel_matrix(self, Za: Dict[str, np.ndarray], Zb: Dict[str, np.ndarray]) -> np.ndarray:
        """Noise-free covariance between two sets of rows, with the fitted hyper-parameters.

        Args:
            Za: Processed blocks of the first set.
            Zb: Processed blocks of the second set.

        Returns:
            Covariance matrix of shape ``(len(Za), len(Zb))``.
        """
        with torch.no_grad():
            ta, tb = self._to_tensor(Za), self._to_tensor(Zb)
            same = Za is Zb or all(np.array_equal(np.asarray(Za[b]), np.asarray(Zb[b]))
                                   for b in self.dims)
            cached = self._cache_distances(ta, tb, same=same)
            return self._assemble(self.params_, ta, tb, cached, same=same).numpy()

    def predict(self, Z: Dict[str, np.ndarray], return_std: bool = False):
        """Posterior mean, and optionally the posterior standard deviation.

        Args:
            Z: Processed blocks of the rows to predict.
            return_std: Also return the epistemic standard deviation (noise-free).

        Returns:
            ``mean`` or ``(mean, std)``, each of shape ``(len(Z),)``.
        """
        with torch.no_grad():
            Zt = self._to_tensor(Z)
            train = self._train_side()
            cross = self._assemble(self.params_, Zt, self.Z_train_,
                                   self._cache_distances(Zt, self.Z_train_, train=train),
                                   train=train)
            mean = (self.params_["mean"] + (cross @ self.alpha_).squeeze(1)).numpy()
            if not return_std:
                return mean
            v = torch.linalg.solve_triangular(self.chol_, cross.T.to(self.chol_.dtype), upper=False)
            var = self._diag(self.params_, Zt) - (v * v).sum(0)
            return mean, torch.sqrt(var.clamp_min(0.0)).numpy()

    # ---------------------------------------------------------------- fixed-context fast path

    def fold_context(self, context: Dict[str, np.ndarray]) -> dict:
        """Collapse every context-only part of the kernel against the training rows.

        With the context held fixed, the covariance between a test row and training row ``j``
        is

            K(i, j) = sum_(b in mol) s_b . RBF_b(i, j)  +  mol(i, j) . w_j  +  c_j

        where ``mol`` is the unweighted molecular kernel ``RBF_fp + RBF_desc``, ``w_j`` gathers
        the product terms with one molecular side, and ``c_j`` gathers the four context RBFs,
        the linear block and any product whose two sides are both context. A batch then pays
        only for the two molecular distance matrices instead of all ten terms.

        Args:
            context: One encoded context, as ``{block: (1, dim) array}`` for the context blocks.

        Returns:
            A dict with ``const`` and ``mol_weight`` (both of length ``n_train``), a
            ``block_weight`` vector per molecular block (for interactions that name one block
            rather than ``"mol"``) and the scalar ``prior_var``, to pass to
            :meth:`predict_in_context`.

        Raises:
            ValueError: If an interaction has molecular kernels on both sides, which cannot be
                reduced to a per-row scalar.
        """
        for inter in self.interactions:
            left, right = self._sides(inter)
            if self._is_mol_side(left) and self._is_mol_side(right):
                raise ValueError(
                    f"interaction {inter!r} has a molecular kernel on both sides and cannot be "
                    "folded into a fixed context; score it through predict() instead")

        ctx_blocks = [b for b in self.dims if b not in MOL_KERNEL_BLOCKS]
        missing = [b for b in ctx_blocks if b not in context]
        if missing:
            raise ValueError(f"fold_context needs every context block; missing {missing}")

        with torch.no_grad():
            train = self._train_side()
            Zc = {b: torch.as_tensor(np.asarray(context[b]), dtype=self.dtype)
                  for b in ctx_blocks}
            cached = {b: self._sq_dists(Zc[b], self.Z_train_[b], b_sq=train["sq"][b])
                      for b in self.rbf_blocks
                      if b not in self.ard_blocks and b in ctx_blocks}
            rbf = {b: self._rbf(b, self.params_, Zc, self.Z_train_, cached, train=train)[0]
                   for b in self.rbf_blocks if b in ctx_blocks}

            const = torch.zeros(self.n_train_, dtype=self.dtype)
            mol_weight = torch.zeros(self.n_train_, dtype=self.dtype)
            # A side naming one molecular block is a different term from the "mol" side, which
            # is the SUM of the molecular RBFs; it needs its own weight vector.
            block_weight = {b: torch.zeros(self.n_train_, dtype=self.dtype)
                            for b in MOL_KERNEL_BLOCKS}
            prior = torch.zeros((), dtype=self.dtype)

            for b in self.rbf_blocks:
                scale = torch.exp(self.params_[f"sc:rbf:{b}"])
                prior = prior + scale                      # RBF(x, x) == 1 for every block
                if b in ctx_blocks:
                    const = const + scale * rbf[b]
            for b in self.linear_blocks:
                scale = torch.exp(self.params_[f"sc:linear:{b}"])
                const = const + scale * (Zc[b] @ self.Z_train_[b].T)[0]
                prior = prior + scale * (Zc[b] * Zc[b]).sum()
            for inter in self.interactions:
                left, right = self._sides(inter)
                scale = torch.exp(self.params_[f"sc:prod:{inter}"])
                # Each RBF is 1 on the diagonal, so a side's self-covariance is 1 -- except
                # the "mol" side, which is the SUM of the two molecular RBFs and so is 2.
                prior = prior + scale * self._side_diag(left) * self._side_diag(right)
                mol_side, ctx_side = ((left, right) if self._is_mol_side(left)
                                      else (right, left))
                if not self._is_mol_side(mol_side):
                    const = const + scale * rbf[left] * rbf[right]
                elif mol_side == "mol":
                    mol_weight = mol_weight + scale * rbf[ctx_side]
                else:
                    block_weight[mol_side] = block_weight[mol_side] + scale * rbf[ctx_side]

            # Which molecular blocks carry a nonzero weight is a property of the fold, not of
            # the batch: decide it here, once, instead of testing it on every prediction.
            active = [b for b in MOL_KERNEL_BLOCKS if bool(torch.any(block_weight[b] != 0))]
            return {"const": const, "mol_weight": mol_weight, "block_weight": block_weight,
                    "prior_var": prior, "active_blocks": active}

    def _is_mol_side(self, side: str) -> bool:
        """Whether one side of an interaction is the molecular kernel."""
        return side == "mol" or side in MOL_KERNEL_BLOCKS

    @staticmethod
    def _side_diag(side: str) -> float:
        """Self-covariance of one side of a product term."""
        return float(len(MOL_KERNEL_BLOCKS)) if side == "mol" else 1.0

    def predict_in_context(self, mol_blocks: Dict[str, np.ndarray], fold: dict,
                           return_std: bool = False):
        """Predict for molecules in the context a :meth:`fold_context` call prepared.

        Args:
            mol_blocks: The processed molecular blocks of the batch.
            fold: The dict returned by :meth:`fold_context`.
            return_std: Also return the posterior standard deviation.

        Returns:
            ``mean`` or ``(mean, std)``, each of shape ``(n_molecules,)``.
        """
        with torch.no_grad():
            train = self._train_side()
            Zm = {b: torch.as_tensor(np.asarray(mol_blocks[b]), dtype=self.dtype)
                  for b in MOL_KERNEL_BLOCKS}
            cached = {b: self._sq_dists(Zm[b], self.Z_train_[b], b_sq=train["sq"][b])
                      for b in MOL_KERNEL_BLOCKS if b not in self.ard_blocks}
            rbf = {b: self._rbf(b, self.params_, Zm, self.Z_train_, cached, train=train)
                   for b in MOL_KERNEL_BLOCKS}
            scaled = None
            for b in MOL_KERNEL_BLOCKS:
                term = torch.exp(self.params_[f"sc:rbf:{b}"]) * rbf[b]
                scaled = term if scaled is None else scaled + term
            mol = sum(rbf[b] for b in MOL_KERNEL_BLOCKS)

            cross = scaled + mol * fold["mol_weight"][None, :] + fold["const"][None, :]
            active = fold.get("active_blocks")
            if active is None:                  # a fold built before the weights were resolved once
                active = [b for b, w in fold.get("block_weight", {}).items()
                          if bool(torch.any(w != 0))]
            for block in active:
                cross = cross + rbf[block] * fold["block_weight"][block][None, :]
            mean = (self.params_["mean"] + (cross @ self.alpha_).squeeze(1)).numpy()
            if not return_std:
                return mean
            v = torch.linalg.solve_triangular(self.chol_, cross.T.to(self.chol_.dtype), upper=False)
            var = fold["prior_var"] - (v * v).sum(0)
            return mean, torch.sqrt(var.clamp_min(0.0)).numpy()

    def kernel_report(self, max_rows: int = 400) -> dict:
        """Fitted lengthscales, the share of prior variance each term carries, and the noise.

        Args:
            max_rows: Rows used to estimate the per-term shares.

        Returns:
            Dict with ``lengthscale`` (per RBF block), ``weight`` (per term, summing to 1)
            and ``noise``.
        """
        with torch.no_grad():
            n = min(self.n_train_, max_rows)
            Z = {b: a[:n] for b, a in self.Z_train_.items()}
            cached = self._cache_distances(Z, Z, same=True)
            rbf = {b: self._rbf(b, self.params_, Z, Z, cached, same=True)
                   for b in self.rbf_blocks}
            shares = {}
            for b in self.rbf_blocks:
                shares[f"rbf:{b}"] = float((torch.exp(self.params_[f"sc:rbf:{b}"])
                                            * rbf[b]).diagonal().mean())
            for b in self.linear_blocks:
                shares[f"linear:{b}"] = float((torch.exp(self.params_[f"sc:linear:{b}"])
                                               * (Z[b] * Z[b]).sum(1)).mean())
            for inter in self.interactions:
                left, right = self._sides(inter)
                prod = (self._side_kernel(left, rbf) * self._side_kernel(right, rbf)).diagonal()
                shares[f"prod:{inter}"] = float(
                    (torch.exp(self.params_[f"sc:prod:{inter}"]) * prod).mean())
            total = sum(shares.values()) or 1.0
            lengthscale = {}
            for b in self.rbf_blocks:
                ls = self._lengthscale(b, self.params_).numpy()
                lengthscale[b] = ls if b in self.ard_blocks else ls[0]
            return {"lengthscale": lengthscale,
                    "weight": {k: v / total for k, v in shares.items()},
                    "noise": self.noise_}
