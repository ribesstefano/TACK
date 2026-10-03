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


class AdditiveProductGP:
    """Exact GP over processed blocks, with cached distances and exact posterior variance.

    Args:
        dims: Mapping of block name to width (as produced by ``BlockPreprocessor.dims_``).
        interactions: Product kernel terms, each ``"<a>*<b>"`` where a side is either a block
            name or ``"mol"`` (the sum of the molecular RBFs).
        ard_blocks: Blocks given one lengthscale per column.
        rbf_blocks: Blocks given an RBF kernel (default: every block except ``linear_blocks``).
        linear_blocks: Blocks given a linear kernel.
        jitter: Starting diagonal jitter for the Cholesky factorisation.
        dtype: Torch dtype for the computation.
    """

    def __init__(self, dims: Dict[str, int], *,
                 interactions: Sequence[str] = DEFAULT_INTERACTIONS,
                 ard_blocks: Sequence[str] = ("descriptors",),
                 rbf_blocks: Optional[Sequence[str]] = None,
                 linear_blocks: Sequence[str] = ("assay_time",),
                 jitter: float = 1e-6,
                 dtype: Union[str, torch.dtype] = torch.float64):
        self.dims = dict(dims)
        self.linear_blocks = [b for b in linear_blocks if b in self.dims]
        self.rbf_blocks = list(rbf_blocks) if rbf_blocks is not None else [
            b for b in self.dims if b not in self.linear_blocks]
        self.ard_blocks = [b for b in ard_blocks if b in self.rbf_blocks]
        self.interactions = [t for t in interactions if self._term_available(t)]
        self.jitter = jitter
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.terms = ([f"rbf:{b}" for b in self.rbf_blocks]
                      + [f"linear:{b}" for b in self.linear_blocks]
                      + [f"prod:{t}" for t in self.interactions])

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
        params["noise"] = torch.tensor(np.log(0.5), dtype=self.dtype, requires_grad=True)
        params["mean"] = torch.tensor(0.0, dtype=self.dtype, requires_grad=True)
        return params

    # ---------------------------------------------------------------- kernel pieces

    def _to_tensor(self, Z: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        return {b: torch.as_tensor(np.asarray(Z[b]), dtype=self.dtype) for b in self.dims}

    @staticmethod
    def _sq_dists(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Pairwise squared euclidean distances, clamped at zero for numerical safety."""
        d2 = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * (A @ B.T)
        return d2.clamp_min(0.0)

    def _cache_distances(self, Za: Dict[str, torch.Tensor],
                         Zb: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Squared distances of the isotropic RBF blocks, computed once per (Za, Zb) pair.

        ARD blocks are absent on purpose: their distances depend on the lengthscales and are
        recomputed per step in :meth:`_rbf`.
        """
        return {b: self._sq_dists(Za[b], Zb[b])
                for b in self.rbf_blocks if b not in self.ard_blocks}

    def _rbf(self, block: str, params: Dict[str, torch.Tensor],
             Za: Dict[str, torch.Tensor], Zb: Dict[str, torch.Tensor],
             cached: Dict[str, torch.Tensor]) -> torch.Tensor:
        """RBF kernel of one block, from the cached distances or recomputed for ARD."""
        ls = torch.exp(params[f"ls:{block}"])
        if block in self.ard_blocks:
            d2 = self._sq_dists(Za[block] / ls, Zb[block] / ls)
            return torch.exp(-0.5 * d2)
        return torch.exp(-0.5 * cached[block] / (ls[0] ** 2))

    def _assemble(self, params: Dict[str, torch.Tensor],
                  Za: Dict[str, torch.Tensor], Zb: Dict[str, torch.Tensor],
                  cached: Dict[str, torch.Tensor]) -> torch.Tensor:
        """The full covariance between two sets of rows (noise-free)."""
        rbf = {b: self._rbf(b, params, Za, Zb, cached) for b in self.rbf_blocks}
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

    def _cholesky(self, K: torch.Tensor) -> torch.Tensor:
        """Cholesky factor with escalating jitter; raises if the kernel stays singular."""
        eye = torch.eye(K.shape[0], dtype=self.dtype)
        jitter = self.jitter
        while jitter <= 1e-2:
            try:
                return torch.linalg.cholesky(K + eye * jitter)
            except Exception:
                jitter *= 10
        raise RuntimeError("GP kernel is not positive definite even with 1e-2 jitter")

    # ---------------------------------------------------------------- fitting

    def _neg_log_mll(self, params, Z, y, cached) -> torch.Tensor:
        """Negative exact log marginal likelihood."""
        n = len(y)
        K = self._assemble(params, Z, Z, cached) + torch.eye(n, dtype=self.dtype) * torch.exp(
            params["noise"])
        L = torch.linalg.cholesky(K + torch.eye(n, dtype=self.dtype) * self.jitter)
        resid = (y - params["mean"]).unsqueeze(1)
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
            rng = np.random.default_rng(seed)
            n = len(y)
            sub = (np.arange(n) if n <= max_hyper_points
                   else np.sort(rng.choice(n, max_hyper_points, replace=False)))
            Z_sub = self._to_tensor({b: np.asarray(Z[b])[sub] for b in self.dims})
            y_sub = torch.as_tensor(y[sub], dtype=self.dtype)
            cached = self._cache_distances(Z_sub, Z_sub)
            best = (np.inf, None)
            for restart in range(n_restarts):
                torch.manual_seed(seed * 1000 + restart)
                params = self._init_params(np.random.default_rng(seed * 1000 + restart))
                opt = torch.optim.Adam(params.values(), lr=lr)
                for _ in range(n_iter):
                    opt.zero_grad()
                    loss = self._neg_log_mll(params, Z_sub, y_sub, cached)
                    if not torch.isfinite(loss):
                        break
                    loss.backward()
                    opt.step()
                with torch.no_grad():
                    final = float(self._neg_log_mll(params, Z_sub, y_sub, cached))
                if np.isfinite(final) and final < best[0]:
                    best = (final, {k: v.detach().clone() for k, v in params.items()})
            if best[1] is None:
                raise RuntimeError("GP marginal-likelihood optimisation diverged in every restart")
            state = {"params": best[1], "neg_mll": best[0], "n_hyper": int(len(y_sub))}
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
        self.params_ = {k: v.detach().clone() for k, v in state["params"].items()}
        self.Z_train_ = self._to_tensor(Z)
        self.y_train_ = torch.as_tensor(y, dtype=self.dtype)
        self.n_train_ = len(y)
        self.n_hyper_ = int(state.get("n_hyper", len(y)))
        self.noise_ = float(torch.exp(self.params_["noise"]))
        with torch.no_grad():
            cached = self._cache_distances(self.Z_train_, self.Z_train_)
            K = self._assemble(self.params_, self.Z_train_, self.Z_train_, cached)
            K = K + torch.eye(self.n_train_, dtype=self.dtype) * self.noise_
            self.chol_ = self._cholesky(K)
            resid = (self.y_train_ - self.params_["mean"]).unsqueeze(1)
            self.alpha_ = torch.cholesky_solve(resid, self.chol_)

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
            return self._assemble(self.params_, ta, tb, self._cache_distances(ta, tb)).numpy()

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
            cross = self._assemble(self.params_, Zt, self.Z_train_,
                                   self._cache_distances(Zt, self.Z_train_))
            mean = (self.params_["mean"] + (cross @ self.alpha_).squeeze(1)).numpy()
            if not return_std:
                return mean
            v = torch.linalg.solve_triangular(self.chol_, cross.T, upper=False)
            var = self._diag(self.params_, Zt) - (v * v).sum(0)
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
            cached = self._cache_distances(Z, Z)
            rbf = {b: self._rbf(b, self.params_, Z, Z, cached) for b in self.rbf_blocks}
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
                ls = torch.exp(self.params_[f"ls:{b}"]).numpy()
                lengthscale[b] = ls if b in self.ard_blocks else ls[0]
            return {"lengthscale": lengthscale,
                    "weight": {k: v / total for k, v in shares.items()},
                    "noise": self.noise_}
