"""A generic ensemble of fusion members with a cached biological context.

Every member consumes the *same* design matrix, which is what makes the class generic: a
member may be a :class:`~tackai.fusion.models.GPInteraction`, a
:class:`~tackai.fusion.models.XGBoostFusion`, or a mixture of both, and the ensemble never
needs to know which. It also makes the context cache possible: encode one experimental
context once, transform it once per member, and then every new batch of molecules only pays
for its own featurisation.

The API mirrors :class:`tackai.ensemble_predictor.EnsemblePredictor`: build with
:meth:`FusionEnsemble.from_pretrained`, score with :meth:`FusionEnsemble.predict`.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from tackai.fusion.blocks import BLOCK_ORDER
from tackai.fusion.context import CONTEXT_BLOCKS, ContextEncoder
from tackai.fusion.data import TASK_LABELS, TASK_SUPPORT, TASK_TYPES, FusionData
from tackai.fusion.features import MolFeaturizer

Z95 = 1.959963984540054          # two-sided 95% normal quantile


@dataclass
class FusionPrediction:
    """Ensemble prediction for a batch of molecules in one context.

    Attributes:
        mean: Weighted ensemble prediction, in the task's own units (a probability for the
            binary task). NaN where the SMILES could not be parsed.
        std: Total uncertainty by the law of total variance — the weighted mean of the
            members' predictive variances plus the weighted spread of their means.
        member_predictions: Per-member prediction.
        member_std: Per-member predictive standard deviation (zeros for a member without one).
        weights: Member weights used for the average.
        task: Task name.
        label_name: Human-readable label of the predicted quantity.
        ok: False where the input SMILES could not be parsed.
        ci_lower_95 / ci_upper_95: ``mean +/- 1.96 * std``, clipped to the task's own support
            where it has one (Dmax is a fraction, activity a probability), so an interval
            never reports an impossible value. This is an interval on the latent mean: it
            covers the ensemble's disagreement and each member's posterior variance, not the
            assay's own measurement noise, which the GP fits separately and excludes.
    """

    mean: np.ndarray
    std: np.ndarray
    member_predictions: Dict[str, np.ndarray]
    member_std: Dict[str, np.ndarray]
    weights: Dict[str, float]
    task: str
    label_name: str
    ok: np.ndarray
    ci_lower_95: np.ndarray = field(default=None)
    ci_upper_95: np.ndarray = field(default=None)
    smiles: Optional[List[str]] = None

    def __post_init__(self):
        support = TASK_SUPPORT.get(self.task)
        if self.ci_lower_95 is None:
            self.ci_lower_95 = self.mean - Z95 * self.std
            if support is not None:
                self.ci_lower_95 = np.clip(self.ci_lower_95, support[0], support[1])
        if self.ci_upper_95 is None:
            self.ci_upper_95 = self.mean + Z95 * self.std
            if support is not None:
                self.ci_upper_95 = np.clip(self.ci_upper_95, support[0], support[1])

    def to_frame(self) -> pd.DataFrame:
        """One row per molecule, with the prediction, its uncertainty and the interval."""
        frame = pd.DataFrame({"prediction": self.mean, "std": self.std,
                              "ci_lower_95": self.ci_lower_95, "ci_upper_95": self.ci_upper_95,
                              "ok": self.ok})
        if self.smiles is not None:
            frame.insert(0, "smiles", self.smiles)
        return frame

    def summary(self) -> str:
        """One-line description of the batch."""
        valid = np.isfinite(self.mean)
        if not valid.any():
            return f"{self.label_name}: no valid molecule in {len(self.mean)} input(s)"
        return (f"{self.label_name}: {np.mean(self.mean[valid]):.3f} mean over {valid.sum()} "
                f"molecule(s), typical uncertainty {np.mean(self.std[valid]):.3f} "
                f"({len(self.weights)} members)")


@dataclass
class FusionContext:
    """A biological context encoded once and pre-transformed for every member.

    Attributes:
        values: The encoded context columns, shape ``(1, n_context_columns)``.
        per_member: For each member, its preprocessor's output for the context blocks.
        source: The record the context was built from.
        folds: For each member that supports it, the context-only kernel terms collapsed to
            two vectors, so a batch pays only for the molecular kernels.
    """

    values: np.ndarray
    per_member: List[Dict[str, np.ndarray]]
    source: dict
    folds: List[Optional[dict]] = field(default_factory=list)


class FusionEnsemble:
    """Weighted ensemble of fusion members sharing one design matrix and one context.

    Args:
        members: Fitted estimators, all consuming the same block layout.
        data: The :class:`FusionData` whose encoder and layout the members were fitted with.
        task: Task name (``"dmax"``, ``"pdc50"`` or ``"activity"``).
        weights: Member weights (default: equal).
    """

    def __init__(self, members: Sequence, data: FusionData, task: str,
                 weights: Optional[Union[Sequence[float], Dict[str, float]]] = None):
        self.members = list(members)
        self.data = data
        self.task = task
        self.names = [f"member_{i:02d}" for i in range(len(self.members))]
        self.set_weights(weights)

    # ---------------------------------------------------------------- construction

    @classmethod
    def fit(cls, factory: Union[Callable, Sequence[Callable]], data: FusionData, task: str, *,
            splits=None, n_members: Optional[int] = 5, n_repeats: int = 5, n_folds: int = 5,
            n_jobs: int = 1, verbose: bool = False) -> "FusionEnsemble":
        """Fit one member per cross-validation fold.

        Five members are the default because the speed study found five about as accurate as
        twenty-five at a quarter of the cost.

        Args:
            factory: Callable returning a fresh estimator, or one callable per member. Each is
                called with ``task_type`` and ``random_state``.
            data: Training data.
            task: Task name.
            splits: Pre-computed ``splits[repeat][fold]``; defaults to the task's own splits.
            n_members: Number of folds to fit (ignored when ``factory`` is a list).
            n_repeats: Repeats to draw folds from.
            n_folds: Folds per repeat.
            n_jobs: Parallel worker processes for the fits.
            verbose: Print progress per member.

        Returns:
            A fitted :class:`FusionEnsemble`.
        """
        if task not in TASK_TYPES:
            raise ValueError(f"task must be one of {sorted(TASK_TYPES)}, got {task!r}")
        _, X, y, groups = data.task_rows(task)
        splits = splits if splits is not None else data.splits(task, n_repeats, n_folds)
        folds = [fold for repeat in splits for fold in repeat]

        factories = list(factory) if isinstance(factory, (list, tuple)) else None
        count = len(factories) if factories is not None else min(n_members, len(folds))
        if count > len(folds):
            raise ValueError(f"asked for {count} members but only {len(folds)} folds exist")
        make = (lambda k: factories[k]) if factories is not None else (lambda k: factory)

        def fit_one(k):
            train, _ = folds[k]
            est = make(k)(task_type=TASK_TYPES[task], random_state=k)
            if verbose:
                print(f"fitting member {k} on {len(train)} rows")
            return est.fit(X[train], y[train], groups[train])

        if n_jobs and n_jobs > 1:
            from joblib import Parallel, delayed
            members = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(fit_one)(k) for k in range(count))
        else:
            members = [fit_one(k) for k in range(count)]
        return cls(members, data, task)

    @classmethod
    def from_pretrained(cls, model_id: Union[str, Path], *, revision: Optional[str] = None,
                        token: Optional[str] = None,
                        data: Optional[FusionData] = None) -> "FusionEnsemble":
        """Load a saved ensemble from a local directory or a Hugging Face Hub repo.

        Args:
            model_id: Directory written by :meth:`save`, or a Hub repo id.
            revision: Hub revision, for a repo id.
            token: Hub token, for a private repo.
            data: Reuse this :class:`FusionData` instead of rebuilding an encoder-only one
                from the manifest (inference needs only the encoder and the layout).

        Returns:
            The loaded ensemble.
        """
        import joblib

        path = Path(model_id)
        if not path.exists():
            from huggingface_hub import snapshot_download
            path = Path(snapshot_download(repo_id=str(model_id), revision=revision, token=token))
        manifest = json.loads((path / "manifest.json").read_text())
        members = [joblib.load(path / name) for name in manifest["member_files"]]
        if data is None:
            encoder = ContextEncoder(protein_space=manifest.get("protein_space", "per_block"))
            featurizer = MolFeaturizer(radius=manifest["fingerprint"][0],
                                       fp_size=manifest["fingerprint"][1])
            data = FusionData(encoder=encoder, featurizer=featurizer)
        cls._check_layout(manifest, data, members)
        return cls(members, data, manifest["task"], weights=manifest.get("weights"))

    @staticmethod
    def _check_layout(manifest: dict, data: FusionData, members: Sequence) -> None:
        """Refuse to score with members whose block layout no longer matches the encoder.

        Re-fitting a context PCA changes a cached table's width. The members still slice the
        columns they were fitted with, so every context block after the changed one lands at
        the wrong offset and the predictions are quietly wrong rather than absent.

        Args:
            manifest: The saved manifest.
            data: The encoder the predictions would be made with.
            members: The loaded members.

        Raises:
            ValueError: If any block's width differs between the manifest, the encoder and
                the members.
        """
        expected = manifest.get("block_dims")
        if not expected:
            return
        for name, actual in (("the embedding cache", data.dims),
                             ("the fitted members", getattr(members[0], "pre_", None)
                              and members[0].pre_.dims_)):
            if not actual:
                continue
            bad = {b: (expected[b], actual[b]) for b in expected
                   if b in actual and expected[b] != actual[b]}
            if bad:
                detail = ", ".join(f"{b}: trained with {e}, {name} has {a}"
                                   for b, (e, a) in sorted(bad.items()))
                raise ValueError(
                    f"block layout mismatch between this ensemble and {name} ({detail}). The "
                    "cached embedding tables have changed since the members were fitted; "
                    "refit the ensemble or restore the tables it was trained on.")

    def save(self, path: Union[str, Path]) -> Path:
        """Write every member plus a manifest describing the layout it expects.

        Args:
            path: Target directory (created if needed).

        Returns:
            The directory written to.
        """
        import joblib
        import sklearn
        import torch
        import xgboost

        import tackai
        from tackai.fusion.features import FP_RADIUS, FP_SIZE

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        files = []
        for name, member in zip(self.names, self.members):
            filename = f"{name}.joblib"
            joblib.dump(member, path / filename)
            files.append(filename)
        (path / "manifest.json").write_text(json.dumps({
            "task": self.task,
            "n_members": len(self.members),
            "member_files": files,
            "member_types": [type(m).__name__ for m in self.members],
            "weights": self.weights,
            "block_dims": self.data.dims,
            "block_order": BLOCK_ORDER,
            "protein_space": self.data.encoder.protein_space,
            "fingerprint": [self.data.featurizer.radius, self.data.featurizer.fp_size],
            "versions": {"tackai": getattr(tackai, "__version__", "unknown"),
                         "numpy": np.__version__, "torch": torch.__version__,
                         "xgboost": xgboost.__version__, "sklearn": sklearn.__version__},
        }, indent=1))
        return path

    def set_weights(self, weights: Optional[Union[Sequence[float], Dict[str, float]]]) -> None:
        """Set the member weights, normalised to sum to one (default: equal).

        Args:
            weights: Sequence aligned with the members, or a mapping of member name to weight.
        """
        if weights is None:
            values = np.ones(len(self.members))
        elif isinstance(weights, dict):
            unknown = sorted(set(weights) - set(self.names))
            if unknown:
                raise ValueError(f"unknown member name(s) in weights: {unknown}; "
                                 f"this ensemble has {self.names}")
            missing = sorted(set(self.names) - set(weights))
            if missing:
                raise ValueError(f"no weight given for member(s): {missing}")
            values = np.array([float(weights[name]) for name in self.names])
        else:
            values = np.asarray(list(weights), dtype=float)
            if len(values) != len(self.members):
                raise ValueError(f"expected {len(self.members)} weights, got {len(values)}")
        total = values.sum()
        if total <= 0:
            raise ValueError("member weights must sum to a positive number")
        self.weights = {name: float(w) for name, w in zip(self.names, values / total)}

    @property
    def available_tasks(self) -> List[str]:
        """The tasks this ensemble can predict (one, the task it was fitted on)."""
        return [self.task]

    # ---------------------------------------------------------------- context cache

    def transform_context(self, record: dict) -> FusionContext:
        """Encode one experimental context and pre-transform it for every member.

        Scoring many molecules against one context is the common case in a screening or
        reinforcement-learning loop; this does the context's share of the work once.

        Args:
            record: Mapping with ``poi_seq``, ``e3_seq``, ``cell_id``, ``assay`` and
                optionally ``assay_time``.

        Returns:
            A :class:`FusionContext` to pass to :meth:`predict`.

        Raises:
            KeyError: If a POI or E3 sequence is not in the cached embedding tables.
        """
        values = self.data.encode_context(record)
        row = np.zeros((1, self.data.n_columns), dtype=np.float64)
        row[:, self.data.context_columns] = values
        context_blocks = [b for b in BLOCK_ORDER if b in CONTEXT_BLOCKS]
        per_member = [m.pre_.transform_blocks(row, only=context_blocks) for m in self.members]
        folds = [self._fold_member(member, blocks)
                 for member, blocks in zip(self.members, per_member)]
        return FusionContext(values=values, per_member=per_member, source=dict(record),
                             folds=folds)

    @staticmethod
    def _fold_member(member, context_blocks: Dict[str, np.ndarray]) -> Optional[dict]:
        """Collapse a member's context-only kernel terms, when its model can do that.

        Six of the GP's ten kernel terms depend on the context alone and are the same for
        every molecule in a batch; folding them is what makes the context cache pay for
        itself. A member whose model has no such shortcut simply gets None.
        """
        model = getattr(member, "model_", None)
        if not hasattr(model, "fold_context"):
            return None
        try:
            return model.fold_context(context_blocks)
        except ValueError:
            return None        # an interaction this context cannot fold; use the ordinary path

    # ---------------------------------------------------------------- prediction

    def predict(self, samples, context: Optional[FusionContext] = None,
                return_individual: bool = True, return_std: bool = True) -> FusionPrediction:
        """Score molecules, either against a cached context or from full records.

        Args:
            samples: SMILES string(s) when ``context`` is given; otherwise dicts (or a
                DataFrame) carrying both the molecule and its context.
            context: A :class:`FusionContext` from :meth:`transform_context`. The fast path.
            return_individual: Keep the per-member predictions in the result.
            return_std: Ask every member that can for a predictive standard deviation.

        Returns:
            A :class:`FusionPrediction`. Rows whose SMILES RDKit cannot parse come back as
            NaN with ``ok=False`` rather than aborting the batch.
        """
        if context is not None:
            smiles = [samples] if isinstance(samples, str) else list(samples)
            ok, scores = self._predict_with_context(smiles, context, return_std)
        else:
            records = samples.to_dict("records") if isinstance(samples, pd.DataFrame) else list(
                [samples] if isinstance(samples, dict) else samples)
            smiles = [r.get("smiles") for r in records]
            ok, scores = self._predict_from_records(records, return_std)
        return self._aggregate(scores, ok, smiles, return_individual)

    def predict_matrix(self, X, return_individual: bool = True,
                       return_std: bool = True) -> FusionPrediction:
        """Score rows that are already encoded as a design matrix.

        This is the path an evaluation takes: a held-out fold of a :class:`FusionData` is
        already featurised, and re-encoding it from records would only repeat the work.

        Args:
            X: Design matrix with this ensemble's block layout.
            return_individual: Keep the per-member predictions in the result.
            return_std: Ask every member that can for a predictive standard deviation.

        Returns:
            A :class:`FusionPrediction` whose ``ok`` is all True (an encoded row is valid by
            construction).
        """
        X = np.asarray(X)
        n = len(X)
        ok = np.ones(n, dtype=bool)
        per_member = []
        for member in self.members:
            if n == 0:
                per_member.append((np.empty(0), np.empty(0)))
                continue
            Z = member.pre_.transform(X)
            per_member.append(self._member_scores(member, Z, return_std))
        return self._aggregate(per_member, ok, [None] * n, return_individual)

    def _predict_with_context(self, smiles: List[str], context: FusionContext,
                              return_std: bool):
        """Fast path: featurise the molecules, reuse each member's transformed context."""
        n = len(smiles)
        ok = np.zeros(n, dtype=bool)
        per_member = []
        if n == 0:
            return ok, [(np.empty(0), np.empty(0)) for _ in self.members]

        fp, desc, ok = self.data.featurizer.featurize(smiles)
        valid = np.flatnonzero(ok)
        mol_row = np.zeros((n, self.data.n_columns), dtype=np.float64)
        mol_row[:, self.data.index["fingerprint"]] = fp
        mol_row[:, self.data.index["descriptors"]] = desc

        folds = context.folds or [None] * len(self.members)
        for member, ctx_blocks, fold in zip(self.members, context.per_member, folds):
            mean = np.full(n, np.nan)
            std = np.zeros(n)
            if len(valid):
                Z = dict(member.pre_.transform_blocks(mol_row[valid], only=["fingerprint",
                                                                            "descriptors"]))
                if fold is not None:
                    mean[valid], std[valid] = self._folded_scores(member, Z, fold, return_std)
                else:
                    for block, row in ctx_blocks.items():
                        Z[block] = np.repeat(row, len(valid), axis=0)
                    mean[valid], std[valid] = self._member_scores(member, Z, return_std)
            per_member.append((mean, std))
        return ok, per_member

    def _folded_scores(self, member, mol_blocks: Dict[str, np.ndarray], fold: dict,
                       return_std: bool):
        """A member's prediction through its folded context, in the reported units."""
        wants_std = return_std and member.supports_std
        result = member.model_.predict_in_context(mol_blocks, fold, return_std=wants_std)
        score, std = result if wants_std else (result, None)
        if member.task_type != "binary":
            scaled = score * member.y_std_ + member.y_mean_
            return scaled, (std * member.y_std_ if std is not None else np.zeros(len(score)))
        if member.native_binary:
            return np.clip(score, 0.0, 1.0), (std if std is not None else np.zeros(len(score)))
        probability = member._platt(score)
        if std is None:
            return probability, np.zeros(len(score))
        high, low = member._platt(score + std), member._platt(score - std)
        return probability, np.abs(high - low) / 2.0

    def _predict_from_records(self, records: List[dict], return_std: bool):
        """Ordinary path: encode each record in full, then score it with every member."""
        n = len(records)
        if n == 0:
            return np.zeros(0, dtype=bool), [(np.empty(0), np.empty(0)) for _ in self.members]

        X = self.data.encode(records)
        _, _, ok = self.data.featurizer.featurize([r.get("smiles") for r in records])
        valid = np.flatnonzero(ok)
        per_member = []
        for member in self.members:
            mean = np.full(n, np.nan)
            std = np.zeros(n)
            if len(valid):
                Z = member.pre_.transform(X[valid])
                mean[valid], std[valid] = self._member_scores(member, Z, return_std)
            per_member.append((mean, std))
        return ok, per_member

    def _member_scores(self, member, Z: Dict[str, np.ndarray], return_std: bool):
        """One member's prediction on processed blocks, in the reported units."""
        if return_std and member.supports_std:
            score, std = member._predict_model(member.model_, Z, return_std=True)
            if member.task_type != "binary":
                return score * member.y_std_ + member.y_mean_, std * member.y_std_
            if member.native_binary:
                return np.clip(score, 0.0, 1.0), std
            high, low = member._platt(score + std), member._platt(score - std)
            return member._platt(score), np.abs(high - low) / 2.0
        score = member._predict_model(member.model_, Z)
        return member._to_original_units(score), np.zeros(len(score))

    def _aggregate(self, per_member, ok, smiles, return_individual: bool) -> FusionPrediction:
        """Weighted mean, and the law of total variance over the members."""
        weights = np.array([self.weights[name] for name in self.names])[:, None]
        means = np.array([m for m, _ in per_member], dtype=float)
        stds = np.array([s for _, s in per_member], dtype=float)
        if means.size == 0:
            mean = np.empty(0)
            std = np.empty(0)
        else:
            mean = (weights * means).sum(axis=0)
            within = (weights * stds ** 2).sum(axis=0)
            between = (weights * (means - mean) ** 2).sum(axis=0)
            std = np.sqrt(within + between)
        member_predictions = ({name: means[i] for i, name in enumerate(self.names)}
                              if return_individual and means.size else {})
        member_std = ({name: stds[i] for i, name in enumerate(self.names)}
                      if return_individual and stds.size else {})
        return FusionPrediction(mean=mean, std=std, member_predictions=member_predictions,
                                member_std=member_std, weights=dict(self.weights),
                                task=self.task, label_name=TASK_LABELS[self.task],
                                ok=np.asarray(ok, dtype=bool), smiles=list(smiles))
