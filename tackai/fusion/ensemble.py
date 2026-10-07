"""A generic ensemble of fusion members with a cached biological context.

Every member consumes the *same* design matrix, which is what makes the class generic: a
member may be a :class:`~tackai.fusion.gp.GPInteraction` or any other estimator sharing its
``fit``/``predict``/``report`` contract, and the ensemble never needs to know which. It also
makes the context cache possible: encode one experimental context once, fold it into every
member that can, and then every new batch of molecules only pays for its own featurisation.

The API mirrors :class:`tackai.ensemble_predictor.EnsemblePredictor`: build with
:meth:`FusionEnsemble.from_pretrained`, score with :meth:`FusionEnsemble.predict`.
"""
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

from tackai.fusion.context import ContextEncoder
from tackai.fusion.data import BLOCK_ORDER, TASK_LABELS, TASK_SUPPORT, TASK_TYPES, FusionData
from tackai.fusion.mol_encoder import DESCRIPTOR_NAMES, MolEncoder
from tackai.fusion.gp import GPInteraction
from tackai.fusion.training import check_labels

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
    """A biological context encoded once and folded into every member that can.

    Attributes:
        values: The encoded context columns, shape ``(1, n_context_columns)``.
        blocks: The same context as ``{block: (1, dim) array}``.
        source: The record the context was built from.
        folds: For each member that supports it, the context-only kernel terms collapsed to
            two vectors, so a batch pays only for the molecular kernels.
    """

    values: np.ndarray
    blocks: Dict[str, np.ndarray]
    source: dict
    folds: List[Optional[dict]] = field(default_factory=list)


class FusionEnsemble:
    """Weighted ensemble of fusion members sharing one design matrix and one context.

    Args:
        members: Fitted estimators, all consuming the design matrix of ``data``.
        data: The :class:`FusionData` whose encoder and layout the members were fitted with.
        task: Task name (``"dmax"``, ``"pdc50"`` or ``"activity"``).
        weights: Member weights (default: equal).

    Raises:
        ValueError: If a member that carries a block layout (a GP) was fitted with one other
            than ``data.blocks_indexes``.
    """

    def __init__(self, members: Sequence, data: FusionData, task: str,
                 weights: Optional[Union[Sequence[float], Dict[str, float]]] = None):
        if task not in TASK_TYPES:
            raise ValueError(f"task must be one of {sorted(TASK_TYPES)}, got {task!r}")
        self.members = list(members)
        self.data = data
        self.task = task
        self.names = [f"member_{i:02d}" for i in range(len(self.members))]
        self._check_members()
        self.set_weights(weights)

    def _check_members(self) -> None:
        """Refuse a member whose block layout differs from the data's.

        The ensemble hands every member the one design matrix of ``data`` and, on the cached
        context path, raw blocks cut by ``data``'s layout. A member fitted with another layout
        would read the wrong columns without any error.
        """
        expected = self.data.blocks_indexes
        for name, member in zip(self.names, self.members):
            layout = getattr(member, "blocks", None)
            if layout is None:
                continue
            if list(layout) != list(expected) or any(
                    not np.array_equal(layout[b], expected[b]) for b in expected):
                raise ValueError(
                    f"{name} was fitted with a different block layout than the data this "
                    "ensemble is built on; pass blocks=data.blocks_indexes when fitting")

    # ---------------------------------------------------------------- construction

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
            featurizer = MolEncoder(radius=manifest["fingerprint"][0],
                                       fp_size=manifest["fingerprint"][1],
                                    descriptors=manifest.get("descriptors", DESCRIPTOR_NAMES))
            data = FusionData(encoder=encoder, featurizer=featurizer)
        cls._check_layout(manifest, data)
        return cls(members, data, manifest["task"], weights=manifest.get("weights"))

    @staticmethod
    def _check_layout(manifest: dict, data: FusionData) -> None:
        """Refuse to score with members whose block layout no longer matches the encoder.

        Re-fitting a context PCA changes a cached table's width. The members still slice the
        columns they were fitted with, so every context block after the changed one lands at
        the wrong offset and the predictions are quietly wrong rather than absent.

        Args:
            manifest: The saved manifest.
            data: The encoder the predictions would be made with.

        Raises:
            ValueError: If any block's width differs between the manifest and the encoder.
        """
        expected = manifest.get("block_dims")
        if not expected:
            return
        bad = {b: (expected[b], data.dims[b]) for b in expected
               if b in data.dims and expected[b] != data.dims[b]}
        if bad:
            detail = ", ".join(f"{b}: trained with {e}, the embedding cache has {a}"
                               for b, (e, a) in sorted(bad.items()))
            raise ValueError(
                f"block layout mismatch between this ensemble and the embedding cache "
                f"({detail}). The cached embedding tables have changed since the members were "
                "fitted; refit the ensemble or restore the tables it was trained on.")

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
        from tackai.fusion.mol_encoder import FP_RADIUS, FP_SIZE

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
            "descriptors": self.data.featurizer.descriptors,
            "versions": {"tackai": getattr(tackai, "__version__", "unknown"),
                         "numpy": np.__version__, "torch": torch.__version__,
                         "xgboost": xgboost.__version__, "sklearn": sklearn.__version__},
        }, indent=1))
        return path

    def push_to_hub(self, repo_id: str, *, private: bool = False,
                    commit_message: str = "Update fusion ensemble",
                    staging_dir: Optional[Union[str, Path]] = None) -> str:
        """Save this ensemble and upload it to a Hugging Face Hub model repo.

        The inverse of :meth:`from_pretrained` with a Hub ``model_id``.

        Args:
            repo_id: Hugging Face Hub model repo id to create (if needed) and upload to.
            private: Create the repo as private if it does not exist yet.
            commit_message: Commit message for the upload.
            staging_dir: Directory to write the saved ensemble into before uploading.
                Default: a temporary directory removed once the upload finishes.

        Returns:
            The commit sha ``upload_folder`` reports.
        """
        if staging_dir is not None:
            directory = self.save(staging_dir)
        else:
            with tempfile.TemporaryDirectory() as tmp:
                return self._upload(self.save(Path(tmp) / "fusion_ensemble"), repo_id,
                                    private, commit_message)
        return self._upload(directory, repo_id, private, commit_message)

    @staticmethod
    def _upload(directory: Path, repo_id: str, private: bool, commit_message: str) -> str:
        """Create (if needed) the model repo and upload every file in ``directory``."""
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id, repo_type="model", exist_ok=True, private=private)
        commit = api.upload_folder(repo_id=repo_id, repo_type="model",
                                   folder_path=str(directory), commit_message=commit_message)
        return getattr(commit, "oid", str(commit))

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

    def calibrate(self, X, y) -> "FusionEnsemble":
        """Fit each member's score-to-probability map on rows no member was fitted on.

        A member whose model does not itself output probabilities -- the GP -- scores on a
        latent scale, and only a held-out set can turn that into a probability. The set must be
        held out from every member's training rows: fitted on scores the members have already
        seen, the map comes out overconfident.

        The scores are taken from the same member scores :meth:`predict` uses, so the map is
        fitted on the scale it will be applied to.

        Args:
            X: Design matrix of the calibration rows, in this ensemble's block layout.
            y: Binary labels (0 or 1) for those rows.

        Returns:
            self

        Raises:
            ValueError: If the task is not binary, if every member outputs probabilities
                natively and there is nothing to calibrate, or if the labels are empty,
                non-finite or hold a single class.
        """
        if TASK_TYPES[self.task] != "binary":
            raise ValueError(f"only a binary task needs calibration; this ensemble predicts "
                             f"{self.task!r}")
        if all(m.native_binary for m in self.members):
            raise ValueError("every member outputs probabilities natively; nothing to calibrate")
        check_labels(y, "binary", "calibration labels")
        labels = np.asarray(y).astype(int)
        X = np.asarray(X)
        for member in self.members:
            if member.native_binary:
                continue
            score = member._predict_model(member.model_, X)
            member.calibrator_ = LogisticRegression(C=1e4).fit(score[:, None], labels)
        return self

    @property
    def available_tasks(self) -> List[str]:
        """The tasks this ensemble can predict (one, the task it was fitted on)."""
        return [self.task]

    @property
    def promoted(self) -> List[bool]:
        """Per member: whether its GP's factorisation had to be promoted to float64."""
        return [bool(getattr(getattr(m, "model_", None), "promoted_", False))
                for m in self.members]

    # ---------------------------------------------------------------- context cache

    def transform_context(self, record: dict) -> FusionContext:
        """Encode one experimental context and fold it into every member that can.

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
        blocks = self.data.encoder.encode_context(record)
        values = self.data.encode_context(record)
        folds = [self._fold_member(member, blocks) for member in self.members]
        return FusionContext(values=values, blocks=blocks, source=dict(record), folds=folds)

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

    def predict(self, samples: Union[Sequence[str], pd.DataFrame, Sequence[dict]],
                context: Optional[FusionContext] = None,
                return_individual: bool = True,
                return_std: bool = True) -> FusionPrediction:
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
            if isinstance(samples, list):
                smiles = samples
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
        per_member = [self._member_scores(member, X, return_std) if n
                      else (np.empty(0), np.empty(0)) for member in self.members]
        return self._aggregate(per_member, ok, [None] * n, return_individual)

    def _predict_with_context(self, smiles: List[str], context: FusionContext,
                              return_std: bool):
        """Fast path: featurise the molecules once, score through each member's folded context."""
        n = len(smiles)
        if n == 0:
            return np.zeros(0, dtype=bool), [(np.empty(0), np.empty(0)) for _ in self.members]

        fp, desc, ok = self.data.featurizer.featurize(smiles)
        valid = np.flatnonzero(ok)
        if len(valid) < n:                       # copy only when something is invalid
            fp, desc = fp[valid], desc[valid]
        mol = {"fingerprint": fp, "descriptors": desc}

        folds = context.folds or [None] * len(self.members)
        X = None                                 # built once, only if some member needs it
        per_member = []
        for member, fold in zip(self.members, folds):
            mean, std = np.full(n, np.nan), np.zeros(n)
            if len(valid):
                if fold is not None:
                    mean[valid], std[valid] = self._member_scores(
                        member, mol, return_std, fold=fold)
                else:
                    if X is None:
                        X = self.data.assemble_features(context.values, fp, desc)
                    mean[valid], std[valid] = self._member_scores(member, X, return_std)
            per_member.append((mean, std))
        return ok, per_member

    def _predict_from_records(self, records: List[dict], return_std: bool):
        """Ordinary path: encode each record in full, then score it with every member."""
        n = len(records)
        if n == 0:
            return np.zeros(0, dtype=bool), [(np.empty(0), np.empty(0)) for _ in self.members]

        X, ok = self.data.encode(records, return_ok=True)
        valid = np.flatnonzero(ok)
        Xv = X if len(valid) == n else X[valid]
        per_member = []
        for member in self.members:
            mean, std = np.full(n, np.nan), np.zeros(n)
            if len(valid):
                mean[valid], std[valid] = self._member_scores(member, Xv, return_std)
            per_member.append((mean, std))
        return ok, per_member

    def _member_scores(self, member, X, return_std: bool, fold: Optional[dict] = None):
        """One member's prediction, in the reported units.

        Args:
            X: Design-matrix rows, or -- with ``fold`` given -- the raw molecular blocks to
                score through that member's folded context.
            return_std: Ask for a predictive standard deviation, if this member has one.
            fold: A :meth:`FusionEnsemble._fold_member` result for this member, to take the
                fast context-folded path instead of the ordinary design-matrix one.
        """
        wants_std = return_std and isinstance(member, GPInteraction)
        if fold is not None:
            result = member.model_.predict_in_context(X, fold, return_std=wants_std)
        elif wants_std:
            result = member._predict_model(member.model_, X, return_std=True)
        else:
            return member.predict(X), None
        score, std = result if wants_std else (result, None)
        return member.report(score, std)

    def _stacking_predict(self, member, X) -> "Tuple[np.ndarray, Optional[np.ndarray]]":
        """(mean, sigma) for one member, in its own reported units, for the stacking fit.

        Used only by fit_stacking/calibrate_stacking/predict_stacked — predict()/predict_matrix()
        are unaffected and remain GP-only, as documented in the design spec.

        Args:
            member: A GPInteraction, xgboost.XGBRegressor, or xgboost.XGBClassifier.
            X: Design matrix of the rows to score.

        Returns:
            ``(mean, sigma)``. ``sigma`` is the GP's own predictive std for a GPInteraction, or
            None for any XGBoost member -- Option A's constant residual sigma for an XGBoost
            regressor is computed once from D_fit by the caller (fit_stacking), not here.

        Raises:
            TypeError: If ``member`` is none of the three supported types.
        """
        X = np.asarray(X)
        if isinstance(member, GPInteraction):
            mean, sigma = member.predict(X, return_std=True)
            return mean, sigma
        if isinstance(member, xgb.XGBClassifier):
            return member.predict_proba(X)[:, 1], None
        if isinstance(member, xgb.XGBRegressor):
            return member.predict(X), None
        raise TypeError(
            f"unsupported stacking member type {type(member).__name__!r} for {member!r}; "
            "expected GPInteraction, xgboost.XGBRegressor, or xgboost.XGBClassifier")

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
