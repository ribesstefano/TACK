"""FastAPI application serving one fusion ensemble over HTTP.

Design notes worth keeping in mind before changing anything here:

* **One process, one ensemble, one worker.** Each uvicorn worker would hold its own
  copy of the ensemble *and* its own :class:`~tackai.fusion.mol_encoder.MolEncoder`
  memo cache, so replicating workers multiplies memory and throws away cache hits
  without buying parallelism that matters. Scale with batch size instead — the
  featuriser and XGBoost both vectorise over rows.
* **Requests are serialised by a lock.** The endpoints are defined ``def`` (not
  ``async def``), so starlette runs them in a thread pool; without the lock two
  concurrent batches would mutate ``MolEncoder.cache`` and ``MolEncoder.stats``
  concurrently. Serialising costs nothing for the intended single-client loop and
  removes a whole class of rare corruption.
* **A bad molecule is not an error.** An unparseable SMILES yields ``None`` in place,
  not a 4xx for the whole batch. A generative loop sends whatever its policy proposes
  and needs an array it can align with its population; failing the request would abort
  a run thousands of steps in over one malformed string.
* **The context is encoded once and memoised.** In a generative run the experimental
  context is constant, so :meth:`~tackai.fusion.data.FusionData.transform_context` is
  pure overhead after the first call. Cached by the request body's own JSON, which is
  exact rather than merely plausible as a cache key.
"""
from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from fastapi import FastAPI, HTTPException

from tackai.serve.schemas import (ContextSpec, InfoResponse, PredictRecordsRequest,
                                  PredictRequest, PredictResponse, Record, TaskInfo)

logger = logging.getLogger("tackai.serve")

# Contexts per run are few (usually exactly one); the cap only stops an unbounded client
# from growing the dict forever.
MAX_CONTEXT_CACHE = 32


def _nan_to_none(values: np.ndarray, ok: np.ndarray) -> List[Optional[float]]:
    """Mask out rows whose SMILES did not parse, as JSON-safe floats.

    Also maps any non-finite model output to ``None``: ``float('nan')`` is not valid
    JSON, and silently emitting it produces a response some clients reject and others
    parse as a number.
    """
    out: List[Optional[float]] = []
    for value, good in zip(np.asarray(values, dtype=float).ravel(), ok):
        out.append(float(value) if good and np.isfinite(value) else None)
    return out


def _output_keys(ensemble) -> List[str]:
    """Keys this ensemble's `predict` will return, asked of the ensemble itself.

    Used to shape an empty batch like a full one. `help()` already derives this from the
    task and the calibration state, so there is no second copy of that logic here.
    """
    try:
        return list(ensemble.help()["output_keys"])
    except Exception:
        # Introspection is a convenience, never a reason to fail a request.
        return []


class Service:
    """The loaded ensembles, their shared encoder, and the state shared across requests.

    One ensemble per task, all consuming the *same* design matrix: the context blocks and
    the molecular features do not depend on the task, so a request featurises once and
    every loaded ensemble scores those same rows.

    That sharing is what makes serving all three tasks affordable, but it is not free.
    Measured on 128 freshly proposed molecules (a cold featuriser cache, as in a
    generative loop) against the published ensembles: featurising costs ~710 ms and
    dominates, while predicting rises from ~60 ms for one task to ~195 ms for three — so
    three tasks cost roughly 18% more per request than one, not three times as much.
    Where molecules repeat and the memo cache is warm, featurisation falls away and the
    ratio approaches the full 3x, so a loop that only ever reads `dmax` is still better
    served by `--task dmax`.

    Args:
        ensembles: Mapping of task name to loaded ensemble, in the order to report.
        data: Encoder-only :class:`~tackai.fusion.data.FusionData` shared by every task.
        model_source: Where the ensembles came from, for ``/info``.
        default_context: Context applied when a request omits one.
    """

    def __init__(self, ensembles: Dict[str, object], data, *, model_source: str,
                 default_context: Optional[ContextSpec] = None):
        if not ensembles:
            raise ValueError("at least one ensemble is required")
        self.ensembles = dict(ensembles)
        self.data = data
        self.model_source = model_source
        self.default_context = default_context
        self.lock = threading.Lock()
        self._context_cache: Dict[str, np.ndarray] = {}

    @property
    def tasks(self) -> List[str]:
        """The loaded task names, in report order."""
        return list(self.ensembles)

    def encoded_context(self, context: Optional[ContextSpec]) -> np.ndarray:
        """Encode (and memoise) one context, falling back to the server's default."""
        context = context or self.default_context
        if context is None:
            raise HTTPException(
                status_code=422,
                detail="No context given and the server was started without a default. "
                       "Post a 'context' object, or start `tack serve --context FILE`.",
            )
        key = context.cache_key()
        if key not in self._context_cache:
            if len(self._context_cache) >= MAX_CONTEXT_CACHE:
                self._context_cache.clear()
            record = context.model_dump(exclude_none=True)
            try:
                self._context_cache[key] = self.data.transform_context(record)
            except Exception as e:
                # An absent POI/E3/cell key raises by design rather than falling back to
                # an "unknown" vector, so this is a client-side mistake worth naming.
                raise HTTPException(
                    status_code=422, detail=f"Could not encode context {record}: {e}"
                ) from e
        return self._context_cache[key]

    def score(self, X: np.ndarray, ok: np.ndarray) -> PredictResponse:
        """Run every loaded ensemble over one already-built design matrix.

        Each task's entry is its ensemble's own ``predict`` dict, converted to lists and
        masked where the SMILES did not parse. Keys are passed through rather than
        renamed, so the payload tracks the Python API by construction.
        """
        predictions: Dict[str, Dict[str, List[Optional[float]]]] = {}
        for task, ensemble in self.ensembles.items():
            if X.shape[0] == 0:
                # No rows to predict, but the keys must still be the ones this task
                # would have produced, so an empty batch is shaped like a full one.
                predictions[task] = {k: [] for k in _output_keys(ensemble)}
                continue
            predictions[task] = {k: _nan_to_none(v, ok)
                                 for k, v in ensemble.predict(X).items()}
        return PredictResponse(tasks=self.tasks, predictions=predictions,
                               n_invalid=int((~ok).sum()))

    def predict_shared(self, smiles: List[str], context: Optional[ContextSpec]
                       ) -> PredictResponse:
        """Score many molecules under one context (the fast path)."""
        encoded = self.encoded_context(context)
        if not smiles:
            return self.score(np.empty((0, self.data.n_columns), dtype=np.float32),
                              np.empty(0, dtype=bool))
        # featurize rather than `assemble`, which discards the ok mask we must report.
        fp, desc, ok = self.data.featurizer.featurize(smiles)
        return self.score(self.data.assemble_features(encoded, fp, desc), ok)

    def predict_records(self, records: List[Record]) -> PredictResponse:
        """Score molecules that each carry their own context (the general path)."""
        payload = [r.model_dump(exclude_none=True) for r in records]
        try:
            X, ok = self.data.transform(payload, return_ok=True)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Could not encode records: {e}") from e
        return self.score(X, ok)

    def info(self) -> InfoResponse:
        """Describe every loaded ensemble and the matrix they share."""
        try:
            from importlib.metadata import version
            tackai_version = version("tackai")
        except Exception:
            tackai_version = "unknown"
        tasks = {
            task: TaskInfo(
                task_type=ensemble.task,
                n_models=len(ensemble.models),
                calibrated=(ensemble.q_hat_ is not None
                            or ensemble.temperature_ is not None),
                output_keys=_output_keys(ensemble),
            )
            for task, ensemble in self.ensembles.items()
        }
        return InfoResponse(
            tasks=tasks,
            model_source=self.model_source,
            has_default_context=self.default_context is not None,
            n_columns=int(self.data.n_columns),
            tackai_version=tackai_version,
        )

    def close(self) -> None:
        """Release the featuriser's worker pool, if it started one."""
        self.data.featurizer.close()


def _expected_n_features(ensemble) -> Optional[int]:
    """How many columns the loaded members were fitted on, or ``None`` if unreported.

    Asked of the members rather than the ensemble because only they know: the ensemble
    never sees the design matrix's width until a prediction is already under way.
    """
    for model in ensemble.models:
        n = getattr(model, "n_features_in_", None)
        if n is None:
            try:
                n = model.get_booster().num_features()
            except Exception:
                n = None
        if n:
            return int(n)
    return None


def _check_layout(ensembles: Dict[str, object], data) -> None:
    """Refuse a design matrix whose width the loaded models cannot consume.

    Without this the mismatch surfaces as an xgboost ``Feature shape mismatch`` on every
    request — a 500 per batch for the whole life of the server, when the problem was
    knowable before it ever started listening. The usual cause is the protein embedding
    space: ``per_block`` and ``combined`` give different POI and E3 widths, so a server
    reading the local cache under the default space cannot serve models trained in the
    other one.

    Raises:
        ValueError: If the widths disagree, naming the likely remedy.
    """
    mismatched = {}
    for task, ensemble in ensembles.items():
        expected = _expected_n_features(ensemble)
        if expected is not None and expected != data.n_columns:
            mismatched[task] = expected
    if not mismatched:
        return
    detail = ", ".join(f"{task} on {n}" for task, n in mismatched.items())
    raise ValueError(
        f"The loaded models were fitted on a different width than this server builds "
        f"({data.n_columns} columns): {detail}. Block widths are {data.dims}.\n"
        "The POI and E3 widths depend on the protein embedding space, so this almost "
        "always means the space does not match the one the models were trained in. "
        "Either pass --context-repo ailab-bio/TACK-Context-Encoding, whose manifest "
        "records the space its tables were built in, or name the space explicitly with "
        "--protein-space combined|per_block against the local cache."
    )


def load_service(*, model: str, tasks: Sequence[str],
                 context: Optional[ContextSpec] = None,
                 context_repo: Optional[str] = None,
                 protein_space: Optional[str] = None,
                 token: Optional[str] = None,
                 n_workers: int = 1) -> Service:
    """Load one ensemble per task, plus the context tables, into a :class:`Service`.

    Each task's ensemble lives in the subfolder named after it, which is how the
    published repo is laid out, so the task name *is* the subfolder — there is no
    separate knob that could disagree with it.

    Args:
        model: Local directory or Hugging Face repo id holding the ensembles, one per
            task subfolder. A local directory that is itself a single ensemble (no task
            subfolders) is also accepted when exactly one task is requested.
        tasks: Task names to load, in report order (``dmax``, ``pdc50``, ``activity``).
        context: Default context applied when a request omits one.
        context_repo: Where to read the context embedding tables from. ``None`` uses the
            local ``TACKAI_CACHE`` only, via a plain
            :class:`~tackai.fusion.data.FusionData`; a repo id or directory goes through
            :meth:`~tackai.fusion.data.FusionData.from_pretrained`, which may download
            and which reads the protein space its manifest records.
        protein_space: ``"per_block"`` or ``"combined"``, choosing the POI/E3 embedding
            space and therefore those blocks' widths. ``None`` leaves it to the encoder
            default (``per_block``) for the local cache, or to the manifest when
            `context_repo` is given.
        token: Hugging Face token, for private repos.
        n_workers: Featuriser worker processes.

    Returns:
        A ready :class:`Service`.

    Raises:
        ValueError: If the resulting design matrix is not the width the loaded models
            were fitted on.
    """
    from tackai.fusion.context import ContextEncoder
    from tackai.fusion.data import FusionData
    from tackai.fusion.mol_encoder import MolEncoder
    from tackai.fusion.xgb_ensemble import XGBStackedEnsemble

    featurizer = MolEncoder(n_workers=n_workers)
    if context_repo is None:
        encoder = ContextEncoder(protein_space=protein_space) if protein_space else None
        data = FusionData(encoder=encoder, featurizer=featurizer)
    else:
        data = FusionData.from_pretrained(context_repo, token=token, featurizer=featurizer,
                                          protein_space=protein_space)

    ensembles = {}
    for task in tasks:
        try:
            ensemble = XGBStackedEnsemble.from_pretrained(model, subfolder=task, token=token)
            source = f"{model}/{task}"
        except FileNotFoundError:
            # A local directory that holds one ensemble directly, rather than a task
            # subfolder per ensemble. Only unambiguous for a single requested task.
            if len(tasks) > 1 or not Path(model).is_dir():
                raise
            ensemble = XGBStackedEnsemble.from_pretrained(model, token=token)
            source = model
        logger.info(f"Loaded {len(ensemble.models)} models ({ensemble.task}) "
                    f"for {task} from {source}")
        ensembles[task] = ensemble

    # Checked here so a width mismatch stops startup rather than failing every request.
    _check_layout(ensembles, data)
    service = Service(ensembles, data, model_source=model, default_context=context)
    if context is not None:
        service.encoded_context(None)
    return service


def create_app(service: Service) -> FastAPI:
    """Build the FastAPI app around an already-loaded :class:`Service`."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        service.close()

    app = FastAPI(
        title="TACK fusion inference",
        description=__doc__,
        version=service.info().tackai_version,
        lifespan=lifespan,
    )
    app.state.service = service

    @app.get("/health")
    def health() -> dict:
        """Liveness probe: cheap, and touches neither the model nor the cache."""
        return {"status": "ok"}

    @app.get("/info", response_model=InfoResponse)
    def info() -> InfoResponse:
        """What this server is serving."""
        return service.info()

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        """Score molecules under one shared context."""
        with service.lock:
            return service.predict_shared(request.smiles, request.context)

    @app.post("/predict/records", response_model=PredictResponse)
    def predict_records(request: PredictRecordsRequest) -> PredictResponse:
        """Score molecules that each carry their own context."""
        with service.lock:
            return service.predict_records(request.records)

    return app
