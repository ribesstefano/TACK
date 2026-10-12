"""Request and response models for the inference server.

Field names deliberately match :data:`tackai.fusion.data.RECORD_FIELDS` rather than the
legacy CSV column names (``POI_Sequence``, ``Cell_Line_ID``, …) so a JSON body is the
same shape as a record handed to :meth:`~tackai.fusion.data.FusionData.transform`. A
client that already builds fusion records can post them unchanged.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContextSpec(BaseModel):
    """One experimental context, shared by every molecule in a request.

    Mirrors the context half of a fusion record: everything except ``smiles``. A field
    left unset falls back to the encoder's own missing-key behaviour, which is
    deliberately asymmetric (a blank cell line gets the "unknown" vector, an absent POI
    accession raises) — see :mod:`tackai.fusion.context`.
    """

    model_config = ConfigDict(extra="forbid")

    # The two sequences are ~450-1300 residues, so they carry no inline example: a
    # truncated one would be invalid if copied. `reinvent/example_context.json` holds a
    # complete, working context instead.
    poi_seq: Optional[str] = Field(
        None, description="Full POI amino-acid sequence, as held in the cached POI table. "
                          "See reinvent/example_context.json for a complete example.")
    e3_seq: Optional[str] = Field(
        None, description="Full E3 ligase amino-acid sequence, as held in the cached E3 "
                          "table (e.g. CRBN, VHL).")
    cell_id: Optional[str] = Field(None, description="Cellosaurus cell line identifier.",
                                   examples=["CVCL_0030"])
    assay: Optional[str] = Field(None, description="Assay type; free text is canonised, and "
                                                  "an unknown type falls back to the "
                                                  "'unknown' vector with a warning.",
                                 examples=["Western blot"])
    assay_time: Optional[float] = Field(None, description="Treatment time in hours.",
                                        examples=[24.0])

    def cache_key(self) -> str:
        """Stable key for memoising this context's encoded vector."""
        return self.model_dump_json()


class Record(ContextSpec):
    """A full inference record: one molecule together with its own context."""

    smiles: Optional[str] = Field(None, description="SMILES string.")


class PredictRequest(BaseModel):
    """Many molecules under one shared context — the generative-loop path.

    ``context`` may be omitted when the server was started with a default context, which
    is the usual arrangement for a REINVENT run: the context is fixed for the whole run,
    so repeating it in every request is noise.
    """

    # `context: null` in the example so "Try it out" at /docs works as-is against a
    # server started with --context, which is the arrangement the README recommends.
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "smiles": ["Cc1sc2c(c1C)C(c1ccc(Cl)cc1)=N[C@@H](CC(=O)Nc1ccc(OCCOCCOCCOCCNc3cccc4"
                   "c3C(=O)N(C3CCC(=O)NC3=O)C4=O)cc1)c1nnc(C)n1-2"],
        "context": None,
    }})

    smiles: List[str] = Field(..., description="SMILES to score, in order.")
    context: Optional[ContextSpec] = Field(
        None, description="Overrides the server's --context default. Omit (or send null) "
                          "to use that default.")


class PredictRecordsRequest(BaseModel):
    """Molecules with differing contexts — the general path.

    Slower than :class:`PredictRequest` because the context blocks are re-encoded per
    row instead of once per batch, so prefer the shared-context form when the context
    really is shared.
    """

    model_config = ConfigDict(extra="forbid")

    records: List[Record] = Field(..., description="Records to score, in order.")

    @model_validator(mode="after")
    def _non_empty(self) -> "PredictRecordsRequest":
        # An empty list is accepted by `smiles` (a degenerate but well-defined batch);
        # here it almost always means the client built its records wrongly.
        if not self.records:
            raise ValueError("records must not be empty")
        return self


class PredictResponse(BaseModel):
    """One entry per loaded task, each holding that ensemble's own output.

    ``predictions[task]`` carries exactly the keys
    :meth:`~tackai.fusion.xgb_ensemble.XGBStackedEnsemble.predict` returns for that
    ensemble — ``mean``/``std``/``std_noise``/``std_disagreement`` for a regression task,
    plus ``lower``/``upper`` once it is calibrated, and
    ``proba``/``entropy_total``/``entropy_aleatoric``/``entropy_epistemic`` for a binary
    one. Nothing is renamed or reduced on the way out, so a client that knows the Python
    API already knows this payload, and a key added to the ensemble appears here without
    a change to the server.

    Every list has the same length as the input, with ``None`` wherever RDKit could not
    parse the SMILES. Callers rely on that alignment: a generative loop needs one value
    per proposed molecule and cannot recover from a silently shortened array.
    """

    model_config = ConfigDict(json_schema_extra={"example": {
        "tasks": ["dmax", "pdc50", "activity"],
        "predictions": {
            "dmax": {"mean": [0.936], "std": [0.109], "std_noise": [0.097],
                     "std_disagreement": [0.049], "lower": [0.807], "upper": [1.066]},
            "pdc50": {"mean": [8.881], "std": [0.854], "std_noise": [0.740],
                      "std_disagreement": [0.426], "lower": [6.856], "upper": [10.906]},
            "activity": {"proba": [0.872], "entropy_total": [0.382],
                         "entropy_aleatoric": [0.154], "entropy_epistemic": [0.156]},
        },
        "n_invalid": 0,
    }})

    tasks: List[str] = Field(..., description="Tasks this response covers, in report order.",
                             examples=[["dmax", "pdc50", "activity"]])
    predictions: Dict[str, Dict[str, List[Optional[float]]]] = Field(
        ..., description="Per task, that ensemble's own `predict` output as lists, one "
                         "value per input molecule and `null` where the SMILES did not "
                         "parse.")
    n_invalid: int = Field(0, description="How many SMILES failed to parse.", examples=[0])


# A task name keys a mapping rather than naming a field, so the generated OpenAPI schema
# has nothing to show for it and renders placeholders ("additionalProp1", "string", 0).
# These examples replace that with what the published ensembles actually return, which is
# the only version of this payload anyone reads at /docs.
_REGRESSION_KEYS = ["mean", "std", "std_noise", "std_disagreement", "lower", "upper"]
_CLASSIFICATION_KEYS = ["proba", "entropy_total", "entropy_aleatoric", "entropy_epistemic"]

_TASK_INFO_EXAMPLE = {
    "dmax": {"task_type": "regression", "n_models": 25, "calibrated": True,
             "output_keys": _REGRESSION_KEYS},
    "pdc50": {"task_type": "regression", "n_models": 25, "calibrated": True,
              "output_keys": _REGRESSION_KEYS},
    "activity": {"task_type": "classification", "n_models": 25, "calibrated": True,
                 "output_keys": _CLASSIFICATION_KEYS},
}


class TaskInfo(BaseModel):
    """What one loaded ensemble is and what its entry in `predictions` will hold."""

    model_config = ConfigDict(json_schema_extra={"example": _TASK_INFO_EXAMPLE["dmax"]})

    task_type: str = Field(..., description="'regression' or 'classification'.",
                           examples=["regression"])
    n_models: int = Field(..., description="Members in this ensemble.", examples=[25])
    calibrated: bool = Field(..., description="Whether conformal (regression) or temperature "
                                              "(classification) calibration is fitted. The "
                                              "published ensembles are calibrated, which is "
                                              "what adds 'lower'/'upper' to a regression task.",
                             examples=[True])
    output_keys: List[str] = Field(..., description="Keys this task's predictions will carry.",
                                   examples=[_REGRESSION_KEYS])


class InfoResponse(BaseModel):
    """What this server is serving, for a client to check it reached the right one."""

    model_config = ConfigDict(json_schema_extra={"example": {
        "tasks": _TASK_INFO_EXAMPLE,
        "model_source": "ailab-bio/TACK-Fast-Ensemble",
        "has_default_context": True,
        "n_columns": 1400,
        "tackai_version": "2.0.0",
    }})

    tasks: Dict[str, TaskInfo] = Field(
        ..., description="One entry per loaded task, keyed by task name.")
    model_source: str = Field(..., description="Directory or Hub repo the ensembles came from.",
                              examples=["ailab-bio/TACK-Fast-Ensemble"])
    has_default_context: bool = Field(
        ..., description="Whether the server was started with --context, so requests may "
                         "omit one.", examples=[True])
    n_columns: int = Field(..., description="Width of the shared design matrix. Must equal "
                                           "what the members were fitted on; the server "
                                           "refuses to start otherwise.", examples=[1400])
    tackai_version: str = Field(..., description="Version of tackai serving this.",
                                examples=["2.0.0"])
