"""Serve trusted local research candidates with task-specific molecular features.

Each distinct molecular contract gets its own FusionData encoder; tasks with an
identical contract share featurization. Biological context tables remain fixed
and their fingerprint is checked against every artifact before serving.
"""
import json
import threading
from pathlib import Path

from tackai.fusion.candidate_artifact import CandidateArtifact
from tackai.fusion.context import ContextEncoder
from tackai.fusion.data import FusionData
from tackai.fusion.molecular_spec import MolecularFeatureEncoder
from tackai.serve.app import Service, _check_layout
from tackai.serve.schemas import InfoResponse, PredictResponse


class CandidateService:
    """Combine same-contract services into the existing batched HTTP API."""

    def __init__(self, services, tasks, model_source):
        self.services = list(services)
        self.tasks = list(tasks)
        self.model_source = str(model_source)
        self.lock = threading.Lock()

    def _combine(self, method, *args):
        outputs = [getattr(service, method)(*args) for service in self.services]
        invalid_counts = {output.n_invalid for output in outputs}
        if len(invalid_counts) != 1:
            raise ValueError("Molecular feature encoders disagree on invalid input count")
        masks = []
        for output in outputs:
            first_task = output.tasks[0]
            first_values = next(iter(output.predictions[first_task].values()))
            masks.append(tuple(value is None for value in first_values))
        if len(set(masks)) != 1:
            raise ValueError("Molecular feature encoders disagree on invalid input positions")
        predictions = {task: values for output in outputs for task, values in output.predictions.items()}
        return PredictResponse(tasks=self.tasks,
                               predictions={task: predictions[task] for task in self.tasks},
                               n_invalid=outputs[0].n_invalid)

    def predict_shared(self, smiles, context):
        return self._combine("predict_shared", smiles, context)

    def predict_records(self, records):
        return self._combine("predict_records", records)

    def info(self):
        infos = [service.info() for service in self.services]
        task_info = {task: info.tasks[task] for info in infos for task in info.tasks}
        widths = {task: info.n_columns for info in infos for task in info.tasks}
        unique_widths = set(widths.values())
        return InfoResponse(tasks={task: task_info[task] for task in self.tasks},
                            model_source=self.model_source,
                            has_default_context=infos[0].has_default_context,
                            n_columns=next(iter(unique_widths)) if len(unique_widths) == 1 else None,
                            n_columns_by_task=widths, tackai_version=infos[0].tackai_version)

    def close(self):
        for service in self.services:
            service.close()


def load_candidate_service(*, model, tasks, context=None, context_repo=None,
                           protein_space=None, token=None, n_workers=1):
    """Load calibrated local candidate artifacts with frozen context verification.

    Artifacts are joblib/pickle files and must come from a trusted source. The
    parent directory contains a subdirectory per task. This does not modify or
    replace the default published XGBoost loading path.
    """
    root = Path(model)
    if not root.is_dir() or not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("A local candidate directory and distinct tasks are required")
    artifacts = {}
    for task in tasks:
        path = root / task
        if not path.is_dir() and len(tasks) == 1:
            path = root
        artifact = CandidateArtifact.from_pretrained(path)
        if artifact.recipe.get("task") != task:
            raise ValueError(f"Candidate recipe does not match requested task {task}")
        if artifact.task != ("classification" if task == "activity" else "regression"):
            raise ValueError(f"Candidate prediction type does not match task {task}")
        if (artifact.temperature_ is None if task == "activity" else artifact.q_hat_ is None):
            raise ValueError(f"Candidate for {task} is not calibrated")
        artifacts[task] = artifact

    grouped = {}
    for task, artifact in artifacts.items():
        key = json.dumps(artifact.molecular_spec.to_dict(), sort_keys=True)
        grouped.setdefault(key, {})[task] = artifact
    services = []
    encoder = None
    try:
        for ensembles in grouped.values():
            feature_spec = next(iter(ensembles.values())).molecular_spec
            featurizer = MolecularFeatureEncoder(feature_spec, n_workers=n_workers)
            if encoder is None and context_repo is not None:
                data = FusionData.from_pretrained(context_repo, token=token, featurizer=featurizer,
                                                  protein_space=protein_space)
                encoder = data.encoder
            else:
                if encoder is None:
                    encoder = ContextEncoder(protein_space=protein_space) if protein_space else ContextEncoder()
                data = FusionData(encoder=encoder, featurizer=featurizer)
            service = Service(ensembles, data, model_source=str(model), default_context=context)
            services.append(service)
            actual_context = encoder.fingerprint()
            for task, artifact in ensembles.items():
                if artifact.context_contract.get("context_fingerprint") != actual_context:
                    raise ValueError(f"Frozen context fingerprint mismatch for {task}")
                if artifact.context_contract.get("feature_dims") != data.dims:
                    raise ValueError(f"Frozen feature layout mismatch for {task}")
            _check_layout(ensembles, data)
            if context is not None:
                service.encoded_context(None)
    except Exception:
        for service in services:
            service.close()
        raise
    return CandidateService(services, tasks, model)
