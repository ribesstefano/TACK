"""Candidate serving checks feature contracts and keeps batches aligned."""
import numpy as np
import pytest
from fusion_fixtures import ASSAYS, CELLS, SEQS, SMILES
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from tackai.fusion.candidate_artifact import CandidateArtifact
from tackai.fusion.context import ContextEncoder
from tackai.fusion.data import FusionData
from tackai.fusion.molecular_spec import MolecularFeatureEncoder, MolecularFeatureSpec
from tackai.serve.app import create_app
from tackai.serve.candidate import load_candidate_service
from tackai.serve.schemas import ContextSpec


def _context():
    return dict(poi_seq=SEQS["poi"][0], e3_seq=SEQS["e3"][0],
                cell_id=CELLS[0], assay=ASSAYS[0], assay_time=24.)


def _write_candidate(root, task, width, *, fingerprint=None, calibrated=True):
    spec = MolecularFeatureSpec(radius=2, size=width)
    encoder = ContextEncoder(protein_space="combined")
    data = FusionData(encoder=encoder, featurizer=MolecularFeatureEncoder(spec, n_workers=1))
    rng = np.random.default_rng(width)
    X = rng.normal(size=(50, data.n_columns)).astype(np.float32)
    classification = task == "activity"
    y = (np.arange(50) % 2).astype(float) if classification else rng.normal(size=50)
    cls = ExtraTreesClassifier if classification else ExtraTreesRegressor
    model = cls(n_estimators=3, random_state=2, n_jobs=1).fit(X[:25], y[:25])
    artifact = CandidateArtifact([model], "classification" if classification else "regression",
        spec, recipe={"task": task}, context_contract={
            "context_fingerprint": fingerprint or encoder.fingerprint(), "feature_dims": data.dims})
    if calibrated:
        if not classification:
            artifact.fit_residual_scale(X[25:35], y[25:35])
        artifact.calibrate(X[35:], y[35:], alpha=.2)
    artifact.save_pretrained(root / task)
    data.featurizer.close()


def test_mixed_molecular_features_http_alignment_and_context(fake_cache, tmp_path):
    root = tmp_path / "candidates"
    for task, width in (("dmax", 32), ("pdc50", 64), ("activity", 32)):
        _write_candidate(root, task, width)
    service = load_candidate_service(model=root, tasks=["dmax", "pdc50", "activity"],
                                     context=ContextSpec(**_context()), protein_space="combined")
    with TestClient(create_app(service)) as client:
        info = client.get("/info").json()
        assert info["n_columns"] is None
        assert info["n_columns_by_task"]["pdc50"] > info["n_columns_by_task"]["dmax"]
        assert all(info["tasks"][task]["calibrated"] for task in info["tasks"])
        molecules = [SMILES[0], "invalid-molecule", SMILES[1]]
        body = client.post("/predict", json={"smiles": molecules}).json()
        assert body["tasks"] == ["dmax", "pdc50", "activity"]
        assert body["n_invalid"] == 1
        for task in body["tasks"]:
            for values in body["predictions"][task].values():
                assert len(values) == 3 and values[1] is None
                assert values[0] is not None and values[2] is not None
        assert "mean" in body["predictions"]["dmax"]
        assert "proba" in body["predictions"]["activity"]
        records = [dict(smiles=smiles, **_context()) for smiles in molecules]
        by_record = client.post("/predict/records", json={"records": records}).json()
        assert by_record["predictions"] == body["predictions"]


@pytest.mark.parametrize("bad_context,uncalibrated", [(True, False), (False, True)])
def test_candidate_server_refuses_wrong_context_or_uncalibrated(fake_cache, tmp_path,
                                                                 bad_context, uncalibrated):
    root = tmp_path / "candidates"
    _write_candidate(root, "dmax", 32, fingerprint="wrong" if bad_context else None,
                     calibrated=not uncalibrated)
    with pytest.raises(ValueError, match="fingerprint|not calibrated"):
        load_candidate_service(model=root, tasks=["dmax"], protein_space="combined")
