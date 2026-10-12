"""Tests for the `tack serve` HTTP layer.

Everything here runs against the synthetic `fake_cache` and a two-member ensemble fitted
on noise: the point is the request/response contract — alignment, invalid-SMILES
handling, context reuse — not prediction quality.
"""
import numpy as np
import pytest
from fusion_fixtures import ASSAYS, CELLS, SEQS, SMILES

pytest.importorskip("fastapi", reason="needs the `serve` extra")

from fastapi.testclient import TestClient  # noqa: E402

from tackai.fusion.data import FusionData  # noqa: E402
from tackai.fusion.mol_encoder import MolEncoder  # noqa: E402
from tackai.fusion.xgb_ensemble import XGBStackedEnsemble  # noqa: E402
from tackai.serve.app import (Service, _check_layout, _expected_n_features,  # noqa: E402
                              create_app)
from tackai.serve.schemas import ContextSpec  # noqa: E402

BAD_SMILES = "this-is-not-a-molecule"


def _context() -> dict:
    return {"poi_seq": SEQS["poi"][0], "e3_seq": SEQS["e3"][0],
            "cell_id": CELLS[0], "assay": ASSAYS[0], "assay_time": 24.0}


def _ensemble(n_columns: int, task: str = "regression", n_models: int = 2):
    """Two XGBoost members fitted on noise of the right width."""
    from xgboost import XGBClassifier, XGBRegressor

    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, n_columns)).astype(np.float32)
    models = []
    for seed in range(n_models):
        if task == "regression":
            model = XGBRegressor(n_estimators=3, max_depth=2, random_state=seed)
            model.fit(X, rng.normal(size=40))
        else:
            model = XGBClassifier(n_estimators=3, max_depth=2, random_state=seed)
            model.fit(X, rng.integers(0, 2, size=40))
        models.append(model)
    return XGBStackedEnsemble(models)


@pytest.fixture
def service(fake_cache):
    """A Service over the synthetic cache serving all three tasks, with a default context."""
    data = FusionData(featurizer=MolEncoder(n_workers=1))
    ensembles = {"dmax": _ensemble(data.n_columns),
                 "pdc50": _ensemble(data.n_columns),
                 "activity": _ensemble(data.n_columns, task="classification")}
    service = Service(ensembles, data, model_source="test",
                      default_context=ContextSpec(**_context()))
    yield service
    service.close()


@pytest.fixture
def client(service):
    with TestClient(create_app(service)) as test_client:
        yield test_client


def test_health_and_info(client):
    assert client.get("/health").json() == {"status": "ok"}
    info = client.get("/info").json()
    assert set(info["tasks"]) == {"dmax", "pdc50", "activity"}
    assert info["tasks"]["dmax"]["task_type"] == "regression"
    assert info["tasks"]["activity"]["task_type"] == "classification"
    assert info["tasks"]["dmax"]["n_models"] == 2
    assert info["tasks"]["dmax"]["calibrated"] is False
    assert "mean" in info["tasks"]["dmax"]["output_keys"]
    assert info["has_default_context"] is True


def test_predict_uses_default_context(client):
    response = client.post("/predict", json={"smiles": SMILES[:4]})
    assert response.status_code == 200
    body = response.json()
    assert body["tasks"] == ["dmax", "pdc50", "activity"]
    assert set(body["predictions"]) == {"dmax", "pdc50", "activity"}
    assert body["n_invalid"] == 0
    for task in ("dmax", "pdc50", "activity"):
        for values in body["predictions"][task].values():
            assert len(values) == 4


def test_payload_keys_are_the_ensembles_own(client, service):
    """The response must not rename or reduce what `ensemble.predict` returns."""
    body = client.post("/predict", json={"smiles": SMILES[:2]}).json()
    for task, ensemble in service.ensembles.items():
        import numpy as np
        X = np.zeros((2, service.data.n_columns), dtype=np.float32)
        assert set(body["predictions"][task]) == set(ensemble.predict(X))
    assert {"mean", "std", "std_noise", "std_disagreement"} == set(body["predictions"]["dmax"])
    assert {"proba", "entropy_total", "entropy_aleatoric",
            "entropy_epistemic"} == set(body["predictions"]["activity"])


def test_invalid_smiles_is_none_not_an_error(client):
    """One malformed molecule must not fail a batch, and must stay in position."""
    smiles = [SMILES[0], BAD_SMILES, SMILES[1]]
    body = client.post("/predict", json={"smiles": smiles}).json()
    dmax = body["predictions"]["dmax"]
    assert dmax["mean"][1] is None and dmax["std"][1] is None
    assert dmax["mean"][0] is not None and dmax["mean"][2] is not None
    assert body["predictions"]["activity"]["proba"][1] is None
    assert body["n_invalid"] == 1
    assert len(dmax["mean"]) == 3


def test_all_invalid_batch(client):
    body = client.post("/predict", json={"smiles": [BAD_SMILES, BAD_SMILES]}).json()
    assert body["predictions"]["dmax"]["mean"] == [None, None]
    assert body["n_invalid"] == 2


def test_empty_batch_keeps_the_task_keys(client):
    """An empty batch is shaped like a full one, so a client can index it blindly."""
    body = client.post("/predict", json={"smiles": []}).json()
    assert body["n_invalid"] == 0
    assert body["predictions"]["dmax"]["mean"] == []
    assert {"mean", "std"} <= set(body["predictions"]["dmax"])
    assert "proba" in body["predictions"]["activity"]


def test_explicit_context_overrides_default(client, service):
    other = {**_context(), "cell_id": CELLS[1]}
    body = client.post("/predict", json={"smiles": SMILES[:2], "context": other}).json()
    assert len(body["predictions"]["dmax"]["mean"]) == 2
    assert ContextSpec(**other).cache_key() in service._context_cache
    # A second request without a context encodes the default separately, rather than
    # reusing whichever context happened to come through last.
    client.post("/predict", json={"smiles": SMILES[:2]})
    assert len(service._context_cache) == 2


def test_context_is_encoded_once_per_batch(client, service):
    """The whole point of the shared-context path: repeated requests re-use the vector."""
    service._context_cache.clear()
    for _ in range(3):
        client.post("/predict", json={"smiles": SMILES[:2], "context": _context()})
    assert len(service._context_cache) == 1


def test_records_path_with_heterogeneous_contexts(client, tiny_records):
    response = client.post("/predict/records", json={"records": tiny_records})
    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]["dmax"]["mean"]) == len(tiny_records)
    assert body["n_invalid"] == 0


def test_records_must_not_be_empty(client):
    assert client.post("/predict/records", json={"records": []}).status_code == 422


def test_unknown_field_is_rejected(client):
    """`extra="forbid"` turns a typo'd context key into a 422, not a silent default."""
    bad = {**_context(), "cell_line_id": CELLS[0]}
    assert client.post("/predict", json={"smiles": SMILES[:1], "context": bad}).status_code == 422


def test_missing_context_without_default_is_422(fake_cache):
    data = FusionData(featurizer=MolEncoder(n_workers=1))
    service = Service({"dmax": _ensemble(data.n_columns)}, data, model_source="test")
    try:
        with TestClient(create_app(service)) as test_client:
            response = test_client.post("/predict", json={"smiles": SMILES[:1]})
            assert response.status_code == 422
            assert "no context" in response.json()["detail"].lower()
    finally:
        service.close()


def test_a_single_task_server_serves_only_that_task(fake_cache):
    data = FusionData(featurizer=MolEncoder(n_workers=1))
    service = Service({"activity": _ensemble(data.n_columns, task="classification")},
                      data, model_source="test",
                      default_context=ContextSpec(**_context()))
    try:
        with TestClient(create_app(service)) as test_client:
            assert list(test_client.get("/info").json()["tasks"]) == ["activity"]
            body = test_client.post("/predict", json={"smiles": SMILES[:3]}).json()
            assert body["tasks"] == ["activity"]
            assert set(body["predictions"]) == {"activity"}
            assert all(0.0 <= v <= 1.0 for v in body["predictions"]["activity"]["proba"])
    finally:
        service.close()


def test_expected_n_features_reads_the_members(fake_cache):
    data = FusionData(featurizer=MolEncoder(n_workers=1))
    assert _expected_n_features(_ensemble(data.n_columns)) == data.n_columns


def test_a_matching_layout_passes(fake_cache):
    data = FusionData(featurizer=MolEncoder(n_workers=1))
    _check_layout({"dmax": _ensemble(data.n_columns)}, data)  # must not raise


def test_a_layout_mismatch_is_refused_before_serving(fake_cache):
    """The real failure mode behind `Feature shape mismatch`.

    A design matrix narrower than the members were fitted on is knowable at load time;
    letting it through means an xgboost ValueError on every request for the life of the
    server. The message has to name the protein space, which is what actually differs.
    """
    data = FusionData(featurizer=MolEncoder(n_workers=1))
    wrong = _ensemble(data.n_columns + 46)  # per_block vs combined, as published
    with pytest.raises(ValueError, match="protein embedding space"):
        _check_layout({"dmax": wrong}, data)


def test_the_mismatch_message_names_the_offending_task(fake_cache):
    data = FusionData(featurizer=MolEncoder(n_workers=1))
    ensembles = {"dmax": _ensemble(data.n_columns),
                 "pdc50": _ensemble(data.n_columns + 46)}
    with pytest.raises(ValueError, match="pdc50 on"):
        _check_layout(ensembles, data)


def test_openapi_schema_is_served(client):
    schema = client.get("/openapi.json").json()
    assert "/predict" in schema["paths"]
    assert "/predict/records" in schema["paths"]
