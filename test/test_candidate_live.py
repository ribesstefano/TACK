"""A loopback subprocess verifies candidate CLI startup and HTTP behavior."""
import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")
from fusion_fixtures import SMILES
from test_candidate_serve import _context, _write_candidate


def test_candidate_cli_on_loopback(fake_cache, tmp_path):
    root = tmp_path / "candidates"
    _write_candidate(root, "dmax", 32)
    _write_candidate(root, "activity", 64)
    port_file = tmp_path / "port"
    log = tmp_path / "server.log"
    command = [sys.executable, "-m", "tackai.cli", "serve", "--candidate-dir", str(root),
               "--task", "dmax", "--task", "activity", "--context", json.dumps(_context()),
               "--local-context", "--protein-space", "combined", "--port", "0",
               "--port-file", str(port_file), "--log-level", "warning"]
    with log.open("w+") as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 120
            while not port_file.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(.1)
            if not port_file.exists():
                output.flush()
                pytest.fail(f"Candidate server did not bind: {log.read_text()}")
            base = f"http://127.0.0.1:{int(port_file.read_text())}"
            while True:
                try:
                    with urllib.request.urlopen(base + "/health", timeout=2) as response:
                        assert json.load(response) == {"status": "ok"}
                    break
                except urllib.error.URLError:
                    if process.poll() is not None or time.monotonic() >= deadline:
                        output.flush()
                        pytest.fail(f"Candidate server did not become healthy: {log.read_text()}")
                    time.sleep(.1)
            request = urllib.request.Request(base + "/predict", method="POST",
                data=json.dumps({"smiles": [SMILES[0], "invalid-molecule", SMILES[1]]}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=30) as response:
                body = json.load(response)
            assert body["n_invalid"] == 1
            assert body["predictions"]["dmax"]["mean"][1] is None
            assert body["predictions"]["activity"]["proba"][1] is None
            assert all(len(values) == 3 for task in body["predictions"].values()
                       for values in task.values())
            client_path = Path(__file__).resolve().parents[1] / (
                "reinvent/reinvent_plugins/components/client_tack.py")
            spec = importlib.util.spec_from_file_location("client_tack", client_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            client = module.TackClient(url=base, max_batch=2, retries=0)
            chunked = client.predict([SMILES[0], "invalid-molecule", SMILES[1]])
            assert chunked["n_invalid"] == 1
            assert client.select(chunked, "dmax") == body["predictions"]["dmax"]["mean"]
            assert client.select(chunked, "activity") == body["predictions"]["activity"]["proba"]
            tackos_source = os.environ.get("TACKOS_TEST_SOURCE")
            if tackos_source:
                with pytest.MonkeyPatch.context() as patch:
                    patch.syspath_prepend(tackos_source)
                    from tackos.models.tack_http_predictor import TACKHTTPPredictor

                    predictor = TACKHTTPPredictor(base, tasks=["dmax", "activity"], max_batch=2)
                    result = predictor.predict([SMILES[0], "invalid-molecule", SMILES[1]])
                    assert result["dmax"][0] == body["predictions"]["dmax"]["mean"][0]
                    assert result["activity"][2] == body["predictions"]["activity"]["proba"][2]
                    assert all(value[1] != value[1] for value in result.values())
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        assert not port_file.exists()
