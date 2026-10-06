"""Publishing the fusion context embeddings to the Hub, and installing them back into
TACKAI_CACHE via FusionData.from_pretrained."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import publish_fusion_context as pfc  # noqa: E402

from fusion_fixtures import (ASSAY_FILE, ASSAY_PCA_FILE, CELL_FILE, COMBINED_FILE, E3_FILE,
                             POI_FILE)
from tackai.fusion.context import CONTEXT_FILES


def _all_context_filenames():
    from tackai.fusion.context import CONTEXT_FILES
    return [filename for files in CONTEXT_FILES.values() for filename in files.values()]


def test_fake_cache_has_every_context_files_entry(fake_cache):
    for block, files in CONTEXT_FILES.items():
        for role, filename in files.items():
            assert (fake_cache / filename).exists(), f"missing {block}/{role}: {filename}"


def test_default_context_repo_is_exported():
    from tackai.fusion import DEFAULT_CONTEXT_REPO
    assert DEFAULT_CONTEXT_REPO == "ailab-bio/TACK-fusion-context"


def test_stage_copies_all_ten_files_and_writes_a_manifest(fake_cache, tmp_path):
    out = tmp_path / "staged"
    manifest = pfc.stage(fake_cache, out)
    for name in _all_context_filenames():
        assert (out / name).exists()
    assert (out / "manifest.json").exists()
    assert manifest["block_dims"] == {"e3": 7, "cell": 47, "poi": 51, "assay": 8,
                                      "assay_time": 1}
    assert manifest["combined_dim"] == 52


def test_stage_hashes_match_the_staged_bytes(fake_cache, tmp_path):
    out = tmp_path / "staged"
    manifest = pfc.stage(fake_cache, out)
    for name, entry in manifest["files"].items():
        assert pfc.sha256_of(out / name) == entry["sha256"]


def test_stage_pca_model_entries_record_the_reduced_dimension(fake_cache, tmp_path):
    from tackai.fusion.context import CONTEXT_FILES
    manifest = pfc.stage(fake_cache, tmp_path / "staged")
    for block, files in CONTEXT_FILES.items():
        table_dim = manifest["files"][files["table"]]["dim"]
        reduced_dim = manifest["files"][files["pca_model"]]["reduced_dim"]
        assert table_dim == reduced_dim, block


def test_stage_refuses_a_partial_set(fake_cache, tmp_path, capsys):
    missing_name = POI_FILE.replace(".npz", "_model.npz")
    (fake_cache / missing_name).unlink()
    with pytest.raises(SystemExit):
        pfc.stage(fake_cache, tmp_path / "staged")
    assert missing_name in capsys.readouterr().out


def test_stage_refuses_an_existing_output_directory(fake_cache, tmp_path):
    out = tmp_path / "staged"
    out.mkdir()
    with pytest.raises(SystemExit):
        pfc.stage(fake_cache, out)


def test_upload_calls_create_repo_and_upload_folder(fake_cache, tmp_path, monkeypatch):
    out = tmp_path / "staged"
    pfc.stage(fake_cache, out)
    calls = {}

    class FakeApi:
        def whoami(self):
            return {"name": "test-user"}

        def create_repo(self, repo_id, repo_type, exist_ok, private):
            calls["create_repo"] = (repo_id, repo_type, exist_ok, private)

        def upload_folder(self, repo_id, repo_type, folder_path, commit_message):
            calls["upload_folder"] = (repo_id, repo_type, folder_path, commit_message)
            return type("Commit", (), {"oid": "abc123"})()

    monkeypatch.setattr("huggingface_hub.HfApi", lambda: FakeApi())
    sha = pfc.upload(out, "ailab-bio/TACK-fusion-context", private=False,
                     commit_message="test commit")
    assert calls["create_repo"] == ("ailab-bio/TACK-fusion-context", "dataset", True, False)
    assert calls["upload_folder"] == ("ailab-bio/TACK-fusion-context", "dataset", str(out),
                                      "test commit")
    assert sha == "abc123"
