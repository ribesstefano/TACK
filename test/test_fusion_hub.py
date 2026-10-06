"""Publishing the fusion context embeddings to the Hub, and installing them back into
TACKAI_CACHE via FusionData.from_pretrained."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import publish_fusion_context as pfc  # noqa: E402

from fusion_fixtures import (ASSAY_FILE, ASSAY_PCA_FILE, CELL_FILE, COMBINED_FILE, E3_FILE,
                             POI_FILE)
from tackai.fusion.context import CONTEXT_FILES
from tackai.fusion.data import FusionData


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


def test_from_pretrained_installs_into_an_empty_cache(fake_cache, tmp_path):
    staged = tmp_path / "staged"
    pfc.stage(fake_cache, staged)
    target = tmp_path / "fresh_cache"          # does not exist yet
    data = FusionData.from_pretrained(staged, cache_dir=target)
    assert data.table is None
    assert data.dims == {"fingerprint": 1024, "descriptors": 216, "e3": 7, "cell": 47,
                         "poi": 51, "assay": 8, "assay_time": 1}
    for name in _all_context_filenames():
        assert (target / name).exists()
    with pytest.raises(ValueError, match="no table"):
        _ = data.X


def test_from_pretrained_can_encode_a_context(fake_cache, tmp_path, tiny_records):
    staged = tmp_path / "staged"
    pfc.stage(fake_cache, staged)
    data = FusionData.from_pretrained(staged, cache_dir=tmp_path / "fresh_cache")
    row = tiny_records[0]
    ctx = data.encode_context({"poi_seq": row["poi_seq"], "e3_seq": row["e3_seq"],
                               "cell_id": row["cell_id"], "assay": row["assay"],
                               "assay_time": row["assay_time"]})
    assert ctx.shape == (1, len(data.context_columns))


def test_from_pretrained_second_call_is_a_no_op(fake_cache, tmp_path):
    staged = tmp_path / "staged"
    pfc.stage(fake_cache, staged)
    target = tmp_path / "fresh_cache"
    FusionData.from_pretrained(staged, cache_dir=target)
    before = {p.name: p.stat().st_mtime_ns for p in target.iterdir()}
    FusionData.from_pretrained(staged, cache_dir=target)
    after = {p.name: p.stat().st_mtime_ns for p in target.iterdir()}
    assert before == after


def test_from_pretrained_supports_the_combined_protein_space(fake_cache, tmp_path):
    """The combined table is published even though the default protein_space never reads
    it; protein_space="combined" must still work after a from-Hub install (spec test #9)."""
    staged = tmp_path / "staged"
    pfc.stage(fake_cache, staged)
    data = FusionData.from_pretrained(staged, cache_dir=tmp_path / "fresh_cache",
                                      protein_space="combined")
    assert data.dims["poi"] == 52 and data.dims["e3"] == 52


def test_from_pretrained_rejects_a_tampered_cached_file(fake_cache, tmp_path):
    staged = tmp_path / "staged"
    pfc.stage(fake_cache, staged)
    target = tmp_path / "fresh_cache"
    FusionData.from_pretrained(staged, cache_dir=target)
    rng = np.random.default_rng(0)
    np.savez(target / POI_FILE, **{s: rng.normal(size=51).astype(np.float32) for s in "xy"})
    with pytest.raises(ValueError, match=POI_FILE):
        FusionData.from_pretrained(staged, cache_dir=target)


def test_from_pretrained_force_download_overwrites(fake_cache, tmp_path):
    staged = tmp_path / "staged"
    pfc.stage(fake_cache, staged)
    target = tmp_path / "fresh_cache"
    FusionData.from_pretrained(staged, cache_dir=target)
    rng = np.random.default_rng(0)
    np.savez(target / POI_FILE, **{s: rng.normal(size=51).astype(np.float32) for s in "xy"})
    data = FusionData.from_pretrained(staged, cache_dir=target, force_download=True)
    assert data.dims["poi"] == 51


def test_from_pretrained_rejects_a_missing_manifest(tmp_path):
    empty_source = tmp_path / "empty_source"
    empty_source.mkdir()
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        FusionData.from_pretrained(empty_source, cache_dir=tmp_path / "cache")


def test_from_pretrained_rejects_a_block_width_mismatch(fake_cache, tmp_path):
    staged = tmp_path / "staged"
    pfc.stage(fake_cache, staged)
    manifest_path = staged / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["block_dims"]["cell"] = 48
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="cell"):
        FusionData.from_pretrained(staged, cache_dir=tmp_path / "fresh_cache")


def test_from_pretrained_descriptors_none_gives_fingerprint_only(fake_cache, tmp_path):
    staged = tmp_path / "staged"
    pfc.stage(fake_cache, staged)
    data = FusionData.from_pretrained(staged, cache_dir=tmp_path / "fresh_cache",
                                      descriptors=None)
    assert data.dims["descriptors"] == 0


def test_from_pretrained_gives_guidance_when_huggingface_hub_is_missing(tmp_path, monkeypatch):
    """A repo id (not a local path) with huggingface_hub unimportable must raise a clear
    ImportError, not a bare traceback. Setting the module to None in sys.modules is the
    standard trick to make `from huggingface_hub import ...` raise ImportError on demand."""
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(ImportError, match="huggingface_hub"):
        FusionData.from_pretrained("ailab-bio/TACK-fusion-context", cache_dir=tmp_path / "cache")


def test_from_pretrained_uses_snapshot_download_for_a_repo_id(fake_cache, tmp_path, monkeypatch):
    staged = tmp_path / "staged"
    pfc.stage(fake_cache, staged)
    calls = {}

    def fake_snapshot_download(repo_id, repo_type, revision, token):
        calls["args"] = (repo_id, repo_type, revision, token)
        return str(staged)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    data = FusionData.from_pretrained("ailab-bio/TACK-fusion-context",
                                      cache_dir=tmp_path / "fresh_cache")
    assert calls["args"] == ("ailab-bio/TACK-fusion-context", "dataset", None, None)
    assert data.dims["cell"] == 47
