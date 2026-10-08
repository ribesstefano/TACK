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
from tackai.fusion.context import CONTEXT_FILES, ContextEncoder
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


def test_stage_writes_a_readme_mentioning_the_repo_and_from_pretrained(fake_cache, tmp_path):
    out = tmp_path / "staged"
    pfc.stage(fake_cache, out, repo_id="ailab-bio/TACK-fusion-context")
    readme = (out / "README.md").read_text()
    assert "ailab-bio/TACK-fusion-context" in readme
    assert "from_pretrained" in readme


def test_stage_readme_defaults_to_the_default_context_repo(fake_cache, tmp_path):
    from tackai.fusion.context import DEFAULT_CONTEXT_REPO
    out = tmp_path / "staged"
    pfc.stage(fake_cache, out)
    readme = (out / "README.md").read_text()
    assert DEFAULT_CONTEXT_REPO in readme


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
    ctx = data.transform_context({"poi_seq": row["poi_seq"], "e3_seq": row["e3_seq"],
                                  "cell_id": row["cell_id"], "assay": row["assay"],
                                  "assay_time": row["assay_time"]})
    assert ctx.shape == (1, len(data.context_columns))


def test_from_pretrained_second_call_is_a_no_op(fake_cache, tmp_path, monkeypatch):
    """shutil.copy2 preserves the source mtime, so a stat()-based before/after comparison
    cannot tell a real no-op from a silent, identical re-copy. Assert on the actual signal: no
    copy2 call happens once every file already matches."""
    staged = tmp_path / "staged"
    pfc.stage(fake_cache, staged)
    target = tmp_path / "fresh_cache"
    FusionData.from_pretrained(staged, cache_dir=target)

    copy_calls = []
    monkeypatch.setattr("tackai.fusion.data.shutil.copy2",
                        lambda src, dst: copy_calls.append((src, dst)))
    FusionData.from_pretrained(staged, cache_dir=target)
    assert copy_calls == []


def test_from_pretrained_supports_the_combined_protein_space(fake_cache, tmp_path):
    """The combined table is published even though the default protein_space never reads
    it; protein_space="combined" must still work after a from-Hub install (spec test #9)."""
    staged = tmp_path / "staged"
    pfc.stage(fake_cache, staged)
    data = FusionData.from_pretrained(staged, cache_dir=tmp_path / "fresh_cache",
                                      protein_space="combined")
    assert data.dims["poi"] == 52 and data.dims["e3"] == 52


def test_stage_with_no_protein_space_leaves_the_manifest_key_none(fake_cache, tmp_path):
    """The maintainer CLI stages a whole cache, not one encoder's choice, so it must not
    commit to a space the manifest didn't actually observe."""
    staged = tmp_path / "staged"
    manifest = pfc.stage(fake_cache, staged)
    assert manifest["protein_space"] is None


def test_push_to_hub_records_the_encoders_protein_space_in_the_manifest(fake_cache, tmp_path):
    data = FusionData(encoder=ContextEncoder(cache_dir=fake_cache, protein_space="combined"))
    staged = tmp_path / "staged"

    data.push_to_hub("ailab-bio/TACK-fusion-context", staging_dir=staged, dry_run=True)

    manifest = json.loads((staged / "manifest.json").read_text())
    assert manifest["protein_space"] == "combined"


def test_from_pretrained_defaults_to_the_manifests_recorded_protein_space(fake_cache, tmp_path):
    """The bug this guards against: pushing a 'combined' encoder and loading it back with no
    protein_space argument must reproduce 'combined', not silently fall back to 'per_block'
    and hand back a context vector of the wrong width."""
    data = FusionData(encoder=ContextEncoder(cache_dir=fake_cache, protein_space="combined"))
    staged = tmp_path / "staged"
    data.push_to_hub("ailab-bio/TACK-fusion-context", staging_dir=staged, dry_run=True)

    reloaded = FusionData.from_pretrained(staged, cache_dir=tmp_path / "fresh_cache")

    assert reloaded.encoder.protein_space == "combined"
    assert reloaded.dims == data.dims
    assert len(reloaded.context_columns) == len(data.context_columns)


def test_from_pretrained_rejects_a_protein_space_contradicting_the_manifest(fake_cache, tmp_path):
    data = FusionData(encoder=ContextEncoder(cache_dir=fake_cache, protein_space="combined"))
    staged = tmp_path / "staged"
    data.push_to_hub("ailab-bio/TACK-fusion-context", staging_dir=staged, dry_run=True)

    with pytest.raises(ValueError, match="per_block"):
        FusionData.from_pretrained(staged, cache_dir=tmp_path / "fresh_cache",
                                   protein_space="per_block")


def test_from_pretrained_accepts_an_explicit_protein_space_matching_the_manifest(
        fake_cache, tmp_path):
    data = FusionData(encoder=ContextEncoder(cache_dir=fake_cache, protein_space="combined"))
    staged = tmp_path / "staged"
    data.push_to_hub("ailab-bio/TACK-fusion-context", staging_dir=staged, dry_run=True)

    reloaded = FusionData.from_pretrained(staged, cache_dir=tmp_path / "fresh_cache",
                                          protein_space="combined")
    assert reloaded.encoder.protein_space == "combined"


def test_from_pretrained_falls_back_to_per_block_for_a_manifest_without_the_key(
        fake_cache, tmp_path):
    """A manifest staged before this fix (or by the maintainer CLI, which never commits to a
    space) has no 'protein_space' key at all; from_pretrained must not choke on it."""
    staged = tmp_path / "staged"
    pfc.stage(fake_cache, staged)  # the CLI path: protein_space stays None in the manifest

    data = FusionData.from_pretrained(staged, cache_dir=tmp_path / "fresh_cache")

    assert data.encoder.protein_space == "per_block"


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
    """Checking dims["poi"] alone would pass even if force_download silently failed to
    overwrite: the tampered file in this test has the same width (51) as the real one, by
    construction. Assert the installed bytes actually match the manifest's hash instead."""
    staged = tmp_path / "staged"
    pfc.stage(fake_cache, staged)
    target = tmp_path / "fresh_cache"
    FusionData.from_pretrained(staged, cache_dir=target)
    rng = np.random.default_rng(0)
    np.savez(target / POI_FILE, **{s: rng.normal(size=51).astype(np.float32) for s in "xy"})
    FusionData.from_pretrained(staged, cache_dir=target, force_download=True)
    manifest = json.loads((staged / "manifest.json").read_text())
    assert pfc.sha256_of(target / POI_FILE) == manifest["files"][POI_FILE]["sha256"]


def test_from_pretrained_conflict_leaves_the_cache_untouched(fake_cache, tmp_path):
    """A hash conflict on one file must not have already copied the others in before raising
    — otherwise the cache can end up holding a Hub table next to a stale local PCA side file,
    silently mixing two PCA fits instead of refusing to. POI_FILE sorts after CELL_FILE and
    ASSAY_FILE alphabetically, so a single-pass copy-as-you-go loop would already have copied
    them before reaching the conflict."""
    staged = tmp_path / "staged"
    pfc.stage(fake_cache, staged)
    target = tmp_path / "fresh_cache"
    target.mkdir()
    rng = np.random.default_rng(0)
    np.savez(target / POI_FILE, **{s: rng.normal(size=51).astype(np.float32) for s in "xy"})
    with pytest.raises(ValueError, match=POI_FILE):
        FusionData.from_pretrained(staged, cache_dir=target)
    assert not (target / CELL_FILE).exists()
    assert not (target / ASSAY_FILE).exists()


def test_from_pretrained_rejects_a_corrupt_source_file(fake_cache, tmp_path):
    staged = tmp_path / "staged"
    pfc.stage(fake_cache, staged)
    (staged / POI_FILE).write_bytes(b"not a valid npz")
    with pytest.raises(ValueError, match="corrupt"):
        FusionData.from_pretrained(staged, cache_dir=tmp_path / "fresh_cache")


def test_from_pretrained_rejects_a_manifest_missing_a_required_file(fake_cache, tmp_path):
    staged = tmp_path / "staged"
    pfc.stage(fake_cache, staged)
    manifest_path = staged / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["files"][POI_FILE]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="version"):
        FusionData.from_pretrained(staged, cache_dir=tmp_path / "fresh_cache")


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


# ---------------------------------------------------------------------- push_to_hub


class _FakeHubApi:
    """Records create_repo/upload_folder calls instead of touching the network."""

    def __init__(self, calls):
        self.calls = calls

    def whoami(self):
        return {"name": "test-user"}

    def create_repo(self, repo_id, repo_type, exist_ok, private):
        self.calls["create_repo"] = (repo_id, repo_type, exist_ok, private)

    def upload_folder(self, repo_id, repo_type, folder_path, commit_message):
        self.calls["upload_folder"] = (repo_id, repo_type, folder_path, commit_message)
        return type("Commit", (), {"oid": "abc123"})()


def test_push_to_hub_stages_to_a_given_directory_and_uploads(fake_cache, tmp_path, monkeypatch):
    data = FusionData(encoder=ContextEncoder(cache_dir=fake_cache))
    calls = {}
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: _FakeHubApi(calls))
    staged = tmp_path / "staged"

    sha = data.push_to_hub("ailab-bio/TACK-fusion-context", commit_message="test commit",
                           staging_dir=staged)

    assert sha == "abc123"
    for name in _all_context_filenames():
        assert (staged / name).exists()
    assert (staged / "manifest.json").exists()
    assert "ailab-bio/TACK-fusion-context" in (staged / "README.md").read_text()
    assert calls["create_repo"] == ("ailab-bio/TACK-fusion-context", "dataset", True, False)
    assert calls["upload_folder"][2] == str(staged)
    assert calls["upload_folder"][3] == "test commit"


def test_push_to_hub_without_staging_dir_uses_and_cleans_up_a_temp_directory(
        fake_cache, monkeypatch):
    data = FusionData(encoder=ContextEncoder(cache_dir=fake_cache))
    calls = {}
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: _FakeHubApi(calls))

    data.push_to_hub("ailab-bio/TACK-fusion-context")

    staged_path = Path(calls["upload_folder"][2])
    assert not staged_path.exists(), "the temporary staging directory must not survive the call"


def test_push_to_hub_dry_run_stages_without_uploading(fake_cache, tmp_path, monkeypatch):
    data = FusionData(encoder=ContextEncoder(cache_dir=fake_cache))
    calls = {}
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: _FakeHubApi(calls))
    staged = tmp_path / "staged"

    result = data.push_to_hub("ailab-bio/TACK-fusion-context", staging_dir=staged, dry_run=True)

    assert result is None
    assert calls == {}
    for name in _all_context_filenames():
        assert (staged / name).exists()


def test_push_to_hub_dry_run_requires_a_staging_dir(fake_cache):
    data = FusionData(encoder=ContextEncoder(cache_dir=fake_cache))
    with pytest.raises(ValueError, match="staging_dir"):
        data.push_to_hub("ailab-bio/TACK-fusion-context", dry_run=True)


def test_push_to_hub_refuses_an_existing_staging_dir(fake_cache, tmp_path):
    data = FusionData(encoder=ContextEncoder(cache_dir=fake_cache))
    staged = tmp_path / "staged"
    staged.mkdir()
    with pytest.raises(SystemExit):
        data.push_to_hub("ailab-bio/TACK-fusion-context", staging_dir=staged)


def test_push_to_hub_output_round_trips_through_from_pretrained(fake_cache, tmp_path,
                                                                 monkeypatch):
    data = FusionData(encoder=ContextEncoder(cache_dir=fake_cache))
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: _FakeHubApi({}))
    staged = tmp_path / "staged"

    data.push_to_hub("ailab-bio/TACK-fusion-context", staging_dir=staged)
    reloaded = FusionData.from_pretrained(staged, cache_dir=tmp_path / "fresh_cache")

    assert reloaded.dims == data.dims
