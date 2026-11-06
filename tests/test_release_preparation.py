import hashlib
import json
from pathlib import Path

import pytest

from breprecover.cli import validate_config
from tools import prepare_release
from tools.prepare_release import ROOT, prepare


def test_example_config_uses_all_required_paths(tmp_path: Path) -> None:
    config = json.loads((ROOT / "examples/config.example.json").read_text())
    artifacts = config["artifacts"]
    for relative in artifacts.values():
        artifact = tmp_path / relative
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("{}\n")
    (tmp_path / artifacts["shape_policy_metadata"]).write_text(
        json.dumps({"selected_dev_policy": {"threshold": 0.561}})
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    result = validate_config(config_path)
    assert result["valid"] is True
    assert len(result["artifacts"]) == 6


def test_archive_contains_allowlisted_text_and_manifest(tmp_path: Path) -> None:
    destination = tmp_path / "candidate"
    result = prepare(destination)
    manifest = json.loads((destination / "MANIFEST.sha256.json").read_text())
    assert result["status"] == "prepared_for_review_only"
    assert manifest["file_count"] == len(manifest["files"])
    assert manifest["license_present"] is True
    assert manifest["license_decision"] == "MIT"
    assert manifest["status"] == "MIT_selected_provenance_review_pending"
    assert manifest["public_release_ready"] is False
    for entry in manifest["files"]:
        data = (destination / entry["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]
        assert not entry["path"].startswith(("build/", "data/", "results/"))


def test_archive_refuses_existing_destination(tmp_path: Path) -> None:
    destination = tmp_path / "existing"
    destination.mkdir()
    sentinel = destination / "keep"
    sentinel.write_text("untouched")
    with pytest.raises(FileExistsError):
        prepare(destination)
    assert sentinel.read_text() == "untouched"


def test_archive_without_license_reports_license_pending(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    monkeypatch.setattr(prepare_release, "ROOT", source)
    (source / "src" / "breprecover").mkdir(parents=True)
    (source / "src" / "breprecover" / "__init__.py").write_text("\n")

    destination = tmp_path / "candidate"
    prepare(destination)
    manifest = json.loads((destination / "MANIFEST.sha256.json").read_text())
    assert manifest["license_present"] is False
    assert manifest["license_decision"] is None
    assert manifest["status"] == "rights_and_license_pending"
    assert manifest["public_release_ready"] is False


def test_archive_rejects_destination_inside_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside"):
        prepare(ROOT / "not-a-release")


def test_archive_rejects_allowlisted_symlink_and_binary(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(prepare_release, "ROOT", tmp_path / "source")
    source = prepare_release.ROOT
    (source / "src").mkdir(parents=True)
    (source / "src" / "valid.py").write_text("pass\n")
    (source / "docs").mkdir()
    (source / "docs" / "readme.md").write_text("sample\n")
    (source / "tests").mkdir()
    (source / "tests" / "test_sample.py").write_text("pass\n")
    target = tmp_path / "fixture.txt"
    target.write_text("fixture")
    link = source / "src" / "release-symlink-fixture.txt"
    output = tmp_path / "symlink-output"
    try:
        link.symlink_to(target)
        with pytest.raises(ValueError, match="symlink"):
            prepare(output)
    finally:
        link.unlink(missing_ok=True)

    binary = source / "src" / "release-binary-fixture.bin"
    try:
        binary.write_bytes(b"\x00\x01")
        with pytest.raises(ValueError, match="unexpected file type"):
            prepare(tmp_path / "binary-output")
    finally:
        binary.unlink(missing_ok=True)


def test_archive_rejects_symlinked_allowlist_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(prepare_release, "ROOT", tmp_path / "source")
    external = tmp_path / "external"
    external.mkdir()
    (external / "leak.py").write_text("pass\n")
    prepare_release.ROOT.mkdir()
    (prepare_release.ROOT / "src").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        prepare(tmp_path / "root-link-output")


def test_secret_scan_detects_fixture_without_literal_token() -> None:
    from tools.prepare_release import _looks_like_secret

    value = "".join(("gh", "p_", "a" * 30))
    assert _looks_like_secret(value.encode())
