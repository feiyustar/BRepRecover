"""Prepare a review-only source archive from an explicit file allowlist."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import re

ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_SIZE = 2_000_000
ALLOWED_ROOTS = ("src/", "docs/", "tests/", "examples/", "envs/")
ALLOWED_FILES = {
    "README.md", "pyproject.toml", "requirements-core-test.txt",
    ".gitignore", "tools/prepare_release.py", "LICENSE", "NOTICE", "CITATION.cff",
}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".yml", ".yaml", ".toml"}
IGNORED_DIRS = {
    "__pycache__", ".pytest_cache", "build", "dist", ".git", ".egg-info",
    ".venv", "venv", "data", "datasets", "features", "feature_cache",
    "captures", "outputs", "results", "weights", "checkpoints", "runs", "wandb",
}
MIT_LICENSE_TEXT = """MIT License

Copyright (c) 2026 BRepRecover contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the \"Software\"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def _license_status(files: list[tuple[Path, str, bytes]]) -> str:
    license_bytes = next((data for _, path, data in files if path == "LICENSE"), None)
    if license_bytes is None:
        return "rights_and_license_pending"
    if license_bytes.decode("utf-8") == MIT_LICENSE_TEXT:
        return "MIT_selected_provenance_review_pending"
    return "license_present_unverified_provenance_review_pending"


def _allowed(path: str) -> bool:
    parts = Path(path).parts
    if any(part in IGNORED_DIRS or part.endswith(".egg-info") for part in parts):
        return False
    if Path(path).suffix not in TEXT_SUFFIXES and path not in ALLOWED_FILES:
        return False
    return path in ALLOWED_FILES or path.startswith(ALLOWED_ROOTS)


def _walk_allowlisted() -> list[Path]:
    files: list[Path] = []
    for root_name in ALLOWED_ROOTS:
        root = ROOT / root_name.rstrip("/")
        if root.is_symlink():
            raise ValueError(f"symlink rejected: {root.relative_to(ROOT).as_posix()}")
        if not root.exists():
            continue
        for current, dirs, names in os.walk(root, followlinks=False):
            current_path = Path(current)
            relative_dir = current_path.relative_to(ROOT).as_posix()
            dirs[:] = [name for name in dirs if name not in IGNORED_DIRS and not name.endswith(".egg-info")]
            for name in list(dirs):
                child = current_path / name
                if child.is_symlink():
                    raise ValueError(f"symlink rejected: {child.relative_to(ROOT).as_posix()}")
            for name in names:
                child = current_path / name
                relative = child.relative_to(ROOT).as_posix()
                if child.is_symlink():
                    raise ValueError(f"symlink rejected: {relative}")
                if child.suffix in {".pyc", ".pyo"}:
                    continue
                if child.suffix not in TEXT_SUFFIXES and relative not in ALLOWED_FILES:
                    raise ValueError(f"unexpected file type rejected: {relative}")
                if _allowed(relative):
                    files.append(child)
    for name in ALLOWED_FILES:
        child = ROOT / name
        for parent in child.parents:
            if parent == ROOT.parent:
                break
            if parent.is_symlink():
                raise ValueError(f"symlink rejected: {parent.relative_to(ROOT).as_posix()}")
        if child.exists() or child.is_symlink():
            if child.is_symlink():
                raise ValueError(f"symlink rejected: {name}")
            files.append(child)
    return sorted(set(files))


def _looks_like_secret(data: bytes) -> bool:
    text = data.decode("utf-8", errors="strict")
    patterns = (
        r"(?im)^\s*(?:api[_-]?key|access[_-]?token|secret[_-]?key)\s*[:=]\s*['\"]?\S{12,}",
        r"(?:ghp|github_pat)_[A-Za-z0-9_]{20,}",
        r"AKIA[A-Z0-9]{16}",
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def prepare(destination: Path) -> dict[str, object]:
    destination = destination.expanduser().resolve(strict=False)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"destination already exists: {destination}")
    if destination == ROOT or ROOT in destination.parents:
        raise ValueError("destination must be outside the source tree")
    files = _walk_allowlisted()
    checked: list[tuple[Path, str, bytes]] = []
    for source in files:
        relative = source.relative_to(ROOT).as_posix()
        if source.is_symlink():
            raise ValueError(f"symlink rejected: {relative}")
        mode = source.stat().st_mode
        if not stat.S_ISREG(mode):
            raise ValueError(f"non-regular file rejected: {relative}")
        if source.stat().st_size > MAX_FILE_SIZE:
            raise ValueError(f"oversize file rejected: {relative}")
        data = source.read_bytes()
        if b"\0" in data:
            raise ValueError(f"binary file rejected: {relative}")
        if _looks_like_secret(data):
            raise ValueError(f"credential marker rejected in: {relative}")
        checked.append((source, relative, data))
    if not checked:
        raise ValueError("allowlist produced an empty archive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    entries = []
    try:
        for source, relative, data in checked:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            entries.append({"path": relative, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        status = _license_status(checked)
        manifest = {"schema": "breprecover-source-archive.v1", "file_count": len(entries), "manifest_included_in_file_list": False, "public_release_ready": False, "license_decision": "MIT" if status == "MIT_selected_provenance_review_pending" else None, "license_present": any(path == "LICENSE" for _, path, _ in checked), "status": status, "files": entries}
        (destination / "MANIFEST.sha256.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    except Exception:
        for child in sorted(destination.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        destination.rmdir()
        raise
    return {"status": "prepared_for_review_only", "destination": str(destination), "file_count": len(entries), "manifest": str(destination / "MANIFEST.sha256.json")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="new directory outside source tree")
    args = parser.parse_args(argv)
    try:
        result = prepare(args.destination)
    except Exception as exc:
        parser.exit(2, f"prepare_release: {type(exc).__name__}: {exc}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
