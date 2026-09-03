"""Tests for --folder mode discovery (ingest/folder_source.py).

Hermetic: builds real files under safe_tmp_path. Never touches corpus/.
"""

import pytest
import yaml

from kbmcp.ingest import folder_source as fs
from kbmcp.ingest.manifest import load_manifest


def _write(root, rel, text="hello world"):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def test_discovers_files_recursively(safe_tmp_path):
    _write(safe_tmp_path, "a.md")
    _write(safe_tmp_path, "sub/b.pdf")
    rows = fs.discover(safe_tmp_path)
    assert {r["doc_id"] for r in rows} == {"a", "sub-b"}


def test_same_basename_in_different_dirs_does_not_collide(safe_tmp_path):
    _write(safe_tmp_path, "one/notes.md")
    _write(safe_tmp_path, "two/notes.md")
    rows = fs.discover(safe_tmp_path)
    assert len({r["doc_id"] for r in rows}) == 2


def test_skips_dotfiles_and_dot_directories(safe_tmp_path):
    _write(safe_tmp_path, "keep.md")
    _write(safe_tmp_path, ".hidden.md")
    _write(safe_tmp_path, ".git/config")
    rows = fs.discover(safe_tmp_path)
    assert [r["doc_id"] for r in rows] == ["keep"]


def test_version_is_a_content_hash_and_tracks_content(safe_tmp_path):
    _write(safe_tmp_path, "a.md", "first")
    v1 = fs.discover(safe_tmp_path)[0]["version"]
    _write(safe_tmp_path, "a.md", "second")
    v2 = fs.discover(safe_tmp_path)[0]["version"]
    assert len(v1) == 64 and v1 != v2


def test_format_is_mapped_into_valid_formats(safe_tmp_path):
    _write(safe_tmp_path, "p.pdf")
    _write(safe_tmp_path, "h.html")
    _write(safe_tmp_path, "n.md")
    by_id = {r["doc_id"]: r["format"] for r in fs.discover(safe_tmp_path)}
    assert by_id == {"p": "pdf", "h": "html", "n": "other"}


def test_license_is_autofilled_and_ok(safe_tmp_path):
    _write(safe_tmp_path, "a.md")
    row = fs.discover(safe_tmp_path)[0]
    assert row["license"] == fs.LOCAL_LICENSE
    assert row["license_ok"] is True


def test_discovery_is_deterministic(safe_tmp_path):
    for name in ("c.md", "a.md", "b.md"):
        _write(safe_tmp_path, name)
    assert fs.discover(safe_tmp_path) == fs.discover(safe_tmp_path)
    assert [r["doc_id"] for r in fs.discover(safe_tmp_path)] == ["a", "b", "c"]


def test_written_manifest_round_trips_through_load_manifest(safe_tmp_path):
    """The synthesized manifest must satisfy the real loader's validation --
    load_manifest requires license, license_ok is true, and a valid format."""
    src = safe_tmp_path / "docs"
    src.mkdir()
    _write(src, "a.md")
    _write(src, "sub/b.pdf")
    out = fs.write_manifest(fs.discover(src), safe_tmp_path / "folder-manifest.yaml")
    entries = load_manifest(out)
    assert len(entries) == 2
    assert all(e.license_ok for e in entries)


def test_empty_folder_raises_rather_than_building_nothing(safe_tmp_path):
    with pytest.raises(fs.FolderSourceError):
        fs.discover(safe_tmp_path)


def test_punctuation_only_difference_collides_and_raises(safe_tmp_path):
    """a-b.md and a_b.md both slug to 'a-b'. Without the check, one file would
    silently vanish from the CKB."""
    _write(safe_tmp_path, "a-b.md")
    _write(safe_tmp_path, "a_b.md")
    with pytest.raises(fs.FolderSourceError) as exc:
        fs.discover(safe_tmp_path)
    assert "a-b" in str(exc.value)          # names the colliding slug
    assert "a-b.md" in str(exc.value) or "a_b.md" in str(exc.value)   # and a real file


def test_slugs_that_reduce_to_empty_collide_and_raise(safe_tmp_path):
    """Filenames of pure punctuation both fall back to 'source'."""
    _write(safe_tmp_path, "___.md")
    _write(safe_tmp_path, "!!!.md")
    with pytest.raises(fs.FolderSourceError):
        fs.discover(safe_tmp_path)


def test_a_single_punctuation_only_filename_is_still_usable(safe_tmp_path):
    """One such file must NOT raise -- the fallback slug is legitimate on its own."""
    _write(safe_tmp_path, "___.md")
    rows = fs.discover(safe_tmp_path)
    assert [r["doc_id"] for r in rows] == ["source"]
