import tempfile
from pathlib import Path

import pytest

from kbmcp.ingest.manifest import load_manifest, ManifestError

VALID = """
- doc_id: ucc-article-2
  url: https://www.law.cornell.edu/ucc/2
  domain: law_contract
  format: html
  license: "Cornell LII - public/open"
  license_ok: true
  version: "2026-oldid-001"
  tier_roles: [T1, T4]
  references: [ucc-article-1]
  collision_terms: [performance, breach]
"""


# NOTE: uses tempfile.TemporaryDirectory() instead of pytest's tmp_path fixture.
# tmp_path fails on this machine: a pre-existing pytest-of-aniru bookkeeping
# dir under the Windows temp folder is ACL-locked (PermissionError: WinError 5),
# even for its owning user. tempfile.TemporaryDirectory() does not touch that
# directory and works correctly here. (Same issue and fix as tests/test_config.py.)
def _write(tmpdir, text):
    p = Path(tmpdir) / "manifest.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_load_valid_manifest():
    with tempfile.TemporaryDirectory() as tmpdir:
        entries = load_manifest(_write(tmpdir, VALID))
        assert len(entries) == 1
        e = entries[0]
        assert e.doc_id == "ucc-article-2"
        assert e.domain == "law_contract"
        assert e.format == "html"
        assert e.references == ["ucc-article-1"]
        assert e.collision_terms == ["performance", "breach"]


def test_byo_minimal_entry_without_domain_or_tiers():
    # A bring-your-own-corpus entry (e.g. auto-generated from a local folder)
    # may omit domain and tier_roles — domain defaults to "unspecified".
    byo = """
- doc_id: my-local-doc
  url: file:///home/me/docs/report.pdf
  format: pdf
  license: "user-owned"
  license_ok: true
  version: "sha256:abcd"
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        entries = load_manifest(_write(tmpdir, byo))
        assert len(entries) == 1
        assert entries[0].domain == "unspecified"
        assert entries[0].tier_roles == []
        assert entries[0].format == "pdf"


def test_free_form_domain_is_accepted():
    ok = VALID.replace("law_contract", "my_custom_domain")
    with tempfile.TemporaryDirectory() as tmpdir:
        entries = load_manifest(_write(tmpdir, ok))
        assert entries[0].domain == "my_custom_domain"


def test_invalid_format_raises():
    bad = VALID.replace("format: html", "format: docx")
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ManifestError):
            load_manifest(_write(tmpdir, bad))


def test_license_not_ok_raises():
    bad = VALID.replace("license_ok: true", "license_ok: false")
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ManifestError):
            load_manifest(_write(tmpdir, bad))


def test_missing_required_field_raises():
    bad = VALID.replace('  version: "2026-oldid-001"\n', "")
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ManifestError):
            load_manifest(_write(tmpdir, bad))


def test_duplicate_slug_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ManifestError):
            load_manifest(_write(tmpdir, VALID + VALID.split("\n", 1)[1]))
