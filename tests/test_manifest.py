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


def _write(tmp_path, text):
    p = tmp_path / "manifest.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_load_valid_manifest(tmp_path):
    entries = load_manifest(_write(tmp_path, VALID))
    assert len(entries) == 1
    e = entries[0]
    assert e.doc_id == "ucc-article-2"
    assert e.domain == "law_contract"
    assert e.format == "html"
    assert e.references == ["ucc-article-1"]
    assert e.collision_terms == ["performance", "breach"]


def test_byo_minimal_entry_without_domain_or_tiers(tmp_path):
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
    entries = load_manifest(_write(tmp_path, byo))
    assert len(entries) == 1
    assert entries[0].domain == "unspecified"
    assert entries[0].tier_roles == []
    assert entries[0].format == "pdf"


def test_free_form_domain_is_accepted(tmp_path):
    ok = VALID.replace("law_contract", "my_custom_domain")
    entries = load_manifest(_write(tmp_path, ok))
    assert entries[0].domain == "my_custom_domain"


def test_invalid_format_raises(tmp_path):
    bad = VALID.replace("format: html", "format: docx")
    with pytest.raises(ManifestError):
        load_manifest(_write(tmp_path, bad))


def test_license_not_ok_raises(tmp_path):
    bad = VALID.replace("license_ok: true", "license_ok: false")
    with pytest.raises(ManifestError):
        load_manifest(_write(tmp_path, bad))


def test_missing_required_field_raises(tmp_path):
    bad = VALID.replace('  version: "2026-oldid-001"\n', "")
    with pytest.raises(ManifestError):
        load_manifest(_write(tmp_path, bad))


def test_duplicate_slug_raises(tmp_path):
    with pytest.raises(ManifestError):
        load_manifest(_write(tmp_path, VALID + VALID.split("\n", 1)[1]))
