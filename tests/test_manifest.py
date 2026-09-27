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


def _write(tmpdir, text):
    p = tmpdir / "manifest.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_load_valid_manifest(safe_tmp_path):
    entries = load_manifest(_write(safe_tmp_path, VALID))
    assert len(entries) == 1
    e = entries[0]
    assert e.doc_id == "ucc-article-2"
    assert e.domain == "law_contract"
    assert e.format == "html"
    assert e.references == ["ucc-article-1"]
    assert e.collision_terms == ["performance", "breach"]


def test_byo_minimal_entry_without_domain_or_tiers(safe_tmp_path):
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
    entries = load_manifest(_write(safe_tmp_path, byo))
    assert len(entries) == 1
    assert entries[0].domain == "unspecified"
    assert entries[0].tier_roles == []
    assert entries[0].format == "pdf"


def test_free_form_domain_is_accepted(safe_tmp_path):
    ok = VALID.replace("law_contract", "my_custom_domain")
    entries = load_manifest(_write(safe_tmp_path, ok))
    assert entries[0].domain == "my_custom_domain"


def test_invalid_format_raises(safe_tmp_path):
    bad = VALID.replace("format: html", "format: docx")
    with pytest.raises(ManifestError):
        load_manifest(_write(safe_tmp_path, bad))


def test_license_not_ok_raises(safe_tmp_path):
    bad = VALID.replace("license_ok: true", "license_ok: false")
    with pytest.raises(ManifestError):
        load_manifest(_write(safe_tmp_path, bad))


def test_missing_required_field_raises(safe_tmp_path):
    bad = VALID.replace('  version: "2026-oldid-001"\n', "")
    with pytest.raises(ManifestError):
        load_manifest(_write(safe_tmp_path, bad))


def test_duplicate_slug_raises(safe_tmp_path):
    with pytest.raises(ManifestError):
        load_manifest(_write(safe_tmp_path, VALID + VALID.split("\n", 1)[1]))


def _write_file(path, body):
    # Named distinctly from the module's existing `_write(tmpdir, text)` above
    # (which appends "manifest.yaml" to a directory arg): redefining `_write`
    # here would shadow that helper for every earlier test in this file, since
    # a module-level name is resolved at call time against the final
    # namespace -- the last `def` wins regardless of source order.
    path.write_text(body, encoding="utf-8")
    return path


_BASE = ("- doc_id: s1\n  url: http://x/1\n  format: html\n"
         "  license: x\n  license_ok: true\n  version: v1\n")


def test_expect_terms_defaults_to_empty(safe_tmp_path):
    p = _write_file(safe_tmp_path / "m.yaml", _BASE)
    assert load_manifest(p)[0].expect_terms == []


def test_expect_terms_round_trips_a_list(safe_tmp_path):
    p = _write_file(safe_tmp_path / "m.yaml",
               _BASE + '  expect_terms: ["smoke ball", "Bowen"]\n')
    assert load_manifest(p)[0].expect_terms == ["smoke ball", "Bowen"]


def test_expect_terms_as_a_bare_string_is_rejected(safe_tmp_path):
    # list("Bowen") silently becomes ['B','o','w','e','n'] -- five one-character
    # terms that all match almost any document. Must fail loudly instead.
    p = _write_file(safe_tmp_path / "m.yaml", _BASE + '  expect_terms: "Bowen"\n')
    with pytest.raises(ManifestError, match="must be a list"):
        load_manifest(p)


def test_expect_terms_rejects_a_blank_term(safe_tmp_path):
    # "" is a substring of every string, so a blank term is a check that can
    # never fail -- worse than no check, because it reads as coverage.
    p = _write_file(safe_tmp_path / "m.yaml", _BASE + '  expect_terms: ["ok", "   "]\n')
    with pytest.raises(ManifestError, match="non-empty"):
        load_manifest(p)


def test_expect_terms_rejects_a_non_string_term(safe_tmp_path):
    p = _write_file(safe_tmp_path / "m.yaml", _BASE + "  expect_terms: [22]\n")
    with pytest.raises(ManifestError, match="non-empty"):
        load_manifest(p)


def test_expect_terms_rejects_a_mapping(safe_tmp_path):
    # A YAML mapping is iterable over its keys, so an unguarded list(raw)
    # would silently turn {a: 1} into ["a"] -- a one-character term that
    # matches almost any document. Same hazard the bare-string guard exists
    # to prevent, so it must be rejected the same way.
    p = _write_file(safe_tmp_path / "m.yaml", _BASE + "  expect_terms: {a: 1}\n")
    with pytest.raises(ManifestError, match="must be a list"):
        load_manifest(p)


def test_expect_terms_rejects_a_non_iterable(safe_tmp_path):
    # An int is not iterable at all -- list(raw) would raise a bare TypeError
    # instead of the clean ManifestError every other malformed-manifest case
    # raises.
    p = _write_file(safe_tmp_path / "m.yaml", _BASE + "  expect_terms: 5\n")
    with pytest.raises(ManifestError, match="must be a list"):
        load_manifest(p)
