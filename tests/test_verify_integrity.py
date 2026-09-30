"""Tests for the corpus-integrity gate (ingest/verify_integrity.py).

Hermetic: tiny CKBs under safe_tmp_path. Never touches ckb/ckb.sqlite except
the one end-to-end test added in Task 5.
"""

import pytest

from kbmcp.db import ops
from kbmcp.db.schema import create_all_tables
from kbmcp.ingest import verify_integrity as vi
from kbmcp.models.ids import doc_id as mk_doc_id


def _ckb(path, docs):
    """docs: {slug: text}. Each slug's url IS the slug, matching _manifest()."""
    if path.exists():
        path.unlink()
    conn = ops.get_db(path)
    try:
        create_all_tables(conn)
        for slug, text in docs.items():
            did = mk_doc_id(slug, "v1")
            ops.insert_source(conn, canonical_url=slug, url_original=slug, domain="d",
                              format="html", license="x", license_ok=True, version="v1")
            ops.insert_doc(conn, doc_id=did, canonical_url=slug, domain="d", format="html")
            for i, chunk_text in enumerate(text if isinstance(text, list) else [text]):
                ops.insert_chunk(conn, chunk_id=f"{slug}-c{i}", doc_id=did, chunk_index=i,
                                 chunk_type="text", text=chunk_text)
    finally:
        conn.close()


def _yaml_dq_escape(t):
    """Escape a term for embedding in a YAML double-quoted flow scalar.

    A raw embedded newline byte in a double-quoted scalar gets FOLDED to a
    space by the YAML spec (PyYAML included) -- it is not preserved. The
    two-character escape sequence backslash-n IS preserved through a
    double-quoted scalar, same as JSON, so terms containing "\\n" (e.g. one
    spanning the chunk-join boundary) must be escaped this way to round-trip.
    """
    return t.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _manifest(path, entries):
    """entries: {slug: [terms]}. url == slug so doc_id matches _ckb()."""
    rows = ""
    for slug, terms in entries.items():
        rows += (f"- doc_id: {slug}\n  url: {slug}\n  domain: d\n  format: html\n"
                 f"  license: x\n  license_ok: true\n  version: v1\n")
        if terms:
            listed = ", ".join(f'"{_yaml_dq_escape(t)}"' for t in terms)
            rows += f"  expect_terms: [{listed}]\n"
    path.write_text(rows, encoding="utf-8")


def _raw(dirpath, slugs):
    dirpath.mkdir(exist_ok=True)
    for s in slugs:
        (dirpath / f"{s}.meta.json").write_text('{"resolved_version": "v1"}',
                                                encoding="utf-8")


def _run(tmp, docs, entries, **kw):
    ckb = tmp / "t.sqlite"
    man = tmp / "m.yaml"
    raw = tmp / "raw"
    _ckb(ckb, docs)
    _manifest(man, entries)
    _raw(raw, entries)
    kw.setdefault("queries_path", tmp / "absent-queries.jsonl")
    kw.setdefault("fixtures_path", tmp / "absent-fixtures.yaml")
    return vi.verify_integrity(ckb, man, raw, **kw)


def test_a_present_term_passes(safe_tmp_path):
    res = _run(safe_tmp_path, {"a": "Bowen LJ delivered the judgment"},
               {"a": ["Bowen"]}, coverage=False)
    assert vi.all_passed(res)
    assert [r.check for r in res] == ["expect_terms"]


def test_an_absent_term_fails_and_names_the_term(safe_tmp_path):
    res = _run(safe_tmp_path, {"a": "page navigation and a revision history line"},
               {"a": ["smoke ball", "Bowen"]}, coverage=False)
    assert not vi.all_passed(res)
    assert "smoke ball" in res[0].detail and "Bowen" in res[0].detail


def test_matching_is_case_sensitive(safe_tmp_path):
    """Consistent with verify_structure's must_contain and verify_gold's
    anchor_present. A loud casing failure beats a quiet pass on unread text."""
    res = _run(safe_tmp_path, {"a": "the Smoke Ball advertisement"},
               {"a": ["smoke ball"]}, coverage=False)
    assert not vi.all_passed(res)


def test_doubled_whitespace_in_a_pdf_parse_fails_a_single_spaced_term(safe_tmp_path):
    """Review Focus 4: Docling emits doubled spaces in PDF text
    ("application  not  declared" in law-cpra-amendment), so a multi-word term
    typed with single spaces fails on text that visibly contains it. The gate
    does NOT normalize -- it reports the term so the cause is visible, and
    authoring resolves terms against the live CKB first."""
    res = _run(safe_tmp_path, {"a": "sensitive  personal  information"},
               {"a": ["sensitive personal information"]}, coverage=False)
    assert not vi.all_passed(res)
    assert "sensitive personal information" in res[0].detail


def test_a_term_spanning_the_chunk_join_matches_on_the_newline(safe_tmp_path):
    """Chunks are joined with "\\n", byte-for-byte as verify_structure joins."""
    res = _run(safe_tmp_path, {"a": ["ends here", "starts there"]},
               {"a": ["ends here\nstarts there"]}, coverage=False)
    assert vi.all_passed(res)


def test_a_term_only_in_context_fails(safe_tmp_path):
    """M4 contexts are model output; a hallucinated summary must not satisfy a
    gate about the source document (spec Sec.6)."""
    ckb = safe_tmp_path / "t.sqlite"
    _ckb(ckb, {"a": "unrelated body text"})
    conn = ops.get_db(ckb)
    try:
        conn.execute("UPDATE chunks SET context = ? WHERE doc_id = ?",
                     ("this chunk discusses Bowen LJ", mk_doc_id("a", "v1")))
        conn.commit()
    finally:
        conn.close()
    man = safe_tmp_path / "m.yaml"
    _manifest(man, {"a": ["Bowen"]})
    raw = safe_tmp_path / "raw"
    _raw(raw, ["a"])
    res = vi.verify_integrity(ckb, man, raw, coverage=False,
                              queries_path=safe_tmp_path / "nq.jsonl",
                              fixtures_path=safe_tmp_path / "nf.yaml")
    assert not vi.all_passed(res)


def test_a_source_with_no_expect_terms_produces_no_verification_result(safe_tmp_path):
    res = _run(safe_tmp_path, {"a": "anything"}, {"a": []}, coverage=False)
    assert res == []


def test_an_unbuilt_document_is_reported_as_unbuilt(safe_tmp_path):
    """Review Focus 1: no pin record means doc_id_for_slug returns None and the
    blob is empty. Blaming 'missing text' would send the reader hunting for a
    parse bug instead of a missing fetch."""
    ckb = safe_tmp_path / "t.sqlite"
    _ckb(ckb, {})
    man = safe_tmp_path / "m.yaml"
    _manifest(man, {"a": ["Bowen"]})
    raw = safe_tmp_path / "raw"
    raw.mkdir()  # deliberately empty -- no a.meta.json
    res = vi.verify_integrity(ckb, man, raw, coverage=False,
                              queries_path=safe_tmp_path / "nq.jsonl",
                              fixtures_path=safe_tmp_path / "nf.yaml")
    assert not vi.all_passed(res)
    assert "not built" in res[0].detail


def test_two_slugs_sharing_a_url_do_not_borrow_each_others_content(safe_tmp_path):
    """Review Focus 5: identical url+version means one doc_id, so slug b's term
    could pass on slug a's text. Flag the collision rather than score it.

    Ordering: the term-less slug ("a") is listed first, the terms-declaring
    slug ("b") second. This is the ordering the ORIGINAL (order-dependent)
    implementation happened to catch -- see the reversed-order test below for
    the ordering that hid the bug."""
    ckb = safe_tmp_path / "t.sqlite"
    _ckb(ckb, {"a": "Bowen LJ"})
    man = safe_tmp_path / "m.yaml"
    man.write_text(
        "- doc_id: a\n  url: same\n  domain: d\n  format: html\n  license: x\n"
        "  license_ok: true\n  version: v1\n"
        "- doc_id: b\n  url: same\n  domain: d\n  format: html\n  license: x\n"
        "  license_ok: true\n  version: v1\n  expect_terms: [\"Bowen\"]\n",
        encoding="utf-8")
    raw = safe_tmp_path / "raw"
    _raw(raw, ["a", "b"])
    res = vi.verify_integrity(ckb, man, raw, coverage=False,
                              queries_path=safe_tmp_path / "nq.jsonl",
                              fixtures_path=safe_tmp_path / "nf.yaml")
    assert not vi.all_passed(res)
    assert "shares a doc_id" in res[0].detail


def test_collision_is_reported_when_the_declaring_slug_is_listed_first(safe_tmp_path):
    """Fix round 1 (Important finding): the original check was order-dependent.
    It recorded the first-seen slug for a doc_id without flagging it (nothing
    to compare against yet), and only checked for a collision inside
    `if e.expect_terms`. So when the TERMS-DECLARING slug ("a") came first, it
    saw no prior record and was scored normally against the shared document's
    real text (which genuinely contains "Bowen", so it passed); when the
    term-less slug ("b") came second, it was skipped entirely because it
    declared no terms. Old behaviour: a single passing expect_terms result,
    all_passed True, collision never reported. Both slugs must now be flagged."""
    ckb = safe_tmp_path / "t.sqlite"
    _ckb(ckb, {"shared-url": "Bowen LJ delivered the judgment"})
    man = safe_tmp_path / "m.yaml"
    man.write_text(
        "- doc_id: a\n  url: shared-url\n  domain: d\n  format: html\n  license: x\n"
        "  license_ok: true\n  version: v1\n  expect_terms: [\"Bowen\"]\n"
        "- doc_id: b\n  url: shared-url\n  domain: d\n  format: html\n  license: x\n"
        "  license_ok: true\n  version: v1\n",
        encoding="utf-8")
    raw = safe_tmp_path / "raw"
    _raw(raw, ["a", "b"])
    res = vi.verify_integrity(ckb, man, raw, coverage=False,
                              queries_path=safe_tmp_path / "nq.jsonl",
                              fixtures_path=safe_tmp_path / "nf.yaml")
    assert not vi.all_passed(res)
    assert {r.slug for r in res} == {"a", "b"}
    assert all(r.check == "doc_id_collision" for r in res)
    assert all(not r.passed for r in res)


def test_collision_is_reported_when_neither_slug_declares_expect_terms(safe_tmp_path):
    """A collision is a manifest defect independent of expect_terms -- two
    named sources collapsing to one document must be reported even when
    neither slug declares any terms to check."""
    ckb = safe_tmp_path / "t.sqlite"
    _ckb(ckb, {"shared-url": "anything at all"})
    man = safe_tmp_path / "m.yaml"
    man.write_text(
        "- doc_id: a\n  url: shared-url\n  domain: d\n  format: html\n  license: x\n"
        "  license_ok: true\n  version: v1\n"
        "- doc_id: b\n  url: shared-url\n  domain: d\n  format: html\n  license: x\n"
        "  license_ok: true\n  version: v1\n",
        encoding="utf-8")
    raw = safe_tmp_path / "raw"
    _raw(raw, ["a", "b"])
    res = vi.verify_integrity(ckb, man, raw, coverage=False,
                              queries_path=safe_tmp_path / "nq.jsonl",
                              fixtures_path=safe_tmp_path / "nf.yaml")
    assert not vi.all_passed(res)
    assert {r.slug for r in res} == {"a", "b"}
    assert all(r.check == "doc_id_collision" for r in res)


def test_a_collision_slug_is_not_also_scored_against_the_shared_text(safe_tmp_path):
    """A slug in a collision group gets the collision result INSTEAD of an
    expect_terms result, not in addition to one -- even when it declares
    terms that are genuinely absent from the shared document's text, the
    reported check must be doc_id_collision, not expect_terms."""
    ckb = safe_tmp_path / "t.sqlite"
    _ckb(ckb, {"shared-url": "unrelated content with no matching phrase"})
    man = safe_tmp_path / "m.yaml"
    man.write_text(
        "- doc_id: a\n  url: shared-url\n  domain: d\n  format: html\n  license: x\n"
        "  license_ok: true\n  version: v1\n  expect_terms: [\"Bowen\"]\n"
        "- doc_id: b\n  url: shared-url\n  domain: d\n  format: html\n  license: x\n"
        "  license_ok: true\n  version: v1\n  expect_terms: [\"Bowen\"]\n",
        encoding="utf-8")
    raw = safe_tmp_path / "raw"
    _raw(raw, ["a", "b"])
    res = vi.verify_integrity(ckb, man, raw, coverage=False,
                              queries_path=safe_tmp_path / "nq.jsonl",
                              fixtures_path=safe_tmp_path / "nf.yaml")
    assert len(res) == 2  # one result per slug, not a collision plus a scoring
    assert all(r.check == "doc_id_collision" for r in res)
    assert not any(r.check == "expect_terms" for r in res)
