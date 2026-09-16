"""Tests for the gold-resolution helper (eval/resolve.py)."""

import pytest

from kbmcp.db import ops
from kbmcp.db.schema import create_all_tables
from kbmcp.eval import resolve as r
from kbmcp.models.ids import doc_id as mk_doc_id

DID = mk_doc_id("u", "v1")


def _ckb(path, texts):
    conn = ops.get_db(path)
    create_all_tables(conn)
    ops.insert_source(conn, canonical_url="u", url_original="u", domain="d",
                      format="html", license="x", license_ok=True, version="v1")
    ops.insert_doc(conn, doc_id=DID, canonical_url="u", domain="d", format="html")
    for i, t in enumerate(texts):
        ops.insert_chunk(conn, chunk_id=f"c{i}", doc_id=DID, chunk_index=i,
                         chunk_type="text", text=t)
    return conn


def _by_slug_and_raw(safe_tmp_path):
    man = safe_tmp_path / "m.yaml"
    man.write_text("- doc_id: u\n  url: u\n  domain: d\n  format: html\n"
                   "  license: x\n  license_ok: true\n  version: v1\n", encoding="utf-8")
    raw = safe_tmp_path / "raw"
    raw.mkdir()
    (raw / "u.meta.json").write_text('{"resolved_version": "v1"}', encoding="utf-8")
    from kbmcp.ingest.manifest import load_manifest
    return {e.doc_id: e for e in load_manifest(man)}, str(raw)


def test_finds_the_chunk_containing_a_phrase(safe_tmp_path):
    conn = _ckb(safe_tmp_path / "t.sqlite", ["alpha beta", "gamma delta"])
    try:
        hits = r.find_chunks(conn, "gamma")
        assert [h["chunk_id"] for h in hits] == ["c1"]
        assert "gamma" in hits[0]["snippet"]
    finally:
        conn.close()


def test_returns_every_match_so_ambiguity_is_visible(safe_tmp_path):
    conn = _ckb(safe_tmp_path / "t.sqlite", ["shared term here", "also shared term"])
    try:
        assert len(r.find_chunks(conn, "shared term")) == 2
    finally:
        conn.close()


def test_gold_ref_for_returns_a_usable_ref(safe_tmp_path):
    conn = _ckb(safe_tmp_path / "t.sqlite", ["alpha beta", "gamma delta"])
    by_slug, raw = _by_slug_and_raw(safe_tmp_path)
    try:
        ref = r.gold_ref_for(conn, "u", "gamma", by_slug, raw)
        assert ref.doc == "u" and ref.chunk_id == "c1" and ref.anchor == "gamma"
    finally:
        conn.close()


def test_gold_ref_for_raises_when_the_phrase_matches_nothing(safe_tmp_path):
    conn = _ckb(safe_tmp_path / "t.sqlite", ["alpha beta"])
    by_slug, raw = _by_slug_and_raw(safe_tmp_path)
    try:
        with pytest.raises(r.ResolveError):
            r.gold_ref_for(conn, "u", "absent phrase", by_slug, raw)
    finally:
        conn.close()


def test_gold_ref_for_raises_when_the_phrase_is_ambiguous(safe_tmp_path):
    """An anchor matching two chunks cannot identify one, and picking the first
    silently would bind the label to an arbitrary chunk."""
    conn = _ckb(safe_tmp_path / "t.sqlite", ["shared term here", "also shared term"])
    by_slug, raw = _by_slug_and_raw(safe_tmp_path)
    try:
        with pytest.raises(r.ResolveError) as exc:
            r.gold_ref_for(conn, "u", "shared term", by_slug, raw)
        assert "2" in str(exc.value)
    finally:
        conn.close()


def test_search_can_be_scoped_to_one_document(safe_tmp_path):
    conn = _ckb(safe_tmp_path / "t.sqlite", ["alpha", "beta"])
    by_slug, raw = _by_slug_and_raw(safe_tmp_path)
    try:
        assert len(r.find_chunks(conn, "alpha", slug="u", by_slug=by_slug,
                                 raw_dir=raw)) == 1
    finally:
        conn.close()


def test_the_ambiguity_count_is_not_capped_by_the_result_limit(safe_tmp_path):
    """find_chunks caps its list; the ambiguity message must still report the
    TRUE total, because an author uses it to judge how much to lengthen the
    anchor."""
    conn = _ckb(safe_tmp_path / "t.sqlite", [f"shared phrase item {i}" for i in range(14)])
    by_slug, raw = _by_slug_and_raw(safe_tmp_path)
    try:
        with pytest.raises(r.ResolveError) as exc:
            r.gold_ref_for(conn, "u", "shared phrase", by_slug, raw)
        assert "14" in str(exc.value), f"expected the true count, got: {exc.value}"
    finally:
        conn.close()


def test_count_chunks_agrees_with_find_chunks_when_under_the_limit(safe_tmp_path):
    """The two queries must describe the same match set, or the count would
    describe something other than the sample."""
    conn = _ckb(safe_tmp_path / "t.sqlite", ["alpha one", "alpha two", "beta"])
    try:
        assert r.count_chunks(conn, "alpha") == len(r.find_chunks(conn, "alpha")) == 2
    finally:
        conn.close()
