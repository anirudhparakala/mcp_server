"""Tests for the FTS5-backed lexical index (index/bm25_store.py).

Hermetic: every test builds a tiny CKB under safe_tmp_path. Never depends on
ckb/ckb.sqlite existing.
"""

import sqlite3

import pytest

from kbmcp.index import bm25_store as bs


# The 6 realistic queries measured against the live corpus. Three of them
# (indices 0, 2, 4) raise sqlite3.OperationalError if passed to MATCH raw.
HAZARD_QUERIES = [
    "What does Article 22(1) say?",
    'liquidated damages "penalty" clause',
    "GDPR OR NOT AND",
    "recovery of damages*",
    "protein intake -- how much?",
    "§ 2202 statute of frauds",
]


def test_to_match_query_quotes_every_token():
    assert bs.to_match_query("shall not") == '"shall" OR "not"'


def test_to_match_query_strips_fts5_operators():
    # OR/NOT/AND become quoted literals, not operators
    assert bs.to_match_query("GDPR OR NOT AND") == '"GDPR" OR "OR" OR "NOT" OR "AND"'


def test_to_match_query_keeps_digits_and_splits_punctuation():
    assert bs.to_match_query("Article 22(1)") == '"Article" OR "22" OR "1"'


def test_to_match_query_empty_for_punctuation_only():
    assert bs.to_match_query("--- ??? ***") == ""
    assert bs.to_match_query("") == ""


def test_to_match_query_never_emits_a_double_quote_inside_a_token():
    # Injection safety: \w+ cannot match a quote, so quoting is safe by construction
    for q in HAZARD_QUERIES:
        for token in bs.to_match_query(q).split(" OR "):
            if token:
                assert token.startswith('"') and token.endswith('"')
                assert '"' not in token[1:-1]


from kbmcp.config import load_corpus_config
from kbmcp.db import ops
from kbmcp.db.schema import create_all_tables


def test_create_all_tables_creates_bm25_meta(safe_tmp_path):
    conn = ops.get_db(safe_tmp_path / "t.sqlite")
    try:
        create_all_tables(conn)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(bm25_meta)")}
        assert cols == {"id", "chunk_count", "chunks_digest", "tokenize",
                        "weights_json", "schema_version", "built_at"}
    finally:
        conn.close()


def test_bm25_meta_rejects_a_second_row(safe_tmp_path):
    """Both halves of single-row enforcement: a different id is out of domain,
    and a duplicate id=1 collides with the primary key. The second case is the
    one that matters -- it is why build() must clear the row before inserting."""
    conn = ops.get_db(safe_tmp_path / "t.sqlite")
    try:
        create_all_tables(conn)
        ins = ("INSERT INTO bm25_meta (id, chunk_count, chunks_digest, tokenize, "
               "weights_json, schema_version, built_at) VALUES (?,1,'d','t','{}',1,'now')")
        conn.execute(ins, (1,))
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(ins, (2,))   # CHECK (id = 1) rejects a different id
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(ins, (1,))   # PRIMARY KEY rejects a duplicate row
        assert conn.execute("SELECT COUNT(*) FROM bm25_meta").fetchone()[0] == 1
    finally:
        conn.close()


def test_config_exposes_bm25_block():
    cfg = load_corpus_config("config/corpus_config.yaml")
    assert cfg.bm25["tokenize"] == "porter unicode61"
    assert cfg.bm25["weights"] == {"context": 1.0, "text": 2.0}
    assert cfg.bm25["top_k"] == 50
    assert "k1" not in cfg.bm25 and "b" not in cfg.bm25  # fixed by FTS5, not tunable


def _ckb(path, rows):
    """rows: [(chunk_id, context, text)] -> an open connection to a tiny CKB."""
    conn = ops.get_db(path)
    create_all_tables(conn)
    ops.insert_source(conn, canonical_url="u", url_original="u", domain="d",
                      format="html", license="x", license_ok=True, version="v1")
    ops.insert_doc(conn, doc_id="doc1", canonical_url="u", domain="d", format="html")
    for i, (cid, ctx, txt) in enumerate(rows):
        ops.insert_chunk(conn, chunk_id=cid, doc_id="doc1", chunk_index=i,
                         chunk_type="text", text=txt, context=ctx)
    return conn


CFG = {"tokenize": "porter unicode61",
       "weights": {"context": 1.0, "text": 2.0}, "top_k": 50}


def test_build_indexes_every_chunk(safe_tmp_path):
    conn = _ckb(safe_tmp_path / "t.sqlite",
                [("c0", "ctx zero", "text zero"), ("c1", "ctx one", "text one")])
    try:
        assert bs.BM25Store(conn, CFG).build() == 2
        assert conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0] == 2
    finally:
        conn.close()


def test_build_writes_fingerprint(safe_tmp_path):
    conn = _ckb(safe_tmp_path / "t.sqlite", [("c0", "ctx", "text")])
    try:
        bs.BM25Store(conn, CFG).build()
        row = conn.execute("SELECT * FROM bm25_meta WHERE id = 1").fetchone()
        assert row["chunk_count"] == 1
        assert len(row["chunks_digest"]) == 64
        assert row["tokenize"] == "porter unicode61"
        assert row["schema_version"] == bs.SCHEMA_VERSION
    finally:
        conn.close()


def test_build_is_deterministic(safe_tmp_path):
    rows = [("c0", "ctx zero", "text zero"), ("c1", "ctx one", "text one")]
    digests = []
    for name in ("a.sqlite", "b.sqlite"):
        conn = _ckb(safe_tmp_path / name, rows)
        try:
            bs.BM25Store(conn, CFG).build()
            digests.append(conn.execute(
                "SELECT chunks_digest FROM bm25_meta").fetchone()[0])
        finally:
            conn.close()
    assert digests[0] == digests[1]


def test_digest_changes_when_text_changes_but_ids_do_not(safe_tmp_path):
    """chunk_id does not depend on text -- M4 added context to every existing
    chunk_id without changing any of them. The digest must notice."""
    conn = _ckb(safe_tmp_path / "t.sqlite", [("c0", "", "original")])
    try:
        bs.BM25Store(conn, CFG).build()
        before = conn.execute("SELECT chunks_digest FROM bm25_meta").fetchone()[0]
        conn.execute("UPDATE chunks SET context = 'added later' WHERE chunk_id = 'c0'")
        conn.commit()
        bs.BM25Store(conn, CFG).build()
        after = conn.execute("SELECT chunks_digest FROM bm25_meta").fetchone()[0]
        assert before != after
    finally:
        conn.close()


def test_rebuild_is_idempotent(safe_tmp_path):
    conn = _ckb(safe_tmp_path / "t.sqlite", [("c0", "ctx", "text")])
    try:
        store = bs.BM25Store(conn, CFG)
        store.build()
        store.build()
        assert conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM bm25_meta").fetchone()[0] == 1
    finally:
        conn.close()


def test_build_rejects_an_unsafe_tokenize_value(safe_tmp_path):
    conn = _ckb(safe_tmp_path / "t.sqlite", [("c0", "ctx", "text")])
    try:
        bad = dict(CFG, tokenize="unicode61'); DROP TABLE chunks; --")
        with pytest.raises(bs.BM25BuildError):
            bs.BM25Store(conn, bad).build()
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 1
    finally:
        conn.close()
