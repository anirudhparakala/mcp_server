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
    """Insertion order differs between the two CKBs, so this passes only because
    corpus_digest sorts by chunk_id -- delete that ORDER BY and it fails."""
    rows = [("c0", "ctx zero", "text zero"), ("c1", "ctx one", "text one")]
    digests = []
    for name, ordered in (("a.sqlite", rows), ("b.sqlite", list(reversed(rows)))):
        conn = _ckb(safe_tmp_path / name, ordered)
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
        bs.BM25Store(conn, CFG).build()          # a good index exists first
        bad = dict(CFG, tokenize="unicode61'); DROP TABLE chunks; --")
        with pytest.raises(bs.BM25BuildError):
            bs.BM25Store(conn, bad).build()
        # validation runs BEFORE the DROP, so the existing index survives intact
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0] == 1
    finally:
        conn.close()


def test_build_raises_actionable_error_when_bm25_meta_is_missing(safe_tmp_path):
    """Simulates a CKB built before bm25_meta existed in the schema -- true of
    every CKB already in the wild, including the shipped release asset. build()
    deliberately owns only chunks_fts (not the content tables), so a missing
    bm25_meta must fail with an actionable BM25BuildError telling the caller to
    run create_all_tables, not a raw sqlite3.OperationalError."""
    conn = _ckb(safe_tmp_path / "t.sqlite", [("c0", "ctx", "text")])
    try:
        conn.execute("DROP TABLE bm25_meta")
        conn.commit()
        with pytest.raises(bs.BM25BuildError, match="create_all_tables"):
            bs.BM25Store(conn, CFG).build()
    finally:
        conn.close()


# BM25 IDF collapses on a corpus of 1-3 documents: with a term in most of them,
# FTS5 clamps the score to ~1.7e-06. Ranking order still holds, but the tests
# would be asserting a clamping artifact instead of ranking behaviour. Padding
# the corpus with non-matching chunks restores realistic IDF (scores 0.9-2.7).
# Measured 2026-08-31; ordering is identical padded or not.
PAD = [(f"pad{i}", "", "unrelated filler material about barbells") for i in range(8)]


def _built(safe_tmp_path, rows, cfg=None):
    conn = _ckb(safe_tmp_path / "t.sqlite", rows)
    bs.BM25Store(conn, cfg or CFG).build()
    return conn, bs.BM25Store(conn, cfg or CFG)


def test_query_returns_positive_descending_scores(safe_tmp_path):
    conn, store = _built(safe_tmp_path, [
        ("c0", "", "protein synthesis protein protein"),
        ("c1", "", "protein synthesis"),
        ("c2", "", "unrelated material"),
    ] + PAD)
    try:
        out = store.query("protein")
        assert [cid for cid, _ in out] == ["c0", "c1"]
        assert all(s > 0 for _, s in out)
        assert out[0][1] >= out[1][1]
    finally:
        conn.close()


def test_context_only_term_retrieves_the_chunk(safe_tmp_path):
    """The whole point of M4: a term that appears ONLY in the generated context
    must still retrieve its chunk."""
    conn, store = _built(safe_tmp_path, [
        ("c0", "This section of the GDPR concerns automated decision-making.",
         "The data subject shall have the right not to be subject to a decision."),
        ("c1", "", "Unrelated text about barbell training."),
    ] + PAD)
    try:
        assert [cid for cid, _ in store.query("GDPR")] == ["c0"]
    finally:
        conn.close()


def test_text_hit_outranks_context_only_hit(safe_tmp_path):
    conn, store = _built(safe_tmp_path, [
        ("ctx_only", "hypertrophy", "filler filler filler"),
        ("in_text", "filler", "hypertrophy filler filler"),
    ] + PAD)
    try:
        assert [cid for cid, _ in store.query("hypertrophy")][0] == "in_text"
    finally:
        conn.close()


def test_stemming_matches_singular_and_plural(safe_tmp_path):
    """The reference's hand-rolled stemmer failed 7 of 9 legal pairs; porter
    must collide damage/damages."""
    conn, store = _built(safe_tmp_path, [("c0", "", "liquidated damage clause")] + PAD)
    try:
        assert [cid for cid, _ in store.query("damages")] == ["c0"]
    finally:
        conn.close()


def test_negation_is_indexed_not_discarded(safe_tmp_path):
    """The reference dropped 'not' as a stopword, making these two identical."""
    conn, store = _built(safe_tmp_path, [
        ("neg", "", "the principal shall not be subject to transfer"),
        ("pos", "", "the principal shall be subject to transfer"),
    ] + PAD)
    try:
        assert [cid for cid, _ in store.query("not")] == ["neg"]
    finally:
        conn.close()


def test_query_honours_top_k(safe_tmp_path):
    conn, store = _built(safe_tmp_path, [(f"c{i}", "", "protein") for i in range(5)])
    try:
        assert len(store.query("protein", top_k=2)) == 2
    finally:
        conn.close()


def test_hazard_queries_do_not_raise(safe_tmp_path):
    conn, store = _built(safe_tmp_path, [("c0", "", "Article 22 damages protein")])
    try:
        for q in HAZARD_QUERIES:
            store.query(q)   # must not raise
    finally:
        conn.close()


def test_empty_and_punctuation_only_queries_return_empty(safe_tmp_path):
    conn, store = _built(safe_tmp_path, [("c0", "", "anything")])
    try:
        assert store.query("") == []
        assert store.query("--- ???") == []
    finally:
        conn.close()


def test_query_before_build_raises(safe_tmp_path):
    conn = _ckb(safe_tmp_path / "t.sqlite", [("c0", "", "text")])
    try:
        with pytest.raises(bs.BM25NotBuiltError):
            bs.BM25Store(conn, CFG).query("text")
    finally:
        conn.close()


def test_query_raises_when_index_is_present_but_incomplete(safe_tmp_path):
    """A build that dies after the DDL but before the fingerprint leaves
    chunks_fts present but empty. That must raise, not return zero hits."""
    conn, store = _built(safe_tmp_path, [("c0", "", "protein")] + PAD)
    try:
        conn.execute("DELETE FROM bm25_meta")
        conn.commit()
        with pytest.raises(bs.BM25NotBuiltError):
            store.query("protein")
    finally:
        conn.close()


def test_is_stale_false_right_after_build(safe_tmp_path):
    conn, store = _built(safe_tmp_path, [("c0", "ctx", "text")])
    try:
        assert store.is_stale() is False
    finally:
        conn.close()


def test_is_stale_true_before_any_build(safe_tmp_path):
    conn = _ckb(safe_tmp_path / "t.sqlite", [("c0", "", "text")])
    try:
        assert bs.BM25Store(conn, CFG).is_stale() is True
    finally:
        conn.close()


def test_is_stale_true_after_a_chunk_is_added(safe_tmp_path):
    conn, store = _built(safe_tmp_path, [("c0", "", "text")])
    try:
        ops.insert_chunk(conn, chunk_id="c1", doc_id="doc1", chunk_index=1,
                         chunk_type="text", text="new")
        assert store.is_stale() is True
    finally:
        conn.close()


def test_is_stale_true_after_context_is_added(safe_tmp_path):
    conn, store = _built(safe_tmp_path, [("c0", "", "text")])
    try:
        conn.execute("UPDATE chunks SET context = 'later' WHERE chunk_id = 'c0'")
        conn.commit()
        assert store.is_stale() is True
    finally:
        conn.close()


def test_is_stale_true_when_config_changes(safe_tmp_path):
    conn, _ = _built(safe_tmp_path, [("c0", "", "text")])
    try:
        reweighted = bs.BM25Store(conn, dict(CFG, weights={"context": 5.0, "text": 1.0}))
        assert reweighted.is_stale() is True
    finally:
        conn.close()


def test_cli_builds_the_index(safe_tmp_path):
    conn = _ckb(safe_tmp_path / "t.sqlite", [("c0", "", "text")])
    conn.close()
    assert bs.main(["--ckb", str(safe_tmp_path / "t.sqlite"),
                    "--config", "config/corpus_config.yaml"]) == 0
    conn = ops.get_db(safe_tmp_path / "t.sqlite")
    try:
        assert conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0] == 1
    finally:
        conn.close()
