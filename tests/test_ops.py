from kbmcp.db.schema import create_all_tables
from kbmcp.db import ops


def _fresh_db():
    conn = ops.get_db(":memory:")
    create_all_tables(conn)
    return conn


def _seed_source_and_doc(conn):
    ops.insert_source(
        conn,
        canonical_url="https://example.com/x",
        url_original="https://example.com/x",
        domain="ai",
        format="html",
        license="CC-BY",
        license_ok=True,
        version="v1",
        tier_roles=["T1"],
    )
    ops.insert_doc(
        conn,
        doc_id="doc-sha",
        canonical_url="https://example.com/x",
        domain="ai",
        format="html",
        title="Example",
    )


def test_insert_and_get_chunk_roundtrip():
    conn = _fresh_db()
    _seed_source_and_doc(conn)
    ops.insert_chunk(
        conn,
        chunk_id="chunk-sha",
        doc_id="doc-sha",
        chunk_index=0,
        text="hello world",
        chunk_type="paragraph",
        heading_path=["Intro"],
    )
    got = ops.get_chunk(conn, "chunk-sha")
    assert got is not None
    assert got["text"] == "hello world"
    assert got["chunk_index"] == 0
    assert isinstance(got["heading_path_json"], list)
    assert got["heading_path_json"] == ["Intro"]


def test_chunk_id_exists_true_and_false():
    conn = _fresh_db()
    _seed_source_and_doc(conn)
    ops.insert_chunk(
        conn, chunk_id="c1", doc_id="doc-sha", chunk_index=0, text="t", chunk_type="paragraph"
    )
    assert ops.chunk_id_exists(conn, "c1") is True
    assert ops.chunk_id_exists(conn, "nope") is False


def test_foreign_key_enforced_on_chunk_without_doc():
    import sqlite3
    import pytest

    conn = _fresh_db()  # foreign_keys ON, no doc inserted
    with pytest.raises(sqlite3.IntegrityError):
        ops.insert_chunk(
            conn, chunk_id="c1", doc_id="missing", chunk_index=0, text="t", chunk_type="paragraph"
        )
