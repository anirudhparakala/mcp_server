import sqlite3

from kbmcp.db.schema import create_all_tables


def test_creates_all_five_tables():
    conn = sqlite3.connect(":memory:")
    create_all_tables(conn)
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    names = {r[0] for r in rows}
    assert {"sources", "docs", "chunks", "edges", "ingest_runs"} <= names


def test_create_all_tables_is_idempotent():
    conn = sqlite3.connect(":memory:")
    create_all_tables(conn)
    create_all_tables(conn)  # must not raise on second call
    count = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='chunks'"
    ).fetchone()[0]
    assert count == 1
