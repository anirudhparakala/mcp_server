from kbmcp.db import ops
from kbmcp.db.schema import create_all_tables


def _seed(conn):
    ops.insert_source(conn, canonical_url="u", url_original="u", domain="ai", format="html",
                      license="x", license_ok=True, version="v1")
    ops.insert_doc(conn, doc_id="doc1", canonical_url="u", domain="ai", format="html", title="T")
    for i in range(3):
        ops.insert_chunk(conn, chunk_id=f"c{i}", doc_id="doc1", chunk_index=i, text=f"t{i}", chunk_type="text")


def test_get_chunks_for_doc_ordered():
    conn = ops.get_db(":memory:"); create_all_tables(conn); _seed(conn)
    rows = ops.get_chunks_for_doc(conn, "doc1")
    assert [r["chunk_index"] for r in rows] == [0, 1, 2]
    assert rows[1]["text"] == "t1"


def test_count_rows():
    conn = ops.get_db(":memory:"); create_all_tables(conn); _seed(conn)
    assert ops.count_rows(conn, "chunks") == 3
    assert ops.count_rows(conn, "docs") == 1
