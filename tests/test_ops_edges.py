from kbmcp.db import ops
from kbmcp.db.schema import create_all_tables


def _ckb():
    conn = ops.get_db(":memory:")
    create_all_tables(conn)
    ops.insert_source(conn, canonical_url="u", url_original="u", domain="law_aireg",
                      format="html", license="x", license_ok=True, version="v1")
    ops.insert_doc(conn, doc_id="D", canonical_url="u", domain="law_aireg", format="html")
    for i in range(3):
        ops.insert_chunk(conn, chunk_id=f"c{i}", doc_id="D", chunk_index=i,
                         text=f"t{i}", chunk_type="text")
    return conn


def test_edge_id_is_deterministic_and_field_sensitive():
    a = ops.edge_id("c0", "c1", "references", "Article 22")
    assert a == ops.edge_id("c0", "c1", "references", "Article 22")
    assert a != ops.edge_id("c0", "c1", "references", "Article 23")
    assert a != ops.edge_id("c0", "c1", "adjacent", "Article 22")
    assert a != ops.edge_id("c1", "c0", "references", "Article 22")
    assert len(a) == 64


def test_edge_id_separator_prevents_boundary_collisions():
    """Same house rule as models/ids.py: 'ab'+'c' must not equal 'a'+'bc'."""
    assert ops.edge_id("ab", "c", "references", "p") != ops.edge_id("a", "bc", "references", "p")


def test_insert_and_read_back_a_resolved_edge():
    conn = _ckb()
    eid = ops.insert_edge(conn, from_chunk="c0", to_chunk="c2", edge_type="references",
                          provenance="Article 22", confidence=1.0, created_at="2026-08-11T00:00:00Z")
    rows = ops.get_edges_from(conn, "c0")
    assert len(rows) == 1
    assert rows[0]["edge_id"] == eid
    assert rows[0]["to_chunk"] == "c2" and rows[0]["edge_type"] == "references"
    assert rows[0]["provenance"] == "Article 22" and rows[0]["confidence"] == 1.0
    conn.close()


def test_insert_edge_is_idempotent():
    conn = _ckb()
    for _ in range(3):
        ops.insert_edge(conn, from_chunk="c0", to_chunk="c1", edge_type="adjacent")
    assert ops.count_rows(conn, "edges") == 1        # re-runs must not accumulate duplicates
    conn.close()


def test_unresolved_reference_is_recorded_without_a_target_chunk():
    conn = _ckb()
    ops.insert_edge(conn, from_chunk="c0", to_external_ref="Directive 95/46/EC",
                    edge_type="references", provenance="Article 29 of Directive 95/46/EC")
    row = ops.get_edges_from(conn, "c0")[0]
    assert row["to_chunk"] is None
    assert row["to_external_ref"] == "Directive 95/46/EC"
    conn.close()


def test_delete_edges_of_type_leaves_other_types_alone():
    conn = _ckb()
    ops.insert_edge(conn, from_chunk="c0", to_chunk="c1", edge_type="adjacent")
    ops.insert_edge(conn, from_chunk="c0", to_chunk="c2", edge_type="references",
                    provenance="Article 22")
    assert ops.delete_edges_of_type(conn, "references") == 1
    assert ops.count_rows(conn, "edges") == 1
    assert ops.get_edges_from(conn, "c0")[0]["edge_type"] == "adjacent"
    conn.close()


def test_edges_respect_the_chunk_foreign_key():
    import sqlite3
    conn = _ckb()
    try:
        ops.insert_edge(conn, from_chunk="does-not-exist", to_chunk="c1", edge_type="adjacent")
        raised = False
    except sqlite3.IntegrityError:
        raised = True
    assert raised, "from_chunk must be a real chunk (FK enforced by get_db's PRAGMA)"
    conn.close()
