from kbmcp.db import ops
from kbmcp.db.schema import create_all_tables
from kbmcp.ingest import graph


def _ckb(conn_path=":memory:"):
    conn = ops.get_db(conn_path)
    create_all_tables(conn)
    ops.insert_source(conn, canonical_url="gdpr", url_original="gdpr", domain="law_aireg",
                      format="html", license="x", license_ok=True, version="v1")
    ops.insert_doc(conn, doc_id="GDPR", canonical_url="gdpr", domain="law_aireg", format="html")
    ops.insert_chunk(conn, chunk_id="g0", doc_id="GDPR", chunk_index=0, chunk_type="text",
                     text="Article 21\nRight to object\n1.\nThe data subject shall have")
    ops.insert_chunk(conn, chunk_id="g1", doc_id="GDPR", chunk_index=1, chunk_type="text",
                     text="Article 22\nAutomated individual decision-making\n1.\nThe data subject")
    ops.insert_chunk(conn, chunk_id="g2", doc_id="GDPR", chunk_index=2, chunk_type="text",
                     text="continues the text of the same article, citing Article 22 inline")
    return conn


def test_anchor_index_maps_each_anchor_to_its_defining_chunk():
    conn = _ckb()
    idx = graph.build_anchor_index(conn)
    assert idx[("GDPR", "article", "22")] == "g1"
    assert idx[("GDPR", "article", "21")] == "g0"
    conn.close()


def test_anchor_index_ignores_inline_citations():
    """g2 only CITES Article 22; it must not claim to define it."""
    conn = _ckb()
    idx = graph.build_anchor_index(conn)
    assert idx[("GDPR", "article", "22")] == "g1"     # not g2
    conn.close()


def test_first_definer_wins_when_an_anchor_repeats():
    """Insertion order is deliberately the OPPOSITE of chunk_index order: the
    later restatement (chunk_index=3) is inserted BEFORE the true section body
    (chunk_index=1). This pins the ORDER BY doc_id, chunk_index clause in
    build_anchor_index -- without it, "first definer wins" would silently fall
    back to insertion/rowid order instead of chunk_index order, and a fixture
    that happens to insert chunks in chunk_index order would mask that
    regression. Do not "tidy" this back to insertion order.
    """
    conn = ops.get_db(":memory:")
    create_all_tables(conn)
    ops.insert_source(conn, canonical_url="gdpr", url_original="gdpr", domain="law_aireg",
                      format="html", license="x", license_ok=True, version="v1")
    ops.insert_doc(conn, doc_id="GDPR", canonical_url="gdpr", domain="law_aireg", format="html")
    ops.insert_chunk(conn, chunk_id="g_annex", doc_id="GDPR", chunk_index=3, chunk_type="text",
                     text="Article 22\nrestated later in an annex")
    ops.insert_chunk(conn, chunk_id="g_body", doc_id="GDPR", chunk_index=1, chunk_type="text",
                     text="Article 22\nAutomated individual decision-making\n1.\nThe data subject")
    idx = graph.build_anchor_index(conn)
    assert idx[("GDPR", "article", "22")] == "g_body"   # lowest chunk_index wins, not first inserted
    conn.close()


def test_persist_anchors_fills_the_citation_anchors_column():
    conn = _ckb()
    updated = graph.persist_anchors(conn)
    assert updated >= 2
    assert ops.get_chunk(conn, "g1")["citation_anchors"] == {"article": ["22"]}
    assert ops.get_chunk(conn, "g2")["citation_anchors"] == {}   # cites, defines nothing
    conn.close()


def test_persist_anchors_is_idempotent():
    conn = _ckb()
    graph.persist_anchors(conn)
    first = ops.get_chunk(conn, "g1")["citation_anchors"]
    graph.persist_anchors(conn)
    assert ops.get_chunk(conn, "g1")["citation_anchors"] == first
    conn.close()


def test_anchor_index_reaches_heading_path_only_anchors():
    """leginfo docs (CA Commercial Code) carry the section number as the LAST
    heading_path element and never repeat it in the chunk text -- calling
    extract_anchors(text) alone would silently drop these anchors."""
    conn = _ckb()
    ops.insert_source(conn, canonical_url="ucc-article-1", url_original="ucc-article-1",
                      domain="law_contract", format="html", license="x", license_ok=True,
                      version="v1")
    ops.insert_doc(conn, doc_id="UCC1", canonical_url="ucc-article-1", domain="law_contract",
                   format="html")
    ops.insert_chunk(conn, chunk_id="u0", doc_id="UCC1", chunk_index=0, chunk_type="text",
                     text="General definitions and principles of interpretation apply here.",
                     heading_path=["DIVISION 1", "CHAPTER 2", "1303."])
    idx = graph.build_anchor_index(conn)
    assert idx[("UCC1", "section", "1303")] == "u0"
    conn.close()
