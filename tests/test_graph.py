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


def _two_doc_ckb():
    conn = ops.get_db(":memory:")
    create_all_tables(conn)
    for slug, dom in (("gdpr", "law_aireg"), ("aiact", "law_aireg"), ("edpb", "law_aireg")):
        ops.insert_source(conn, canonical_url=slug, url_original=slug, domain=dom,
                          format="html", license="x", license_ok=True, version="v1")
        ops.insert_doc(conn, doc_id=slug.upper(), canonical_url=slug, domain=dom, format="html")
    ops.insert_chunk(conn, chunk_id="gdpr22", doc_id="GDPR", chunk_index=0, chunk_type="text",
                     text="Article 22\nAutomated individual decision-making\n1.\nThe data subject")
    ops.insert_chunk(conn, chunk_id="aiact22", doc_id="AIACT", chunk_index=0, chunk_type="text",
                     text="Article 22\nAuthorised representatives\n1.\nProviders established")
    ops.insert_chunk(conn, chunk_id="edpb0", doc_id="EDPB", chunk_index=0, chunk_type="text",
                     text="The prohibition in Article 22(1) applies to solely automated decisions.")
    ops.insert_chunk(conn, chunk_id="edpb1", doc_id="EDPB", chunk_index=1, chunk_type="text",
                     text="See Article 9999 which is not in the corpus at all.")
    return conn


def test_reference_resolves_into_the_declared_scope():
    conn = _two_doc_ckb()
    idx = graph.build_anchor_index(conn)
    from kbmcp.ingest.citations import Reference
    target = graph.resolve_reference(Reference("article", "22", "Article 22(1)"),
                                     from_doc_id="EDPB", scope_doc_ids={"EDPB", "GDPR"},
                                     anchor_index=idx)
    assert target == "gdpr22"          # NOT aiact22 -- the AI Act is out of scope
    conn.close()


def test_ambiguous_reference_across_scope_yields_no_edge():
    """If both in-scope docs define Article 22, guessing would be worse than nothing."""
    conn = _two_doc_ckb()
    idx = graph.build_anchor_index(conn)
    from kbmcp.ingest.citations import Reference
    target = graph.resolve_reference(Reference("article", "22", "Article 22"),
                                     from_doc_id="EDPB", scope_doc_ids={"EDPB", "GDPR", "AIACT"},
                                     anchor_index=idx)
    assert target is None
    conn.close()


def test_out_of_corpus_reference_is_unresolved():
    conn = _two_doc_ckb()
    idx = graph.build_anchor_index(conn)
    from kbmcp.ingest.citations import Reference
    assert graph.resolve_reference(Reference("article", "9999", "Article 9999"),
                                   from_doc_id="EDPB", scope_doc_ids={"EDPB", "GDPR"},
                                   anchor_index=idx) is None
    conn.close()


def test_scope_for_is_self_plus_declared_references():
    class E:
        def __init__(self, slug, refs):
            self.doc_id, self.references = slug, refs
    entries = {"edpb": E("edpb", ["gdpr"]), "gdpr": E("gdpr", [])}
    assert graph.scope_for("edpb", entries) == {"edpb", "gdpr"}
    assert graph.scope_for("gdpr", entries) == {"gdpr"}


def test_adjacent_edges_link_consecutive_chunks_within_a_document():
    conn = _two_doc_ckb()
    n = graph.build_adjacent_edges(conn)
    assert n == 1                                   # only EDPB has 2 chunks
    rows = ops.get_edges_from(conn, "edpb0")
    assert [r["to_chunk"] for r in rows if r["edge_type"] == "adjacent"] == ["edpb1"]
    conn.close()


def test_adjacent_edges_never_cross_a_document_boundary():
    conn = _two_doc_ckb()
    graph.build_adjacent_edges(conn)
    for row in conn.execute("SELECT from_chunk, to_chunk FROM edges WHERE edge_type='adjacent'"):
        a = ops.get_chunk(conn, row["from_chunk"])["doc_id"]
        b = ops.get_chunk(conn, row["to_chunk"])["doc_id"]
        assert a == b
    conn.close()


def test_build_graph_is_rerunnable_without_duplicating_edges(safe_tmp_path):
    ckb = safe_tmp_path / "ckb.sqlite"
    src = _two_doc_ckb()
    disk = ops.get_db(ckb)
    src.backup(disk)
    src.close()
    disk.close()

    manifest = safe_tmp_path / "m.yaml"
    manifest.write_text(
        "- doc_id: gdpr\n  url: gdpr\n  domain: law_aireg\n  format: html\n"
        "  license: x\n  license_ok: true\n  version: v1\n"
        "- doc_id: aiact\n  url: aiact\n  domain: law_aireg\n  format: html\n"
        "  license: x\n  license_ok: true\n  version: v1\n"
        "- doc_id: edpb\n  url: edpb\n  domain: law_aireg\n  format: html\n"
        "  license: x\n  license_ok: true\n  version: v1\n  references: [gdpr]\n",
        encoding="utf-8")
    cfg = {"extractors": ["legal", "academic"], "resolve": True,
           "edge_types": ["adjacent", "references"], "record_unresolved": True,
           "max_targets_per_reference": 1}

    first = graph.build_graph(ckb, manifest, safe_tmp_path / "raw", cfg,
                              doc_id_for=lambda slug: slug.upper())
    second = graph.build_graph(ckb, manifest, safe_tmp_path / "raw", cfg,
                              doc_id_for=lambda slug: slug.upper())
    assert first["references_resolved"] == second["references_resolved"] == 1
    conn = ops.get_db(ckb)
    assert ops.count_rows(conn, "edges") == first["adjacent"] + first["references_resolved"] \
        + first["references_unresolved"]
    conn.close()
