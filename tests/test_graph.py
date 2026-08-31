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


# --- min_anchor_body_chars (review round 2, finding B): a text-derived header match
# followed by too little body in its own chunk must not claim the anchor -- the
# chunker routinely ends a chunk right after the NEXT article's header line, e.g.
# "...\nArticle 30\nRecords of processing activities\n1." (36 chars of body). ---

def _doc_with_two_chunks(conn, *, doc_id, slug, chunk0_text, chunk1_text):
    ops.insert_source(conn, canonical_url=slug, url_original=slug, domain="law_aireg",
                      format="html", license="x", license_ok=True, version="v1")
    ops.insert_doc(conn, doc_id=doc_id, canonical_url=slug, domain="law_aireg", format="html")
    ops.insert_chunk(conn, chunk_id=f"{doc_id}-c0", doc_id=doc_id, chunk_index=0,
                     chunk_type="text", text=chunk0_text)
    ops.insert_chunk(conn, chunk_id=f"{doc_id}-c1", doc_id=doc_id, chunk_index=1,
                     chunk_type="text", text=chunk1_text)


def test_header_split_from_its_body_anchors_the_next_chunk():
    """The concrete measured case: chunk 0 ends right after the header line with
    only a few characters of body; the real body lives in chunk 1. With the
    threshold configured, chunk 1 -- not chunk 0 -- must claim the anchor."""
    conn = ops.get_db(":memory:")
    create_all_tables(conn)
    _doc_with_two_chunks(
        conn, doc_id="GDPR2", slug="gdpr2",
        chunk0_text="...end of Article 29's text.\nArticle 30\nRecords of processing "
                     "activities\n1.",                              # 36 chars after "Article 30"
        chunk1_text="Each controller and each processor shall maintain a record of "
                     "processing activities under its responsibility. " * 3,  # long real body
    )
    idx = graph.build_anchor_index(conn, min_anchor_body_chars=200)
    assert idx[("GDPR2", "article", "30")] == "GDPR2-c1"
    conn.close()


def test_header_with_sufficient_body_still_anchors_itself():
    """A normal header-then-body chunk (the common case) must still anchor itself
    once the threshold is applied, not defer to the next chunk."""
    conn = ops.get_db(":memory:")
    create_all_tables(conn)
    _doc_with_two_chunks(
        conn, doc_id="GDPR3", slug="gdpr3",
        chunk0_text="Article 30\nRecords of processing activities\n1.\n"
                     "Each controller and each processor shall maintain a record of "
                     "processing activities under its responsibility. " * 3,
        chunk1_text="unrelated later content",
    )
    idx = graph.build_anchor_index(conn, min_anchor_body_chars=200)
    assert idx[("GDPR3", "article", "30")] == "GDPR3-c0"
    conn.close()


def test_no_qualifying_chunk_falls_back_to_the_header_chunk():
    """When the header is the LAST chunk of the document (no next chunk to defer
    to), losing the anchor entirely is worse than a marginal one -- it must still
    fall back to the original header chunk."""
    conn = ops.get_db(":memory:")
    create_all_tables(conn)
    ops.insert_source(conn, canonical_url="gdpr4", url_original="gdpr4", domain="law_aireg",
                      format="html", license="x", license_ok=True, version="v1")
    ops.insert_doc(conn, doc_id="GDPR4", canonical_url="gdpr4", domain="law_aireg", format="html")
    ops.insert_chunk(conn, chunk_id="GDPR4-c0", doc_id="GDPR4", chunk_index=0, chunk_type="text",
                     text="...end of Article 29's text.\nArticle 30\nRecords of processing "
                          "activities\n1.")     # only chunk in the doc; no successor to defer to
    idx = graph.build_anchor_index(conn, min_anchor_body_chars=200)
    assert idx[("GDPR4", "article", "30")] == "GDPR4-c0"
    conn.close()


def test_heading_path_derived_anchor_is_unaffected_by_the_body_threshold():
    """min_anchor_body_chars applies to TEXT-derived anchors only -- heading_path
    anchors (leginfo docs) already point at their own section's chunk and must
    resolve identically no matter how large the threshold is."""
    conn = ops.get_db(":memory:")
    create_all_tables(conn)
    ops.insert_source(conn, canonical_url="ucc-article-1b", url_original="ucc-article-1b",
                      domain="law_contract", format="html", license="x", license_ok=True,
                      version="v1")
    ops.insert_doc(conn, doc_id="UCC1B", canonical_url="ucc-article-1b", domain="law_contract",
                   format="html")
    ops.insert_chunk(conn, chunk_id="u0b", doc_id="UCC1B", chunk_index=0, chunk_type="text",
                     text="short",   # deliberately shorter than any realistic threshold
                     heading_path=["DIVISION 1", "CHAPTER 2", "1303."])
    idx = graph.build_anchor_index(conn, min_anchor_body_chars=200)
    assert idx[("UCC1B", "section", "1303")] == "u0b"
    conn.close()


def test_default_min_anchor_body_chars_preserves_prior_unthresholded_behaviour():
    """Existing callers that don't pass min_anchor_body_chars must keep working
    exactly as before: a short body must not get bumped to the next chunk."""
    conn = ops.get_db(":memory:")
    create_all_tables(conn)
    _doc_with_two_chunks(
        conn, doc_id="GDPR5", slug="gdpr5",
        chunk0_text="Article 30\nshort body\n",
        chunk1_text="unrelated later content",
    )
    idx = graph.build_anchor_index(conn)   # no threshold argument at all
    assert idx[("GDPR5", "article", "30")] == "GDPR5-c0"
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


def test_resolve_false_does_not_wipe_existing_reference_edges(safe_tmp_path):
    """resolve: false must SKIP rebuilding references, not DELETE them -- an
    operator using resolve: false for a fast partial rebuild must not lose
    reference edges a prior run already built (review round 1, finding 2)."""
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
    assert first["references_resolved"] == 1
    conn = ops.get_db(ckb)
    before = [r for r in ops.get_edges_from(conn, "edpb0") if r["edge_type"] == "references"]
    assert len(before) == 1
    conn.close()

    cfg_no_resolve = dict(cfg, resolve=False)
    second = graph.build_graph(ckb, manifest, safe_tmp_path / "raw", cfg_no_resolve,
                               doc_id_for=lambda slug: slug.upper())
    assert second["references_resolved"] == 0    # skipped this run, not rebuilt

    conn = ops.get_db(ckb)
    after = [r for r in ops.get_edges_from(conn, "edpb0") if r["edge_type"] == "references"]
    assert len(after) == 1                        # the earlier edge survived
    assert after[0]["to_chunk"] == before[0]["to_chunk"] == "gdpr22"
    conn.close()


def test_build_graph_counts_ambiguous_separately_and_emits_no_edge(safe_tmp_path):
    """EDPB's Article 22(1) has two in-scope candidates once the manifest scopes
    EDPB to both GDPR and the AI Act. That must land in stats['ambiguous'], not
    be folded into references_unresolved, and must produce NO edge at all --
    not even an unresolved placeholder (review round 1, finding 1)."""
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
        "  license: x\n  license_ok: true\n  version: v1\n"
        "  references: [gdpr, aiact]\n",
        encoding="utf-8")
    cfg = {"extractors": ["legal", "academic"], "resolve": True,
           "edge_types": ["references"], "record_unresolved": True,
           "max_targets_per_reference": 1}

    stats = graph.build_graph(ckb, manifest, safe_tmp_path / "raw", cfg,
                              doc_id_for=lambda slug: slug.upper())
    assert stats["ambiguous"] == 1                # Article 22(1) -- GDPR and AIACT both define it
    assert stats["references_unresolved"] == 1    # Article 9999 -- nowhere in the corpus
    assert stats["references_resolved"] == 0

    conn = ops.get_db(ckb)
    refs = [r for r in ops.get_edges_from(conn, "edpb0") if r["edge_type"] == "references"]
    assert refs == []                              # ambiguous -> no edge, not even a placeholder
    conn.close()


def test_build_graph_honours_configured_max_targets_per_reference(safe_tmp_path):
    """Changing max_targets_per_reference must actually change behaviour, not
    just be accepted: at the default (1) a 2-candidate reference is ambiguous;
    raising it to 2 must reclassify that same reference as unresolved instead
    (still no edge -- a single candidate is still required to emit one). This
    proves the config value is read, not merely accepted (review round 1,
    finding 3)."""
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
        "  license: x\n  license_ok: true\n  version: v1\n"
        "  references: [gdpr, aiact]\n",
        encoding="utf-8")
    base_cfg = {"extractors": ["legal", "academic"], "resolve": True,
                "edge_types": ["references"], "record_unresolved": True}

    strict = graph.build_graph(ckb, manifest, safe_tmp_path / "raw",
                               dict(base_cfg, max_targets_per_reference=1),
                               doc_id_for=lambda slug: slug.upper())
    assert strict["ambiguous"] == 1
    assert strict["references_unresolved"] == 1

    lenient = graph.build_graph(ckb, manifest, safe_tmp_path / "raw",
                                dict(base_cfg, max_targets_per_reference=2),
                                doc_id_for=lambda slug: slug.upper())
    assert lenient["ambiguous"] == 0               # 2 candidates <= configured max of 2
    assert lenient["references_unresolved"] == 2   # Article 22(1) AND Article 9999: still no edge
    assert lenient["references_resolved"] == 0


def test_build_graph_records_a_per_chunk_error_and_keeps_going(safe_tmp_path, monkeypatch):
    """An unexpected failure resolving one chunk's citations must not abort the
    whole pass: it is recorded in stats['errors'], and a chunk that sorts AFTER
    the failing one must still be processed (review round 1, finding 1)."""
    conn = ops.get_db(":memory:")
    create_all_tables(conn)
    for slug, dom in (("gdpr", "law_aireg"), ("edpb", "law_aireg"), ("later", "law_aireg")):
        ops.insert_source(conn, canonical_url=slug, url_original=slug, domain=dom,
                          format="html", license="x", license_ok=True, version="v1")
        ops.insert_doc(conn, doc_id=slug.upper(), canonical_url=slug, domain=dom, format="html")
    ops.insert_chunk(conn, chunk_id="gdpr22", doc_id="GDPR", chunk_index=0, chunk_type="text",
                     text="Article 22\nAutomated individual decision-making\n1.\nThe data subject")
    ops.insert_chunk(conn, chunk_id="edpb0", doc_id="EDPB", chunk_index=0, chunk_type="text",
                     text="See Article 9999 which triggers the injected failure.")
    ops.insert_chunk(conn, chunk_id="later0", doc_id="LATER", chunk_index=0, chunk_type="text",
                     text="This document cites Article 22 too.")
    ckb = safe_tmp_path / "ckb.sqlite"
    disk = ops.get_db(ckb)
    conn.backup(disk)
    conn.close()
    disk.close()

    manifest = safe_tmp_path / "m.yaml"
    manifest.write_text(
        "- doc_id: gdpr\n  url: gdpr\n  domain: law_aireg\n  format: html\n"
        "  license: x\n  license_ok: true\n  version: v1\n"
        "- doc_id: edpb\n  url: edpb\n  domain: law_aireg\n  format: html\n"
        "  license: x\n  license_ok: true\n  version: v1\n  references: [gdpr]\n"
        "- doc_id: later\n  url: later\n  domain: law_aireg\n  format: html\n"
        "  license: x\n  license_ok: true\n  version: v1\n  references: [gdpr]\n",
        encoding="utf-8")
    cfg = {"extractors": ["legal", "academic"], "resolve": True,
           "edge_types": ["references"], "record_unresolved": True,
           "max_targets_per_reference": 1}

    real_extract_references = graph.extract_references

    def flaky(text, *, extractors):
        if "9999" in text:
            raise ValueError("boom")
        return real_extract_references(text, extractors=extractors)

    monkeypatch.setattr(graph, "extract_references", flaky)

    stats = graph.build_graph(ckb, manifest, safe_tmp_path / "raw", cfg,
                              doc_id_for=lambda slug: slug.upper())
    assert len(stats["errors"]) == 1
    assert "edpb0" in stats["errors"][0]
    assert stats["references_resolved"] == 1      # LATER sorts after EDPB and still ran
    conn2 = ops.get_db(ckb)
    conn2.close()


def test_build_graph_honours_configured_min_anchor_body_chars(safe_tmp_path):
    """min_anchor_body_chars must actually reach build_anchor_index through
    build_graph's cfg, not just be accepted (review round 2, finding B)."""
    conn = ops.get_db(":memory:")
    create_all_tables(conn)
    for slug in ("gdpr", "citer"):
        ops.insert_source(conn, canonical_url=slug, url_original=slug, domain="law_aireg",
                          format="html", license="x", license_ok=True, version="v1")
        ops.insert_doc(conn, doc_id=slug.upper(), canonical_url=slug, domain="law_aireg",
                       format="html")
    ops.insert_chunk(conn, chunk_id="gdpr-c0", doc_id="GDPR", chunk_index=0, chunk_type="text",
                     text="...end of Article 29's text.\nArticle 30\nRecords of processing "
                          "activities\n1.")                       # header, thin body
    ops.insert_chunk(conn, chunk_id="gdpr-c1", doc_id="GDPR", chunk_index=1, chunk_type="text",
                     text="Each controller and each processor shall maintain a record of "
                          "processing activities under its responsibility. " * 3)  # real body
    ops.insert_chunk(conn, chunk_id="citer-c0", doc_id="CITER", chunk_index=0, chunk_type="text",
                     text="As required by Article 30, records must be kept.")

    ckb = safe_tmp_path / "ckb.sqlite"
    disk = ops.get_db(ckb)
    conn.backup(disk)
    conn.close()
    disk.close()

    manifest = safe_tmp_path / "m.yaml"
    manifest.write_text(
        "- doc_id: gdpr\n  url: gdpr\n  domain: law_aireg\n  format: html\n"
        "  license: x\n  license_ok: true\n  version: v1\n"
        "- doc_id: citer\n  url: citer\n  domain: law_aireg\n  format: html\n"
        "  license: x\n  license_ok: true\n  version: v1\n  references: [gdpr]\n",
        encoding="utf-8")
    base_cfg = {"extractors": ["legal", "academic"], "resolve": True,
                "edge_types": ["references"], "record_unresolved": True}

    unthresholded = graph.build_graph(ckb, manifest, safe_tmp_path / "raw", base_cfg,
                                      doc_id_for=lambda slug: slug.upper())
    assert unthresholded["references_resolved"] == 1
    conn = ops.get_db(ckb)
    edge = [r for r in ops.get_edges_from(conn, "citer-c0") if r["edge_type"] == "references"][0]
    assert edge["to_chunk"] == "gdpr-c0"          # default: no threshold, lands on the thin chunk
    conn.close()

    thresholded = graph.build_graph(ckb, manifest, safe_tmp_path / "raw",
                                    dict(base_cfg, min_anchor_body_chars=200),
                                    doc_id_for=lambda slug: slug.upper())
    assert thresholded["references_resolved"] == 1
    conn = ops.get_db(ckb)
    edge = [r for r in ops.get_edges_from(conn, "citer-c0") if r["edge_type"] == "references"][0]
    assert edge["to_chunk"] == "gdpr-c1"          # configured: defers to the chunk with real body
    conn.close()
