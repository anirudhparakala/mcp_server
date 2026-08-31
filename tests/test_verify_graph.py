"""Tests for the cross-reference ground-truth gate (verify_graph.py), which
consumes corpus/benchmark/graph_expectations.yaml the way verify_structure.py
consumes structure_fixtures.yaml -- see that module for the pattern this
mirrors. Hermetic: builds tiny in-memory-derived CKBs, never depends on
ckb/ckb.sqlite existing.
"""

from kbmcp.db import ops
from kbmcp.db.schema import create_all_tables
from kbmcp.ingest import verify_graph as vg
from kbmcp.models.ids import doc_id as mk_doc_id


def _manifest_and_pins(safe_tmp_path, sources):
    """sources: [(slug, url, resolved_version)]. Writes a manifest.yaml and
    matching corpus/raw/<slug>.meta.json pin records, and returns
    {slug: derived doc_id} computed the same way doc_id_for_slug does, so
    tests can insert matching rows into a hermetic DB."""
    raw_dir = safe_tmp_path / "raw"
    raw_dir.mkdir()
    lines = []
    derived = {}
    for slug, url, version in sources:
        lines.append(
            f"- doc_id: {slug}\n  url: {url}\n  domain: law_aireg\n  format: html\n"
            f"  license: x\n  license_ok: true\n  version: v1\n"
        )
        (raw_dir / f"{slug}.meta.json").write_text(
            f'{{"resolved_version": "{version}"}}', encoding="utf-8"
        )
        derived[slug] = mk_doc_id(url, version)
    manifest = safe_tmp_path / "m.yaml"
    manifest.write_text("".join(lines), encoding="utf-8")
    return manifest, raw_dir, derived


def _ckb_with_reference_edge(path, *, from_did, to_did, resolved=True):
    conn = ops.get_db(path)
    create_all_tables(conn)
    ops.insert_source(conn, canonical_url="from-url", url_original="from-url",
                      domain="law_aireg", format="html", license="x",
                      license_ok=True, version="v1")
    ops.insert_doc(conn, doc_id=from_did, canonical_url="from-url", domain="law_aireg",
                   format="html")
    ops.insert_chunk(conn, chunk_id="from-c0", doc_id=from_did, chunk_index=0,
                     chunk_type="text", text="cites the other document")

    ops.insert_source(conn, canonical_url="to-url", url_original="to-url",
                      domain="law_aireg", format="html", license="x",
                      license_ok=True, version="v1")
    ops.insert_doc(conn, doc_id=to_did, canonical_url="to-url", domain="law_aireg",
                   format="html")
    ops.insert_chunk(conn, chunk_id="to-c0", doc_id=to_did, chunk_index=0,
                     chunk_type="text", text="defines the cited section")

    if resolved:
        ops.insert_edge(conn, from_chunk="from-c0", to_chunk="to-c0",
                        edge_type="references", provenance="ref", confidence=1.0,
                        created_at="2026-01-01T00:00:00Z")
    else:
        ops.insert_edge(conn, from_chunk="from-c0", to_external_ref="article:99",
                        edge_type="references", provenance="ref", confidence=0.0,
                        created_at="2026-01-01T00:00:00Z")
    conn.close()


def test_active_expectation_passes_when_resolved_edge_exists(safe_tmp_path):
    manifest, raw_dir, derived = _manifest_and_pins(
        safe_tmp_path, [("citer", "http://citer", "v1"), ("cited", "http://cited", "v1")]
    )
    ckb = safe_tmp_path / "ckb.sqlite"
    _ckb_with_reference_edge(ckb, from_did=derived["citer"], to_did=derived["cited"])

    exp = safe_tmp_path / "expectations.yaml"
    exp.write_text(
        "- name: citer-to-cited\n  from_slug: citer\n  to_slug: cited\n  min_edges: 1\n",
        encoding="utf-8",
    )

    results = vg.verify_graph(ckb, exp, manifest, raw_dir)
    assert len(results) == 1
    r = results[0]
    assert r.name == "citer-to-cited"
    assert r.passed is True
    assert r.pending is False
    assert r.edges_observed == 1
    assert vg.all_passed(results) is True


def test_active_expectation_fails_and_fails_the_gate(safe_tmp_path):
    manifest, raw_dir, derived = _manifest_and_pins(
        safe_tmp_path, [("citer", "http://citer", "v1"), ("cited", "http://cited", "v1")]
    )
    ckb = safe_tmp_path / "ckb.sqlite"
    # Only an UNRESOLVED edge exists (to_chunk NULL) -- must not count.
    _ckb_with_reference_edge(ckb, from_did=derived["citer"], to_did=derived["cited"],
                             resolved=False)

    exp = safe_tmp_path / "expectations.yaml"
    exp.write_text(
        "- name: citer-to-cited\n  from_slug: citer\n  to_slug: cited\n  min_edges: 1\n",
        encoding="utf-8",
    )

    results = vg.verify_graph(ckb, exp, manifest, raw_dir)
    assert len(results) == 1
    r = results[0]
    assert r.passed is False
    assert r.edges_observed == 0
    assert vg.all_passed(results) is False


def test_pending_expectation_is_reported_but_excluded_from_all_passed(safe_tmp_path):
    manifest, raw_dir, derived = _manifest_and_pins(
        safe_tmp_path,
        [("citer", "http://citer", "v1"), ("cited", "http://cited", "v1"),
         ("other-citer", "http://other-citer", "v1"), ("other-cited", "http://other-cited", "v1")],
    )
    ckb = safe_tmp_path / "ckb.sqlite"
    # The active expectation passes...
    _ckb_with_reference_edge(ckb, from_did=derived["citer"], to_did=derived["cited"])
    # ...but the pending pair never resolves (no edge at all between them).
    conn = ops.get_db(ckb)
    ops.insert_source(conn, canonical_url="oc-url", url_original="oc-url", domain="law_aireg",
                      format="html", license="x", license_ok=True, version="v1")
    ops.insert_doc(conn, doc_id=derived["other-citer"], canonical_url="oc-url",
                   domain="law_aireg", format="html")
    ops.insert_chunk(conn, chunk_id="oc-c0", doc_id=derived["other-citer"], chunk_index=0,
                     chunk_type="text", text="cites nothing resolvable")
    ops.insert_source(conn, canonical_url="ocd-url", url_original="ocd-url", domain="law_aireg",
                      format="html", license="x", license_ok=True, version="v1")
    ops.insert_doc(conn, doc_id=derived["other-cited"], canonical_url="ocd-url",
                   domain="law_aireg", format="html")
    ops.insert_chunk(conn, chunk_id="ocd-c0", doc_id=derived["other-cited"], chunk_index=0,
                     chunk_type="text", text="defines something else")
    conn.close()

    exp = safe_tmp_path / "expectations.yaml"
    exp.write_text(
        "- name: active-pair\n  from_slug: citer\n  to_slug: cited\n  min_edges: 1\n"
        "- name: pending-pair\n  from_slug: other-citer\n  to_slug: other-cited\n"
        "  min_edges: 1\n  pending: true\n",
        encoding="utf-8",
    )

    results = vg.verify_graph(ckb, exp, manifest, raw_dir)
    assert len(results) == 2
    by_name = {r.name: r for r in results}
    assert by_name["active-pair"].passed is True
    pending_result = by_name["pending-pair"]
    assert pending_result.pending is True
    assert pending_result.passed is False           # honestly reported as failing...
    assert pending_result.edges_observed == 0
    assert vg.all_passed(results) is True            # ...but does not fail the gate
