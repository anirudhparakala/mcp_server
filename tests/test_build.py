from pathlib import Path
from kbmcp.ingest import build
from kbmcp.ingest.fetch import write_meta
from kbmcp.db import ops
from kbmcp.models.ids import doc_id as mk_doc_id, chunk_id as mk_chunk_id

FIX = Path(__file__).resolve().parent / "fixtures"
CFG = {"parse": {"table_mode": "accurate", "ocr": False},
       "chunk": {"tokenizer": "Qwen/Qwen3-Embedding-0.6B", "max_tokens": 512}}


def _mini_corpus(tmp):
    raw = tmp / "raw"; raw.mkdir()
    (raw / "tiny.html").write_bytes((FIX / "tiny.html").read_bytes())
    write_meta(raw, {"doc_id": "tiny", "raw_filename": "tiny.html", "resolved_version": "v1",
                     "content_hash": "x", "format": "html", "source_url": "https://ex/tiny"})
    manifest = tmp / "m.yaml"
    manifest.write_text("- doc_id: tiny\n  url: https://ex/tiny\n  domain: ai\n  format: html\n"
                        "  license: x\n  license_ok: true\n  version: v1\n", encoding="utf-8")
    return manifest, raw


def test_build_ckb_populates_docs_and_chunks(safe_tmp_path):
    manifest, raw = _mini_corpus(safe_tmp_path)
    ckb = safe_tmp_path / "ckb.sqlite"
    stats = build.build_ckb(manifest, raw, safe_tmp_path / "parsed", ckb, CFG)
    assert stats["docs"] == 1 and stats["chunks"] >= 1 and not stats["errors"]
    conn = ops.get_db(ckb)
    did = mk_doc_id("https://ex/tiny", "v1")
    chunks = ops.get_chunks_for_doc(conn, did)
    assert len(chunks) >= 1
    # house-rule chunk_id determinism
    assert chunks[0]["chunk_id"] == mk_chunk_id("https://ex/tiny", "v1", 0)
    conn.close()  # release the sqlite handle so safe_tmp_path cleanup can delete it (Windows)


def test_build_ckb_is_idempotent(safe_tmp_path):
    manifest, raw = _mini_corpus(safe_tmp_path)
    ckb = safe_tmp_path / "ckb.sqlite"
    build.build_ckb(manifest, raw, safe_tmp_path / "parsed", ckb, CFG)
    build.build_ckb(manifest, raw, safe_tmp_path / "parsed", ckb, CFG)  # must not double-insert or raise
    conn = ops.get_db(ckb)
    assert ops.count_rows(conn, "docs") == 1
    conn.close()


def test_build_ckb_force_rebuilds_existing_doc(safe_tmp_path):
    manifest, raw = _mini_corpus(safe_tmp_path)
    ckb = safe_tmp_path / "ckb.sqlite"; parsed = safe_tmp_path / "parsed"
    build.build_ckb(manifest, raw, parsed, ckb, CFG)
    stats = build.build_ckb(manifest, raw, parsed, ckb, CFG, force=True)  # must not PK-collide
    assert stats["docs"] == 1 and not stats["errors"]
    conn = ops.get_db(ckb)
    assert ops.count_rows(conn, "docs") == 1
    conn.close()


def test_build_ckb_cleans_partial_state_on_chunk_failure(safe_tmp_path, monkeypatch):
    manifest, raw = _mini_corpus(safe_tmp_path)
    ckb = safe_tmp_path / "ckb.sqlite"; parsed = safe_tmp_path / "parsed"
    # M4 moved parse+chunk behind contextualize.load_document, so the seam that
    # fails a doc AFTER its docs row is inserted (the case _delete_doc exists for)
    # is now the first insert_chunk. Failing earlier would never create a partial
    # row and so would not exercise the cleanup this test is about.
    real = build.ops.insert_chunk
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")   # fail the first doc mid-build
        return real(*a, **k)

    monkeypatch.setattr(build.ops, "insert_chunk", flaky)
    stats = build.build_ckb(manifest, raw, parsed, ckb, CFG)
    assert stats["errors"] and stats["docs"] == 0
    conn = ops.get_db(ckb); assert ops.count_rows(conn, "docs") == 0; conn.close()  # partial row cleaned
    # a later run must REBUILD the previously-failed doc, not skip it
    stats2 = build.build_ckb(manifest, raw, parsed, ckb, CFG)
    assert stats2["docs"] == 1 and not stats2["errors"]
    conn2 = ops.get_db(ckb); assert ops.count_rows(conn2, "docs") == 1; conn2.close()


CTX_CFG = {
    "enabled": True, "model": "claude-haiku-4-5", "temperature": 0, "max_tokens": 150,
    "cache_ttl": "1h", "window_target_tokens": 6000, "window_max_tokens": 8000,
    "min_cacheable_tokens": 4096, "chars_per_token": 4, "max_retries": 5,
    "price_in_per_mtok": 1.0, "price_out_per_mtok": 5.0,
    "cache_write_multiplier": 2.0, "cache_read_multiplier": 0.1,
}


def test_build_applies_pinned_contexts_without_any_client(safe_tmp_path, monkeypatch):
    from test_contextualize import _FakeClient

    from kbmcp.ingest import contextualize as ctx

    manifest, raw = _mini_corpus(safe_tmp_path)
    contexts_dir = safe_tmp_path / "contexts"
    cfg = {**CFG, "contextualize": {**CTX_CFG, "contexts_dir": str(contexts_dir)}}

    # First build with a fake client writes the pins...
    monkeypatch.setattr(ctx, "make_client", lambda c: (_FakeClient(), "ok"))
    build.build_ckb(manifest, raw, safe_tmp_path / "parsed", safe_tmp_path / "a.sqlite", cfg)

    # ...then a build with NO credentials must still populate context from the pins.
    monkeypatch.setattr(ctx, "make_client", lambda c: (None, "no credentials"))
    stats = build.build_ckb(manifest, raw, safe_tmp_path / "parsed", safe_tmp_path / "b.sqlite", cfg)
    assert stats["contexts"] == stats["chunks"] and stats["contexts_missing"] == 0
    conn = ops.get_db(safe_tmp_path / "b.sqlite")
    rows = ops.get_chunks_for_doc(conn, mk_doc_id("https://ex/tiny", "v1"))
    conn.close()
    assert rows and all((r["context"] or "").strip() for r in rows)


def test_build_without_key_or_pins_succeeds_with_empty_context(safe_tmp_path, monkeypatch, capsys):
    from kbmcp.ingest import contextualize as ctx

    manifest, raw = _mini_corpus(safe_tmp_path)
    cfg = {**CFG, "contextualize": {**CTX_CFG, "contexts_dir": str(safe_tmp_path / "contexts")}}
    monkeypatch.setattr(ctx, "make_client", lambda c: (None, "no credentials"))
    stats = build.build_ckb(manifest, raw, safe_tmp_path / "parsed", safe_tmp_path / "ckb.sqlite", cfg)
    assert stats["docs"] == 1 and not stats["errors"]          # build still SUCCEEDS
    assert stats["contexts"] == 0 and stats["contexts_missing"] == stats["chunks"]
    assert "[note]" in capsys.readouterr().err                 # one loud note, not an error
    conn = ops.get_db(safe_tmp_path / "ckb.sqlite")
    rows = ops.get_chunks_for_doc(conn, mk_doc_id("https://ex/tiny", "v1"))
    conn.close()
    assert rows and all(r["context"] is None for r in rows)


def test_build_no_context_flag_skips_generation_entirely(safe_tmp_path, monkeypatch):
    from test_contextualize import _FakeClient

    from kbmcp.ingest import contextualize as ctx

    manifest, raw = _mini_corpus(safe_tmp_path)
    client = _FakeClient()
    monkeypatch.setattr(ctx, "make_client", lambda c: (client, "ok"))
    cfg = {**CFG, "contextualize": {**CTX_CFG, "contexts_dir": str(safe_tmp_path / "contexts")}}
    stats = build.build_ckb(manifest, raw, safe_tmp_path / "parsed", safe_tmp_path / "ckb.sqlite",
                            cfg, no_context=True)
    assert client.messages.calls == []
    assert stats["contexts"] == 0


def test_build_require_context_reports_the_shortfall(safe_tmp_path, monkeypatch):
    from kbmcp.ingest import contextualize as ctx

    manifest, raw = _mini_corpus(safe_tmp_path)
    cfg = {**CFG, "contextualize": {**CTX_CFG, "contexts_dir": str(safe_tmp_path / "contexts")}}
    monkeypatch.setattr(ctx, "make_client", lambda c: (None, "no credentials"))
    stats = build.build_ckb(manifest, raw, safe_tmp_path / "parsed", safe_tmp_path / "ckb.sqlite",
                            cfg, require_context=True)
    assert any("context" in e["error"].lower() for e in stats["errors"])


def test_build_counters_exclude_a_document_rolled_back_mid_insert(safe_tmp_path, monkeypatch, capsys):
    """A doc that fails PARTWAY through its inserts is rolled back, so none of its
    chunks or contexts may remain in the totals -- otherwise the build reports
    nonsense like '2 of 0 chunks have no context'."""
    from test_contextualize import _FakeClient

    from kbmcp.ingest import contextualize as ctx

    manifest, raw = _mini_corpus(safe_tmp_path)
    cfg = {**CFG, "contextualize": {**CTX_CFG, "contexts_dir": str(safe_tmp_path / "contexts")}}
    monkeypatch.setattr(ctx, "make_client", lambda c: (_FakeClient(), "ok"))

    real = build.ops.insert_chunk
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:          # fixture yields 2 chunks; fail the SECOND
            raise RuntimeError("boom")
        return real(*a, **k)

    monkeypatch.setattr(build.ops, "insert_chunk", flaky)
    stats = build.build_ckb(manifest, raw, safe_tmp_path / "parsed",
                            safe_tmp_path / "ckb.sqlite", cfg)

    assert stats["docs"] == 0 and stats["errors"]        # doc failed and was rolled back
    assert stats["chunks"] == 0                          # no partially-inserted chunks counted
    assert stats["contexts"] == 0                        # ... and none of its contexts either
    assert stats["contexts_missing"] == 0
    assert "of 0 chunks" not in capsys.readouterr().err  # the incoherent note never appears
    conn = ops.get_db(safe_tmp_path / "ckb.sqlite")
    assert ops.count_rows(conn, "chunks") == 0 and ops.count_rows(conn, "docs") == 0
    conn.close()


def test_build_note_never_counts_a_rolled_back_document(safe_tmp_path, monkeypatch, capsys):
    """The keyless variant of the rollback case: without a client every chunk is
    'missing' a context, so a rolled-back doc used to produce the incoherent
    '2 of 0 chunks have no context' note."""
    from kbmcp.ingest import contextualize as ctx

    manifest, raw = _mini_corpus(safe_tmp_path)
    cfg = {**CFG, "contextualize": {**CTX_CFG, "contexts_dir": str(safe_tmp_path / "contexts")}}
    monkeypatch.setattr(ctx, "make_client", lambda c: (None, "no credentials"))

    real = build.ops.insert_chunk
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom")
        return real(*a, **k)

    monkeypatch.setattr(build.ops, "insert_chunk", flaky)
    stats = build.build_ckb(manifest, raw, safe_tmp_path / "parsed",
                            safe_tmp_path / "ckb.sqlite", cfg)

    assert stats["docs"] == 0 and stats["chunks"] == 0
    assert stats["contexts_missing"] == 0        # the rolled-back doc contributes nothing
    err = capsys.readouterr().err
    assert "of 0 chunks" not in err              # the incoherent note must not appear
    assert "[note]" not in err                   # nothing was built, so nothing to note


class _Interrupting:
    """Simulates Ctrl-C during the long contextualization pass."""

    def __init__(self):
        self.messages = self

    def create(self, **kwargs):
        raise KeyboardInterrupt("user pressed ctrl-c mid-run")


def test_interrupt_during_contextualization_leaves_no_half_built_doc(safe_tmp_path, monkeypatch):
    """A KeyboardInterrupt is NOT caught by `except Exception`, so if the docs row
    were written before contextualization it would survive with zero chunks and
    every later non-force build would skip that document forever."""
    from kbmcp.ingest import contextualize as ctx

    manifest, raw = _mini_corpus(safe_tmp_path)
    cfg = {**CFG, "contextualize": {**CTX_CFG, "contexts_dir": str(safe_tmp_path / "contexts")}}
    ckb = safe_tmp_path / "ckb.sqlite"

    monkeypatch.setattr(ctx, "make_client", lambda c: (_Interrupting(), "ok"))
    try:
        build.build_ckb(manifest, raw, safe_tmp_path / "parsed", ckb, cfg)
    except KeyboardInterrupt:
        pass                                        # propagates, as it should

    conn = ops.get_db(ckb)
    assert ops.count_rows(conn, "docs") == 0        # nothing half-built
    assert ops.count_rows(conn, "chunks") == 0
    conn.close()                                    # handle was released despite the interrupt

    # the interrupted document must be REBUILDABLE, not skipped forever
    monkeypatch.setattr(ctx, "make_client", lambda c: (None, "no credentials"))
    stats = build.build_ckb(manifest, raw, safe_tmp_path / "parsed", ckb, cfg)
    assert stats["docs"] == 1 and stats["chunks"] >= 1


def test_doc_with_zero_chunks_is_not_treated_as_already_built(safe_tmp_path):
    """Defence in depth for the same class of partial state: a docs row with no
    chunks must not make a later build skip the document."""
    from kbmcp.ingest.parse import parse_source

    manifest, raw = _mini_corpus(safe_tmp_path)
    ckb = safe_tmp_path / "ckb.sqlite"
    did = mk_doc_id("https://ex/tiny", "v1")
    conn = ops.get_db(ckb)
    build.create_all_tables(conn)
    ops.insert_source(conn, canonical_url="https://ex/tiny", url_original="https://ex/tiny",
                      domain="ai", format="html", license="x", license_ok=True, version="v1")
    ops.insert_doc(conn, doc_id=did, canonical_url="https://ex/tiny", domain="ai",
                   format="html", title="orphan")
    assert ops.count_rows(conn, "chunks") == 0
    conn.close()

    stats = build.build_ckb(manifest, raw, safe_tmp_path / "parsed", ckb, CFG)
    assert stats["docs"] == 1 and stats["chunks"] >= 1   # rebuilt, not skipped
    conn = ops.get_db(ckb)
    assert ops.count_rows(conn, "docs") == 1
    conn.close()


def test_full_build_prunes_docs_no_longer_in_the_manifest(safe_tmp_path):
    """Changing a source's URL or version changes its doc_id. Without reconciliation
    the OLD doc and its chunks linger forever — stale content that still gets
    embedded, indexed and served."""
    manifest, raw = _mini_corpus(safe_tmp_path)
    ckb = safe_tmp_path / "ckb.sqlite"
    build.build_ckb(manifest, raw, safe_tmp_path / "parsed", ckb, CFG)

    # simulate a previous build under a different URL (=> different doc_id)
    stale = mk_doc_id("https://ex/tiny-OLD-URL", "v1")
    conn = ops.get_db(ckb)
    ops.insert_source(conn, canonical_url="https://ex/tiny-OLD-URL",
                      url_original="https://ex/tiny-OLD-URL", domain="ai", format="html",
                      license="x", license_ok=True, version="v1")
    ops.insert_doc(conn, doc_id=stale, canonical_url="https://ex/tiny-OLD-URL",
                   domain="ai", format="html", title="stale")
    ops.insert_chunk(conn, chunk_id="stale-c0", doc_id=stale, chunk_index=0,
                     text="Code:\nSelect Code\nCONS", chunk_type="text")
    assert ops.count_rows(conn, "docs") == 2
    conn.close()

    stats = build.build_ckb(manifest, raw, safe_tmp_path / "parsed", ckb, CFG, force=True)
    assert stats["pruned"] == 1
    conn = ops.get_db(ckb)
    assert ops.count_rows(conn, "docs") == 1                    # stale doc gone
    assert ops.get_chunk(conn, "stale-c0") is None              # and its chunks
    conn.close()


def test_only_build_never_prunes(safe_tmp_path):
    """--only builds a subset; pruning there would delete every other document."""
    manifest, raw = _mini_corpus(safe_tmp_path)
    ckb = safe_tmp_path / "ckb.sqlite"
    build.build_ckb(manifest, raw, safe_tmp_path / "parsed", ckb, CFG)
    conn = ops.get_db(ckb)
    other = mk_doc_id("https://ex/other", "v1")
    ops.insert_source(conn, canonical_url="https://ex/other", url_original="https://ex/other",
                      domain="ai", format="html", license="x", license_ok=True, version="v1")
    ops.insert_doc(conn, doc_id=other, canonical_url="https://ex/other", domain="ai",
                   format="html", title="other")
    conn.close()

    stats = build.build_ckb(manifest, raw, safe_tmp_path / "parsed", ckb, CFG,
                            only={"tiny"}, force=True)
    assert stats["pruned"] == 0
    conn = ops.get_db(ckb)
    assert ops.count_rows(conn, "docs") == 2                    # untouched
    conn.close()


def test_build_does_not_prune_when_a_document_errored(safe_tmp_path, monkeypatch):
    """A load failure means we cannot know that doc's identity; deleting other rows
    on the strength of an incomplete picture would be destructive."""
    manifest, raw = _mini_corpus(safe_tmp_path)
    ckb = safe_tmp_path / "ckb.sqlite"
    build.build_ckb(manifest, raw, safe_tmp_path / "parsed", ckb, CFG)
    conn = ops.get_db(ckb)
    other = mk_doc_id("https://ex/other", "v1")
    ops.insert_source(conn, canonical_url="https://ex/other", url_original="https://ex/other",
                      domain="ai", format="html", license="x", license_ok=True, version="v1")
    ops.insert_doc(conn, doc_id=other, canonical_url="https://ex/other", domain="ai",
                   format="html", title="other")
    conn.close()

    from kbmcp.ingest import contextualize as ctx
    monkeypatch.setattr(ctx, "load_document",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    stats = build.build_ckb(manifest, raw, safe_tmp_path / "parsed", ckb, CFG, force=True)
    assert stats["errors"] and stats["pruned"] == 0
    conn = ops.get_db(ckb)
    assert ops.count_rows(conn, "docs") == 2                    # nothing deleted
    conn.close()


def test_force_rebuild_does_not_orphan_edges_or_raise_fk_error(safe_tmp_path):
    """review round 2, finding A: _delete_doc only deleted chunks, but
    edges.from_chunk/to_chunk are FOREIGN KEY REFERENCES chunks.chunk_id and
    get_db turns on PRAGMA foreign_keys. Once the M5 graph has run, `build --force`
    -- the documented rebuild path, and the exact command that applies newly
    generated context pins -- aborted with sqlite3.IntegrityError on any document
    with outgoing or incoming edges. _delete_doc must clear edges on BOTH sides
    before deleting chunks, and no orphan edge rows may survive a rebuild."""
    manifest, raw = _mini_corpus(safe_tmp_path)
    ckb = safe_tmp_path / "ckb.sqlite"
    stats = build.build_ckb(manifest, raw, safe_tmp_path / "parsed", ckb, CFG)
    assert stats["docs"] == 1 and not stats["errors"]

    tiny_did = mk_doc_id("https://ex/tiny", "v1")
    conn = ops.get_db(ckb)
    tiny_chunks = ops.get_chunks_for_doc(conn, tiny_did)
    assert len(tiny_chunks) >= 1
    tiny_chunk_id = tiny_chunks[0]["chunk_id"]

    # An unrelated chunk in the SAME doc's own space is enough to exercise both
    # directions of the FK without pulling in a second manifest entry (which the
    # prune pass at the end of a full build would otherwise delete on its own,
    # muddying what this test is isolating).
    ops.insert_chunk(conn, chunk_id="ghost-c0", doc_id=tiny_did, chunk_index=999,
                     text="ghost chunk standing in for a real cross-referencing chunk",
                     chunk_type="text")
    ops.insert_edge(conn, from_chunk=tiny_chunk_id, to_chunk="ghost-c0",
                    edge_type="references", provenance="out", confidence=1.0,
                    created_at="2026-08-11T00:00:00Z")
    ops.insert_edge(conn, from_chunk="ghost-c0", to_chunk=tiny_chunk_id,
                    edge_type="references", provenance="in", confidence=1.0,
                    created_at="2026-08-11T00:00:00Z")
    assert ops.count_rows(conn, "edges") == 2
    conn.close()

    # Must not raise sqlite3.IntegrityError.
    stats2 = build.build_ckb(manifest, raw, safe_tmp_path / "parsed", ckb, CFG, force=True)
    assert stats2["docs"] == 1 and not stats2["errors"]

    conn = ops.get_db(ckb)
    orphans = conn.execute(
        "SELECT edge_id FROM edges WHERE from_chunk = ? OR to_chunk = ?",
        (tiny_chunk_id, tiny_chunk_id),
    ).fetchall()
    assert orphans == []                       # neither direction survived as an orphan
    assert ops.count_rows(conn, "edges") == 0
    conn.close()
