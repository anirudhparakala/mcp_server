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
