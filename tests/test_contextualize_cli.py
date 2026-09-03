from pathlib import Path

from test_contextualize import _FakeClient  # rootdir-inserted; see tests/ layout

from kbmcp.db import ops
from kbmcp.db.schema import create_all_tables
from kbmcp.ingest import contextualize as ctx
from kbmcp.ingest.chunk import ChunkRecord
from kbmcp.ingest.fetch import write_meta

FIX = Path(__file__).resolve().parent / "fixtures"
CFG = {
    "enabled": True, "model": "claude-haiku-4-5", "temperature": 0, "max_tokens": 150,
    "cache_ttl": "1h", "window_target_tokens": 60, "window_max_tokens": 100,
    "min_cacheable_tokens": 40, "chars_per_token": 4, "max_retries": 5,
    "price_in_per_mtok": 1.0, "price_out_per_mtok": 5.0,
    "cache_write_multiplier": 2.0, "cache_read_multiplier": 0.1,
}


def _recs(n, size=120):
    return [ChunkRecord(chunk_index=i, text=("w" * size), heading_path=["D"]) for i in range(n)]


def test_estimate_document_reports_tokens_and_dollars():
    est = ctx.estimate_document(_recs(8), CFG)
    assert est["chunks"] == 8 and est["windows"] >= 1
    assert est["output_tokens"] == 8 * CFG["max_tokens"]
    assert est["usd"] > 0
    # a cacheable window must charge ONE write plus reads, not N full prefixes
    assert est["cache_read_tokens"] > 0 or est["cacheable_windows"] == 0


def test_estimate_charges_one_cache_write_and_k_minus_one_reads_per_window():
    """The number this produces is what a human approves real spend from, so pin
    the arithmetic: one write per cacheable window, (k-1) reads (not k), per-chunk
    tokens always billed, and the multipliers actually applied."""
    from kbmcp.ingest.context_windows import est_tokens

    cfg = {**CFG, "min_cacheable_tokens": 1,          # force the window to be cacheable
           "window_target_tokens": 10_000, "window_max_tokens": 10_000}
    recs = _recs(3, size=400)
    est = ctx.estimate_document(recs, cfg)

    assert est["windows"] == 1 and est["cacheable_windows"] == 1 and est["chunks"] == 3
    prefix = est["cache_write_tokens"]                # exactly ONE write per cacheable window
    assert prefix > 0
    assert est["cache_read_tokens"] == prefix * 2     # (k-1) reads — a k/k-1 flip fails here

    per_chunk = est_tokens("w" * 400, 4) + est_tokens(ctx.TASK_INSTRUCTION, 4)
    assert est["input_tokens"] == per_chunk * 3       # per-chunk tokens billed regardless
    assert est["output_tokens"] == 3 * 150

    # multipliers written out independently of the implementation
    expected_usd = round((est["input_tokens"] * 1.0
                          + prefix * 1.0 * 2.0
                          + est["cache_read_tokens"] * 1.0 * 0.1
                          + est["output_tokens"] * 5.0) / 1_000_000, 4)
    assert est["usd"] == expected_usd

    # below the cache floor the SAME window bills its prefix once per chunk instead
    uncacheable = ctx.estimate_document(recs, {**cfg, "min_cacheable_tokens": 10 ** 9})
    assert uncacheable["cacheable_windows"] == 0
    assert uncacheable["cache_write_tokens"] == 0 and uncacheable["cache_read_tokens"] == 0
    assert uncacheable["input_tokens"] == per_chunk * 3 + prefix * 3
    assert uncacheable["usd"] > est["usd"]            # caching must be the cheaper path


def test_estimate_handles_a_single_chunk_window_without_negative_reads():
    cfg = {**CFG, "min_cacheable_tokens": 1, "window_target_tokens": 10_000,
           "window_max_tokens": 10_000}
    est = ctx.estimate_document(_recs(1, size=400), cfg)
    assert est["chunks"] == 1 and est["cache_read_tokens"] == 0   # k-1 == 0, never negative


def test_estimate_scales_with_chunk_count():
    small = ctx.estimate_document(_recs(4), CFG)
    big = ctx.estimate_document(_recs(16), CFG)
    assert big["usd"] > small["usd"]


def test_context_coverage_flags_empty_and_suspicious(safe_tmp_path):
    ckb = safe_tmp_path / "ckb.sqlite"
    conn = ops.get_db(ckb)
    create_all_tables(conn)
    ops.insert_source(conn, canonical_url="u", url_original="u", domain="ai", format="html",
                      license="x", license_ok=True, version="v1")
    ops.insert_doc(conn, doc_id="D", canonical_url="u", domain="ai", format="html", title="T")
    body = "The quick brown fox jumps over the lazy dog and keeps running."
    ops.insert_chunk(conn, chunk_id="c0", doc_id="D", chunk_index=0, text=body, chunk_type="text",
                     context="This passage from T describes an animal moving quickly.")
    ops.insert_chunk(conn, chunk_id="c1", doc_id="D", chunk_index=1, text=body, chunk_type="text",
                     context=None)
    ops.insert_chunk(conn, chunk_id="c2", doc_id="D", chunk_index=2, text=body, chunk_type="text",
                     context="short")
    ops.insert_chunk(conn, chunk_id="c3", doc_id="D", chunk_index=3, text=body, chunk_type="text",
                     context=body[:40])            # verbatim copy of the chunk
    conn.close()
    cov = ctx.context_coverage(ckb)
    assert cov["total"] == 4 and cov["with_context"] == 3
    assert cov["empty"] == ["c1"]
    assert set(cov["suspicious"]) == {"c2", "c3"}


def _mini_corpus(tmp):
    raw = tmp / "raw"
    raw.mkdir()
    (raw / "tiny.html").write_bytes((FIX / "tiny.html").read_bytes())
    write_meta(raw, {"doc_id": "tiny", "raw_filename": "tiny.html", "resolved_version": "v1",
                     "content_hash": "x", "format": "html", "source_url": "https://ex/tiny"})
    manifest = tmp / "m.yaml"
    manifest.write_text("- doc_id: tiny\n  url: https://ex/tiny\n  domain: ai\n  format: html\n"
                        "  license: x\n  license_ok: true\n  version: v1\n", encoding="utf-8")
    return manifest, raw


def test_cli_estimate_makes_no_api_call_and_writes_no_pins(safe_tmp_path, monkeypatch, capsys):
    manifest, raw = _mini_corpus(safe_tmp_path)
    called = []
    monkeypatch.setattr(ctx, "make_client", lambda cfg: (called.append(1), (None, "blocked"))[1])
    rc = ctx.main(["--manifest", str(manifest), "--raw-dir", str(raw),
                   "--parsed-dir", str(safe_tmp_path / "parsed"),
                   "--contexts-dir", str(safe_tmp_path / "contexts"),
                   "--config", "config/corpus_config.yaml", "--estimate"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "estimate" in err.lower() and "usd" in err.lower()
    assert not called                              # --estimate must not even build a client
    assert not (safe_tmp_path / "contexts").exists() or \
        not list((safe_tmp_path / "contexts").glob("*.json"))


def test_cli_generates_and_pins_with_an_injected_client(safe_tmp_path, monkeypatch):
    manifest, raw = _mini_corpus(safe_tmp_path)
    monkeypatch.setattr(ctx, "make_client", lambda cfg: (_FakeClient(), "ok"))
    rc = ctx.main(["--manifest", str(manifest), "--raw-dir", str(raw),
                   "--parsed-dir", str(safe_tmp_path / "parsed"),
                   "--contexts-dir", str(safe_tmp_path / "contexts"),
                   "--config", "config/corpus_config.yaml"])
    assert rc == 0
    record = ctx.load_pin_file(safe_tmp_path / "contexts", "tiny")
    assert record["model"] == "claude-haiku-4-5"
    assert len(record["contexts"]) >= 1


def test_cli_without_credentials_exits_nonzero_and_says_why(safe_tmp_path, monkeypatch, capsys):
    manifest, raw = _mini_corpus(safe_tmp_path)
    monkeypatch.setattr(ctx, "make_client", lambda cfg: (None, "no Anthropic credentials resolved"))
    rc = ctx.main(["--manifest", str(manifest), "--raw-dir", str(raw),
                   "--parsed-dir", str(safe_tmp_path / "parsed"),
                   "--contexts-dir", str(safe_tmp_path / "contexts"),
                   "--config", "config/corpus_config.yaml"])
    assert rc == 1
    assert "credentials" in capsys.readouterr().err.lower()


def test_load_document_returns_meta_version_records_and_title(safe_tmp_path):
    """One shared loader: build.py (Task 6) consumes the same 5-tuple, so the
    contextualize CLI and the build cannot drift on how a document is loaded."""
    from kbmcp.config import load_corpus_config
    from kbmcp.ingest.manifest import load_manifest

    manifest, raw = _mini_corpus(safe_tmp_path)
    entry = load_manifest(manifest)[0]
    cfg = load_corpus_config("config/corpus_config.yaml").raw
    meta, version, dl_doc, records, title = ctx.load_document(
        entry, raw, safe_tmp_path / "parsed", cfg)
    assert meta["raw_filename"] == "tiny.html"
    assert version == "v1"
    assert records and records[0].chunk_index == 0
    assert isinstance(title, str) and title
