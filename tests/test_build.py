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
