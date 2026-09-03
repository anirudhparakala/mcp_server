from pathlib import Path
from kbmcp.ingest import parse, chunk

FIX = Path(__file__).resolve().parent / "fixtures"
PCFG = {"table_mode": "accurate", "ocr": False}
CCFG = {"tokenizer": "Qwen/Qwen3-Embedding-0.6B", "max_tokens": 512}


def test_chunk_document_yields_contiguous_indices(safe_tmp_path):
    dl = parse.parse_source("tiny", FIX / "tiny.html", safe_tmp_path, PCFG)
    recs = chunk.chunk_document(dl, CCFG)
    assert len(recs) >= 1
    assert [r.chunk_index for r in recs] == list(range(len(recs)))  # 0-based contiguous
    assert any("widgets" in r.text or "Hello world" in r.text for r in recs)


def test_chunk_document_is_deterministic(safe_tmp_path):
    dl = parse.parse_source("tiny", FIX / "tiny.html", safe_tmp_path, PCFG)
    a = chunk.chunk_document(dl, CCFG)
    b = chunk.chunk_document(dl, CCFG)
    assert [(r.chunk_index, r.text) for r in a] == [(r.chunk_index, r.text) for r in b]


def test_chunk_document_marks_table_chunks(safe_tmp_path):
    dl = parse.parse_source("tbl", FIX / "tiny_table.html", safe_tmp_path, PCFG)
    recs = chunk.chunk_document(dl, CCFG)
    assert any(r.chunk_type == "table" for r in recs)
    tbl = next(r for r in recs if r.chunk_type == "table")
    assert "Protein" in tbl.text and tbl.table is not None
