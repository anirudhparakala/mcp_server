from pathlib import Path
from kbmcp.ingest import parse

FIX = Path(__file__).resolve().parent / "fixtures"
CFG = {"table_mode": "accurate", "ocr": False}


def test_parse_source_produces_doclingdocument_and_caches(safe_tmp_path):
    dl = parse.parse_source("tiny", FIX / "tiny.html", safe_tmp_path, CFG)
    md = dl.export_to_markdown()
    assert "Hello world paragraph one." in md
    assert parse.parsed_path_for(safe_tmp_path, "tiny").exists()  # committed cache written


def test_parse_source_is_cache_first(safe_tmp_path):
    parse.parse_source("tiny", FIX / "tiny.html", safe_tmp_path, CFG)
    # corrupt the raw path; a cache hit must NOT touch it
    dl = parse.parse_source("tiny", safe_tmp_path / "does-not-exist.html", safe_tmp_path, CFG)
    assert "Hello world paragraph one." in dl.export_to_markdown()


def test_parse_source_extracts_table(safe_tmp_path):
    dl = parse.parse_source("tbl", FIX / "tiny_table.html", safe_tmp_path, CFG)
    md = dl.export_to_markdown()
    assert "Protein" in md and "0.8 g/kg" in md
    assert len(dl.tables) >= 1  # TableFormer recovered a table structure
