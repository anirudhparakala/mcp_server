"""Tests for local-source raw-cache extensions (ingest/fetch.py).

Docling infers input format from the file extension. Caching a .md or .docx as
.bin makes it unparseable, which would break --folder mode for every format
except PDF and HTML.
"""

from kbmcp.ingest import fetch


def test_remote_behaviour_is_unchanged():
    """The 55 committed pin records must keep resolving to the same paths."""
    assert fetch._ext_for("pdf") == ".pdf"
    assert fetch._ext_for("html") == ".html"
    assert fetch._ext_for("other") == ".bin"


def test_local_source_keeps_its_real_suffix():
    assert fetch._ext_for("other", "C:/docs/notes.md") == ".md"
    assert fetch._ext_for("other", "/home/u/report.docx") == ".docx"
    assert fetch._ext_for("other", "sheet.xlsx") == ".xlsx"


def test_local_pdf_and_html_still_normalise():
    """A local .htm should cache as .html, matching the declared format."""
    assert fetch._ext_for("pdf", "/docs/paper.pdf") == ".pdf"
    assert fetch._ext_for("html", "/docs/page.htm") == ".html"


def test_local_file_with_no_suffix_falls_back():
    assert fetch._ext_for("other", "/docs/README") == ".bin"


def test_suffix_is_lowercased_and_bounded():
    assert fetch._ext_for("other", "/docs/NOTES.MD") == ".md"
    # a pathological "suffix" must not become the filename
    assert fetch._ext_for("other", "/docs/x." + "a" * 40) == ".bin"


def test_committed_pin_records_still_resolve():
    """Regression guard for the 55 real sources: every committed *.meta.json must
    still name a raw file whose path _ext_for would produce today."""
    import json
    from pathlib import Path

    metas = sorted(Path("corpus/raw").glob("*.meta.json"))
    assert len(metas) >= 55, f"expected the committed pin records, found {len(metas)}"
    for m in metas:
        rec = json.loads(m.read_text(encoding="utf-8"))
        raw_name = rec.get("raw_filename")
        if not raw_name:
            continue
        expected_ext = fetch._ext_for(rec.get("format", "other"))
        assert raw_name.endswith(expected_ext), (
            f"{m.name}: pinned raw file {raw_name!r} no longer matches _ext_for "
            f"({expected_ext!r}) -- the extension change broke an existing pin"
        )
