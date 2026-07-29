from pathlib import Path

from kbmcp.ingest import fetch
from kbmcp.ingest.manifest import SourceEntry


def _entry(**kw):
    base = dict(
        doc_id="d", url="https://example.com/x", format="html",
        license="CC-BY", license_ok=True, version="v1",
    )
    base.update(kw)
    return SourceEntry(**base)


def test_resolve_recipe_dispatch():
    assert fetch.resolve_recipe(_entry(url="file:///tmp/x.pdf")) == "local"
    assert fetch.resolve_recipe(_entry(url="https://arxiv.org/abs/1706.03762")) == "arxiv"
    assert fetch.resolve_recipe(_entry(url="https://www.law.cornell.edu/ucc/2")) == "generic"


def test_arxiv_id_strips_prefix_version_and_ext():
    assert fetch._arxiv_id("https://arxiv.org/abs/1706.03762") == "1706.03762"
    assert fetch._arxiv_id("https://arxiv.org/pdf/1706.03762v5.pdf") == "1706.03762"
    assert fetch._arxiv_id("https://arxiv.org/html/2005.11401v4") == "2005.11401"


def test_arxiv_urls_builds_versioned_html_and_pdf():
    html_url, pdf_url = fetch._arxiv_urls("https://arxiv.org/abs/1706.03762", "v7")
    assert html_url == "https://arxiv.org/html/1706.03762v7"
    assert pdf_url == "https://arxiv.org/pdf/1706.03762v7"


def test_read_local_returns_bytes_and_path(safe_tmp_path):
    f = safe_tmp_path / "doc.pdf"
    f.write_bytes(b"%PDF-1.4 data")
    data, path = fetch._read_local(f.as_uri())
    assert data == b"%PDF-1.4 data"
    assert Path(path) == f


def test_read_local_handles_percent_in_filename(safe_tmp_path):
    f = safe_tmp_path / "report%41.pdf"
    f.write_bytes(b"%PDF-1.4 data")
    data, path = fetch._read_local(f.as_uri())
    assert data == b"%PDF-1.4 data"
    assert Path(path) == f


def test_read_local_handles_spaces_and_ampersand_in_filename(safe_tmp_path):
    f = safe_tmp_path / "sp ace & plain.pdf"
    f.write_bytes(b"%PDF-1.4 data")
    data, path = fetch._read_local(f.as_uri())
    assert data == b"%PDF-1.4 data"
    assert Path(path) == f


def test_resolve_recipe_scheme_less_path_is_local():
    # A repo-relative path (authored / committed BYO source) routes to the local recipe.
    assert fetch.resolve_recipe(_entry(url="corpus/authored/waiver.html")) == "local"


def test_read_local_reads_repo_relative_path(safe_tmp_path, monkeypatch):
    # A scheme-less relative path is resolved from the current working directory.
    (safe_tmp_path / "sub").mkdir()
    (safe_tmp_path / "sub" / "doc.html").write_bytes(b"<html>rel</html>")
    monkeypatch.chdir(safe_tmp_path)
    data, path = fetch._read_local("sub/doc.html")
    assert data == b"<html>rel</html>"
    assert Path(path) == Path("sub/doc.html")
