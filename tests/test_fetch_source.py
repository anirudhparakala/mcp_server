import httpx
import pytest

from kbmcp.ingest import fetch
from kbmcp.ingest.manifest import SourceEntry

CFG = {"timeout_s": 5, "retries": 1, "user_agent": "t", "render": {"enabled": False, "hosts": []}}


def _entry(**kw):
    base = dict(
        doc_id="wiki-x", url="https://en.wikipedia.org/w/index.php?oldid=1",
        format="html", license="CC-BY-SA", license_ok=True, version="oldid-1",
    )
    base.update(kw)
    return SourceEntry(**base)


def _ok_transport(content=b"<html>page</html>"):
    return httpx.MockTransport(
        lambda req: httpx.Response(200, content=content, headers={"content-type": "text/html"})
    )


def test_fetch_source_pins_raw_and_meta(safe_tmp_path):
    r = fetch.fetch_source(_entry(), safe_tmp_path, CFG, transport=_ok_transport())
    assert r.status == "ok"
    assert r.raw_path.exists() and r.raw_path.read_bytes() == b"<html>page</html>"
    assert r.meta_path.exists()
    assert r.content_hash == fetch.content_hash(b"<html>page</html>")
    assert r.resolved_version == "oldid-1"
    meta = fetch.read_meta(safe_tmp_path, "wiki-x")
    assert meta["license_ok"] is True and meta["raw_filename"] == "wiki-x.html"


def test_fetch_source_is_cache_first(safe_tmp_path):
    fetch.fetch_source(_entry(), safe_tmp_path, CFG, transport=_ok_transport())
    boom = httpx.MockTransport(lambda req: httpx.Response(500))  # would fail if used
    r2 = fetch.fetch_source(_entry(), safe_tmp_path, CFG, transport=boom)
    assert r2.status == "cached"


def test_fetch_source_local_uses_content_hash_version(safe_tmp_path):
    f = safe_tmp_path / "local.pdf"
    f.write_bytes(b"%PDF bytes")
    e = _entry(doc_id="local-doc", url=f.as_uri(), format="pdf", version="ignored")
    r = fetch.fetch_source(e, safe_tmp_path, CFG)
    assert r.status == "ok"
    assert r.resolved_version == f"sha256:{fetch.content_hash(b'%PDF bytes')}"
    assert r.raw_path.name == "local-doc.pdf"


def test_fetch_source_arxiv_falls_back_to_pdf(safe_tmp_path):
    def handler(req):
        if "/html/" in req.url.path:
            return httpx.Response(404)
        return httpx.Response(200, content=b"%PDF", headers={"content-type": "application/pdf"})

    e = _entry(doc_id="arxiv-x", url="https://arxiv.org/abs/1706.03762", format="pdf", version="v7")
    r = fetch.fetch_source(e, safe_tmp_path, CFG, transport=httpx.MockTransport(handler))
    assert r.status == "ok" and r.format == "pdf"
    assert r.raw_path.name == "arxiv-x.pdf"
    assert r.resolved_version == "v7"


def test_fetch_source_raises_on_http_error(safe_tmp_path):
    boom = httpx.MockTransport(lambda req: httpx.Response(404))
    with pytest.raises(fetch.FetchError):
        fetch.fetch_source(_entry(), safe_tmp_path, CFG, transport=boom)


def test_fetch_all_collects_errors_without_raising(safe_tmp_path):
    good = _entry(doc_id="good")
    bad = _entry(doc_id="bad", url="https://arxiv.org/abs/9999.99999", format="pdf", version="v1")

    def handler(req):
        if "arxiv.org" in req.url.host:
            return httpx.Response(404)  # both html and pdf fail -> error
        return httpx.Response(200, content=b"ok", headers={"content-type": "text/html"})

    results = fetch.fetch_all([good, bad], safe_tmp_path, CFG, transport=httpx.MockTransport(handler))
    by_id = {r.doc_id: r.status for r in results}
    assert by_id == {"good": "ok", "bad": "error"}


def test_fetch_all_does_not_raise_on_malformed_entry(safe_tmp_path):
    good = _entry(doc_id="good")
    bad = _entry(doc_id="bad", url=123)

    def handler(req):
        return httpx.Response(200, content=b"ok", headers={"content-type": "text/html"})

    results = fetch.fetch_all([good, bad], safe_tmp_path, CFG, transport=httpx.MockTransport(handler))
    by_id = {r.doc_id: r for r in results}
    assert by_id["good"].status == "ok"
    assert by_id["bad"].status == "error"
    assert by_id["bad"].error


def test_fetch_source_missing_local_file_raises_fetcherror(safe_tmp_path):
    missing = safe_tmp_path / "nope.pdf"
    e = _entry(doc_id="local-missing", url=missing.as_uri(), format="pdf", version="ignored")
    with pytest.raises(fetch.FetchError):
        fetch.fetch_source(e, safe_tmp_path, CFG)


def test_render_required_without_renderer_raises(safe_tmp_path):
    cfg = {"timeout_s": 5, "retries": 1, "user_agent": "t",
           "render": {"enabled": True, "hosts": ["eur-lex.europa.eu"]}}
    e = _entry(doc_id="eurlex-x", url="https://eur-lex.europa.eu/eli/reg/2016/679/oj",
               version="consolidated-2016-05-04")
    with pytest.raises(fetch.FetchError):
        fetch.fetch_source(e, safe_tmp_path, cfg)
