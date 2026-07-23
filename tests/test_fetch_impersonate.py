import httpx
import pytest

from kbmcp.ingest import fetch
from kbmcp.ingest.manifest import SourceEntry


def _entry(url, **kw):
    base = dict(doc_id="d", url=url, format="html", license="x", license_ok=True, version="v1")
    base.update(kw)
    return SourceEntry(**base)


CFG = {
    "timeout_s": 5, "retries": 1, "user_agent": "t",
    "render": {"enabled": False, "hosts": []},
    "impersonate": {"enabled": True, "browser": "chrome", "hosts": ["law.justia.com", "nycourts.gov", "sec.gov"]},
}


def test_needs_impersonate_matches_flagged_hosts():
    assert fetch._needs_impersonate(_entry("https://law.justia.com/cases/x.html"), CFG) is True
    assert fetch._needs_impersonate(_entry("https://www.sec.gov/Archives/x.htm"), CFG) is True
    assert fetch._needs_impersonate(_entry("https://en.wikipedia.org/wiki/X"), CFG) is False


def test_needs_impersonate_false_when_disabled():
    cfg = {**CFG, "impersonate": {"enabled": False, "hosts": ["law.justia.com"]}}
    assert fetch._needs_impersonate(_entry("https://law.justia.com/x"), cfg) is False


def test_fetch_source_routes_flagged_host_through_impersonator(safe_tmp_path):
    calls = {}

    def fake_imp(url, cfg):
        calls["url"] = url
        return b"<html>opinion</html>", url, "text/html", 200

    e = _entry("https://law.justia.com/cases/hadley.html", doc_id="case-x")
    r = fetch.fetch_source(e, safe_tmp_path, CFG, impersonator=fake_imp)
    assert r.status == "ok"
    assert r.raw_path.read_bytes() == b"<html>opinion</html>"
    assert calls["url"] == e.url


def test_fetch_source_non_flagged_host_uses_httpx_not_impersonator(safe_tmp_path):
    def fake_imp(url, cfg):
        raise AssertionError("must not impersonate a non-flagged host")

    ok = httpx.MockTransport(lambda req: httpx.Response(200, content=b"<html>wiki</html>", headers={"content-type": "text/html"}))
    e = _entry("https://en.wikipedia.org/wiki/Consideration", doc_id="wiki-x")
    r = fetch.fetch_source(e, safe_tmp_path, CFG, transport=ok, impersonator=fake_imp)
    assert r.status == "ok"
    assert r.raw_path.read_bytes() == b"<html>wiki</html>"
