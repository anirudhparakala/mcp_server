import httpx
import pytest

from kbmcp.ingest import fetch

CFG = {"timeout_s": 5, "retries": 2, "user_agent": "test-agent/1.0"}


def test_http_get_returns_bytes_and_metadata():
    def handler(request):
        assert request.headers["user-agent"] == "test-agent/1.0"
        return httpx.Response(200, content=b"<html>ok</html>", headers={"content-type": "text/html"})

    data, final_url, ctype, status = fetch._http_get(
        "https://example.com/x", CFG, transport=httpx.MockTransport(handler)
    )
    assert data == b"<html>ok</html>"
    assert status == 200
    assert ctype is not None and "text/html" in ctype


def test_http_get_raises_fetcherror_on_404():
    handler = lambda request: httpx.Response(404)
    with pytest.raises(fetch.FetchError):
        fetch._http_get("https://example.com/missing", CFG, transport=httpx.MockTransport(handler))


def test_http_get_follows_redirect_and_records_final_url():
    def handler(request):
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "https://example.com/end"})
        return httpx.Response(200, content=b"final")

    data, final_url, ctype, status = fetch._http_get(
        "https://example.com/start", CFG, transport=httpx.MockTransport(handler)
    )
    assert data == b"final"
    assert final_url.endswith("/end")


def test_http_get_retries_transport_error_then_succeeds():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("boom")
        return httpx.Response(200, content=b"ok")

    data, *_ = fetch._http_get("https://example.com/x", CFG, transport=httpx.MockTransport(handler))
    assert data == b"ok"
    assert calls["n"] == 2


def test_http_get_raises_after_exhausting_retries():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        raise httpx.ConnectError("down")

    with pytest.raises(fetch.FetchError):
        fetch._http_get("https://example.com/x", CFG, transport=httpx.MockTransport(handler))
    assert calls["n"] == CFG["retries"] + 1


def test_http_get_retries_5xx_then_succeeds():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503)
        return httpx.Response(200, content=b"ok")

    data, *_ = fetch._http_get("https://example.com/x", CFG, transport=httpx.MockTransport(handler))
    assert data == b"ok"
    assert calls["n"] == 2
