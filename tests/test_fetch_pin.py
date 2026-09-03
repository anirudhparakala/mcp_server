from kbmcp.ingest import fetch


def test_content_hash_is_deterministic_known_vector():
    # sha256("hello world") — frozen vector; guards the exact hashing.
    assert (
        fetch.content_hash(b"hello world")
        == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    )


def test_meta_roundtrip(safe_tmp_path):
    meta = {
        "doc_id": "x", "raw_filename": "x.html", "recipe": "generic",
        "resolved_version": "v1", "content_hash": "abc", "content_type": "text/html",
        "final_url": "https://u", "format": "html", "fetched_at": "2026-07-17T00:00:00Z",
        "source_url": "https://u", "license": "CC-BY", "license_ok": True,
    }
    p = fetch.write_meta(safe_tmp_path, meta)
    assert p.exists()
    assert fetch.read_meta(safe_tmp_path, "x") == meta


def test_read_meta_missing_returns_none(safe_tmp_path):
    assert fetch.read_meta(safe_tmp_path, "nope") is None


def test_is_cached_true_only_when_raw_and_meta_exist(safe_tmp_path):
    assert fetch.is_cached(safe_tmp_path, "x") is False
    (safe_tmp_path / "x.html").write_bytes(b"data")
    fetch.write_meta(
        safe_tmp_path,
        {"doc_id": "x", "raw_filename": "x.html", "resolved_version": "v1", "content_hash": "abc"},
    )
    assert fetch.is_cached(safe_tmp_path, "x") is True


def test_is_cached_false_when_raw_bytes_missing(safe_tmp_path):
    # meta committed but bytes gitignored/absent (fresh-clone case) -> re-fetch.
    fetch.write_meta(
        safe_tmp_path,
        {"doc_id": "y", "raw_filename": "y.pdf", "resolved_version": "v1", "content_hash": "abc"},
    )
    assert fetch.is_cached(safe_tmp_path, "y") is False


def test_is_cached_false_when_meta_lacks_raw_filename(safe_tmp_path):
    # meta record present but missing the raw_filename key entirely -> must not
    # fall back to treating raw_dir itself as the "raw file" (it always exists).
    fetch.write_meta(
        safe_tmp_path,
        {"doc_id": "z", "resolved_version": "v1", "content_hash": "abc"},
    )
    assert fetch.is_cached(safe_tmp_path, "z") is False
