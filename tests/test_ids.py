from kbmcp.models.ids import doc_id, chunk_id


def test_doc_id_is_deterministic_and_64_hex():
    a = doc_id("https://example.com/x", "v1")
    b = doc_id("https://example.com/x", "v1")
    assert a == b
    assert len(a) == 64
    assert all(c in "0123456789abcdef" for c in a)


def test_doc_id_changes_with_version():
    assert doc_id("u", "v1") != doc_id("u", "v2")


def test_chunk_id_varies_with_index():
    assert chunk_id("https://example.com/x", "v1", 0) != chunk_id("https://example.com/x", "v1", 1)


def test_doc_id_frozen_vector():
    # Guards the exact serialization (unit-separator + utf-8 + sha256).
    # Changing the scheme would break citations and gold labels — house rule.
    assert doc_id("u", "v") == "60cedb20de8f964f7894f1d68ea14786b32581ddf68e6f1a434246068b421c30"


def test_chunk_id_frozen_vector():
    assert chunk_id("u", "v", 0) == "b031bfb4c16489a6756f381befef44f71a2354119da59e86ed233973d0557ca6"
