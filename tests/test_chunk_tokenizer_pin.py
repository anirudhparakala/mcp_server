"""Tests for the Qwen3 tokenizer revision pin (ingest/chunk.py).

Hermetic: the resolver is injected, so no network call and no model download.
"""

import pytest

from kbmcp.ingest import chunk as ch


CFG = {"tokenizer": "Qwen/Qwen3-Embedding-0.6B",
       "tokenizer_revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
       "max_tokens": 800}


def test_matching_revision_passes_and_returns_it():
    got = ch.check_tokenizer_revision(CFG, resolver=lambda m: CFG["tokenizer_revision"])
    assert got == CFG["tokenizer_revision"]


def test_drift_raises_and_names_both_revisions():
    with pytest.raises(ch.TokenizerDriftError) as exc:
        ch.check_tokenizer_revision(CFG, resolver=lambda m: "ffffffffffffffffffffffffffffffffffffffff")
    msg = str(exc.value)
    assert CFG["tokenizer_revision"][:12] in msg      # the pin
    assert "ffffffffffff" in msg                       # what was resolved
    assert "--allow-tokenizer-drift" in msg            # the escape hatch


def test_allow_drift_suppresses_the_failure():
    got = ch.check_tokenizer_revision(
        CFG, allow_drift=True, resolver=lambda m: "ffffffffffffffffffffffffffffffffffffffff")
    assert got == "ffffffffffffffffffffffffffffffffffffffff"


def test_unpinned_config_is_permitted_and_returns_resolved():
    """A BYO user with no pin in their config must not be blocked."""
    got = ch.check_tokenizer_revision({"tokenizer": "X"}, resolver=lambda m: "abc123")
    assert got == "abc123"


def test_unresolvable_revision_is_not_fatal():
    """Offline or a hub outage must not break a build that would otherwise work."""
    def boom(model):
        raise OSError("network unreachable")
    assert ch.check_tokenizer_revision(CFG, resolver=boom) is None


def test_revision_is_part_of_the_chunker_cache_key():
    """_chunker is lru_cache'd; if revision is not in the key, a second call with a
    different revision silently returns the first chunker and drift is undetectable."""
    import inspect
    params = list(inspect.signature(ch._chunker.__wrapped__).parameters)
    assert params == ["model", "max_tokens", "revision"]
