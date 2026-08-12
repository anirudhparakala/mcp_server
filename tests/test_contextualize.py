import json

from kbmcp.ingest.chunk import ChunkRecord
from kbmcp.ingest import contextualize as ctx
from kbmcp.models.ids import chunk_id as mk_chunk_id

URL, VER = "https://ex/doc", "v1"


def _recs(*texts):
    return [ChunkRecord(chunk_index=i, text=t, heading_path=["D"]) for i, t in enumerate(texts)]


def _pin(records, contexts, *, model="claude-haiku-4-5", prompt_version=None):
    return {
        "doc_id": "slug",
        "model": model,
        "prompt_version": ctx.PROMPT_VERSION if prompt_version is None else prompt_version,
        "generated_at": "2026-08-11T00:00:00Z",
        "contexts": {
            mk_chunk_id(URL, VER, r.chunk_index): {
                "context": contexts[i],
                "text_sha256": ctx.text_sha256(r.text),
                "window_index": 0,
            }
            for i, r in enumerate(records)
        },
    }


def test_save_then_load_roundtrips_deterministically(safe_tmp_path):
    recs = _recs("alpha", "beta")
    rec = _pin(recs, ["ctx a", "ctx b"])
    p1 = ctx.save_pin_file(safe_tmp_path, "slug", rec)
    first = p1.read_text(encoding="utf-8")
    ctx.save_pin_file(safe_tmp_path, "slug", rec)
    assert p1.read_text(encoding="utf-8") == first          # byte-identical rewrite
    assert first.endswith("\n")
    assert json.loads(first) == rec
    assert ctx.load_pin_file(safe_tmp_path, "slug") == rec


def test_load_pin_file_returns_empty_dict_when_absent(safe_tmp_path):
    assert ctx.load_pin_file(safe_tmp_path, "nope") == {}


def test_valid_pinned_returns_contexts_keyed_by_chunk_id():
    recs = _recs("alpha", "beta")
    got = ctx.valid_pinned(_pin(recs, ["ctx a", "ctx b"]), recs, canonical_url=URL, version=VER,
                           model="claude-haiku-4-5", prompt_version=ctx.PROMPT_VERSION)
    assert got == {mk_chunk_id(URL, VER, 0): "ctx a", mk_chunk_id(URL, VER, 1): "ctx b"}


def test_valid_pinned_drops_entry_whose_chunk_text_changed():
    recs = _recs("alpha", "beta")
    record = _pin(recs, ["ctx a", "ctx b"])
    changed = _recs("alpha", "beta REWRITTEN BY A NEW CHUNKER")
    got = ctx.valid_pinned(record, changed, canonical_url=URL, version=VER,
                           model="claude-haiku-4-5", prompt_version=ctx.PROMPT_VERSION)
    assert got == {mk_chunk_id(URL, VER, 0): "ctx a"}       # chunk 1 dropped, will regenerate


def test_valid_pinned_drops_everything_on_model_or_prompt_change():
    recs = _recs("alpha", "beta")
    stale_model = _pin(recs, ["a", "b"], model="claude-3-haiku-20240307")
    stale_prompt = _pin(recs, ["a", "b"], prompt_version=ctx.PROMPT_VERSION - 1)
    for record in (stale_model, stale_prompt):
        assert ctx.valid_pinned(record, recs, canonical_url=URL, version=VER,
                                model="claude-haiku-4-5", prompt_version=ctx.PROMPT_VERSION) == {}


def test_valid_pinned_handles_empty_or_missing_record():
    recs = _recs("alpha")
    assert ctx.valid_pinned({}, recs, canonical_url=URL, version=VER,
                            model="claude-haiku-4-5", prompt_version=ctx.PROMPT_VERSION) == {}


def test_valid_pinned_drops_blank_contexts():
    recs = _recs("alpha", "beta")
    record = _pin(recs, ["ctx a", "   "])
    got = ctx.valid_pinned(record, recs, canonical_url=URL, version=VER,
                           model="claude-haiku-4-5", prompt_version=ctx.PROMPT_VERSION)
    assert list(got) == [mk_chunk_id(URL, VER, 0)]


def test_valid_pinned_handles_malformed_record_shapes():
    """Covers malformed record shapes: non-dict entry, null contexts, missing contexts.
    Valid entries are preserved; malformed are dropped."""
    recs = _recs("alpha", "beta", "gamma")

    # Non-dict entry: valid entries still come back
    record_nondict = _pin(recs, ["ctx a", "ctx b", "ctx c"])
    record_nondict["contexts"][mk_chunk_id(URL, VER, 1)] = "not a dict"
    got_nondict = ctx.valid_pinned(record_nondict, recs, canonical_url=URL, version=VER,
                                   model="claude-haiku-4-5", prompt_version=ctx.PROMPT_VERSION)
    assert mk_chunk_id(URL, VER, 0) in got_nondict  # Valid entry
    assert mk_chunk_id(URL, VER, 1) not in got_nondict  # Malformed dropped
    assert mk_chunk_id(URL, VER, 2) in got_nondict  # Valid entry

    # Null contexts: no entries, but no crash
    record_null = {"doc_id": "slug", "model": "claude-haiku-4-5", "prompt_version": ctx.PROMPT_VERSION, "generated_at": "2026-08-11T00:00:00Z", "contexts": None}
    got_null = ctx.valid_pinned(record_null, recs, canonical_url=URL, version=VER,
                                model="claude-haiku-4-5", prompt_version=ctx.PROMPT_VERSION)
    assert got_null == {}

    # Missing contexts key: no entries, but no crash
    record_missing = {"doc_id": "slug", "model": "claude-haiku-4-5", "prompt_version": ctx.PROMPT_VERSION, "generated_at": "2026-08-11T00:00:00Z"}
    got_missing = ctx.valid_pinned(record_missing, recs, canonical_url=URL, version=VER,
                                   model="claude-haiku-4-5", prompt_version=ctx.PROMPT_VERSION)
    assert got_missing == {}


class _FakeUsage:
    def __init__(self, cache_read=0, cache_write=0, inp=100, out=20):
        self.input_tokens = inp
        self.output_tokens = out
        self.cache_creation_input_tokens = cache_write
        self.cache_read_input_tokens = cache_read


class _FakeBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _FakeMessage:
    def __init__(self, text, usage):
        self.content = [_FakeBlock(text)]
        self.usage = usage
        self.stop_reason = "end_turn"


class _FakeMessages:
    """Records every request; first call per prefix 'writes' cache, later ones 'read' it."""

    def __init__(self, replies=None, raise_on=()):
        self.calls = []
        self._replies = list(replies or [])
        self._seen_prefixes = set()
        self._raise_on = set(raise_on)

    def create(self, **kwargs):
        n = len(self.calls)
        self.calls.append(kwargs)
        if n in self._raise_on:
            raise RuntimeError("boom")
        prefix = kwargs["messages"][0]["content"][0]["text"]
        usage = _FakeUsage(cache_read=500) if prefix in self._seen_prefixes else _FakeUsage(cache_write=500)
        self._seen_prefixes.add(prefix)
        reply = self._replies[n] if n < len(self._replies) else f"context {n}"
        return _FakeMessage(reply, usage)


class _FakeClient:
    def __init__(self, replies=None, raise_on=()):
        self.messages = _FakeMessages(replies, raise_on)


CFG = {
    "enabled": True, "model": "claude-haiku-4-5", "temperature": 0, "max_tokens": 150,
    "cache_ttl": "1h", "contexts_dir": "corpus/contexts", "window_target_tokens": 60,
    "window_max_tokens": 100, "min_cacheable_tokens": 4096, "chars_per_token": 4,
    "max_retries": 5, "price_in_per_mtok": 1.0, "price_out_per_mtok": 5.0,
    "cache_write_multiplier": 2.0, "cache_read_multiplier": 0.1,
}


def test_build_messages_caches_only_the_window_block():
    msgs = ctx.build_messages("# Title\n(https://ex)", "WINDOW TEXT", "CHUNK TEXT", cache_ttl="1h")
    assert len(msgs) == 1 and msgs[0]["role"] == "user"
    blocks = msgs[0]["content"]
    assert len(blocks) == 2
    assert blocks[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert "WINDOW TEXT" in blocks[0]["text"] and "# Title" in blocks[0]["text"]
    assert "cache_control" not in blocks[1]        # volatile suffix must NOT be a breakpoint
    assert "CHUNK TEXT" in blocks[1]["text"]
    assert "CHUNK TEXT" not in blocks[0]["text"]   # chunk must not leak into the cached prefix


def test_generator_sends_locked_model_params_and_returns_context_and_usage():
    client = _FakeClient(replies=["This chunk states the protein RDA."])
    gen = ctx.ContextGenerator(client, CFG)
    context, usage = gen.generate("# T", "window", "chunk")
    assert context == "This chunk states the protein RDA."
    assert usage["cache_creation_input_tokens"] == 500
    kwargs = client.messages.calls[0]
    assert kwargs["model"] == "claude-haiku-4-5"
    assert kwargs["max_tokens"] == 150
    assert kwargs["temperature"] == 0
    assert "thinking" not in kwargs               # Haiku 4.5: no thinking parameter
    assert isinstance(kwargs["system"], list)


def test_contexts_for_document_generates_one_call_per_chunk_and_pins(safe_tmp_path):
    recs = _recs("alpha " * 30, "beta " * 30, "gamma " * 30)
    client = _FakeClient()
    out = ctx.contexts_for_document(
        "slug", recs, canonical_url=URL, version=VER, doc_title="T", source_url=URL,
        contexts_dir=safe_tmp_path, cfg=CFG, client=client,
    )
    assert out["generated"] == 3 and out["pinned"] == 0 and out["missing"] == 0
    assert len(client.messages.calls) == 3
    assert set(out["contexts"]) == {mk_chunk_id(URL, VER, i) for i in range(3)}
    record = ctx.load_pin_file(safe_tmp_path, "slug")
    assert record["model"] == "claude-haiku-4-5"
    assert record["prompt_version"] == ctx.PROMPT_VERSION
    assert len(record["contexts"]) == 3


def test_contexts_for_document_reuses_pins_and_makes_no_api_call(safe_tmp_path):
    recs = _recs("alpha " * 30, "beta " * 30, "gamma " * 30)
    first = _FakeClient()
    ctx.contexts_for_document("slug", recs, canonical_url=URL, version=VER, doc_title="T",
                              source_url=URL, contexts_dir=safe_tmp_path, cfg=CFG, client=first)
    second = _FakeClient()
    out = ctx.contexts_for_document("slug", recs, canonical_url=URL, version=VER, doc_title="T",
                                    source_url=URL, contexts_dir=safe_tmp_path, cfg=CFG, client=second)
    assert second.messages.calls == []            # nothing regenerated
    assert out["pinned"] == 3 and out["generated"] == 0
    assert len(out["contexts"]) == 3


def test_contexts_for_document_without_client_is_offline_and_never_fails(safe_tmp_path):
    recs = _recs("alpha " * 30, "beta " * 30)
    out = ctx.contexts_for_document("slug", recs, canonical_url=URL, version=VER, doc_title="T",
                                    source_url=URL, contexts_dir=safe_tmp_path, cfg=CFG, client=None)
    assert out["contexts"] == {} and out["missing"] == 2 and out["generated"] == 0
    assert out["errors"] == []
    assert not ctx.contexts_path_for(safe_tmp_path, "slug").exists()   # nothing to pin


def test_contexts_for_document_resumes_after_a_failure(safe_tmp_path):
    recs = _recs("alpha " * 30, "beta " * 30, "gamma " * 30)
    flaky = _FakeClient(raise_on={1})             # second chunk blows up
    out = ctx.contexts_for_document("slug", recs, canonical_url=URL, version=VER, doc_title="T",
                                    source_url=URL, contexts_dir=safe_tmp_path, cfg=CFG, client=flaky)
    assert out["generated"] == 2 and len(out["errors"]) == 1 and out["missing"] == 1
    healthy = _FakeClient()
    out2 = ctx.contexts_for_document("slug", recs, canonical_url=URL, version=VER, doc_title="T",
                                     source_url=URL, contexts_dir=safe_tmp_path, cfg=CFG, client=healthy)
    assert len(healthy.messages.calls) == 1       # only the previously-failed chunk
    assert out2["pinned"] == 2 and out2["generated"] == 1 and out2["missing"] == 0


def test_chunks_in_one_window_share_a_byte_identical_cached_prefix(safe_tmp_path):
    recs = _recs(*["body " * 20 for _ in range(4)])
    client = _FakeClient()
    ctx.contexts_for_document("slug", recs, canonical_url=URL, version=VER, doc_title="T",
                              source_url=URL, contexts_dir=safe_tmp_path,
                              cfg={**CFG, "window_target_tokens": 10_000, "window_max_tokens": 10_000},
                              client=client)
    prefixes = {c["messages"][0]["content"][0]["text"] for c in client.messages.calls}
    assert len(prefixes) == 1                     # one window -> one cache entry, 3 reads
    reads = [c for c in client.messages.calls[1:]]
    assert len(reads) == 3


def test_force_regenerates_even_when_pins_are_valid(safe_tmp_path):
    recs = _recs("alpha " * 30)
    ctx.contexts_for_document("slug", recs, canonical_url=URL, version=VER, doc_title="T",
                              source_url=URL, contexts_dir=safe_tmp_path, cfg=CFG, client=_FakeClient())
    again = _FakeClient()
    ctx.contexts_for_document("slug", recs, canonical_url=URL, version=VER, doc_title="T",
                              source_url=URL, contexts_dir=safe_tmp_path, cfg=CFG, client=again, force=True)
    assert len(again.messages.calls) == 1
