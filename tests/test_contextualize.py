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
