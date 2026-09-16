"""Tests for the gold verification gate (eval/verify_gold.py).

Hermetic: tiny CKBs under safe_tmp_path. Never touches ckb/ckb.sqlite.
"""

import json

import pytest

from kbmcp.db import ops
from kbmcp.db.schema import create_all_tables
from kbmcp.eval import verify_gold as vg
from kbmcp.models.ids import doc_id as mk_doc_id

DID = mk_doc_id("u", "v1")


def _ckb(path, text="the quick brown fox jumps"):
    # Several tests call _run() (and hence _ckb()) twice against the same
    # safe_tmp_path/"t.sqlite" file (e.g. once per min_per_tier value). Rebuild
    # fresh each time rather than INSERTing into a file that may already hold
    # this same source/doc/chunk -- otherwise the second call hits a stale
    # sqlite3.IntegrityError on sources.canonical_url that has nothing to do
    # with what the test is actually checking.
    if path.exists():
        path.unlink()
    conn = ops.get_db(path)
    try:
        create_all_tables(conn)
        ops.insert_source(conn, canonical_url="u", url_original="u", domain="d",
                          format="html", license="x", license_ok=True, version="v1")
        ops.insert_doc(conn, doc_id=DID, canonical_url="u", domain="d", format="html")
        ops.insert_chunk(conn, chunk_id="c0", doc_id=DID, chunk_index=0,
                         chunk_type="text", text=text)
    finally:
        conn.close()


def _manifest(path, slugs=("u",)):
    """Each slug gets a DISTINCT url on purpose: doc_id is sha256(url, version,
    ...), so reusing one url would make every slug resolve to the SAME doc_id and
    silently defeat the doc_matches check this file is testing."""
    rows = "".join(
        f"- doc_id: {s}\n  url: {s}\n  domain: d\n  format: html\n"
        f"  license: x\n  license_ok: true\n  version: v1\n" for s in slugs)
    path.write_text(rows, encoding="utf-8")


def _raw(dirpath, slugs=("u",)):
    dirpath.mkdir(exist_ok=True)
    for s in slugs:
        (dirpath / f"{s}.meta.json").write_text('{"resolved_version": "v1"}',
                                                encoding="utf-8")


def _write(path, items):
    path.write_text("\n".join(json.dumps(i) for i in items) + "\n", encoding="utf-8")


def _ok_item(**kw):
    d = dict(id="T1-001", tier="T1", query="q", answerable=True,
             gold=[{"doc": "u", "chunk_id": "c0", "anchor": "quick brown"}],
             distractor_docs=[], collision_term=None, notes="n")
    d.update(kw)
    return d


def _run(safe_tmp_path, items, min_per_tier=0):
    ckb = safe_tmp_path / "t.sqlite"
    _ckb(ckb)
    man = safe_tmp_path / "m.yaml"
    _manifest(man)
    raw = safe_tmp_path / "raw"
    _raw(raw)
    q = safe_tmp_path / "q.jsonl"
    _write(q, items)
    return vg.verify_gold(str(ckb), str(q), str(man), str(raw), min_per_tier=min_per_tier)


def test_a_valid_item_passes(safe_tmp_path):
    assert vg.all_passed(_run(safe_tmp_path, [_ok_item()])) is True


def test_a_chunk_id_absent_from_the_ckb_fails(safe_tmp_path):
    item = _ok_item(gold=[{"doc": "u", "chunk_id": "MISSING", "anchor": "quick"}])
    results = _run(safe_tmp_path, [item])
    assert vg.all_passed(results) is False
    assert any("MISSING" in r.detail for r in results if not r.passed)


def test_an_anchor_absent_from_its_chunk_fails(safe_tmp_path):
    """The drift guard: a rebuild that moves text must be caught here, not
    discovered later as inexplicably bad recall."""
    item = _ok_item(gold=[{"doc": "u", "chunk_id": "c0", "anchor": "NOT IN THE TEXT"}])
    results = _run(safe_tmp_path, [item])
    assert vg.all_passed(results) is False
    assert any("anchor" in r.check for r in results if not r.passed)


def test_a_gold_doc_that_disagrees_with_the_chunk_fails(safe_tmp_path):
    _manifest(safe_tmp_path / "m2.yaml", slugs=("u", "other"))
    item = _ok_item(gold=[{"doc": "other", "chunk_id": "c0", "anchor": "quick brown"}])
    ckb = safe_tmp_path / "t.sqlite"
    _ckb(ckb)
    raw = safe_tmp_path / "raw"
    _raw(raw, slugs=("u", "other"))
    q = safe_tmp_path / "q.jsonl"
    _write(q, [item])
    results = vg.verify_gold(str(ckb), str(q), str(safe_tmp_path / "m2.yaml"),
                             str(raw), min_per_tier=0)
    assert vg.all_passed(results) is False


def test_duplicate_ids_fail(safe_tmp_path):
    results = _run(safe_tmp_path, [_ok_item(), _ok_item()])
    assert vg.all_passed(results) is False


def test_an_id_prefix_that_disagrees_with_its_tier_fails(safe_tmp_path):
    results = _run(safe_tmp_path, [_ok_item(id="T2-001", tier="T1")])
    assert vg.all_passed(results) is False


def test_t7_must_be_unanswerable_with_empty_gold(safe_tmp_path):
    bad_gold = _ok_item(id="T7-001", tier="T7", answerable=False)
    bad_answerable = _ok_item(id="T7-002", tier="T7", answerable=True, gold=[])
    for item in (bad_gold, bad_answerable):
        assert vg.all_passed(_run(safe_tmp_path, [item])) is False


def test_a_non_t7_item_with_empty_gold_fails(safe_tmp_path):
    assert vg.all_passed(_run(safe_tmp_path, [_ok_item(gold=[])])) is False


def test_t4_and_t5_need_two_gold_refs_in_two_distinct_documents(safe_tmp_path):
    """Both refs in the SAME document is not multi-hop and not a conflict."""
    same_doc = [{"doc": "u", "chunk_id": "c0", "anchor": "quick"},
                {"doc": "u", "chunk_id": "c0", "anchor": "brown"}]
    for tier, ident in (("T4", "T4-001"), ("T5", "T5-001")):
        results = _run(safe_tmp_path, [_ok_item(id=ident, tier=tier, gold=same_doc)])
        assert vg.all_passed(results) is False, f"{tier} accepted a single-document pair"


def test_t2_needs_a_distractor_and_t6_needs_a_collision_term(safe_tmp_path):
    t2 = _ok_item(id="T2-001", tier="T2", distractor_docs=[])
    t6 = _ok_item(id="T6-001", tier="T6", collision_term=None)
    for item in (t2, t6):
        assert vg.all_passed(_run(safe_tmp_path, [item])) is False


def test_a_distractor_slug_absent_from_the_manifest_fails(safe_tmp_path):
    item = _ok_item(id="T2-001", tier="T2", distractor_docs=["no-such-slug"])
    assert vg.all_passed(_run(safe_tmp_path, [item])) is False


def test_the_min_per_tier_rule_is_enforced(safe_tmp_path):
    """min_per_tier=0 disables the check, which is how the authoring tasks build
    the file up one tier at a time."""
    items = [_ok_item(id=f"T1-{i:03d}") for i in range(3)]
    assert vg.all_passed(_run(safe_tmp_path, items, min_per_tier=15)) is False
    assert vg.all_passed(_run(safe_tmp_path, items, min_per_tier=0)) is True


def test_a_tier_entirely_absent_from_the_file_fails(safe_tmp_path):
    """The case the previous implementation missed: 15 T1 items and nothing else
    passed at min_per_tier=15, because only tiers PRESENT in the file were
    counted. The exit criterion is >=15 in every tier."""
    items = [_ok_item(id=f"T1-{i:03d}") for i in range(15)]
    results = _run(safe_tmp_path, items, min_per_tier=15)
    assert vg.all_passed(results) is False
    failed_tiers = {r.detail.split(":")[0] for r in results
                    if not r.passed and r.check == "tier_count"}
    assert "T7" in failed_tiers, "an absent T7 must be named in the failures"


def test_cli_returns_1_on_failure_and_writes_nothing_to_stdout(safe_tmp_path, capsys):
    ckb = safe_tmp_path / "t.sqlite"
    _ckb(ckb)
    man = safe_tmp_path / "m.yaml"
    _manifest(man)
    raw = safe_tmp_path / "raw"
    _raw(raw)
    q = safe_tmp_path / "q.jsonl"
    _write(q, [_ok_item(gold=[{"doc": "u", "chunk_id": "MISSING", "anchor": "x"}])])
    rc = vg.main(["--ckb", str(ckb), "--queries", str(q), "--manifest", str(man),
                  "--raw-dir", str(raw), "--min-per-tier", "0"])
    assert rc == 1
    assert capsys.readouterr().out == ""
