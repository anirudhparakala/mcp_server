"""Tests for the benchmark schema (eval/models.py).

Hermetic: pure dataclasses and JSONL round-tripping; no CKB, no corpus.
"""

import json

import pytest

from kbmcp.eval import models as m


def _item(**kw):
    d = dict(id="T1-001", tier="T1", query="what is x?", answerable=True,
             gold=[{"doc": "d1", "chunk_id": "c0", "anchor": "x is"}],
             distractor_docs=[], collision_term=None, notes="baseline lookup")
    d.update(kw)
    return d


def test_round_trips_through_jsonl(safe_tmp_path):
    items = [m.item_from_dict(_item()),
             m.item_from_dict(_item(id="T7-001", tier="T7", answerable=False, gold=[]))]
    path = safe_tmp_path / "q.jsonl"
    m.dump_queries(items, path)
    assert m.load_queries(path) == items


def test_one_item_per_line(safe_tmp_path):
    """Line-oriented so a diff shows which items changed and a malformed item is
    one bad line, not an unparseable file."""
    items = [m.item_from_dict(_item(id=f"T1-{i:03d}")) for i in range(5)]
    path = safe_tmp_path / "q.jsonl"
    m.dump_queries(items, path)
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 5


def test_a_malformed_line_is_reported_with_its_line_number(safe_tmp_path):
    path = safe_tmp_path / "q.jsonl"
    path.write_text('{"id": "T1-001", "tier": "T1", "query": "q", "answerable": true, "gold": []}\nNOT JSON\n', encoding="utf-8")
    with pytest.raises(m.BenchmarkError) as exc:
        m.load_queries(path)
    assert "line 2" in str(exc.value)


def test_an_unknown_tier_is_rejected():
    with pytest.raises(m.BenchmarkError) as exc:
        m.item_from_dict(_item(tier="T9"))
    assert "T9" in str(exc.value)


def test_gold_refs_require_all_three_fields():
    for bad in ({"doc": "d1", "chunk_id": "c0"},
                {"doc": "d1", "anchor": "x"},
                {"chunk_id": "c0", "anchor": "x"}):
        with pytest.raises(m.BenchmarkError):
            m.item_from_dict(_item(gold=[bad]))


def test_an_empty_anchor_is_rejected():
    """An empty anchor would vacuously 'appear' in every chunk, disabling the
    drift check that is the whole reason anchors exist."""
    with pytest.raises(m.BenchmarkError):
        m.item_from_dict(_item(gold=[{"doc": "d1", "chunk_id": "c0", "anchor": "  "}]))


def test_items_are_frozen():
    """FrozenInstanceError specifically -- a bare Exception would also pass on an
    unrelated AttributeError from a typo in the attribute name."""
    import dataclasses

    item = m.item_from_dict(_item())
    with pytest.raises(dataclasses.FrozenInstanceError):
        item.tier = "T2"


def test_tiers_constant_is_the_seven_agreed_tiers():
    assert m.TIERS == ("T1", "T2", "T3", "T4", "T5", "T6", "T7")


def test_a_non_list_gold_is_rejected_by_type():
    for bad in ({"doc": "d1", "chunk_id": "c0", "anchor": "x"}, "c0", 3):
        with pytest.raises(m.BenchmarkError):
            m.item_from_dict(_item(gold=bad))


def test_a_non_dict_gold_entry_is_rejected():
    with pytest.raises(m.BenchmarkError):
        m.item_from_dict(_item(gold=["c0"]))


def test_a_string_distractor_docs_is_rejected_not_shredded():
    """"doc1" would otherwise become ('d','o','c','1') with no error."""
    with pytest.raises(m.BenchmarkError):
        m.item_from_dict(_item(distractor_docs="case-hadley-v-baxendale"))


def test_a_quoted_boolean_answerable_is_rejected():
    """A quoted "false" is truthy and would silently flip a T7 item."""
    with pytest.raises(m.BenchmarkError):
        m.item_from_dict(_item(answerable="false"))


def test_ids_sort_numerically_within_a_tier(safe_tmp_path):
    items = [m.item_from_dict(_item(id=f"T1-{n}")) for n in (10, 2, 1)]
    path = safe_tmp_path / "q.jsonl"
    m.dump_queries(items, path)
    ids = [json.loads(l)["id"] for l in path.read_text(encoding="utf-8").splitlines()]
    assert ids == ["T1-1", "T1-2", "T1-10"]
