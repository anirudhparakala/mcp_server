"""Benchmark schema — the ground truth Phase 3 measures retrieval against.

Design: docs/superpowers/specs/2026-09-16-phase1-benchmark-design.md

Departs from the RAG reference because we author our own labels rather than
consuming BEIR/HotpotQA: gold is CHUNK-level (doc-level is too coarse to measure
chunk retrieval), `tier` is validated rather than a free-form category string,
and generation-dependent fields are gone (Decision 1 is retrieval-only).
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

TIERS = ("T1", "T2", "T3", "T4", "T5", "T6", "T7")


class BenchmarkError(ValueError):
    """Raised when a benchmark item or file is malformed."""


@dataclass(frozen=True)
class GoldRef:
    """One chunk that answers a query.

    `chunk_id` is what eval scores against -- exact and fast. `anchor` is a short
    verbatim quote that must appear in that chunk, and is the ground truth of
    record: chunk_id depends on version and chunk_index, so a re-fetch or
    re-chunk changes it, and without the anchor a drifted label would silently
    score against the wrong chunk.
    """

    doc: str
    chunk_id: str
    anchor: str


@dataclass(frozen=True)
class BenchmarkItem:
    id: str
    tier: str
    query: str
    answerable: bool
    gold: tuple = ()
    distractor_docs: tuple = ()
    collision_term: str | None = None
    notes: str = ""


def _require(d: dict, key: str, where: str):
    value = d.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise BenchmarkError(f"{where}: missing or empty required field {key!r}")
    return value


def _require_list(d: dict, key: str, where: str) -> list:
    """A string is iterable, so a bare `"distractor_docs": "doc1"` would silently
    become ('d','o','c','1'). Anything not a list is rejected by type, not by
    whether it happens to iterate."""
    value = d.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise BenchmarkError(
            f"{where}: {key!r} must be a list, got {type(value).__name__} "
            f"({value!r}). A single value still needs to be a one-element list."
        )
    return value


def item_from_dict(d: dict) -> BenchmarkItem:
    """Validate and build one item. Raises BenchmarkError rather than admitting a
    malformed item -- a silently-wrong benchmark corrupts every published number."""
    ident = d.get("id") or "<no id>"
    tier = _require(d, "tier", ident)
    if tier not in TIERS:
        raise BenchmarkError(f"{ident}: unknown tier {tier!r}; expected one of {TIERS}")
    gold = []
    for i, g in enumerate(_require_list(d, "gold", ident)):
        where = f"{ident} gold[{i}]"
        if not isinstance(g, dict):
            raise BenchmarkError(f"{ident} gold[{i}]: must be an object with doc/chunk_id/anchor, got {type(g).__name__}")
        gold.append(GoldRef(doc=_require(g, "doc", where),
                            chunk_id=_require(g, "chunk_id", where),
                            anchor=_require(g, "anchor", where)))
    answerable = d.get("answerable", True)
    if not isinstance(answerable, bool):
        raise BenchmarkError(
            f"{ident}: 'answerable' must be a JSON boolean (true/false), got "
            f"{type(answerable).__name__} ({answerable!r}). A quoted \"false\" is "
            "truthy and would silently flip the item to answerable."
        )
    return BenchmarkItem(
        id=_require(d, "id", ident),
        tier=tier,
        query=_require(d, "query", ident),
        answerable=answerable,
        gold=tuple(gold),
        distractor_docs=tuple(_require_list(d, "distractor_docs", ident)),
        collision_term=d.get("collision_term"),
        notes=d.get("notes", ""),
    )


def item_to_dict(item: BenchmarkItem) -> dict:
    out = asdict(item)
    out["gold"] = [asdict(g) for g in item.gold]
    out["distractor_docs"] = list(item.distractor_docs)
    return out


def load_queries(path) -> list:
    """Parse a JSONL benchmark file. A malformed line names its line number."""
    items = []
    text = Path(path).read_text(encoding="utf-8")
    for n, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BenchmarkError(f"{path} line {n}: not valid JSON ({exc})") from exc
        items.append(item_from_dict(raw))
    return items


def _sort_key(item):
    """Sort by tier then numeric suffix, so T1-2 precedes T1-10 regardless of
    zero-padding. Plain lexicographic order would interleave them and make
    later diffs noisy."""
    tier, _, rest = item.id.partition("-")
    return (tier, int(rest) if rest.isdigit() else 0, item.id)


def dump_queries(items, path) -> None:
    """Write JSONL, one item per line, sorted by id for a stable diff."""
    lines = [json.dumps(item_to_dict(i), ensure_ascii=False, sort_keys=True)
             for i in sorted(items, key=_sort_key)]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
