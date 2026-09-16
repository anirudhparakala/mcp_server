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


def item_from_dict(d: dict) -> BenchmarkItem:
    """Validate and build one item. Raises BenchmarkError rather than admitting a
    malformed item -- a silently-wrong benchmark corrupts every published number."""
    ident = d.get("id") or "<no id>"
    tier = _require(d, "tier", ident)
    if tier not in TIERS:
        raise BenchmarkError(f"{ident}: unknown tier {tier!r}; expected one of {TIERS}")
    gold = []
    for i, g in enumerate(d.get("gold") or []):
        where = f"{ident} gold[{i}]"
        gold.append(GoldRef(doc=_require(g, "doc", where),
                            chunk_id=_require(g, "chunk_id", where),
                            anchor=_require(g, "anchor", where)))
    return BenchmarkItem(
        id=_require(d, "id", ident),
        tier=tier,
        query=_require(d, "query", ident),
        answerable=bool(d.get("answerable", True)),
        gold=tuple(gold),
        distractor_docs=tuple(d.get("distractor_docs") or ()),
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


def dump_queries(items, path) -> None:
    """Write JSONL, one item per line, sorted by id for a stable diff."""
    lines = [json.dumps(item_to_dict(i), ensure_ascii=False, sort_keys=True)
             for i in sorted(items, key=lambda x: x.id)]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
