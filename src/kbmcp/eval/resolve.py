"""Gold resolution helper: turn a verbatim phrase into a `GoldRef`.

Authoring ~105 benchmark items by hand-copying chunk_ids out of SQL would be
error-prone and slow. This module is the tool the authoring tasks use instead:
find the chunk(s) a phrase appears in, and fail loudly (rather than guessing)
when the phrase doesn't identify exactly one chunk.

Match is a plain substring (`instr(text, ?) > 0`), not full-text search: an
anchor must be a *verbatim* substring of the chunk text, because
`verify_gold` later checks exactly that (`ref.anchor in chunk["text"]`). An
FTS match would accept anchors verify_gold would then reject.
"""

import sqlite3

from ..ingest.graph import doc_id_for_slug
from . import models

_SNIPPET_RADIUS = 80


class ResolveError(ValueError):
    """Raised when a phrase does not identify exactly one chunk."""


def find_chunks(
    conn: sqlite3.Connection,
    phrase: str,
    *,
    slug: str | None = None,
    by_slug: dict | None = None,
    raw_dir=None,
    limit: int = 10,
) -> list[dict]:
    """All chunks whose text contains `phrase` as a verbatim substring.

    Scoped to one document when `slug` is given (resolved to a doc_id via
    `graph.doc_id_for_slug` -- the one place that mapping is computed).
    """
    doc_id_to_slug = None
    if by_slug is not None:
        doc_id_to_slug = {
            doc_id_for_slug(s, by_slug, raw_dir): s for s in by_slug
        }

    if slug is not None:
        target_doc_id = doc_id_for_slug(slug, by_slug, raw_dir)
        rows = conn.execute(
            "SELECT chunk_id, doc_id, text FROM chunks "
            "WHERE instr(text, ?) > 0 AND doc_id = ? ORDER BY chunk_id LIMIT ?",
            (phrase, target_doc_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT chunk_id, doc_id, text FROM chunks "
            "WHERE instr(text, ?) > 0 ORDER BY chunk_id LIMIT ?",
            (phrase, limit),
        ).fetchall()

    hits = []
    for row in rows:
        text = row["text"]
        idx = text.find(phrase)
        start = max(0, idx - _SNIPPET_RADIUS)
        end = min(len(text), idx + len(phrase) + _SNIPPET_RADIUS)
        resolved_slug = doc_id_to_slug.get(row["doc_id"]) if doc_id_to_slug else None
        hits.append({
            "chunk_id": row["chunk_id"],
            "doc_id": row["doc_id"],
            "slug": resolved_slug,
            "snippet": text[start:end],
        })
    return hits


def gold_ref_for(conn: sqlite3.Connection, slug: str, phrase: str, by_slug: dict,
                  raw_dir) -> "models.GoldRef":
    """Resolve `phrase` within document `slug` to a `GoldRef`.

    Raises `ResolveError` when the phrase matches zero chunks (the query's
    premise is wrong -- fix the query, not the anchor) or more than one
    (the anchor doesn't identify a single chunk -- lengthen it).
    """
    hits = find_chunks(conn, phrase, slug=slug, by_slug=by_slug, raw_dir=raw_dir)
    if len(hits) == 0:
        raise ResolveError(f"phrase {phrase!r} matched no chunks in {slug!r}")
    if len(hits) > 1:
        chunk_ids = [h["chunk_id"] for h in hits]
        raise ResolveError(
            f"phrase {phrase!r} is ambiguous in {slug!r}: matched {len(hits)} "
            f"chunks {chunk_ids!r}; lengthen the anchor"
        )
    return models.GoldRef(doc=slug, chunk_id=hits[0]["chunk_id"], anchor=phrase)
