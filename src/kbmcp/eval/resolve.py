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


def _scope_where(phrase: str, slug, by_slug, raw_dir):
    """The WHERE clause and params shared by `find_chunks` and `count_chunks`.

    Kept as one function so the sampled list and the true total always
    describe the same match set -- a hand-duplicated WHERE clause in each
    caller could drift and make the count describe a different query than
    the sample it accompanies.
    """
    if slug is not None:
        target_doc_id = doc_id_for_slug(slug, by_slug, raw_dir)
        return "instr(text, ?) > 0 AND doc_id = ?", (phrase, target_doc_id)
    return "instr(text, ?) > 0", (phrase,)


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

    Capped at `limit` -- fine for browsing, but callers that need the true
    match count (e.g. an ambiguity message) must use `count_chunks` instead.
    """
    doc_id_to_slug = None
    if by_slug is not None:
        doc_id_to_slug = {
            doc_id_for_slug(s, by_slug, raw_dir): s for s in by_slug
        }

    where, params = _scope_where(phrase, slug, by_slug, raw_dir)
    rows = conn.execute(
        f"SELECT chunk_id, doc_id, text FROM chunks WHERE {where} "
        "ORDER BY chunk_id LIMIT ?",
        (*params, limit),
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


def count_chunks(conn: sqlite3.Connection, phrase: str, *, slug: str | None = None,
                  by_slug: dict | None = None, raw_dir=None) -> int:
    """Total chunks containing `phrase` -- uncapped.

    find_chunks caps its result list, which is fine for browsing but wrong for
    an ambiguity message: an author uses the count to judge how much to lengthen
    an anchor, and "matched 10" when the truth is 47 understates the problem.
    """
    where, params = _scope_where(phrase, slug, by_slug, raw_dir)
    row = conn.execute(
        f"SELECT COUNT(*) FROM chunks WHERE {where}", params
    ).fetchone()
    return row[0]


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
        total = count_chunks(conn, phrase, slug=slug, by_slug=by_slug, raw_dir=raw_dir)
        sample = [h["chunk_id"] for h in hits]
        raise ResolveError(
            f"phrase {phrase!r} is ambiguous in {slug!r}: matched {total} chunks "
            f"(showing {len(sample)}): {sample}. Lengthen the anchor until it "
            "identifies exactly one chunk."
        )
    return models.GoldRef(doc=slug, chunk_id=hits[0]["chunk_id"], anchor=phrase)
