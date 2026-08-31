"""Lexical BM25 index over the CKB, backed by SQLite FTS5.

Design: docs/superpowers/specs/2026-08-31-phase1-bm25-index-design.md

FTS5 lives inside ckb.sqlite rather than in a sidecar artifact, so the shipped
CKB stays a single file and the index cannot drift from the chunks it indexes.
"""

import re

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def to_match_query(text: str) -> str:
    """Turn free user text into a safe FTS5 MATCH expression.

    A raw query string is MATCH *syntax*: bare punctuation, a trailing `*`, or
    a bare AND/OR/NOT is a syntax error, and unbalanced quotes are worse. Each
    `\\w+` run is extracted and double-quoted, then OR-joined for recall.

    Quoting is injection-safe by construction: `\\w` cannot match `"`, so no
    token can close its own quote.

    Returns "" when the text has no word characters at all; callers must treat
    that as "no query" rather than passing it to MATCH.
    """
    tokens = _TOKEN_RE.findall(text or "")
    return " OR ".join('"%s"' % t for t in tokens)
