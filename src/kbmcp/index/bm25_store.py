"""Lexical BM25 index over the CKB, backed by SQLite FTS5.

Design: docs/superpowers/specs/2026-08-31-phase1-bm25-index-design.md

FTS5 lives inside ckb.sqlite rather than in a sidecar artifact, so the shipped
CKB stays a single file and the index cannot drift from the chunks it indexes.
"""

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

SCHEMA_VERSION = 1

# The tokenizer string is interpolated into DDL (FTS5 does not accept it as a
# bound parameter), so it is allowlisted rather than trusted.
_SAFE_TOKENIZE_RE = re.compile(r"[a-z0-9_ ]+")

DEFAULT_TOKENIZE = "porter unicode61"
DEFAULT_WEIGHTS = {"context": 1.0, "text": 2.0}
DEFAULT_TOP_K = 50


class BM25BuildError(RuntimeError):
    """Raised when the index cannot be built."""


class BM25NotBuiltError(RuntimeError):
    """Raised when querying before build()."""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def corpus_digest(conn) -> tuple:
    """(chunk_count, sha256 hex) over every chunk's id, context and text.

    Ordered by chunk_id for determinism, and fields are joined with the ASCII
    unit separator, matching the house ID scheme's collision-safe convention.

    Covers CONTENT, not just ids: chunk_id is sha256(url, version, chunk_index)
    and does not depend on text, so an id-only digest would miss a re-parse or a
    contextualization pass.
    """
    h = hashlib.sha256()
    n = 0
    for cid, ctx, txt in conn.execute(
        "SELECT chunk_id, COALESCE(context, ''), text FROM chunks ORDER BY chunk_id"
    ):
        for part in (cid, ctx, txt):
            h.update(part.encode("utf-8"))
            h.update(b"\x1f")
        n += 1
    return n, h.hexdigest()


class BM25Store:
    """FTS5-backed lexical index over the CKB's chunks.

    The index lives in the same database as the chunks it indexes and is always
    rebuilt wholesale, so it cannot partially diverge from them.
    """

    def __init__(self, conn: sqlite3.Connection, cfg: dict) -> None:
        cfg = cfg or {}
        self.conn = conn
        self.tokenize = cfg.get("tokenize", DEFAULT_TOKENIZE)
        self.weights = dict(cfg.get("weights", DEFAULT_WEIGHTS))
        self.top_k = cfg.get("top_k", DEFAULT_TOP_K)

    def build(self) -> int:
        """Drop and rebuild the FTS5 index over all chunks; return the count."""
        if not _SAFE_TOKENIZE_RE.fullmatch(self.tokenize):
            raise BM25BuildError(
                f"unsafe bm25.tokenize value {self.tokenize!r}: expected only "
                "lowercase letters, digits, underscores and spaces"
            )
        self.conn.execute("DROP TABLE IF EXISTS chunks_fts")
        try:
            self.conn.execute(
                "CREATE VIRTUAL TABLE chunks_fts USING fts5("
                "chunk_id UNINDEXED, context, text, "
                f"tokenize='{self.tokenize}')"
            )
        except sqlite3.OperationalError as exc:
            if "no such module" in str(exc).lower():
                raise BM25BuildError(
                    "SQLite FTS5 is not available in this Python's sqlite3 build, "
                    "so the lexical index cannot be created. FTS5 ships with "
                    "standard CPython builds; a custom or minimal SQLite may omit "
                    "it. Reinstall Python from python.org or rebuild SQLite with "
                    "-DSQLITE_ENABLE_FTS5."
                ) from exc
            raise
        self.conn.execute(
            "INSERT INTO chunks_fts (chunk_id, context, text) "
            "SELECT chunk_id, COALESCE(context, ''), text FROM chunks "
            "ORDER BY chunk_id"
        )
        count, digest = corpus_digest(self.conn)
        self.conn.execute("DELETE FROM bm25_meta")
        self.conn.execute(
            "INSERT INTO bm25_meta (id, chunk_count, chunks_digest, tokenize, "
            "weights_json, schema_version, built_at) VALUES (1, ?, ?, ?, ?, ?, ?)",
            (count, digest, self.tokenize, json.dumps(self.weights, sort_keys=True),
             SCHEMA_VERSION, _now()),
        )
        self.conn.commit()
        return count


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
