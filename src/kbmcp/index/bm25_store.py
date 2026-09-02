"""Lexical BM25 index over the CKB, backed by SQLite FTS5.

Design: docs/superpowers/specs/2026-08-31-phase1-bm25-index-design.md

FTS5 lives inside ckb.sqlite rather than in a sidecar artifact, so the shipped
CKB stays a single file and the index cannot drift from the chunks it indexes.
"""

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from datetime import datetime, timezone

from ..config import load_corpus_config
from ..db import ops
from ..db.schema import create_all_tables

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

    Ordered by chunk_id for determinism, and fields are separated by the ASCII
    unit separator (0x1F), matching the house ID scheme's join convention --
    the corpus's extracted text does not contain a literal 0x1F (measured: 0 of
    2803 live chunks), so this is not the collision-proof guarantee the ID
    scheme has over its own fields, just a separator absent from real text.

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


def invalidate(conn: sqlite3.Connection) -> None:
    """Drop the BM25 fingerprint so is_stale()/query() treat the index as not
    built. A no-op when bm25_meta does not exist (a CKB predating the lexical
    index has nothing to invalidate).

    Ownership: this module owns the index, so anything that rewrites chunks
    out from under it -- e.g. ingest/build.py's --force rebuild -- calls this
    rather than hand-writing `DELETE FROM bm25_meta` itself. Does not touch
    chunks_fts: leaving the stale table behind is harmless, since
    BM25Store._index_exists() requires the bm25_meta row too, and the next
    real build() drops and repopulates chunks_fts wholesale anyway.
    """
    try:
        conn.execute("DELETE FROM bm25_meta")
        conn.commit()
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc).lower():
            return  # nothing to invalidate on a CKB that predates bm25_meta
        raise


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

    def _row_cursor(self) -> sqlite3.Cursor:
        """A cursor with row_factory = sqlite3.Row, independent of the caller's
        connection-level setting.

        __init__ accepts any connection and never sets or checks
        conn.row_factory -- only kbmcp.db.ops.get_db happens to set it. Phase 2
        opens the shipped read-only CKB directly (e.g. a plain
        sqlite3.connect("file:...?mode=ro", uri=True)), which leaves it at the
        tuple default. Setting row_factory on a per-call cursor rather than on
        self.conn keeps this store's dict-style row access working without
        mutating a connection a caller may be sharing elsewhere.
        """
        cur = self.conn.cursor()
        cur.row_factory = sqlite3.Row
        return cur

    def build(self) -> int:
        """Drop and rebuild the FTS5 index over all chunks; return the count.

        Precondition: `conn` must already carry the full CKB schema (i.e.
        `kbmcp.db.schema.create_all_tables(conn)` has been run on it). build()
        owns only the `chunks_fts` table -- it deliberately does not create
        `bm25_meta` or any content table itself, since that split belongs to
        db/schema.py.
        """
        if not _SAFE_TOKENIZE_RE.fullmatch(self.tokenize):
            raise BM25BuildError(
                f"unsafe bm25.tokenize value {self.tokenize!r}: expected only "
                "lowercase letters, digits, underscores and spaces"
            )

        # Clear the fingerprint FIRST, in its own committed transaction. SQLite
        # auto-commits DDL but not DML, so a crash after the DROP/CREATE would
        # otherwise leave chunks_fts empty while the PREVIOUS build's bm25_meta
        # row survives the rollback -- is_stale() would then report a completely
        # empty index as fresh, and every query would silently return no hits.
        # Failing "not built" is recoverable; failing "fresh but empty" is not.
        try:
            self.conn.execute("DELETE FROM bm25_meta")
            self.conn.commit()
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc).lower():
                raise BM25BuildError(
                    "the CKB has no bm25_meta table, so the index fingerprint cannot "
                    "be recorded. This happens on a CKB built before the lexical index "
                    "existed. Call kbmcp.db.schema.create_all_tables(conn) on it first "
                    "(the `python -m kbmcp.index.bm25_store` CLI does this for you)."
                ) from exc
            raise

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
        try:
            self.conn.execute(
                "INSERT INTO bm25_meta (id, chunk_count, chunks_digest, tokenize, "
                "weights_json, schema_version, built_at) VALUES (1, ?, ?, ?, ?, ?, ?)",
                (count, digest, self.tokenize, json.dumps(self.weights, sort_keys=True),
                 SCHEMA_VERSION, _now()),
            )
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc).lower():
                raise BM25BuildError(
                    "the CKB has no bm25_meta table, so the index fingerprint cannot "
                    "be recorded. This happens on a CKB built before the lexical index "
                    "existed. Call kbmcp.db.schema.create_all_tables(conn) on it first "
                    "(the `python -m kbmcp.index.bm25_store` CLI does this for you)."
                ) from exc
            raise
        self.conn.commit()
        return count

    def _index_exists(self) -> bool:
        """True only when a COMPLETE index is present.

        Presence of chunks_fts alone is not enough: a build that fails after the
        DDL but before the fingerprint write leaves the table present but empty,
        because SQLite auto-commits DDL outside the surrounding transaction. A
        presence-only check would let query() return zero hits for a broken index
        instead of raising.
        """
        has_table = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'chunks_fts'"
        ).fetchone() is not None
        if not has_table:
            return False
        try:
            return self.conn.execute(
                "SELECT 1 FROM bm25_meta WHERE id = 1").fetchone() is not None
        except sqlite3.OperationalError:
            return False   # bm25_meta absent entirely -> not built

    def is_stale(self) -> bool:
        """True when the index does not match the chunks currently in the DB.

        Guards the desync class that bit M5: an index built from a superseded
        chunk set returns chunk_ids that no longer exist, silently.
        """
        if not self._index_exists():
            return True
        row = self._row_cursor().execute(
            "SELECT * FROM bm25_meta WHERE id = 1").fetchone()
        if row is None:
            return True
        count, digest = corpus_digest(self.conn)
        return (
            row["chunk_count"] != count
            or row["chunks_digest"] != digest
            or row["tokenize"] != self.tokenize
            or json.loads(row["weights_json"]) != self.weights
            or row["schema_version"] != SCHEMA_VERSION
        )

    def query(self, text: str, top_k=None) -> list:
        """Top lexical matches as (chunk_id, score), best first.

        Scores are returned POSITIVE and DESCENDING. FTS5's bm25() is negative
        with more-negative meaning a better match; negating here keeps the
        contract downstream RRF and not-found code expects.
        """
        if not self._index_exists():
            raise BM25NotBuiltError(
                "BM25 index not built: run BM25Store.build() (or "
                "`python -m kbmcp.index.bm25_store`) first"
            )
        match = to_match_query(text)
        if not match:
            return []
        limit = self.top_k if top_k is None else top_k
        rows = self._row_cursor().execute(
            "SELECT chunk_id, bm25(chunks_fts, 0.0, ?, ?) AS score FROM chunks_fts "
            "WHERE chunks_fts MATCH ? ORDER BY score ASC, chunk_id ASC LIMIT ?",
            (float(self.weights.get("context", 1.0)),
             float(self.weights.get("text", 2.0)), match, limit),
        ).fetchall()
        return [(r["chunk_id"], -float(r["score"])) for r in rows]


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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m kbmcp.index.bm25_store")
    p.add_argument("--ckb", default="ckb/ckb.sqlite")
    p.add_argument("--config", default="config/corpus_config.yaml")
    a = p.parse_args(argv)

    conn = ops.get_db(a.ckb)
    try:
        create_all_tables(conn)
        count = BM25Store(conn, load_corpus_config(a.config).bm25).build()
        print(f"bm25: indexed {count} chunks", file=sys.stderr)
    except BM25BuildError as exc:
        print(f"bm25: build failed: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
