"""CKB SQLite operations (document side: sources, docs, chunks).

JSON-valued columns are serialized with json.dumps here so callers pass native
Python lists/dicts. Edge ops live in the cross-ref-graph milestone.
"""

import hashlib
import json
import sqlite3
from typing import Any, Optional

from .schema import create_all_tables


def get_db(path) -> sqlite3.Connection:
    """Open a CKB connection with foreign keys enforced and Row access."""
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def insert_source(
    conn: sqlite3.Connection,
    *,
    canonical_url: str,
    url_original: str,
    domain: str,
    format: str,
    license: str,
    license_ok: bool,
    version: str,
    raw_cache: Optional[str] = None,
    content_hash: Optional[str] = None,
    tier_roles: Optional[list] = None,
    collision_terms: Optional[list] = None,
    references: Optional[list] = None,
    conflict_with: Optional[list] = None,
    rationale: Optional[str] = None,
    fetched_at: Optional[str] = None,
) -> None:
    conn.execute(
        "INSERT INTO sources (canonical_url, url_original, domain, format, license, "
        "license_ok, version, raw_cache, content_hash, tier_roles_json, "
        "collision_terms_json, references_json, conflict_with_json, rationale, fetched_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            canonical_url, url_original, domain, format, license, int(license_ok), version,
            raw_cache, content_hash, json.dumps(tier_roles or []),
            json.dumps(collision_terms or []), json.dumps(references or []),
            json.dumps(conflict_with or []), rationale, fetched_at,
        ),
    )
    conn.commit()


def insert_doc(
    conn: sqlite3.Connection,
    *,
    doc_id: str,
    canonical_url: str,
    domain: str,
    format: str,
    title: Optional[str] = None,
    doc_type: Optional[str] = None,
    structure: Any = None,
    created_at: Optional[str] = None,
) -> None:
    conn.execute(
        "INSERT INTO docs (doc_id, canonical_url, title, domain, format, doc_type, "
        "structure_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            doc_id, canonical_url, title, domain, format, doc_type,
            json.dumps(structure) if structure is not None else None, created_at,
        ),
    )
    conn.commit()


def insert_chunk(
    conn: sqlite3.Connection,
    *,
    chunk_id: str,
    doc_id: str,
    chunk_index: int,
    text: str,
    chunk_type: str,
    context: Optional[str] = None,
    heading_path: Optional[list] = None,
    table: Any = None,
    citation_anchors: Optional[dict] = None,
) -> None:
    conn.execute(
        "INSERT INTO chunks (chunk_id, doc_id, chunk_index, text, context, "
        "heading_path_json, chunk_type, table_json, citation_anchors_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            chunk_id, doc_id, chunk_index, text, context,
            json.dumps(heading_path or []), chunk_type,
            json.dumps(table) if table is not None else None,
            json.dumps(citation_anchors or {}),
        ),
    )
    conn.commit()


def chunk_id_exists(conn: sqlite3.Connection, chunk_id: str) -> bool:
    row = conn.execute("SELECT 1 FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
    return row is not None


def get_chunks_for_doc(conn: sqlite3.Connection, doc_id: str) -> list:
    """All chunks for a doc, ordered by chunk_index, JSON columns parsed (see get_chunk)."""
    rows = conn.execute(
        "SELECT * FROM chunks WHERE doc_id = ? ORDER BY chunk_index", (doc_id,)
    ).fetchall()
    return [get_chunk(conn, r["chunk_id"]) for r in rows]


def count_rows(conn: sqlite3.Connection, table: str) -> int:
    if table not in {"sources", "docs", "chunks", "edges", "ingest_runs"}:
        raise ValueError(f"unknown table: {table}")
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def get_chunk(conn: sqlite3.Connection, chunk_id: str) -> Optional[dict]:
    """Fetch a chunk by ID; _json columns are returned parsed (list/dict), not as raw strings."""
    row = conn.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
    if row is None:
        return None
    result = dict(row)
    # Deserialize JSON columns and normalize to clean key names
    result["heading_path"] = json.loads(result.pop("heading_path_json"))
    if result["table_json"] is not None:
        result["table"] = json.loads(result.pop("table_json"))
    else:
        result.pop("table_json")
        result["table"] = None
    result["citation_anchors"] = json.loads(result.pop("citation_anchors_json"))
    return result


def edge_id(from_chunk: str, to_chunk, edge_type: str, provenance) -> str:
    """Deterministic edge ID so re-running the graph pass regenerates, not duplicates.

    Fields are serialized with JSON and then hashed so no field value can forge a boundary,
    even if it contains control characters. JSON escaping ensures the four-tuple is injective.
    """
    payload = json.dumps([from_chunk, to_chunk, edge_type, provenance], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def insert_edge(
    conn: sqlite3.Connection,
    *,
    from_chunk: str,
    to_chunk: Optional[str] = None,
    to_external_ref: Optional[str] = None,
    edge_type: str,
    provenance: Optional[str] = None,
    confidence: Optional[float] = None,
    created_at: Optional[str] = None,
) -> str:
    """Insert one edge, returning its deterministic ID. Idempotent by edge_id."""
    eid = edge_id(from_chunk, to_chunk, edge_type, provenance)
    conn.execute(
        "INSERT OR REPLACE INTO edges (edge_id, from_chunk, to_chunk, to_external_ref, "
        "edge_type, provenance, confidence, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (eid, from_chunk, to_chunk, to_external_ref, edge_type, provenance, confidence, created_at),
    )
    conn.commit()
    return eid


def get_edges_from(conn: sqlite3.Connection, chunk_id: str) -> list:
    rows = conn.execute(
        "SELECT * FROM edges WHERE from_chunk = ? ORDER BY edge_type, edge_id", (chunk_id,)
    ).fetchall()
    return [dict(r) for r in rows]


def delete_edges_of_type(conn: sqlite3.Connection, edge_type: str) -> int:
    cur = conn.execute("DELETE FROM edges WHERE edge_type = ?", (edge_type,))
    conn.commit()
    return cur.rowcount


def set_citation_anchors(conn: sqlite3.Connection, chunk_id: str, anchors: dict) -> None:
    """Populate the chunks.citation_anchors column left empty by M3."""
    conn.execute("UPDATE chunks SET citation_anchors_json = ? WHERE chunk_id = ?",
                 (json.dumps(anchors or {}), chunk_id))
    conn.commit()
