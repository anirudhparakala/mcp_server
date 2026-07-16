"""CKB SQLite operations (document side: sources, docs, chunks).

JSON-valued columns are serialized with json.dumps here so callers pass native
Python lists/dicts. Edge ops live in the cross-ref-graph milestone.
"""

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


def get_chunk(conn: sqlite3.Connection, chunk_id: str) -> Optional[dict]:
    """Fetch a chunk by ID; _json columns are returned parsed (list/dict), not as raw strings."""
    row = conn.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
    if row is None:
        return None
    result = dict(row)
    # Deserialize JSON columns
    result["heading_path_json"] = json.loads(result["heading_path_json"])
    if result["table_json"] is not None:
        result["table_json"] = json.loads(result["table_json"])
    result["citation_anchors_json"] = json.loads(result["citation_anchors_json"])
    return result
