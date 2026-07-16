"""CKB SQLite schema — 5 tables (sources, docs, chunks, edges, ingest_runs).

Trimmed from the RAG project's 9-table schema per the ingest design spec §7:
session/metrics tables are dropped. JSON-valued columns hold lists/dicts as
text (json.dumps) — see db/ops.py.
"""

import sqlite3

_DDL = """
CREATE TABLE IF NOT EXISTS sources (
    canonical_url        TEXT PRIMARY KEY,
    url_original         TEXT NOT NULL,
    domain               TEXT NOT NULL,
    format               TEXT NOT NULL,
    license              TEXT NOT NULL,
    license_ok           INTEGER NOT NULL,
    version              TEXT NOT NULL,
    raw_cache            TEXT,
    content_hash         TEXT,
    tier_roles_json      TEXT NOT NULL DEFAULT '[]',
    collision_terms_json TEXT NOT NULL DEFAULT '[]',
    references_json      TEXT NOT NULL DEFAULT '[]',
    conflict_with_json   TEXT NOT NULL DEFAULT '[]',
    rationale            TEXT,
    fetched_at           TEXT
);

CREATE TABLE IF NOT EXISTS docs (
    doc_id         TEXT PRIMARY KEY,
    canonical_url  TEXT NOT NULL REFERENCES sources(canonical_url),
    title          TEXT,
    domain         TEXT NOT NULL,
    format         TEXT NOT NULL,
    doc_type       TEXT,
    structure_json TEXT,
    created_at     TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id              TEXT PRIMARY KEY,
    doc_id                TEXT NOT NULL REFERENCES docs(doc_id),
    chunk_index           INTEGER NOT NULL,
    text                  TEXT NOT NULL,
    context               TEXT,
    heading_path_json     TEXT NOT NULL DEFAULT '[]',
    chunk_type            TEXT NOT NULL,
    table_json            TEXT,
    citation_anchors_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE (doc_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS edges (
    edge_id         TEXT PRIMARY KEY,
    from_chunk      TEXT NOT NULL REFERENCES chunks(chunk_id),
    to_chunk        TEXT REFERENCES chunks(chunk_id),
    to_external_ref TEXT,
    edge_type       TEXT NOT NULL,
    provenance      TEXT,
    confidence      REAL,
    created_at      TEXT
);

CREATE TABLE IF NOT EXISTS ingest_runs (
    ingest_run_id TEXT PRIMARY KEY,
    started_at    TEXT,
    finished_at   TEXT,
    status        TEXT,
    config_json   TEXT,
    stats_json    TEXT
);
"""


def create_all_tables(conn: sqlite3.Connection) -> None:
    """Create all CKB tables if they do not already exist (idempotent)."""
    conn.executescript(_DDL)
    conn.commit()
