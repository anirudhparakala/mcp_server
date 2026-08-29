"""Stage E (part 2) — resolve extracted citations into an edges graph.

Deterministic and offline. Two passes over the built CKB:
  1. persist_anchors  -- record what each chunk DEFINES into chunks.citation_anchors
                         (the column M3 deliberately left empty).
  2. build_graph      -- resolve what each chunk CITES to a target chunk and emit
                         `references` edges, plus structural `adjacent` edges.

Re-runnable: it clears its own edge types before rebuilding, and edge IDs are
derived from their fields, so repeated runs converge instead of accumulating.
"""

import json

from .citations import anchors_to_dict, extract_anchors
from ..db import ops


def build_anchor_index(conn) -> dict:
    """{(doc_id, kind, value): chunk_id} -- the FIRST chunk that defines each anchor.

    Documents restate section identifiers (annexes, tables of contents), and the
    definition is the earliest occurrence, so later restatements must not steal
    the anchor away from the section body.

    Anchors come from two sources that don't overlap in this corpus: standalone
    header lines in a chunk's text (EUR-Lex GDPR/AI-Act), and the last element of
    a chunk's heading_path (leginfo CA Commercial Code, where the section number
    lives in the breadcrumb, not the text) -- see citations.extract_anchors.
    """
    rows = conn.execute(
        "SELECT chunk_id, doc_id, text, heading_path_json FROM chunks "
        "ORDER BY doc_id, chunk_index"
    ).fetchall()
    index: dict = {}
    for row in rows:
        heading_path = json.loads(row["heading_path_json"])
        for anchor in extract_anchors(row["text"], heading_path):
            key = (row["doc_id"], anchor.kind, anchor.value)
            index.setdefault(key, row["chunk_id"])   # first definer wins
    return index


def persist_anchors(conn) -> int:
    """Write each chunk's anchors into chunks.citation_anchors. Returns rows updated."""
    rows = conn.execute(
        "SELECT chunk_id, doc_id, text, heading_path_json FROM chunks "
        "ORDER BY doc_id, chunk_index"
    ).fetchall()
    updated = 0
    for row in rows:
        heading_path = json.loads(row["heading_path_json"])
        anchors = extract_anchors(row["text"], heading_path)
        ops.set_citation_anchors(conn, row["chunk_id"], anchors_to_dict(anchors))
        updated += 1
    return updated
