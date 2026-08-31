"""Stage E (part 2) — resolve extracted citations into an edges graph.

Deterministic and offline. Two passes over the built CKB:
  1. persist_anchors  -- record what each chunk DEFINES into chunks.citation_anchors
                         (the column M3 deliberately left empty).
  2. build_graph      -- resolve what each chunk CITES to a target chunk and emit
                         `references` edges, plus structural `adjacent` edges.

Re-runnable: it clears its own edge types before rebuilding, and edge IDs are
derived from their fields, so repeated runs converge instead of accumulating.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .citations import (
    anchors_to_dict,
    extract_anchors,
    extract_heading_path_anchor,
    extract_references,
    extract_text_anchors,
)
from ..db import ops


def build_anchor_index(conn, min_anchor_body_chars: int = 0) -> dict:
    """{(doc_id, kind, value): chunk_id} -- the FIRST chunk that defines each anchor.

    Documents restate section identifiers (annexes, tables of contents), and the
    definition is the earliest occurrence, so later restatements must not steal
    the anchor away from the section body.

    Anchors come from two sources that don't overlap in this corpus: standalone
    header lines in a chunk's text (EUR-Lex GDPR/AI-Act), and the last element of
    a chunk's heading_path (leginfo CA Commercial Code, where the section number
    lives in the breadcrumb, not the text) -- see citations.extract_anchors.

    min_anchor_body_chars (review round 2, finding B): the chunker routinely ends
    a chunk right after the NEXT article's header line, e.g.
    "...\\nArticle 30\\nRecords of processing activities\\n1." (36 chars of body) --
    measured live, 58 of 196 cross-document edges landed on such a chunk. When a
    TEXT-derived header match is followed by fewer than this many characters in
    its own chunk, that chunk does not claim the anchor; the next chunk of the
    same document does instead (the chunker having split the header from its
    body). If no next chunk exists, losing the anchor entirely is worse than a
    marginal one, so it falls back to the original header chunk. heading_path-
    derived anchors already point at their own section's chunk and are never
    affected by this threshold. Defaults to 0 (no thresholding) so existing
    callers that don't pass it keep their prior behaviour exactly.
    """
    rows = conn.execute(
        "SELECT chunk_id, doc_id, text, heading_path_json FROM chunks "
        "ORDER BY doc_id, chunk_index"
    ).fetchall()

    by_doc: dict = {}
    for row in rows:
        by_doc.setdefault(row["doc_id"], []).append(row)

    index: dict = {}
    for doc_id, doc_rows in by_doc.items():
        for i, row in enumerate(doc_rows):
            heading_path = json.loads(row["heading_path_json"])
            hp_anchor = extract_heading_path_anchor(heading_path)
            if hp_anchor is not None:
                key = (doc_id, hp_anchor.kind, hp_anchor.value)
                index.setdefault(key, row["chunk_id"])   # first definer wins

            text = row["text"] or ""
            for anchor, end in extract_text_anchors(text):
                key = (doc_id, anchor.kind, anchor.value)
                if key in index:
                    continue                              # first definer wins
                body_len = len(text) - end
                if body_len >= min_anchor_body_chars:
                    index[key] = row["chunk_id"]
                elif i + 1 < len(doc_rows):
                    index[key] = doc_rows[i + 1]["chunk_id"]   # header/body split across chunks
                else:
                    index[key] = row["chunk_id"]          # no successor -- don't lose the anchor
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


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def scope_for(slug: str, entries_by_slug: dict) -> set:
    """Manifest slugs a document's references may resolve into: itself + declared refs.

    Bare citations carry no instrument name (measured: 1 of 173 EDPB references names
    one, and it names a document outside the corpus), and the same article number is
    defined by several regulations. The manifest's `references:` field is the only
    reliable statement of which instruments a document is about.
    """
    entry = entries_by_slug.get(slug)
    scope = {slug}
    if entry is not None:
        scope.update(getattr(entry, "references", None) or [])
    return scope


def candidates_for(reference, *, scope_doc_ids: set, anchor_index: dict) -> list:
    """Chunk IDs within `scope_doc_ids` that define `reference`.

    The raw candidate list, not collapsed to a single verdict -- callers that need
    to tell "out of corpus" (0 candidates) apart from "ambiguous" (>1 candidates)
    for stats/config purposes use this instead of resolve_reference.
    """
    targets = []
    for doc_id in scope_doc_ids:
        chunk = anchor_index.get((doc_id, reference.kind, reference.value))
        if chunk is not None:
            targets.append(chunk)
    return targets


def resolve_reference(reference, *, from_doc_id: str, scope_doc_ids: set, anchor_index: dict):
    """Target chunk_id, or None when out of corpus or ambiguous within the scope."""
    targets = candidates_for(reference, scope_doc_ids=scope_doc_ids, anchor_index=anchor_index)
    if len(targets) != 1:
        return None            # 0 = out of corpus, >1 = ambiguous; guessing is worse
    return targets[0]


def build_adjacent_edges(conn) -> int:
    """Link consecutive chunks inside each document (never across documents)."""
    rows = conn.execute(
        "SELECT chunk_id, doc_id FROM chunks ORDER BY doc_id, chunk_index"
    ).fetchall()
    made = 0
    for prev, cur in zip(rows, rows[1:]):
        if prev["doc_id"] != cur["doc_id"]:
            continue
        ops.insert_edge(conn, from_chunk=prev["chunk_id"], to_chunk=cur["chunk_id"],
                        edge_type="adjacent", confidence=1.0, created_at=_now())
        made += 1
    return made


def build_graph(ckb_path, manifest_path, raw_dir, cfg: dict, *, doc_id_for=None) -> dict:
    """Resolve every chunk's citations into edges. Re-runnable and deterministic."""
    from ..models.ids import doc_id as mk_doc_id
    from .fetch import read_meta
    from .manifest import load_manifest

    entries = load_manifest(manifest_path)
    by_slug = {e.doc_id: e for e in entries}

    def _doc_id(slug: str):
        if doc_id_for is not None:
            return doc_id_for(slug)
        entry = by_slug.get(slug)
        meta = read_meta(raw_dir, slug) if entry is not None else None
        if entry is None or meta is None:
            return None
        return mk_doc_id(entry.url, meta["resolved_version"])

    stats = {"anchors": 0, "references_resolved": 0, "references_unresolved": 0,
             "ambiguous": 0, "adjacent": 0, "errors": []}
    conn = ops.get_db(ckb_path)
    try:
        stats["anchors"] = persist_anchors(conn)
        anchor_index = build_anchor_index(
            conn, min_anchor_body_chars=cfg.get("min_anchor_body_chars", 0))

        edge_types = cfg.get("edge_types", ["adjacent", "references"])
        resolve_refs = cfg.get("resolve", True) and "references" in edge_types

        # Only delete an edge type this run will actually rebuild: `references`
        # rebuilding is additionally gated on `resolve`, so deleting it when
        # resolve=False would wipe previously-built edges and never restore them.
        for edge_type in edge_types:
            if edge_type == "references" and not resolve_refs:
                continue
            ops.delete_edges_of_type(conn, edge_type)   # rebuild, never accumulate

        if "adjacent" in edge_types:
            stats["adjacent"] = build_adjacent_edges(conn)

        if not resolve_refs:
            return stats

        # doc_id -> the scope of doc_ids its references may resolve into
        scope_by_doc = {}
        for slug in by_slug:
            did = _doc_id(slug)
            if did is None:
                continue
            resolved = {_doc_id(s) for s in scope_for(slug, by_slug)}
            scope_by_doc[did] = {d for d in resolved if d is not None}

        max_targets = cfg.get("max_targets_per_reference", 1)
        extractors = tuple(cfg.get("extractors", ["legal", "academic"]))
        rows = conn.execute(
            "SELECT chunk_id, doc_id, text FROM chunks ORDER BY doc_id, chunk_index"
        ).fetchall()
        for row in rows:
            scope = scope_by_doc.get(row["doc_id"], {row["doc_id"]})
            try:
                for ref in extract_references(row["text"], extractors=extractors):
                    candidates = candidates_for(ref, scope_doc_ids=scope,
                                                anchor_index=anchor_index)
                    if len(candidates) == 1:
                        target = candidates[0]
                        if target == row["chunk_id"]:
                            continue    # self-citation: cites the section it defines
                        ops.insert_edge(conn, from_chunk=row["chunk_id"], to_chunk=target,
                                        edge_type="references", provenance=ref.raw,
                                        confidence=1.0, created_at=_now())
                        stats["references_resolved"] += 1
                    elif len(candidates) > max_targets:
                        stats["ambiguous"] += 1        # too many candidates; guessing is worse
                    else:
                        if cfg.get("record_unresolved", True):
                            ops.insert_edge(conn, from_chunk=row["chunk_id"],
                                            to_external_ref=f"{ref.kind}:{ref.value}",
                                            edge_type="references", provenance=ref.raw,
                                            confidence=0.0, created_at=_now())
                        stats["references_unresolved"] += 1
            except Exception as exc:
                # One bad chunk must not lose the whole graph -- record and continue.
                stats["errors"].append(f"{row['chunk_id']}: {exc}")
    finally:
        conn.close()
    return stats


def main(argv=None) -> int:
    from ..config import load_corpus_config

    p = argparse.ArgumentParser(prog="python -m kbmcp.ingest.graph")
    p.add_argument("--ckb", default="ckb/ckb.sqlite")
    p.add_argument("--manifest", default="corpus/manifest.yaml")
    p.add_argument("--raw-dir", default="corpus/raw")
    p.add_argument("--config", default="config/corpus_config.yaml")
    a = p.parse_args(argv)

    cfg = load_corpus_config(a.config).graph
    stats = build_graph(a.ckb, a.manifest, a.raw_dir, cfg)
    print(f"anchors={stats['anchors']} adjacent={stats['adjacent']} "
          f"references_resolved={stats['references_resolved']} "
          f"unresolved={stats['references_unresolved']} errors={len(stats['errors'])}",
          file=sys.stderr)
    return 0 if not stats["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
