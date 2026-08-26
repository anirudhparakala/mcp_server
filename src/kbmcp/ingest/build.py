"""Build stage — parse + chunk each manifest source into ckb/ckb.sqlite (docs + chunks).

Dev-only. No embeddings/context/edges here (M2 fetch done; M4/M5 add context/graph;
M6 wires the full build_corpus.py). Deterministic: doc_id/chunk_id from models.ids
using canonical_url = entry.url and version = the pin record's resolved_version.
"""

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ..config import load_corpus_config
from ..db import ops
from ..db.schema import create_all_tables
from ..models.ids import chunk_id as mk_chunk_id
from ..models.ids import doc_id as mk_doc_id
from . import contextualize as ctxmod
from .fetch import read_meta
from .manifest import load_manifest


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_ckb(manifest_path, raw_dir, parsed_dir, ckb_path, cfg, *, only=None, force=False,
              reparse=False, no_context=False, require_context=False) -> dict:
    entries = load_manifest(manifest_path)
    conn = ops.get_db(ckb_path)
    create_all_tables(conn)
    stats = {"docs": 0, "chunks": 0, "contexts": 0, "contexts_missing": 0, "pruned": 0,
             "errors": []}

    # Contextual Retrieval is pins-first: valid pins are applied with or without
    # credentials, and the API is called only for chunks that have none. Locked
    # Decision 5/6 -- no key and no pins is a NOTE, not a failure.
    ctx_cfg = cfg.get("contextualize", {})
    contexts_dir = ctx_cfg.get("contexts_dir", "corpus/contexts")
    if not ctx_cfg.get("enabled"):
        client, client_reason = None, "disabled in config (contextualize.enabled)"
    elif no_context:
        client, client_reason = None, "skipped (--no-context)"
    else:
        client, client_reason = ctxmod.make_client(ctx_cfg)

    try:
        _run_build(conn, entries, raw_dir, parsed_dir, cfg, ctx_cfg, contexts_dir, client,
                   client_reason, stats, only=only, force=force, reparse=reparse,
                   require_context=require_context)
    finally:
        # Always release the sqlite handle -- including on KeyboardInterrupt during
        # the long contextualization pass. A leaked handle locks ckb/ckb.sqlite on
        # Windows and fails temp-dir cleanup in tests.
        conn.close()
    return stats


def _run_build(conn, entries, raw_dir, parsed_dir, cfg, ctx_cfg, contexts_dir, client,
               client_reason, stats, *, only, force, reparse, require_context) -> None:
    for e in entries:
        if only and e.doc_id not in only:
            continue
        did = None
        try:
            # --force rebuilds the DB from the committed parse; only --reparse re-runs
            # Docling (which overwrites the committed corpus/parsed/ reproducibility anchor).
            meta, version, dl_doc, records, title = ctxmod.load_document(
                e, raw_dir, parsed_dir, cfg, reparse=reparse)
            did = mk_doc_id(e.url, version)
            if not (force or reparse) and _doc_present(conn, did):
                continue

            # Contextualization runs BEFORE any row is written. It is the slow,
            # networked part (minutes per document), and `except Exception` below
            # cannot catch a KeyboardInterrupt: writing the docs row first meant a
            # Ctrl-C mid-run left docs=1/chunks=0, which every later non-force
            # build then SKIPPED as already present -- losing the document
            # permanently and silently. Nothing touches the DB until it returns.
            ctx_out = ctxmod.contexts_for_document(
                e.doc_id, records, canonical_url=e.url, version=version,
                doc_title=title, source_url=e.url, contexts_dir=contexts_dir,
                cfg=ctx_cfg, client=client,
            )
            contexts = ctx_out["contexts"]

            _delete_doc(conn, did)  # clear any prior/partial rows -> atomic (re)build
            _upsert_source_and_doc(conn, e, meta, did, dl_doc)

            # Accumulate this document's numbers LOCALLY and fold them into stats
            # only once every row is in. An insert can raise partway through, and
            # the handler below rolls the whole document back -- counting as we go
            # would leave a rolled-back doc's chunks and contexts in the totals
            # (e.g. reporting "2 of 0 chunks have no context").
            doc_chunks = 0
            for rec in records:
                cid = mk_chunk_id(e.url, version, rec.chunk_index)
                ops.insert_chunk(
                    conn, chunk_id=cid, doc_id=did,
                    chunk_index=rec.chunk_index, text=rec.text, chunk_type=rec.chunk_type,
                    context=contexts.get(cid),
                    heading_path=rec.heading_path, table=rec.table,
                    citation_anchors=rec.citation_anchors,
                )
                doc_chunks += 1

            stats["chunks"] += doc_chunks
            stats["contexts"] += len(contexts)
            stats["contexts_missing"] += max(0, len(records) - len(contexts))
            for err in ctx_out["errors"]:
                stats["errors"].append(
                    {"doc_id": e.doc_id,
                     "error": f"context chunk {err['chunk_index']}: {err['error']}"})
            stats["docs"] += 1
        except Exception as exc:  # noqa: BLE001 — one bad source must not abort the build
            if did is not None:
                _delete_doc(conn, did)  # drop partial rows so a later run rebuilds cleanly
            stats["errors"].append({"doc_id": e.doc_id, "error": str(exc)})

    # Reconcile the DB against the manifest. Changing a source's URL or version
    # changes its doc_id, so the PREVIOUS doc and its chunks would otherwise linger
    # forever as stale content that still gets indexed and served. Only on a
    # complete, error-free full build: with --only we are looking at a subset, and
    # after an error we cannot know the missing document's identity -- pruning on
    # either incomplete picture would delete good data.
    if only is None and not stats["errors"]:
        expected = set()
        for e in entries:
            meta = read_meta(raw_dir, e.doc_id)
            if meta is not None:
                expected.add(mk_doc_id(e.url, meta["resolved_version"]))
        for row in conn.execute("SELECT doc_id FROM docs").fetchall():
            if row["doc_id"] not in expected:
                _delete_doc(conn, row["doc_id"])
                stats["pruned"] += 1

    if stats["contexts_missing"]:
        msg = (f"{stats['contexts_missing']} of {stats['chunks']} chunks have no context "
               f"({client_reason}). The CKB is usable, but Contextual Retrieval improves recall "
               f"~49%; set ANTHROPIC_API_KEY (or run `ant auth login`) and rebuild with --force.")
        if require_context:
            stats["errors"].append({"doc_id": "*", "error": f"--require-context: {msg}"})
        else:
            print(f"[note] {msg}", file=sys.stderr)

    conn.execute(
        "INSERT INTO ingest_runs (ingest_run_id, started_at, finished_at, status, stats_json) "
        "VALUES (?, ?, ?, ?, ?)",
        (f"build-{_now()}-{uuid.uuid4().hex[:8]}", _now(), _now(),
         "ok" if not stats["errors"] else "partial", json.dumps(stats)),
    )
    conn.commit()


def _doc_present(conn, did) -> bool:
    """A doc counts as built only if it actually has chunks.

    Defence in depth: a docs row with zero chunks is partial state (e.g. an
    interrupt slipped between the row insert and the chunk loop). Treating it as
    'present' would skip that document on every later build, forever.
    """
    return conn.execute(
        "SELECT 1 FROM docs d WHERE d.doc_id = ? "
        "AND EXISTS (SELECT 1 FROM chunks c WHERE c.doc_id = d.doc_id)", (did,)
    ).fetchone() is not None


def _delete_doc(conn, did) -> None:
    """Remove a doc and its chunks (chunks first, for the FK); commit. No-op if absent."""
    conn.execute("DELETE FROM chunks WHERE doc_id = ?", (did,))
    conn.execute("DELETE FROM docs WHERE doc_id = ?", (did,))
    conn.commit()


def _upsert_source_and_doc(conn, e, meta, did, dl_doc) -> None:
    if conn.execute("SELECT 1 FROM sources WHERE canonical_url = ?", (e.url,)).fetchone() is None:
        ops.insert_source(
            conn, canonical_url=e.url, url_original=meta.get("source_url", e.url), domain=e.domain,
            format=meta.get("format", e.format), license=e.license, license_ok=e.license_ok,
            version=meta["resolved_version"], raw_cache=meta.get("raw_filename"),
            content_hash=meta.get("content_hash"), tier_roles=e.tier_roles,
            collision_terms=e.collision_terms, references=e.references, conflict_with=e.conflict_with,
            rationale=e.rationale, fetched_at=meta.get("fetched_at"),
        )
    title = getattr(dl_doc, "name", None) or e.doc_id
    ops.insert_doc(conn, doc_id=did, canonical_url=e.url, domain=e.domain,
                   format=meta.get("format", e.format), title=title, created_at=_now())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m kbmcp.ingest.build")
    p.add_argument("--manifest", default="corpus/manifest.yaml")
    p.add_argument("--raw-dir", default="corpus/raw")
    p.add_argument("--parsed-dir", default="corpus/parsed")
    p.add_argument("--ckb", default="ckb/ckb.sqlite")
    p.add_argument("--config", default="config/corpus_config.yaml")
    p.add_argument("--only", action="append", default=None)
    p.add_argument("--force", action="store_true", help="rebuild the DB from the committed parse")
    p.add_argument("--reparse", action="store_true",
                   help="re-run Docling and OVERWRITE the committed parse (implies --force)")
    p.add_argument("--no-context", action="store_true",
                   help="skip Contextual Retrieval generation (pinned contexts are still applied)")
    p.add_argument("--require-context", action="store_true",
                   help="fail the build if any chunk ends up without a context")
    a = p.parse_args(argv)
    Path(a.ckb).parent.mkdir(parents=True, exist_ok=True)
    cfg = load_corpus_config(a.config).raw
    stats = build_ckb(a.manifest, a.raw_dir, a.parsed_dir, a.ckb, cfg,
                      only=set(a.only) if a.only else None, force=a.force, reparse=a.reparse,
                      no_context=a.no_context, require_context=a.require_context)
    print(f"docs={stats['docs']} chunks={stats['chunks']} contexts={stats['contexts']} "
          f"errors={len(stats['errors'])}", file=sys.stderr)
    for err in stats["errors"]:
        print(f"  [error] {err['doc_id']}: {err['error']}", file=sys.stderr)
    return 0 if not stats["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
