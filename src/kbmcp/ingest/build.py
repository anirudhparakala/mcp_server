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
from .chunk import chunk_document
from .fetch import read_meta
from .manifest import load_manifest
from .parse import parse_source


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_ckb(manifest_path, raw_dir, parsed_dir, ckb_path, cfg, *, only=None, force=False, reparse=False) -> dict:
    entries = load_manifest(manifest_path)
    conn = ops.get_db(ckb_path)
    create_all_tables(conn)
    stats = {"docs": 0, "chunks": 0, "errors": []}
    for e in entries:
        if only and e.doc_id not in only:
            continue
        did = None
        try:
            meta = read_meta(raw_dir, e.doc_id)
            if meta is None:
                raise FileNotFoundError(f"no pin record for {e.doc_id} (run fetch first)")
            version = meta["resolved_version"]
            did = mk_doc_id(e.url, version)
            if not (force or reparse) and _doc_present(conn, did):
                continue
            _delete_doc(conn, did)  # clear any prior/partial rows -> atomic (re)build
            raw_path = Path(raw_dir) / meta["raw_filename"]
            # --force rebuilds the DB from the committed parse; only --reparse re-runs
            # Docling (which overwrites the committed corpus/parsed/ reproducibility anchor).
            dl_doc = parse_source(e.doc_id, raw_path, parsed_dir, cfg["parse"], force=reparse)
            _upsert_source_and_doc(conn, e, meta, did, dl_doc)
            for rec in chunk_document(dl_doc, cfg["chunk"]):
                ops.insert_chunk(
                    conn, chunk_id=mk_chunk_id(e.url, version, rec.chunk_index), doc_id=did,
                    chunk_index=rec.chunk_index, text=rec.text, chunk_type=rec.chunk_type,
                    heading_path=rec.heading_path, table=rec.table, citation_anchors=rec.citation_anchors,
                )
                stats["chunks"] += 1
            stats["docs"] += 1
        except Exception as exc:  # noqa: BLE001 — one bad source must not abort the build
            if did is not None:
                _delete_doc(conn, did)  # drop partial rows so a later run rebuilds cleanly
            stats["errors"].append({"doc_id": e.doc_id, "error": str(exc)})
    conn.execute(
        "INSERT INTO ingest_runs (ingest_run_id, started_at, finished_at, status, stats_json) "
        "VALUES (?, ?, ?, ?, ?)",
        (f"build-{_now()}-{uuid.uuid4().hex[:8]}", _now(), _now(),
         "ok" if not stats["errors"] else "partial", json.dumps(stats)),
    )
    conn.commit()
    conn.close()  # release the sqlite file (Windows temp-dir cleanup fails on an open handle)
    return stats


def _doc_present(conn, did) -> bool:
    return conn.execute("SELECT 1 FROM docs WHERE doc_id = ?", (did,)).fetchone() is not None


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
    a = p.parse_args(argv)
    Path(a.ckb).parent.mkdir(parents=True, exist_ok=True)
    cfg = load_corpus_config(a.config).raw
    stats = build_ckb(a.manifest, a.raw_dir, a.parsed_dir, a.ckb, cfg,
                      only=set(a.only) if a.only else None, force=a.force, reparse=a.reparse)
    print(f"docs={stats['docs']} chunks={stats['chunks']} errors={len(stats['errors'])}", file=sys.stderr)
    for err in stats["errors"]:
        print(f"  [error] {err['doc_id']}: {err['error']}", file=sys.stderr)
    return 0 if not stats["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
