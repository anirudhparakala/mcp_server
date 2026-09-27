"""Corpus-integrity gate: is the built document the document the manifest said?

Complements the other two ingest gates. `verify_structure` asks whether a
document's tables and cross-references survived parse+chunk; this asks the prior
question -- whether the bytes we parsed came from the page we meant. Carlill
passed every existing check for five milestones while containing no case text,
because its manifest URL was a Wikisource redirect stub and nothing in the
corpus contradicted it.

Matching mirrors `verify_structure` exactly: case-sensitive exact substring
against chunk `text` ordered by `chunk_index` and joined with "\\n". `context` is
never matched -- M4's generated contexts are model output, and a hallucinated
summary must not satisfy a gate about the source document.
"""

from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from ..db import ops
from .graph import doc_id_for_slug
from .manifest import load_manifest

# This file lives at <root>/src/kbmcp/ingest/, so parents[3] is the repo root.
# Anchored to the PACKAGE, never the process cwd: as cwd-relative literals these
# silently "did not exist" outside the repo root, so a legitimate shipped build
# launched from elsewhere skipped its own gates and exited 0 (the M6 lesson).
_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CKB = _ROOT / "ckb" / "ckb.sqlite"
DEFAULT_MANIFEST = _ROOT / "corpus" / "manifest.yaml"
DEFAULT_RAW_DIR = _ROOT / "corpus" / "raw"
DEFAULT_QUERIES = _ROOT / "corpus" / "benchmark" / "queries.jsonl"
DEFAULT_FIXTURES = _ROOT / "corpus" / "benchmark" / "structure_fixtures.yaml"


@dataclass
class IntegrityResult:
    slug: str
    check: str  # "expect_terms" | "coverage"
    passed: bool
    detail: str


def _blob(conn, doc_id: str) -> str:
    """Chunk text ordered by chunk_index, joined with "\\n" -- byte-for-byte the
    blob verify_structure builds, so the two gates cannot disagree about what
    "present in the document" means."""
    return "\n".join(c["text"] for c in ops.get_chunks_for_doc(conn, doc_id))


def _covered_slugs(queries_path, fixtures_path) -> set:
    return set()  # replaced in Task 4


def verify_integrity(ckb_path, manifest_path, raw_dir, *,
                     queries_path=DEFAULT_QUERIES,
                     fixtures_path=DEFAULT_FIXTURES,
                     coverage=True) -> list[IntegrityResult]:
    """Verify declared expect_terms, and (when `coverage`) that every source has
    a content check at all.

    `coverage=False` is the BYO path: a BYO user's expect_terms are their own
    assertion about their own documents and are still verified, but requiring
    them to declare one per source would impose this corpus's standards on
    theirs.
    """
    entries = load_manifest(manifest_path)
    by_slug = {e.doc_id: e for e in entries}
    external = _covered_slugs(queries_path, fixtures_path) if coverage else set()

    # doc_id is sha256(url, version), so two slugs with the same url+version
    # collapse to one document and would score against each other's text.
    seen_doc_ids: dict[str, str] = {}
    results: list[IntegrityResult] = []

    with closing(ops.get_db(ckb_path)) as conn:
        for e in entries:
            slug = e.doc_id
            did = doc_id_for_slug(slug, by_slug, raw_dir)
            collision = seen_doc_ids.get(did) if did else None
            if did:
                seen_doc_ids.setdefault(did, slug)

            if e.expect_terms:
                if did is None:
                    detail = ("not built: no manifest entry resolved to a doc_id "
                              "(no fetch pin in raw_dir), so there is no text to check")
                    passed = False
                elif collision:
                    detail = (f"shares a doc_id with {collision!r} (identical url and "
                              "pinned version), so its terms would score against that "
                              "document's text")
                    passed = False
                else:
                    missing = [t for t in e.expect_terms if t not in _blob(conn, did)]
                    passed = not missing
                    detail = "ok" if passed else f"missing: {missing}"
                results.append(IntegrityResult(slug, "expect_terms", passed, detail))

            if coverage:
                ok = bool(e.expect_terms) or slug in external
                results.append(IntegrityResult(
                    slug, "coverage", ok,
                    "ok" if ok else ("no content check: no gold anchor, no structure "
                                     "fixture, no expect_terms")))
    return results


def all_passed(results) -> bool:
    return all(r.passed for r in results)
