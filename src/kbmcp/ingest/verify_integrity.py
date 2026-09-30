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

import json
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..db import ops
from .graph import doc_id_for_slug
from .manifest import load_manifest


class IntegrityError(ValueError):
    """Raised when the gate's own inputs are unusable (as distinct from a source
    failing its check)."""


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
    check: str  # "expect_terms" | "doc_id_collision" | "coverage"
    passed: bool
    detail: str


def _blob(conn, doc_id: str) -> str:
    """Chunk text ordered by chunk_index, joined with "\\n" -- byte-for-byte the
    blob verify_structure builds, so the two gates cannot disagree about what
    "present in the document" means."""
    return "\n".join(c["text"] for c in ops.get_chunks_for_doc(conn, doc_id))


def _covered_slugs(queries_path, fixtures_path) -> set:
    """Slugs carrying an EXTERNAL content check: a gold anchor or a structure
    fixture. Those are verified by their own gates -- re-verifying here would
    mean two places to fix when the assertion changes -- so this only asks
    whether the claim exists.

    graph_expectations.yaml is deliberately excluded: an edge between two
    documents is a weaker claim about either one's content than a quote is.
    """
    q, f = Path(queries_path), Path(fixtures_path)
    if not q.exists() and not f.exists():
        raise IntegrityError(
            f"coverage inputs unavailable: neither {q} nor {f} exists, so every "
            "source would fail 'no content check' for want of an input rather "
            "than a defect. Pass --no-coverage for a BYO corpus."
        )

    covered: set = set()
    if q.exists():
        for n, line in enumerate(q.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise IntegrityError(f"{q}: line {n} is not valid JSON: {exc}") from exc
            for g in item.get("gold") or []:
                covered.add(g["doc"])
    if f.exists():
        for fx in yaml.safe_load(f.read_text(encoding="utf-8")) or []:
            slug = fx.get("slug")
            if slug:
                covered.add(slug)
    return covered


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
    # Collision groups are computed up front, over ALL entries, before the
    # per-entry loop below -- not incrementally as entries are visited. An
    # incremental "have I seen this doc_id before" check is order-dependent:
    # the first-seen slug in a colliding pair never sees a collision (nothing
    # recorded yet), and a term-less slug is skipped entirely if the check
    # only runs inside `if e.expect_terms`. Either gap lets a real collision
    # go unreported depending on manifest ordering. Grouping first makes every
    # member of a collision visible regardless of iteration order or whether
    # it declares expect_terms.
    dids: dict[str, str] = {}
    doc_id_slugs: dict[str, list] = {}
    for e in entries:
        did = doc_id_for_slug(e.doc_id, by_slug, raw_dir)
        dids[e.doc_id] = did
        if did is not None:
            doc_id_slugs.setdefault(did, []).append(e.doc_id)
    collisions = {did: slugs for did, slugs in doc_id_slugs.items() if len(slugs) > 1}

    results: list[IntegrityResult] = []

    with closing(ops.get_db(ckb_path)) as conn:
        for e in entries:
            slug = e.doc_id
            did = dids[slug]
            group = collisions.get(did) if did is not None else None

            if group:
                # Two (or more) named sources collapsing to one document is a
                # manifest defect in its own right, independent of whether
                # this slug declares expect_terms -- report every slug in the
                # group, and do not also score it against the shared text.
                others = [s for s in group if s != slug]
                detail = (f"shares a doc_id with {others!r} (identical url and "
                          "pinned version), so its terms would score against that "
                          "document's text")
                results.append(IntegrityResult(slug, "doc_id_collision", False, detail))
            elif e.expect_terms:
                if did is None:
                    detail = ("not built: no manifest entry resolved to a doc_id "
                              "(no fetch pin in raw_dir), so there is no text to check")
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


def main(argv=None) -> int:
    import argparse
    import sys

    p = argparse.ArgumentParser(prog="python -m kbmcp.ingest.verify_integrity")
    p.add_argument("--ckb", default=str(DEFAULT_CKB))
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    p.add_argument("--queries", default=str(DEFAULT_QUERIES))
    p.add_argument("--fixtures", default=str(DEFAULT_FIXTURES))
    p.add_argument("--no-coverage", action="store_true",
                   help="verify declared expect_terms only; skip the "
                        "every-source-is-covered rule (BYO corpora)")
    a = p.parse_args(argv)

    results = verify_integrity(
        a.ckb, a.manifest, a.raw_dir, queries_path=a.queries,
        fixtures_path=a.fixtures, coverage=not a.no_coverage)

    for r in results:
        if not r.passed:
            print(f"  [FAIL] {r.slug} ({r.check}): {r.detail}", file=sys.stderr)

    # Summarize by whatever check names actually appear in `results`, not a
    # hard-coded pair. A hard-coded coverage/expect_terms split would count a
    # doc_id_collision result (Task 3's review fix, a third check name) in
    # neither line -- all_passed() below still covers it for the exit code,
    # but the operator-facing summary would under-report a real failure. This
    # form stays honest if a fourth check name is ever added.
    by_check: dict = {}
    for r in results:
        by_check.setdefault(r.check, []).append(r)
    for check in sorted(by_check):
        rs = by_check[check]
        print(f"integrity    {check} {sum(r.passed for r in rs)}/{len(rs)}",
              file=sys.stderr)

    ok = all_passed(results)
    print(f"integrity: {'ok' if ok else 'FAILED'}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
