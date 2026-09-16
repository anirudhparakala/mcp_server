"""Gold verification gate (stage for eval/benchmark): assert every benchmark
item's gold labels still point at the right chunks.

Gold labels bind to `chunk_id = sha256(canonical_url, version, chunk_index)`,
so re-fetching or re-chunking a source changes them -- this has already
happened once in this project (see the UCC re-fetch, M5). Each gold label
also carries an `anchor` -- a verbatim quote that must still appear in its
chunk -- so drift is a loud failure here rather than silently-wrong retrieval
scoring. Mirrors verify_structure.py / verify_graph.py's shape and naming:
items -> results list -> all_passed.
"""

from collections import Counter
from contextlib import closing
from dataclasses import dataclass

from ..db import ops
from . import models
from ..ingest.graph import doc_id_for_slug
from ..ingest.manifest import load_manifest


@dataclass
class CheckResult:
    item_id: str
    check: str
    passed: bool
    detail: str


def verify_gold(ckb_path, queries_path, manifest_path, raw_dir, *, min_per_tier=15) -> list[CheckResult]:
    items = models.load_queries(queries_path)
    entries = load_manifest(manifest_path)
    by_slug = {e.doc_id: e for e in entries}

    results: list[CheckResult] = []
    seen_ids: set[str] = set()

    # closing() guarantees the sqlite handle is released even if an item is
    # malformed (Windows temp-dir cleanup fails on an open handle).
    with closing(ops.get_db(ckb_path)) as conn:
        for item in items:
            for ref in item.gold:
                chunk = ops.get_chunk(conn, ref.chunk_id)

                # 1. chunk_exists
                exists = chunk is not None
                results.append(CheckResult(
                    item.id, "chunk_exists", exists,
                    "ok" if exists else f"chunk_id {ref.chunk_id!r} not found in CKB",
                ))
                if not exists:
                    # anchor_present and doc_matches need the chunk; skip them
                    # rather than raising, so the rest of the item's checks
                    # (and every other item) still run.
                    continue

                # 2. anchor_present
                anchor_ok = ref.anchor in chunk["text"]
                results.append(CheckResult(
                    item.id, "anchor_present", anchor_ok,
                    "ok" if anchor_ok else
                    f"anchor {ref.anchor!r} not found in chunk {ref.chunk_id!r} text",
                ))

                # 3. doc_matches
                expected_doc_id = doc_id_for_slug(ref.doc, by_slug, raw_dir)
                doc_ok = expected_doc_id == chunk["doc_id"]
                results.append(CheckResult(
                    item.id, "doc_matches", doc_ok,
                    "ok" if doc_ok else
                    f"gold doc {ref.doc!r} resolves to doc_id {expected_doc_id!r}, "
                    f"but chunk {ref.chunk_id!r} belongs to doc_id {chunk['doc_id']!r}",
                ))

            # 4. tier_valid / id_unique / id_prefix
            tier_ok = item.tier in models.TIERS
            results.append(CheckResult(
                item.id, "tier_valid", tier_ok,
                "ok" if tier_ok else f"unknown tier {item.tier!r}; expected one of {models.TIERS}",
            ))

            id_unique = item.id not in seen_ids
            results.append(CheckResult(
                item.id, "id_unique", id_unique,
                "ok" if id_unique else f"duplicate item id {item.id!r}",
            ))
            seen_ids.add(item.id)

            prefix = item.id.split("-")[0]
            prefix_ok = prefix == item.tier
            results.append(CheckResult(
                item.id, "id_prefix", prefix_ok,
                "ok" if prefix_ok else
                f"id prefix {prefix!r} disagrees with tier {item.tier!r}",
            ))

            # 5. t7_equivalence
            is_t7 = item.tier == "T7"
            is_unanswerable = item.answerable is False
            is_empty_gold = not item.gold
            t7_ok = is_t7 == is_unanswerable == is_empty_gold
            results.append(CheckResult(
                item.id, "t7_equivalence", t7_ok,
                "ok" if t7_ok else
                f"T7 <-> unanswerable <-> empty-gold disagree "
                f"(tier=={item.tier!r} answerable=={item.answerable!r} "
                f"gold has {len(item.gold)} ref(s))",
            ))

            # 6. multi_doc (T4/T5 only)
            if item.tier in ("T4", "T5"):
                doc_count = len({g.doc for g in item.gold})
                multi_ok = doc_count >= 2
                results.append(CheckResult(
                    item.id, "multi_doc", multi_ok,
                    "ok" if multi_ok else
                    f"tier {item.tier} needs gold refs spanning >= 2 documents, "
                    f"got {doc_count}",
                ))

            # 7. tier_fields (T2 needs a distractor, T6 needs a collision_term)
            if item.tier == "T2":
                t2_ok = len(item.distractor_docs) > 0
                results.append(CheckResult(
                    item.id, "tier_fields", t2_ok,
                    "ok" if t2_ok else "tier T2 needs a non-empty distractor_docs",
                ))
            if item.tier == "T6":
                t6_ok = bool(item.collision_term)
                results.append(CheckResult(
                    item.id, "tier_fields", t6_ok,
                    "ok" if t6_ok else "tier T6 needs a non-empty collision_term",
                ))

            # 8. slug_known (distractor_docs entries and gold refs' doc)
            for slug in item.distractor_docs:
                slug_ok = slug in by_slug
                results.append(CheckResult(
                    item.id, "slug_known", slug_ok,
                    "ok" if slug_ok else
                    f"distractor_docs slug {slug!r} is not in the manifest",
                ))
            for ref in item.gold:
                slug_ok = ref.doc in by_slug
                results.append(CheckResult(
                    item.id, "slug_known", slug_ok,
                    "ok" if slug_ok else
                    f"gold doc slug {ref.doc!r} is not in the manifest",
                ))

    # 9. tier_count -- once, across the whole file, not per item.
    # A tier absent entirely (0 items) is not held to min_per_tier here -- a
    # partial/in-progress queries file legitimately omits whole tiers, and
    # this file only judges tiers it actually contains. Only tiers with at
    # least one item are checked against the floor.
    counts = Counter(item.tier for item in items)
    for tier in models.TIERS:
        n = counts.get(tier, 0)
        if n == 0:
            continue
        count_ok = n >= min_per_tier
        results.append(CheckResult(
            tier, "tier_count", count_ok,
            "ok" if count_ok else
            f"tier {tier} has {n} item(s), need >= {min_per_tier}",
        ))

    return results


def all_passed(results) -> bool:
    return all(r.passed for r in results)


def main(argv=None) -> int:
    import argparse
    import sys

    p = argparse.ArgumentParser(prog="python -m kbmcp.eval.verify_gold")
    p.add_argument("--ckb", default="ckb/ckb.sqlite")
    p.add_argument("--queries", default="corpus/benchmark/queries.jsonl")
    p.add_argument("--manifest", default="corpus/manifest.yaml")
    p.add_argument("--raw-dir", default="corpus/raw")
    p.add_argument("--min-per-tier", type=int, default=15)
    a = p.parse_args(argv)

    try:
        results = verify_gold(a.ckb, a.queries, a.manifest, a.raw_dir,
                              min_per_tier=a.min_per_tier)
    except models.BenchmarkError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for r in results:
        if not r.passed:
            print(f"  [FAIL] {r.item_id} {r.check}: {r.detail}", file=sys.stderr)

    tier_counts = {r.item_id: r.detail for r in results if r.check == "tier_count"}
    for tier in models.TIERS:
        print(f"  tier {tier}: {tier_counts.get(tier, 'not checked')}", file=sys.stderr)

    ok = all_passed(results)
    print(f"gold gate: {sum(r.passed for r in results)}/{len(results)} checks "
          f"{'ok' if ok else 'FAILED'}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
