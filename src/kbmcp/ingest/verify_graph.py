"""Cross-reference ground-truth gate (stage H): assert the graph pass's resolved
`references` edges reach the ground-truth pairs recorded in
corpus/benchmark/graph_expectations.yaml. Mirrors verify_structure.py's shape
and naming -- see that module for the pattern (fixtures -> results list ->
all_passed).

Deterministic and offline: reads the expectations YAML and the built CKB only.
"""

from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..db import ops
from .graph import doc_id_for_slug
from .manifest import load_manifest


@dataclass
class ExpectationResult:
    name: str
    from_slug: str
    to_slug: str
    min_edges: int
    passed: bool
    pending: bool
    edges_observed: int
    detail: str


def _resolved_edge_count(conn, from_doc_id: str, to_doc_id: str) -> int:
    """Resolved `references` edges from any chunk of `from_doc_id` to any chunk
    of `to_doc_id`. Resolved means to_chunk IS NOT NULL."""
    row = conn.execute(
        "SELECT COUNT(*) FROM edges "
        "WHERE edge_type = 'references' AND to_chunk IS NOT NULL "
        "AND from_chunk IN (SELECT chunk_id FROM chunks WHERE doc_id = ?) "
        "AND to_chunk IN (SELECT chunk_id FROM chunks WHERE doc_id = ?)",
        (from_doc_id, to_doc_id),
    ).fetchone()
    return row[0]


def verify_graph(ckb_path, expectations_path, manifest_path, raw_dir) -> list[ExpectationResult]:
    expectations = yaml.safe_load(Path(expectations_path).read_text(encoding="utf-8")) or []
    entries = load_manifest(manifest_path)
    by_slug = {e.doc_id: e for e in entries}

    results = []
    # closing() guarantees the sqlite handle is released even if an expectation
    # is malformed (Windows temp-dir cleanup fails on an open handle).
    with closing(ops.get_db(ckb_path)) as conn:
        for exp in expectations:
            name = exp["name"]
            from_slug = exp["from_slug"]
            to_slug = exp["to_slug"]
            min_edges = exp["min_edges"]
            pending = bool(exp.get("pending", False))

            from_did = doc_id_for_slug(from_slug, by_slug, raw_dir)
            to_did = doc_id_for_slug(to_slug, by_slug, raw_dir)

            if from_did is None or to_did is None:
                unresolved = from_slug if from_did is None else to_slug
                edges_observed = 0
                detail = (
                    f"could not derive a doc_id for slug {unresolved!r} "
                    "(no manifest entry or no fetch pin in raw_dir)"
                )
            else:
                edges_observed = _resolved_edge_count(conn, from_did, to_did)
                detail = (
                    f"{edges_observed} resolved reference edge(s) {from_slug} -> {to_slug} "
                    f"(need >= {min_edges})"
                )

            passed = edges_observed >= min_edges
            results.append(
                ExpectationResult(
                    name=name, from_slug=from_slug, to_slug=to_slug, min_edges=min_edges,
                    passed=passed, pending=pending, edges_observed=edges_observed, detail=detail,
                )
            )
    return results


def all_passed(results) -> bool:
    """True iff every ACTIVE (non-pending) expectation passed. A pending
    expectation is reported but never allowed to fail the gate."""
    return all(r.passed for r in results if not r.pending)
