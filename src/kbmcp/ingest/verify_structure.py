"""Structure-fixtures gate (stage G): assert the 5 structure-critical docs'
tables/cross-references survived parse+chunk. The build fails if any regress.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

from ..db import ops


@dataclass
class FixtureResult:
    doc_id: str
    name: str
    passed: bool
    detail: str


def verify_structure(ckb_path, fixtures_path) -> list:
    fixtures = yaml.safe_load(Path(fixtures_path).read_text(encoding="utf-8")) or []
    conn = ops.get_db(ckb_path)
    results = []
    for fx in fixtures:
        did = fx["doc_id"]
        chunks = ops.get_chunks_for_doc(conn, did)
        blob_all = "\n".join(c["text"] for c in chunks)
        blob_tables = "\n".join(c["text"] for c in chunks if c["chunk_type"] == "table")
        need_text = fx.get("must_contain", [])
        need_table = fx.get("must_contain_in_table", [])
        missing = [s for s in need_text if s not in blob_all] + [s for s in need_table if s not in blob_tables]
        results.append(FixtureResult(
            doc_id=did, name=fx["name"], passed=not missing,
            detail="ok" if not missing else f"missing: {missing}",
        ))
    conn.close()
    return results


def all_passed(results) -> bool:
    return all(r.passed for r in results)
