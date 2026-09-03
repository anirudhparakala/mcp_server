"""Structure-fixtures gate (stage G): assert the 5 structure-critical docs'
tables/cross-references survived parse+chunk. The build fails if any regress.
"""

from contextlib import closing
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


def verify_structure(ckb_path, fixtures_path) -> list[FixtureResult]:
    fixtures = yaml.safe_load(Path(fixtures_path).read_text(encoding="utf-8")) or []
    results = []
    # closing() guarantees the sqlite handle is released even if a fixture is
    # malformed (Windows temp-dir cleanup fails on an open handle).
    with closing(ops.get_db(ckb_path)) as conn:
        for fx in fixtures:
            name = fx["name"]
            did = fx["doc_id"]
            need_text = fx.get("must_contain", [])
            need_table = fx.get("must_contain_in_table", [])
            if not (need_text or need_table):
                raise ValueError(
                    f"fixture {name!r} has no must_contain/must_contain_in_table assertions"
                )
            chunks = ops.get_chunks_for_doc(conn, did)
            blob_all = "\n".join(c["text"] for c in chunks)
            blob_tables = "\n".join(c["text"] for c in chunks if c["chunk_type"] == "table")
            missing = [s for s in need_text if s not in blob_all] + [
                s for s in need_table if s not in blob_tables
            ]
            results.append(
                FixtureResult(
                    doc_id=did,
                    name=name,
                    passed=not missing,
                    detail="ok" if not missing else f"missing: {missing}",
                )
            )
    return results


def all_passed(results) -> bool:
    return all(r.passed for r in results)
