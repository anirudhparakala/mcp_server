"""Tests for the verify-gate CLIs.

Before this task, `python -m kbmcp.ingest.verify_structure` exited 0 having done
nothing, because the module had no __main__ block -- a false pass.
"""

import pytest

from kbmcp.db import ops
from kbmcp.db.schema import create_all_tables
from kbmcp.ingest import verify_graph as vg
from kbmcp.ingest import verify_structure as vs
from kbmcp.models.ids import doc_id as mk_doc_id


DID = mk_doc_id("u", "v1")   # fixtures key on the sha256 doc_id, not the slug


def _ckb_with_chunk(path, text):
    conn = ops.get_db(path)
    try:
        create_all_tables(conn)
        ops.insert_source(conn, canonical_url="u", url_original="u", domain="d",
                          format="html", license="x", license_ok=True, version="v1")
        ops.insert_doc(conn, doc_id=DID, canonical_url="u", domain="d", format="html")
        ops.insert_chunk(conn, chunk_id="c0", doc_id=DID, chunk_index=0,
                         chunk_type="text", text=text)
    finally:
        conn.close()


def test_structure_cli_returns_0_when_fixtures_pass(safe_tmp_path):
    ckb = safe_tmp_path / "t.sqlite"
    _ckb_with_chunk(ckb, "the quick brown fox")
    fx = safe_tmp_path / "fx.yaml"
    fx.write_text(
        f"- name: f1\n  slug: u\n  doc_id: {DID}\n  must_contain: ['quick brown']\n",
        encoding="utf-8")
    assert vs.main(["--ckb", str(ckb), "--fixtures", str(fx)]) == 0


def test_structure_cli_returns_1_when_a_fixture_fails(safe_tmp_path):
    ckb = safe_tmp_path / "t.sqlite"
    _ckb_with_chunk(ckb, "the quick brown fox")
    fx = safe_tmp_path / "fx.yaml"
    fx.write_text(
        f"- name: f1\n  slug: u\n  doc_id: {DID}\n  must_contain: ['ABSENT STRING']\n",
        encoding="utf-8")
    assert vs.main(["--ckb", str(ckb), "--fixtures", str(fx)]) == 1


def test_structure_cli_writes_nothing_to_stdout(safe_tmp_path, capsys):
    """House rule: stdio JSON-RPC transport -- library output goes to stderr."""
    ckb = safe_tmp_path / "t.sqlite"
    _ckb_with_chunk(ckb, "the quick brown fox")
    fx = safe_tmp_path / "fx.yaml"
    fx.write_text(
        f"- name: f1\n  slug: u\n  doc_id: {DID}\n  must_contain: ['quick']\n",
        encoding="utf-8")
    vs.main(["--ckb", str(ckb), "--fixtures", str(fx)])
    assert capsys.readouterr().out == ""


def test_graph_cli_returns_0_on_an_empty_expectations_file(safe_tmp_path):
    ckb = safe_tmp_path / "t.sqlite"
    _ckb_with_chunk(ckb, "text")
    exp = safe_tmp_path / "exp.yaml"
    exp.write_text("[]\n", encoding="utf-8")
    man = safe_tmp_path / "m.yaml"
    man.write_text(
        "- doc_id: u\n  url: u\n  domain: d\n  format: html\n"
        "  license: x\n  license_ok: true\n  version: v1\n", encoding="utf-8")
    raw = safe_tmp_path / "raw"
    raw.mkdir()
    (raw / "u.meta.json").write_text('{"resolved_version": "v1"}', encoding="utf-8")
    assert vg.main(["--ckb", str(ckb), "--expectations", str(exp),
                    "--manifest", str(man), "--raw-dir", str(raw)]) == 0
