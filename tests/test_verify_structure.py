from kbmcp.ingest import verify_structure as vs
from kbmcp.db import ops
from kbmcp.db.schema import create_all_tables


def _ckb_with_table(path):
    conn = ops.get_db(path); create_all_tables(conn)
    ops.insert_source(conn, canonical_url="u", url_original="u", domain="fitness", format="pdf",
                      license="x", license_ok=True, version="v1")
    ops.insert_doc(conn, doc_id="D", canonical_url="u", domain="fitness", format="pdf", title="DGA")
    ops.insert_chunk(conn, chunk_id="c0", doc_id="D", chunk_index=0,
                     text="Protein RDA 0.8 g/kg per day", chunk_type="table",
                     table={"columns": ["Nutrient", "RDA"], "data": [["Protein", "0.8 g/kg"]]})
    conn.close()


def test_fixture_passes_when_table_snippet_present(safe_tmp_path):
    ckb = safe_tmp_path / "ckb.sqlite"; _ckb_with_table(ckb)
    fx = safe_tmp_path / "fx.yaml"
    fx.write_text('- name: t\n  doc_id: D\n  must_contain_in_table: ["0.8", "Protein"]\n', encoding="utf-8")
    res = vs.verify_structure(ckb, fx)
    assert vs.all_passed(res) and res[0].passed


def test_fixture_fails_when_snippet_missing(safe_tmp_path):
    ckb = safe_tmp_path / "ckb.sqlite"; _ckb_with_table(ckb)
    fx = safe_tmp_path / "fx.yaml"
    fx.write_text('- name: t\n  doc_id: D\n  must_contain: ["1.6 g/kg (absent)"]\n', encoding="utf-8")
    res = vs.verify_structure(ckb, fx)
    assert not vs.all_passed(res) and not res[0].passed
