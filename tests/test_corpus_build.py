"""Tests for the corpus build orchestrator (ingest/corpus_build.py).

Hermetic: every stage is injected, so no network, no Docling, no API calls, and
the real corpus is never touched.
"""

import pytest

from kbmcp.ingest import corpus_build as cb


class _Stages:
    """Recording doubles for every stage, with configurable failure."""

    def __init__(self, **fail):
        self.calls = []
        self.fail = fail

    def fetch(self, entries, raw_dir, cfg, force=False):
        self.calls.append("fetch")
        return self.fail.get("fetch", [])

    def build(self, **kw):
        self.calls.append("build")
        return self.fail.get("build", {"docs": 2, "chunks": 9, "contexts": 9,
                                       "contexts_missing": 0, "pruned": 0, "errors": []})

    def graph(self, **kw):
        self.calls.append("graph")
        return self.fail.get("graph", {"adjacent": 7, "references_resolved": 3,
                                       "references_unresolved": 1, "errors": []})

    def bm25(self, ckb_path, cfg):
        self.calls.append("bm25")
        return self.fail.get("bm25", 9)

    def gates(self, **kw):
        self.calls.append("gates")
        return self.fail.get("gates", {"structure": (5, 5), "graph": (3, 3)})


def _args(**kw):
    import argparse
    d = dict(manifest="m.yaml", folder=None, ckb="ckb.sqlite", config="c.yaml",
             raw_dir="raw", parsed_dir="parsed", force=False, reparse=False,
             no_context=False, require_context=False, allow_tokenizer_drift=False,
             only=None, skip_fetch=False)
    d.update(kw)
    return argparse.Namespace(**d)


def test_stages_run_in_the_load_bearing_order(monkeypatch):
    """graph must follow build (build_ckb clears edges) and bm25 must follow both
    (build_ckb invalidates the index whenever it writes chunks)."""
    s = _Stages()
    out = cb.run(_args(), stages=s)
    assert s.calls == ["fetch", "build", "graph", "bm25", "gates"]
    assert out["errors"] == []


def test_a_failing_stage_does_not_stop_later_stages():
    s = _Stages(build={"docs": 1, "chunks": 3, "contexts": 0, "contexts_missing": 3,
                       "pruned": 0, "errors": [{"doc_id": "x", "error": "boom"}]})
    out = cb.run(_args(), stages=s)
    assert s.calls == ["fetch", "build", "graph", "bm25", "gates"]
    assert out["errors"]


def test_exit_code_is_1_when_any_stage_errored():
    s = _Stages(build={"docs": 1, "chunks": 3, "contexts": 0, "contexts_missing": 3,
                       "pruned": 0, "errors": [{"doc_id": "x", "error": "boom"}]})
    assert cb.exit_code(cb.run(_args(), stages=s)) == 1


def test_exit_code_is_0_on_a_clean_run():
    assert cb.exit_code(cb.run(_args(), stages=_Stages())) == 0


def test_fetch_errors_are_collected_by_name():
    class R:
        def __init__(self, d, st, err=None):
            self.doc_id, self.status, self.error = d, st, err
    s = _Stages(fetch=[R("good", "ok"), R("dead-link", "error", "404")])
    out = cb.run(_args(), stages=s)
    assert any("dead-link" in str(e) for e in out["errors"])


def test_skip_fetch_does_not_call_the_fetcher():
    s = _Stages()
    cb.run(_args(skip_fetch=True), stages=s)
    assert "fetch" not in s.calls


def test_manifest_and_folder_together_is_an_error():
    with pytest.raises(cb.CorpusBuildError):
        cb.run(_args(folder="somewhere"), stages=_Stages())


def test_summary_names_every_failed_source(capsys):
    s = _Stages(build={"docs": 1, "chunks": 3, "contexts": 0, "contexts_missing": 3,
                       "pruned": 0,
                       "errors": [{"doc_id": "alpha", "error": "boom"},
                                  {"doc_id": "beta", "error": "bang"}]})
    out = cb.run(_args(), stages=s)
    cb.print_summary(out)
    err = capsys.readouterr().err
    assert "alpha" in err and "beta" in err


def test_summary_goes_to_stderr_not_stdout(capsys):
    out = cb.run(_args(), stages=_Stages())
    cb.print_summary(out)
    cap = capsys.readouterr()
    assert cap.out == ""
    assert "bm25" in cap.err


# ---------------------------------------------------------------------------
# Coverage for paths the plan's own nine tests do not reach: folder mode, the
# gates' skip-when-absent rule, the tokenizer preflight, and the CLI surface.
# ---------------------------------------------------------------------------


def test_folder_mode_materialises_a_manifest_beside_the_ckb(safe_tmp_path):
    """build_ckb and build_graph both take a manifest PATH, so folder mode must
    write one -- next to the CKB, never inside the user's document folder."""
    docs = safe_tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("alpha", encoding="utf-8")
    (docs / "b.md").write_text("beta", encoding="utf-8")
    ckb = safe_tmp_path / "out" / "t.sqlite"

    s = _Stages()
    cb.run(_args(manifest=None, folder=str(docs), ckb=str(ckb)), stages=s)

    written = ckb.parent / "folder-manifest.yaml"
    assert written.exists(), "folder mode must materialise a manifest"
    assert not (docs / "folder-manifest.yaml").exists(), \
        "the user's document folder must not be written into"
    import yaml
    rows = yaml.safe_load(written.read_text(encoding="utf-8"))
    assert {r["doc_id"] for r in rows} == {"a", "b"}


def test_folder_mode_failure_propagates(safe_tmp_path):
    """An empty folder must not quietly produce an empty CKB."""
    empty = safe_tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(Exception):
        cb.run(_args(manifest=None, folder=str(empty),
                     ckb=str(safe_tmp_path / "t.sqlite")), stages=_Stages())


def test_gates_are_skipped_when_their_fixture_files_are_absent(safe_tmp_path):
    """A BYO corpus legitimately has neither fixture file; that is a skip, not
    a failure."""
    st = cb._DefaultStages()
    result = st.gates(ckb_path=str(safe_tmp_path / "nope.sqlite"),
                      manifest_path=str(safe_tmp_path / "m.yaml"),
                      raw_dir=str(safe_tmp_path),
                      fixtures_path=str(safe_tmp_path / "absent-fixtures.yaml"),
                      expectations_path=str(safe_tmp_path / "absent-exp.yaml"))
    assert result == {"structure": None, "graph": None}


def test_tokenizer_preflight_is_skipped_when_no_config_file(safe_tmp_path):
    """run() must stay network-free for injected-stage callers: with no config
    on disk there is no pin to protect, so no hub round-trip happens."""
    called = []
    import kbmcp.ingest.chunk as chunk_mod
    orig = chunk_mod.check_tokenizer_revision
    try:
        chunk_mod.check_tokenizer_revision = lambda *a, **k: called.append(1)
        cb.run(_args(config=str(safe_tmp_path / "absent.yaml")), stages=_Stages())
    finally:
        chunk_mod.check_tokenizer_revision = orig
    assert called == []


def test_cli_rejects_manifest_and_folder_together_without_a_traceback():
    """main() must turn the error into an exit code, not a stack trace."""
    assert cb.main(["--manifest", "m.yaml", "--folder", "f"]) == 1


def test_arg_parser_defaults_manifest_to_none_so_folder_mode_is_reachable():
    """If --manifest defaulted to a path, every --folder run would trip the
    mutual-exclusion check."""
    a = cb.build_arg_parser().parse_args(["--folder", "somewhere"])
    assert a.manifest is None and a.folder == "somewhere"


def test_summary_reports_skipped_gates_distinctly_from_failed_ones(capsys):
    s = _Stages(gates={"structure": None, "graph": None})
    cb.print_summary(cb.run(_args(), stages=s))
    err = capsys.readouterr().err
    assert "skip" in err.lower()


def test_neither_manifest_nor_folder_falls_back_to_the_standard_manifest():
    """The fallback lives in run(), not argparse, so --folder alone works."""
    s = _Stages()
    a = cb.build_arg_parser().parse_args([])
    cb.run(a, stages=s)
    assert a.manifest == cb.DEFAULT_MANIFEST


def test_folder_mode_never_runs_the_eval_gates(safe_tmp_path):
    """corpus/benchmark/*.yaml are ground truth for THIS project's manifest.
    A BYO user working from a clone has those files on disk, so an
    existence-check alone would assert our corpus's facts against their
    documents and fail every run. Measured before the fix: a 3-file folder
    build reported structure 0/5, graph 0/3, exit 1."""
    docs = safe_tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("alpha", encoding="utf-8")

    class _GateSpy(_Stages):
        def gates(self, **kw):
            self.calls.append("gates")
            raise AssertionError("gates must not run in folder mode")

    s = _GateSpy()
    out = cb.run(_args(manifest=None, folder=str(docs),
                       ckb=str(safe_tmp_path / "out" / "t.sqlite")), stages=s)
    assert out["stages"]["gates"] == {"structure": None, "graph": None}
    assert not any(e["stage"] == "gates" for e in out["errors"])


def test_manifest_mode_still_runs_the_gates():
    """The folder-mode skip must not disable gates for the shipped corpus."""
    s = _Stages()
    cb.run(_args(), stages=s)
    assert "gates" in s.calls
