"""Corpus build orchestrator (Stage: end-to-end) -- turns a manifest, or a plain
folder of documents, into a queryable CKB.

Design: docs/superpowers/specs/2026-09-02-phase1-corpus-build-design.md

This module owns SEQUENCING, the summary, and exit codes -- nothing else. Every
stage (fetch, build, graph, bm25, the two gates) already exists as a tested
library API; this file wires them together in the load-bearing order and never
reimplements their logic.

Stage order (see spec Sec.4 -- both orderings are load-bearing):
  fetch -> build_ckb -> graph -> bm25 -> gates
  1. graph must follow build_ckb: build_ckb --force deletes and rewrites every
     chunk, and _delete_doc clears edges first. Building the graph earlier
     leaves it empty.
  2. bm25 must follow both: build_ckb calls bm25_store.invalidate(conn)
     whenever it writes chunks, so the index must be (re)built after the
     chunks are final.

Failure policy (spec Sec.6): continue through every stage, exit non-zero, print
a loud itemized summary. A per-source failure must never discard the rest of a
BYO corpus; a partial CKB is still written and usable.

`run()` is fully hermetic when a `stages` double is injected (tests): it never
touches the manifest or config files itself for the stage calls -- it only
threads PATHS through, and each `_DefaultStages` method loads exactly the
manifest/config slice the real function underneath it needs, right before
delegating. The one direct read `run()` performs is a config-existence-guarded
tokenizer revision check, which is skipped (not defaulted-and-called) when the
config file is absent, so it makes no network call in that case either.
"""

import argparse
import sys
from pathlib import Path

import yaml

from ..config import load_corpus_config
from ..db import ops
from ..db.schema import create_all_tables
from ..index.bm25_store import BM25Store
from . import build as build_mod
from . import chunk as chunk_mod
from . import folder_source
from . import graph as graph_mod
from . import manifest as manifest_mod
from . import verify_graph
from . import verify_structure

DEFAULT_MANIFEST = "corpus/manifest.yaml"

# The shipped manifest, anchored to the PACKAGE rather than the process cwd.
# corpus_build.py lives at <root>/src/kbmcp/ingest/, so parents[3] is the repo
# root. Resolving DEFAULT_MANIFEST against cwd instead would make gate
# applicability depend on where the command was launched from: running the
# shipped build from outside the repo with an absolute --manifest silently
# skipped both gates, which is a worse failure than the one that motivated
# the check. On an installed (non-repo) layout this path simply will not
# exist, and the fixture-existence check skips the gates anyway.
_SHIPPED_MANIFEST = Path(__file__).resolve().parents[3] / "corpus" / "manifest.yaml"
# Anchored to the package for the same reason as _SHIPPED_MANIFEST: as bare
# cwd-relative literals these silently "did not exist" whenever the build was
# launched from outside the repo root, so both gates skipped and the run
# reported a clean exit while verifying nothing.
_BENCH_DIR = Path(__file__).resolve().parents[3] / "corpus" / "benchmark"
_STRUCTURE_FIXTURES_DEFAULT = str(_BENCH_DIR / "structure_fixtures.yaml")
_GRAPH_EXPECTATIONS_DEFAULT = str(_BENCH_DIR / "graph_expectations.yaml")


class CorpusBuildError(ValueError):
    """Raised for an invalid orchestrator invocation (e.g. --manifest + --folder)."""


class _DefaultStages:
    """Thin wrappers over the real stage APIs.

    Each method loads exactly the manifest/config slice the real function
    underneath it needs, then delegates -- no business logic of its own.
    """

    def fetch(self, manifest_path, raw_dir, config_path, force=False):
        # fetch_all's signature has no manifest_path slot -- it needs a
        # pre-loaded entries list -- so this wrapper does one extra step
        # (load, then delegate) that the other wrappers don't need.
        from . import fetch as fetch_mod
        entries = manifest_mod.load_manifest(manifest_path)
        cfg = load_corpus_config(config_path).fetch
        return fetch_mod.fetch_all(entries, raw_dir, cfg, force=force)

    def build(self, **kw):
        config_path = kw.pop("config_path")
        cfg = load_corpus_config(config_path).raw
        return build_mod.build_ckb(cfg=cfg, **kw)

    def graph(self, **kw):
        config_path = kw.pop("config_path")
        cfg = load_corpus_config(config_path).graph
        return graph_mod.build_graph(cfg=cfg, **kw)

    def bm25(self, ckb_path, config_path):
        cfg = load_corpus_config(config_path).bm25
        conn = ops.get_db(ckb_path)
        try:
            create_all_tables(conn)
            return BM25Store(conn, cfg).build()
        finally:
            conn.close()

    def gates(self, *, ckb_path, manifest_path, raw_dir, applicable=True,
              fixtures_path=_STRUCTURE_FIXTURES_DEFAULT,
              expectations_path=_GRAPH_EXPECTATIONS_DEFAULT):
        """Run each gate only when its fixture file exists (spec Sec.8) AND the
        fixtures describe the corpus being built (`applicable`, see
        _gates_apply_to). A missing or inapplicable fixture is a skip-with-note,
        not a failure -- a BYO corpus legitimately has no ground truth."""
        if not applicable:
            return {"structure": None, "graph": None}

        result = {}

        if Path(fixtures_path).exists():
            fixture_results = verify_structure.verify_structure(ckb_path, fixtures_path)
            result["structure"] = (sum(r.passed for r in fixture_results), len(fixture_results))
        else:
            result["structure"] = None

        if Path(expectations_path).exists():
            exp_results = verify_graph.verify_graph(
                ckb_path, expectations_path, manifest_path, raw_dir)
            active = [r for r in exp_results if not r.pending]
            result["graph"] = (sum(r.passed for r in active), len(active))
        else:
            result["graph"] = None

        return result


def _gates_apply_to(args, manifest_path) -> bool:
    """True only when corpus/benchmark/*.yaml describe the corpus being built.

    Those fixtures are ground truth for THIS project's shipped manifest --
    specific doc_ids, specific cited articles. A BYO user working from a clone of
    this repo has them on disk, so keying the gates on mere file existence
    asserts our corpus's facts about their documents and fails every run
    (measured: a 3-file folder build reported structure 0/5, graph 0/3, exit 1).
    Folder mode never matches; neither does --manifest pointed at someone else's
    manifest. This deliberately narrows spec Sec.8, which only anticipated
    fixtures being ABSENT.
    """
    if args.folder or not manifest_path:
        return False
    try:
        return Path(manifest_path).resolve() == _SHIPPED_MANIFEST.resolve()
    except OSError:
        return False


def run(args, *, stages=None) -> dict:
    """Run the full fetch -> build -> graph -> bm25 -> gates pipeline.

    Returns {"stages": {...}, "errors": [...]}. Every stage runs to
    completion regardless of earlier failures; `errors` accumulates a
    normalized {"stage", "doc_id", "error"} entry per failure, from any
    stage, so `exit_code()`/`print_summary()` can report honestly.
    """
    if args.manifest and args.folder:
        raise CorpusBuildError(
            "--manifest and --folder are mutually exclusive; pass exactly one")

    # Neither given: fall back to the standard manifest. The fallback lives here
    # rather than in argparse's default so that --folder alone does not collide
    # with a pre-populated --manifest (which made folder mode unusable). Held in
    # a local rather than written back to args -- run() must not mutate its
    # caller's namespace.
    requested_manifest = args.manifest or (None if args.folder else DEFAULT_MANIFEST)

    stages = stages or _DefaultStages()
    out = {"stages": {}, "errors": []}

    if args.folder:
        rows = folder_source.discover(args.folder)
        out_path = Path(args.ckb).parent / "folder-manifest.yaml"
        manifest_path = str(folder_source.write_manifest(rows, out_path))
    else:
        manifest_path = requested_manifest

    Path(args.ckb).parent.mkdir(parents=True, exist_ok=True)

    # Tokenizer revision check -- before any stage, TokenizerDriftError
    # propagates straight out of run() (main() catches it). Skipped, not
    # defaulted-and-called, when the config file is absent: there is no pin
    # to protect, and skipping keeps run() network-free for injected-stage
    # (test) callers instead of making a pointless hub round-trip.
    if Path(args.config).exists():
        chunk_cfg = load_corpus_config(args.config).chunk
        chunk_mod.check_tokenizer_revision(chunk_cfg, allow_drift=args.allow_tokenizer_drift)

    raw_dir = args.raw_dir
    parsed_dir = args.parsed_dir
    ckb_path = args.ckb
    only = set(args.only) if args.only else None
    if only and manifest_path and Path(manifest_path).exists():
        # An --only typo would otherwise select no sources, build nothing, and
        # exit 0 -- the same silent-no-op family as the prune wipe above.
        try:
            known = {e.doc_id for e in manifest_mod.load_manifest(manifest_path)}
        except Exception:  # noqa: BLE001 -- a bad manifest is the stages' error to report
            known = None
        if known is not None:
            unknown = sorted(only - known)
            if unknown:
                raise CorpusBuildError(
                    f"--only names {len(unknown)} slug(s) absent from the manifest: "
                    f"{', '.join(unknown)}. Nothing would be built.")

    # 1. fetch
    if args.skip_fetch:
        out["stages"]["fetch"] = None
    else:
        try:
            fetch_results = stages.fetch(manifest_path, raw_dir, args.config, force=args.force)
        except Exception as exc:  # noqa: BLE001 -- one bad stage must not abort the run
            fetch_results = []
            out["errors"].append({"stage": "fetch", "doc_id": None, "error": str(exc)})
        out["stages"]["fetch"] = fetch_results
        for r in fetch_results:
            if r.status == "error":
                out["errors"].append({"stage": "fetch", "doc_id": r.doc_id, "error": r.error})

    # 2. build
    try:
        build_result = stages.build(
            manifest_path=manifest_path, raw_dir=raw_dir, parsed_dir=parsed_dir,
            ckb_path=ckb_path, config_path=args.config, only=only, force=args.force,
            reparse=args.reparse, no_context=args.no_context,
            require_context=args.require_context,
        )
    except Exception as exc:  # noqa: BLE001 -- one bad stage must not abort the run
        build_result = {"docs": 0, "chunks": 0, "contexts": 0, "contexts_missing": 0,
                         "pruned": 0, "errors": [{"doc_id": "*", "error": str(exc)}]}
    out["stages"]["build"] = build_result
    _collect_errors(out, "build", build_result.get("errors", []))

    # 3. graph -- must follow build (build_ckb clears edges on --force/prune).
    try:
        graph_result = stages.graph(
            ckb_path=ckb_path, manifest_path=manifest_path, raw_dir=raw_dir,
            config_path=args.config,
        )
    except Exception as exc:  # noqa: BLE001 -- one bad stage must not abort the run
        graph_result = {"adjacent": 0, "references_resolved": 0, "references_unresolved": 0,
                         "errors": [str(exc)]}
    out["stages"]["graph"] = graph_result
    _collect_errors(out, "graph", graph_result.get("errors", []))

    # 4. bm25 -- must follow both (build_ckb invalidates it whenever it writes chunks).
    try:
        bm25_count = stages.bm25(ckb_path, args.config)
    except Exception as exc:  # noqa: BLE001 -- one bad stage must not abort the run
        bm25_count = 0
        out["errors"].append({"stage": "bm25", "doc_id": None, "error": str(exc)})
    out["stages"]["bm25"] = bm25_count

    # 5. gates
    #
    # Run only when the fixtures actually describe the corpus being built --
    # see _gates_apply_to.
    try:
        gates_result = stages.gates(
            ckb_path=ckb_path, manifest_path=manifest_path, raw_dir=raw_dir,
            applicable=_gates_apply_to(args, manifest_path))
    except Exception as exc:  # noqa: BLE001 -- one bad stage must not abort the run
        gates_result = {}
        out["errors"].append({"stage": "gates", "doc_id": None, "error": str(exc)})
    out["stages"]["gates"] = gates_result

    # A run that indexed nothing is a failure even when every stage "succeeded".
    # build_ckb records no error for a document that yields zero chunks, and with
    # the gates skipped for a BYO corpus nothing else would catch it -- a folder
    # of unparseable files would report "3 docs / 0 chunks / 0 indexed / exit 0",
    # exactly the partial-state-that-looks-complete this pipeline exists to avoid.
    # Judge the END STATE, not this run's delta. Checking only "docs>0 and
    # chunks==0" missed the destructive case: build_ckb prunes every doc absent
    # from the manifest, so an empty or shrunken manifest WIPES a populated CKB
    # and reports docs=0/chunks=0 -- which the delta check waved through as
    # "nothing was asked for". Measured: a seeded 1-doc CKB went to 0/0 at exit 0.
    # stages.bm25 returns the whole CKB's indexed count, so it is the honest
    # end-state signal.
    indexed = out["stages"].get("bm25")
    if indexed == 0:
        out["errors"].append({
            "stage": "bm25", "doc_id": None,
            "error": ("the CKB contains 0 indexed chunks after this run -- nothing is "
                      "retrievable. If you expected content, check that the manifest lists "
                      "your sources (a manifest missing a source PRUNES it from the CKB) "
                      "and that the inputs parsed."),
        })
    for name, gate in (gates_result or {}).items():
        if gate is not None and gate[0] < gate[1]:
            out["errors"].append({
                "stage": "gates", "doc_id": name,
                "error": f"{name} gate: {gate[0]}/{gate[1]} passed",
            })

    return out


def _collect_errors(out: dict, stage: str, errors) -> None:
    """Normalize a stage's own `errors` list into out["errors"] entries."""
    for e in errors or []:
        if isinstance(e, dict):
            out["errors"].append({
                "stage": stage, "doc_id": e.get("doc_id"), "error": e.get("error", str(e)),
            })
        else:
            out["errors"].append({"stage": stage, "doc_id": None, "error": str(e)})


def exit_code(result: dict) -> int:
    return 1 if result.get("errors") else 0


def print_summary(result: dict) -> None:
    """Write the spec Sec.6 summary block to stderr -- never stdout (stdio
    JSON-RPC transport; a stray stdout write corrupts the protocol)."""
    stages = result.get("stages", {})

    fetch = stages.get("fetch")
    if fetch is None:
        print("fetch        skipped (--skip-fetch)", file=sys.stderr)
    else:
        failed = sum(1 for r in fetch if r.status == "error")
        print(f"fetch        {len(fetch) - failed} ok    {failed} failed", file=sys.stderr)

    build = stages.get("build") or {}
    print(f"build        {build.get('docs', 0)} docs  {build.get('chunks', 0)} chunks   "
          f"{build.get('contexts', 0)} contexts", file=sys.stderr)

    graph = stages.get("graph") or {}
    resolved = graph.get("references_resolved", 0)
    unresolved = graph.get("references_unresolved", 0)
    edges = graph.get("adjacent", 0) + resolved + unresolved
    print(f"graph        {edges} edges  ({resolved} resolved, {unresolved} unresolved)",
          file=sys.stderr)

    print(f"bm25         {stages.get('bm25', 0)} indexed", file=sys.stderr)

    build = stages.get("build") or {}
    missing = build.get("contexts_missing") or 0
    pruned = build.get("pruned") or 0
    if missing:
        print(f"             {missing} chunk(s) WITHOUT context -- set ANTHROPIC_API_KEY "
              "and rebuild with --force to add them", file=sys.stderr)
    if pruned:
        print(f"             {pruned} document(s) PRUNED (absent from the manifest and "
              "removed from the CKB)", file=sys.stderr)

    gates = stages.get("gates") or {}
    structure_gate = gates.get("structure")
    if structure_gate is None:
        print("structure    skipped (the benchmark fixtures describe the shipped "
              "manifest, not this corpus)", file=sys.stderr)
    else:
        print(f"structure    {structure_gate[0]}/{structure_gate[1]}"
              "   (ground truth for the shipped 55-source manifest)", file=sys.stderr)

    graph_gate = gates.get("graph")
    if graph_gate is None:
        print("graph gate   skipped (the benchmark expectations describe the shipped "
              "manifest, not this corpus)", file=sys.stderr)
    else:
        print(f"graph gate   {graph_gate[0]}/{graph_gate[1]}", file=sys.stderr)

    errors = result.get("errors") or []
    if errors:
        print("", file=sys.stderr)
        print(f"failed sources ({len(errors)}):", file=sys.stderr)
        for e in errors:
            if isinstance(e, dict):
                stage = e.get("stage", "?")
                doc = e.get("doc_id") or "-"
                msg = e.get("error", "")
                print(f"  [{stage}] {doc}: {msg}", file=sys.stderr)
            else:
                print(f"  {e}", file=sys.stderr)

    print(f"exit {exit_code(result)}", file=sys.stderr)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python scripts/build_corpus.py",
        description="Build a CKB from a manifest or a folder of documents.",
    )
    # Defaults to None, NOT to the manifest path: a non-None default would make
    # every --folder invocation trip the mutual-exclusion check in run(), which
    # made folder mode -- the headline BYO feature -- unusable from the CLI.
    # run() substitutes DEFAULT_MANIFEST when neither option is given.
    p.add_argument("--manifest", default=None,
                    help=f"manifest path (default {DEFAULT_MANIFEST}; "
                         "mutually exclusive with --folder)")
    p.add_argument("--folder", default=None, help="BYO local-folder mode")
    p.add_argument("--ckb", default="ckb/ckb.sqlite")
    p.add_argument("--config", default="config/corpus_config.yaml")
    p.add_argument("--raw-dir", dest="raw_dir", default="corpus/raw")
    p.add_argument("--parsed-dir", dest="parsed_dir", default="corpus/parsed")
    p.add_argument("--force", action="store_true",
                    help="rebuild the CKB from the committed parse")
    p.add_argument("--reparse", action="store_true",
                    help="re-run Docling; overwrites the committed parse")
    p.add_argument("--no-context", dest="no_context", action="store_true",
                    help="skip Contextual Retrieval generation (pins still applied)")
    p.add_argument("--require-context", dest="require_context", action="store_true",
                    help="fail if any chunk ends up without a context")
    p.add_argument("--allow-tokenizer-drift", dest="allow_tokenizer_drift", action="store_true",
                    help="accept a tokenizer revision mismatch")
    p.add_argument("--only", action="append", default=None,
                    help="restrict the build stage to this doc_id slug (repeatable)")
    p.add_argument("--skip-fetch", dest="skip_fetch", action="store_true",
                    help="use the existing raw cache; offline rebuild")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        result = run(args)
    except (CorpusBuildError, chunk_mod.TokenizerDriftError,
            folder_source.FolderSourceError, manifest_mod.ManifestError,
            yaml.YAMLError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print_summary(result)
    return exit_code(result)


if __name__ == "__main__":
    raise SystemExit(main())
