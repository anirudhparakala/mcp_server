from kbmcp.ingest import fetch
from kbmcp.ingest.manifest import SourceEntry


def _entry(**kw):
    base = dict(doc_id="d", url="https://example.com/x", format="html",
                license="x", license_ok=True, version="v1")
    base.update(kw)
    return SourceEntry(**base)


def test_arg_parser_only_is_repeatable_and_dry_run_is_flag():
    args = fetch.build_arg_parser().parse_args(["--only", "a", "--only", "b", "--dry-run"])
    assert args.only == ["a", "b"]
    assert args.dry_run is True
    assert args.manifest == "corpus/manifest.yaml"


def test_plan_fetches_filters_by_only_and_uses_arxiv_html_target():
    entries = [
        _entry(doc_id="wiki", url="https://en.wikipedia.org/w/index.php?oldid=1"),
        _entry(doc_id="paper", url="https://arxiv.org/abs/1706.03762", format="pdf", version="v7"),
    ]
    assert fetch.plan_fetches(entries, only={"paper"}) == [
        ("paper", "arxiv", "https://arxiv.org/html/1706.03762v7")
    ]


def test_plan_fetches_generic_target_is_the_url():
    entries = [_entry(doc_id="wiki", url="https://en.wikipedia.org/w/index.php?oldid=1")]
    assert fetch.plan_fetches(entries) == [
        ("wiki", "generic", "https://en.wikipedia.org/w/index.php?oldid=1")
    ]
