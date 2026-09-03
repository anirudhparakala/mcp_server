from pathlib import Path

from kbmcp.ingest import fetch
from kbmcp.ingest.manifest import SourceEntry

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG = str(REPO_ROOT / "config" / "corpus_config.yaml")


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


def test_main_real_run_pins_local_source_and_returns_zero(safe_tmp_path):
    src = safe_tmp_path / "doc.pdf"
    src.write_bytes(b"%PDF real")
    raw = safe_tmp_path / "raw"
    manifest = safe_tmp_path / "m.yaml"
    manifest.write_text(
        f"- doc_id: local-1\n"
        f"  url: {src.as_uri()}\n"
        f"  format: pdf\n"
        f"  license: user-owned\n"
        f"  license_ok: true\n"
        f"  version: auto\n",
        encoding="utf-8",
    )
    rc = fetch.main(["--manifest", str(manifest), "--raw-dir", str(raw), "--config", CONFIG])
    assert rc == 0
    assert (raw / "local-1.pdf").read_bytes() == b"%PDF real"
    assert fetch.read_meta(raw, "local-1")["resolved_version"].startswith("sha256:")


def test_main_real_run_returns_one_on_fetch_error(safe_tmp_path):
    raw = safe_tmp_path / "raw"
    manifest = safe_tmp_path / "m.yaml"
    manifest.write_text(
        f"- doc_id: gone\n"
        f"  url: {(safe_tmp_path / 'missing.pdf').as_uri()}\n"
        f"  format: pdf\n"
        f"  license: x\n"
        f"  license_ok: true\n"
        f"  version: auto\n",
        encoding="utf-8",
    )
    rc = fetch.main(["--manifest", str(manifest), "--raw-dir", str(raw), "--config", CONFIG])
    assert rc == 1


def test_main_unknown_only_slug_returns_one(safe_tmp_path, capsys):
    manifest = safe_tmp_path / "m.yaml"
    manifest.write_text(
        f"- doc_id: real\n"
        f"  url: {(safe_tmp_path / 'x.pdf').as_uri()}\n"
        f"  format: pdf\n"
        f"  license: x\n"
        f"  license_ok: true\n"
        f"  version: auto\n",
        encoding="utf-8",
    )
    rc = fetch.main(["--manifest", str(manifest), "--only", "typo", "--config", CONFIG])
    assert rc == 1
    assert "unknown" in capsys.readouterr().err.lower()
