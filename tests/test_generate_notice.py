"""Tests for scripts/generate_notice.py — the NOTICE generator.

The NOTICE is the attribution artifact for a published corpus, so the property
that matters most is that no source can be silently dropped: a NOTICE missing a
source is worse than no NOTICE, because it looks complete.

Hermetic: every test builds its own entry list. Never reads corpus/manifest.yaml
except in the one test that deliberately checks the shipped corpus is covered.
"""

import importlib.util
from pathlib import Path

import pytest
import yaml

_SPEC = importlib.util.spec_from_file_location(
    "generate_notice", Path(__file__).resolve().parents[1] / "scripts" / "generate_notice.py"
)
gn = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gn)


def _entry(doc_id, license_="CC BY 4.0", url=None, domain="ai"):
    return {"doc_id": doc_id, "url": url or f"https://example.test/{doc_id}",
            "license": license_, "domain": domain}


def test_every_source_appears_in_the_notice():
    """The load-bearing property: a source that exists must be attributed."""
    entries = [_entry(f"src-{i}", license_=f"License {i % 3}") for i in range(12)]
    text = gn.build_notice(entries)
    for e in entries:
        assert e["doc_id"] in text
        assert e["url"] in text


def test_source_count_is_stated_and_matches():
    """A stated count that disagrees with the body would mask a dropped source."""
    entries = [_entry(f"src-{i}") for i in range(7)]
    text = gn.build_notice(entries)
    assert "7 source" in text
    assert sum(text.count(e["doc_id"]) for e in entries) >= 7


def test_sources_are_grouped_under_their_license():
    """doc_ids here are deliberately distinctive: single letters matched prose in
    the preamble and made an earlier version of this test pass for the wrong reason."""
    entries = [_entry("zeta-one", "CC BY 4.0"), _entry("zeta-two", "CC BY 4.0"),
               _entry("omega-pd", "Public domain")]
    text = gn.build_notice(entries)
    cc_at = text.index("CC BY 4.0\n---")
    pd_at = text.index("Public domain\n---")
    assert text.index("zeta-one") > cc_at and text.index("zeta-two") > cc_at
    assert text.index("omega-pd") > pd_at
    # and each sits under its OWN heading, not merely after some heading
    assert text.index("omega-pd") - pd_at < 200


def test_output_is_deterministic_regardless_of_input_order():
    entries = [_entry("a", "L1"), _entry("b", "L2"), _entry("c", "L1")]
    assert gn.build_notice(entries) == gn.build_notice(list(reversed(entries)))


def test_missing_license_raises_rather_than_emitting_blank_attribution():
    """Silently attributing a source to '' is exactly the failure this guards."""
    with pytest.raises(gn.NoticeError):
        gn.build_notice([{"doc_id": "x", "url": "https://example.test/x", "domain": "ai"}])
    with pytest.raises(gn.NoticeError):
        gn.build_notice([_entry("x", license_="   ")])


def test_missing_url_or_doc_id_raises():
    with pytest.raises(gn.NoticeError):
        gn.build_notice([{"doc_id": "x", "license": "CC BY 4.0", "domain": "ai"}])
    with pytest.raises(gn.NoticeError):
        gn.build_notice([{"url": "https://example.test/x", "license": "CC BY 4.0", "domain": "ai"}])


def test_removal_flagged_sources_are_called_out():
    """Sources the manifest marks as removal candidates must be visible as such,
    not buried -- they are the ones a challenge would land on first."""
    entries = [_entry("ok", "CC BY 4.0"),
               _entry("risky", "(c) Someone. Shipped fair-use + attribution; first candidate "
                               "for one-line removal if challenged.")]
    text = gn.build_notice(entries)
    assert "risky" in text
    assert "removal" in text.lower()


def test_shipped_manifest_is_fully_covered():
    """Not hermetic by design: asserts the NOTICE covers the real 55-source corpus."""
    manifest = Path(__file__).resolve().parents[1] / "corpus" / "manifest.yaml"
    entries = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    text = gn.build_notice(entries)
    missing = [e["doc_id"] for e in entries if e["doc_id"] not in text]
    assert missing == [], f"sources absent from NOTICE: {missing}"
    assert f"{len(entries)} source" in text


def test_committed_notice_is_current():
    """The committed NOTICE must match what the manifest generates now. Catches a
    manifest edit that was never reflected in the published attribution file."""
    root = Path(__file__).resolve().parents[1]
    notice = root / "NOTICE"
    assert notice.exists(), "NOTICE is missing; run scripts/generate_notice.py"
    entries = yaml.safe_load((root / "corpus" / "manifest.yaml").read_text(encoding="utf-8"))
    expected = gn.build_notice(entries)
    actual = notice.read_text(encoding="utf-8")
    assert actual == expected, (
        "NOTICE is stale relative to corpus/manifest.yaml — "
        "regenerate with: venv\\Scripts\\python.exe scripts/generate_notice.py"
    )


def test_noncommercial_sources_are_flagged():
    """NC terms constrain how a downstream user may deploy the shipped index, so
    they must be visible rather than left for the reader to spot in prose."""
    text = gn.build_notice([_entry("nc-src", "CC BY-NC (Br J Sports Med / BMJ)")])
    assert "NONCOMMERCIAL" in text


def test_removal_marker_catches_all_flagged_shipped_sources():
    """Regression: the first marker was the literal phrase 'removal if challenged',
    which silently missed the NC source whose tail reads 'removal if the shipped
    index's NC status is challenged'. All three flagged sources must be caught."""
    manifest = Path(__file__).resolve().parents[1] / "corpus" / "manifest.yaml"
    entries = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    flagged = [e for e in entries if "first candidate for" in e["license"].lower()]
    assert len(flagged) == 3, f"expected 3 flagged sources, found {len(flagged)}"
    text = gn.build_notice(entries)
    for e in flagged:
        line = next(ln for ln in text.splitlines() if e["license"] in ln)
        assert "REMOVAL CANDIDATE" in line, f"{e['doc_id']} not flagged in NOTICE"
