"""Generate the NOTICE attribution file from corpus/manifest.yaml.

The corpus ships with chunk text from third-party sources (Decision 6), so every
source needs visible attribution and its license recorded. Generating that from
the manifest rather than hand-maintaining it means the two cannot drift: the
manifest already carries a per-source `license` field, and a test asserts the
committed NOTICE matches what this produces.

BYO-corpus users should re-run this against their own manifest before publishing
an index built from their own sources.

Usage:
    venv\\Scripts\\python.exe scripts/generate_notice.py
    venv\\Scripts\\python.exe scripts/generate_notice.py --check   # CI-style, no write
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import yaml

# A manifest license string carrying this phrasing marks a source the user already
# identified as the first thing to drop if its inclusion is challenged. Those must
# be visible in the NOTICE, not buried among the rest. The phrase is matched loosely
# on purpose: the three flagged sources word their tails differently ("...one-line
# removal if challenged" vs "...removal if the shipped index's NC status is
# challenged"), and a narrower match silently missed the NonCommercial one.
_REMOVAL_MARKER = "first candidate for"

# NonCommercial terms constrain how a downstream user may deploy the shipped index,
# so they are surfaced explicitly rather than left for the reader to notice.
_NONCOMMERCIAL_MARKER = "-nc"

_PREAMBLE = """\
NOTICE
======

This project ships a prebuilt knowledge base ("the CKB") containing text
extracted from {n} third-party sources, together with machine-generated
per-chunk context summaries derived from that text.

The **software** in this repository is licensed under the MIT License; see
LICENSE. This NOTICE covers the **corpus content**, which is NOT covered by
that license and remains under the terms of its respective sources, listed
below.

Corpus content is included for retrieval research and evaluation: the server
returns short cited evidence chunks in response to queries and performs no
generation. Sources that are public domain, Creative Commons, or licensed for
reuse are included under those terms, with attribution below. A small number of
copyrighted items are included on a fair-use basis for this research purpose;
they are marked "REMOVAL CANDIDATE" and the manifest is structured so any single
source can be removed in one line.

If you hold rights in any material listed here and object to its inclusion,
please open an issue in this repository and it will be removed.

This file is generated from corpus/manifest.yaml by scripts/generate_notice.py.
Do not edit it by hand -- edit the manifest and regenerate.

This is a record of attribution, not legal advice.

"""


class NoticeError(ValueError):
    """Raised when a manifest entry cannot be attributed."""


def _require(entry, field, index):
    value = entry.get(field)
    if value is None or not str(value).strip():
        ident = entry.get("doc_id") or f"entry #{index}"
        raise NoticeError(
            f"{ident}: missing required field {field!r}. Every source must carry "
            "doc_id, url and license -- a source that cannot be attributed must not "
            "be shipped."
        )
    return str(value).strip()


def build_notice(entries) -> str:
    """Render the NOTICE text for a list of manifest entries.

    Groups sources under their license string, sorted, so the output is stable
    regardless of manifest order. Raises NoticeError rather than emitting a blank
    attribution for an incomplete entry.
    """
    by_license = defaultdict(list)
    for i, entry in enumerate(entries):
        doc_id = _require(entry, "doc_id", i)
        url = _require(entry, "url", i)
        license_ = _require(entry, "license", i)
        by_license[license_].append((doc_id, url, entry.get("domain", "")))

    out = [_PREAMBLE.format(n=len(entries))]
    out.append(f"Sources ({len(entries)} source(s), grouped by license)")
    out.append("=" * 62)
    out.append("")

    for license_ in sorted(by_license):
        low = license_.lower()
        tags = []
        if _REMOVAL_MARKER in low:
            tags.append("REMOVAL CANDIDATE")
        if _NONCOMMERCIAL_MARKER in low or "noncommercial" in low:
            tags.append("NONCOMMERCIAL")
        prefix = ("[" + ", ".join(tags) + "] ") if tags else ""
        out.append(prefix + license_)
        out.append("-" * 62)
        for doc_id, url, domain in sorted(by_license[license_]):
            out.append(f"  {doc_id}" + (f"  [{domain}]" if domain else ""))
            out.append(f"    {url}")
        out.append("")

    return "\n".join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python scripts/generate_notice.py")
    p.add_argument("--manifest", default="corpus/manifest.yaml")
    p.add_argument("--out", default="NOTICE")
    p.add_argument("--check", action="store_true",
                   help="exit 1 if the file on disk differs; write nothing")
    a = p.parse_args(argv)

    entries = yaml.safe_load(Path(a.manifest).read_text(encoding="utf-8"))
    text = build_notice(entries)
    out = Path(a.out)

    if a.check:
        if not out.exists():
            print(f"{a.out} is missing", file=sys.stderr)
            return 1
        if out.read_text(encoding="utf-8") != text:
            print(f"{a.out} is stale; regenerate with scripts/generate_notice.py",
                  file=sys.stderr)
            return 1
        print(f"{a.out} is current ({len(entries)} sources)", file=sys.stderr)
        return 0

    out.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {a.out}: {len(entries)} sources", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
