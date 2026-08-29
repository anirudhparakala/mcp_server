"""Stage E (part 1) — deterministic citation extraction.

Pure and offline: no I/O, no DB, no LLM. Two different things are pulled out of a
chunk's text and must never be confused:

  ANCHOR    -- this chunk DEFINES a section ("Article 22" on its own line, as a
               heading). Anchors are what references resolve TO.
  REFERENCE -- this chunk CITES a section ("...the prohibition in Article 22(1)").
               References are what edges are built FROM.

Both are the same string; only position distinguishes them. Treating an inline
citation as an anchor would make every document appear to define what it cites,
and the resolver would wire every reference back to its own chunk.

Pattern choices are measurement-driven (see the plan's measurements table):
"[20]"-style bibliography citations and "incorporated by reference" are
deliberately not extracted.

Anchors come from two independent sources that do not overlap in this corpus:
line-start text (EUR-Lex GDPR/AI-Act documents, which have `heading_path == []`)
and, when supplied, the last element of a chunk's `heading_path` (leginfo CA
Commercial Code documents, where the chunker preserves the section number --
e.g. "1201." -- as the final breadcrumb element rather than as its own text
line). Only the *last* element is examined and it must match the section shape
in full: breadcrumb elements like "CHAPTER 2. ... [1201 - 1206]" contain digits
too, and a substring match there would anchor the wrong section.
"""

import re
from dataclasses import dataclass

# --- anchors: standalone header lines only (re.M, anchored both ends) ---
_ANCHOR_PATTERNS = (
    ("article", re.compile(r"^[ \t]*Article[ \t]+(\d+[a-z]?)[ \t]*$", re.M)),
    ("recital", re.compile(r"^[ \t]*Recital[ \t]+(\d+)[ \t]*$", re.M)),
    # leginfo renders "1303." as its own line above the section body
    ("section", re.compile(r"^[ \t]*(\d{4})\.[ \t]*$", re.M)),
)

# --- anchors: heading_path last element (leginfo docs; see module docstring).
# Anchored both ends (fullmatch semantics) so a breadcrumb element containing
# digits, e.g. "CHAPTER 2. ... [1201 - 1206]", never matches as a substring.
_HEADING_PATH_SECTION_RE = re.compile(r"^(\d+[a-z]?)\.?$")

# --- references: inline prose ---
_LEGAL_REFERENCE_PATTERNS = (
    # "Article 22", "Article 22(1)", "Article 22a" -- the subsection is dropped:
    # the corpus anchors at article granularity, so 22(1) resolves to Article 22.
    ("article", re.compile(r"\bArticle[ \t]+(\d+[a-z]?)(?:\(\d+\))?")),
    ("recital", re.compile(r"\bRecital[ \t]+(\d+)")),
    # "(Section 1303)", "Section 1303", "§ 1303", "§ 2-201"
    ("section", re.compile(r"(?:§[ \t]*|\bSections?[ \t]+)(\d+(?:[-.]\d+)*)")),
    ("case", re.compile(r"\b([A-Z][A-Za-z.&'\-]+(?:[ \t]+[A-Z][A-Za-z.&'\-]+)?"
                        r"[ \t]+v\.[ \t]+[A-Z][A-Za-z.&'\-]+(?:[ \t]+[A-Z][A-Za-z.&'\-]+)?)")),
)

_ACADEMIC_REFERENCE_PATTERNS = (
    ("arxiv", re.compile(r"\barXiv:(\d{4}\.\d{4,5})", re.I)),
    ("doi", re.compile(r"\b(?:doi:[ \t]*)?(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", re.I)),
)

_EXTRACTOR_SETS = {
    "legal": _LEGAL_REFERENCE_PATTERNS,
    "academic": _ACADEMIC_REFERENCE_PATTERNS,
}


@dataclass(frozen=True)
class Anchor:
    kind: str
    value: str


@dataclass(frozen=True)
class Reference:
    kind: str
    value: str
    raw: str


def _anchor_line_spans(text: str) -> list:
    """Character spans of every standalone header line, so references inside them
    are not double-counted as citations."""
    spans = []
    for _, pattern in _ANCHOR_PATTERNS:
        for m in pattern.finditer(text):
            spans.append((m.start(), m.end()))
    return spans


def extract_anchors(text: str, heading_path=None) -> list:
    """Section identifiers this chunk DEFINES.

    Combines two sources, de-duplicated, source-then-pattern order preserved:
    standalone header lines in `text`, and -- when `heading_path` is given and
    non-empty -- its last element, if that element (in full) is a section
    identifier such as "1201." (leginfo CA Commercial Code chunks carry the
    section number there instead of as a text line).
    """
    found = []
    seen = set()
    if text:
        for kind, pattern in _ANCHOR_PATTERNS:
            for m in pattern.finditer(text):
                key = (kind, m.group(1))
                if key not in seen:
                    seen.add(key)
                    found.append(Anchor(kind, m.group(1)))
    if heading_path:
        m = _HEADING_PATH_SECTION_RE.match(heading_path[-1].strip())
        if m:
            key = ("section", m.group(1))
            if key not in seen:
                seen.add(key)
                found.append(Anchor("section", m.group(1)))
    return found


def extract_references(text: str, *, extractors=("legal", "academic")) -> list:
    """Citations this chunk MAKES, de-duplicated, in first-appearance order."""
    if not text:
        return []
    skip = _anchor_line_spans(text)

    def _inside_anchor(pos: int) -> bool:
        return any(start <= pos < end for start, end in skip)

    hits = []
    for name in extractors:
        for kind, pattern in _EXTRACTOR_SETS.get(name, ()):
            for m in pattern.finditer(text):
                if _inside_anchor(m.start()):
                    continue
                value = m.group(1)
                hits.append((m.start(), Reference(kind, value.lower() if kind == "case" else value,
                                                  m.group(0).strip())))
    hits.sort(key=lambda pair: pair[0])
    out, seen = [], set()
    for _, ref in hits:
        key = (ref.kind, ref.value)
        if key not in seen:
            seen.add(key)
            out.append(ref)
    return out


def anchors_to_dict(anchors) -> dict:
    """{kind: sorted[value]} for the chunks.citation_anchors column."""
    grouped: dict = {}
    for a in anchors:
        grouped.setdefault(a.kind, set()).add(a.value)
    return {k: sorted(v) for k, v in sorted(grouped.items())}
