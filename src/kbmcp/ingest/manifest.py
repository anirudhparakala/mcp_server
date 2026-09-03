"""corpus/manifest.yaml loader + validator.

`SourceEntry.doc_id` is the manifest SLUG (a stable human handle used for
cross-reference resolution), not the sha256 docs.doc_id (derived at build time
from url+version via models.ids.doc_id).

The loader is shared by the curated eval corpus AND by BYO-corpus manifests
(hand-written or auto-generated from a local folder). BYO entries may omit
`domain` and `tier_roles` (eval-only fields), so those are optional here, and
`domain` is free-form (NOT restricted to the eval corpus's four domains).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

# Known eval-corpus domains (informational only — NOT enforced, so BYO works).
KNOWN_DOMAINS = {"ai", "law_aireg", "law_contract", "fitness"}
VALID_FORMATS = {"html", "pdf", "other"}
_REQUIRED = ["doc_id", "url", "format", "license", "license_ok", "version"]


class ManifestError(ValueError):
    """Raised when a manifest entry is malformed or violates a corpus rule."""


@dataclass(frozen=True)
class SourceEntry:
    doc_id: str  # manifest slug (human handle), NOT the sha256 docs.doc_id
    url: str
    format: str
    license: str
    license_ok: bool
    version: str
    domain: str = "unspecified"  # free-form; eval corpus uses KNOWN_DOMAINS by convention
    tier_roles: list = field(default_factory=list)  # eval-only; empty for BYO
    raw_cache: Optional[str] = None
    collision_terms: list = field(default_factory=list)
    references: list = field(default_factory=list)
    conflict_with: list = field(default_factory=list)
    rationale: Optional[str] = None


def load_manifest(path) -> list[SourceEntry]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ManifestError("manifest root must be a list of source entries")

    entries: list[SourceEntry] = []
    seen: set[str] = set()
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            raise ManifestError(f"entry {i} must be a mapping")
        for key in _REQUIRED:
            if key not in row:
                raise ManifestError(f"entry {i} missing required field: {key}")
        slug = row["doc_id"]
        if row["format"] not in VALID_FORMATS:
            raise ManifestError(f"entry {i} ({slug}): invalid format {row['format']!r}")
        if row["license_ok"] is not True:
            raise ManifestError(f"entry {i} ({slug}): license_ok must be true")
        if slug in seen:
            raise ManifestError(f"duplicate doc_id slug: {slug}")
        seen.add(slug)
        entries.append(
            SourceEntry(
                doc_id=slug,
                url=row["url"],
                format=row["format"],
                license=row["license"],
                license_ok=bool(row["license_ok"]),
                version=row["version"],
                domain=row.get("domain", "unspecified"),
                tier_roles=list(row.get("tier_roles", [])),
                raw_cache=row.get("raw_cache"),
                collision_terms=list(row.get("collision_terms", [])),
                references=list(row.get("references", [])),
                conflict_with=list(row.get("conflict_with", [])),
                rationale=row.get("rationale"),
            )
        )
    return entries
