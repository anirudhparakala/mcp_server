"""--folder mode: turn a directory of documents into manifest-shaped rows.

Decision 6 makes local-file ingest first-class, so pointing at a folder must need
no flags. License is auto-filled because a --folder user is almost always indexing
documents they already own; attribution exists for this project's PUBLISHED corpus
of third-party material, not for someone's private notes. No NOTICE is generated
for a folder-built corpus.
"""

import hashlib
import re
from pathlib import Path

import yaml

LOCAL_LICENSE = "Local file (user-supplied); not redistributed by this project"

_SLUG_RE = re.compile(r"[^a-z0-9]+")


class FolderSourceError(ValueError):
    """Raised when a folder cannot produce a usable source list."""


def slug_for(rel_path) -> str:
    """A stable doc_id slug from a path relative to the folder root.

    Uses the whole relative path, not the basename, so notes.md in two different
    subdirectories do not collide.
    """
    stem = str(Path(rel_path).with_suffix("")).replace("\\", "/")
    slug = _SLUG_RE.sub("-", stem.lower()).strip("-")
    return slug or "source"


def _format_for(suffix: str) -> str:
    s = suffix.lower()
    if s == ".pdf":
        return "pdf"
    if s in (".html", ".htm"):
        return "html"
    return "other"


def discover(folder) -> list:
    """Manifest-shaped rows for every file under `folder`, sorted by doc_id.

    Skips dotfiles and anything inside a dot-directory. Raises rather than
    returning [] for an empty folder: silently building an empty CKB is the kind
    of quiet no-op this project keeps getting bitten by.
    """
    root = Path(folder).resolve()
    if not root.is_dir():
        raise FolderSourceError(f"not a directory: {root}")

    rows, seen = [], {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part.startswith(".") for part in rel.parts):
            continue
        slug = slug_for(rel)
        if slug in seen:
            raise FolderSourceError(
                f"two files produce the same doc_id {slug!r}: {seen[slug]} and {rel}. "
                "Rename one of them."
            )
        seen[slug] = rel
        rows.append({
            "doc_id": slug,
            "url": str(path),
            "format": _format_for(path.suffix),
            "license": LOCAL_LICENSE,
            "license_ok": True,
            "version": hashlib.sha256(path.read_bytes()).hexdigest(),
            "domain": "unspecified",
        })

    if not rows:
        raise FolderSourceError(
            f"no files found under {root} (dotfiles and dot-directories are skipped)"
        )
    return sorted(rows, key=lambda r: r["doc_id"])


def write_manifest(rows, out_path) -> Path:
    """Materialize the synthesized manifest beside the CKB.

    build_ckb and build_graph both take a manifest PATH and call load_manifest
    themselves, so folder mode needs a real file. It is written next to the CKB
    rather than into the user's document folder, and doubles as provenance:
    exactly what was discovered, with content hashes.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(rows, sort_keys=False, allow_unicode=True),
                   encoding="utf-8")
    return out
