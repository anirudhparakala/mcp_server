"""Deterministic, stable IDs for docs and chunks.

House rule (CLAUDE.md): chunk IDs are sha256(canonical_url + version + chunk_index).
Fields are joined with the ASCII unit-separator (\x1f) so no field value can
forge a boundary, then UTF-8 encoded and sha256 hex-digested. NEVER change this
serialization — citations and eval gold labels depend on it.
"""

import hashlib

_SEP = "\x1f"  # ASCII unit separator; extremely unlikely inside a URL or version


def doc_id(canonical_url: str, version: str) -> str:
    """Stable 64-char sha256 hex ID for a document version."""
    payload = f"{canonical_url}{_SEP}{version}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def chunk_id(canonical_url: str, version: str, chunk_index: int) -> str:
    """Stable 64-char sha256 hex ID for a chunk within a document version."""
    payload = f"{canonical_url}{_SEP}{version}{_SEP}{chunk_index}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
