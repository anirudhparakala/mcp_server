"""Cache-first, reproducible source fetcher (dev-only, [corpus] extra).

Stage 1 of the two-stage ingest pipeline (fetch -> build). Fetches each manifest
source and pins the raw bytes plus a resolved version into corpus/raw/, so the
later build stage runs fully offline and deterministically.

Pinning contract (see ingest design spec §9):
  - Versions are AUTHOR-PINNED: the manifest carries each source's exact
    version-specific URL + version string; the fetcher records that version
    VERBATIM (local files: version = "sha256:<hash>"). Never resolve "latest".
  - The pin RECORD (corpus/raw/<doc_id>.meta.json) is committed to git; the raw
    bytes are gitignored (local cache, re-fetchable + hash-verified).
  - Bespoke recipe logic is limited to arXiv (abs -> html/pdf) and file:// reads;
    everything else is a generic HTTP GET.
"""

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, unquote
from urllib.request import url2pathname

import httpx

from ..config import load_corpus_config
from .manifest import SourceEntry, load_manifest


@dataclass(frozen=True)
class FetchResult:
    """Outcome of pinning one source."""

    doc_id: str
    status: str  # "ok" | "cached" | "error"
    recipe: Optional[str] = None
    raw_path: Optional[Path] = None
    meta_path: Optional[Path] = None
    resolved_version: Optional[str] = None
    content_hash: Optional[str] = None
    content_type: Optional[str] = None
    final_url: Optional[str] = None
    format: Optional[str] = None
    fetched_at: Optional[str] = None
    error: Optional[str] = None


def content_hash(data: bytes) -> str:
    """sha256 hex of raw bytes (integrity check + local-file version handle)."""
    return hashlib.sha256(data).hexdigest()


def _ext_for(fmt: str) -> str:
    return {"pdf": ".pdf", "html": ".html"}.get(fmt, ".bin")


def meta_path_for(raw_dir, doc_id: str) -> Path:
    return Path(raw_dir) / f"{doc_id}.meta.json"


def write_meta(raw_dir, meta: dict) -> Path:
    """Write the committed pin-record sidecar (deterministic key order)."""
    p = meta_path_for(raw_dir, meta["doc_id"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return p


def read_meta(raw_dir, doc_id: str) -> Optional[dict]:
    p = meta_path_for(raw_dir, doc_id)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def is_cached(raw_dir, doc_id: str) -> bool:
    """True only when the meta record AND the raw file it names both exist."""
    meta = read_meta(raw_dir, doc_id)
    if meta is None:
        return False
    return (Path(raw_dir) / meta.get("raw_filename", "")).exists()


class FetchError(Exception):
    """Raised when a source cannot be fetched or pinned."""


def _http_get(url: str, cfg: dict, *, transport=None):
    """GET `url` with retries/timeout/UA; return (data, final_url, content_type, status).

    Retries on transport errors and 5xx up to cfg['retries'] times; raises
    FetchError immediately on 4xx (fail fast) and after exhausting retries.
    A fresh client per attempt keeps injected MockTransport tests simple.
    """
    timeout = cfg.get("timeout_s", 30)
    retries = cfg.get("retries", 2)
    headers = {"User-Agent": cfg.get("user_agent", "kbmcp-corpus-builder/0.1")}
    last_err = None
    for _ in range(retries + 1):
        try:
            with httpx.Client(
                timeout=timeout, follow_redirects=True, transport=transport, headers=headers
            ) as client:
                resp = client.get(url)
        except httpx.TransportError as exc:
            last_err = f"transport error: {exc}"
            continue
        if resp.status_code >= 500:
            last_err = f"server error {resp.status_code}"
            continue
        if resp.status_code >= 400:
            raise FetchError(f"{url} -> HTTP {resp.status_code}")
        return resp.content, str(resp.url), resp.headers.get("content-type"), resp.status_code
    raise FetchError(f"{url} failed after {retries + 1} attempt(s): {last_err}")
