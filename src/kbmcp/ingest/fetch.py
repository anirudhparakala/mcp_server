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
from urllib.parse import urlparse
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
    raw_filename = meta.get("raw_filename")
    if not raw_filename:
        return False
    return (Path(raw_dir) / raw_filename).exists()


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


def resolve_recipe(entry: SourceEntry) -> str:
    """Pick the fetch recipe. Only arXiv and local need bespoke logic."""
    if entry.url.startswith("file:"):
        return "local"
    host = urlparse(entry.url).netloc.lower()
    if "arxiv.org" in host:
        return "arxiv"
    return "generic"


def _arxiv_id(url: str) -> str:
    tail = urlparse(url).path.rsplit("/", 1)[-1]
    if tail.endswith(".pdf"):
        tail = tail[:-4]
    return re.sub(r"v\d+$", "", tail)


def _arxiv_urls(url: str, version: str) -> tuple[str, str]:
    """(html_url, pdf_url) for the pinned version, e.g. version='v7'."""
    aid = _arxiv_id(url)
    return f"https://arxiv.org/html/{aid}{version}", f"https://arxiv.org/pdf/{aid}{version}"


def _read_local(url: str) -> tuple[bytes, str]:
    """Read bytes for a file:// URL (BYO local-file path); returns (data, abspath)."""
    path = Path(url2pathname(urlparse(url).path))
    return path.read_bytes(), str(path)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _needs_render(entry: SourceEntry, cfg: dict) -> bool:
    render = cfg.get("render") or {}
    if not render.get("enabled", False):
        return False
    host = urlparse(entry.url).netloc.lower()
    return any(h.lower() in host for h in render.get("hosts", []))


def fetch_source(entry, raw_dir, cfg, *, force=False, transport=None, renderer=None) -> FetchResult:
    """Fetch and pin one source (cache-first). Raises FetchError on fetch failure."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    recipe = resolve_recipe(entry)

    if not force and is_cached(raw_dir, entry.doc_id):
        meta = read_meta(raw_dir, entry.doc_id)
        return FetchResult(
            doc_id=entry.doc_id, status="cached", recipe=recipe,
            raw_path=raw_dir / meta["raw_filename"], meta_path=meta_path_for(raw_dir, entry.doc_id),
            resolved_version=meta.get("resolved_version"), content_hash=meta.get("content_hash"),
            content_type=meta.get("content_type"), final_url=meta.get("final_url"),
            format=meta.get("format"), fetched_at=meta.get("fetched_at"),
        )

    if recipe == "local":
        try:
            data, final_url = _read_local(entry.url)
        except (OSError, ValueError) as exc:
            raise FetchError(f"{entry.doc_id}: cannot read local file {entry.url}: {exc}") from exc
        fmt, content_type = entry.format, None
        resolved_version = f"sha256:{content_hash(data)}"
    elif recipe == "arxiv":
        html_url, pdf_url = _arxiv_urls(entry.url, entry.version)
        try:
            data, final_url, content_type, _ = _http_get(html_url, cfg, transport=transport)
            fmt = "html"
        except FetchError:
            data, final_url, content_type, _ = _http_get(pdf_url, cfg, transport=transport)
            fmt = "pdf"
        resolved_version = entry.version
    else:  # generic
        if _needs_render(entry, cfg):
            if renderer is None:
                raise FetchError(f"{entry.doc_id}: rendering required but no renderer configured")
            data, final_url = renderer(entry.url, cfg)
            content_type = "text/html"
        else:
            data, final_url, content_type, _ = _http_get(entry.url, cfg, transport=transport)
        fmt = entry.format
        resolved_version = entry.version

    raw_filename = f"{entry.doc_id}{_ext_for(fmt)}"
    (raw_dir / raw_filename).write_bytes(data)
    meta = {
        "doc_id": entry.doc_id, "recipe": recipe, "resolved_version": resolved_version,
        "content_hash": content_hash(data), "content_type": content_type,
        "final_url": final_url, "format": fmt, "fetched_at": _now_iso(),
        "source_url": entry.url, "license": entry.license, "license_ok": entry.license_ok,
        "raw_filename": raw_filename,
    }
    meta_p = write_meta(raw_dir, meta)
    return FetchResult(
        doc_id=entry.doc_id, status="ok", recipe=recipe, raw_path=raw_dir / raw_filename,
        meta_path=meta_p, resolved_version=resolved_version, content_hash=meta["content_hash"],
        content_type=content_type, final_url=final_url, format=fmt, fetched_at=meta["fetched_at"],
    )


def fetch_all(entries, raw_dir, cfg, *, force=False, transport=None, renderer=None) -> list[FetchResult]:
    """Fetch every source, capturing per-source failures as error results."""
    results = []
    for entry in entries:
        try:
            results.append(
                fetch_source(entry, raw_dir, cfg, force=force, transport=transport, renderer=renderer)
            )
        except Exception as exc:  # noqa: BLE001 — one bad source must not abort the run
            try:
                recipe = resolve_recipe(entry)
            except Exception:  # noqa: BLE001 — recipe is best-effort context on the error path
                recipe = None
            results.append(
                FetchResult(doc_id=entry.doc_id, status="error", recipe=recipe, error=str(exc))
            )
    return results


def plan_fetches(entries, only=None):
    """Dry-run plan: (doc_id, recipe, target_url) per selected source (no network)."""
    plan = []
    for entry in entries:
        if only and entry.doc_id not in only:
            continue
        recipe = resolve_recipe(entry)
        target = _arxiv_urls(entry.url, entry.version)[0] if recipe == "arxiv" else entry.url
        plan.append((entry.doc_id, recipe, target))
    return plan


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m kbmcp.ingest.fetch",
        description="Fetch and pin manifest sources into corpus/raw/ (dev-only).",
    )
    p.add_argument("--manifest", default="corpus/manifest.yaml")
    p.add_argument("--raw-dir", default=None, help="override fetch.raw_dir from config")
    p.add_argument("--config", default="config/corpus_config.yaml")
    p.add_argument("--only", action="append", default=None, help="fetch only this doc_id (repeatable)")
    p.add_argument("--force", action="store_true", help="re-fetch even if cached")
    p.add_argument("--dry-run", action="store_true", help="print the plan; do not fetch")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = load_corpus_config(args.config).fetch
    raw_dir = args.raw_dir or cfg.get("raw_dir", "corpus/raw")
    entries = load_manifest(args.manifest)
    only = set(args.only) if args.only else None

    if args.dry_run:
        for doc_id, recipe, target in plan_fetches(entries, only):
            print(f"{doc_id}\t{recipe}\t{target}")
        return 0

    selected = [e for e in entries if not only or e.doc_id in only]
    results = fetch_all(selected, raw_dir, cfg, force=args.force)
    ok = 0
    for r in results:
        if r.status == "error":
            print(f"[error]  {r.doc_id}  ERROR: {r.error}", file=sys.stderr)
        else:
            ok += 1
            print(f"[{r.status:>6}] {r.doc_id}  ({r.recipe}, {r.format}, {r.resolved_version})", file=sys.stderr)
    print(f"{ok}/{len(results)} sources pinned", file=sys.stderr)
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
