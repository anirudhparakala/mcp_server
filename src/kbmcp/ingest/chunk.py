"""Stage C — Docling HybridChunker: DoclingDocument -> deterministic ChunkRecords.

chunk_index is the 0-based contiguous yield order of the chunker; it feeds the
house-rule chunk_id = sha256(url + version + chunk_index). context and
citation_anchors are left empty here (populated in M4 contextualize / M5 graph).
"""

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc import DoclingDocument, DocItemLabel, RefItem, TableItem
from transformers import AutoTokenizer


# Site chrome that Docling faithfully extracts but that carries no corpus content.
# Deliberately NARROW, one signature per publisher. A structural heuristic
# (short-line ratio) was measured first and REJECTED: 31 of 2795 corpus chunks have
# >=70% short lines and nearly all are real content -- arXiv LaTeX math, author
# blocks, Wikipedia pseudocode, model-card citation snippets. Matching on shape
# would have destroyed them; matching on "Cookie"/"Privacy Policy" keywords wrongly
# caught a Wikipedia bibliography entry. Each rule below is anchored to text that
# only that publisher's navigation emits.
_LINK_LINE = re.compile(r"^\[[^\]]+\]\(https?://[^)]+\)$")


def is_boilerplate(text: str) -> bool:
    """True only for site navigation/footer chrome, never for document content."""
    stripped = (text or "").strip()
    if not stripped:
        return False

    # leginfo.legislature.ca.gov renders a California code-selector menu above
    # every section: "Code:\nSelect Code\nCONS\nBPC\nCIV\n..."
    if stripped.startswith("Code:") and "Select Code" in stripped[:200]:
        return True

    # law.justia.com appends a footer that is almost entirely markdown links.
    lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
    if len(lines) >= 5:
        links = sum(1 for ln in lines if _LINK_LINE.match(ln))
        if links / len(lines) >= 0.85:
            return True

    return False


@dataclass(frozen=True)
class ChunkRecord:
    chunk_index: int
    text: str
    heading_path: list = field(default_factory=list)
    chunk_type: str = "text"
    table: Optional[dict] = None
    citation_anchors: dict = field(default_factory=dict)


@lru_cache(maxsize=2)
def _chunker(model: str, max_tokens: int) -> HybridChunker:
    tok = HuggingFaceTokenizer(tokenizer=AutoTokenizer.from_pretrained(model), max_tokens=max_tokens)
    return HybridChunker(tokenizer=tok)


def _classify(dl_doc: DoclingDocument, ck) -> tuple[str, Optional[dict]]:
    """Return (chunk_type, table_grid_or_None) for a chunk from its doc_items.

    NOTE (docling 2.118.0 deviation): the DocItem instances found on
    ``ck.meta.doc_items`` are generic proxy objects (not the concrete
    ``TableItem`` subclass), so ``isinstance(it, TableItem)`` never matches
    there. Identify a table item by ``it.label == DocItemLabel.TABLE`` and
    resolve the concrete item via its ``self_ref`` against ``dl_doc`` before
    calling ``export_to_dataframe``.
    """
    items = list(getattr(ck.meta, "doc_items", []) or [])
    for it in items:
        if getattr(it, "label", None) == DocItemLabel.TABLE:
            resolved = RefItem(cref=it.self_ref).resolve(dl_doc)
            if isinstance(resolved, TableItem):
                return "table", resolved.export_to_dataframe(dl_doc).to_dict(orient="split")
    return "text", None


def chunk_document(dl_doc: DoclingDocument, chunk_cfg: dict) -> list[ChunkRecord]:
    chunker = _chunker(
        chunk_cfg.get("tokenizer", "Qwen/Qwen3-Embedding-0.6B"),
        int(chunk_cfg.get("max_tokens", 512)),
    )
    drop_chrome = bool(chunk_cfg.get("drop_boilerplate", True))
    records: list[ChunkRecord] = []
    for ck in chunker.chunk(dl_doc=dl_doc):
        if drop_chrome and is_boilerplate(ck.text):
            continue
        headings = list(getattr(ck.meta, "headings", []) or [])
        ctype, table = _classify(dl_doc, ck)
        # chunk_index is assigned AFTER filtering so it stays 0-based contiguous:
        # it feeds chunk_id = sha256(url + version + chunk_index) (house rule).
        records.append(
            ChunkRecord(chunk_index=len(records), text=ck.text, heading_path=headings,
                        chunk_type=ctype, table=table)
        )
    return records
