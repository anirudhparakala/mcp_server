"""Stage B — Docling parse: pinned raw bytes -> DoclingDocument (committed cache).

Dev-only ([corpus] extra). Runs Docling once per source and commits the parsed
DoclingDocument to corpus/parsed/<doc_id>.json; the committed parse is the
reproducibility anchor for deterministic chunk_ids (Docling model inference is
not guaranteed bit-identical across runs/GPUs). Cache-first: an existing parsed
JSON is loaded and Docling is NOT re-run.
"""

from functools import lru_cache
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument


def parsed_path_for(parsed_dir, doc_id: str) -> Path:
    return Path(parsed_dir) / f"{doc_id}.json"


@lru_cache(maxsize=2)
def _converter(table_mode: str, ocr: bool) -> DocumentConverter:
    opts = PdfPipelineOptions(do_ocr=ocr, do_table_structure=True)
    opts.table_structure_options.mode = (
        TableFormerMode.ACCURATE if table_mode == "accurate" else TableFormerMode.FAST
    )
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )


def parse_source(doc_id: str, raw_path, parsed_dir, parse_cfg: dict, *, force: bool = False) -> DoclingDocument:
    """Parse one pinned source into a DoclingDocument (cache-first, committed)."""
    parsed_dir = Path(parsed_dir)
    parsed_dir.mkdir(parents=True, exist_ok=True)
    cache = parsed_path_for(parsed_dir, doc_id)
    if cache.exists() and not force:
        return DoclingDocument.load_from_json(cache)
    conv = _converter(parse_cfg.get("table_mode", "accurate"), bool(parse_cfg.get("ocr", False)))
    dl_doc = conv.convert(Path(raw_path)).document
    dl_doc.save_as_json(cache)
    return dl_doc
