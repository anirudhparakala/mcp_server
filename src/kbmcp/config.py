"""Loader for config/corpus_config.yaml (dev-only build configuration)."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class CorpusConfig:
    fetch: dict
    parse: dict
    chunk: dict
    contextualize: dict
    graph: dict
    bm25: dict
    verify: dict
    raw: dict


def load_corpus_config(path) -> CorpusConfig:
    """Parse the corpus build config; missing sections default to empty dicts."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return CorpusConfig(
        fetch=data.get("fetch", {}),
        parse=data.get("parse", {}),
        chunk=data.get("chunk", {}),
        contextualize=data.get("contextualize", {}),
        graph=data.get("graph", {}),
        bm25=data.get("bm25", {}),
        verify=data.get("verify", {}),
        raw=data,
    )
