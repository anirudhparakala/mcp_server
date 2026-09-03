from pathlib import Path

from kbmcp.config import load_corpus_config

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "corpus_config.yaml"


def test_loads_real_config_sections():
    cfg = load_corpus_config(CONFIG_PATH)
    assert cfg.chunk["target_tokens"] == 512
    assert cfg.contextualize["model"] == "claude-haiku-4-5"
    assert cfg.contextualize["section_scoped"] is True
    assert cfg.parse["ocr"] is False


def test_missing_section_defaults_to_empty_dict(safe_tmp_path):
    p = safe_tmp_path / "c.yaml"
    p.write_text("chunk:\n  target_tokens: 256\n", encoding="utf-8")
    cfg = load_corpus_config(p)
    assert cfg.chunk["target_tokens"] == 256
    assert cfg.graph == {}


def test_contextualize_block_carries_m4_window_and_pricing_keys():
    cfg = load_corpus_config("config/corpus_config.yaml").contextualize
    assert cfg["model"] == "claude-haiku-4-5"
    assert cfg["temperature"] == 0
    assert cfg["max_tokens"] == 150
    assert cfg["cache_ttl"] == "1h"
    assert cfg["contexts_dir"] == "corpus/contexts"
    # window budget must clear Haiku 4.5's 4096-token minimum cacheable prefix
    assert cfg["window_target_tokens"] > cfg["min_cacheable_tokens"]
    assert cfg["window_max_tokens"] >= cfg["window_target_tokens"]
    assert cfg["chars_per_token"] >= 1
    assert cfg["price_in_per_mtok"] > 0 and cfg["price_out_per_mtok"] > 0
