import tempfile
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


def test_missing_section_defaults_to_empty_dict():
    # pytest's tmp_path fixture fails on this machine: a pre-existing
    # pytest-of-aniru bookkeeping dir under the Windows temp folder is
    # ACL-locked (PermissionError: WinError 5), even for its owning user.
    # tempfile.TemporaryDirectory() does not touch that directory and
    # works correctly here. See task-5-report.md for the verified repro.
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "c.yaml"
        p.write_text("chunk:\n  target_tokens: 256\n", encoding="utf-8")
        cfg = load_corpus_config(p)
        assert cfg.chunk["target_tokens"] == 256
        assert cfg.graph == {}
