"""Site-chrome filtering.

Deliberately NARROW. A structural heuristic was tried first and rejected: 31 of
2795 corpus chunks have >=70% short lines, and nearly all are legitimate content
(arXiv LaTeX math, author blocks, Wikipedia pseudocode, HF model-card citation
snippets). Dropping on that signal would destroy real corpus content, so each
signature below targets one publisher's chrome exactly.
"""

from kbmcp.ingest.chunk import ChunkRecord, chunk_document, is_boilerplate

LEGINFO_NAV = (
    "Code:\nSelect Code\nCONS\nBPC\nCIV\nCCP\nCOM\nCORP\nEDC\nELEC\nEVID\nFAM\nFIN\n"
    "FGC\nFAC\nGOV\nHNC\nHSC\nINS\nLAB\nMVC\nPEN\nPROB\nPCC\nPRC\nPUC\nRTC\nSHC\nUIC\n"
    "VEH\nWAT\nWIC\nSection:\n1 or 2 or 1001\nSearch\ninformation"
)

JUSTIA_FOOTER = (
    "[Justia Connect](https://connect.justia.com/)\n"
    "[Legal Portal](https://www.justia.com/)\n"
    "[Company](https://company.justia.com/)\n"
    "[Help](https://help.justia.com/)\n"
    "[Terms of Service](https://www.justia.com/terms-of-service/)\n"
    "[Privacy Policy](https://www.justia.com/privacy-policy/)"
)


def test_leginfo_code_selector_menu_is_boilerplate():
    assert is_boilerplate(LEGINFO_NAV) is True


def test_justia_link_footer_is_boilerplate():
    assert is_boilerplate(JUSTIA_FOOTER) is True


# --- false-positive guards: every one of these is REAL corpus content ---

def test_arxiv_latex_math_is_not_boilerplate():
    """93% short lines, but it is the Transformer paper's optimizer section."""
    text = ("We used the Adam optimizer [\n[20](#bib.bib20)\n] with \u03b2 1 = 0.9 "
            "\\beta_{1}=0.9 , \u03b2 2 = 0.98 \\beta_{2}=0.98\n" + "\n".join("=" for _ in range(40)))
    assert is_boilerplate(text) is False


def test_author_block_is_not_boilerplate():
    text = ("Ashish Vaswani\nGoogle Brain\navaswani@google.com\n&Noam Shazeer\n1 1\n"
            "footnotemark: 1\nGoogle Brain\nnoam@google.com\n&Niki Parmar\nGoogle Research\n")
    assert is_boilerplate(text) is False


def test_wikipedia_reference_entry_mentioning_privacy_policy_is_not_boilerplate():
    """The first keyword-based attempt wrongly ate this: it is a bibliography entry."""
    text = ('115. [\u2191](#cite_ref-121) Chen, Brian X. (23 May 2018). '
            '["Getting a Flood of G.D.P.R.-Related Privacy Policy Updates? Read Them"]'
            '(https://www.nytimes.com/2018/05/23/technology/personaltech/what-you-should-look-for-'
            'europe-data-law.html). The New York Times. Retrieved 4 June 2018.')
    assert is_boilerplate(text) is False


def test_model_card_citation_block_is_not_boilerplate():
    text = ("If you find this repository useful, please consider giving a star and citation\n"
            "```\n@misc{li2023making,\n  title={Making Large Language Models},\n"
            "  author={Chaofan Li},\n  year={2023}\n}\n```")
    assert is_boilerplate(text) is False


def test_a_chunk_of_ordinary_prose_is_not_boilerplate():
    assert is_boilerplate("Subject to the provisions of this division on breach in installment "
                          "contracts (Section 2612), the buyer may reject the whole.") is False


def test_empty_and_whitespace_are_not_boilerplate():
    assert is_boilerplate("") is False
    assert is_boilerplate("   \n \n ") is False


# --- integration: chunk_document drops them and keeps indices contiguous ---

class _FakeChunk:
    def __init__(self, text):
        self.text = text
        self.meta = type("M", (), {"headings": ["H"], "doc_items": []})()


def test_chunk_document_drops_boilerplate_and_reindexes(monkeypatch):
    """chunk_index must stay 0-based contiguous after a drop -- it feeds chunk_id."""
    from kbmcp.ingest import chunk as chunk_mod

    fakes = [_FakeChunk(LEGINFO_NAV), _FakeChunk("real section one"),
             _FakeChunk(JUSTIA_FOOTER), _FakeChunk("real section two")]
    monkeypatch.setattr(chunk_mod, "_chunker", lambda *a, **k: type(
        "C", (), {"chunk": staticmethod(lambda dl_doc: iter(fakes))})())

    recs = chunk_document(object(), {"tokenizer": "x", "max_tokens": 512,
                                     "drop_boilerplate": True})
    assert [r.text for r in recs] == ["real section one", "real section two"]
    assert [r.chunk_index for r in recs] == [0, 1]      # contiguous, no gap


def test_chunk_document_keeps_boilerplate_when_disabled(monkeypatch):
    from kbmcp.ingest import chunk as chunk_mod

    fakes = [_FakeChunk(LEGINFO_NAV), _FakeChunk("real section one")]
    monkeypatch.setattr(chunk_mod, "_chunker", lambda *a, **k: type(
        "C", (), {"chunk": staticmethod(lambda dl_doc: iter(fakes))})())

    recs = chunk_document(object(), {"tokenizer": "x", "max_tokens": 512,
                                     "drop_boilerplate": False})
    assert len(recs) == 2 and [r.chunk_index for r in recs] == [0, 1]


def test_chunk_record_is_unchanged_in_shape():
    r = ChunkRecord(chunk_index=0, text="t")
    assert r.chunk_type == "text" and r.heading_path == [] and r.table is None
