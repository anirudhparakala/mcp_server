from kbmcp.ingest import citations as cit

# Verbatim shapes taken from the real corpus (GDPR, EDPB, CA Commercial Code, arXiv).
GDPR_HEADER_CHUNK = """rights and freedoms of others.
Section 4
Right to object and automated individual decision-making
Article 22
Automated individual decision-making, including profiling
1.
The data subject shall have the right not to be subject to a decision based solely
on automated processing.
Article 23
Restrictions
"""

EDPB_CITING_CHUNK = (
    "The prohibition in Article 22(1) applies. As noted in Article 29 of Directive "
    "95/46/EC and Article 15 of the same instrument, controllers must inform the data "
    "subject. See also Recital 71."
)

UCC_CITING_CHUNK = (
    "may be explained or supplemented:\n(a) By course of dealing, course of performance, "
    "or usage of trade (Section 1303); and\n(b) By evidence of consistent additional terms"
)


def test_anchor_is_found_only_on_a_standalone_header_line():
    anchors = cit.extract_anchors(GDPR_HEADER_CHUNK)
    assert cit.Anchor("article", "22") in anchors
    assert cit.Anchor("article", "23") in anchors


def test_inline_citation_is_not_mistaken_for_an_anchor():
    """The single most damaging failure mode: every doc would 'define' what it cites."""
    assert cit.extract_anchors(EDPB_CITING_CHUNK) == []
    assert cit.extract_anchors(UCC_CITING_CHUNK) == []


def test_references_are_extracted_from_inline_prose():
    refs = cit.extract_references(EDPB_CITING_CHUNK)
    kinds = {(r.kind, r.value) for r in refs}
    assert ("article", "22") in kinds          # Article 22(1) -> article 22
    assert ("article", "29") in kinds
    assert ("recital", "71") in kinds


def test_section_reference_in_parentheses_is_extracted():
    refs = cit.extract_references(UCC_CITING_CHUNK)
    assert ("section", "1303") in {(r.kind, r.value) for r in refs}


def test_a_chunk_that_defines_an_article_does_not_also_reference_it():
    refs = cit.extract_references(GDPR_HEADER_CHUNK)
    assert ("article", "22") not in {(r.kind, r.value) for r in refs}


def test_academic_identifiers_are_extracted():
    text = "as shown in arXiv:1706.03762 and doi:10.1145/1571941.1572114 we find"
    refs = cit.extract_references(text)
    kinds = {(r.kind, r.value) for r in refs}
    assert ("arxiv", "1706.03762") in kinds
    assert ("doi", "10.1145/1571941.1572114") in kinds


def test_case_citation_is_extracted():
    refs = cit.extract_references("as held in Hadley v. Baxendale the damages were")
    assert ("case", "hadley v. baxendale") in {(r.kind, r.value) for r in refs}


def test_numbered_bibliography_citations_are_ignored():
    """[20] is bibliography-relative; 1686 occurrences, deliberately out of scope."""
    refs = cit.extract_references("We used the Adam optimizer [20] with beta_1 = 0.9")
    assert refs == []


def test_references_are_deduplicated_in_first_appearance_order():
    refs = cit.extract_references("Article 22 then Article 5 then Article 22 again")
    assert [(r.kind, r.value) for r in refs] == [("article", "22"), ("article", "5")]


def test_extractor_selection_is_honoured():
    text = "Article 22 and arXiv:1706.03762"
    legal_only = cit.extract_references(text, extractors=("legal",))
    assert {r.kind for r in legal_only} == {"article"}
    academic_only = cit.extract_references(text, extractors=("academic",))
    assert {r.kind for r in academic_only} == {"arxiv"}


def test_anchors_to_dict_is_sorted_and_json_ready():
    anchors = [cit.Anchor("article", "23"), cit.Anchor("article", "22"),
               cit.Anchor("section", "1303")]
    assert cit.anchors_to_dict(anchors) == {"article": ["22", "23"], "section": ["1303"]}


def test_empty_text_yields_nothing():
    assert cit.extract_anchors("") == [] and cit.extract_references("") == []


# --- heading_path section anchors (leginfo docs: no text-line heading, but the
# chunker preserves the section number as the last heading_path element) ---

def test_heading_path_section_anchor_is_extracted():
    anchors = cit.extract_anchors("some section body text", heading_path=[
        "Code Section Group", "Code Text", "Commercial Code - COM",
        "DIVISION 1. GENERAL PROVISIONS [1101 - 1310]",
        "CHAPTER 2. General Definitions ... [1201 - 1206]", "1201.",
    ])
    assert cit.Anchor("section", "1201") in anchors


def test_heading_path_section_anchor_is_extracted_for_another_section():
    anchors = cit.extract_anchors("some other section body", heading_path=[
        "CHAPTER 2. Form, Formation and Readjustment of Contract [2201 - 2210]",
        "2202.",
    ])
    assert cit.Anchor("section", "2202") in anchors


def test_heading_path_breadcrumb_elements_are_not_mistaken_for_section_anchors():
    """The whole element must match the section shape -- a substring match here
    would anchor the wrong section. This is the false-positive guard that matters
    most: breadcrumb elements contain digits too."""
    assert cit.extract_anchors("body", heading_path=[
        "DIVISION 1. GENERAL PROVISIONS [1101 - 1310]",
    ]) == []
    assert cit.extract_anchors("body", heading_path=[
        "CHAPTER 2. Form, Formation and Readjustment of Contract [2201 - 2210]",
    ]) == []


def test_heading_path_none_or_empty_behaves_exactly_as_before():
    assert cit.extract_anchors(GDPR_HEADER_CHUNK, heading_path=None) == cit.extract_anchors(GDPR_HEADER_CHUNK)
    assert cit.extract_anchors(GDPR_HEADER_CHUNK, heading_path=[]) == cit.extract_anchors(GDPR_HEADER_CHUNK)
    assert cit.extract_anchors("", heading_path=None) == []
    assert cit.extract_anchors("", heading_path=[]) == []


def test_text_header_and_heading_path_section_anchors_combine():
    anchors = cit.extract_anchors(GDPR_HEADER_CHUNK, heading_path=["Some Doc", "1201."])
    assert cit.Anchor("article", "22") in anchors
    assert cit.Anchor("article", "23") in anchors
    assert cit.Anchor("section", "1201") in anchors
