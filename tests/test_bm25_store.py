"""Tests for the FTS5-backed lexical index (index/bm25_store.py).

Hermetic: every test builds a tiny CKB under safe_tmp_path. Never depends on
ckb/ckb.sqlite existing.
"""

import pytest

from kbmcp.index import bm25_store as bs


# The 6 realistic queries measured against the live corpus. Three of them
# (indices 0, 2, 4) raise sqlite3.OperationalError if passed to MATCH raw.
HAZARD_QUERIES = [
    "What does Article 22(1) say?",
    'liquidated damages "penalty" clause',
    "GDPR OR NOT AND",
    "recovery of damages*",
    "protein intake -- how much?",
    "§ 2202 statute of frauds",
]


def test_to_match_query_quotes_every_token():
    assert bs.to_match_query("shall not") == '"shall" OR "not"'


def test_to_match_query_strips_fts5_operators():
    # OR/NOT/AND become quoted literals, not operators
    assert bs.to_match_query("GDPR OR NOT AND") == '"GDPR" OR "OR" OR "NOT" OR "AND"'


def test_to_match_query_keeps_digits_and_splits_punctuation():
    assert bs.to_match_query("Article 22(1)") == '"Article" OR "22" OR "1"'


def test_to_match_query_empty_for_punctuation_only():
    assert bs.to_match_query("--- ??? ***") == ""
    assert bs.to_match_query("") == ""


def test_to_match_query_never_emits_a_double_quote_inside_a_token():
    # Injection safety: \w+ cannot match a quote, so quoting is safe by construction
    for q in HAZARD_QUERIES:
        for token in bs.to_match_query(q).split(" OR "):
            if token:
                assert token.startswith('"') and token.endswith('"')
                assert '"' not in token[1:-1]
