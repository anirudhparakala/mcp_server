from kbmcp.ingest.chunk import ChunkRecord
from kbmcp.ingest import context_windows as cw

CFG = {"window_target_tokens": 60, "window_max_tokens": 100, "chars_per_token": 4}


def _rec(i, text, heading=None):
    return ChunkRecord(chunk_index=i, text=text, heading_path=list(heading or []))


def test_est_tokens_is_deterministic_and_at_least_one():
    assert cw.est_tokens("", 4) == 1
    assert cw.est_tokens("a" * 40, 4) == 10
    assert cw.est_tokens("a" * 41, 4) == 11  # rounds up
    assert cw.est_tokens("x" * 100, 4) == cw.est_tokens("x" * 100, 4)


def test_windows_partition_every_chunk_exactly_once():
    recs = [_rec(i, "word " * 40, ["Doc", f"S{i // 3}"]) for i in range(12)]
    wins = cw.build_windows(recs, CFG)
    seen = [i for w in wins for i in w.chunk_indices]
    assert seen == list(range(12))               # contiguous, ascending, no gaps
    assert len(seen) == len(set(seen))           # no chunk in two windows
    assert [w.window_index for w in wins] == list(range(len(wins)))


def test_window_text_contains_its_chunks_and_headings():
    recs = [_rec(0, "alpha body", ["Doc", "Intro"]), _rec(1, "beta body", ["Doc", "Intro"])]
    wins = cw.build_windows(recs, CFG)
    assert len(wins) == 1
    assert "alpha body" in wins[0].text and "beta body" in wins[0].text
    assert "Intro" in wins[0].text
    assert wins[0].text.count("Intro") == 1      # heading emitted once per run, not per chunk


def test_sections_are_kept_whole_when_they_fit():
    # two 3-chunk sections, each ~30 est tokens; target 60 fits both, so one window
    small = "c" * 40
    recs = [_rec(i, small, ["Doc", "A" if i < 3 else "B"]) for i in range(6)]
    wins = cw.build_windows(recs, {"window_target_tokens": 100, "window_max_tokens": 200, "chars_per_token": 4})
    assert len(wins) == 1
    # a tighter target must split BETWEEN the sections, not inside one
    wins = cw.build_windows(recs, {"window_target_tokens": 40, "window_max_tokens": 200, "chars_per_token": 4})
    assert [w.chunk_indices for w in wins] == [(0, 1, 2), (3, 4, 5)]


def test_oversized_single_section_is_split_at_chunk_boundaries():
    big = "z" * 400  # ~100 est tokens each, one section, max 100 -> one chunk per window
    recs = [_rec(i, big, ["Doc", "Same"]) for i in range(4)]
    wins = cw.build_windows(recs, {"window_target_tokens": 60, "window_max_tokens": 100, "chars_per_token": 4})
    assert [w.chunk_indices for w in wins] == [(0,), (1,), (2,), (3,)]
    assert all(w.est_tokens <= 100 or len(w.chunk_indices) == 1 for w in wins)


def test_chunks_without_headings_are_windowed_too():
    recs = [_rec(i, "no heading text", []) for i in range(3)]
    wins = cw.build_windows(recs, CFG)
    assert [i for w in wins for i in w.chunk_indices] == [0, 1, 2]


def test_build_windows_is_deterministic():
    recs = [_rec(i, f"body {i} " * 20, ["Doc", f"S{i // 2}"]) for i in range(9)]
    a = cw.build_windows(recs, CFG)
    b = cw.build_windows(recs, CFG)
    assert [(w.window_index, w.chunk_indices, w.text) for w in a] == \
           [(w.window_index, w.chunk_indices, w.text) for w in b]


def test_empty_input_yields_no_windows():
    assert cw.build_windows([], CFG) == []
