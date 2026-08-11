"""Context windows — partition a document's chunks into contiguous, token-budgeted runs.

Pure and offline: no I/O, no API, deterministic given the same records + config.

Each window is a run of WHOLE heading_path sections packed up to
``window_target_tokens``; a single section larger than ``window_max_tokens`` is
split at chunk boundaries. Windows PARTITION the document — every chunk belongs
to exactly one window — so all chunks in a window share one byte-identical
prompt prefix and therefore hit the Anthropic prompt cache (see the 2026-08-11
amendment to ingest design spec section 5).

Token counts here are a cheap offline estimate (chars / chars_per_token), not
Claude's tokenizer; the budgets carry enough slack to absorb the error.
"""

from dataclasses import dataclass
from math import ceil


def est_tokens(text: str, chars_per_token: int = 4) -> int:
    """Deterministic offline token estimate (never returns 0)."""
    return max(1, ceil(len(text) / max(1, int(chars_per_token))))


@dataclass(frozen=True)
class ContextWindow:
    window_index: int
    chunk_indices: tuple
    text: str
    est_tokens: int


def render_chunk(record) -> str:
    """One chunk as it appears inside a window (heading line handled by the caller)."""
    return record.text


def _heading_line(heading_path) -> str:
    path = [str(h) for h in (heading_path or []) if str(h).strip()]
    return "## " + " > ".join(path) if path else "## (no heading)"


def _sections(records) -> list:
    """Maximal runs of consecutive records sharing the same heading_path."""
    runs = []
    for rec in records:
        key = tuple(rec.heading_path or [])
        if runs and runs[-1][0] == key:
            runs[-1][1].append(rec)
        else:
            runs.append((key, [rec]))
    return runs


def _render(section_runs) -> str:
    """Render [(heading_key, [records]), ...] as window text, one heading line per run."""
    blocks = []
    for key, recs in section_runs:
        body = "\n\n".join(render_chunk(r) for r in recs)
        blocks.append(f"{_heading_line(list(key))}\n{body}")
    return "\n\n".join(blocks)


def _emit(windows, section_runs, chars_per_token) -> None:
    if not section_runs:
        return
    text = _render(section_runs)
    indices = tuple(r.chunk_index for _, recs in section_runs for r in recs)
    windows.append(
        ContextWindow(
            window_index=len(windows),
            chunk_indices=indices,
            text=text,
            est_tokens=est_tokens(text, chars_per_token),
        )
    )


def build_windows(records, cfg: dict) -> list:
    """Partition records (ascending chunk_index) into contiguous ContextWindows."""
    if not records:
        return []
    target = int(cfg.get("window_target_tokens", 6000))
    hard_max = max(target, int(cfg.get("window_max_tokens", 8000)))
    cpt = int(cfg.get("chars_per_token", 4))

    windows: list = []
    current: list = []          # [(heading_key, [records])] accumulated for the open window
    current_tokens = 0

    for key, recs in _sections(records):
        section_tokens = est_tokens(_render([(key, recs)]), cpt)

        # A single section over the hard cap: close the open window, then split it
        # at chunk boundaries into sub-windows that each stay under the cap.
        if section_tokens > hard_max:
            _emit(windows, current, cpt)
            current, current_tokens = [], 0
            part: list = []
            part_tokens = 0
            for rec in recs:
                rec_tokens = est_tokens(_render([(key, [rec])]), cpt)
                if part and part_tokens + rec_tokens > hard_max:
                    _emit(windows, [(key, part)], cpt)
                    part, part_tokens = [], 0
                part.append(rec)
                part_tokens += rec_tokens
            _emit(windows, [(key, part)], cpt)
            continue

        if current and current_tokens + section_tokens > target:
            _emit(windows, current, cpt)
            current, current_tokens = [], 0
        current.append((key, recs))
        current_tokens += section_tokens

    _emit(windows, current, cpt)
    return windows
