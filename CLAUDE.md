# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development workflow: superpowers, always

All development in this repo uses the **superpowers** skills. This is not optional:

- **Before any feature/design work:** `superpowers:brainstorming`
- **Before writing code for a multi-step task:** `superpowers:writing-plans` — every phase gets its own executable plan in `docs/superpowers/plans/` before implementation starts
- **Implementing:** `superpowers:executing-plans` or `superpowers:subagent-driven-development`, with `superpowers:test-driven-development` (RED → GREEN → refactor; never write implementation before a failing test)
- **Any bug or unexpected behavior:** `superpowers:systematic-debugging` before proposing fixes
- **Before claiming anything works:** `superpowers:verification-before-completion` — run the commands, show the output

The user's cadence: review each plan step, implement, test, debug, verify exit criteria, then move to the next step. One phase at a time; do not start a phase whose executable plan hasn't been written and reviewed.

## What this project is

A **retrieval-only knowledge-base MCP server**: hybrid retrieval (BM25 + dense + RRF fusion → graph expansion → cross-encoder rerank → not-found gate) over a deliberately hard, mixed-domain corpus, exposed as three MCP tools (`search`, `fetch_document`, `list_documents`) over stdio. The client LLM synthesizes answers; the server returns cited evidence chunks and an explicit `not_found` signal.

**Read these before doing anything:**
- `docs/PLAN.md` — high-level goals, non-goals, success criteria
- `docs/superpowers/plans/2026-07-09-kb-mcp-server-master-plan.md` — the master technical plan: architecture, interfaces, port surface map, tier taxonomy, per-phase contracts and exit criteria

> **Note:** `docs/superpowers/` is gitignored — plans exist only on the primary dev machine. If you are on a fresh clone and the master plan is missing, stop and ask the user for it; do not improvise architecture from this file alone.

## Locked decisions — do not relitigate

1. **Retrieval-only.** No generation, no answer verification on the server. Client LLM does synthesis.
2. **RAG code is reference, not a copy source (amended 2026-07-15).** `C:\Users\aniru\RAG` is read for inspiration and technique — never transcribed verbatim. The user considers the old implementation functional but not optimal, and wants to deliberately enhance each module rather than port it as-is. **HARD STOP:** before invoking `superpowers:writing-plans` or writing any implementation code for a module listed in the master plan's §2 "Port Surface Map," stop, explicitly tell the user "this module needs an enhancement-planning pass before I proceed" naming the module, invoke `superpowers:brainstorming`, and wait for the user's actual input in the conversation. Do not treat reading the RAG source yourself as satisfying this — the pass is not complete until the user has weighed in. This supersedes the original "copy + adapt" framing; §2's category labels ("Copy near-verbatim" etc.) now describe what the old code does, not an instruction to transcribe it.
3. **No Ollama anywhere.** Embedder is `Snowflake/snowflake-arctic-embed-l-v2.0` via sentence-transformers (same weights as the RAG project's Ollama model). Runtime = pip install only.
4. **Stateless server.** No session state, no coref, no warm-start. Fresh internal state per call.
5. **Two install paths — demo + bring-your-own-corpus (Shape C, amended 2026-07-16).** (a) A prebuilt CKB ships as a release asset for a zero-build demo/proof over the 60-doc hard corpus; (b) bring-your-own-corpus — users install the `[corpus]` extra and run `scripts/build_corpus.py` against their own `manifest.yaml` to build a CKB over their own documents. The 60-doc corpus is the eval/proof fixture **and** the demo content, not the only supported content. This is an open-source project meant to be used by others, so BYO is first-class. Contextual Retrieval (Haiku) must be **optional/config-gated** (`contextualize.enabled`) so the BYO path works without an API key. Supersedes the earlier "ingest is dev-only" framing.
6. **Usability decisions (2026-07-16, with user).** Code license: **MIT**. Shipped index: **ship all chunk text with a fair-use + per-source attribution `NOTICE`** (keep the per-source `license` field so any source can be swapped/removed in one line if challenged; prefer CC-licensed equivalents where a swap is trivial). BYO corpus input: support **both** a `manifest.yaml` **and** a friction-free **local-folder** mode (`build_corpus.py --folder`) — local-file ingest is first-class; local files' `version` defaults to a content hash. Contextual Retrieval **defaults OFF when no API key is present** (build proceeds on raw text with a note); the shipped corpus is built with it ON.

## House rules (carried over from the RAG project)

- **YAML config over hard-coded constants** — all tunables live in `config/*.yaml`.
- **Deterministic, stable IDs** — chunk IDs are `sha256(canonical_url + version + chunk_index)` conceptually; the implementation (`src/kbmcp/models/ids.py`) joins fields with the ASCII unit-separator `\x1f` before hashing (prevents boundary-collision, e.g. `"ab"+"c"` vs `"a"+"bc"`) — never break this scheme, in concept or exact serialization (citations and eval gold labels depend on it). `config/corpus_config.yaml`'s `ids.scheme` field must stay in sync with this description.
- **Reproducible artifacts** — corpus builds, eval runs, and reports must be re-runnable and committed where they are proof artifacts (eval reports especially).
- **Never write to stdout in server code** — stdio transport; logs go to stderr.

## Environment & commands

- Windows 11, PowerShell. Venv at `venv\` — Python 3.11.9 (matches project pin `>=3.11,<3.13`).
- **Always invoke the venv interpreter directly** — `venv\Scripts\python.exe -m pytest ...` — because shell activation does not persist between Claude Code tool calls.
- Repo status: **Phase 0 complete.** Phase 1 (corpus construction) in progress — Milestone 1 (Foundations & Data Model: `[corpus]` extra, deterministic sha256 IDs, 5-table SQLite schema + ops, `corpus_config.yaml` loader, BYO-compatible manifest loader) complete and reviewed (task reviews + whole-milestone review + independent adversarial review). Remaining Phase 1 milestones: fetch & pin, Docling parse & chunk, Contextual Retrieval, cross-ref graph, BM25 index + `build_corpus.py`, benchmark + gold resolution.
- Working conventions:
  - Install: `venv\Scripts\python.exe -m pip install -e ".[dev]"`
  - Tests: `venv\Scripts\python.exe -m pytest tests\test_x.py::test_name -v`
  - Toy MCP server (Phase 0 reference, not production): `examples\toy_server.py`, registered with Claude Code as `kb-toy`
- Planned conventions (not yet implemented — see master plan for phases):
  - Package layout: `src/kbmcp/`; console entry point `kbmcp`
  - Corpus build (dev-only): `python scripts/build_corpus.py`
  - Eval: `python -m kbmcp.eval run` → per-tier report in `eval/runs/<ts>/report.md`
