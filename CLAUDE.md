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
2. **Copy + adapt from `C:\Users\aniru\RAG`.** That repo is the source for the retrieval pipeline, ingest pipeline, and eval framework. The master plan's §2 "Port Surface Map" says exactly what is copied verbatim, rewritten, or dropped. Read the RAG original before porting a module.
3. **No Ollama anywhere.** Embedder is `Snowflake/snowflake-arctic-embed-l-v2.0` via sentence-transformers (same weights as the RAG project's Ollama model). Runtime = pip install only.
4. **Stateless server.** No session state, no coref, no warm-start. Fresh internal state per call.
5. **Prebuilt index ships to users** (release asset); ingest is dev-only behind a `[corpus]` extra.

## House rules (carried over from the RAG project)

- **YAML config over hard-coded constants** — all tunables live in `config/*.yaml`.
- **Deterministic, stable IDs** — chunk IDs are `sha256(canonical_url + version + chunk_index)`; never break this (citations and eval gold labels depend on it).
- **Reproducible artifacts** — corpus builds, eval runs, and reports must be re-runnable and committed where they are proof artifacts (eval reports especially).
- **Never write to stdout in server code** — stdio transport; logs go to stderr.

## Environment & commands

- Windows 11, PowerShell. Venv at `venv\` — Python 3.11.9 (matches project pin `>=3.11,<3.13`).
- **Always invoke the venv interpreter directly** — `venv\Scripts\python.exe -m pytest ...` — because shell activation does not persist between Claude Code tool calls.
- Repo status: **pre-implementation** — Phase 0 not started; no `src/`, no tests, no `pyproject.toml` yet. Update this section as commands become real.
- Planned conventions (from the master plan — confirm against `pyproject.toml` once it exists):
  - Tests: `venv\Scripts\python.exe -m pytest tests\test_x.py::test_name -v`
  - Package layout: `src/kbmcp/`; console entry point `kbmcp`
  - Corpus build (dev-only): `python scripts/build_corpus.py`
  - Eval: `python -m kbmcp.eval run` → per-tier report in `eval/runs/<ts>/report.md`
