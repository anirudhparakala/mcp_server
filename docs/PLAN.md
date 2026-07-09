# PLAN.md — Knowledge-Base MCP Server

## Overview

An MCP server that exposes a retrieval engine over a deliberately hard,
self-curated document corpus, built by porting the hybrid retrieval +
reranking + citation-verification pipeline from an existing RAG project into
MCP tool form. Built to be usable by any MCP-capable client (Claude Code,
Claude Desktop, Cursor, etc).

This is a rough plan, not an implementation plan. It defines what gets built
and in what order, not how. Implementation detail is a separate pass.

## Problem / Motivation

Knowledge-base MCP servers are a known gap in the current MCP ecosystem —
most public options are thin wrappers around simple note-taking tools and
fall apart on anything requiring real retrieval quality (multi-hop reasoning,
disambiguation, synthesis across documents, correctly saying "not found").
Most "knowledge base" demos also cheat by using easy, clean documents. This
project deliberately does the opposite: the corpus is built to be hard.

## Goals

1. A working MCP server exposing real hybrid retrieval, not keyword search.
2. A self-built document corpus that is genuinely difficult to retrieve over
   correctly — not simple lookup material.
3. An evaluation layer proving retrieval quality on that hard corpus, not
   just a demo that looks fine on easy queries.
4. Something a handful of real people can actually run and use.

## Non-Goals

- Not a general-purpose "MCP for any document" tool.
- Not competing with Notion/Obsidian-style note-taking MCP servers.
- Not optimizing for corpus size. A smaller, genuinely hard corpus beats a
  huge easy one.
- Not building new retrieval algorithms from scratch — reusing and adapting
  the existing hybrid retrieval + reranking pipeline, not reinventing it.

## Corpus Strategy (what "hard" means here)

The corpus is built from internet-accessible sources only, chosen and
assembled specifically to stress retrieval, not just to fill a knowledge
base. Hardness comes from things like:

- Documents that require synthesizing information across multiple sources to
  answer a single question (multi-hop).
- Sources that contain overlapping or partially contradictory information,
  where the server has to represent that rather than average it away.
- Deep technical material where surface keyword overlap is misleading.
- Queries deliberately phrased ambiguously, requiring disambiguation before
  retrieval is even useful.
- Cases where the correct answer is "not in the corpus" — the server must
  say so instead of fabricating.

The corpus should be organized into difficulty tiers (reusing the tiered-eval
approach from the earlier RAG project) so retrieval quality can be measured
per tier, not just as one blended number.

## High-Level Phases

### Phase 0 — MCP fundamentals
Get to the point of running one existing simple MCP server end to end and
understanding the tool/resource/host/client model well enough to build one.

### Phase 1 — Corpus construction
Source and assemble the hard document collection from public sources.
Organize it into the difficulty tiers described above. This phase produces
the dataset, not any retrieval code.

### Phase 2 — Retrieval core port
Adapt the existing hybrid retrieval + reranking + citation-verification
pipeline to run against the new corpus, as a standalone callable service
(no MCP yet). Prove it works correctly on a sample of queries per tier
before moving on.

### Phase 3 — Evaluation pass
Run the retrieval core against the full tiered corpus and produce real
accuracy/citation-quality numbers per tier, using the same evaluation
methodology as the earlier RAG project. This is what proves "hard" and
"working" aren't just claims.

### Phase 4 — MCP server wrapper
Expose the proven retrieval core as a small set of MCP tools (search, fetch
document, and whatever else the corpus genuinely needs — kept minimal).

### Phase 5 — Hardening
Failure handling, logging, minimal auth, graceful degradation on bad or
adversarial queries.

### Phase 6 — Ship
README, publish to GitHub and relevant MCP registries/listings, write the
resume-facing narrative.

### Phase 7 — Real usage
Get real people to run real queries against it, collect what broke, do one
feedback-driven revision pass.

## Success Criteria

- Retrieval eval numbers exist per difficulty tier, not just overall.
- The server correctly declines to answer when the answer isn't in the
  corpus, at a measured rate, not just anecdotally.
- At least one real external user has used it and given feedback.
- A stranger can clone the repo and get it running from the README alone.

## Explicitly Out of Scope for v1

- Multi-corpus support.
- Enterprise auth (OAuth, SSO).
- Horizontal scaling / remote hosted deployment.
- Any UI beyond the MCP tool interface itself.
