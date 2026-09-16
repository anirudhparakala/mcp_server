# kbmcp

Retrieval-only knowledge-base MCP server. See `docs/PLAN.md`.

## Licensing

- **Software:** MIT — see [LICENSE](LICENSE).
- **Corpus content** under `corpus/` (and any knowledge base built from it) is
  **not** covered by that license. It comes from 55 third-party sources that
  retain their own terms — see [NOTICE](NOTICE) for per-source attribution.

`NOTICE` is generated from `corpus/manifest.yaml`; regenerate it after any
manifest change:

```
venv\Scripts\python.exe scripts/generate_notice.py
venv\Scripts\python.exe scripts/generate_notice.py --check   # verify, don't write
```

If you hold rights in any listed material and object to its inclusion, open an
issue and it will be removed.

## Building a corpus

Two paths, both via one command:

```
# Your own documents — no manifest, no flags
venv\Scripts\python.exe scripts/build_corpus.py --folder ./my-docs --ckb ckb/mine.sqlite

# A curated manifest (the shipped 55-source corpus)
venv\Scripts\python.exe scripts/build_corpus.py
```

It runs fetch → parse/chunk/contextualize → cross-reference graph → BM25 index →
verification gates, prints a per-stage summary, and exits non-zero if anything
failed. A partial corpus is still written and usable — one dead link does not
discard the rest.

Contextual Retrieval needs `ANTHROPIC_API_KEY`; without it the build proceeds on
raw text and says so. `--no-context` skips it explicitly.
