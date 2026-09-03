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
