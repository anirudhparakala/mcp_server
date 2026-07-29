# kbmcp corpus

The curated, deliberately-hard, mixed-domain corpus that is both the **eval/proof
fixture** and the **demo content** for the kbmcp retrieval server. Sources were
hand-selected around specific retrieval traps (7-tier taxonomy — see
`docs/superpowers/specs/2026-07-16-phase1-corpus-design.md`) and web-verified,
version-pinned, and fetched in Milestone 2 (2026-07-23).

## Contents (55 sources)

| Domain | Count | Mix |
|---|---|---|
| AI / RAG / LLM (`ai`) | 18 | arXiv papers, 2 author-self-archived papers, Wikipedia, HF model cards |
| Legal — AI regulation (`law_aireg`) | 11 | EUR-Lex (GDPR, EU AI Act), CA statutes, EDPB, ICO, NIST, Wikipedia |
| Legal — contract law (`law_contract`) | 12 | CA Commercial/Civil Code, 5 public-domain cases, a SEC EDGAR contract, Wikipedia |
| Fitness / nutrition (`fitness`) | 14 | JISSN/PMC open-access papers, USDA DGA, Wikipedia, 1 authored waiver |

Formats: 33 HTML, 22 PDF. The full per-source record — URL, license, pinned
version, tier roles, collision terms, cross-references — lives in
[`manifest.yaml`](manifest.yaml).

## Licensing posture

The shipped index carries chunk text under a **fair-use + per-source attribution**
posture (normal for a showcase OSS project). Every source's **actual license** is
recorded in `manifest.yaml`'s `license` field, so any single source can be
swapped or removed in one line if ever challenged. License mix:

- **Open licenses:** arXiv non-exclusive; CC BY 4.0 (JISSN/PMC papers, AlphaFold,
  HF model cards Apache-2.0); CC BY-SA 4.0 (Wikipedia, Wikisource); **CC BY-NC**
  (Morton 2018 — kept as load-bearing, flagged first-to-remove); CC0 (the one
  authored document below).
- **Public domain / government works:** NIST, USDA (17 U.S.C. §105); California
  statutes and pre-modern judicial opinions (government edicts).
- **Reuse-authorized:** EUR-Lex / EDPB (EU Decision 2011/833/EU, with
  acknowledgement); ICO (UK Open Government Licence).
- **Public record:** the SEC EDGAR contract filing.
- **Author self-archived (publisher ©, fair-use + attribution):** BM25 monograph
  (© Now Publishers) and RRF paper (© ACM) — the two least-defensible sources,
  first candidates for removal if challenged.
- **Authored for this corpus (CC0):** [`authored/sample-gym-liability-waiver.html`](authored/sample-gym-liability-waiver.html)
  — an original, generic gym liability-waiver + informed-consent document
  (placeholder facility, no real entity), created because no cleanly-licensed real
  waiver exists. Serves the "waiver/release/consent" cross-domain collision.

## Reproducibility & pinning

- **Versions are author-pinned:** arXiv `vN`, Wikipedia/Wikisource `oldid`,
  EUR-Lex consolidation date, EDGAR accession, else retrieval date. Local/authored
  files use a content hash (`sha256:…`).
- **Pin records are committed, raw bytes are not:** each source has a committed
  `raw/<doc_id>.meta.json` (resolved version + sha256 content-hash + provenance);
  the raw bytes (`raw/*.pdf`, `raw/*.html`) are gitignored and re-fetchable +
  hash-verified from the record. `raw/.gitkeep` keeps the directory tracked.
- Re-running `python -m kbmcp.ingest.fetch` is cache-first and reproducible.

### Notable fetch routes (learned during verification)

- **PMC** (`pmc.ncbi.nlm.nih.gov`) reCAPTCHA-blocks bots → PMC open-access
  articles are fetched via the **Europe PMC PDF endpoint**
  (`europepmc.org/articles/PMC<id>?pdf=render`).
- **Justia / nycourts.gov / SEC EDGAR** block Python's TLS fingerprint → fetched
  via **`curl_cffi` browser TLS impersonation** (config `fetch.impersonate.hosts`).
- **Authored / committed local files** use a repo-relative path (resolved from the
  repo root), not a machine-specific `file://` path.

## Sources swapped or dropped during verification (audit trail, 2026-07-23)

All decisions made with the user:

| Source | Action | Reason |
|---|---|---|
| AlphaFold | kept (⚠️→OK) | Nature Open Access CC BY 4.0 (via Europe PMC); sourced from PMC |
| BM25 monograph, RRF paper | kept | publisher ©, author self-archived; load-bearing (RRF is a T1 anchor + this server's fusion) |
| OECD AI Principles | **dropped** | bespoke "not altered / may not be sold" license conflicts with a chunked KB; non-load-bearing |
| UCC Art 1 & 2 (Cornell LII) | **swapped** → CA Commercial Code | Cornell LII UCC is ALI/NCCUSL © (academic-use only); CA-enacted code is public domain |
| Wex ×3 (Cornell LII) | **swapped** → Wikipedia | Wex is CC BY-NC-SA 2.5; Wikipedia equivalents are CC BY-SA 4.0 |
| WHO/FAO TR 935 | **dropped** | all-rights-reserved (pre-CC-IGO) + JS-gated; redundant RDA cross-check |
| Schoenfeld frequency/volume metas | **swapped** → Swinton 2024 | classic papers are paywalled/not in PMC; open dose-response meta substitutes |
| Gym waiver | **authored** | no cleanly-licensed real waiver exists; original CC0 document |

## Model self-references (meta)

The two HF model-card sources point at this server's **actual** Phase-2 models,
modernized 2026-07-23 against 2026 SOTA: embedder `Qwen/Qwen3-Embedding-0.6B` and
reranker `BAAI/bge-reranker-v2-m3` (both Apache-2.0).
