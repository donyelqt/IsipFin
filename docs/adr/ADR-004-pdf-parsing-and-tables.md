# ADR-004: Document processing pipeline — pypdf text, pdfplumber tables, section-path provenance

- **Status:** Accepted
- **Date:** 2026-10-13
- **Deciders:** Doniele Antonio
- **Supersedes:** none

## Context

The brief asks for *"pipelines for robust document processing (e.g., text extraction, splitting,
cleaning)"* and *"improving text and table output formatting."* The reference corpus (Microsoft
FY2023 Annual Report) is a ~130-page, text-layer PDF with a dense TOC, multi-page financial
tables, footnotes, and per-page headers/footers. The output of this pipeline is the raw material
every retrieval quality claim depends on.

## Decision

**`doc-ingest` module — two-stage parse, structure-aware:**

1. **Text:** `pypdf` extracts per-page text (1-based page numbers). No layout analysis needed for
   the body.
2. **Table detection + extraction:** a page heuristically flagged as table-bearing
   (regex for ≥4 pipe-aligned or whitespace-columned numeric runs, or a caption matching
   `(table|note)\s+\d+`) is re-parsed with `pdfplumber` `.extract_tables()`. Each detected grid
   is serialized to a **markdown table** and emitted as `Block(kind="table", text=md, page, section_path)`.
3. **Structure:** heading detection by (a) `pypdf` font-size/style metadata when present,
   (b) regex fallback: ALL-CAPS lines, `^Note \d+`, numbered sections
   (`^(\d+\.|[A-Z]\.)\s+[A-Z]`). Every subsequent block inherits and extends the current
   `section_path`.
4. **Cleaning** (applied per block, order matters):

| Rule | Rationale |
|---|---|
| Strip repeated page headers/footers (lines recurring on ≥60% of pages) | Removes "Microsoft 2023 Annual Report" boilerplate that would dominate retrieval |
| De-hyphenate line-broken words (`operat-\ning` → `operating`) | BM25 keys on exact tokens; broken words lose recall |
| Collapse whitespace runs; drop blocks < 20 chars after cleaning | Kills OCR-ish noise without losing content |
| Preserve numbers **exactly as printed** (thousands separators, parentheses for negatives) | Financial answers must round-trip to the document; normalization happens at display, never at storage |
| Keep table header rows with every table block | Detached numeric rows are unreadable to the LLM |

5. **Provenance invariants:** every `Block` carries `page` (1-based) and `section_path`; pages
   with no extractable text are recorded with `text_layer: false`, surfaced in the UI as a
   warning, and never silently dropped.
6. **Idempotency:** `doc_id = sha256(pdf bytes)`; re-uploading the same file reuses
   `data/index/` artifacts and the `documents` row — ingestion is a no-op.
7. **Tables spanning pages:** when page N+1's first rows repeat page N's header signature, merge
   into one table block with the page range preserved.

## Alternatives Considered

| Alternative | Pros | Cons | Verdict |
|---|---|---|---|
| Unstructured.io | Excellent layout pipeline, community connectors | Heavy dependency tree (torch-adjacent optional deps), a service process to run in Docker, overkill for a clean text-layer PDF | rejected |
| GROBID | Strong on academic/scientific structure | Built for papers, not 10-K-style financials; Java service dependency | rejected |
| Raw text-only pypdf | Simplest | Tables degrade to unreadable number streams — the exact failure the exam's "table formatting" bullet targets | rejected as sole path |
| Camelot/Tabula | Mature table extraction | pdfplumber's table API is simpler and handles this report's ruled tables well; Tabula needs Java | rejected |
| **pypdf + pdfplumber on demand (chosen)** | Two light deps; tables treated specially | Two parsers to keep consistent | **accepted** |

## Consequences

- Table blocks retrievable as tables → synthesis prompt can render markdown tables, feeding the
  exam's "improving text and table output formatting" criterion directly (see
  `docs/prompts/table-and-number-formatting.md`).
- Markdown serialization loses visual layout (merged cells, footnotes) — accepted; the LLM
  reads aligned markdown reliably, and the citation still points at the PDF page for humans.
- Failure modes explicitly handled: scanned page → warning + low-quality flag, not silence;
  multi-column sections → footer/header stripping plus blank-line splitting keeps columns
  partially ordered (worst case still retrievable, flagged in journal as known limitation).
- `Block`/`ParsedDocument` dataclasses (frozen in `docs/SPEC.md` §9) are the contract
  `chunking-index` consumes — parsing changes never leak into retrieval code.
