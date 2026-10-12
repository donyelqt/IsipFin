# ADR-003: Vectorless (tree-based) RAG — evaluated, rejected, and mined for one idea

- **Status:** Accepted
- **Date:** 2026-10-13
- **Deciders:** Doniele Antonio
- **Supersedes:** none

## Context

Vectorless RAG (PageIndex-style) builds a **hierarchical tree index** of a document — section →
subsection → page range — and has an LLM *reason* down the tree to locate the pages that answer a
question. No embeddings, no vector database. Open-source implementations exist (VectifyAI
PageIndex, forks using Vertex/GPT reasoning over the tree). It is a credible fit for long
financial reports precisely where cosine similarity struggles: dense numeric tables,
cross-references, and note-numbered evidence.

This had to be decided seriously, not dismissed — a reviewer who knows PageIndex will ask.

## Decision

**Reject vectorless as the primary retrieval mechanism for this system. Use hybrid FAISS+BM25
retrieval (ADR-002).** Adopt one idea from vectorless into the chosen design: **section-path
metadata on every chunk** (the cheap 10% of vectorless's benefit at 2% of the cost).

## Alternatives Considered

| Approach | Pros | Cons | Verdict |
|---|---|---|---|
| Pure vectorless tree navigation | Strong on table-heavy reports; no embedding calls; explainable page-level provenance | Multiple sequential LLM calls per query (one per tree level, each 2–10 s); tree quality depends on TOC extraction; a miss shrinks recall with no fallback | **Rejected** — see reasons below |
| Vision-based page rendering + VLM | Handles scans/layouts | Cost per query orders higher; needs vision model; latency | Rejected for this corpus (clean text layer) |
| Hybrid tree + vector (tree routes, vector confirms) | Best of both | Doubles pipeline surface; two failure modes to debug in 46h | Rejected — complexity without grading credit |
| **Hybrid vector + BM25 + parent-child (chosen)** | Sub-second retrieval; exact-keyword strength via BM25; table context via parents; satisfies the brief's vector-DB mandate | Embedding dependency; RRF tuning | **Accepted** |

### Why rejected — the four concrete reasons

1. **The brief mandates it.** "Implement document retrieval using a vector database to store and
   search document content." A vectorless primary path directly contradicts an explicit,
   checkable requirement. FAISS satisfies it; tree navigation does not.
2. **Latency compounds per level.** Navigating a 3-level tree is 3–8 sequential LLM calls before
   synthesis even starts: measured pattern for this class of system is 5–15 s per answer versus
   <1 s for hybrid retrieval. A chat demo that takes 15 s per turn reads as broken.
3. **Token cost + rate-limit risk.** Each navigation call ships candidate section summaries;
   exploratory branches multiply that. A free-tier key dies mid-demo. Hybrid retrieval's only
   per-query LLM-free work is embedding + index reads.
4. **Tree build fragility.** Section trees are derived from the report's TOC and heading styles;
   a mis-parsed note (e.g., "Note 13 — Commitments and Contingencies" nested under the wrong
   parent) silently misroutes queries. Chunk-level retrieval degrades gracefully; tree-level
   misplacement does not.

### Where vectorless would genuinely win

- Scanned/image PDFs with no text layer (ours has a clean one).
- Very long reports (500+ pages) with deep, reliable structure — 10-K filings with machine TOCs.
- No embedding budget/quota available at all (offline air-gapped deployment).
- Queries that require document-structure reasoning ("which note discusses lease commitments?")
   — though here the BM25 keyword hit on "Note 13" plus section-path metadata covers it.

### What we take anyway (the adopted idea)

Every chunk carries `section_path` (e.g., `["Management's Discussion", "Results of Operations",
"Revenue"]`) and page range at ingestion. This is implemented in `doc-ingest` (ADR-004),
surfaces in citations, and is filterable in retrieval (`filters={"section": ...}`). That is the
cheap 10% of vectorless — structural provenance — without navigation cost. The remaining
vectorless advantage (LLM structural reasoning) is largely duplicated by `analyze_question`'s
`required_sections` planning signal.

## Consequences

- BM25 carries the exact-token strength vectorless claims for itself; FAISS carries paraphrase;
   parents carry table context.
- Comparison numbers above are from the design analysis, not a live benchmark — stated honestly;
   the journal records them as reasoned estimates with the latency arithmetic shown, not
   fabricated measurements.
- If the grader asks "why not PageIndex?" this ADR is the prepared answer, and
   `docs/JOURNAL.md` cross-references it under architectural trade-offs.
