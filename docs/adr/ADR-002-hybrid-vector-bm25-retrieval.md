# ADR-002: Hybrid retrieval — FAISS vectors + BM25 with RRF, over parent-child chunks

- **Status:** Accepted
- **Date:** 2026-10-13
- **Deciders:** Doniele Antonio
- **Supersedes:** none

## Context

The brief requires *"document retrieval using a vector database to store and search document
content."* The reference corpus is Microsoft's FY2023 Annual Report: ~130 pages mixing narrative
MD&A, dense financial tables (income statement, segment revenue, Note 13 commitments), and
footnotes.

Two retrieval philosophies apply, and each fails alone on this corpus:

- **Dense (embedding) search** is strong on paraphrased questions ("why did margin fall?") but
  weak on *exact-token* financial lookups — "operating income FY2023", "Note 13", "$88,523",
  "diluted EPS" — where the query string is itself the best key. Embeddings also smear
  numerically adjacent facts: "$211,915 / $219,790 / $245,122" live in adjacent table rows and
  embed to nearly the same vector.
- **Sparse (BM25) search** nails those exact line items and rare section names, but misses
  semantic paraphrases entirely ("what drove the cloud business" vs. "Microsoft Cloud revenue").

Retrieval quality gates the whole answer; this is the highest-leverage component in the system.

## Decision

**Hybrid retrieval in `chunking-index` (`src/core/retrieval.py`) — persisted under `data/index/`:**

1. **Chunking:** detect headings + section paths during ingestion, build **parent chunks**
   (~1000 tokens, aligned to sections/tables) and **child chunks** (~250 tokens, overlap 50)
   via `RecursiveCharacterTextSplitter` with financial-friendly separators (blank line, table
   rows, sentence, word — in that order, so tables stay intact).
2. **Dense index:** embed child chunks (OpenAI-compatible embeddings endpoint), store in
   **FAISS** (`data/index/faiss/`) — this is the mandated vector database.
3. **Sparse index:** `rank_bm25.BM25Okapi` over child token lists, pickled to
   `data/index/bm25.pkl`.
4. **Fusion:** per sub-query, retrieve top-k from each; fuse with **Reciprocal Rank Fusion**,
   `score(d) = Σ 1/(k + rank_i(d))` with `k = 60`.
5. **Parent expansion:** fused child hits are expanded to their **parent chunk** text before
   synthesis — the LLM sees the full table/section context, the citation points at the parent's
   page range. Expansion happens *after* fusion so ranking is decided at child granularity.
6. **Source of truth:** `data/index/chunks.jsonl` holds texts + provenance; FAISS stores vectors
   only; BM25 stores token lists only. Rebuild any index from the JSONL.

**Parameters (defaults, tunable via sidebar/`k`):**

| Parameter | Default | Why |
|---|---|---|
| child size / overlap | 250 / 50 tokens | precise matching without cutting tables |
| parent size | ~1000 tokens | full table + heading context for the LLM |
| fused top-k (children → parents) | 8 | bounds context window; parent expansion sets final context |
| dense index type | `IndexFlatIP` over L2-normalized vectors | exact search = cosine similarity; at a few thousand vectors IVF/HNSW approximation buys nothing and costs recall + a training step. Switch to `IndexIVFFlat` only if the corpus grows past ~50k chunks — then it's a one-line change in `src/core/index.py` |
| RRF `k` | 60 | standard constant; dampens rank disagreements at the top |
| score gate `tau` | 0.02 (RRF units) | below → one rewrite retry, then answer with what we have |
| rewrite retries | 1 | bounded cost; prevents retrieval loops |

## Alternatives Considered

| Alternative | Pros | Cons | Verdict |
|---|---|---|---|
| Pure dense vector search | Simple, one index | Semantic drift on numbers; exact line items missed | rejected |
| Pure BM25 | Exact keyword strength | No paraphrase handling; brittle to table wording | rejected |
| Weighted-sum fusion | Tunable weight | Cosine ∈ [0,1] vs BM25 unbounded — scales incomparable, weights twitch per query | rejected; RRF is rank-only and scale-free |
| Cross-encoder reranker (top-50 → top-8) | Best precision on paper | Extra model, 100s of ms–s latency per query, another failure mode in a 46h build | deferred (noted in journal as next step) |
| MMR diversity | Reduces redundancy | We *want* redundancy across sub-queries on multi-part questions; diversity hurts citation clustering | rejected |
| ColBERT late interaction | Strong recall | Heavy dependency, deployment complexity | rejected for this scope |
| Qdrant / Weaviate server | Metadata filtering, scale, hybrid queries built in, persistence | **A second compose service** for a single-container demo; the corpus is ~130 pages (thousands of chunks — the loaded index is tens of MB, so startup load is trivial); a server adds a network hop, a healthcheck, a data volume, and a second failure mode between the app and its index. The index is **disk-persisted** (`data/index/faiss/`, written by `write_index`, reloaded via `read_index`), not in-memory — ADR-008 rejected the compose-service split for the same reason | rejected for this scope; the upgrade path is written, not built |
| Chroma (persistent local client) | Closer to drop-in, persists to disk, no server | Convenience over correctness here: metadata filtering happens *after* the dense search, so section filtering can't shape what FAISS/BM25 return; and our filtering must run *before* fusion to preserve parent-context provenance | rejected — filtering before fusion is load-bearing for citations |
| Vectorless tree navigation | No embeddings | **Rejected — see `docs/adr/ADR-003-vectorless-rag-evaluated.md`** | — |

## Consequences

- Two lightweight artifacts must stay in sync with `chunks.jsonl`; rebuild is a single script
  (`scripts/build_index.py`), so drift is recoverable.
- **In-place removal is FAISS's one real limitation, and it is documented rather than papered
  over.** `IndexFlatIP` has no `remove_ids` (arbitrary deletion was only added in newer FAISS via
  `IDSelector`-based removal on some index types, and not reliably on `IndexFlat`). So removing a
  single uploaded document means rebuilding the index from the surviving rows of
  `chunks.jsonl` — a one-command, sub-second operation at this corpus size. Adding is genuinely
  incremental (`add_with_ids`). The design consequence: **delete is a rebuild, add is an append**,
  and `chunks.jsonl` remains the source of truth so the rebuild is lossless. If per-document
  removal ever becomes a hot path, the switch is a Qdrant Cloud / pgvector migration, decided by
  measurement rather than by this exam.
- **Scaling ceiling is explicit.** `IndexFlatIP` is exact and O(n) per query — fine to ~50k chunks
  (tens of ms). Past that, `IndexIVFFlat` (approximate, needs training) then `IndexHNSWFlat`.
  Corpus is ~130 pages today; both thresholds are recorded in the parameter table so the choice is
  revisited with evidence, not vibes.
- Parent expansion doubles as table-context insurance: even if a child matched a caption row,
  the LLM receives the whole table.
- Questions that span multiple years ("compare FY23 vs FY22") now return both parent chunks —
  the exact scenario parent expansion exists for.
- Cost: retrieval runs once per sub-query (≤4), each sub-query is one embed call + two index
  reads — well under a second on CPU for a 130-page corpus.
- Grounding check (ADR-007) is the safety net if retrieval still misses: claims unsupported by
  retrieved text get trimmed and flagged, not silently emitted.
