# Stress Test: Delivery and demo risks

> Delivery-risk review against the 46-hour sprint. Register first, then the demo script that
> protects the submission, then the kill-switch list.

## Register

| # | Risk | Prob | Impact | Early-warning signal | Mitigation | Contingency (Plan B) |
|---|---|---|---|---|---|---|
| 1 | **API key/quota failure at demo time** (grader has no key, or quota dies mid-session) | High | Fatal | First synthesis call returns 401/429 | `MOCK_LLM=1` fully scripted path documented in README; `.env.example` explains the keyless demo | Ship the mock path as the default demo; journal explains the scripted mode mirrors real node flow |
| 2 | **`docker compose up` fails** (Windows/Mac Docker Desktop, no key set) | Med | Fatal | Any image build error in CI or first cold clone | CI runs `docker build` smoke on every PR; compose passes env through; `.env.example` copied by the README command sequence | README documents the native path (`pip install -r requirements.txt && streamlit run app.py`) as fallback; both tested before submit |
| 3 | **Context-length overflow** from parent expansion on a 130-page report | Med | High | `LLMError.ContextLength` in dev sessions; tokens_in per turn climbing | Global context budget in `GraphState` (greedy pack to ~6k tokens, see STRESS-TEST-RAG #6); retrieval k bounded by sidebar | Synthesis prompt handles truncation gracefully; k lowered automatically on repeated overflow and logged |
| 4 | **Nondeterministic answer quality flaps between demo runs** | Med | Med | Same question gives different answers across runs | Temperature 0.2 synthesis, 0.0 classification/verification; golden QA set pinned; prompts documented in `docs/prompts/` | Re-run golden QA before any demo; if a case regresses, revert to the last prompt version (versioned) |
| 5 | **Grader cannot reproduce** (cold clone breaks: missing corpus, missing key, missing volume) | Med | Fatal | "works on my machine" | README is a 4-command reproduction (`clone → cp .env → download corpus or MOCK_LLM=1 → compose up`); `scripts/download_corpus.py` fetches the PDF; `data/` on a named volume | Mock mode + empty-state UI ("upload a PDF to begin") means the app is never broken-looking |
| 6 | **Scope creep eats journal time** (extra features past the Must set) | Med | High | MoSCoW "Should/Could" items starting while Must items are unchecked | Kill-switch list below; `docs/TODO.md` ordered so all Must items precede everything else | Cut Could → then Should (in listed order); journal explicitly records what was cut and why |
| 7 | **Upload failure on an unusual PDF** (scanned, malformed) | Med | Med | ParseError in dev tests | Per-file error card in UI; other files still ingest; page-level `text_layer: false` warning | Partial ingestion with a visible quality banner — never a silent empty index |
| 8 | **FAQ: reviewer runs `pytest` and it fails** (no key, no network) | Med | High | Any network-dependent test in CI | All tests run against `MockLLM` / fixtures; no test touches the network; CI runs the full suite | Mock-mode tests are hermetic by design; CI proves it on every push |

## The demo script (order matters)

| Step | Action | Expected observable | Most likely break | Recovery |
|---|---|---|---|---|
| 1 | `docker compose up --build` | Streamlit at :8501, empty state "upload a PDF" | Missing `.env` | Copy from `.env.example` (README command) |
| 2 | Upload Microsoft FY2023 annual report | Index builds; sidebar shows doc, pages, sections; build latency in debug panel | `ParseError` on odd pages | Per-page warning card; other pages indexed |
| 3 | Ask "What was Microsoft's revenue in FY2023?" | **$211,915 million**, citation card → Income Statements page, debug panel shows 1 retrieval pass | Wrong-year figure | Stop; check extraction contract fix (STRESS-TEST-RAG #1) |
| 4 | Ask "Compare operating income FY23 vs FY22" | $88,523M vs $83,383M, both cited; debug shows decomposition into sub-queries | One year missing | Check parent expansion; re-run golden QA |
| 5 | Ask follow-up "which segment drove that?" | Resolves "that" from history without repeating the question; segment revenue cited | Pronoun drift | History window / `cited_pages` fix (STRESS-TEST-RAG #3) |
| 6 | Ask "What was revenue in 2019?" | Honest refusal naming fiscal-year coverage, no number invented | Classify whiplash | `missing_aspect` copy (STRESS-TEST-RAG #7) |

Steps 3–6 map 1:1 to `docs/SPEC.md` §10 success criteria — run them in order before every submit.

## Kill-switch features (cut in this order at hour 40 if red)

| Cut order | Feature | Exam credit lost | Why safe |
|---|---|---|---|
| 1 | Streaming synthesis (already a "Could") | none — not in the brief | Listed as Could in `docs/SPEC.md` MoSCoW |
| 2 | Rename/delete sessions (keep create/switch) | minimal | History persistence, the graded item, still works |
| 3 | Multi-document upload UI (keep single-doc path) | small | "Uploading multiple documents" is a "may include" bullet |
| 4 | LangSmith env integration | none | Optional add-on by design (ADR-007) |

**Non-negotiable Musts** (protect with time, never cut): cited answers for the revenue question,
decomposition on a two-part question, follow-up resolution, honest refusals, vector-DB
retrieval, Docker reproduction, JOURNAL.md. If any of these is red at hour 40, cut everything
else in the table above.
