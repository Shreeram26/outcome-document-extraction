# DESIGN.md

## What the service does

Takes a document (PDF/DOCX/XLSX/CSV) plus a field schema and returns extracted
fields as contract-valid JSON. `POST /jobs` acknowledges in <2s and processes
in the background (10 in-flight, 429 beyond); `GET /jobs/{id}` is tenant-scoped
(`X-Organization-Id`, 404 for foreign jobs); terminal statuses are immutable;
idempotent resubmission returns the existing job. Results are validated with
the assignment's own `tools/validate.py` before a job is marked terminal.

## Extraction pipeline (the interesting part)

**1. Parsing (`app/parsing/`).** One parser per format returning a common
`ParsedDocument` with source locations (page numbers, `Sheet!A1` refs) so every
extracted value can carry a citation. XLSX uses `read_only`+`data_only` (cached
formula values), CSV sniffs delimiters, PDFs flag image-only pages as scanned.

**2. Deterministic-first extraction (`app/extraction/deterministic.py`).**
Before any LLM call: labelled-value matching (field names/keys/quoted
description phrases, exact > quoted > token-subset, ambiguity rejected),
table detection with scored header candidates, generic row filtering (blanks,
summary words, duplicate primaries, non-numeric rows in numeric-heavy tables),
and sheet-relevance pools so auxiliary sheets cannot poison document-level
values. Roughly 30% of the sample corpus (boxscore, rentroll rows) resolves
with zero LLM calls.

**3. Coverage-driven LLM fallback (`app/extraction/engine.py`).** The key
lesson from the accuracy campaign: field-driven expansion (stop when every
field has a value) silently loses ROWS — on budget_001 it left 15 of 55 records
unsent and follow-up stages asked only for unresolved fields, producing
2-field rows. Multi-record mode is now **coverage-driven**: stages run while
the previous context was trimmed (unsent content exists), every call uses the
FULL schema so new records carry all fields, and chunks are row-bounded
(4,000 chars ≈ 15–25 rows) so responses finish inside the completion budget.
Single-entry mode stays field-driven (cheaper). Table mode adds one bounded
LLM call for doc-level fields the deterministic stage left unresolved.

**4. Response handling.** Tolerant JSON parsing (fences, prose, bare values,
currency coercion) into `FieldResult(value, citation, confidence)`; null
discipline enforced (null ⇒ empty citation); merged by primary key with
conflict recording; deterministic values never overwritten; contract
validated with `tools/validate.py` before a job goes terminal.

**5. Cost control (`app/extraction/ledger.py`).** One thread-safe
CostLedger: every call reserves its estimated maximum cost before launching
(input ≈ chars/4, output bounded by an enforced max_tokens=16000, per-model
pricing × 1.25 safety factor); refusals happen before network use. Settled
with provider-reported `usage.cost`; RETAINED on watchdog timeouts (the
provider may still bill); reconciled when abandoned requests complete;
released only on definitive pre-response failures. Metadata distinguishes
`cost_usd` (conservative exposure) from `cost_settled_usd`.

## Model findings (measured, see experiments/)

- `anthropic/claude-haiku-5.5` — default. Direct JSON, cheap ($0.034 for
  budget_001 at 0.971), best engine-level record coverage (40 rows even with
  the old field-driven engine).
- `gpt-6-luna` + `response_format: json_schema` — same quality at ~⅓ the
  cost, 2/2 clean parses. json_schema works on openai-provider models but
  **intermittently HTTP-400s on anthropic-provider models** → kept opt-in via
  gateway `extra_params`, not a global default.
- Reasoning models (gemini-3.8-flash, glm-5.3-flash) consumed their entire
  completion budget on hidden reasoning and returned no content at 16k/32k
  tokens — avoided.
- Tool calling: unusable (400s / inconsistent shapes). Field-level
  cheap→strong escalation: measured +$0.137 for zero gain — rejected.
- Watchdog: provider keep-alives defeat client read timeouts; a worker-thread
  watchdog (150s) converts hangs into handled errors. A timed-out request is
  still billed by the provider, so its reservation is retained and
  reconciled when the abandoned request completes.

## Trade-offs and limitations

- Scanned PDFs (lease_001/003) have no vision stage: the engine correctly
  skips them and returns contract-valid null-heavy results (scores 0.15/0.075).
  A vision pass is the main accuracy opportunity left.
- rentroll per-row fields stay null when deterministic resolution misses them;
  the single table-fallback call fills document-level fields but not 325 rows.
- PDF tables are extracted as text (PyMuPDF `find_tables` mangles borderless
  headers — measured); the LLM infers columns from spacing.
- t12-style grids can emit extra summary rows; dedupe keeps first by primary.
- In-memory job store: restarts lose job history (acceptable for the
  assignment; a persistent store is the obvious production upgrade).
- Abandoned-but-later-billed requests reconcile into the run ledger after the
  document's own exposure snapshot — per-doc exposure is an estimate for
  timed-out calls (conservative).

## What I would do with more time

Vision stage for scanned PDFs; per-row LLM fill for table docs; primary-key
reconciliation against parsed source rows to drive targeted re-extraction of
missing records; persistent job store; per-model structured-output capability
flags with automatic fallback.
