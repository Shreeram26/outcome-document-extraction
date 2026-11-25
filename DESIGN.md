# DESIGN.md

## Architecture

The service accepts a document (PDF/DOCX/XLSX/CSV) and a field schema, returning
extracted values as contract-valid JSON. The pipeline is deterministic-first:
rule-based extraction runs before any LLM call, and only unresolved fields reach
the model. A FastAPI backend wraps this pipeline with async job processing,
tenant isolation, idempotency, and cost safeguards.

## Document parsing

One parser per format produces a `ParsedDocument` with source locations (page
numbers for PDFs, `Sheet!Cell` refs for XLSX/CSV) so every extracted value can
carry a citation.

- PDF: PyMuPDF extracts page-wise text; pages with no text layer and an image
  are flagged as scanned (candidates for vision).
- DOCX: `iter_inner_content()` preserves paragraph/table reading order; tables
  get `rXcY` cell references.
- XLSX: `read_only=True` + `data_only=True` (cached formula values, not formula
  strings); every cell carries its A1 reference; trailing empty cells are
  trimmed per row.
- CSV: delimiter sniffing (`,` `;` `\t` `|`), UTF-8 BOM tolerant, single sheet
  named `"Sheet1"` (matches reference outputs).

## Deterministic-first extraction

Before any LLM call, the deterministic stage resolves fields using two generic
mechanisms:

1. **Labelled values**: a field's display name, key, or quoted phrases from its
   description are matched against `Label: value` patterns in the document.
   Match quality is tiered (exact name/key > quoted description > token-subset).
   A source label claimed by multiple fields at equal strength is contested and
   left unresolved rather than guessed. Sheet relevance pools prevent auxiliary
   sheets from poisoning document-level values.
2. **Table columns**: when a primary field exists and a header row matches enough
   field labels, one entry is produced per data row. Candidate tables are scored
   on data-row count, primary uniqueness, data-row proportion, mapped-field
   count, and match quality. Row filtering drops blanks, summary rows (exact
   word match on primary), label-only rows, duplicate primaries, and non-numeric
   rows in numeric-heavy tables.

Block-structured tables (primary row followed by empty-primary continuation
rows) are handled by a continuation-enrichment pass: values under header-mapped
columns fill the record's null fields, and cells matching quoted phrases from
field descriptions claim rows for those fields. Ambiguous cells are dropped
rather than guessed.

In the sample set, boxscore_001 resolves entirely deterministically (score 1.000)
and rentroll_001 resolves 325 rows + 12 fields deterministically.

## Coverage-driven LLM fallback

Fields the deterministic stage leaves unresolved are extracted by a
vision-language model. The fallback is budget-conscious and progressive:

1. **Targeted** (6k chars): context built from blocks matching the unresolved
   fields' labels plus the document head.
2. **Expanded** (12k chars): larger context budget.
3. **Full** (4k-char chunks): the document split at structural boundaries, each
   chunk receiving only fields whose labels appear in it.

Multi-record mode is **coverage-driven**: stages run while the previous context
was trimmed (unsent content exists), regardless of field coverage. This was the
key fix for budget_001 — field-driven expansion left 15 of 55 records unsent.
Each call uses the FULL schema so new records carry every field. Grid fallback
(failed header match on XLSX/CSV) uses row-bounded chunks (4k chars) to prevent
output truncation.

## Record-completeness prompt

For primary-key schemas, the system prompt adds: "Extract EVERY record present
in the supplied context — do not stop early. Each entry must contain EVERY
field; set a field to null only when it is genuinely absent for that record."
A 60-character citation cap limits output verbosity. These changes raised
budget_001 from 13% to ~97% field completeness.

## Schema-description defaults

Field descriptions may explicitly state an absent-value default ("false
otherwise", "0 if none", "empty list if none"). The engine parses these
generically and applies them only when: (a) the field is null in the final
entries, (b) the document has extractable text, (c) the last extraction attempt
did not fail, and (d) the default is type-compatible with observed values for
that field. Defaults never override extracted values. This fixes lease_002's
boolean/count fields (`letter_of_credit_required: false`, `renewal_options_count: 0`).

## Scanned-page vision fallback

PDFs whose pages have no text layer are extracted by rendering those pages to
PNG (200 dpi) and sending them as base64 `image_url` content blocks to a
vision-capable model. Pages are batched (5 per call) to stay within payload
limits; results from all batches are merged (nulls filled). Documents with a
text layer are never sent to vision.

Model: `anthropic/claude-haiku-5.5`, selected from a controlled 3-model
experiment on lease_001 pages 2+18 (200 dpi). Results: accuracy 0.70, zero
hallucinations, 100% citation coverage, $0.0034/call. Alternatives: qwen3-vl
(higher completeness 0.825 but zero citations and 2 hallucinations), gemini-3.8-flash
(lower accuracy, 6× cost, reasoning-token hang at 16k tokens). The experiment
cost $0.028 total.

## Model gateway and cost ledger

All LLM calls go through `LlmGateway`, restricted to the assignment's
`ALLOWED_MODELS` frozenset, configured via `LLM_BASE_URL`/`LLM_API_KEY` env
vars. A hard wall-clock watchdog (150s) converts provider hangs into handled
errors — provider keep-alives defeat client read timeouts.

Cost is controlled by a thread-safe `CostLedger`: every call reserves its
estimated maximum cost before launching (input ≈ chars/4, output bounded by
`max_tokens`, per-model pricing × 1.25 safety factor). Refusals happen before
network use. Reservations are settled with provider-reported `usage.cost`,
retained on timeout (the provider may still bill), reconciled on late
completion, and released on definitive pre-response failures. `metadata.cost_usd`
reports conservative exposure (settled + outstanding); `cost_settled_usd`
reports the provider-confirmed portion.

## Response parsing and validation

Tolerant JSON parsing handles markdown fences, prose-wrapped JSON, bare
scalars, missing/unknown fields, and currency-string coercion. Every result is
validated with the assignment's `tools/validate.py` before a job is marked
terminal. Null discipline is enforced: null ⇒ empty citation, confidence 0.

## Merge and conflict handling

Deterministic values are never overwritten by LLM values. Multi-record results
merge by primary key: new primary values append records; known primary values
fill null fields only; conflicting non-null values are recorded in
`result.conflicts`. Schema-description defaults are applied last (only to
genuinely absent fields).

## Backend service

FastAPI with async job processing. `POST /jobs` validates inputs (400), checks
the 50 MB limit (413), creates an idempotent job (hash of org + filename +
content + schema), and returns 202 in <2s. A bounded worker pool (10 in-flight)
processes jobs; submissions beyond capacity get 429 + Retry-After.
`GET /jobs/{job_id}` is tenant-scoped (404 for foreign orgs). Terminal
statuses are immutable. Results are contract-validated before going terminal.

## Accuracy results

| Document | Score | Records | Method |
|---|---|---|---|
| boxscore_001 | 1.000 | 10/10 | deterministic |
| budget_001 | 0.974 | 55/55 | coverage-driven LLM |
| lease_001 | 0.725 | 1/1 | vision (scanned) |
| lease_002 | 0.650 | 1/1 | text LLM + defaults |
| lease_003 | 0.825 | 1/1 | vision (scanned) |
| om_001 | 0.862 | 1/1 | text LLM |
| rentroll_001 | 0.831 | 325/325 | deterministic + table fallback |
| t12_001 | 0.670 | 127/119 | grid fallback |
| **Mean** | **0.817** | | |

API spend: ~$0.78 total of $10 (development + validation + final outputs).

## What broke and how we fixed it

- **Field-driven expansion lost rows**: budget_001's 15 unsent records were
  caused by stopping when field coverage was complete. Fixed by coverage-driven
  stages.
- **Aux-sheet poisoning**: rentroll's `report_period` grabbed "Prior Post Month"
  from a parameters sheet. Fixed by sheet-relevance pools.
- **Truncation at 16k tokens**: reasoning models (gemini, glm) consumed the
  completion budget on hidden reasoning. Fixed by row-bounded chunking (t12) and
  model selection (haiku).
- **Possessive apostrophes corrupted quoted labels**: "record's 'GRAND TOTAL'"
  produced a phantom "s" label. Fixed with word-boundary guards on the quote
  regex.
- **Column-mapped fill overwrote category values**: continuation rows whose
  category cell was also a mapped column had their label treated as a value.
  Fixed by per-row category-claim resolution.

## What we'd do with more time

Vision for mixed PDFs (text + scanned pages in one document); per-row LLM fill
for table docs with continuation rows; primary-key reconciliation against parsed
source rows for targeted re-extraction; persistent job store; per-model
structured-output capability flags with automatic fallback; prompt refinement
for lease_002's 8 missed fields (rent schedule, free-text clauses).
