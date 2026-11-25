# Model-Routing Experiment Results

Date: 2026-11-25. Aggregate API spend this pass: **≈ $0.31** (cap $1.50; prior
session spend $0.2611 → total ≈ $0.57 of $10 credits). 10 experiment records,
6 planned configurations, 2 documents (budget_001: 55-row PDF table;
t12_001: 119-row XLSX grid). All runs sequential, ledger-enforced.

## Method-support findings (probes + runs)

| Technique | Verdict | Evidence |
|---|---|---|
| JSON prompt (baseline) | ✅ reliable-ish | works on all models; **haiku flaky** (1 of 2 identical runs unparseable even after repair) |
| `response_format: json_schema` | ⚠️ model-dependent | works on openai-provider models (gpt-6-luna 2/2); **anthropic-provider intermittently 400s** (9/9 accepted, then consistent 400s) — unusable for production today |
| Tool/function calling | ❌ | sonnet: HTTP 400; haiku/luna: tool-call shape inconsistent with our extraction |
| Pydantic / TypedDict | ➖ no separate test | `parse_extraction_response` already performs local typed validation + coercion; TypedDict adds no runtime enforcement — documented, not tested |
| Repair/retry (1 attempt) | ➖ no benefit observed | 1 attempt made (haiku flaky run); repair response also unparseable — 0% success in sample |
| Field-level escalation (cheap → strong) | ❌ negative result | +$0.137, **zero** fields filled, 8 conflicts — missing *rows* are not recoverable by field-level escalation |

## Comparison (budget_001, identical targeted context)

| Config | Model | Method | Score | Records | Cost | Reliable? |
|---|---|---|---|---|---|---|
| baseline (engine, prior run) | haiku-5.5 | json_prompt | 0.114 (40 rows) | 40/55 | $0.0069 | ✅ |
| E1 | haiku-5.5 | json_prompt | parse_error | — | $0.0170 | ❌ flaky |
| E2 | gpt-6-luna | json_schema | **0.3052** | 17/55 | **$0.0046** | ✅ 2/2 |
| E3 | sonnet-5.5 | json_prompt | 0.3052 | 17/55 | $0.1375 (30× luna) | ✅ |
| E6 | luna → sonnet escalation | json_prompt | 0.3052 | 17/55 | $0.1470 | no gain |

(t12_001: all models ≈ score 0.0275, 4/119 rows — single targeted call covers
only the header region; full row coverage needs the engine's row-batched grid
path, out of scope for these single-call experiments.)

## Evidence-based conclusions

1. **`response_format: json_schema` is the best reliability lever where
   supported** (gpt-6-luna: 2/2 clean parses, no repair, half haiku's cost).
   It is NOT safe to enable globally: anthropic-provider models intermittently
   reject it with HTTP 400.
2. **Model choice is configurable via `LLM_MODEL`** (already supported by the
   gateway): haiku produced the best single engine run (40 records) but shows
   json_prompt flakiness; luna + json_schema matched quality at ~½–⅓ cost in
   all runs; sonnet adds cost, not quality, for this document class.
3. **Blind escalation is rejected by evidence**: the strong model duplicated
   the cheap model's output at 30× cost. Escalation should trigger only on
   *missing records* (e.g. row-coverage shortfall), not unresolved fields —
   and record-coverage expansion is a chunking/staging concern, not a model
   concern.
4. **Small-sample caveat**: 1–2 runs per cell, one hard document. Conclusions
   are directional, not statistical.

## Production changes made

- `LlmGateway.complete(..., extra_params=...)`: generic pass-through for
  `response_format`/`tools`-style parameters (tested), enabling structured
  output per model without touching the extraction pipeline.
- No pipeline defaults changed: the evidence does not yet justify switching
  the default model or enabling json_schema globally (anthropic 400 risk).
