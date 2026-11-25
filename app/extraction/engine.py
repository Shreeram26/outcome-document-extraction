"""Extraction engine: deterministic first, progressive LLM fallback second.

Flow per document:
1. Deterministic stage (labelled values + table columns). Table documents whose
   header matches the schema resolve entirely here — no LLM call.
2. LLM fallback, budget-conscious and progressive:
   - stage "targeted": small context built from blocks matching the unresolved
     fields' labels (plus the document head) — one call;
   - stage "expanded": the same selection with a larger budget — one call;
   - stage "full": only as a last resort, the document split at structural
     boundaries (pages / paragraphs / rows), where each chunk receives only the
     unresolved fields whose labels appear in that chunk.
   Each call sends only the fields still unresolved. Documents with no
   extractable text (scanned) are never sent.
3. Merge: deterministic values are never overwritten; grid rows merge by their
   primary identifier with conflict detection; every schema field is present in
   every entry; missing values are null with empty citations.

Budgets (per document): max LLM calls, max context characters per call, and a
conservative cost exposure limit. Cost is accounted through reservations: every
call reserves its estimated maximum cost before launching (input tokens from
message sizes, output bounded by an enforced max_tokens, per-model pricing and
a safety factor — see CostLedger via the gateway), the reservation is settled
with the provider-reported cost on completion and RETAINED on local timeouts
(the provider may still bill the request), so exposure never under-counts.
When a budget is hit, extraction stops gracefully — successful values are kept,
the rest stay null with warnings. Every call attempt is recorded (fields,
context size, duration, status, reserved cost, settled cost, tokens, cumulative
exposure) in `ExtractionResult.call_log`."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from ..parsing.models import ParsedDocument, Sheet
from .context import select_context
from .deterministic import DeterministicResult, deterministic_extract, norm_label
from .ledger import usage_cost
from .vision import (
    build_vision_messages,
    estimate_vision_cost,
    render_scanned_pages,
    vision_page_batches,
)
from .llm import (
    LlmBadResponseError,
    LlmBudgetError,
    LlmError,
    LlmGateway,
    LlmResponse,
    LlmTimeoutError,
)
from .prompting import build_messages, render_document
from .responses import ExtractionParseError, FieldResult, ParsedExtraction, parse_extraction_response
from .schema import ExtractionField, ExtractionSchema, sub_schema

DEFAULT_CHUNK_CHARS = 12_000
DEFAULT_FULL_CHUNK_CHARS = 4_000
DEFAULT_MAX_CALLS = 8
DEFAULT_MAX_COST_USD = 0.50
TARGETED_CHARS = 6_000
MIN_CONSECUTIVE_FAILURES = 2


@dataclass
class ExtractionResult:
    entries: list[dict[str, FieldResult]]
    warnings: list[str] = field(default_factory=list)
    llm: LlmResponse | None = None
    llm_responses: list[LlmResponse] = field(default_factory=list)
    mode: str = "single"
    provenance: dict[str, str] = field(default_factory=dict)
    call_log: list[dict] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    cost_exposure_usd: float = 0.0
    cost_settled_usd: float = 0.0


class ExtractionEngine:
    def __init__(
        self,
        gateway: LlmGateway,
        model: str,
        chunk_chars: int = DEFAULT_CHUNK_CHARS,
        full_chunk_chars: int = DEFAULT_FULL_CHUNK_CHARS,
        max_calls: int = DEFAULT_MAX_CALLS,
        max_cost_usd: float = DEFAULT_MAX_COST_USD,
        vision_model: str | None = None,
        vision_dpi: int = 200,
        vision_batch_size: int = 5,
    ):
        self.gateway = gateway
        self.model = model
        self.chunk_chars = chunk_chars
        self.full_chunk_chars = full_chunk_chars
        self.max_calls = max_calls
        self.max_cost_usd = max_cost_usd
        self.vision_model = vision_model or model
        self.vision_dpi = vision_dpi
        self.vision_batch_size = vision_batch_size
        self._call_log: list[dict] = []
        self._doc_spend = 0.0
        self._doc_settled = 0.0
        self._cost_exhausted = False
        self._consecutive_failures = 0
        self._last_attempt_failed = False

    def extract(self, schema: ExtractionSchema, document: ParsedDocument) -> ExtractionResult:
        self._call_log = []
        self._doc_spend = 0.0
        self._doc_settled = 0.0
        self._cost_exhausted = False
        self._consecutive_failures = 0
        self._last_attempt_failed = False
        det = deterministic_extract(schema, document)
        warnings = list(det.warnings)
        conflicts: list[str] = []
        result = self._dispatch_extraction(schema, document, det, warnings, conflicts)
        self._apply_schema_defaults(schema, result, document)
        return result

    def _dispatch_extraction(
        self,
        schema: ExtractionSchema,
        document: ParsedDocument,
        det: DeterministicResult,
        warnings: list[str],
        conflicts: list[str],
    ) -> ExtractionResult:
        if det.mode == "table":
            provenance = {
                key: "deterministic" if det.entries[0][key].value is not None else "unresolved"
                for key in schema.fields
            }
            unresolved_after = [
                key for key, src in provenance.items() if src == "unresolved"
            ]
            if unresolved_after and _document_has_text(document):
                filled = self._table_llm_fallback(
                    schema, document, det, unresolved_after, warnings
                )
                for key in filled:
                    provenance[key] = "llm"
            return ExtractionResult(
                entries=det.entries, warnings=warnings, llm=None, llm_responses=[],
                mode="table", provenance=provenance, call_log=list(self._call_log),
                cost_exposure_usd=round(self._doc_spend, 6),
                cost_settled_usd=round(self._doc_settled, 6),
            )
        if schema.primary_key and document.kind in ("xlsx", "csv"):
            return self._fallback_grid(schema, document, det, warnings, conflicts)
        return self._fallback_progressive(schema, document, det, warnings, conflicts)

    # ---------- budget / call plumbing ----------

    def _calls_left(self) -> bool:
        return len(self._call_log) < self.max_calls

    def _record(
        self,
        fields: list[str],
        context_chars: int,
        duration_ms: int,
        status: str,
        response: LlmResponse | None = None,
        estimate: float = 0.0,
        actual_usd: float | None = None,
        note: str = "",
    ) -> dict:
        """Pure log writer — cost mutations happen in `_attempt` only, so
        settled cost is never counted twice."""
        if actual_usd is None and response is not None:
            actual_usd = usage_cost(response)
        record = {
            "call": len(self._call_log) + 1,
            "model": self.model,
            "fields": list(fields),
            "context_chars": context_chars,
            "duration_ms": duration_ms,
            "status": status,
            "tokens_in": response.tokens_in if response else 0,
            "tokens_out": response.tokens_out if response else 0,
            "reserved_usd": round(estimate, 6),
            "cost_usd": round(actual_usd, 6) if actual_usd is not None else 0.0,
            "cumulative_cost_usd": round(self._doc_spend, 6),
            "note": note,
        }
        self._call_log.append(record)
        return record

    def _attempt(
        self, sub: ExtractionSchema, context_doc: ParsedDocument, stage: str, warnings: list[str]
    ) -> tuple[ParsedExtraction | None, LlmResponse | None]:
        """One LLM attempt with budget checks, a pre-call cost reservation (via
        the gateway's ledger), observability, and graceful failure."""
        if not self._calls_left():
            warnings.append(f"call budget reached ({self.max_calls} calls); remaining fields left null")
            return None, None
        rendered = render_document(context_doc)
        fields = list(sub.fields)
        messages = build_messages(sub, context_doc)
        estimate = self.gateway.estimate_cost_usd(messages, self.model)
        if self.max_cost_usd is not None and self._doc_spend + estimate > self.max_cost_usd:
            self._cost_exhausted = True
            warnings.append(
                f"cost budget reached (${self._doc_spend + estimate:.4f} projected"
                f" >= ${self.max_cost_usd:.2f}); remaining fields left null"
            )
            return None, None
        started = time.monotonic()
        self._doc_spend += estimate
        try:
            response = self.gateway.complete(self.model, messages)
        except LlmBudgetError as exc:
            self._doc_spend -= estimate
            duration = int((time.monotonic() - started) * 1000)
            self._record(fields, len(rendered), duration, "budget_refused", estimate=estimate, note=str(exc)[:150])
            self._last_attempt_failed = True
            self._cost_exhausted = True
            warnings.append(f"llm fallback failed (budget): {str(exc)[:150]}")
            return None, None
        except LlmTimeoutError as exc:
            duration = int((time.monotonic() - started) * 1000)
            self._record(fields, len(rendered), duration, "timeout", estimate=estimate, note=str(exc)[:150])
            self._last_attempt_failed = True
            self._consecutive_failures += 1
            warnings.append(f"llm fallback failed (timeout): {str(exc)[:150]}")
            return None, None
        except LlmBadResponseError as exc:
            duration = int((time.monotonic() - started) * 1000)
            self._apply_billed_cost(exc.actual_cost_usd, estimate)
            self._record(
                fields, len(rendered), duration, "error", estimate=estimate,
                actual_usd=exc.actual_cost_usd, note=str(exc)[:150],
            )
            self._consecutive_failures += 1
            warnings.append(f"llm fallback failed (error): {str(exc)[:150]}")
            return None, None
        except LlmError as exc:
            duration = int((time.monotonic() - started) * 1000)
            self._doc_spend -= estimate
            self._record(fields, len(rendered), duration, "error", estimate=estimate, note=str(exc)[:150])
            self._last_attempt_failed = True
            self._consecutive_failures += 1
            warnings.append(f"llm fallback failed (error): {str(exc)[:150]}")
            return None, None
        duration = int((time.monotonic() - started) * 1000)
        actual = usage_cost(response)
        if actual is not None:
            self._doc_spend += actual - estimate
            self._doc_settled += actual
        try:
            parsed = parse_extraction_response(response.content, sub)
        except ExtractionParseError as exc:
            note = f"{exc}; raw: {exc.raw_content[:100]!r}"
            if response.finish_reason == "length":
                note += "; response truncated at max_tokens"
            self._record(
                fields, len(rendered), duration, "parse_error", response=response,
                estimate=estimate, note=note,
            )
            self._consecutive_failures = 0
            warnings.append(f"llm fallback failed (parse): {exc}; raw: {exc.raw_content[:100]!r}")
            if response.finish_reason == "length":
                warnings.append(
                    "response hit the max_tokens limit; increase --max-tokens or"
                    " reduce chunk size"
                )
            return None, None
        if response.finish_reason == "length":
            warnings.append(
                "response parsed but hit the max_tokens limit; increase --max-tokens"
                " or reduce chunk size"
            )
        self._record(fields, len(rendered), duration, "success", response=response, estimate=estimate)
        self._last_attempt_failed = False
        self._consecutive_failures = 0
        return parsed, response

    def _apply_billed_cost(self, billed: float | None, estimate: float):
        """A response arrived (the request may be billed): never refund the
        reservation. Book the provider cost when known (signed adjustment —
        exposure converges to the actual bill); otherwise keep the conservative
        estimate."""
        if billed is not None:
            self._doc_spend += billed - estimate
            self._doc_settled += billed

    def _gateway_broken(self, warnings: list[str]) -> bool:
        if self._consecutive_failures >= MIN_CONSECUTIVE_FAILURES:
            warnings.append(
                f"LLM gateway failed {self._consecutive_failures} times consecutively;"
                " stopping fallback"
            )
            return True
        return False

    # ---------- strategies ----------

    def _fallback_progressive(
        self,
        schema: ExtractionSchema,
        document: ParsedDocument,
        det: DeterministicResult,
        warnings: list[str],
        conflicts: list[str],
    ) -> ExtractionResult:
        entry = dict(det.entries[0])
        responses: list[LlmResponse] = []
        multi = schema.primary_key is not None
        records: list[dict[str, FieldResult]] = []
        by_primary: dict[str, dict[str, FieldResult]] = {}

        def unresolved() -> list[str]:
            if not multi or not records:
                return [k for k in schema.fields if entry[k].value is None]
            return [k for k in schema.fields if all(rec[k].value is None for rec in records)]

        if not _document_has_text(document):
            if document.scanned_pages:
                return self._vision_fallback(schema, document, det, warnings, conflicts)
            warnings.append(
                "no extractable text and no scanned pages; LLM fallback skipped"
            )
            det_resolved = {k for k, fr in det.entries[0].items() if fr.value is not None}
            return self._finish_single(schema, [entry], warnings, responses, det_resolved, conflicts)
        det_resolved = {k for k, fr in det.entries[0].items() if fr.value is not None}
        stages = (
            ("targeted", min(TARGETED_CHARS, self.chunk_chars)),
            ("expanded", self.chunk_chars),
            ("full", None),
        )
        # Multi-record mode is COVERAGE-driven: a stage runs when the previous
        # context was trimmed (unsent content exists), regardless of field
        # coverage — rows still unseen would otherwise be lost. Each call uses
        # the FULL schema so every new record carries every field. Single-entry
        # mode stays field-driven (stops when all fields are resolved).
        context_trimmed = False
        for stage, budget in stages:
            if not multi:
                if not unresolved():
                    break
            elif stage != "targeted" and not context_trimmed:
                break
            if not self._calls_left():
                warnings.append(f"call budget reached ({self.max_calls} calls); remaining fields left null")
                break
            if self._cost_exhausted:
                break
            if self._gateway_broken(warnings):
                break
            plan, context_trimmed = self._build_stage_plan(
                stage, budget, schema, document, multi, unresolved
            )
            for note in [note for _, note in plan if note]:
                warnings.append(note)
            for context_doc, _note in plan:
                still = unresolved()
                if not multi and not still:
                    break
                if not self._calls_left():
                    warnings.append(f"call budget reached ({self.max_calls} calls); remaining fields left null")
                    break
                if self._cost_exhausted:
                    break
                if self._gateway_broken(warnings):
                    break
                relevant = (
                    self._relevant_fields(schema, still, context_doc)
                    if stage == "full" and not multi
                    else list(schema.fields)
                    if multi
                    else still
                )
                if not relevant:
                    continue
                sub = sub_schema(schema, relevant)
                parsed, response = self._attempt(sub, context_doc, stage, warnings)
                if parsed is None:
                    continue
                responses.append(response)
                warnings.extend(f"llm: {w}" for w in parsed.warnings)
                if not parsed.entries:
                    continue
                if multi:
                    self._merge_records(
                        schema, parsed.entries, records, by_primary,
                        det.entries[0], conflicts, warnings,
                    )
                else:
                    if len(parsed.entries) > 1:
                        warnings.append(
                            f"llm: returned {len(parsed.entries)} entries; merged the first"
                        )
                    self._merge_single(entry, parsed.entries[0], conflicts)
        if records:
            return self._finish_single(schema, records, warnings, responses, det_resolved, conflicts)
        return self._finish_single(schema, [entry], warnings, responses, det_resolved, conflicts)

    def _merge_records(
        self,
        schema: ExtractionSchema,
        parsed_entries: list[dict[str, FieldResult]],
        records: list[dict[str, FieldResult]],
        by_primary: dict[str, dict[str, FieldResult]],
        det_single: dict[str, FieldResult],
        conflicts: list[str],
        warnings: list[str],
    ):
        """Merge LLM entries into records keyed by the primary field. New
        primary values append records (document-level deterministic values are
        repeated into them); known primary values fill null fields only, and
        conflicting non-null values are recorded without overwriting. Entries
        without a usable primary value are dropped with a warning."""
        dropped = 0
        for llm_entry in parsed_entries:
            primary_fr = llm_entry.get(schema.primary_key)
            pvalue = primary_fr.value if primary_fr is not None else None
            pnorm = norm_label(pvalue) if pvalue is not None else ""
            if not pnorm:
                dropped += 1
                continue
            existing = by_primary.get(pnorm)
            if existing is None:
                record: dict[str, FieldResult] = {}
                for key in schema.fields:
                    fr = llm_entry.get(key)
                    if (fr is None or fr.value is None) and det_single[key].value is not None:
                        fr = det_single[key]
                    record[key] = fr or FieldResult(value=None)
                by_primary[pnorm] = record
                records.append(record)
                continue
            for key in schema.fields:
                fr = llm_entry.get(key)
                if fr is None:
                    continue
                if existing[key].value is None:
                    existing[key] = fr
                elif fr.value is not None and fr.value != existing[key].value:
                    conflicts.append(
                        f"{schema.primary_key}={pvalue!r} field {key!r}: kept"
                        f" {existing[key].value!r}; also got {fr.value!r}"
                    )
        if dropped:
            warnings.append(
                f"llm: {dropped} entry(ies) dropped for missing primary value"
            )

    def _vision_fallback(
        self,
        schema: ExtractionSchema,
        document: ParsedDocument,
        det: DeterministicResult,
        warnings: list[str],
        conflicts: list[str],
    ) -> ExtractionResult:
        """Extract fields from scanned pages using a vision-capable model.

        Pages are rendered to PNG at the configured DPI, batched, and sent
        sequentially to the gateway. Results from all batches are merged
        into a single entry (filling nulls). Deterministic values are never
        overwritten."""
        entry = dict(det.entries[0])
        responses: list[LlmResponse] = []
        det_resolved = {k for k, fr in det.entries[0].items() if fr.value is not None}

        scanned = sorted(document.scanned_pages)
        if not scanned:
            warnings.append("no scanned pages detected; vision fallback skipped")
            return self._finish_single(schema, [entry], warnings, responses, det_resolved, conflicts)

        from .vision import render_scanned_pages, vision_page_batches, build_vision_messages

        page_images = render_scanned_pages(document, dpi=self.vision_dpi)
        if not page_images:
            warnings.append("no pages could be rendered for vision extraction")
            return self._finish_single(schema, [entry], warnings, responses, det_resolved, conflicts)

        batches = vision_page_batches(scanned, batch_size=self.vision_batch_size)
        schema_fields_raw = {
            key: {"name": f.name, "description": f.description}
            for key, f in schema.fields.items()
        }

        for batch_pages in batches:
            if not self._calls_left():
                warnings.append(f"call budget reached ({self.max_calls} calls); remaining fields left null")
                break
            if self._cost_exhausted:
                break
            if self._gateway_broken(warnings):
                break

            batch_images = {pn: page_images[pn] for pn in batch_pages if pn in page_images}
            if not batch_images:
                continue
            messages = build_vision_messages(schema_fields_raw, batch_images, batch_pages)
            num_images = len(batch_images)

            started = time.monotonic()
            try:
                response = self.gateway.complete(
                    self.vision_model, messages, max_tokens=8000,
                )
            except LlmBudgetError as exc:
                self._cost_exhausted = True
                warnings.append(f"vision fallback failed (budget): {str(exc)[:150]}")
                break
            except (LlmTimeoutError, LlmBadResponseError, LlmError) as exc:
                duration = int((time.monotonic() - started) * 1000)
                self._record(
                    list(schema.fields), 0, duration, "error",
                    estimate=0, note=f"vision: {str(exc)[:120]}",
                )
                self._consecutive_failures += 1
                self._last_attempt_failed = True
                warnings.append(f"vision fallback failed: {str(exc)[:150]}")
                continue

            duration = int((time.monotonic() - started) * 1000)
            actual = usage_cost(response.raw)
            if actual is not None:
                self._doc_settled += actual
            responses.append(response)
            self._consecutive_failures = 0

            try:
                parsed = parse_extraction_response(response.content, schema)
            except ExtractionParseError as exc:
                warnings.append(f"vision parse error: {str(exc)[:150]}")
                continue

            self._record(
                list(schema.fields), 0, duration, "success",
                response=response, estimate=0,
            )
            warnings.extend(f"vision: {w}" for w in parsed.warnings)

            for llm_entry in parsed.entries:
                for key, fr in llm_entry.items():
                    if entry[key].value is None and fr.value is not None:
                        entry[key] = fr
                    elif entry[key].value is not None and fr.value is not None and fr.value != entry[key].value:
                        conflicts.append(
                            f"{key}: kept {entry[key].value!r}; vision also returned {fr.value!r}"
                        )
            if all(entry[k].value is not None for k in schema.fields):
                break

        return self._finish_single(schema, [entry], warnings, responses, det_resolved, conflicts)

    def _build_stage_plan(
        self,
        stage: str,
        budget: int,
        schema: ExtractionSchema,
        document: ParsedDocument,
        multi: bool,
        unresolved: Callable[[], list[str]],
    ) -> tuple[list[tuple[ParsedDocument, str | None]], bool]:
        """Build the extraction plan for one stage: a list of (context, note)
        pairs plus a flag indicating whether the context was trimmed (meaning
        unsent content exists)."""
        if stage == "full":
            chunk_size = self.full_chunk_chars if multi else self.chunk_chars
            plan = [(chunk, None) for chunk in self._chunks(document, max_chars=chunk_size)]
            return plan, False
        still = unresolved()
        context_doc, note = select_context(
            document, [schema.fields[k] for k in still], max_chars=budget
        )
        return [(context_doc, note)], note is not None

    def _merge_single(self, entry: dict[str, FieldResult], llm_entry: dict[str, FieldResult], conflicts: list[str]):
        for key, fr in llm_entry.items():
            if entry[key].value is None:
                entry[key] = fr
            elif fr.value is not None and fr.value != entry[key].value:
                conflicts.append(
                    f"{key}: kept {entry[key].value!r}; llm also returned {fr.value!r}"
                )

    def _fallback_grid(
        self,
        schema: ExtractionSchema,
        document: ParsedDocument,
        det: DeterministicResult,
        warnings: list[str],
        conflicts: list[str],
    ) -> ExtractionResult:
        """Table documents whose header could not be matched: progressive stages,
        entries merged by primary identifier with conflict detection."""
        if not _document_has_text(document):
            if document.scanned_pages:
                return self._vision_fallback(schema, document, det, warnings, conflicts)
            warnings.append(
                "no extractable text and no scanned pages; LLM fallback skipped"
            )
            return self._finish_grid(schema, det.entries, warnings, [], "single", det)
        entries: list[dict[str, FieldResult]] = []
        by_primary: dict[str, dict[str, FieldResult]] = {}
        responses: list[LlmResponse] = []
        pcol_key = schema.primary_key
        stages = (
            ("targeted", min(TARGETED_CHARS, self.chunk_chars)),
            ("expanded", self.chunk_chars),
            ("full", None),
        )
        for stage, budget in stages:
            if not self._calls_left():
                warnings.append(f"call budget reached ({self.max_calls} calls); remaining fields left null")
                break
            if self._cost_exhausted:
                break
            if self._gateway_broken(warnings):
                break
            if stage == "full":
                plan = self._chunks(document, max_chars=self.full_chunk_chars)
            else:
                context_doc, note = select_context(
                    document, list(schema.fields.values()), max_chars=budget
                )
                if note:
                    warnings.append(note)
                plan = [context_doc]
            for context_doc in plan:
                if not self._calls_left():
                    warnings.append(f"call budget reached ({self.max_calls} calls); remaining fields left null")
                    break
                if self._cost_exhausted:
                    break
                if self._gateway_broken(warnings):
                    break
                parsed, response = self._attempt(schema, context_doc, stage, warnings)
                if parsed is None:
                    continue
                responses.append(response)
                warnings.extend(f"llm: {w}" for w in parsed.warnings)
                for new_entry in parsed.entries:
                    pvalue = new_entry[pcol_key].value
                    pnorm = norm_label(pvalue) if pvalue is not None else ""
                    if not pnorm:
                        continue
                    if pnorm in by_primary:
                        conflicts.append(
                            f"row {pvalue!r}: duplicate across calls; kept the first"
                        )
                        continue
                    by_primary[pnorm] = new_entry
                    entries.append(new_entry)
        return self._finish_grid(schema, entries, warnings, responses, "single", det, conflicts)

    def _table_llm_fallback(
        self,
        schema: ExtractionSchema,
        document: ParsedDocument,
        det: DeterministicResult,
        unresolved: list[str],
        warnings: list[str],
    ) -> list[str]:
        """One bounded targeted call to fill fields the deterministic table
        stage left unresolved (e.g. document-level values living on an
        auxiliary sheet). A single-entry response is treated as document-level
        and repeated into every record; multi-entry responses merge by primary
        (fill nulls only). Returns the keys that were filled."""
        if not self._calls_left() or self._cost_exhausted or self._gateway_broken(warnings):
            return []
        context_doc, note = select_context(
            document, [schema.fields[k] for k in unresolved],
            max_chars=min(TARGETED_CHARS, self.chunk_chars),
        )
        if note:
            warnings.append(note)
        parsed, _response = self._attempt(schema, context_doc, "table_fallback", warnings)
        if parsed is None or not parsed.entries:
            return []
        filled: list[str] = []
        pkey = schema.primary_key
        if len(parsed.entries) == 1:
            for key in unresolved:
                fr = parsed.entries[0].get(key)
                if fr is not None and fr.value is not None:
                    for entry in det.entries:
                        if entry[key].value is None:
                            entry[key] = fr
                    filled.append(key)
            return filled
        for llm_entry in parsed.entries:
            primary_fr = llm_entry.get(pkey)
            pvalue = primary_fr.value if primary_fr is not None else None
            pnorm = norm_label(pvalue) if pvalue is not None else ""
            if not pnorm:
                continue
            for entry in det.entries:
                entry_primary = entry.get(pkey)
                if entry_primary is None or norm_label(entry_primary.value) != pnorm:
                    continue
                for key in unresolved:
                    fr = llm_entry.get(key)
                    if fr is not None and fr.value is not None and entry[key].value is None:
                        entry[key] = fr
                        if key not in filled:
                            filled.append(key)
        return filled

    def _apply_schema_defaults(self, schema: ExtractionSchema, result: ExtractionResult, document: ParsedDocument):
        """Apply schema-description defaults to fields that are genuinely
        absent (null) in the final entries. Skipped when the document has no
        extractable text (fields may exist but were unreadable) or when the
        LLM gateway failed repeatedly (extraction did not complete). Defaults
        never override extracted non-null values and are skipped when the
        observed values for the field are type-incompatible."""
        if not _document_has_text(document):
            return
        if self._last_attempt_failed or self._consecutive_failures >= MIN_CONSECUTIVE_FAILURES:
            return
        defaults = {k: f for k, f in schema.fields.items() if f.has_default}
        if not defaults or not result.entries:
            return
        applied: list[str] = []
        for entry in result.entries:
            for key, f in defaults.items():
                fr = entry.get(key)
                if fr is not None and fr.value is not None:
                    continue
                samples = [
                    e[key].value for e in result.entries
                    if e is not entry and key in e and e[key].value is not None
                ]
                if not _default_compatible(f.default_value, samples):
                    continue
                entry[key] = FieldResult(value=f.default_value, citation="", confidence=0.9)
                if key not in applied:
                    applied.append(key)
        for key in applied:
            if result.provenance.get(key) in (None, "unresolved"):
                result.provenance[key] = "default"
        if applied:
            result.warnings.append(f"schema defaults applied for: {', '.join(applied)}")

    # ---------- shared helpers ----------

    def _finish_single(
        self,
        schema: ExtractionSchema,
        entries: list[dict[str, FieldResult]],
        warnings: list[str],
        responses: list[LlmResponse],
        det_resolved: set[str],
        conflicts: list[str],
    ) -> ExtractionResult:
        first = entries[0]
        unresolved = {k for k, fr in first.items() if fr.value is None}
        provenance = {
            k: (
                "deterministic"
                if k not in unresolved and k in det_resolved
                else "llm" if k not in unresolved
                else "unresolved"
            )
            for k in schema.fields
        }
        return ExtractionResult(
            entries=entries, warnings=warnings,
            llm=responses[-1] if responses else None, llm_responses=responses,
            mode="single", provenance=provenance, call_log=list(self._call_log),
            conflicts=conflicts,
            cost_exposure_usd=round(self._doc_spend, 6),
            cost_settled_usd=round(self._doc_settled, 6),
        )

    def _finish_grid(
        self,
        schema: ExtractionSchema,
        entries: list[dict[str, FieldResult]],
        warnings: list[str],
        responses: list[LlmResponse],
        mode: str,
        det: DeterministicResult,
        conflicts: list[str] | None = None,
    ) -> ExtractionResult:
        det_resolved = set(det.doc_level) | {
            k for k, fr in det.entries[0].items() if fr.value is not None
        }
        if not entries:
            entries = det.entries
            det_only = True
        else:
            for entry in entries:
                for key, fr in det.doc_level.items():
                    if entry[key].value is None:
                        entry[key] = fr
            det_only = False
        provenance = {
            k: (
                "deterministic"
                if (entries[0][k].value is not None and (det_only or k in det_resolved))
                else "llm" if entries[0][k].value is not None
                else "unresolved"
            )
            for k in schema.fields
        }
        return ExtractionResult(
            entries=entries, warnings=warnings,
            llm=responses[-1] if responses else None, llm_responses=responses,
            mode=mode, provenance=provenance, call_log=list(self._call_log),
            conflicts=conflicts or [],
            cost_exposure_usd=round(self._doc_spend, 6),
            cost_settled_usd=round(self._doc_settled, 6),
        )

    def _relevant_fields(
        self, schema: ExtractionSchema, still: list[str], context_doc: ParsedDocument
    ) -> list[str]:
        """Unresolved fields whose label phrases appear in the chunk; fields with
        no label phrases at all are always considered relevant."""
        from .context import context_phrases

        phrases_by_field = {
            f.key: set(context_phrases([f])) for f in (schema.fields[k] for k in still)
        }
        text = norm_label(render_document(context_doc))
        relevant = []
        for key in still:
            phrases = phrases_by_field[key]
            if not phrases or any(p in text for p in phrases):
                relevant.append(key)
        return relevant

    def _chunks(self, document: ParsedDocument, max_chars: int | None = None) -> list[ParsedDocument]:
        """Split a document into LLM-sized chunks at structural boundaries
        (pages / paragraphs / rows). Cell references and page markers survive
        chunking, so citations stay truthful."""
        if document.kind == "pdf":
            units = [("pdf", page) for page in document.pages]
        elif document.kind == "docx":
            units = [("docx", section) for section in document.sections]
        else:
            units = [
                ("sheet", (sheet, row))
                for sheet in document.sheets
                for row in sheet.rows
            ]
        limit = max_chars if max_chars is not None else self.chunk_chars
        chunks: list[ParsedDocument] = []
        current_pages: list = []
        current_sections: list = []
        current_rows: dict[str, list] = {}
        size = 0

        def flush():
            nonlocal current_pages, current_sections, current_rows, size
            if current_pages or current_sections or current_rows:
                chunks.append(_materialize(document, current_pages, current_sections, current_rows))
            current_pages, current_sections, current_rows, size = [], [], {}, 0

        for kind, unit in units:
            unit_doc = _unit_document(document, kind, unit)
            unit_size = len(render_document(unit_doc))
            if size + unit_size > limit and size:
                flush()
            if kind == "pdf":
                current_pages.append(unit)
            elif kind == "docx":
                current_sections.append(unit)
            else:
                sheet, row = unit
                current_rows.setdefault(sheet.name, []).append(row)
            size += unit_size
        flush()
        return chunks


def _unit_document(document: ParsedDocument, kind: str, unit):
    if kind == "pdf":
        return ParsedDocument(kind="pdf", filename=document.filename, pages=[unit])
    if kind == "docx":
        return ParsedDocument(kind="docx", filename=document.filename, sections=[unit])
    sheet, row = unit
    return ParsedDocument(
        kind=document.kind,
        filename=document.filename,
        sheets=[Sheet(name=sheet.name, rows=(row,), n_rows=1, n_cols=sheet.n_cols)],
    )


def _materialize(
    document: ParsedDocument, pages: list, sections: list, rows_by_sheet: dict[str, list]
) -> ParsedDocument:
    merged = ParsedDocument(kind=document.kind, filename=document.filename)
    if document.kind == "pdf":
        merged.pages = list(pages)
    elif document.kind == "docx":
        merged.sections = list(sections)
    else:
        sheets = []
        for name, rows in rows_by_sheet.items():
            sheets.append(Sheet(name=name, rows=tuple(rows), n_rows=len(rows)))
        merged.sheets = sheets
    return merged


def _document_has_text(document: ParsedDocument) -> bool:
    if document.kind == "pdf":
        return any(p.text.strip() for p in document.pages)
    if document.kind == "docx":
        return any(
            (s.kind == "paragraph" and s.text.strip()) or s.table is not None
            for s in document.sections
        )
    return any(
        cell.value not in (None, "")
        for sheet in document.sheets
        for row in sheet.rows
        for cell in row
    )


def _default_compatible(default, sample_values) -> bool:
    """A default is compatible when observed values for the field share its
    type. No observed values -> compatible (nothing to contradict it)."""
    for value in sample_values:
        if isinstance(default, list):
            return isinstance(value, list)
        if isinstance(default, bool):
            return isinstance(value, bool)
        if isinstance(default, (int, float)):
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if isinstance(default, str):
            return isinstance(value, str)
    return True
