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

from ..parsing.models import ParsedDocument, Sheet
from .context import select_context
from .deterministic import DeterministicResult, deterministic_extract, norm_label
from .ledger import usage_cost
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
        max_calls: int = DEFAULT_MAX_CALLS,
        max_cost_usd: float = DEFAULT_MAX_COST_USD,
    ):
        self.gateway = gateway
        self.model = model
        self.chunk_chars = chunk_chars
        self.max_calls = max_calls
        self.max_cost_usd = max_cost_usd
        self._call_log: list[dict] = []
        self._doc_spend = 0.0
        self._doc_settled = 0.0
        self._cost_exhausted = False
        self._consecutive_failures = 0

    def extract(self, schema: ExtractionSchema, document: ParsedDocument) -> ExtractionResult:
        self._call_log = []
        self._doc_spend = 0.0
        self._doc_settled = 0.0
        self._cost_exhausted = False
        self._consecutive_failures = 0
        det = deterministic_extract(schema, document)
        warnings = list(det.warnings)
        conflicts: list[str] = []
        if det.mode == "table":
            provenance = {
                key: "deterministic" if det.entries[0][key].value is not None else "unresolved"
                for key in schema.fields
            }
            return ExtractionResult(
                entries=det.entries, warnings=warnings, llm=None, llm_responses=[],
                mode="table", provenance=provenance, call_log=list(self._call_log),
                cost_exposure_usd=0.0, cost_settled_usd=0.0,
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
            self._cost_exhausted = True
            warnings.append(f"llm fallback failed (budget): {str(exc)[:150]}")
            return None, None
        except LlmTimeoutError as exc:
            duration = int((time.monotonic() - started) * 1000)
            self._record(fields, len(rendered), duration, "timeout", estimate=estimate, note=str(exc)[:150])
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
        if not _document_has_text(document):
            warnings.append(
                "no extractable text (scanned or empty document); LLM fallback skipped"
            )
            det_resolved = {k for k, fr in det.entries[0].items() if fr.value is not None}
            return self._finish_single(schema, entry, warnings, responses, det_resolved, conflicts)
        det_resolved = {k for k, fr in det.entries[0].items() if fr.value is not None}
        stages = (
            ("targeted", min(TARGETED_CHARS, self.chunk_chars)),
            ("expanded", self.chunk_chars),
            ("full", None),
        )
        for stage, budget in stages:
            if not [k for k in schema.fields if entry[k].value is None]:
                break
            if not self._calls_left():
                warnings.append(f"call budget reached ({self.max_calls} calls); remaining fields left null")
            if self._cost_exhausted:
                break
                break
            if self._gateway_broken(warnings):
                break
            if stage == "full":
                plan = [(chunk, None) for chunk in self._chunks(document)]
            else:
                still = [k for k in schema.fields if entry[k].value is None]
                context_doc, note = select_context(
                    document, [schema.fields[k] for k in still], max_chars=budget
                )
                if note:
                    warnings.append(note)
                plan = [(context_doc, stage)]
            for context_doc, _tag in plan:
                still = [k for k in schema.fields if entry[k].value is None]
                if not still:
                    break
                if not self._calls_left():
                    warnings.append(f"call budget reached ({self.max_calls} calls); remaining fields left null")
                if self._cost_exhausted:
                    break
                    break
                if self._gateway_broken(warnings):
                    break
                relevant = (
                    self._relevant_fields(schema, still, context_doc)
                    if stage == "full"
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
                if parsed.entries:
                    if len(parsed.entries) > 1:
                        warnings.append(
                            f"llm: returned {len(parsed.entries)} entries; merged the first"
                        )
                    self._merge_single(entry, parsed.entries[0], conflicts)
        return self._finish_single(schema, entry, warnings, responses, det_resolved, conflicts)

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
            warnings.append(
                "no extractable text (scanned or empty document); LLM fallback skipped"
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
            if self._cost_exhausted:
                break
                break
            if self._gateway_broken(warnings):
                break
            if stage == "full":
                plan = self._chunks(document)
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
                if self._cost_exhausted:
                    break
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

    # ---------- shared helpers ----------

    def _finish_single(
        self,
        schema: ExtractionSchema,
        entry: dict[str, FieldResult],
        warnings: list[str],
        responses: list[LlmResponse],
        det_resolved: set[str],
        conflicts: list[str],
    ) -> ExtractionResult:
        unresolved = {k for k, fr in entry.items() if fr.value is None}
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
            entries=[entry], warnings=warnings,
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

    def _chunks(self, document: ParsedDocument) -> list[ParsedDocument]:
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
            if size + unit_size > self.chunk_chars and size:
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
