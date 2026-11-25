"""Model response parsing: turn free-form LLM output into predictable FieldResults.

Tolerant of the common failure modes (markdown fences, prose around JSON, bare
values, missing fields, unknown fields) while keeping the output shape fixed:
entries of {schema_key: FieldResult(value, citation, confidence)}."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .schema import ExtractionSchema

_CURRENCY_RE = re.compile(r"^[$€£₹]\s*([+-]?[\d,]+(?:\.\d+)?)$")
_THOUSANDS_RE = re.compile(r"^([+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?)$")
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class ExtractionParseError(ValueError):
    """Raised when a model response cannot be interpreted at all."""

    def __init__(self, message: str, raw_content: str = ""):
        super().__init__(message)
        self.raw_content = raw_content


@dataclass(frozen=True)
class FieldResult:
    value: Any
    citation: str = ""
    confidence: float = 0.0
    page_numbers: list[int] | None = None
    source_locations: list[str] | None = None


@dataclass
class ParsedExtraction:
    entries: list[dict[str, FieldResult]]
    warnings: list[str] = field(default_factory=list)


def extract_json_block(content: str) -> Any:
    """Extract the first JSON value from a response that may be wrapped in
    markdown fences or surrounded by prose."""
    text = content.strip()
    candidates = [text]
    fence = _FENCE_RE.search(text)
    if fence:
        candidates.append(fence.group(1).strip())
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
    raise ExtractionParseError("model response contained no valid JSON", raw_content=content)


def parse_extraction_response(content: str, schema: ExtractionSchema) -> ParsedExtraction:
    """Parse a raw model response into entries aligned with the schema.

    Every entry carries every schema field. Missing fields are filled with null
    and reported as warnings; unknown fields are dropped and reported; bare
    scalar values are wrapped as FieldResult(value=...); currency-formatted
    strings are coerced to numbers (a contract requirement)."""
    data = extract_json_block(content)
    entries_raw = _entries_from(data, schema)
    entries: list[dict[str, FieldResult]] = []
    warnings: list[str] = []
    for i, entry_raw in enumerate(entries_raw):
        entry, entry_warnings = _normalize_entry(i, entry_raw, schema)
        entries.append(entry)
        warnings.extend(entry_warnings)
    return ParsedExtraction(entries=entries, warnings=warnings)


def _entries_from(data: Any, schema: ExtractionSchema) -> list[Any]:
    if isinstance(data, dict):
        if isinstance(data.get("entries"), list):
            return data["entries"]
        if any(k in schema.fields for k in data):
            return [data]
        raise ExtractionParseError('response JSON must be {"entries": [...]} or a list of entries')
    if isinstance(data, list):
        return data
    raise ExtractionParseError(f"unexpected response type {type(data).__name__}")


def _normalize_entry(
    i: int, entry_raw: Any, schema: ExtractionSchema
) -> tuple[dict[str, FieldResult], list[str]]:
    warnings: list[str] = []
    if not isinstance(entry_raw, dict):
        raise ExtractionParseError(f"entry {i} is not an object")
    entry: dict[str, FieldResult] = {}
    for key in schema.fields:
        if key not in entry_raw:
            entry[key] = FieldResult(value=None)
            warnings.append(f"entry {i}: missing field {key!r}, filled with null")
            continue
        entry[key] = _field_result(i, key, entry_raw[key], warnings)
    for key in entry_raw:
        if key not in schema.fields:
            warnings.append(f"entry {i}: unknown field {key!r} ignored")
    return entry, warnings


def _field_result(i: int, key: str, raw: Any, warnings: list[str]) -> FieldResult:
    if isinstance(raw, dict):
        value = raw.get("value")
        citation = raw.get("citation", "")
        confidence = raw.get("confidence", 0.0)
        if not isinstance(citation, str):
            warnings.append(f"entry {i}: field {key!r} citation was not a string, coerced")
            citation = "" if citation is None else str(citation)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            warnings.append(f"entry {i}: field {key!r} confidence was not a number, set to 0.0")
            confidence = 0.0
        confidence = min(max(confidence, 0.0), 1.0)
    else:
        value = raw
        citation = ""
        confidence = 0.0
        warnings.append(f"entry {i}: field {key!r} was a bare value, wrapped without citation")
    value = _coerce_value(value)
    if value is None:
        citation = ""
        confidence = 0.0
    elif not citation:
        warnings.append(f"entry {i}: field {key!r} has a value but no citation")
    return FieldResult(value=value, citation=citation, confidence=confidence)


def _coerce_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_coerce_value(v) for v in value]
    if isinstance(value, str):
        match = _CURRENCY_RE.match(value.strip()) or _THOUSANDS_RE.match(value.strip())
        if match:
            try:
                return float(match.group(1).replace(",", ""))
            except ValueError:
                return value
    return value
