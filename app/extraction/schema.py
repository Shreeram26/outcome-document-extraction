"""Schema normalization: validate user-provided field lists, preserve keys/names/descriptions/primary.

Field descriptions may explicitly state a default for absent values
("false otherwise", "0 if none", "empty list if none"). Such defaults are
parsed out generically and applied by the engine only when a field is
genuinely not found — never overriding an extracted value, and only when
compatible with the values observed for that field."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable


class SchemaError(ValueError):
    """Raised when an extraction schema is malformed."""


@dataclass(frozen=True)
class ExtractionField:
    key: str
    name: str
    description: str
    primary: bool = False
    has_default: bool = False
    default_value: Any = None


_DEFAULT_PATTERNS = [
    (re.compile(r"\bfalse\b\s*(?:otherwise|when|if\s+(?:absent|none|no\b|not\s+stated|not\s+specified|not\s+provided|unavailable))", re.IGNORECASE), False),
    (re.compile(r"\b(?:0|zero)\s*(?:if|when)\b", re.IGNORECASE), 0),
    (re.compile(r"\b(?:empty\s+list|\[\])\s*(?:if|when|otherwise\b)", re.IGNORECASE), []),
]


def parse_field_default(description: str) -> tuple[bool, Any]:
    """Parse an explicit absent-value default from a field description.

    Recognizes schema-driven phrasings only:
    - "false otherwise" / "false if absent|none|not stated|..." -> False
    - "0 if none" / "zero if none" / "0 if ..." -> 0
    - "empty list if none" / "[] if none" -> []

    Descriptions that say "null if ..." intentionally parse to no default.
    Priority: boolean, numeric zero, empty list."""
    if not description:
        return False, None
    for pattern, value in _DEFAULT_PATTERNS:
        if pattern.search(description):
            return True, value
    return False, None


@dataclass(frozen=True)
class ExtractionSchema:
    name: str
    description: str
    fields: dict[str, ExtractionField]

    @property
    def primary_key(self) -> str | None:
        for f in self.fields.values():
            if f.primary:
                return f.key
        return None


def normalize_schema(raw: object) -> ExtractionSchema:
    """Validate a raw field list ({"name", "description", "fields": {key: {name, description, primary}}}).

    Field keys are preserved exactly; missing `name`/`description` default to the key and ""
    respectively. At most one field may be `primary: true`."""
    if not isinstance(raw, dict):
        raise SchemaError("schema must be a JSON object")
    fields_raw = raw.get("fields")
    if not isinstance(fields_raw, dict) or not fields_raw:
        raise SchemaError("schema must have a non-empty `fields` object")
    fields: dict[str, ExtractionField] = {}
    for key, spec in fields_raw.items():
        if not isinstance(key, str) or not key:
            raise SchemaError("field keys must be non-empty strings")
        if spec is None:
            spec = {}
        if not isinstance(spec, dict):
            raise SchemaError(f"field {key!r} must be an object")
        name = spec.get("name")
        description = spec.get("description")
        primary = spec.get("primary", False)
        if name is not None and not isinstance(name, str):
            raise SchemaError(f"field {key!r}: `name` must be a string")
        if description is not None and not isinstance(description, str):
            raise SchemaError(f"field {key!r}: `description` must be a string")
        if not isinstance(primary, bool):
            raise SchemaError(f"field {key!r}: `primary` must be a boolean")
        has_default, default_value = parse_field_default(description or "")
        fields[key] = ExtractionField(
            key=key, name=name or key, description=description or "", primary=primary,
            has_default=has_default, default_value=default_value,
        )
    primaries = [f.key for f in fields.values() if f.primary]
    if len(primaries) > 1:
        raise SchemaError(f"schema has more than one primary field: {primaries}")
    name = raw.get("name")
    description = raw.get("description")
    if name is not None and not isinstance(name, str):
        raise SchemaError("schema `name` must be a string")
    if description is not None and not isinstance(description, str):
        raise SchemaError("schema `description` must be a string")
    return ExtractionSchema(name=name or "", description=description or "", fields=fields)


def sub_schema(schema: ExtractionSchema, keys: Iterable[str]) -> ExtractionSchema:
    """A schema restricted to the given field keys, keeping their original
    definitions; used so the LLM fallback asks only for unresolved fields."""
    keys = list(keys)
    missing = [k for k in keys if k not in schema.fields]
    if missing:
        raise SchemaError(f"unknown field(s): {missing}")
    return ExtractionSchema(
        name=schema.name,
        description=schema.description,
        fields={k: schema.fields[k] for k in keys},
    )
