"""Schema-driven extraction core: deterministic first, LLM fallback second.

    from app.extraction import ExtractionEngine, LlmGateway, normalize_schema

    schema = normalize_schema(raw_schema_json)
    engine = ExtractionEngine(LlmGateway(), model="google/gemini-3.8-flash")
    result = engine.extract(schema, parsed_document)
"""
from .context import DEFAULT_CONTEXT_MAX_CHARS, context_phrases, select_context
from .deterministic import DeterministicResult, deterministic_extract, norm_label, quoted_labels
from .engine import ExtractionEngine, ExtractionResult
from .llm import ALLOWED_MODELS, LlmError, LlmGateway, LlmResponse, LlmTimeoutError
from .prompting import MAX_DOCUMENT_CHARS, SYSTEM_PROMPT, build_messages, render_document
from .responses import (
    ExtractionParseError,
    FieldResult,
    ParsedExtraction,
    extract_json_block,
    parse_extraction_response,
)
from .schema import (
    ExtractionField,
    ExtractionSchema,
    SchemaError,
    normalize_schema,
    sub_schema,
)
from .values import parse_scalar

__all__ = [
    "ALLOWED_MODELS",
    "DEFAULT_CONTEXT_MAX_CHARS",
    "DeterministicResult",
    "ExtractionEngine",
    "ExtractionField",
    "ExtractionParseError",
    "ExtractionResult",
    "ExtractionSchema",
    "FieldResult",
    "LlmError",
    "LlmTimeoutError",
    "LlmGateway",
    "LlmResponse",
    "MAX_DOCUMENT_CHARS",
    "ParsedExtraction",
    "SYSTEM_PROMPT",
    "SchemaError",
    "build_messages",
    "context_phrases",
    "deterministic_extract",
    "extract_json_block",
    "norm_label",
    "normalize_schema",
    "parse_extraction_response",
    "parse_scalar",
    "quoted_labels",
    "render_document",
    "select_context",
    "sub_schema",
]
