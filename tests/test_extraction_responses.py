from __future__ import annotations

import pytest

from app.extraction import (
    ExtractionParseError,
    extract_json_block,
    normalize_schema,
    parse_extraction_response,
)

SCHEMA = normalize_schema(
    {"name": "s", "fields": {"a": {"name": "A"}, "b": {"name": "B"}}}
)

GOOD_ENTRY = {
    "a": {"value": 7, "citation": "[Page 1] a=7", "confidence": 0.95},
    "b": {"value": None, "citation": "", "confidence": 0.0},
}


def _content(entries) -> str:
    import json

    return json.dumps({"entries": entries})


def test_clean_object_response():
    parsed = parse_extraction_response(_content([GOOD_ENTRY]), SCHEMA)
    assert len(parsed.entries) == 1
    assert parsed.entries[0]["a"].value == 7
    assert parsed.entries[0]["a"].citation == "[Page 1] a=7"
    assert parsed.entries[0]["a"].confidence == 0.95
    assert parsed.entries[0]["b"].value is None
    assert parsed.warnings == []


def test_markdown_fenced_json():
    content = "```json\n" + _content([GOOD_ENTRY]) + "\n```"
    parsed = parse_extraction_response(content, SCHEMA)
    assert parsed.entries[0]["a"].value == 7


def test_prose_around_json():
    content = "Here is the extraction:\n" + _content([GOOD_ENTRY]) + "\nHope that helps."
    parsed = parse_extraction_response(content, SCHEMA)
    assert parsed.entries[0]["a"].value == 7


def test_bare_entry_dict_accepted():
    import json

    parsed = parse_extraction_response(json.dumps(GOOD_ENTRY), SCHEMA)
    assert len(parsed.entries) == 1


def test_list_response_accepted():
    parsed = parse_extraction_response(_content([GOOD_ENTRY, GOOD_ENTRY]), SCHEMA)
    assert len(parsed.entries) == 2


def test_missing_field_filled_with_null_and_warning():
    entry = {"a": {"value": 1, "citation": "x", "confidence": 0.5}}
    parsed = parse_extraction_response(_content([entry]), SCHEMA)
    assert parsed.entries[0]["b"].value is None
    assert parsed.entries[0]["b"].citation == ""
    assert any("missing field 'b'" in w for w in parsed.warnings)


def test_unknown_field_dropped_with_warning():
    entry = dict(GOOD_ENTRY, c={"value": 9, "citation": "y", "confidence": 1.0})
    parsed = parse_extraction_response(_content([entry]), SCHEMA)
    assert "c" not in parsed.entries[0]
    assert any("unknown field 'c'" in w for w in parsed.warnings)


def test_bare_scalar_value_wrapped():
    entry = {"a": 5, "b": None}
    parsed = parse_extraction_response(_content([entry]), SCHEMA)
    assert parsed.entries[0]["a"].value == 5
    assert any("bare value" in w for w in parsed.warnings)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("$1,425", 1425.0),
        ("€900.50", 900.5),
        ("1,803", 1803.0),
        ("(500)", "(500)"),
        ("80271", "80271"),
        ("510-500", "510-500"),
        ("2026-09-27", "2026-09-27"),
        ("91.67%", "91.67%"),
        (42, 42),
        (None, None),
    ],
)
def test_currency_coercion_scoped_to_contract_violations(raw, expected):
    entry = {"a": {"value": raw, "citation": "x", "confidence": 1.0}, "b": None}
    parsed = parse_extraction_response(_content([entry]), SCHEMA)
    assert parsed.entries[0]["a"].value == expected


def test_list_value_coerced_itemwise():
    entry = {"a": {"value": ["$1,425", "510-500"], "citation": "x", "confidence": 1.0}, "b": None}
    parsed = parse_extraction_response(_content([entry]), SCHEMA)
    assert parsed.entries[0]["a"].value == [1425.0, "510-500"]


def test_null_value_forces_empty_citation():
    entry = {"a": {"value": None, "citation": "should be dropped", "confidence": 0.5}, "b": None}
    parsed = parse_extraction_response(_content([entry]), SCHEMA)
    assert parsed.entries[0]["a"].citation == ""


def test_value_without_citation_warns():
    entry = {"a": {"value": 5, "confidence": 0.5}, "b": None}
    parsed = parse_extraction_response(_content([entry]), SCHEMA)
    assert any("no citation" in w for w in parsed.warnings)


@pytest.mark.parametrize("bad,expected", [(1.7, 1.0), (-2, 0.0), ("high", 0.0)])
def test_confidence_clamped_or_defaulted(bad, expected):
    entry = {"a": {"value": 5, "citation": "x", "confidence": bad}, "b": None}
    parsed = parse_extraction_response(_content([entry]), SCHEMA)
    assert parsed.entries[0]["a"].confidence == expected


def test_invalid_json_raises_with_raw_content():
    with pytest.raises(ExtractionParseError, match="no valid JSON") as exc:
        parse_extraction_response("not json at all", SCHEMA)
    assert exc.value.raw_content == "not json at all"


def test_non_object_entry_raises():
    with pytest.raises(ExtractionParseError, match="entry 0"):
        parse_extraction_response(_content([5]), SCHEMA)


def test_entries_of_wrong_type_raises():
    with pytest.raises(ExtractionParseError):
        parse_extraction_response('{"entries": "nope"}', SCHEMA)


def test_unrelated_dict_raises():
    with pytest.raises(ExtractionParseError):
        parse_extraction_response('{"foo": 1}', SCHEMA)


def test_extract_json_block_handles_lists():
    assert extract_json_block("[1, 2]") == [1, 2]
    assert extract_json_block('```json\n{"a": 1}\n```') == {"a": 1}
