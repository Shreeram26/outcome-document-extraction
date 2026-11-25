from __future__ import annotations

import pytest

from app.extraction import SchemaError, normalize_schema

RAW = {
    "name": "thing",
    "description": "a thing to extract",
    "fields": {
        "alpha": {"name": "Alpha", "description": "the a field"},
        "beta": {"name": "Beta"},
        "gamma": {"description": "the g field", "primary": True},
    },
}


def test_preserves_keys_names_descriptions_primary():
    schema = normalize_schema(RAW)
    assert list(schema.fields) == ["alpha", "beta", "gamma"]
    assert schema.fields["alpha"].name == "Alpha"
    assert schema.fields["alpha"].description == "the a field"
    assert schema.fields["beta"].name == "Beta"
    assert schema.fields["beta"].description == ""
    assert schema.fields["beta"].primary is False
    assert schema.fields["gamma"].primary is True
    assert schema.primary_key == "gamma"
    assert schema.name == "thing"
    assert schema.description == "a thing to extract"


def test_missing_top_level_name_and_description_default_empty():
    schema = normalize_schema({"fields": {"x": {"name": "X"}}})
    assert schema.name == ""
    assert schema.description == ""


def test_primary_flag_false_explicit():
    schema = normalize_schema({"fields": {"x": {"name": "X", "primary": False}}})
    assert schema.primary_key is None


def test_no_primary_is_fine():
    schema = normalize_schema({"fields": {"x": {"name": "X"}}})
    assert schema.primary_key is None


@pytest.mark.parametrize(
    "bad",
    [
        [],
        "schema",
        {},
        {"name": "n"},
        {"fields": {}},
        {"fields": []},
        {"fields": {"a": 5}},
        {"fields": {"a": {"name": 1}}},
        {"fields": {"a": {"description": 3}}},
        {"fields": {"a": {"primary": "yes"}}},
        {"fields": {"a": {"name": "A", "primary": True}, "b": {"name": "B", "primary": True}}},
        {"fields": {"a": {"name": "A"}}, "name": 7},
        {"fields": {"a": {"name": "A"}}, "description": 7},
    ],
)
def test_invalid_schemas_raise(bad):
    with pytest.raises(SchemaError):
        normalize_schema(bad)


def test_error_message_mentions_offending_field():
    with pytest.raises(SchemaError, match="'a'"):
        normalize_schema({"fields": {"a": {"name": 1}}})


def test_sample_contract_schema_normalizes():
    import json
    from pathlib import Path

    raw = json.loads(Path("contract/schema.example.json").read_text())
    schema = normalize_schema(raw)
    assert schema.primary_key == "code"
    assert list(schema.fields)[0] == "property_name"


def test_sample_schemas_all_normalize():
    import json
    from pathlib import Path

    for path in sorted(Path("sample/schemas").glob("*.json")):
        schema = normalize_schema(json.loads(path.read_text()))
        assert len(schema.fields) > 0
        assert schema.name == path.stem
